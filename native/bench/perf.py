#!/usr/bin/env python3
"""Full performance suite: TTFT, single-stream decode, concurrency, prefill,
decode during a long prefill.

    python3 bench/perf.py
    GB10_BASE_URL=http://127.0.0.1:8888/v1 GB10_MODEL=qwen3.8-27b-sglang \\
      python3 bench/perf.py --levels 1 4 8 16

Env: GB10_BASE_URL / GB10_MODEL / GB10_API_KEY (+ optional GB10_METRICS_URL),
see bench/common.py. Run it against an otherwise idle box — any competing GPU
work skews every number here.

What the columns mean (and the traps they avoid):
  * decode tok/s = tokens after the first one / time after the first one, so
    TTFT and queueing do not leak into the generation speed; `e2e tok/s`
    (tokens / whole request, the figure the README headline uses) is shown
    next to it for comparison.
  * queue: time spent waiting for admission, from the engine's queue-time
    histogram (vLLM request_queue_time, SGLang queue_time) or, if only gauges
    are exported, the peak number of waiting requests. A level above the
    server's cap (SGLang --max-running-requests, itself clamped to
    max-mamba-cache-size / 5; vLLM --max-num-seqs) queues requests in waves:
    TTFT and aggregate then measure admission, not the model — such rows are
    flagged and excluded from the peak.
  * prefill prompts are unique from their very first token and made of varied
    words, so the prefix/radix cache cannot serve part of them; cached tokens
    are checked and the rate is computed over uncached tokens only.
  * warmup: after a restart, kernels are JIT-compiled / graphs captured on
    their first use DURING inference (speculative-decoding paths, the first
    few concurrent batch sizes, the first longer prompt) — seconds of stall
    that land in whatever runs first. The warmup pass touches those shapes
    before anything is measured; it runs before any section, also with
    `--only <section>`; `--only warmup` does just that (useful right after a
    server start), and `--no-warmup` measures the cold server on purpose.
  * GPU: peak temperature / lowest SM clock per section or row from the local
    nvidia-smi (GB10_GPU_WATCH), flagged on throttling or near the 80 °C
    suspend threshold. A server restart mid-run is flagged too.
  * the run header records the engine's effective settings (SGLang
    /get_server_info: revision, draft tokens, request cap ...), and the whole
    output is saved under results/runs/ (GB10_RUN_DIR).
  * decode during a long prefill: what the streams already running see while
    another request's long prompt is prefilled (an agent sending a big
    context while others generate): their tok/s before / during / after it,
    and the longest pause between two chunks of text. SGLang schedules a
    prefill batch before decode whenever one is ready and turns mixed
    prefill+decode batches off with speculative decoding, so the chunks of
    one prompt can run back to back and the decode wait for all of them
    whatever --chunked-prefill-size is; --prefill-decode-interval N runs N
    decode rounds after each prefill batch. This section measures which.
  * PLE ms/op (patched vLLM only, GB10_METRICS_PORT) comes from the
    engine-side metrics sidecar as a delta around each section; each closing
    read waits one sidecar refresh (~6 s per row).
"""
import argparse
import statistics
import sys
import threading
import time
import uuid

import common
from common import (CODE_PROMPT, GpuWatch, PeakWatch, cache_hit_significant, calibrate, chat,
                    decode_rate, delta, metrics, ple_summary, restarted, sidecar, spec_summary,
                    unique_prompt)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _fmt(x, spec, unit=""):
    return "n/a" if x is None else f"{x:{spec}}{unit}"


def _parallel(n, fn):
    """Run fn(i) for i in range(n) concurrently -> (results, errors, wall)."""
    out, errors = [None] * n, []

    def worker(i):
        try:
            out[i] = fn(i)
        except Exception as e:  # noqa: BLE001 - reported by the caller
            errors.append(repr(e))

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in out if r], errors, time.perf_counter() - t0


def warmup(levels):
    """Trigger the first-use JIT compiles / graph captures before anything is timed."""
    print("## 0. Warmup (first-use kernel compiles)", flush=True)
    t0 = time.perf_counter()
    chat("hi", 8)
    chat(CODE_PROMPT, 64, stream=True)                     # speculative-decoding path
    for n in sorted({x for x in levels if x <= 16} | {2, 4, 8}):
        _parallel(n, lambda i: chat(f"[{uuid.uuid4().hex[:8]}] {CODE_PROMPT}", 32, stream=True))
    chat(unique_prompt(uuid.uuid4().hex, 3000), 8, stream=True)  # longer-prompt path
    print(f"   done in {time.perf_counter() - t0:.1f}s\n")


def ttft_probe(n=5):
    print("## 1. Time to first token (streaming, short prompt)")
    vals = [chat("Say hello.", 16, stream=True)["ttft"] for _ in range(n)]
    vals = [v for v in vals if v is not None]
    if not vals:
        print("   no content in any stream — cannot measure TTFT\n")
        return
    print(f"   median {statistics.median(vals) * 1000:.0f} ms   "
          f"min {min(vals) * 1000:.0f}   max {max(vals) * 1000:.0f}\n")


def single_stream(n=5, max_tokens=700):
    print(f"## 2. Single-stream decode (code, thinking off, up to {max_tokens} tokens)")
    rows = []
    s0 = sidecar(settle=True)
    with GpuWatch() as gpu:
        for _ in range(n):
            m0 = metrics()
            r = chat(CODE_PROMPT, max_tokens, stream=True)
            rows.append((r, spec_summary(m0, metrics())))
    for r, spec in rows:
        print(f"   {r['completion_tokens']:>4} tok  decode {_fmt(decode_rate(r), '.1f')} tok/s  "
              f"e2e {r['completion_tokens'] / r['e2e']:.1f} tok/s  ttft {_fmt(r['ttft'], '.2f', 's')}"
              f"{'  ' + spec if spec else ''}")
    short = [r for r, _ in rows if r["completion_tokens"] < max_tokens * 0.5]
    if short:
        print(f"   note: {len(short)} run(s) stopped early (<50% of max_tokens) — "
              "short runs overweight the fixed per-request cost")
    ple = ple_summary(s0, sidecar(settle=True))
    print(f"   MEDIAN decode {_fmt(_med(decode_rate(r) for r, _ in rows), '.1f')} tok/s   "
          f"e2e {statistics.median(r['completion_tokens'] / r['e2e'] for r, _ in rows):.1f} tok/s"
          f"{'   ' + ple if ple else ''}")
    print(f"   {gpu}\n" if str(gpu) else "")


def concurrency(levels, tokens=300):
    print(f"## 3. Concurrency sweep ({tokens} tokens per stream)")
    print(f"   {'streams':>7} {'wall(s)':>8} {'aggregate':>11} {'decode/stream':>14} "
          f"{'TTFT p50':>9} {'TTFT max':>9} {'queue':>9} {'PLE ms/op':>10}")
    print("   " + "-" * 85)
    peak = (0, 0.0)
    s_prev = sidecar(settle=True)
    for n in levels:
        m0, s0 = metrics(), s_prev
        # uuid first: unique from the first token, no prefix reuse
        with PeakWatch() as watch, GpuWatch() as gpu:
            done, errors, wall = _parallel(n, lambda i: chat(
                f"[{uuid.uuid4().hex[:8]}] {CODE_PROMPT} Variant {i}.", tokens, stream=True))
        m1 = metrics()
        cold = restarted(m0, m1)
        if cold:
            print(f"   {n:>7}  !! the engine restarted during this row — its numbers "
                  "(and the next rows') are cold-server numbers")
        s1 = s_prev = sidecar(settle=True)
        ops = delta(s0, s1, "ple_ops")
        ple = delta(s0, s1, "ple_op_ms") / ops if ops else None
        if errors:
            print(f"   {n:>7}  {len(errors)} request(s) failed, e.g. {errors[0][:120]}")
            if not done:
                continue
        agg = sum(r["completion_tokens"] for r in done) / wall
        ttfts = [r["ttft"] for r in done if r["ttft"] is not None]
        q_sum, q_cnt = delta(m0, m1, "queue_sum"), delta(m0, m1, "queue_count")
        queue = q_sum / q_cnt if q_sum is not None and q_cnt else None
        waiting = watch.peak.get("waiting")
        if queue is not None:
            queue_col, queued = _fmt(queue, ">8.2f", "s"), queue > 0.5
        elif waiting is not None:
            queue_col, queued = f"{waiting:>4.0f} wait", waiting > 0
        else:
            queue_col, queued = "n/a", False
        # Waiting = not yet scheduled: more streams than the running-request
        # cap, or the per-step token budget taken by other requests' prefill.
        flag = "  <- queued (request cap / prefill budget)" if queued else ""
        if agg > peak[1] and not flag and not cold:
            peak = (n, agg)
        print(f"   {n:>7} {wall:>8.1f} {agg:>7.1f} t/s {_fmt(_med(decode_rate(r) for r in done), '>9.1f')} t/s "
              f"{_fmt(_med(ttfts), '>8.2f', 's')} {_fmt(max(ttfts) if ttfts else None, '>8.2f', 's')} "
              f"{queue_col:>9} {_fmt(ple, '>10.1f')}{flag}")
        if str(gpu):
            print(f"   {'':>7} {gpu}")
        time.sleep(3)
    if peak[0]:
        print(f"\n   peak aggregate without queueing: {peak[1]:.1f} tok/s at {peak[0]} streams\n")
    else:
        print()


def prefill(targets):
    print("## 4. Long-context prefill (unique prompts, prefix cache checked)")
    s_prev = sidecar(settle=True)
    for target in targets:
        m0, s0 = metrics(), s_prev
        prompt = unique_prompt(uuid.uuid4().hex, target)
        with GpuWatch() as gpu:
            r = chat(prompt, 8, stream=True)
        hits = r["cached_tokens"]
        if hits is None:
            hits = delta(m0, metrics(), "pc_hits")
        s_prev = sidecar(settle=True)
        ple = ple_summary(s0, s_prev)
        fresh = r["prompt_tokens"] - (hits or 0)
        rate = fresh / r["ttft"] if r["ttft"] else None
        cached = "n/a" if hits is None else f"{hits:.0f}"
        print(f"   prompt {r['prompt_tokens']:>7} tok (cached {cached:>5}) -> "
              f"TTFT {_fmt(r['ttft'], '>6.2f', 's')}   prefill ~{_fmt(rate, '>6.0f')} tok/s"
              f"{'   ' + ple if ple else ''}{'   ' + str(gpu) if str(gpu) else ''}")
        if cache_hit_significant(hits, r["prompt_tokens"]):
            print("   !! prefix-cache hit on a unique prompt — the rate above excludes it")
    print()


LONG_PROMPT = (
    "Write a long, complete Python module implementing a small SQL engine: a "
    "tokenizer, a recursive-descent parser for SELECT / WHERE / JOIN / GROUP BY / "
    "ORDER BY, a planner and an executor over in-memory tables, with docstrings, "
    "type hints and a full unittest suite. Write all of the code; do not stop early."
)


def _tokens_at(series, t):
    """Completion tokens a stream had reached at time t (its last chunk at or
    before t; 0 before its first)."""
    n = 0
    for ts, k in series:
        if ts > t:
            break
        n = k
    return n


def _window_rate(all_series, a, b):
    """Aggregate tok/s of the streams over [a, b]."""
    if b <= a:
        return None
    return sum(_tokens_at(s, b) - _tokens_at(s, a) for s in all_series) / (b - a)


def prefill_interference(prompt_tokens, levels, base=4.0, after=3.0, max_tokens=16384):
    print(f"## 5. Decode during a long prefill (one ~{prompt_tokens // 1000}K prompt "
          "while N streams generate)")
    print(f"   {'streams':>7} {'prompt':>8} {'TTFT':>8} {'prefill':>10}   "
          f"{'decode before':>13} {'during':>8} {'after':>8}   {'longest pause':>13} {'step':>7}")
    print("   " + "-" * 96)
    for n in levels:
        series = [[] for _ in range(n)]
        stop_after = [None]
        no_usage = [False]

        def recorder(i):
            def cb(t, k):
                if k is None:          # no per-chunk usage: count chunks instead
                    no_usage[0] = True
                    k = len(series[i]) + 1
                series[i].append((t, k))
                return stop_after[0] is not None and t > stop_after[0]
            return cb

        box = []
        bg = threading.Thread(target=lambda: box.append(_parallel(n, lambda i: chat(
            f"[{uuid.uuid4().hex[:8]}] {LONG_PROMPT} Variant {i}.", max_tokens, stream=True,
            ignore_eos=True, on_chunk=recorder(i)))))
        bg.start()
        deadline = time.perf_counter() + 300
        while not all(series) and bg.is_alive() and time.perf_counter() < deadline:
            time.sleep(0.05)
        if not all(series):
            stop_after[0] = 0.0
            bg.join()
            errors = box[0][1] if box else []
            print(f"   {n:>7}  the decode streams did not start"
                  f"{', e.g. ' + errors[0][:120] if errors else ''}")
            continue
        time.sleep(base)
        t_send = time.perf_counter()
        try:
            with GpuWatch() as gpu:
                r = chat(unique_prompt(uuid.uuid4().hex, prompt_tokens), 8, stream=True)
        except Exception as e:  # noqa: BLE001
            stop_after[0] = 0.0
            bg.join()
            print(f"   {n:>7}  the long prompt failed: {e!r:.120}")
            continue
        t_first = t_send + (r["ttft"] or r["e2e"])
        stop_after[0] = t_first + after
        bg.join()
        _, errors, _ = box[0]
        hits = r["cached_tokens"] or 0
        rate = (r["prompt_tokens"] - hits) / (t_first - t_send)

        before = _window_rate(series, t_send - (base - 0.5), t_send)
        during = _window_rate(series, t_send, t_first)
        end = min(s[-1][0] for s in series)
        after_rate = _window_rate(series, t_first, min(end, t_first + after))
        # Gaps between chunks: the longest that overlaps the prefill, and the
        # usual one before it (one decode step, or a few with spec decoding).
        longest, usual = 0.0, []
        for s in series:
            ts = [t for t, _ in s]
            for a, b in zip(ts, ts[1:]):
                if b > t_send and a < t_first:
                    longest = max(longest, b - a)
                elif b <= t_send:
                    usual.append(b - a)
        unit = " chunks/s" if no_usage[0] else ""
        print(f"   {n:>7} {r['prompt_tokens']:>8} {t_first - t_send:>7.2f}s {rate:>6.0f} t/s   "
              f"{_fmt(before, '>9.1f', ' t/s')} {_fmt(during, '>8.1f')} {_fmt(after_rate, '>8.1f')}   "
              f"{longest:>12.2f}s {_fmt(_med(usual), '>6.2f', 's')}")
        notes = []
        if errors:
            notes.append(f"{len(errors)} decode stream(s) failed, e.g. {errors[0][:100]}")
        if end < t_first:
            notes.append("a decode stream ended before the prompt's first token: "
                         "the numbers above undercount")
        if no_usage[0]:
            notes.append(f"no per-chunk usage from the server: rates are{unit}, not tokens")
        if cache_hit_significant(hits, r["prompt_tokens"]):
            notes.append("prefix-cache hit on the unique prompt — the prefill rate excludes it")
        for note in notes:
            print(f"   {'':>7} !! {note}")
        if str(gpu):
            print(f"   {'':>7} {gpu}")
        time.sleep(3)
    print("   (during: the streams' aggregate while the prompt is prefilled; longest pause:"
          " the longest gap between two chunks of any stream in that time; step: the usual"
          " gap before it)\n")


LEVELS = (1, 2, 4, 8, 16, 24, 32)


def default_levels():
    """Levels up to the server's request cap, the cap itself, and one burst row.

    A sweep past the cap measures admission, not the model, and one that stops
    short of it never shows the full-load aggregate (a cap of 12 with the fixed
    1..32 levels peaked "at 8 streams", then queued). The burst row (1.5x the
    cap) is kept to show what a spike costs in TTFT; it is flagged and left out
    of the peak. Unknown cap (not SGLang): the fixed levels."""
    cap = common.server_info().get("max_running_requests")
    if not isinstance(cap, int) or cap < 1:
        return list(LEVELS)
    return sorted({n for n in LEVELS if n < cap} | {cap, cap + (cap + 1) // 2})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="+", type=int,
                    help="concurrency levels (default: 1 2 4 8 16 24 32 up to the server's "
                         "request cap, the cap, and 1.5x the cap; 1 2 4 8 16 24 32 if unknown)")
    ap.add_argument("--prefill", nargs="+", type=int, default=[2000, 8000, 32000, 100000],
                    help="approx. prompt sizes for the prefill section")
    ap.add_argument("--interference-prompt", type=int, default=32000,
                    help="approx. size of the long prompt in the decode-during-prefill section")
    ap.add_argument("--interference-streams", nargs="+", type=int, default=[1, 4],
                    help="decode streams running while it is prefilled (default: 1 4)")
    ap.add_argument("--only", choices=["warmup", "ttft", "decode", "concurrency", "prefill",
                                       "interference"],
                    help="run a single section (warmup alone: prime a freshly started server)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the warmup pass (measure first-use JIT stalls on purpose)")
    args = ap.parse_args()
    if not args.levels:
        args.levels = default_levels()

    saved = common.save_output("perf")
    common.print_header()
    print("warmup...", flush=True)
    try:
        chat("hi", 8, stream=True)  # streamed: also proves usage comes back in streams
    except Exception as e:  # noqa: BLE001
        sys.exit(f"warmup request failed: {e}")
    calibrate()  # before any measured interval, also with --no-warmup
    print("ok\n")
    m_start = metrics()
    # Warm up before any measured section (a cold server skews whichever
    # section runs first), unless --no-warmup asks for the cold numbers.
    if args.only == "warmup" or not args.no_warmup:
        warmup(args.levels)
    sections = {
        "ttft": ttft_probe,
        "decode": single_stream,
        "concurrency": lambda: concurrency(args.levels),
        "prefill": lambda: prefill(args.prefill),
        "interference": lambda: prefill_interference(args.interference_prompt,
                                                     args.interference_streams),
    }
    for name, fn in sections.items():
        if args.only in (None, name):
            fn()
    if restarted(m_start, metrics()):
        print("!! the engine restarted during this run — see the rows flagged above; "
              "re-run on a stable server before quoting anything")
    if saved:
        print(f"saved -> {saved}")


if __name__ == "__main__":
    main()
