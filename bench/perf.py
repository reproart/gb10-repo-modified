#!/usr/bin/env python3
"""Full performance suite: TTFT, single-stream decode, concurrency, prefill.

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
from common import (CODE_PROMPT, PeakWatch, chat, decode_rate, delta, metrics, ple_summary,
                    sidecar, spec_summary, unique_prompt)


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
          f"{'   ' + ple if ple else ''}\n")


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
        with PeakWatch() as watch:
            done, errors, wall = _parallel(n, lambda i: chat(
                f"[{uuid.uuid4().hex[:8]}] {CODE_PROMPT} Variant {i}.", tokens, stream=True))
        m1 = metrics()
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
        if agg > peak[1] and not flag:
            peak = (n, agg)
        print(f"   {n:>7} {wall:>8.1f} {agg:>7.1f} t/s {_fmt(_med(decode_rate(r) for r in done), '>9.1f')} t/s "
              f"{_fmt(_med(ttfts), '>8.2f', 's')} {_fmt(max(ttfts) if ttfts else None, '>8.2f', 's')} "
              f"{queue_col:>9} {_fmt(ple, '>10.1f')}{flag}")
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
        r = chat(unique_prompt(uuid.uuid4().hex, target), 8, stream=True)
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
              f"{'   ' + ple if ple else ''}")
        if hits:
            print("   !! prefix-cache hit on a unique prompt — the rate above excludes it")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8, 16, 24, 32],
                    help="concurrency levels (default: 1 2 4 8 16 24 32)")
    ap.add_argument("--prefill", nargs="+", type=int, default=[2000, 8000, 32000, 100000],
                    help="approx. prompt sizes for the prefill section")
    ap.add_argument("--only", choices=["warmup", "ttft", "decode", "concurrency", "prefill"],
                    help="run a single section (warmup alone: prime a freshly started server)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the warmup pass (measure first-use JIT stalls on purpose)")
    args = ap.parse_args()

    print(f"endpoint {common.BASE_URL}  model {common.MODEL}")
    if not metrics():
        print(f"note: {common.METRICS_URL} unreachable — queue / spec / prefix-cache "
              "columns show n/a (set GB10_METRICS_URL; SGLang needs --enable-metrics)")
    print("warmup...", flush=True)
    try:
        chat("hi", 8)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"warmup request failed: {e}")
    print("ok\n")
    # Warm up before any measured section (a cold server skews whichever
    # section runs first), unless --no-warmup asks for the cold numbers.
    if args.only == "warmup" or not args.no_warmup:
        warmup(args.levels)
    sections = {
        "ttft": ttft_probe,
        "decode": single_stream,
        "concurrency": lambda: concurrency(args.levels),
        "prefill": lambda: prefill(args.prefill),
    }
    for name, fn in sections.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
