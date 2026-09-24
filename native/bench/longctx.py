#!/usr/bin/env python3
"""Concurrent long-context test: N streams, each with its own ~CTX-token prompt.

    python3 bench/longctx.py
    python3 bench/longctx.py --ctx 60000 --streams 1 2 4 8

Env: GB10_BASE_URL / GB10_MODEL / GB10_API_KEY (+ optional GB10_METRICS_URL),
see bench/common.py. GB10_LONGCTX_TOKENS / GB10_LONGCTX_STREAMS set the
defaults of --ctx / --streams.

Every prompt is unique from its first token (and made of varied words), so
the prefix/radix cache cannot deduplicate them: you measure prefill cost and
KV capacity, not cache hits. Cached tokens are checked per row and a hit is
reported loudly instead of being assumed away.

Expect linear scaling: one long prefill fills the per-step token budget (SGLang
serialises prefill, one sequence per batch; vLLM chunks it), so concurrent long
prompts are prefilled one after another — TTFT grows linearly, s/stream stays
flat, and the waiting shows up as "queue" even below the running-request cap.

KV capacity: the engine only admits a request when its KV fits, so a full pool
shows up as fewer requests running at once (the rest wait in "queue"), rarely
as preemption; the peak-usage line under each row turns that into a capacity
estimate.

By default each request answers "OK" right after its prefill and frees its KV
before the next one is admitted, so total ctx can exceed the pool without
preemption — that tests prefill, not capacity. --gen N forces exactly N
generated tokens per stream (ignore_eos), so earlier streams stay resident
while later ones prefill: a real capacity test. N ~ 1000 is enough — while
another stream prefills, a resident stream only decodes a few hundred tokens.
The "gen" column shows what was actually generated (a gateway may drop
ignore_eos).

This is the thermally heaviest workload in the suite (sustained prefill
measured 74 °C against an 80 °C suspend threshold): with a local nvidia-smi
every row reports its peak temperature and lowest SM clock, and throttling is
flagged. The output is saved under results/runs/ like perf.py's.
"""
import argparse
import os
import statistics
import threading
import sys
import time
import uuid

import common
from common import (GpuWatch, PeakWatch, cache_hit_significant, calibrate, chat, delta, metrics,
                    restarted, unique_prompt)


def run_level(n, ctx, gen):
    forced = gen is not None
    out, errors = [None] * n, []
    tag = uuid.uuid4().hex[:8]  # fresh per row: no cross-row reuse either

    def worker(i):
        try:
            out[i] = chat(unique_prompt(f"{tag}-{i}", ctx), gen if forced else 24,
                          stream=True, ignore_eos=forced)
        except Exception as e:  # noqa: BLE001 - reported below
            errors.append(repr(e))

    m0 = metrics()
    t0 = time.perf_counter()
    with PeakWatch(every=2.0) as watch, GpuWatch() as gpu:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    wall = time.perf_counter() - t0
    return [r for r in out if r], errors, wall, m0, metrics(), watch.peak, gpu


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("GB10_LONGCTX_TOKENS", "83000")),
                    help="approx. prompt tokens per stream (default 83000)")
    ap.add_argument("--gen", type=int, default=None,
                    help="force exactly N generated tokens per stream (ignore_eos) to keep "
                         "KV resident — a capacity test; default: a short 'OK' answer")
    ap.add_argument("--streams", nargs="+", type=int,
                    default=[int(x) for x in os.environ.get("GB10_LONGCTX_STREAMS", "4,8,12").split(",")],
                    help="concurrency levels (default: 4 8 12)")
    args = ap.parse_args()

    saved = common.save_output("longctx")
    common.print_header(f"  ctx ~{args.ctx} tok/stream")
    print("warmup...", flush=True)
    try:
        chat("hi", 8, stream=True)  # streamed: also proves usage comes back in streams
    except Exception as e:  # noqa: BLE001
        sys.exit(f"warmup request failed: {e}")
    calibrate()
    chat(unique_prompt(uuid.uuid4().hex, 400), 16)
    print("ok\n")
    print(f"{'streams':>7} {'ctx each':>9} {'total ctx':>10} {'gen':>6} {'cached':>7} {'wall':>8} "
          f"{'TTFT p50':>9} {'TTFT max':>9} {'s/stream':>9} {'queue':>7} {'preempt':>8}")
    print("-" * 99)

    for n in args.streams:
        done, errors, wall, m0, m1, peak, gpu = run_level(n, args.ctx, args.gen)
        if restarted(m0, m1):
            print(f"{n:>7}  !! the engine restarted during this row (OOM or earlyoom, "
                  "then a systemd restart?) — its numbers are not comparable")
        if errors:
            print(f"{n:>7}  {len(errors)} request(s) failed, e.g. {errors[0][:120]}"
                  " — check the server log (journalctl -u gb10-sglang; earlyoom or OOM?)")
            if not done:
                continue
        ttfts = [r["ttft"] for r in done if r["ttft"] is not None]
        total = sum(r["prompt_tokens"] for r in done)
        reported = [r["cached_tokens"] for r in done if r["cached_tokens"] is not None]
        hits = sum(reported) if reported else delta(m0, m1, "pc_hits")
        q_sum, q_cnt = delta(m0, m1, "queue_sum"), delta(m0, m1, "queue_count")
        queue = q_sum / q_cnt if q_sum is not None and q_cnt else None
        pre = delta(m0, m1, "preempted")

        def f(x, spec, unit=""):
            return "n/a" if x is None else f"{x:{spec}}{unit}"

        gen_avg = statistics.mean(r["completion_tokens"] for r in done)
        print(f"{n:>7} {done[0]['prompt_tokens']:>9} {total:>10} {gen_avg:>6.0f} {f(hits, '.0f'):>7} {wall:>7.1f}s "
              f"{f(statistics.median(ttfts) if ttfts else None, '.1f', 's'):>9} "
              f"{f(max(ttfts) if ttfts else None, '.1f', 's'):>9} {wall / n:>8.1f}s "
              f"{f(queue, '.1f', 's'):>7} {f(pre, '.0f'):>8}")
        if "kv_usage" in peak:
            running = int(peak["running"]) if "running" in peak else None
            line = (f"        peak KV pool usage {100 * peak['kv_usage']:.1f}%, "
                    f"up to {running if running is not None else '?'} of {n} requests running at once")
            if running and peak["kv_usage"] > 0.05:
                # Admission waits until the KV fits, so a full pool shows up as
                # fewer running requests (the rest wait), not as preemption.
                # Estimate from the peak: running streams of this size (prompt
                # + generated; an upper bound — one may still be mid-prefill)
                # over the usage they produced.
                per = statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in done)
                line += (f" -> the pool holds ~{running / peak['kv_usage']:.1f} streams of "
                         f"~{per / 1e3:.0f}k (~{running * per / peak['kv_usage'] / 1e6:.2f}M tokens)")
            print(line)
        if str(gpu):
            print(f"        {gpu}")
        if cache_hit_significant(hits, total):
            print(f"        !! {hits:.0f} prefix-cache hit tokens on unique prompts — "
                  "the numbers above are optimistic")
        time.sleep(4)
    if saved:
        print(f"\nsaved -> {saved}")


if __name__ == "__main__":
    main()
