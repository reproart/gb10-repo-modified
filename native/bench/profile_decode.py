#!/usr/bin/env python3
"""Where a decode step's time goes: a short GPU profile, summed per kernel group.

    python3 bench/profile_decode.py                 # profile the running server
    python3 bench/profile_decode.py --trace FILE    # summarize an existing trace

Asks the server (SGLang /start_profile) to record `--steps` scheduler steps
with the torch profiler, runs one streaming code answer while it records,
then reads the Chrome trace the server wrote and prints:

  * GPU busy time per kernel group (MoE experts, GDN, QSA / attention, the
    n-gram table gather, dense GEMMs, lm_head / sampling, copies, ...) and the
    top kernels by total time;
  * how much of the profiled span the GPU was idle (waiting on the CPU,
    launches or page faults outside kernels).

A kernel that reads host memory through the page tables (the PLE gather on
GB10) shows its fault time inside its own duration, so a slow table read
lands in the "ple gather" row.

The trace is written by the server process, so this script must run on the
same machine (the default output dir is /tmp/gb10-profile). Profiling slows
the steps it records; compare shares, not tok/s.
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict

# name regex -> group, first match wins
GROUPS = [
    ("ple gather", r"gather_ple|ple_rows|qwen4_ple|ngram"),
    ("norm / elementwise", r"rmsnorm|layer_norm|layernorm"),
    ("hyper-connections", r"hc_mix|hc_combine|hyper_conn"),
    ("GDN (linear attn)", r"delta_rule|chunk_gated|gated_delta|fused_recurrent|fla_|deltanet|causal_conv|conv1d|gdn|mamba|ssm"),
    ("QSA / attention", r"qsa|indexer|persistent_topk|flash|attn|attention|xqa|paged|merge_state"),
    ("MoE routing", r"moe_fused_gate|topk|router|route_radix|gating"),
    ("MoE experts", r"marlin_moe|moe|expert|groupproblemshape|fused_experts"),
    ("dense GEMM, FP8/FP4 Marlin", r"marlin"),
    ("dense GEMM, BF16", r"wmma|gemvx|gemv|bf16.*gemm|gemm.*bf16|cublas|nvjet|sm\d+_xmma"),
    ("dense GEMM, other", r"gemm|cutlass|matmul|mm_"),
    ("norm / elementwise", r"norm|rms|silu|gelu|act_and_mul|elementwise|vectorized|add_|mul_|copy_kernel|cat|index"),
    ("sampling / spec", r"sample|argmax|verify|accept|eagle|spec|logits|tree"),
    ("memcpy", r"memcpy|memset|copy"),
]


def group_of(name: str) -> str:
    low = name.lower()
    for group, pat in GROUPS:
        if re.search(pat, low):
            return group
    return "other"


def load_trace(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def summarize(trace: dict, top: int = 20) -> str:
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    kernels = [e for e in events
               if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kernels:
        return "no GPU kernel events in the trace (was the GPU activity recorded?)"
    by_group = defaultdict(float)
    by_name = defaultdict(lambda: [0.0, 0])
    intervals = []
    for e in kernels:
        dur = float(e.get("dur", 0.0))
        ts = float(e.get("ts", 0.0))
        name = e.get("name", "?")
        by_group[group_of(name) if e["cat"] == "kernel" else "memcpy"] += dur
        rec = by_name[name]
        rec[0] += dur
        rec[1] += 1
        intervals.append((ts, ts + dur))
    intervals.sort()
    span = intervals[-1][1] - intervals[0][0]
    busy, cur_s, cur_e = 0.0, *intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    total = sum(by_group.values())

    out = [f"profiled span {span / 1e3:.1f} ms, GPU busy {busy / 1e3:.1f} ms "
           f"({100 * busy / span:.0f}%), idle {100 * (1 - busy / span):.0f}%",
           f"{len(kernels)} kernels, {total / 1e3:.1f} ms of kernel time "
           "(overlapping streams can exceed busy time)", "",
           f"{'group':<22}{'ms':>10}{'share':>8}"]
    for g, d in sorted(by_group.items(), key=lambda kv: -kv[1]):
        out.append(f"{g:<22}{d / 1e3:>10.1f}{100 * d / total:>7.0f}%")
    out += ["", f"top {top} kernels:", f"{'ms':>9}{'calls':>7}{'avg us':>9}  name"]
    for name, (d, n) in sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]:
        out.append(f"{d / 1e3:>9.1f}{n:>7}{d / n:>9.1f}  [{group_of(name)}] {name[:110]}")
    return "\n".join(out)


def newest_trace(directory: str, since: float) -> str | None:
    files = [f for f in glob.glob(os.path.join(directory, "**", "*.trace.json*"), recursive=True)
             if os.path.getmtime(f) >= since]
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", help="summarize this trace file instead of profiling")
    ap.add_argument("--steps", type=int, default=40, help="scheduler steps to record (default 40)")
    ap.add_argument("--output-dir", default="/tmp/gb10-profile")
    ap.add_argument("--tokens", type=int, default=400, help="answer length while recording")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if args.trace:
        print(summarize(load_trace(args.trace), args.top))
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import urllib.request

    import common

    common.print_header()
    os.makedirs(args.output_dir, exist_ok=True)
    print("warmup...", flush=True)
    common.chat(common.CODE_PROMPT, 64, stream=True)
    started = time.time()
    body = json.dumps({"output_dir": args.output_dir, "num_steps": args.steps,
                       "activities": ["CPU", "GPU"]}).encode()
    req = urllib.request.Request(f"{common.ENGINE_ROOT}/start_profile", data=body,
                                 headers=common.HEADERS, method="POST")
    # start_profile returns when profiling starts; the request below provides the steps.
    t = threading.Thread(target=lambda: urllib.request.urlopen(req, timeout=600).read(), daemon=True)
    t.start()
    time.sleep(1.0)
    r = common.chat(common.CODE_PROMPT, args.tokens, stream=True)
    print(f"answer while profiling: {r['completion_tokens']} tokens in {r['e2e']:.1f} s "
          "(slowed by the profiler)")
    t.join(timeout=600)
    print("waiting for the trace file...", flush=True)
    path = None
    for _ in range(120):
        path = newest_trace(args.output_dir, started)
        if path:
            size = os.path.getsize(path)
            time.sleep(2.0)
            if os.path.getsize(path) == size:
                break
        time.sleep(1.0)
    if not path:
        sys.exit(f"no trace appeared in {args.output_dir} (server-side path; is the server local?)")
    print(f"trace: {path}\n")
    print(summarize(load_trace(path), args.top))


if __name__ == "__main__":
    main()
