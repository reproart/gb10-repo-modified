#!/usr/bin/env python3
"""Where a decode step's time goes: a short GPU profile, summed per kernel group.

    python3 bench/profile_decode.py                 # profile the running server
    python3 bench/profile_decode.py --trace FILE    # summarize an existing trace

Asks the server to record with the torch profiler while it runs one
streaming code answer, then reads the Chrome trace the server wrote and
prints:

  * GPU busy time per kernel group (MoE experts, GDN, QSA / attention, the
    n-gram table gather, dense GEMMs, lm_head / sampling, copies, ...) and the
    top kernels by total time;
  * how much of the profiled span the GPU was idle (waiting on the CPU,
    launches or page faults outside kernels).

"ms" is the kernels' own durations. SGLang runs some work on a second
stream under CUDA graphs (on Qwen4-Exp the MoE router beside the shared
expert, the QSA indexer beside the qkv projection), and a kernel sharing the
GPU runs longer without making the step longer. "wall" splits every
stretch of GPU time evenly among the kernels running in it, so it adds up
to the busy time: a group whose wall is well below its ms is mostly
overlapped, and a faster kernel there buys little.

A kernel that reads host memory through the page tables (the PLE gather on
GB10) shows its fault time inside its own duration, so a slow table read
lands in the "ple gather" row.

SGLang (/get_server_info answers): /start_profile records `--steps`
scheduler steps into --output-dir. vLLM: the profiler endpoints exist only
when the server was started with a profiler config, which also fixes the
trace directory (vllm-b12x: PROFILE=1 ./serve.sh, traces to
/tmp/gb10-profile on the host); this script brackets the answer with
/start_profile and /stop_profile, so `--tokens` sets the length (with MTP,
~4 tokens a step: 160 tokens ~ 40 steps).

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
    # cuBLASLt names its FP8 kernels nvjet_smXXX_qq... (q: e4m3 operands),
    # e.g. a ModelOpt FP8 layer through torch._scaled_mm; tst... is BF16
    ("dense GEMM, FP8 (cuBLASLt)", r"nvjet_sm\d+_q"),
    ("dense GEMM, NVFP4", r"e2m1|nvfp4"),
    ("dense GEMM, BF16", r"wmma|gemvx|gemv|skinny_gemm|bf16.*gemm|gemm.*bf16|cublas|nvjet|sm\d+_xmma"),
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


def wall_shares(kernels) -> tuple[dict, dict]:
    """Split GPU time among the kernels running at each moment: a stretch with
    n kernels in flight gives each 1/n of it. Returns (per group, per name) in
    the trace's time unit; each sums to the GPU busy time."""
    edges = []
    for i, e in enumerate(kernels):
        ts, dur = float(e.get("ts", 0.0)), float(e.get("dur", 0.0))
        name = e.get("name", "?")
        group = group_of(name) if e["cat"] == "kernel" else "memcpy"
        edges.append((ts, 1, group, name))
        edges.append((ts + dur, -1, group, name))
    edges.sort(key=lambda x: (x[0], x[1]))  # ends before starts at one instant
    by_group, by_name = defaultdict(float), defaultdict(float)
    active_group, active_name = defaultdict(int), defaultdict(int)
    n_active, prev = 0, None
    for t, step, group, name in edges:
        if n_active and prev is not None and t > prev:
            share = (t - prev) / n_active
            for g, c in active_group.items():
                by_group[g] += share * c
            for k, c in active_name.items():
                by_name[k] += share * c
        prev = t
        n_active += step
        for d, key in ((active_group, group), (active_name, name)):
            d[key] += step
            if not d[key]:
                del d[key]
    return by_group, by_name


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
    wall_group, wall_name = wall_shares(kernels)
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
           f"{'group':<27}{'ms':>9}{'wall ms':>9}{'share':>7}"]
    wall_total = sum(wall_group.values()) or 1.0
    for g, d in sorted(by_group.items(), key=lambda kv: -wall_group.get(kv[0], 0.0)):
        w = wall_group.get(g, 0.0)
        out.append(f"{g:<27}{d / 1e3:>9.1f}{w / 1e3:>9.1f}{100 * w / wall_total:>6.0f}%")
    out += ["", f"top {top} kernels (by wall time):",
            f"{'ms':>9}{'wall ms':>9}{'calls':>7}{'avg us':>9}  name"]
    for name, (d, n) in sorted(by_name.items(), key=lambda kv: -wall_name.get(kv[0], 0.0))[:top]:
        out.append(f"{d / 1e3:>9.1f}{wall_name.get(name, 0.0) / 1e3:>9.1f}{n:>7}{d / n:>9.1f}"
                   f"  [{group_of(name)}] {name[:100]}")
    return "\n".join(out)


def newest_trace(directory: str, since: float) -> str | None:
    """The largest trace written since `since`: the engine's, when the server
    also leaves a smaller CPU-only one (vLLM's API process)."""
    files = [f for f in glob.glob(os.path.join(directory, "**", "*.trace.json*"), recursive=True)
             if os.path.getmtime(f) >= since]
    return max(files, key=os.path.getsize) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", help="summarize this trace file instead of profiling")
    ap.add_argument("--steps", type=int, default=40, help="scheduler steps to record (default 40)")
    ap.add_argument("--output-dir", default="/tmp/gb10-profile")
    ap.add_argument("--tokens", type=int, default=None,
                    help="answer length while recording (default 400 on SGLang, 160 on vLLM)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    if args.trace:
        print(summarize(load_trace(args.trace), args.top))
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import urllib.error
    import urllib.request

    import common

    common.print_header()
    sglang = bool(common.server_info())
    os.makedirs(args.output_dir, exist_ok=True)
    print("warmup...", flush=True)
    common.chat(common.CODE_PROMPT, 64, stream=True)
    started = time.time()

    def post(path, body=None, timeout=600):
        req = urllib.request.Request(f"{common.ENGINE_ROOT}{path}", data=body or b"",
                                     headers=common.HEADERS, method="POST")
        return urllib.request.urlopen(req, timeout=timeout).read()

    if sglang:
        tokens = args.tokens or 400
        body = json.dumps({"output_dir": args.output_dir, "num_steps": args.steps,
                           "activities": ["CPU", "GPU"]}).encode()
        # start_profile returns when profiling starts; the request below provides the steps.
        t = threading.Thread(target=lambda: post("/start_profile", body), daemon=True)
        t.start()
        time.sleep(1.0)
    else:
        tokens = args.tokens or 160
        try:
            post("/start_profile")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                sys.exit("the server has no /start_profile: vLLM opens it only with a profiler "
                         "config at launch (vllm-b12x: PROFILE=1 ./serve.sh)")
            raise
    r = common.chat(common.CODE_PROMPT, tokens, stream=True)
    print(f"answer while profiling: {r['completion_tokens']} tokens in {r['e2e']:.1f} s "
          "(slowed by the profiler)")
    if sglang:
        t.join(timeout=600)
    else:
        print("stopping the profiler (vLLM writes the trace now)...", flush=True)
        try:
            post("/stop_profile")
        except (TimeoutError, urllib.error.URLError) as e:
            print(f"/stop_profile did not answer ({e}); looking for the trace anyway", flush=True)
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
