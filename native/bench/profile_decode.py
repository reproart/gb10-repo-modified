#!/usr/bin/env python3
"""Where a decode step's time goes: a short GPU profile, summed per kernel group.

    python3 bench/profile_decode.py                 # profile the running server
    python3 bench/profile_decode.py --prefill 16384 # profile a prefill instead
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
trace directory (vllm-b12x and vllm-flash-ar: PROFILE=1 ./serve.sh, traces
to /tmp/gb10-profile on the host); this script brackets the answer with
/start_profile and /stop_profile, so `--tokens` sets the length (with MTP,
~4 tokens a step: 160 tokens ~ 40 steps).

Which layer a kernel belongs to: with shapes recorded (vllm-flash-ar:
PROFILE=1 PROFILE_SHAPES=1 ./serve.sh, which also runs vLLM eagerly, since a
kernel replayed from a CUDA graph has no launching op in the trace), each top
kernel lists the op that launched it and its input shapes, e.g.
"aten::mm [[1,2560],[2560,65536]]": the weight's size tells the layer. A
second table then sums the time per op and shapes (one row per layer kind),
with the weight a GEMM reads and the rate it reads it at: a decode GEMM
near the memory's bandwidth (~240 GB/s on a GB10) can only get faster by
reading fewer bytes, one far below it by a better kernel. --trace re-reads a
trace already written, e.g. after updating this script.

--prefill N profiles a prefill instead of decode steps: one prompt of ~N tokens
that nothing in the prefix cache shares, answered with one token. On SGLang
the recording covers all but the last chunk of it (--steps defaults to
ceil(N / chunked_prefill_size) - 1, read from /get_server_info), so every
recorded step is a full chunk; on vLLM the profiler brackets the request.

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
    ("hyper-connections", r"hc_mix|hc_combine|hyper_?conn"),
    # vLLM b12x: the MTP head's own prefill GEMMs (BF16) before any GEMM rule
    ("MTP draft", r"mtp_feedback|mtpprefill"),
    ("GDN (linear attn)", r"delta_rule|chunk_gated|gated_delta|fused_recurrent|fla_|deltanet|causal_conv|conv1d|gdn|mamba|ssm"),
    ("QSA / attention", r"qsa|indexer|persistent_topk|flash|attn|attention|xqa|paged|merge_state"),
    ("MoE routing", r"moe_fused_gate|topk|router|route_radix|gating"),
    ("MoE experts", r"marlin_moe|moe|expert|groupproblemshape|fused_experts"),
    # Marlin dense: FP8 / FP4 side layers on SGLang, the int8 lm_head on vllm-flash-ar
    ("dense GEMM, Marlin", r"marlin"),
    # vLLM's blockwise FP8 (vllm-flash-ar's side layers: 128x128 block scales)
    ("dense GEMM, FP8 block-scaled", r"w8a8|block_?scaled_mm|blockwise|block_fp8|fp8_block"),
    # vLLM b12x's CuTe DSL dense GEMM and its weight-only split-K reduce: the
    # block-scaled linear layers (on FP4-Local the MXFP8 attention / GDN /
    # shared-expert projections and the MXFP8 lm_head). Its names carry
    # "bf16" (the activations), so it must come before the BF16 rule.
    ("dense GEMM, b12x block-scaled", r"b12x_libdense_gemm|b12xgemmblockscaled|weightonlysplitk"),
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


# GEMM ops whose weight is one input: its index in "Input Dims". A decode-size
# GEMM reads the whole weight once and does little else, so bytes / time is how
# close it runs to the memory's bandwidth (GB10: 273 GB/s on paper, ~240 in
# practice). MoE expert GEMMs are left out: they read only the routed experts.
WEIGHT_ARG = {"aten::mm": 1, "aten::addmm": 2, "aten::bmm": 1,
              "_C::cutlass_scaled_mm": 2, "_C::marlin_gemm": 2, "_C::gptq_marlin_gemm": 2,
              "gb10::fp8_row_linear": 1}  # (vllm-flash-ar HC_FP8: its FP8 rows; the BF16 tail not counted)
DTYPE_BYTES = [("float8", 1), ("bfloat16", 2), ("half", 2), ("double", 8), ("float", 4),
               ("long", 8), ("short", 2), ("char", 1), ("byte", 1), ("bool", 1), ("int", 4)]


def weight_bytes(op: str, dims: list, types: list) -> int | None:
    i = WEIGHT_ARG.get(op)
    if i is None or i >= len(dims) or not dims[i]:
        return None
    t = (types[i] if i < len(types) else "").lower()
    size = next((b for k, b in DTYPE_BYTES if k in t), None)
    if size is None:
        return None
    n = 1
    for d in dims[i]:
        n *= int(d)
    return n * size


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
    # the op that launched a kernel, with its input shapes (record_shapes): same "External id"
    ops, wbytes = {}, {}
    for e in events:
        a = e.get("args") or {}
        if e.get("ph") == "X" and e.get("cat") == "cpu_op" and "Input Dims" in a and "External id" in a:
            op = f"{e.get('name', '?')} " + json.dumps(a["Input Dims"], separators=(",", ":"))
            ops[a["External id"]] = op
            if op not in wbytes:
                wbytes[op] = weight_bytes(e.get("name", "?"), a["Input Dims"], a.get("Input type") or [])
    shapes = defaultdict(lambda: defaultdict(int))
    # per launching op + shapes: the time of all the kernels it launched, and how
    # many times it ran (an op launching a GEMM and its split-K reduce runs once)
    layer_time, layer_runs = defaultdict(float), defaultdict(set)
    for e in kernels:
        xid = (e.get("args") or {}).get("External id")
        op = ops.get(xid)
        if op:
            shapes[e.get("name", "?")][op] += 1
            layer_time[op] += float(e.get("dur", 0.0))
            layer_runs[op].add(xid)
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
        for op, c in sorted(shapes.get(name, {}).items(), key=lambda kv: -kv[1])[:3]:
            out.append(f"{'':>36}<- {c:>5} x {op[:110]}")
    if not ops:
        out += ["", "(no op shapes in this trace: PROFILE_SHAPES=1 records them, in eager mode)"]
        return "\n".join(out)
    out += ["", f"top {top} layers (op + input shapes; kernel time of everything the op launched;",
            "weight = the GEMM's weight operand, GB/s = weight / avg: near ~240 the layer is",
            "bandwidth-bound, far below it launch- or tile-bound):",
            f"{'ms':>9}{'calls':>7}{'avg us':>9}{'weight MB':>11}{'GB/s':>7}  op"]
    for op, d in sorted(layer_time.items(), key=lambda kv: -kv[1])[:top]:
        n = len(layer_runs[op])
        b = wbytes.get(op)
        bw = f"{b / (d / n * 1e3):>7.0f}" if b and d else f"{'':>7}"
        mb = f"{b / 1e6:>11.1f}" if b else f"{'':>11}"
        out.append(f"{d / 1e3:>9.1f}{n:>7}{d / n:>9.1f}{mb}{bw}  {op[:100]}")
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
    ap.add_argument("--steps", type=int, default=None,
                    help="scheduler steps to record (default 40; with --prefill, its full chunks)")
    ap.add_argument("--output-dir", default="/tmp/gb10-profile")
    ap.add_argument("--tokens", type=int, default=None,
                    help="answer length while recording (default 400 on SGLang, 160 on vLLM)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--prefill", type=int, default=0, metavar="N",
                    help="profile the prefill of a unique ~N-token prompt instead of decode")
    args = ap.parse_args()

    if args.trace:
        print(summarize(load_trace(args.trace), args.top))
        return

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import urllib.error
    import urllib.request

    import common

    common.print_header()
    info = common.server_info()
    sglang = bool(info)
    os.makedirs(args.output_dir, exist_ok=True)
    print("warmup...", flush=True)
    common.chat(common.CODE_PROMPT, 64, stream=True)
    steps = args.steps or 40
    if args.prefill:
        prompt = common.unique_prompt(f"profile-{time.time_ns()}", args.prefill)
        if sglang and not args.steps:
            chunk = int(info.get("chunked_prefill_size") or 0)
            steps = max(1, -(-args.prefill // chunk) - 1) if chunk > 0 else 1
            print(f"prefill of ~{args.prefill} tokens, chunks of {chunk or '?'}: recording "
                  f"{steps} step(s)", flush=True)
    started = time.time()

    def post(path, body=None, timeout=600):
        req = urllib.request.Request(f"{common.ENGINE_ROOT}{path}", data=body or b"",
                                     headers=common.HEADERS, method="POST")
        return urllib.request.urlopen(req, timeout=timeout).read()

    if sglang:
        tokens = args.tokens or 400
        body = json.dumps({"output_dir": args.output_dir, "num_steps": steps,
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
    if args.prefill:
        r = common.chat(prompt, 1, stream=True)
        pt = r.get("prompt_tokens") or args.prefill
        print(f"prefill while profiling: {pt} tokens, TTFT {r['ttft']:.1f} s, "
              f"{pt / max(r['ttft'], 1e-9):.0f} tok/s (slowed by the profiler)")
    else:
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
