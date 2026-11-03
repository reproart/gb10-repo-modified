#!/usr/bin/env python3
"""GPU check and timing for gb10_fp8_rowwise.py (FP8 hyper-connection projections), in the image.

While the server has no requests (or is stopped; it needs ~1 GB and a few minutes):

    docker run --rm --gpus all --entrypoint python3 -v "$PWD/src:/w:ro" qwen38-flash-dgx \\
        /w/bench_fp8_rowwise_gpu.py

For the two hyper-connection projections of Qwen3.8-Flash-Next (down + block_inject
[336 x 10240], the 16 block_inject rows kept BF16; up [10240 x 320]):
1. correctness, BF16 activations: the Triton kernel (1-64 rows) and the torch path (512)
   against x @ W.T for the dequantized weight; the same bits on every run; a row's result
   the same alone as in a batch of up to 16 rows;
2. time per call at decode sizes, replayed from a CUDA graph as the server runs them, each
   call on another copy of the weight (~300 MB of copies: the model reads ~98 different
   mixes a step from memory, so a weight left in the GPU's caches by the call before would
   time the cache, not the memory): cuBLAS on BF16 (what runs without HC_FP8) against the
   FP8 kernel, the rate each reads its weight at, and the saving per decode step at
   1 stream (4 rows, ~98 mixes);
3. prefill sizes (2048 / 8192 rows), eager: cuBLAS on BF16 against the prefill path
   (the weight dequantized to BF16 by one Triton pass, then cuBLAS);
4. a sweep of the kernel's tile / split settings at 4 rows, against the default;
5. what the server relies on at load: prewarm() builds every row tile and the dequantize
   (the status line must name no torch path), and whether a kernel variant launched for
   the first time inside a CUDA graph capture loads there (if not, prewarm is what keeps
   decode on the kernel).
Exit code: the number of FAIL lines. FP8ROW_DRY=1: a dry run of this script on the CPU
(Triton's interpreter, FP16 since it cannot dot BF16; times mean nothing).
"""
import os
import sys
import time

DRY = os.environ.get("FP8ROW_DRY") == "1"
if DRY:
    os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_fp8_rowwise as fr  # noqa: E402

dev = torch.device("cpu" if DRY else "cuda")
DT = torch.float16 if DRY else torch.bfloat16
torch.manual_seed(0)
SMS = 48 if DRY else torch.cuda.get_device_properties(0).multi_processor_count
print(f"{'CPU dry run' if DRY else torch.cuda.get_device_name(0)}, {SMS} SMs; torch {torch.__version__}")
SHAPES = [("hc down + block_inject", 336, 320, 10240), ("hc up", 10240, 10240, 320)]
COPY_BYTES = 300e6  # BF16 copies of a weight cycled through per timing (well past the caches)
if DRY:
    SHAPES = [("hc down + block_inject (small)", 336, 320, 1024), ("hc up (small)", 1024, 1024, 320)]
    COPY_BYTES = 2e6
MIXES = 98  # a decode step at 1 stream: 2 mixes x 48 layers + the final mixer + the MTP layer's
fails = 0


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def make(n, n8, k):
    w = (torch.randn(n, k, device=dev) * 0.02).to(DT)
    q, s = fr.quantize_per_row(w[:n8])
    return w, q, s, w[n8:].contiguous()


def graph_us(fn, calls, reps=5):
    """Time of one call of fn(i) for i in range(calls), replayed from a CUDA graph."""
    def body():
        for i in range(calls):
            fn(i)
    if DRY:
        t = time.perf_counter()
        body()
        return (time.perf_counter() - t) * 1e6 / calls
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    g.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (calls * reps)


def eager_us(fn, reps=10):
    if DRY:
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1e6
    for _ in range(2):
        fn()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / reps


print("\n1. correctness (BF16 activations)")
for name, n, n8, k in SHAPES:
    w, q, s, tail = make(n, n8, k)
    wd = fr.dequantize(q, s, tail)
    xb = torch.randn(64, k, device=dev).to(DT)
    for m in ((1, 4, 20, 512) if DRY else (1, 4, 8, 16, 32, 64, 512)):
        x = xb[:m] if m <= 64 else torch.randn(m, k, device=dev).to(DT)
        ref = x.float() @ wd.T
        path = "kernel" if m <= fr.MAX_ROWS else "torch"
        run = (lambda: fr.triton_mm(x, q, s, tail)) if m <= fr.MAX_ROWS else (lambda: fr.torch_mm(x, q, s, tail))
        y = run()
        same = all(torch.equal(y, run()) for _ in range(3))
        alone = torch.equal(fr.triton_mm(x[:1], q, s, tail)[0], y[0]) if 1 < m <= 16 else True
        op = torch.ops.gb10.fp8_row_linear(x, q, s, tail)
        ok = rel(y, ref) < 1e-2 and same and alone and (DRY or torch.equal(op, y))  # (on the CPU the op takes torch)
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'} {name}, {m:3d} rows ({path}): rel {rel(y, ref):.4f} to the FP8 "
              f"weight, {rel(y, x.float() @ w.float().T):.4f} to the BF16 one; same bits 3 runs: {same}"
              + (f"; alone = in the batch: {alone}" if 1 < m <= 16 else ""))
    dq = fr.dequant_weight(q, s, tail, DT)
    dq_ok = DRY or ("dequant" not in fr._state["bad_tiles"] and torch.equal(dq, fr.dequant_weight(q.cpu(), s.cpu(), tail.cpu(), DT).to(dev)))
    fails += not dq_ok
    print(f"  {'PASS' if dq_ok else 'FAIL'} {name}, the prefill path's dequantize kernel = torch's conversion, bit for bit")

print("\n2. decode sizes: time per call, CUDA graph, a different weight copy each call (from memory)")
saving = {}
copies = {}
for name, n, n8, k in SHAPES:
    c = max(4, int(COPY_BYTES / (n * k * 2)))
    copies[name] = [make(n, n8, k) for _ in range(c)]
    cs = copies[name]
    bf16_bytes = n * k * 2
    fp8_bytes = cs[0][1].numel() + cs[0][2].numel() * 4 + cs[0][3].numel() * 2
    for m in ((4,) if DRY else (1, 4, 8, 16, 32, 64)):
        x = torch.randn(m, k, device=dev).to(DT)
        t_b = graph_us(lambda i: F.linear(x, cs[i][0]), len(cs))
        t_f = graph_us(lambda i: fr.triton_mm(x, cs[i][1], cs[i][2], cs[i][3]), len(cs))
        if m == 4:
            saving[name] = (t_b, t_f)
        print(f"  {name:<23}{m:4d} rows: BF16 {t_b:6.1f} us ({bf16_bytes / t_b / 1e3:4.0f} GB/s), "
              f"FP8 {t_f:6.1f} us ({fp8_bytes / t_f / 1e3:4.0f} GB/s), x{t_b / t_f:.2f}  ({len(cs)} copies)")
tb = sum(v[0] for v in saving.values())
tf = sum(v[1] for v in saving.values())
print(f"  a mix at 4 rows (1 stream with MTP): BF16 {tb:.1f} us, FP8 {tf:.1f} us; x{MIXES} mixes: "
      f"{(tb - tf) * MIXES / 1000:+.2f} ms a decode step saved (positive = faster)")

print("\n3. prefill sizes: time per call, eager (cuBLAS on BF16 vs dequantize + cuBLAS)")
for name, n, n8, k in SHAPES:
    cs = copies[name][:4]
    for m in ((96,) if DRY else (2048, 8192)):
        x = torch.randn(m, k, device=dev).to(DT)
        i = [0]

        def nxt():
            i[0] = (i[0] + 1) % len(cs)
            return cs[i[0]]
        t_b = eager_us(lambda: F.linear(x, nxt()[0]))
        t_f = eager_us(lambda: fr.torch_mm(x, *nxt()[1:]))
        t_d = eager_us(lambda: fr.dequant_weight(*nxt()[1:], DT))
        print(f"  {name:<23}{m:5d} rows: BF16 {t_b:7.1f} us, FP8 (dequantize + cuBLAS) {t_f:7.1f} us, "
              f"{t_f - t_b:+.1f} us a call (the dequantize alone {t_d:.1f} us)")
        del x

print("\n4. kernel settings at 4 rows, from memory as in 2: (BLOCK_N, BLOCK_K, SPLIT) -> us")
for name, n, n8, k in SHAPES:
    cs = copies[name]
    x = torch.randn(4, k, device=dev).to(DT)
    default = fr.config(n, k, n8, SMS)

    def timed(cfg):
        return graph_us(lambda i: fr.triton_mm(x, cs[i][1], cs[i][2], cs[i][3], cfg), len(cs), reps=3)
    results = [(default, timed(default))]
    for bn in ((16, 64) if DRY else (16, 32, 64, 128)):
        if n8 < n and n8 % bn:
            continue
        for bk in ((256,) if DRY else (64, 128, 256) if k >= 1024 else (32, 64, 128)):
            steps = -(-k // bk)
            for split in ((1, 2) if DRY else (1, 2, 4, 8, 12, 16, 24)):
                if split > steps:
                    continue
                per = -(-steps // split)
                cfg = (bn, bk, -(-steps // per), per * bk)
                if any(c == cfg for c, _ in results):
                    continue
                try:
                    results.append((cfg, timed(cfg)))
                except Exception as e:  # noqa: BLE001 - a tile this GPU cannot build
                    print(f"  {name}: {cfg} failed: {type(e).__name__}")
    d = results[0][1]
    results.sort(key=lambda r: r[1])
    best = ", ".join(f"{c[:3]} {t:.1f}" for c, t in results[:5])
    print(f"  {name}: default {default[:3]} {d:.1f} us; best {best}")

print("\n5. load-time prewarm, and a first launch inside a CUDA graph capture")
for name, n, n8, k in SHAPES:
    w, q, s, tail = make(n, n8, k)
    fr.prewarm(q, s, tail, DT)
ok = not fr._state["bad_tiles"]
fails += not ok
print(f"  {'PASS' if ok else 'FAIL'} after prewarm: {fr.status()}")
if not DRY:
    name, n, n8, k = SHAPES[0]
    w, q, s, tail = make(n, n8, k)
    x = torch.randn(4, k, device=dev).to(DT)
    cfg = (16, 128, 5, 2048)  # a variant nothing above built
    err = None
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fr.triton_mm(x, q, s, tail, cfg)
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - the point of the probe
        err = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    print(f"  INFO first launch of a new variant inside a capture: {'loads fine' if err is None else 'FAILS (' + err + ')'}"
          " (prewarm keeps the server off this path either way)")

print("\nALL OK" if not fails else f"\n{fails} FAILED")
sys.exit(fails)
