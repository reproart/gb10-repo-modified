#!/usr/bin/env python3
"""GPU check and timing for gb10_fp8_rowwise.py (FP8 hyper-connection projections), in the image.

While the server has no requests (or is stopped; it needs ~0.2 GB and a few minutes):

    docker run --rm --gpus all --entrypoint python3 -v "$PWD/src:/w:ro" qwen38-flash-dgx \\
        /w/bench_fp8_rowwise_gpu.py

For the two hyper-connection projections of Qwen3.8-Flash-Next (down + block_inject
[336 x 10240], the 16 block_inject rows kept BF16; up [10240 x 320]):
1. correctness, BF16 activations: the Triton kernel (1-64 rows) and the torch path (512)
   against x @ W.T for the dequantized weight; the same bits on every run; a row's result
   the same alone as in a batch of up to 16 rows;
2. time per call, kernels replayed from a CUDA graph as the server runs them: cuBLAS on
   the BF16 weight (what runs now) against the FP8 kernel, and the rate each reads its
   weight at; the saving per decode step at 1 stream (4 rows; ~98 mixes a step);
3. a sweep of the kernel's tile / split settings at 4 rows, against the default.
Exit code: the number of FAIL lines. FP8ROW_DRY=1: a dry run of this script on the CPU
(Triton's interpreter, FP16 since it cannot dot BF16; wall-clock times mean nothing).
"""
import os
import sys
import time

DRY = os.environ.get("FP8ROW_DRY") == "1"
if DRY:
    os.environ["TRITON_INTERPRET"] = "1"

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_fp8_rowwise as fr  # noqa: E402

dev = torch.device("cpu" if DRY else "cuda")
DT = torch.float16 if DRY else torch.bfloat16
torch.manual_seed(0)
SMS = 48 if DRY else torch.cuda.get_device_properties(0).multi_processor_count
print(f"{'CPU dry run' if DRY else torch.cuda.get_device_name(0)}, {SMS} SMs; torch {torch.__version__}")
SHAPES = [("hc down + block_inject", 336, 320, 10240), ("hc up", 10240, 10240, 320)]
if DRY:
    SHAPES = [("hc down + block_inject (small)", 336, 320, 1024), ("hc up (small)", 1024, 1024, 320)]
MIXES = 98  # a decode step at 1 stream: 2 mixes x 48 layers + the final mixer + the MTP layer's
fails = 0


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def graph_us(fn, calls=50, reps=10):
    if DRY:
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1e6
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    g.replay()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (calls * reps)


weights = {}
for name, n, n8, k in SHAPES:
    w = (torch.randn(n, k, device=dev) * 0.02).to(DT)
    q, s = fr.quantize_per_row(w[:n8])
    tail = w[n8:].contiguous()
    weights[name] = (w, q, s, tail)

print("\n1. correctness (BF16 activations)")
for name, n, n8, k in SHAPES:
    w, q, s, tail = weights[name]
    wd = fr.dequantize(q, s, tail)
    xb = torch.randn(64, k, device=dev).to(DT)
    for m in ((1, 4, 20, 512) if DRY else (1, 4, 8, 16, 32, 64, 512)):
        x = xb[:m] if m <= 64 else torch.randn(m, k, device=dev).to(DT)
        ref = x.float() @ wd.T
        path = "kernel" if m <= fr.MAX_ROWS else "torch"
        y = fr.triton_mm(x, q, s, tail) if m <= fr.MAX_ROWS else fr.torch_mm(x, q, s, tail)
        same = all(torch.equal(y, fr.triton_mm(x, q, s, tail) if m <= fr.MAX_ROWS else fr.torch_mm(x, q, s, tail))
                   for _ in range(3))
        alone = True
        if 1 < m <= 16:
            alone = torch.equal(fr.triton_mm(x[:1], q, s, tail)[0], y[0])
        op = torch.ops.gb10.fp8_row_linear(x, q, s, tail)
        ok = rel(y, ref) < 1e-2 and same and alone and (DRY or torch.equal(op, y))  # (on the CPU the op takes torch)
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'} {name}, {m:3d} rows ({path}): rel {rel(y, ref):.4f} to the FP8 "
              f"weight, {rel(y, x.float() @ w.float().T):.4f} to the BF16 one; same bits 3 runs: {same}"
              + (f"; alone = in the batch: {alone}" if 1 < m <= 16 else ""))

print("\n2. time per call (CUDA graph replay), cuBLAS BF16 vs the FP8 kernel")
saving = 0.0
for name, n, n8, k in SHAPES:
    w, q, s, tail = weights[name]
    bf16_bytes = w.numel() * 2
    fp8_bytes = q.numel() + s.numel() * 4 + tail.numel() * 2
    for m in ((4,) if DRY else (1, 4, 8, 16, 32, 64, 512)):
        x = torch.randn(m, k, device=dev).to(DT)
        t_b = graph_us(lambda: F.linear(x, w))
        if m <= fr.MAX_ROWS:
            t_f = graph_us(lambda: fr.triton_mm(x, q, s, tail))
        else:
            t_f = graph_us(lambda: fr.torch_mm(x, q, s, tail))
        if m == 4:
            saving += (t_b - t_f) * MIXES
        print(f"  {name:<23}{m:4d} rows: BF16 {t_b:7.1f} us ({bf16_bytes / t_b / 1e3:4.0f} GB/s), "
              f"FP8 {t_f:7.1f} us ({fp8_bytes / t_f / 1e3:4.0f} GB/s), x{t_b / t_f:.2f}")
print(f"  at 4 rows (1 stream with MTP), ~{MIXES} mixes a step: {saving / 1000:+.2f} ms a decode step "
      "saved (positive = faster)")

print("\n3. kernel settings at 4 rows: (BLOCK_N, BLOCK_K, SPLIT) -> us")
for name, n, n8, k in SHAPES:
    w, q, s, tail = weights[name]
    x = torch.randn(4, k, device=dev).to(DT)
    default = fr.config(n, k, n8, SMS)
    results = [(default, graph_us(lambda: fr.triton_mm(x, q, s, tail, default), calls=20, reps=5))]
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
                    results.append((cfg, graph_us(lambda c=cfg: fr.triton_mm(x, q, s, tail, c), calls=20, reps=5)))
                except Exception as e:  # noqa: BLE001 - a tile this GPU cannot build
                    print(f"  {name}: {cfg} failed: {type(e).__name__}")
    results.sort(key=lambda r: r[1])
    d = next(t for c, t in results if c == default)
    best = ", ".join(f"{c[:3]} {t:.1f}" for c, t in results[:4])
    print(f"  {name}: default {default[:3]} {d:.1f} us; best {best}")

print("\nALL OK" if not fails else f"\n{fails} FAILED")
sys.exit(fails)
