#!/usr/bin/env python3
"""GPU check for patch_det_moe_align.py, in the image, against vLLM's CUDA kernel.

With the server stopped (it needs the GPU for a few seconds):

    docker run --rm --gpus all --entrypoint python3 -v "$PWD/src:/w:ro" qwen38-flash-dgx \
        /w/test_det_moe_align_gpu.py

For each batch (Qwen3.8-Flash-Next: 512 experts, top-10; Marlin's block sizes; some
padding tokens with -1 and ids past the last expert, as vLLM passes them):
1. the same layout as ops.moe_align_block_size: the padded total, every block's expert
   id, the -1 blocks after them, the same routes in each expert's blocks, numel in the
   rest. Only the order inside an expert may differ, and that is the point.
2. the deterministic path gives one layout in every run; how many of the kernel's runs
   moved a route to another block than in its first run (the variation that changes
   Marlin's sums and that the patch removes), and how many only reordered routes inside
   a block (harmless: a block's rows are computed alike wherever they sit). 0 for a
   batch does not prove the kernel stable; more than 0 shows it is not;
3. time per call of both (CUDA events).
Then, on an image built with the patch, moe_align_block_size itself with the switch on:
the wrapper takes the deterministic path for a prefill batch and the kernel for a
decode one.
"""
import importlib
import os
import sys
import time

import torch

from vllm import _custom_ops as ops

E, TOPK = 512, 10
mod = importlib.import_module("vllm.model_executor.layers.fused_moe.moe_align_block_size")
if hasattr(mod, "_deterministic_align"):
    det, where = mod._deterministic_align, "the image's patched moe_align_block_size.py"
else:  # an image built before the patch: the helper from the patcher next to this file
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import patch_det_moe_align as pd
    ns = {}
    exec(compile(pd.HELPER, "helper", "exec"), ns)
    det, where = ns["_deterministic_align"], "patch_det_moe_align.py (this image is not patched)"
dev = torch.device(os.environ.get("DET_ALIGN_TEST_DEVICE", "cuda"))  # cpu: a dry run of this script
print(f"{torch.cuda.get_device_name(0) if dev.type == 'cuda' else dev}; helper from {where}")


def buffers(topk_ids, block):
    n = topk_ids.numel()
    cap = n + E * (block - 1)
    if n < E:
        cap = min(n * block, cap)
    return (torch.empty(cap, dtype=torch.int32, device=dev),
            torch.empty(-(-cap // block), dtype=torch.int32, device=dev),
            torch.empty(1, dtype=torch.int32, device=dev))


def kernel(topk_ids, block):
    s, e, p = buffers(topk_ids, block)
    ops.moe_align_block_size(topk_ids, E, block, s, e, p, None)
    return s, e, p


def ours(topk_ids, block):
    s, e, p = buffers(topk_ids, block)
    det(topk_ids, E, block, s, e, p)
    return s, e, p


def batch(tokens, pad, stray, seed, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.rand(tokens, E, generator=g).topk(TOPK, dim=1).indices  # distinct experts per token
    t = torch.cat([t, torch.full((pad, TOPK), -1, dtype=t.dtype)])
    if stray:
        pos = torch.randperm(t.numel(), generator=g)[:stray]
        t.view(-1)[pos] = E + 1
    return t.to(dtype).to(dev)


def per_expert(s, e, p, block):
    """The layout up to the padded total, with the order inside each expert's run sorted."""
    total = int(p.item())
    owner = e[:total // block].to(torch.int64).repeat_interleave(block)
    key = owner * (2 ** 32) + s[:total].to(torch.int64)
    return total, torch.sort(key).values


def by_block(s, total, block):
    """Which routes each block holds, with the order inside a block sorted."""
    blk = torch.arange(total, device=s.device, dtype=torch.int64) // block
    return torch.sort(blk * (2 ** 32) + s[:total].to(torch.int64)).values


def ms(fn, *a, reps=20):
    for _ in range(3):
        fn(*a)
    if dev.type != "cuda":
        t = time.perf_counter()
        for _ in range(reps):
            fn(*a)
        return (time.perf_counter() - t) * 1000 / reps
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        fn(*a)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / reps


fails = 0
for tokens, block, pad, stray, dtype in [(8192, 64, 0, 0, torch.int32), (4000, 64, 96, 0, torch.int32),
                                         (2048, 64, 0, 13, torch.int64), (512, 32, 16, 0, torch.int32),
                                         (100, 16, 28, 5, torch.int32), (12, 8, 4, 0, torch.int32)]:
    topk_ids = batch(tokens, pad, stray, tokens + block, dtype)
    ks, ke, kp = kernel(topk_ids, block)
    ds, de, dp = ours(topk_ids, block)
    total = int(kp.item())
    blocks = total // block
    problems = []
    if int(dp.item()) != total:
        problems.append(f"padded total {int(dp.item())} vs the kernel's {total}")
    else:
        if not torch.equal(de[:blocks], ke[:blocks]):
            problems.append("block expert ids differ")
        if not bool((de[blocks:] == -1).all()) or not bool((ke[blocks:] == -1).all()):
            problems.append("blocks past the end are not all -1")
        if not torch.equal(per_expert(ds, de, dp, block)[1], per_expert(ks, ke, kp, block)[1]):
            problems.append("the routes in some expert's blocks differ")
        if not bool((ds[total:] == topk_ids.numel()).all()):
            problems.append("the rest does not hold numel")
    det_runs = [ours(topk_ids, block)[0] for _ in range(5)]
    det_same = all(torch.equal(r, det_runs[0]) for r in det_runs)
    if not det_same:
        problems.append("the deterministic path gave two layouts")
    k_runs = [kernel(topk_ids, block)[0] for _ in range(10)]
    first = by_block(ks, total, block)
    k_moved = sum(not torch.equal(by_block(r, total, block), first) for r in k_runs)
    k_inner = sum(not torch.equal(r[:total], ks[:total]) for r in k_runs) - k_moved
    t_k, t_d = ms(kernel, topk_ids, block), ms(ours, topk_ids, block)
    fails += bool(problems)
    print(f"{'FAIL' if problems else 'PASS'} {tokens} tokens + {pad} padding (-1) + {stray} stray ids, "
          f"{str(dtype).split('.')[-1]}, block {block}: {total} rows, {blocks} blocks; "
          f"kernel: a route in another block in {k_moved} of 10 runs, reordered inside blocks only in {k_inner}; "
          f"{t_k * 1000:.0f} us kernel, {t_d * 1000:.0f} us deterministic"
          + ("" if not problems else "\n     " + "; ".join(problems)))

if hasattr(mod, "_DET_ALIGN"):
    calls = []
    real = mod._deterministic_align
    mod._deterministic_align = lambda *a: (calls.append(a[0].shape[0]), real(*a))[1]
    mod._DET_ALIGN = True
    try:
        mod.moe_align_block_size(batch(4000, 0, 0, 1, torch.int32), 64, E, ignore_invalid_experts=True)
        mod.moe_align_block_size(batch(4, 0, 0, 2, torch.int32), 8, E, ignore_invalid_experts=True)
    finally:
        mod._deterministic_align, mod._DET_ALIGN = real, False
    ok = calls == [4000]
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} the wrapper with the switch on: the deterministic path for 4000 tokens, "
          f"the kernel for 4 (it took the deterministic path for {calls or 'none'})")
else:
    print("SKIP the wrapper: this image was built before the patch (./build.sh)")
print("ALL OK" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
