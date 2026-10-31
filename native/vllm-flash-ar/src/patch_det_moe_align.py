#!/usr/bin/env python3
"""Deterministic MoE token alignment for the Marlin MoE path (VLLM_DET_MOE_ALIGN=1).

moe_align_block_size sorts a batch's (token, expert) routes into per-expert blocks of
block_size rows for the grouped MoE GEMM. vLLM's CUDA kernel places the routes inside an
expert with atomicAdd, i.e. in thread-arrival order; when an expert gets more routes than
one block (prefill), that order decides which m-block a route lands in, and Marlin's
split-K stripes depend on the m-block, so the same prompt sums in another order from one
run to the next. bench/consistency.py showed it on this recipe (int4 experts on Marlin
MoE): three cold runs of one greedy prompt, first-token top-5 logprobs 0.16-0.45 nats
apart, with VLLM_MARLIN_USE_ATOMIC_ADD on or off. blazux's NVFP4 path, whose MoE does not
run on Marlin, was deterministic with the same top-k fix.

With VLLM_DET_MOE_ALIGN=1, batches of at least VLLM_DET_MOE_ALIGN_MIN_TOKENS tokens (9)
are aligned by torch ops instead: the same blocks, padding and expert ids as the kernel,
the routes inside an expert in ascending route order. No host sync (CUDA-graph safe).
Smaller batches (decode) keep the kernel: a block holds them whole, rows are independent.
The idea and the torch implementation are prestonlogan's (Flint-vLLM-Single-DGX-Spark,
fast/patch_align.py, Apache-2.0; ~3.7% prefill there); written here against the
moe_align_block_size of this image's vLLM, located by name rather than by exact text, and
checked on CPU against a reference of the kernel's layout (src/test_det_moe_align_cpu.py).

usage: patch_det_moe_align.py <path to vllm/model_executor/layers/fused_moe/moe_align_block_size.py>
"""
import ast
import re
import sys

HELPER = '''

# --- det-moe-align (vllm-flash-ar, after prestonlogan's patch_align.py) ---
import os as _os

_DET_ALIGN = _os.environ.get("VLLM_DET_MOE_ALIGN", "0") == "1"
_DET_MIN_TOKENS = int(_os.environ.get("VLLM_DET_MOE_ALIGN_MIN_TOKENS", "9"))


def _deterministic_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad):
    """moe_align_block_size's outputs with each expert's routes in ascending route order
    (the CUDA kernel: atomicAdd arrival order). Routes to an id outside [0, num_experts)
    (the -1 of padding tokens) are left out, as the kernel leaves them out. Torch ops
    only, no host sync."""
    import torch as _torch
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(_torch.int64)
    n = flat.numel()
    cap = sorted_ids.numel()
    # invalid routes go to an extra bucket after the last expert, which takes no blocks
    key = _torch.where((flat >= 0) & (flat < num_experts), flat, _torch.full_like(flat, num_experts))
    counts = _torch.zeros(num_experts + 1, dtype=_torch.int64, device=dev).scatter_add_(0, key, _torch.ones_like(key))
    padded = ((counts[:num_experts] + block_size - 1) // block_size) * block_size
    pad_end = padded.cumsum(0)
    pad_start = _torch.cat([pad_end - padded, pad_end.new_zeros(1)])
    cnt_start = counts.cumsum(0) - counts
    sv, si = _torch.sort(key, stable=True)
    dest = pad_start[sv] + _torch.arange(n, device=dev) - cnt_start[sv]
    dest = _torch.where(sv < num_experts, dest, _torch.full_like(dest, cap))  # invalid: one slot past the end
    out = _torch.full((cap + 1,), n, dtype=sorted_ids.dtype, device=dev)
    out[dest] = si.to(sorted_ids.dtype)
    sorted_ids.copy_(out[:cap])
    blk_start = _torch.arange(expert_ids.numel(), device=dev, dtype=_torch.int64) * block_size
    e = _torch.searchsorted(pad_end, blk_start, right=True)
    expert_ids.copy_(_torch.where(e >= num_experts, _torch.full_like(e, -1), e).to(_torch.int32))
    num_tokens_post_pad.copy_(pad_end[-1:].to(_torch.int32))
# --- end det-moe-align ---
'''
MARK = "# --- det-moe-align"
NEEDED = ("topk_ids", "num_experts", "block_size", "sorted_ids", "expert_ids", "num_tokens_post_pad")


def patch(src: str) -> str:
    if MARK in src:
        return src
    fn = re.search(r"^def moe_align_block_size\(", src, re.M)
    assert fn, "def moe_align_block_size( not found"
    # the kernel call inside it: "<indent>ops.moe_align_block_size(\n ... \n<indent>)\n"
    call = re.compile(r"^(?P<ind>[ ]+)ops\.moe_align_block_size\(\n(?P<args>.*?)\n(?P=ind)\)\n", re.S | re.M)
    m = call.search(src, fn.end())
    assert m, "the ops.moe_align_block_size(...) call in moe_align_block_size not found"
    nxt = src.find("\ndef ", fn.end())
    assert src.count("ops.moe_align_block_size(", fn.end(), nxt if nxt > 0 else len(src)) == 1, \
        "more than one ops.moe_align_block_size call in moe_align_block_size"
    args = [a.strip().rstrip(",") for a in m.group("args").split("\n") if a.strip()]
    assert tuple(args[:6]) == NEEDED, f"unexpected kernel arguments {args[:6]}"
    ind = m.group("ind")
    cond = "_DET_ALIGN and topk_ids.shape[0] >= _DET_MIN_TOKENS"
    # expert parallelism (an expert_map) keeps the kernel; only names the function has
    signature = src[fn.end():src.index(":\n", fn.end())]
    if re.search(r"\bexpert_map\b", signature):
        cond += " and expert_map is None"
    if any("scatter_idx" in a for a in args[6:]):
        cond += " and scatter_idx is None"
    body = "\n".join(("    " + line) if line else line for line in m.group(0).rstrip("\n").split("\n")) + "\n"
    new = (f"{ind}if {cond}:\n"
           f"{ind}    _deterministic_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)\n"
           f"{ind}else:\n{body}")
    src = src[:m.start()] + new + src[m.end():]
    # the helper before the function (and before any decorator directly above it)
    start = src.index("def moe_align_block_size(")
    line_start = src.rfind("\n", 0, start) + 1
    while True:
        prev = src.rfind("\n", 0, line_start - 1) + 1
        if line_start > 0 and src[prev:line_start].startswith("@"):
            line_start = prev
        else:
            break
    return src[:line_start] + HELPER.lstrip("\n") + "\n\n" + src[line_start:]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
    s = patch(open(path).read())
    ast.parse(s)
    open(path, "w").write(s)
    print("moe_align_block_size: deterministic path patched OK (VLLM_DET_MOE_ALIGN=1 turns it on)")
