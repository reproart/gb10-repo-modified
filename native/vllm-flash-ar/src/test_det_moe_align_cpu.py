#!/usr/bin/env python3
"""CPU checks for patch_det_moe_align.py (torch, no GPU, no vLLM).

1. _deterministic_align against a reference of the CUDA kernel's layout: every expert's
   routes in its own run of blocks, padded to block_size with the route count n, the
   blocks' expert ids, the padded total; the routes inside an expert in ascending order;
   routes to an id outside [0, num_experts) left out (padding tokens carry -1).
   Random batches, 512 experts top-10 (Qwen3.8-Flash-Next) and small cases.
2. The patcher on vLLM's moe_align_block_size.py (MOE_ALIGN_SRC, or a minimal stand-in
   of the PR #53896 layout): the call wrapped once, the helper above the function, the
   file parses, a second run changes nothing.

    python3 src/test_det_moe_align_cpu.py
"""
import ast
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch_det_moe_align as pd  # noqa: E402

ns = {}
exec(compile(pd.HELPER, "helper", "exec"), ns)
det = ns["_deterministic_align"]


def reference(topk_ids, num_experts, block_size):
    flat = topk_ids.reshape(-1).tolist()
    n = len(flat)
    per = [[] for _ in range(num_experts)]
    for i, e in enumerate(flat):
        if 0 <= e < num_experts:  # the kernel skips other ids (get_local_expert_id)
            per[e].append(i)
    ids, blocks = [], []
    for e in range(num_experts):
        if not per[e]:
            continue
        k = -(-len(per[e]) // block_size) * block_size
        ids += per[e] + [n] * (k - len(per[e]))
        blocks += [e] * (k // block_size)
    return ids, blocks, len(ids)


def run(tokens, top_k, num_experts, block_size, seed, pad_tokens=0, stray=0):
    g = torch.Generator().manual_seed(seed)
    rows = [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(tokens)]
    topk = (torch.stack(rows) if rows else torch.empty(0, top_k, dtype=torch.int64)).to(torch.int32)
    if pad_tokens:  # padding tokens: every route -1
        topk = torch.cat([topk, torch.full((pad_tokens, top_k), -1, dtype=torch.int32)])
    if stray:  # a few ids past the last expert, anywhere
        pos = torch.randperm(topk.numel(), generator=g)[:stray]
        topk.view(-1)[pos] = num_experts + torch.arange(stray, dtype=torch.int32) % 3
    cap = topk.numel() + num_experts * (block_size - 1)  # as moe_align_block_size sizes it
    if topk.numel() < num_experts:
        cap = min(topk.numel() * block_size, cap)
    sorted_ids = torch.empty(cap, dtype=torch.int32)
    expert_ids = torch.empty(-(-cap // block_size), dtype=torch.int32)
    post = torch.empty(1, dtype=torch.int32)
    det(topk, num_experts, block_size, sorted_ids, expert_ids, post)
    ids, blocks, total = reference(topk, num_experts, block_size)
    assert post.item() == total, (post.item(), total)
    assert sorted_ids[:total].tolist() == ids, "sorted ids differ from the reference layout"
    assert expert_ids[:len(blocks)].tolist() == blocks, "block expert ids differ"
    assert all(v == -1 for v in expert_ids[len(blocks):].tolist()), "blocks past the end must be -1"
    assert all(v == topk.numel() for v in sorted_ids[total:].tolist()), "the rest must hold numel"
    return total


def main():
    for tokens, top_k, experts, block, seed in [(4096, 10, 512, 16, 1), (300, 10, 512, 64, 2), (9, 10, 512, 16, 3),
                                                 (2, 2, 4, 4, 4), (50, 3, 8, 8, 5)]:
        total = run(tokens, top_k, experts, block, seed)
        print(f"PASS layout: {tokens} tokens x top-{top_k} of {experts} experts, block {block}: {total} rows")
    for tokens, top_k, experts, block, seed, pad, stray in [(500, 10, 512, 16, 11, 12, 0), (40, 10, 512, 8, 12, 3, 7),
                                                            (0, 10, 512, 16, 13, 16, 0), (30, 3, 8, 4, 14, 5, 9)]:
        total = run(tokens, top_k, experts, block, seed, pad, stray)
        print(f"PASS layout with {pad} padding tokens (-1) and {stray} ids past the last expert: "
              f"{tokens} tokens x top-{top_k} of {experts}, block {block}: {total} rows")
    a = run(2048, 10, 512, 16, 7)
    b = run(2048, 10, 512, 16, 7)
    assert a == b
    print("PASS the same batch twice: the same layout")

    src_path = os.environ.get("MOE_ALIGN_SRC")
    src = open(src_path).read() if src_path else STANDIN
    patched = pd.patch(src)
    ast.parse(patched)
    assert patched.count("_deterministic_align(topk_ids") == 2 and patched.count("ops.moe_align_block_size(") == 1  # def + call
    assert patched.index(pd.MARK) < patched.index("def moe_align_block_size(")
    assert pd.patch(patched) == patched
    print(f"PASS patcher on {'MOE_ALIGN_SRC' if src_path else 'a stand-in of the PR #53896 file'}: wrapped once, "
          "helper above, parses, idempotent")
    if "expert_map" in src:
        assert "and expert_map is None:" in patched
    # a version without expert_map: the condition must not name it
    no_map = STANDIN.replace("    expert_map: torch.Tensor | None = None,\n", "") \
        .replace("        expert_map if ignore_invalid_experts else None,\n", "") \
        .replace("    if expert_map is not None and not ignore_invalid_experts:\n        expert_ids = expert_map[expert_ids]\n", "")
    assert "expert_map" not in no_map
    patched = pd.patch(no_map)
    ast.parse(patched)
    assert "expert_map" not in patched.split(pd.MARK.replace("# ---", "# --- end"))[1]
    print("PASS patcher on a signature without expert_map: the condition does not name it")
    print("ALL OK")


STANDIN = '''import torch

from vllm import _custom_ops as ops


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map if ignore_invalid_experts else None,
    )

    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size():
    pass
'''

if __name__ == "__main__":
    main()
