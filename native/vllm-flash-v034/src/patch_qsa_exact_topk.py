#!/usr/bin/env python3
"""Build-time patch: exact, deterministic QSA block top-k for vLLM 0.34 (VLLM_QSA_EXACT_TOPK=1).

vLLM 0.34's Qwen4Exp QSA indexer (models/qwen4_exp/nvidia/ops/qsa_indexer.py, _topk) picks
each query's key blocks with torch.ops._C.persistent_topk on GB10 (cooperative_topk is gated
off for the SM 12.x family). That kernel was non-deterministic on GB10 and could drop real
candidates in the dev image's vLLM (vllm#51782, diagnosed by @k3dani in blazux#3); whether
0.34's build carries @jschmied's fix (vllm#55122) is not visible from its Python, so this is
the exact fallback ../vllm-flash-ar carries as patch 9 (EXACT_TOPK, from blazux 8347e7c),
ported: torch.topk over the visible columns, the columns past a row's visible count masked
to -inf first.

One change from the dev image's version: the selected block ids are put in ascending order.
0.34's expansion kernel reads only the first min(visible, block_topk) slots of a row, so the
visible blocks must come first; torch.topk(sorted=False) does not promise an order, and the
masked columns are exactly the ids at or past the row's visible count, so ascending order puts
them last. It is also the order @jschmied's deterministic kernel writes (index-ordered slots).

Inert unless VLLM_QSA_EXACT_TOPK=1 at runtime (the dev image found it 20-40% slower on long
prefill). usage: patch_qsa_exact_topk.py <path to .../qwen4_exp/nvidia/ops/qsa_indexer.py>
"""
import ast
import sys

MARK = "gb10: exact QSA top-k"
CALL = """    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )
"""
NEW_CALL = """    if _GB10_QSA_EXACT_TOPK:  # gb10: exact QSA top-k (VLLM_QSA_EXACT_TOPK=1)
        _gb10_qsa_exact_topk(logits, visible_blocks, block_indices, block_topk, logits.shape[1])
        return
""" + CALL
HOOK = '''

# --- gb10: exact QSA top-k (VLLM_QSA_EXACT_TOPK=1; vllm-flash-v034/src/patch_qsa_exact_topk.py)
import os as _gb10_os

_GB10_QSA_EXACT_TOPK = _gb10_os.environ.get("VLLM_QSA_EXACT_TOPK", "0").lower() in ("1", "true", "yes")


def _gb10_qsa_exact_topk(logits, visible_blocks, block_indices, block_topk, columns):
    """block_indices[r] = the ids of row r's block_topk highest logits among its first
    visible_blocks[r] columns, in ascending order, then -1. Torch ops, no host sync."""
    cols = torch.arange(columns, device=logits.device, dtype=torch.int32)
    masked = logits[:, :columns].masked_fill(cols[None, :] >= visible_blocks[:, None], float("-inf"))
    k = min(block_topk, columns)
    idx = torch.topk(masked, k, dim=1, sorted=False).indices.sort(dim=1).values
    block_indices[:, :k].copy_(idx)
    if k < block_topk:
        block_indices[:, k:].fill_(-1)
'''


def patch(path):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        return "already installed"
    if src.count(CALL) != 1 or "def _topk(" not in src:
        raise SystemExit(f"patch_qsa_exact_topk: {path}: the persistent_topk call in _topk not found once: "
                         "not the qsa_indexer.py this was written for")
    out = src.replace(CALL, NEW_CALL).rstrip("\n") + HOOK
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return "INSTALLED"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: patch_qsa_exact_topk.py <qsa_indexer.py>")
    print("  QSA exact top-k", patch(sys.argv[1]), "in", sys.argv[1], "(inert unless VLLM_QSA_EXACT_TOPK=1)")
