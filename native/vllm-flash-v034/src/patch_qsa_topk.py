#!/usr/bin/env python3
"""Build-time patch: a deterministic QSA block top-k for vLLM 0.34.

vLLM 0.34's Qwen4Exp QSA indexer (models/qwen4_exp/nvidia/ops/qsa_indexer.py, _topk) picks
each query's key blocks with torch.ops._C.persistent_topk on GB10 (cooperative_topk is gated
off for the SM 12.x family). That kernel is non-deterministic on GB10 (vllm#51782, diagnosed by
@k3dani in blazux#3), in 0.34's build too: ../bench/consistency.py forked from the first token
with it and gave the same text every time with the exact top-k below (2026-11-22). Two
replacements, as ../vllm-flash-ar carries them (its patch 9):

  VLLM_QSA_DET_TOPK=1    @jschmied's deterministic persistent_topk (vllm#55122), built at image
                         build time as /opt/llm/kernel-det/_C_det.so (VLLM_QSA_DET_LIB): index-
                         ordered output slots, bit-exact ties, at the stock kernel's speed. It
                         registers _C_det.persistent_topk with the same arguments as the call
                         here, for k = 512, 1024 or 2048. If it does not load, or k is another
                         value, the exact path runs instead (deterministic, slower) and the log
                         says so: asking for determinism never gets the stock kernel.
  VLLM_QSA_EXACT_TOPK=1  torch.topk over the visible columns, the rest masked to -inf
                         (blazux 8347e7c); wins when both are set. On the dev image it cost
                         20-40% of long prefill.

One change in the exact path from the dev image's version: the chosen block ids are put in
ascending order. 0.34's expansion kernel reads only the first min(visible, block_topk) slots of
a row, so the visible blocks must come first; torch.topk(sorted=False) does not promise that,
and the masked columns are exactly the ids at or past the row's visible count, so ascending
order puts them last. It is also the order @jschmied's kernel writes.

usage: patch_qsa_topk.py <path to .../qwen4_exp/nvidia/ops/qsa_indexer.py>
Inert unless VLLM_QSA_DET_TOPK=1 or VLLM_QSA_EXACT_TOPK=1 at runtime.
"""
import ast
import sys

MARK = "gb10: deterministic QSA top-k"
CALL = """    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )
"""
NEW_CALL = """    # gb10: deterministic QSA top-k (VLLM_QSA_DET_TOPK=1 / VLLM_QSA_EXACT_TOPK=1)
    if _GB10_QSA_EXACT_TOPK or (_GB10_QSA_DET_OP is not None and block_topk not in _GB10_QSA_DET_K):
        _gb10_qsa_exact_topk(logits, visible_blocks, block_indices, block_topk, logits.shape[1])
        return
    if _GB10_QSA_DET_OP is not None:
        topk_op = _GB10_QSA_DET_OP
""" + CALL
HOOK = '''

# --- gb10: deterministic QSA top-k (VLLM_QSA_DET_TOPK=1 / VLLM_QSA_EXACT_TOPK=1; vllm-flash-v034/src/patch_qsa_topk.py)
import os as _gb10_os

try:
    from vllm.logger import init_logger as _gb10_init_logger

    _gb10_logger = _gb10_init_logger("vllm.gb10_qsa_topk")
except ImportError:  # CPU tests
    import logging as _gb10_logging

    _gb10_logger = _gb10_logging.getLogger("vllm.gb10_qsa_topk")

_GB10_QSA_DET_K = (512, 1024, 2048)  # the k values the deterministic kernel is built for
_GB10_QSA_EXACT_TOPK = _gb10_os.environ.get("VLLM_QSA_EXACT_TOPK", "0").lower() in ("1", "true", "yes")
_GB10_QSA_DET_OP = None
if _GB10_QSA_EXACT_TOPK:
    _gb10_logger.info("QSA top-k: exact torch.topk (VLLM_QSA_EXACT_TOPK=1)")
elif _gb10_os.environ.get("VLLM_QSA_DET_TOPK", "0") == "1":
    _gb10_lib = _gb10_os.environ.get("VLLM_QSA_DET_LIB", "/opt/llm/kernel-det/_C_det.so")
    try:
        torch.ops.load_library(_gb10_lib)
        _GB10_QSA_DET_OP = torch.ops._C_det.persistent_topk
        _gb10_logger.info("QSA top-k: deterministic persistent_topk (%s; exact torch.topk for k outside %s)",
                          _gb10_lib, _GB10_QSA_DET_K)
    except Exception as _gb10_e:  # noqa: BLE001 - determinism is kept, only the speed is lost
        _GB10_QSA_EXACT_TOPK = True
        _gb10_logger.warning("QSA top-k: the deterministic kernel did not load (%s: %s); exact torch.topk "
                             "instead (deterministic, slower long prefill)", type(_gb10_e).__name__, _gb10_e)


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
        raise SystemExit(f"patch_qsa_topk: {path}: the persistent_topk call in _topk not found once: "
                         "not the qsa_indexer.py this was written for")
    out = src.replace(CALL, NEW_CALL).rstrip("\n") + HOOK
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return "INSTALLED"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: patch_qsa_topk.py <qsa_indexer.py>")
    print("  QSA top-k", patch(sys.argv[1]), "in", sys.argv[1],
          "(inert unless VLLM_QSA_DET_TOPK=1 or VLLM_QSA_EXACT_TOPK=1)")
