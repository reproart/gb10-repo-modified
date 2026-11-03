#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): FP8 hyper-connection projections (VLLM_HC_FP8=1).

Appends a hook to vLLM's models/qwen3_8_flash_next/nvidia/hyperconnection.py: with
VLLM_HC_FP8=1 at runtime, gb10_fp8_rowwise.install_hc wraps the module's classes so
their input_mix_weight_down(_block_inject) / input_mix_weight_up linear layers keep
their weights in FP8 with per-row scales after loading and run them through
gb10_fp8_rowwise's GEMM (see that module). Without the variable nothing is imported
and nothing changes.

usage: patch_hc_fp8.py <path to hyperconnection.py>
"""
import ast
import sys

MARK = "qwen38-flash-dgx: FP8 hyper-connection projections"
HOOK = '''

# --- qwen38-flash-dgx: FP8 hyper-connection projections (VLLM_HC_FP8=1; src/patch_hc_fp8.py)
import os as _gb10_hc_os

if _gb10_hc_os.environ.get("VLLM_HC_FP8", "0") == "1":
    import gb10_fp8_rowwise as _gb10_fp8_rowwise

    _gb10_fp8_rowwise.install_hc(globals())
'''
EXPECT = ("input_mix_weight_up", "input_mix_weight_down", "ReplicatedLinear")


def patch(path: str) -> str:
    src = open(path).read()
    if MARK in src:
        return "already installed"
    missing = [s for s in EXPECT if s not in src]
    if missing:
        raise SystemExit(f"patch_hc_fp8: {path} lacks {missing}: not the hyperconnection.py this was written for")
    out = src.rstrip("\n") + HOOK
    ast.parse(out)
    open(path, "w").write(out)
    return "INSTALLED"


if __name__ == "__main__":
    print("  FP8 HC hook", patch(sys.argv[1]), "in", sys.argv[1], "(inert unless VLLM_HC_FP8=1)")
