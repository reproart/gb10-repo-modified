#!/usr/bin/env python3
"""Build-time patch (vllm-flash-v034): the LM heads take the checkpoint's quantization.

vLLM 0.34's Qwen4Exp builds its lm_head (models/qwen4_exp/nvidia/model.py) and the MTP
draft's (mtp.py) as ParallelLMHead without a quant_config, so they are BF16 whatever the
checkpoint says. This recipe's checkpoint (Saren's AutoRound hybrid) stores the head as int8
GPTQ (lm_head.qweight / qzeros / scales; "dynamic": {"+:.*lm_head$": {"bits": 8}}), which a
BF16 head cannot load. The same one-line change as ../vllm-flash-ar's Dockerfile makes on the
pinned image: pass quant_config=vllm_config.quant_config. Written for vLLM 0.34.0; each file
must hold exactly one such head, or the build stops.

usage: patch_lm_head_quant.py <model.py> [<mtp.py> ...]
"""
import ast
import sys

MARK = "# gb10: the checkpoint's quantization (int8 GPTQ head)"
ANCHOR = 'prefix=maybe_prefix(prefix, "lm_head"),'


def patch(path: str) -> str:
    with open(path) as f:
        lines = f.read().splitlines(keepends=True)
    if any(MARK in line for line in lines):
        return "already installed"
    hits = [i for i, line in enumerate(lines) if line.strip() == ANCHOR]
    if len(hits) != 1:
        raise SystemExit(f"patch_lm_head_quant: {path}: {len(hits)} lines {ANCHOR!r}, expected 1")
    i = hits[0]
    call = "".join(lines[max(0, i - 4):i])
    if "ParallelLMHead(" not in call:
        raise SystemExit(f"patch_lm_head_quant: {path}: the lm_head prefix line is not in a ParallelLMHead call")
    if "quant_config" in call:
        return "not needed (the head already takes a quant_config)"
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i, f"{indent}quant_config=vllm_config.quant_config,  {MARK}\n")
    out = "".join(lines)
    ast.parse(out)
    with open(path, "w") as f:
        f.write(out)
    return "INSTALLED"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    for p in sys.argv[1:]:
        print(f"  lm_head quant_config: {patch(p)} in {p}")
