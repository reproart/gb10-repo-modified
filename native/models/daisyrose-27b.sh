# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: RadixArk/DaisyRose-27B-NVFP4, RadixArk's finetune of Qwen3.8-27B by
# reinforcement learning only (no SFT, no distillation; ~1,200 long agent tasks),
# a drop-in for the base: same architecture, tokenizer and chat template, the
# vision tower and the MTP head the base's, unchanged. Card: Terminal Bench 4.1
# 5.6 -> 12.0, small gains on most other coding / reasoning / instruction sets.
#
#   hf download RadixArk/DaisyRose-27B-NVFP4 \
#     --revision c4d85c5dfad123426555c22f4e24bb15381f0bdb \
#     --local-dir /models/RadixArk/DaisyRose-27B-NVFP4          # ~22 GB
#   ./serve.sh daisyrose-27b
#
# Read against RadixArk/Qwen3.8-27B-NVFP4, this project's default target, from
# the hub's files (2026-11-19): the same 2,194 tensors and the same layout
# (NVFP4 MLPs and lm_head, FP8 GDN / attention projections, BF16 GDN conv /
# in_proj_a / in_proj_b, embeddings and vision tower), byte-identical
# tokenizer.json, chat_template.jinja and image / video preprocessor configs,
# the same generation_config (temperature 1.0, top_p 0.95, top_k 20, which
# --sampling-defaults model applies). What differs:
#   - config.json is from transformers 5.19: vision_config.model_type
#     "qwen3_5_vision" and rope_parameters "axial". SGLang 0.5.21's own
#     Qwen3_5Config and its transformers 5.12.1 both read it (checked on CPU);
#     the boot log may say "Missing validation function ... 'axial'", harmless:
#     SGLang's vision tower has its own RoPE.
#   - no kv_cache_scheme. RadixArk's base says static FP8 but carries no scales
#     either, so --kv-cache-dtype fp8_e4m3 runs at scale 1.0 on both.
#   - the exclude list names the BF16 layers explicitly; they were left out of
#     quantized_layers before, so the same layers end up BF16.
# So everything else comes from qwen3.8-27b.sh: DFlash2 at 15 draft tokens, FP8
# draft, FP8_GEMM_TUNE=exact, cap 16, fp8 KV, parsers, ABLATE (whose vectors
# were found on Qwen3.8-27B: unvalidated on this model).
#
# The draft. Neither was trained on DaisyRose: z-lab's DFlash2 (the default)
# and RadixArk's DSpark, which the card recommends, are both drafts for
# Qwen3.8-27B. Measured on the GX10 (2026-11-19, results/RESULTS.md "DaisyRose-27B"):
# DFlash2 at 15 draft tokens 90.6 tok/s single-stream (the base 93.8), 343 /
# 430 tok/s at 8 / 16 streams (the base 322 / 430), prefill as the base; DSpark
# at gamma 7 ~50 tok/s single-stream. So DFlash2 stays. SPEC=dspark takes the
# card's draft (qwen3.8-27b-dspark.sh: download and gamma there), SPEC=mtp the
# checkpoint's MTP head (the base's, not retrained with the trunk).
#
# The card's flags not taken: --mamba-ssm-dtype float32 (bfloat16 here, as
# measured for the base; EXTRA_ARGS="--mamba-ssm-dtype float32" for theirs),
# --mamba-full-memory-ratio and --mamba-radix-cache-strategy extra_buffer_lazy
# (a 4-GPU data-parallel recipe; here MAX_RUNNING sizes the GDN pool).
# Thinking: the card evaluates at reasoning effort xhigh (the template's
# default) with no max_tokens cap, and sends earlier turns back with their
# reasoning_content; long thinking runs to tens of thousands of tokens.

MODEL_DIR="${MODEL_DIR:-/models/RadixArk/DaisyRose-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-daisyrose-27b}"

if [ "${SPEC:-}" = dspark ]; then
  # the DSpark profile replaces the DFlash2 flags; to the base profile this is
  # a separate-draft run (attention backend, FP8 tuning rows)
  SPEC=dflash
  # shellcheck source=qwen3.8-27b-dspark.sh
  . "$ROOT/models/qwen3.8-27b-dspark.sh"
else
  # shellcheck source=qwen3.8-27b.sh
  . "$ROOT/models/qwen3.8-27b.sh"
fi
