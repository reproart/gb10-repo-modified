# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B with RadixArk's DSpark draft instead of z-lab's DFlash2.
#
#   ./serve.sh qwen3.8-27b-dspark
#
# Everything else comes from qwen3.8-27b.sh (target, cap, GDN pool, KV and
# parser flags; edit MODEL_DIR there and all profiles follow). DSpark was
# evaluated with RadixArk/Qwen3.8-27B-NVFP4 and Qwen/Qwen3.8-27B-FP8 as
# targets. Not measured on GB10 yet: compare it with the base profile by
# running bench/perf.py against each.
#
# RadixArk/Qwen3.8-27B-DSpark (model card): a 1.86B BF16 draft (five
# attention layers over target features from layers 5/19/33/47/61, plus a
# rank-256 Markov head), trained on 16 future positions and served with
# gamma 7: 7 proposals, verified 8 at a time with the target's bonus token.
# Its card: accept length 3.2-4.5 on code, math and chat with thinking on at
# temperature 1.0 (GB300, SGLang 0.5.17), and 2.3-3.2x over autoregressive
# at one stream on an H200. Those protocols differ from bench/perf.py
# (greedy, thinking off), so the numbers do not compare with ours directly.
#
# Weights (~3.7 GB), once:
#   hf download RadixArk/Qwen3.8-27B-DSpark --local-dir /models/Qwen3.8-27B-DSpark
# The card lists model.safetensors as sha256 2aff025f45823b40ebe726b9dfa4030
# 2f3512bd9a11c3a7347de32a567acd9a7 (3,714,723,322 bytes).

DRAFT_DIR="${DRAFT_DIR:-/models/Qwen3.8-27B-DSpark}"
# Draft proposals per step (gamma, --speculative-dspark-block-size); SGLang
# verifies gamma + 1. 7 is what the card serves; the draft was trained on 16
# positions, so larger values are worth a sweep, like DRAFT_TOKENS for DFlash2.
DSPARK_GAMMA="${DSPARK_GAMMA:-7}"
# The verify width, which sizes the verify buffer as DRAFT_TOKENS does for
# DFlash2: 8 per running request here, against 11 in the base profile.
DRAFT_TOKENS=$((DSPARK_GAMMA + 1))

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"

spec_args() {
  args+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DRAFT_DIR"
    --speculative-draft-model-quantization unquant
    --speculative-draft-attention-backend flashinfer
    --speculative-dspark-block-size "$DSPARK_GAMMA"
    --speculative-num-steps 1
    --speculative-eagle-topk 1
  )
}

# The card's serving environment.
model_env() {
  export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}

model_summary() {
  echo "DSpark, gamma $DSPARK_GAMMA (verify $DRAFT_TOKENS); cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)"
}
