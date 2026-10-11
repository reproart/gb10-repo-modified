# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: inclusionAI Ling-3.0-flash-VL on one DGX Spark: ~125B MoE (5.1B
# active; 512 routed experts, top-8, plus a shared one) over a hybrid backbone
# (Kimi Delta Attention linear layers + gated MLA), with a 0.4B vision encoder
# for images and video. A thinking model with tool calls; no MTP head, so no
# speculative decoding.
#
#   hf download inclusionAI/Ling-3.0-flash-VL-fp4 --local-dir /models/inclusionAI/Ling-3.0-flash-VL-fp4
#   ./serve.sh ling-3.0-flash-vl install     # SGLang 0.5.21
#   ./serve.sh ling-3.0-flash-vl
#
# From the SGLang cookbook's Ling-3.0-flash-VL recipe, whose DGX Spark cells
# are verified (dev build bf254483a1; MMMU-Pro / GSM8K measured on B300):
#   fp4   MXFP4 experts + FP8 e4m3 elsewhere; flashinfer_mxfp4 MoE (the SM12x
#         CUTLASS W4A8 path), picked by SGLang itself. Spark, 8K in / 1K out:
#         TPOT 30.2 ms (~33 tok/s) alone, 111.6 ms at 16 streams (1,070 tok/s
#         in+out); MMMU-Pro 76.24%, GSM8K 96.66%.
#   int4  GPTQ W4A16, group 32, compressed-tensors (Marlin MoE). Spark:
#         TPOT 44.2 ms (~23 tok/s) alone; MMMU-Pro 77.51%, GSM8K 96.97%.
# BF16 (~250 GB) and FP8 (~126 GB) do not fit one Spark.
#
# SGLang 0.5.21 has everything the recipe needs (checked in the wheel):
# BailingMoeV3VLForConditionalGeneration (models/bailing_mm_v3.py), the KDA
# layers, the automatic flashinfer_mxfp4 choice for the mixed FP8/MXFP4
# checkpoint on SM12x (GB10 included), shared-expert fusion turned off for
# the FP4 / INT4 exports, the ling3 reasoning and tool-call parsers, and
# decord2 for video on aarch64. Audio input is not supported by the model.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Which weights (MODEL_DIR, if set, wins):
#   fp4   inclusionAI/Ling-3.0-flash-VL-fp4, the faster one (default)
#   int4  inclusionAI/Ling-3.0-flash-VL-int4
WEIGHTS="${WEIGHTS:-fp4}"
case "$WEIGHTS" in
  fp4)  _dir=/models/inclusionAI/Ling-3.0-flash-VL-fp4 ;;
  int4) _dir=/models/inclusionAI/Ling-3.0-flash-VL-int4 ;;
  *) echo "WEIGHTS must be fp4 or int4, not '$WEIGHTS'" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$_dir}"
DRAFT_DIR=""
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ling-3.0-flash-vl}"
unset _dir

# Context. YARN=1 (the recipe's and the model card's): 262144 tokens through
# a YaRN x2 override of the native 131072 window; static, so a short prompt
# is scaled the same way. YARN=0: the native 128K window, no override.
YARN="${YARN:-1}"
if [ "$YARN" = 1 ]; then
  CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
else
  CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
fi
LING_YARN='{"rope_scaling":{"rope_type":"yarn","factor":2.0,"rope_theta":6000000,"partial_rotary_factor":0.5,"original_max_position_embeddings":131072}}'

# The recipe runs SGLang's default pool; 0.80 is this project's margin for
# DGX OS earlyoom. Each request also holds a fixed-size KDA state, so the cap
# is what concurrency costs: 16 is the recipe's largest Spark measurement.
# An image is ~880 tokens at 720p (up to ~4,100), a video clip several
# thousand: size the cap for image-heavy traffic with that in mind.
MEM_FRACTION="${MEM_FRACTION:-0.80}"
MAX_RUNNING="${MAX_RUNNING:-16}"
# Empty = SGLang's choice (what the recipe ran).
CHUNKED_PREFILL="${CHUNKED_PREFILL:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"

# Thinking is on by default in the chat template; per request:
# "chat_template_kwargs": {"enable_thinking": false}. ling3 is what "auto"
# resolves to from the template.
REASONING_PARSER="${REASONING_PARSER:-ling3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-ling3}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

model_env() {
  # SGLang derives 131072 from the config; the YaRN override allows more.
  if [ "$YARN" = 1 ]; then
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  elif [ "$CONTEXT_LENGTH" -gt 131072 ]; then
    echo "CONTEXT_LENGTH=$CONTEXT_LENGTH needs YARN=1 (the native window is 131072)" >&2
    exit 1
  fi
}

model_args() {
  args+=(
    --tp 1
    --max-running-requests "$MAX_RUNNING"
  )
  [ "$YARN" = 1 ] && args+=(--json-model-override-args "$LING_YARN")
  [ -n "$CHUNKED_PREFILL" ] && args+=(--chunked-prefill-size "$CHUNKED_PREFILL")
  [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
  [ -n "$MOE_RUNNER_BACKEND" ] && args+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  [ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  return 0
}

model_summary() {
  echo "Ling-3.0-flash-VL $WEIGHTS ($([ "$WEIGHTS" = fp4 ] && echo "MXFP4 experts + FP8" || echo "GPTQ INT4 W4A16")), MoE ${MOE_RUNNER_BACKEND:-auto}; context $CONTEXT_LENGTH$([ "$YARN" = 1 ] && echo " (YaRN x2)"); cap $MAX_RUNNING; images + video, no draft"
}
