# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B, a LoRA (r 64, alpha 128, all
# language-model linear layers; TRIZ problem solving, mostly Chinese, trained
# at 2048 tokens) merged into Qwen/Qwen3.8-27B (BF16), FP8 at load.
#
#   hf download Qwen/Qwen3.8-27B --local-dir /models/Qwen3.8-27B
#   hf download Meerkat-AI/Meerkat-TRIZ-v1-Qwen3.8-27B --local-dir /models/Meerkat-TRIZ-v1-Qwen3.8-27B
#   python3 scripts/merge-lora.py /models/Qwen3.8-27B /models/Meerkat-TRIZ-v1-Qwen3.8-27B \
#       /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged
#   ./serve.sh meerkat-triz
#
# merge-lora.py runs in the SGLang venv (torch, safetensors) with the server
# stopped; it says whether the adapter's chat template differs from Qwen's
# (--chat-template adapter takes the adapter's).
#
# Everything else comes from qwen3.8-27b.sh: DFlash2 (trained on the base
# model's hidden states, so expect a lower accept_len after the fine-tune;
# the answers are the target's either way), FP8 draft, cap 12. The merged
# checkpoint is BF16 (~54 GB); QUANTIZATION=fp8 quantizes its linear layers
# while loading (SGLang's online FP8, W8A8 with per-tensor weight scales),
# so it serves in ~28 GB without writing a second checkpoint. Lossy like any
# FP8; QUANTIZATION= (empty) serves BF16 for a reference, slower.
#
# To compare it with the base model on the same questions:
#   bench/answers.py collect questions.jsonl --out runs/triz.jsonl   (here)
#   bench/answers.py collect questions.jsonl --out runs/base.jsonl   (qwen3.8-27b)
#   bench/answers.py compare runs/base.jsonl runs/triz.jsonl > compare.md

# Which weights:
#   fp8   the merged checkpoint converted offline to block FP8 laid out like
#         Qwen/Qwen3.8-27B-FP8 (~29 GB on disk, loads as Qwen's own FP8 does;
#         faster to boot and the same kernels as the qwen3.8-27b FP8 target):
#           python3 scripts/fp8-like-reference.py check /models/Qwen3.8-27B /models/Qwen3.8-27B-FP8
#           python3 scripts/fp8-like-reference.py convert /models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged \
#               /models/Qwen3.8-27B-FP8 /models/Meerkat-TRIZ-v1-Qwen3.8-27B-FP8
#   bf16  the merged BF16 checkpoint, quantized to FP8 while loading
#         (QUANTIZATION=fp8: per-tensor weight scales; 54 GB read each boot).
# Default: fp8 once its directory exists, bf16 before that.
FP8_DIR="${FP8_DIR:-/models/Meerkat-TRIZ-v1-Qwen3.8-27B-FP8}"
BF16_DIR="${BF16_DIR:-/models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged}"
if [ -z "${WEIGHTS:-}" ]; then
  if [ -f "$FP8_DIR/config.json" ]; then WEIGHTS=fp8; else WEIGHTS=bf16; fi
fi
case "$WEIGHTS" in
  fp8)  MODEL_DIR="${MODEL_DIR:-$FP8_DIR}"; QUANTIZATION="${QUANTIZATION-}" ;;
  bf16) MODEL_DIR="${MODEL_DIR:-$BF16_DIR}"; QUANTIZATION="${QUANTIZATION-fp8}" ;;
  *) echo "WEIGHTS must be fp8 or bf16, not '$WEIGHTS'" >&2; exit 2 ;;
esac
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-meerkat-triz}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"

eval "_base_$(declare -f model_args)"

model_args() {
  _base_model_args
  [ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
  return 0
}

model_summary() {
  echo "Meerkat-TRIZ (merged LoRA), $([ "$WEIGHTS" = fp8 ] && echo "FP8 checkpoint (block 128x128)" || echo "BF16 -> ${QUANTIZATION:-BF16} at load"); DFlash2, $DRAFT_TOKENS draft tokens; cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)$(fp8_summary)"
}
