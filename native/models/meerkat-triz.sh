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

MODEL_DIR="${MODEL_DIR:-/models/Meerkat-TRIZ-v1-Qwen3.8-27B-merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-meerkat-triz}"
QUANTIZATION="${QUANTIZATION-fp8}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"

eval "_base_$(declare -f model_args)"

model_args() {
  _base_model_args
  [ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
  return 0
}

model_summary() {
  echo "Meerkat-TRIZ (merged LoRA), ${QUANTIZATION:-BF16}; DFlash2, $DRAFT_TOKENS draft tokens; cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)$(fp8_summary)"
}
