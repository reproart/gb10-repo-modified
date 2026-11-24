# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for primitive-ai/Ling-3.2-air-NVFP4A16 (inclusionAI's
# Ling-3.2-air with NVFP4 W4A16 experts) on vLLM, in the vLLM project's own
# nightly container at the commit the checkpoint's card names.
# Source it, don't run it. Every value is "${VAR:-default}":
#   MAX_MODEL_LEN=131072 GPU_UTIL=0.75 ./serve.sh

VLING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The container. The card runs nightly-a2bcb5657a88..., a tag no longer on
# Docker Hub (2026-11-24); cu134-nightly-<same commit> is, for arm64 too.
# image.lock pins the index digest Docker Hub showed (pushed 2026-11-05,
# arm64 12.2 GB compressed); IMAGE_DIGEST set in the shell wins.
IMAGE="${IMAGE:-vllm/vllm-openai}"
IMAGE_TAG="${IMAGE_TAG:-cu134-nightly-a2bcb5657a881a002e79d0adab3c6b5e39436121}"
if [ -z "${IMAGE_DIGEST:-}" ] && [ -f "$VLING_ROOT/image.lock" ]; then
  IMAGE_DIGEST="$(tr -d '[:space:]' < "$VLING_ROOT/image.lock")"
fi
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
EXPECT_PREFIX="${EXPECT_PREFIX-76a533475156}"
image_ref() {
  if [ -n "$IMAGE_DIGEST" ]; then echo "$IMAGE@$IMAGE_DIGEST"; else echo "$IMAGE:$IMAGE_TAG"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-gb10-vllm-ling}"
# Before the start, with the GPU idle: drop the page cache and compact free
# memory (../scripts/mem-prep.sh). 0 = memory as it is.
MEM_PREP="${MEM_PREP:-1}"

# The weights (72.6 GiB):
#   ../scripts/mem-prep.sh idle && hf download primitive-ai/Ling-3.2-air-NVFP4A16 \
#     --revision d4107e488e0cbbc06c4ebd6552da25e0cbb114b9 \
#     --local-dir /models/primitive-ai/Ling-3.2-air-NVFP4A16
# Mounted at the same path, read-only.
MODEL_DIR="${MODEL_DIR:-/models/primitive-ai/Ling-3.2-air-NVFP4A16}"
MODEL_RO="${MODEL_RO:-1}"

# Kernel and compile caches, kept between runs and apart from the other
# servers'. Root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-vllm-ling}"

# The API: OpenAI-compatible at http://HOST:PORT/v1.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ling-3.2-air}"

# The card's window; 262144 is the checkpoint's own maximum (no YaRN in its
# config.json, so the 1M of the SGLang page does not apply).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# --gpu-memory-utilization. The card's 0.92 is of a 96 GB card; 0.8 of a
# Spark's pool held 104.5 GiB in practice and left 3.4; 0.7 leaves 18
# (README, Measured).
GPU_UTIL="${GPU_UTIL:-0.7}"
# The checkpoint's MTP layer: proposals per step; 0 = off.
SPEC_TOKENS="${SPEC_TOKENS:-0}"
# --moe-backend; empty = vLLM's choice. For W4A16 NVFP4 use one that applies
# the experts' SwiGLU clamp (vLLM refuses the ones that do not).
MOE_BACKEND="${MOE_BACKEND:-}"
# Concurrent requests and prefill chunk; empty = vLLM's own defaults.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
# 0 = text only (--language-model-only: no vision encoder memory).
MULTIMODAL="${MULTIMODAL:-1}"
REASONING_PARSER="${REASONING_PARSER-ling3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-ling3}"

# Refuse to start without this much MemAvailable (GiB). Empty = what
# GPU_UTIL takes of the pool, plus 2: the container claims it at once.
MIN_FREE_GIB="${MIN_FREE_GIB:-}"

# Anything else: vllm serve flags, appended last so they win; and extra
# environment for the container, as space-separated NAME=value pairs.
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

need_docker() {
  if ! command -v docker >/dev/null; then
    echo "docker not found (DGX OS ships it; sudo apt install docker.io otherwise)" >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker is installed but not usable by $(id -un): sudo usermod -aG docker $(id -un)," \
      "then log in again (or run with sudo)" >&2
    exit 1
  fi
}
