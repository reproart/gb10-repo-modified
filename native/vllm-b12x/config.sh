# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for nvidia/Qwen3.8-Flash-Next-FP4-Local on vLLM in the
# community b12x container (eugr/spark-vllm-b12x), the one runtime that reads
# this checkpoint: its NVFP4 scale factors use a new lossless compression and
# its n-gram (PLE) table is NVFP4, neither of which SGLang 0.5.21 or upstream
# vLLM loads. Source it, don't run it. Every value is "${VAR:-default}":
#   MAX_NUM_SEQS=4 MTP_TOKENS=3 ./serve.sh

B12X_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The container. image.lock (written by ./pull.sh, committed) pins the exact
# build by its sha256 digest, so a newer push to :latest never changes what
# runs here; IMAGE_DIGEST set in the shell wins. Docker Hub showed
# 03a956544e9f... for linux/arm64 of :latest on 2026-10-16 (EXPECT_PREFIX:
# pull.sh compares).
IMAGE="${IMAGE:-eugr/spark-vllm-b12x}"
if [ -z "${IMAGE_DIGEST:-}" ] && [ -f "$B12X_ROOT/image.lock" ]; then
  IMAGE_DIGEST="$(tr -d '[:space:]' < "$B12X_ROOT/image.lock")"
fi
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
EXPECT_PREFIX="${EXPECT_PREFIX-03a956544e9f}"
image_ref() {
  if [ -n "$IMAGE_DIGEST" ]; then echo "$IMAGE@$IMAGE_DIGEST"; else echo "$IMAGE:latest"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-gb10-vllm-b12x}"

# The weights (~96 GiB), downloaded once with the venv's hf CLI:
#   hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir /models/nvidia/Qwen3.8-Flash-Next-FP4-Local
# Mounted into the container at the same path, read-only (MODEL_RO=0 if the
# b12x loader turns out to want to write next to the weights).
MODEL_DIR="${MODEL_DIR:-/models/nvidia/Qwen3.8-Flash-Next-FP4-Local}"
MODEL_RO="${MODEL_RO:-1}"

# Kernel and compile caches, kept between runs (first boot compiles) and
# apart from SGLang's: the container runs as root, so they end up root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-vllm-b12x}"

# The API: OpenAI-compatible at http://HOST:PORT/v1. 8888 like ./serve.sh, so
# bench/ works unchanged; the two never run together (one pool of memory).
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-fp4-local}"

# The card's DGX Spark command, as knobs:
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"            # concurrent requests
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"  # prefill chunk
GPU_UTIL="${GPU_UTIL:-0.8}"                  # --gpu-memory-utilization
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# MTP draft tokens with the checkpoint's own head (NVFP4 experts); 0 = off.
MTP_TOKENS="${MTP_TOKENS:-4}"
# Where the n-gram table lives: disk (read through io_uring, the card's Spark
# setting, as our SGLang profile reads it in place) or whatever else the
# container's VLLM_PLE_TABLE_MEMORY accepts.
PLE_MEMORY="${PLE_MEMORY:-disk}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# Refuse to start without this much MemAvailable (GiB): the container takes
# GPU_UTIL of the ~121 GiB pool at once.
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"

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
