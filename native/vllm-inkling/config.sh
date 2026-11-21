# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for thinkingmachines/Inkling-2-Mini on vLLM in the vLLM
# project's own container (vllm/vllm-openai:inkling-2-mini). Support needs
# vLLM >= 0.35.0, not released on 2026-11-21 (nightly only), so this
# directory runs the image the vLLM recipe names, with the recipe's
# dgx_spark_gb10 command as knobs. The SGLang twin is ../sglang-inkling/:
# the same weights, for an engine A/B on one model.
# Source it, don't run it. Every value is "${VAR:-default}":
#   SPEC_TOKENS=0 GPU_UTIL=0.85 ./serve.sh

VINK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The container. image.lock (committed) pins the exact build by its sha256
# digest; IMAGE_DIGEST set in the shell wins. The tag is a multi-platform
# index: Docker Hub showed index c2dc759f5984... (arm64 manifest
# ecdf0ccc663d..., 10.5 GiB compressed) pushed 2026-11-20 21:36 UTC, and that
# index digest is what image.lock holds and what `docker pull` reports.
IMAGE="${IMAGE:-vllm/vllm-openai}"
IMAGE_TAG="${IMAGE_TAG:-inkling-2-mini}"
if [ -z "${IMAGE_DIGEST:-}" ] && [ -f "$VINK_ROOT/image.lock" ]; then
  IMAGE_DIGEST="$(tr -d '[:space:]' < "$VINK_ROOT/image.lock")"
fi
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
EXPECT_PREFIX="${EXPECT_PREFIX-c2dc759f5984}"
image_ref() {
  if [ -n "$IMAGE_DIGEST" ]; then echo "$IMAGE@$IMAGE_DIGEST"; else echo "$IMAGE:$IMAGE_TAG"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-gb10-vllm-inkling}"
# Before the start, with the GPU idle: drop the page cache and compact free
# memory (../scripts/mem-prep.sh). 0 = memory as it is.
MEM_PREP="${MEM_PREP:-1}"

# The weights: the same directory as ../sglang-inkling/ (one download serves
# both), at the same revision:
#   ../scripts/mem-prep.sh idle && hf download thinkingmachines/Inkling-2-Mini \
#     --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
#     --local-dir /models/thinkingmachines/Inkling-2-Mini
# Mounted at the same path, read-only.
MODEL_DIR="${MODEL_DIR:-/models/thinkingmachines/Inkling-2-Mini}"
MODEL_RO="${MODEL_RO:-1}"

# Kernel and compile caches (torch.compile, CUDA graphs, FlashInfer, Triton),
# kept between runs and apart from the other servers'. Root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-vllm-inkling}"

# The API: OpenAI-compatible at http://HOST:PORT/v1. 8888 and the model id
# inkling-2-mini, as ../sglang-inkling/, so bench/ runs compare unchanged.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-2-mini}"

# The recipe's dgx_spark_gb10 command (features reasoning, spec_decoding):
#   --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser inkling
#   --speculative-config {"method":"dspark","num_speculative_tokens":7,
#                         "draft_sample_method":"probabilistic"}
# The recipe caps the context at 262144 on a Spark "to fit in memory"
# (1048576 natively).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# DSpark on the bundled draft head: proposals per step; 0 = off (plain decode).
SPEC_TOKENS="${SPEC_TOKENS:-7}"
# How the draft samples: the recipe's "probabilistic"; empty leaves vLLM's
# default (worth an A/B at temperature 0, where bench/perf.py runs).
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD-probabilistic}"
# --gpu-memory-utilization. The recipe sets none, so vLLM's default 0.9:
# ~108 GiB of a ~120 GiB Spark, leaving ~12 for the OS. ../vllm-b12x/ runs
# 0.8 on a Spark, and this project has seen machines wedge at high fractions
# on unified memory, so 0.8 here; GPU_UTIL=0.9 is the recipe exactly.
GPU_UTIL="${GPU_UTIL:-0.8}"
# Concurrent requests and prefill chunk; empty = vLLM's own defaults.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-}"
REASONING_PARSER="${REASONING_PARSER-inkling}"
# The recipe's tool_calling feature (opt-in there); on by default here, as in
# ../sglang-inkling/. Empty = off.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-inkling}"

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
