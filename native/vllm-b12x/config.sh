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

# Context. Native 262144; YARN=1 extends it with YaRN (Qwen: up to ~1M),
# built from the checkpoint's own config.json (its rope_parameters with
# rope_type yarn, YARN_FACTOR and original_max_position_embeddings = its
# max_position_embeddings), passed as --hf-overrides; MAX_MODEL_LEN then
# defaults to factor x native (4 x 262144 = 1048576). YaRN is static: every
# request is scaled the same way, short ones included, which can cost a
# little quality at ordinary lengths, so leave it off unless a session needs
# more than 262144. KV for 1M fits a Spark in fp8 (the card: 1M tokens of
# FP8 KV with the table on disk); one such request takes most of the pool.
YARN="${YARN:-0}"
YARN_FACTOR="${YARN_FACTOR:-4}"
# Empty = 262144, or factor x native with YARN=1.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# The card's DGX Spark command, as knobs:
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"            # concurrent requests
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"  # prefill chunk
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# MTP draft tokens with the checkpoint's own head (NVFP4 experts); 0 = off.
MTP_TOKENS="${MTP_TOKENS:-4}"
# Where the n-gram (PLE) table lives. The card: "DGX Spark: can run with or
# without PLE offload", 1M tokens of fp8 KV with it. The container's vLLM
# (./show-env.sh): VLLM_PLE_CPU_OFFLOAD (default 1: offloaded) and
# VLLM_PLE_TABLE_MEMORY ram | disk | shared (default unset, "resolved by vLLM
# for b12x"). Its boot line "Resolved Engram configuration: ...
# table_memory='device'" is DeepSeek V4.1's Engram table, not this one.
#   disk  read from the checkpoint's files through io_uring (the card's Spark
#         command). Measured: 71.7 GB loaded into the pool, ~25 GiB less than
#         the checkpoint, and `free` showed it neither in RAM nor page cache.
#   cpu   pinned in host memory (VLLM_PLE_TABLE_MEMORY=ram, the card's RTX PRO
#         setup). A Spark has one memory: the ~25 GiB come out of what the
#         GPU budget leaves, so GPU_UTIL drops to 0.6 (little KV). For
#         comparison only.
#   none  no offload (VLLM_PLE_CPU_OFFLOAD=0): the table loads with the
#         weights into the GPU budget, ~95 GiB together, so GPU_UTIL rises
#         to 0.85: ~5 GiB of KV (~300K fp8 tokens) and ~13 GiB left to the OS.
#         Saves the disk reads per prefill / decode step.
PLE_OFFLOAD="${PLE_OFFLOAD:-disk}"
# --gpu-memory-utilization. Empty = per PLE_OFFLOAD above (0.8 / 0.6 / 0.85).
GPU_UTIL="${GPU_UTIL:-}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# Refuse to start without this much MemAvailable (GiB). Empty = what
# GPU_UTIL takes of the pool, plus 2: the container claims it at once.
MIN_FREE_GIB="${MIN_FREE_GIB:-}"

# PROFILE=1: start vLLM's torch profiler endpoints (/start_profile,
# /stop_profile; off unless configured at launch) and write traces to
# PROFILE_DIR, mounted at the same path in the container so
# bench/profile_decode.py finds them on the host. Off by default: tracing
# costs nothing until started, but keeps the endpoints open.
PROFILE="${PROFILE:-0}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/gb10-profile}"

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
