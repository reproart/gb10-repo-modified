# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for thinkingmachines/Inkling-2-Mini on SGLang in LMSYS's
# own container (lmsysorg/sglang:inkling-2-mini). Inkling-2-Mini support is
# not in an SGLang release yet (sgl-project/sglang#51926, open on
# 2026-11-21), so the stock venv of ../serve.sh cannot load it; this
# directory runs the image the SGLang cookbook names for DGX Spark, with the
# cookbook's DGX Spark recipe as knobs. Once a release carries it, the model
# belongs in ../models/ as a profile and this directory can go.
# Source it, don't run it. Every value is "${VAR:-default}":
#   STRATEGY=balanced MEM_FRACTION=0.6 ./serve.sh

INKLING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The container. image.lock (committed) pins the exact build by its sha256
# digest, so a newer push to the tag never changes what runs here;
# IMAGE_DIGEST set in the shell wins. The tag is a multi-platform index:
# Docker Hub showed index efac853b4681... (arm64 manifest bc811ac7ac68...,
# 15.3 GiB compressed) pushed 2026-11-20 20:16 UTC, and that index digest is
# what image.lock holds and what `docker pull` reports.
IMAGE="${IMAGE:-lmsysorg/sglang}"
IMAGE_TAG="${IMAGE_TAG:-inkling-2-mini}"
if [ -z "${IMAGE_DIGEST:-}" ] && [ -f "$INKLING_ROOT/image.lock" ]; then
  IMAGE_DIGEST="$(tr -d '[:space:]' < "$INKLING_ROOT/image.lock")"
fi
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
EXPECT_PREFIX="${EXPECT_PREFIX-efac853b4681}"
image_ref() {
  if [ -n "$IMAGE_DIGEST" ]; then echo "$IMAGE@$IMAGE_DIGEST"; else echo "$IMAGE:$IMAGE_TAG"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-gb10-sglang-inkling}"
# Before the start, with the GPU idle: drop the page cache and compact free
# memory (../scripts/mem-prep.sh). 0 = memory as it is.
MEM_PREP="${MEM_PREP:-1}"

# The weights (~18.2 GiB: one safetensors file with the bundled DSpark draft
# head), at the revision this directory was written against:
#   ../scripts/mem-prep.sh idle && hf download thinkingmachines/Inkling-2-Mini \
#     --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
#     --local-dir /models/thinkingmachines/Inkling-2-Mini
# Public, Apache-2.0, no token. Mounted at the same path, read-only.
MODEL_DIR="${MODEL_DIR:-/models/thinkingmachines/Inkling-2-Mini}"
MODEL_RO="${MODEL_RO:-1}"

# JIT kernel and compile caches, kept between runs (the first boot compiles;
# MAX_JOBS below bounds how many compile at once). The container runs as
# root, so they end up root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-sglang-inkling}"

# The API: OpenAI-compatible at http://HOST:PORT/v1. 8888 like ../serve.sh,
# so ../bench/ works unchanged; the two never run together (one pool of
# memory).
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-2-mini}"

# The cookbook's two DGX Spark recipes (both "verified", gsm8k 93.56% and
# 93.93% on all 1319 questions):
#   dspark    speculative decoding on the bundled 3-layer DSpark draft head,
#             8 draft tokens verified per step, plus four SGLANG_OPT_* switches
#   balanced  plain decode
# and one not in the cookbook:
#   dflash2   incoai/Inkling-2-Mini-DFlash2, a separate 5-layer block-diffusion
#             draft (BF16, ~0.99 GB), with the compatibility patch from its
#             repository (patches/sglang_inkling_dflash2.py, applied inside the
#             container at each start; it edits 5 SGLang files of this image
#             and nothing else). Its card's command, flag for flag: DFLASH,
#             8 draft tokens (block size 8), FA4 for the draft too; TP1 only.
# Default dflash2 since 2026-11-28 (README: 1.35x vLLM's DSpark on code, 1.8x
# on prose, HumanEval unchanged); dspark is the cookbook recipe.
STRATEGY="${STRATEGY:-dflash2}"
# The DFlash2 draft, at the revision the patch was taken from:
#   ../scripts/mem-prep.sh idle && hf download incoai/Inkling-2-Mini-DFlash2 \
#     --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2 \
#     --local-dir /models/incoai/Inkling-2-Mini-DFlash2
# Public, Apache-2.0. model.safetensors sha256 217980ab00d1...; the patch's
# sha256 is 14223ae2f3aa... (checked by serve.sh).
DRAFT_DIR="${DRAFT_DIR:-/models/incoai/Inkling-2-Mini-DFlash2}"
DFLASH2_PATCH_SHA256=14223ae2f3aa1d661da67cf0143ec33070d952a6e28a119e6cec64fcc0bebe50
# --speculative-num-draft-tokens for dspark and dflash2. The recipes' 8; the 27B's
# DFlash2 optimum moved with the workload (16 for one stream, 10 for many),
# so worth a sweep here too.
DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
# --mem-fraction-static. The recipe's 0.45 is what the cookbook verified on a
# Spark; it leaves the KV pool small. This project runs the 27B at 0.85 and
# recorded wedged machines above it, so 0.6-0.85 is worth measuring, not
# assuming.
MEM_FRACTION="${MEM_FRACTION:-0.45}"
# Concurrent requests and context; empty = SGLang's own choice for the pool.
MAX_RUNNING="${MAX_RUNNING:-}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
# Image input on (the recipe's --enable-multimodal); 0 = text only.
MULTIMODAL="${MULTIMODAL:-1}"
REASONING_PARSER="${REASONING_PARSER-inkling}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER-inkling2}"

# Refuse to start without this much MemAvailable (GiB). Empty = what
# MEM_FRACTION takes of the pool, plus 4 (JIT compilers at first boot).
MIN_FREE_GIB="${MIN_FREE_GIB:-}"

# Anything else: sglang serve flags, appended last so they win; and extra
# environment for the container, as space-separated NAME=value pairs that
# replace a recipe value of the same name (EXTRA_ENV=MAX_JOBS=8).
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
