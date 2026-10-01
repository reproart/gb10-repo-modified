# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for the llama.cpp scripts here (GLM-5.3-Flash GSQ-RCO GGUFs).
# Source it, don't run it. Every value is "${VAR:-default}": override from the
# shell, e.g.  QUANT=3.5bit ./serve-dual.sh

LLAMACPP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The llama.cpp checkout and build. The commit is the one the GGUFs' card
# evaluated with: upstream 8134115f plus PR 27773 (the glm5-next
# architecture), at the PR's head de25343. Both Sparks must run the same
# build: llama.cpp's RPC protocol is not stable across versions.
LLAMA_DIR="${LLAMA_DIR:-$HOME/spark/llama.cpp-glm5}"
LLAMA_REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_COMMIT="${LLAMA_COMMIT:-de25343596a2c924ba62d35b126e811be0d38a49}"
LLAMA_PR="${LLAMA_PR:-27773}"
LLAMA_BIN="$LLAMA_DIR/build/bin"

# The GGUFs (pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF):
#   hf download pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf \
#     GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf GLM-5.3-Flash-mmproj-BF16.gguf \
#     --local-dir /models/GLM-5.3-Flash-GSQ-RCO-GGUF
# Only the machine that runs llama-server needs them; the RPC worker gets its
# share over the network.
GGUF_DIR="${GGUF_DIR:-/models/GLM-5.3-Flash-GSQ-RCO-GGUF}"
MMPROJ_FILE="${MMPROJ_FILE:-$GGUF_DIR/GLM-5.3-Flash-mmproj-BF16.gguf}"

# The API: OpenAI-compatible at http://HOST:PORT/v1. 8889, so it never
# collides with the SGLang server's 8888 (stop that one anyway: one Spark has
# one pool of memory). API_KEY as for serve.sh.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8889}"
API_KEY="${API_KEY:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.3-flash}"

# The card's evaluation runtime: no TF32 in F32 matrix products
# (patches/native-f32-mmf.patch + these two variables). Kept on so outputs
# match what the card measured; STRICT_F32=0 lets llama.cpp use TF32.
STRICT_F32="${STRICT_F32:-1}"

# Any other llama-server flags, appended last so they win.
EXTRA_ARGS="${EXTRA_ARGS:-}"

gguf_for() { echo "$GGUF_DIR/GLM-5.3-Flash-GSQ-RCO-$1.gguf"; }

gib() { awk -v b="$1" 'BEGIN { printf "%.1f", b / 1073741824 }'; }

need_build() {
  if [ ! -x "$LLAMA_BIN/$1" ]; then
    echo "$LLAMA_BIN/$1 not found: run $LLAMACPP_ROOT/build.sh first" >&2
    exit 1
  fi
}

need_file() {
  if [ ! -f "$1" ]; then
    echo "$1 not found (download: see GGUF_DIR in $LLAMACPP_ROOT/config.sh)" >&2
    exit 1
  fi
}

# Refuse to start next to the SGLang server: both want most of the memory.
no_sglang() {
  if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
    echo "an SGLang server is running (pgrep -f sglang.launch_server); stop it first:" \
      "the GPU and the CPU share the same 128 GB" >&2
    exit 1
  fi
}

strict_f32_env() {
  if [ "$STRICT_F32" = 1 ]; then
    export NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_MMF_F32_DISABLE=1
  fi
}

# The flags both launchers share.
common_server_args() {
  args+=(
    --host "$HOST" --port "$PORT" --alias "$SERVED_MODEL_NAME"
    -ngl all --jinja
    --load-mode none --lazy-mode off
    --metrics
  )
  [ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
  return 0
}
