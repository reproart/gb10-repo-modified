#!/usr/bin/env bash
# Build llama.cpp for the GLM-5.3-Flash GSQ-RCO GGUFs on a DGX Spark (GB10):
# the commit their card evaluated with (PR 27773, the glm5-next
# architecture), patches/native-f32-mmf.patch, CUDA for sm_121 and the RPC
# backend (for serve-dual.sh; its server binary is ggml-rpc-server at this
# commit). Run it on both Sparks for the two-machine
# setup: llama-server on one, ggml-rpc-server on the other, same build.
#
#   ./build.sh            clone or update $LLAMA_DIR, patch, build
#   JOBS=8 ./build.sh     fewer parallel compile jobs (default: all cores)
#   UI=0 ./build.sh       no web UI in llama-server (it downloads a prebuilt
#                         one from Hugging Face at build time; API only then)
#
# Needs git, cmake >= 3.24 and the CUDA toolkit (nvcc; DGX OS ships it in
# /usr/local/cuda). ~10-20 minutes on the Spark's 20 cores. Re-running is
# safe: the checkout is reset to the pinned commit, the patch re-applied.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

JOBS="${JOBS:-$(nproc)}"
UI="${UI:-1}"
PATCH="$LLAMACPP_ROOT/patches/native-f32-mmf.patch"

for tool in git cmake; do
  command -v "$tool" >/dev/null || { echo "$tool not found (sudo apt install $tool)" >&2; exit 1; }
done
if ! command -v nvcc >/dev/null; then
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  else
    echo "nvcc not found: install the CUDA toolkit or put /usr/local/cuda/bin on PATH" >&2
    exit 1
  fi
fi

if [ ! -d "$LLAMA_DIR/.git" ]; then
  mkdir -p "$(dirname "$LLAMA_DIR")"
  git init -q "$LLAMA_DIR"
  git -C "$LLAMA_DIR" remote add origin "$LLAMA_REPO"
fi
cd "$LLAMA_DIR"

# The PR's commit is fetchable by hash on GitHub; the PR ref is the fallback.
if ! git cat-file -e "$LLAMA_COMMIT^{commit}" 2>/dev/null; then
  echo "fetching $LLAMA_COMMIT ..."
  git fetch -q --depth 1 origin "$LLAMA_COMMIT" ||
    git fetch -q origin "pull/$LLAMA_PR/head"
fi
git checkout -q --force "$LLAMA_COMMIT"
git clean -q -fd -e build
echo "llama.cpp at $(git log -1 --format='%h %ad %s' --date=short)"

git apply --check "$PATCH"
git apply "$PATCH"
echo "applied $(basename "$PATCH")"

# 121a-real: the Spark's own architecture with its arch-specific features,
# no PTX for other GPUs. GGML_RPC for the two-machine setup.
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DGGML_RPC=ON \
  -DLLAMA_USE_PREBUILT_UI="$([ "$UI" = 1 ] && echo ON || echo OFF)"
cmake --build build -j "$JOBS" --target llama-server ggml-rpc-server llama-cli

echo
"$LLAMA_BIN/llama-server" --version 2>&1 | tail -n 2
echo "built: $LLAMA_BIN/{llama-server,ggml-rpc-server,llama-cli}"
