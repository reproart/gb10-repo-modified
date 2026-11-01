#!/usr/bin/env bash
# Build the image: the official Flash-Next vLLM image (pinned by digest in the
# Dockerfile) + this recipe's patches. Each step prints its "... OK" line; a
# missing one means it did not run. Pulls a ~20 GiB base the first time; a
# rebuild after a patch change takes minutes (layer cache).
# Then src/test_det_moe_align_gpu.py checks DET_MOE_ALIGN's layout against the
# image's CUDA kernel (seconds on the GPU; DET_ALIGN_CHECK=0 skips it).
#   ./build.sh
#   ./build.sh --no-cache        anything docker build takes
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
# ~20 GB of image through the page cache and a CUDA compile: never beside a GPU job.
"$NATIVE_DIR/scripts/mem-prep.sh" idle || exit 1
log "building $IMAGE"
docker build -t "$IMAGE" "$@" "$VFA_ROOT"
log "built $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c8-19)); ./serve.sh starts it (a running server keeps its old image until then)"
# DET_MOE_ALIGN's layout against the image's own CUDA kernel, a few seconds on the
# GPU (still idle: checked above). A failure concerns DET_MOE_ALIGN=1, the default.
if [ "${DET_ALIGN_CHECK:-1}" = 1 ]; then
  log "checking the deterministic MoE alignment against the kernel (src/test_det_moe_align_gpu.py; DET_ALIGN_CHECK=0 skips)"
  if docker run --rm --gpus all --entrypoint python3 -v "$VFA_ROOT/src:/w:ro" "$IMAGE" /w/test_det_moe_align_gpu.py; then
    log "DET_MOE_ALIGN=1 (the default) checks out on this image"
  else
    log "WARNING: the check failed or could not run: start this image with DET_MOE_ALIGN=0 (vLLM's kernel) until it passes"
  fi
fi
