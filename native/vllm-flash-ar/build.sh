#!/usr/bin/env bash
# Build the image: the official Flash-Next vLLM image (pinned by digest in the
# Dockerfile) + this recipe's patches. Each step prints its "... OK" line; a
# missing one means it did not run. Pulls a ~20 GiB base the first time; a
# rebuild after a patch change takes minutes (layer cache).
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
