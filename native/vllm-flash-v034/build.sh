#!/usr/bin/env bash
# Build the image: vllm/vllm-openai:v0.34.0 at the digest in image.lock + the two patches in
# the Dockerfile. The first run pulls the ~10 GB base; a rebuild takes a minute. Each patch
# prints its "... OK" line.
#   ./build.sh
#   ./build.sh --no-cache        anything docker build takes
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
# A base image through the page cache: never beside a GPU job.
"$NATIVE_DIR/scripts/mem-prep.sh" idle || exit 1
ref="$(base_ref)"
docker image inspect "$ref" >/dev/null 2>&1 || { log "pulling $BASE_IMAGE:$BASE_TAG at ${BASE_DIGEST:0:19}..."; docker pull "$ref"; }
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$V034_ROOT/Dockerfile" "$NATIVE_DIR/vllm-flash-ar/src/vllm_fp8_hybrid.py" "$V034_ROOT/src/patch_lm_head_quant.py" "$ctx/"
log "building $IMAGE on $ref"
docker build -t "$IMAGE" --build-arg "BASE=$ref" "$@" "$ctx"
log "built $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c8-19)); ./prepare.sh once, then ./serve.sh"
