#!/usr/bin/env bash
# Fetch the vLLM Inkling-2-Mini container and pin it.
#
#   ./pull.sh        image.lock present (it is committed): pull exactly that
#                    digest
#                    no image.lock: pull the tag, check it against
#                    EXPECT_PREFIX, write image.lock. Commit image.lock:
#                    serve.sh then always runs this build, whatever the tag
#                    becomes.
#   rm image.lock && EXPECT_PREFIX= ./pull.sh   move to the tag's current build on purpose
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
# ~10.5 GB compressed: never beside a GPU job (../scripts/mem-prep.sh idle; a Spark has hung so).
"$VINK_ROOT/../scripts/mem-prep.sh" idle || exit 1
ref="$(image_ref)"
echo "== docker pull $ref (~10.5 GB compressed) =="
docker pull "$ref"

digest="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$ref" |
  awk -F@ -v img="$IMAGE" '$1 == img { print $2; exit }')"
if [ -z "$digest" ]; then
  echo "no repository digest for $ref (docker image inspect $ref)" >&2
  exit 1
fi
echo "digest $digest"

if [ -z "$IMAGE_DIGEST" ]; then
  if [ -n "$EXPECT_PREFIX" ] && [[ "$digest" != "sha256:$EXPECT_PREFIX"* ]]; then
    echo "note: this is not the $EXPECT_PREFIX... Docker Hub showed: the tag moved since." >&2
    echo "  docker buildx imagetools inspect $IMAGE:$IMAGE_TAG   lists the index and its platforms" >&2
    echo "Pinning what was pulled; check before committing image.lock." >&2
  fi
  echo "$digest" > "$VINK_ROOT/image.lock"
  echo "wrote ${VINK_ROOT}/image.lock: commit it, so every serve runs this build"
fi
