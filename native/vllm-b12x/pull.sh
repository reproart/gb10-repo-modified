#!/usr/bin/env bash
# Fetch the b12x vLLM container and pin it.
#
#   ./pull.sh        image.lock present: pull exactly that digest
#                    no image.lock: pull :latest, check it against
#                    EXPECT_PREFIX (the digest Docker Hub showed), write
#                    image.lock. Commit image.lock: serve.sh then always runs
#                    this build, whatever :latest becomes.
#   rm image.lock && ./pull.sh     move to the current :latest on purpose
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
echo "== docker pull $ref (~9 GB compressed) =="
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
    # Docker Hub's tag page lists the per-platform (linux/arm64) manifest;
    # for a multi-platform image the local digest is the index's instead.
    echo "note: this is not the $EXPECT_PREFIX... Docker Hub showed. Either :latest moved" \
      "since, or the image is multi-platform and Hub showed the arm64 manifest:" >&2
    echo "  docker buildx imagetools inspect $IMAGE:latest   lists both" >&2
    echo "Pinning what was pulled; check before committing image.lock." >&2
  fi
  echo "$digest" > "$B12X_ROOT/image.lock"
  echo "wrote ${B12X_ROOT}/image.lock: commit it, so every serve runs this build"
fi
