#!/usr/bin/env bash
# Get the engine image on both nodes, pinned. Without image.lock: pull IMAGE:IMAGE_TAG here, record the
# digest it resolved to in image.lock (commit it). With it: pull exactly that digest, here and on the worker.
#   ./pull.sh
#   rm image.lock && IMAGE_TAG=v2 ./pull.sh     move to another tag on purpose
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
if [ -z "$IMAGE_DIGEST" ]; then
  log "pulling $IMAGE:$IMAGE_TAG (no image.lock yet)"
  docker pull "$IMAGE:$IMAGE_TAG"
  digest=$(docker image inspect "$IMAGE:$IMAGE_TAG" --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    | grep -m1 "^$IMAGE@" | cut -d@ -f2)
  [ -n "$digest" ] || { echo "no repo digest for $IMAGE:$IMAGE_TAG" >&2; exit 1; }
  echo "$digest" > "$VD_ROOT/image.lock"
  IMAGE_DIGEST=$digest
  log "pinned $IMAGE@$digest in image.lock"
else
  log "pulling the pinned $(image_ref)"
  docker pull "$(image_ref)"
fi
log "worker ($WORKER_SSH): pulling $(image_ref)"
wssh "docker pull $(printf '%q' "$(image_ref)")"
log "both nodes hold $(image_ref)"
