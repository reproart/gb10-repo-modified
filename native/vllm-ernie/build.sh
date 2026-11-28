#!/usr/bin/env bash
# Build the image here (docker/Dockerfile, every source pinned) and copy it to the worker.
#   ./build.sh            build, then copy if the worker answers
#   ./build.sh copy       copy only
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
ref="$(image_ref)"
# Pulls and copies never beside a GPU job (../scripts/mem-prep.sh idle; a Spark has hung so).
"$VE_ROOT/../scripts/mem-prep.sh" idle || exit 1
if [ "${1:-}" != copy ]; then
  log "building $ref (base ~10.3 GB compressed)"
  docker build -t "$ref" "$VE_ROOT/docker"
fi
if wssh true 2>/dev/null; then
  "$VE_ROOT/../scripts/mem-prep.sh" idle "$WORKER_SSH" || exit 1
  log "copying $ref to $WORKER_SSH"
  docker save "$ref" | wssh docker load
else
  log "worker $WORKER_SSH not reachable: ./build.sh copy once it is"
fi
