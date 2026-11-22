#!/usr/bin/env bash
# Stop the container (graceful, 60 s), as ../vllm-flash-ar/stop.sh does.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop -t 60 "$NAME" >/dev/null && docker rm "$NAME" >/dev/null
  log "stopped $NAME"
else
  log "$NAME is not running"
fi
