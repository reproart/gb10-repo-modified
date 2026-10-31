#!/usr/bin/env bash
# Stop the server. The container stays (docker logs works); the next ./serve.sh
# replaces it. A container stopped by hand does not come back after a reboot,
# even with RESTART=unless-stopped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop -t 30 "$NAME" >/dev/null && log "stopped $NAME (docker logs $NAME; docker rm $NAME to remove it)"
else
  log "$NAME is not running"
fi
