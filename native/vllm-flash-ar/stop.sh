#!/usr/bin/env bash
# Stop the server. The container stays (docker logs works); the next ./serve.sh
# replaces it. A container stopped by hand does not come back after a reboot,
# even with RESTART=unless-stopped.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
# The watchdog goes too (unless it is the one stopping the server for a restart).
pidf="$HOME/.cache/gb10/vllm-flash-ar-watchdog.pid"
if [ -z "${WATCHDOG_CHILD:-}" ] && [ -f "$pidf" ]; then
  wpid=$(cat "$pidf")
  if grep -qs watchdog.sh "/proc/$wpid/cmdline"; then
    kill "$wpid" && log "watchdog stopped (pid $wpid)"
    for _ in $(seq 20); do [ -d "/proc/$wpid" ] || break; sleep 0.25; done   # its lock goes with it
  fi
fi
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  docker stop -t 30 "$NAME" >/dev/null && log "stopped $NAME (docker logs $NAME; docker rm $NAME to remove it)"
else
  log "$NAME is not running"
fi
