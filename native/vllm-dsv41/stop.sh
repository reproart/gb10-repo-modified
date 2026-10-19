#!/usr/bin/env bash
# Stop and remove both rank containers; their logs go to logs/ first.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

mkdir -p "$VD_ROOT/logs"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker logs "$CONTAINER_NAME" > "$VD_ROOT/logs/head-$stamp.log" 2>&1 || true
wssh "docker logs $CONTAINER_NAME" > "$VD_ROOT/logs/worker-$stamp.log" 2>&1 || true
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1
wssh "docker rm -f $CONTAINER_NAME >/dev/null 2>&1"
log "stopped (logs in logs/*-$stamp.log)"
