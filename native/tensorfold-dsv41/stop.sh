#!/usr/bin/env bash
# Stop and remove both rank containers. Their logs go to logs/ first; the caches stay on disk.
set -uo pipefail
# shellcheck source=config.sh
. "$(dirname "$0")/config.sh"

mkdir -p "$LOGS"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker logs "$RANK0" > "$LOGS/r0-$stamp.log" 2>&1 || true
wssh "docker logs $RANK1" > "$LOGS/r1-$stamp.log" 2>&1 || true
docker stop -t 60 "$RANK0" >/dev/null 2>&1; docker rm "$RANK0" >/dev/null 2>&1
wssh "docker stop -t 60 $RANK1 >/dev/null 2>&1; docker rm $RANK1 >/dev/null 2>&1"
log "stopped (logs in $LOGS/*-$stamp.log)"
