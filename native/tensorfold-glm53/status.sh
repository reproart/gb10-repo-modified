#!/usr/bin/env bash
# Both rank containers, the memory each node has left, and the API's /health.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# shellcheck disable=SC2016  # run by bash here and over ssh
mem='free -g | awk "/Mem:/ {print \"available\", \$7, \"GiB\"}"'
echo "head:   $(docker ps -a --filter "name=^$CONTAINER_NAME$" --format '{{.Status}}' | head -1) $(bash -c "$mem")"
echo "worker: $(wssh "docker ps -a --filter 'name=^$CONTAINER_NAME\$' --format '{{.Status}}' | head -1; $mem" | tr '\n' ' ')"
h=$(curl -fsS -m 10 "http://127.0.0.1:$PORT/health" 2>/dev/null) && echo "API: $h" || echo "API: not serving"
[ -d "$KIT_DIR/.git" ] && echo "recipe: $(git -C "$KIT_DIR" log -1 --format='%h %ad' --date=short) (pinned ${KIT_COMMIT:0:7})"
