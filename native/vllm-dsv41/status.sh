#!/usr/bin/env bash
# Both rank containers, the memory each node has left, and the API.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# shellcheck disable=SC2016  # run by bash here and over ssh
mem='free -g | awk "/Mem:/ {print \"available\", \$7, \"GiB\"}"'
echo "head:   $(docker ps -a --filter "name=^$CONTAINER_NAME$" --format '{{.Status}}' | head -1) $(bash -c "$mem")"
echo "worker: $(wssh "docker ps -a --filter 'name=^$CONTAINER_NAME\$' --format '{{.Status}}' | head -1; $mem" | tr '\n' ' ')"
if curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"$SERVED_MODEL_NAME\""; then
  echo "API: serving $SERVED_MODEL_NAME on :$PORT"
else
  echo "API: not serving"
fi
