#!/usr/bin/env bash
# Both rank containers, memory on both nodes, the API.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
echo "head:   $(docker ps --filter "name=^$CONTAINER_NAME\$" --format '{{.Status}}' | head -1)"; free -g | awk '/Mem:/ {print "        available", $7, "GiB"}'
echo "worker: $(wssh "docker ps --filter name=^$CONTAINER_NAME\$ --format '{{.Status}}'" 2>/dev/null | head -1)"; wssh free -g 2>/dev/null | awk '/Mem:/ {print "        available", $7, "GiB"}'
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | head -c 300; echo
