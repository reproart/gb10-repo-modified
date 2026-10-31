#!/usr/bin/env bash
# The container, the image it runs, the memory left, the API and the metrics sidecar.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
echo "container: $(docker ps -a --filter "name=^$NAME$" --format '{{.Status}} ({{.Image}})' | head -1)"
img=$(docker image inspect -f '{{.Id}} {{.Created}}' "$IMAGE" 2>/dev/null | awk '{ print substr($1, 8, 12), "built", substr($2, 1, 16) }')
echo "image:     $IMAGE ${img:-missing (./build.sh)}"
"$NATIVE_DIR/scripts/mem-prep.sh" status
auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
if m=$(curl -fsS -m 10 ${auth[@]+"${auth[@]}"} "http://127.0.0.1:$PORT/v1/models" 2>/dev/null); then
  echo "API:       :$PORT serves $(echo "$m" | python3 -c 'import json,sys; print(" ".join(d["id"] for d in json.load(sys.stdin)["data"]))')"
else
  echo "API:       not answering on :$PORT"
fi
if [ "$METRICS_PORT" != 0 ] && s=$(curl -fsS -m 5 "http://127.0.0.1:$METRICS_PORT/metrics" 2>/dev/null); then
  echo "sidecar:   $(echo "$s" | grep -E '^vllm:mamba_state_copy_guard_total|^vllm:ple_mmap_ops_total' | tr '\n' ' ')"
fi
