#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (Intel AutoRound int4 + int8 lm_head + fp8 side
# layers, the n-gram table mmapped from NVMe) on vLLM in Docker, one Spark:
# config.sh's settings, the one-server-at-a-time checks, memory prepared, then
# scripts/serve-intel-ar.sh (the recipe's launcher: image probes, the docker
# run), then a wait until the API answers. Rerun to restart with new settings.
#   ./build.sh && ./download.sh        once
#   ./serve.sh
#   MTP=2 SEQS=16 ./serve.sh            any setting from config.sh
#   PIN_PROMPT="You are HomeBot, the household assistant." ./serve.sh
# Then, as for every server here:
#   GB10_METRICS_PORT=18400 python3 ../bench/perf.py
#   python3 ../bench/consistency.py
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker

# One server at a time: each wants most of the 128 GB.
pgrep -f "sglang.launch_server" >/dev/null 2>&1 && {
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2; exit 1; }
for c in gb10-vllm-b12x tfv41-r0 gb10-vllm-dsv41 glm53-flash-tf; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME" &&
   [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT" >&2; exit 1
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE: ./build.sh first" >&2; exit 1; }

# GB10 memory before the start (../scripts/mem-prep.sh); it leaves memory alone
# while the previous container of this recipe still runs (a restart).
if [ "$MEM_PREP" = 1 ]; then "$NATIVE_DIR/scripts/mem-prep.sh" prep || true; fi

"$VFA_ROOT/scripts/serve-intel-ar.sh"
[ "$WAIT_MIN" = 0 ] && exit 0

log "waiting for http://127.0.0.1:$PORT/v1/models (weights ~5 min; the first boot of a new image compiles kernels)"
auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
for _ in $(seq 1 $((WAIT_MIN * 6))); do
  if curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "serving $(echo "$SERVED_NAME" | cut -d' ' -f1) on :$PORT; logs: docker logs -f $NAME"
    exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    log "$NAME stopped during boot:"; docker logs --tail 60 "$NAME" 2>&1; exit 1
  fi
  sleep 10
done
log "not answering after $WAIT_MIN min; still loading? docker logs -f $NAME"
exit 1
