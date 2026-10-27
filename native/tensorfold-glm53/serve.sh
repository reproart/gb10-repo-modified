#!/usr/bin/env bash
# Serve GLM-5.3-Flash on both Sparks with Mia's start.sh: rank 1 on the worker,
# rank 0 here with the API on :8888, then her smoke test. First start ~10 min
# (kernel compiles, prepared weights), later ones under a minute.
#   ./serve.sh                       start (if it already runs: says so)
#   ./serve.sh restart               stop, then start with the current settings
#   PARALLEL=8 ./serve.sh restart    8 requests at once (config.sh: what it costs)
#   CONTEXT=524288 ./serve.sh restart, KV=bf16 ..., DRAFTER=mtp ... (Mia's settings)
#   ./serve.sh restart --max-tokens 65536      more flags go to tensorfold serve
# Then, as for the other servers here:
#   python3 ../bench/perf.py --levels 1 2 4 8
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_kit

# One server at a time: each wants port 8888 and most of the memory of both nodes.
pgrep -f "sglang.launch_server" >/dev/null 2>&1 && {
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2; exit 1; }
for c in gb10-vllm-b12x gb10-vllm-dsv41 tfv41-r0; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
wssh "docker ps --format '{{.Names}}'" | grep -qxE 'tfv41-r1|gb10-vllm-dsv41' && {
  echo "a DeepSeek rank runs on the worker: ../tensorfold-dsv41/stop.sh or ../vllm-dsv41/stop.sh first" >&2; exit 1; }

"$NATIVE_DIR/scripts/host-tuning.sh" --check 2>/dev/null | grep -q "applied$" ||
  echo "note: host tuning not applied here (earlyoom, no swap for the engine): ../scripts/host-tuning.sh" >&2

running=0
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && running=1
if [ "$DROP_CACHES" = 1 ] && { [ "$running" = 0 ] || [ "${1:-}" = restart ]; }; then
  drop='sync; sudo -n sh -c "echo 1 > /proc/sys/vm/drop_caches"'
  if bash -c "$drop" 2>/dev/null && wssh "$drop" 2>/dev/null; then
    log "page cache dropped on both nodes"
  else
    log "page cache not dropped (no passwordless sudo): TensorFold sizes its pool to what is free"
  fi
fi

args=()
[ "${1:-}" = restart ] && { args+=(restart); shift; }
[ -n "$EFFORT" ] && args+=(--reasoning-effort "$EFFORT")
kit_env
log "PARALLEL $PARALLEL, drafter $DRAFTER, effort ${EFFORT:-the recipe default}, weights in $HF_CACHE"
exec "$KIT_DIR/start.sh" "${args[@]}" "$@"
