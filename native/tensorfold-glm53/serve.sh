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
for c in gb10-vllm-b12x gb10-vllm-dsv41 tfv41-r0 qwen38-flash; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
wssh "docker ps --format '{{.Names}}'" | grep -qxE 'tfv41-r1|gb10-vllm-dsv41' && {
  echo "a DeepSeek rank runs on the worker: ../tensorfold-dsv41/stop.sh or ../vllm-dsv41/stop.sh first" >&2; exit 1; }

"$NATIVE_DIR/scripts/host-tuning.sh" --check 2>/dev/null | grep -q "applied$" ||
  echo "note: host tuning not applied here (earlyoom, no swap for the engine): ../scripts/host-tuning.sh" >&2

kit_env
# A restart stops both ranks first, so that the memory below is prepared with the GPUs idle.
if [ "${1:-}" = restart ]; then shift; "$KIT_DIR/stop.sh"; fi
# Both nodes' memory before the ranks take it (../scripts/mem-prep.sh): page cache dropped, free memory compacted.
if [ "$MEM_PREP" = 1 ] && ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  "$NATIVE_DIR/scripts/mem-prep.sh" prep "$WORKER_SSH" || true
fi

args=()
[ -n "$EFFORT" ] && args+=(--reasoning-effort "$EFFORT")
log "PARALLEL $PARALLEL, drafter $DRAFTER, effort ${EFFORT:-the recipe default}, weights in $HF_CACHE"
exec "$KIT_DIR/start.sh" "${args[@]}" "$@"
