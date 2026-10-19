#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw) on sudoingX's tuned vLLM with styles01's 1017
# overlay, TP=2 across both Sparks over RoCE: rank 1 on the worker (headless), rank 0 here, then wait
# until the API answers. Both run detached; ./stop.sh stops them, ./status.sh shows them. Boot ~13 min.
#
#   ./pull.sh && ./repack.sh       once: the image on both nodes, the serving tree on both nodes
#   ./serve.sh
#   MAX_NUM_SEQS=1 GPU_UTIL=0.88 ./serve.sh
#   VISION=0 ./serve.sh            text only: the vision tower's ~0.9 GiB a rank goes to KV
#   DEPTH_PROBE=1 ./serve.sh       boot for re-measuring the depth table (README, "Depth table")
#
# Then, as for the other servers (port 8888, the model id tensorfold-dsv41 serves):
#   python3 ../bench/perf.py --levels 1 2
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
[ -n "$IMAGE_DIGEST" ] || echo "warning: no image.lock, running whatever $ref is (./pull.sh pins it)" >&2

# One server at a time: each wants port 8888 and most of the memory, on both nodes for the pair ones.
pgrep -f "sglang.launch_server" >/dev/null 2>&1 && {
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2; exit 1; }
for c in gb10-vllm-b12x tfv41-r0; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
wssh "docker ps --format '{{.Names}}'" | grep -qx tfv41-r1 && {
  echo "tfv41-r1 runs on the worker: ../tensorfold-dsv41/stop.sh first" >&2; exit 1; }
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = 000 ] || {
  echo "something already serves :$PORT" >&2; exit 1; }

# The image and the serving tree, on both nodes.
docker image inspect "$ref" >/dev/null 2>&1 || { echo "no $ref here: ./pull.sh" >&2; exit 1; }
wssh "docker image inspect $(printf '%q' "$ref") >/dev/null 2>&1" || { echo "no $ref on the worker: ./pull.sh" >&2; exit 1; }
python3 -I "$VD_ROOT/repack.py" --check --out "$SERVE_DIR" >/dev/null || {
  python3 -I "$VD_ROOT/repack.py" --check --out "$SERVE_DIR"; echo "./repack.sh builds it" >&2; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -I - --check --out $(printf '%q' "$SERVE_DIR")" \
  < "$VD_ROOT/repack.py" >/dev/null || { echo "the worker's $SERVE_DIR is not a serving tree: ./repack.sh" >&2; exit 1; }

# The fabric on each node: the netdev's IPv4, its RoCE device, and the RoCE v2 GID index whose
# IPv4-mapped GID is that address (derived at every start: a link flap can move the table).
# shellcheck disable=SC2016  # expanded by the bash that runs it, here or on the worker
fabric='IF=$1 HCA=$2
ip=$(ip -4 -o addr show "$IF" 2>/dev/null | awk "{print \$4}" | cut -d/ -f1 | head -1)
[ -n "$HCA" ] || HCA=$(ls "/sys/class/net/$IF/device/infiniband" 2>/dev/null | head -1)
gid=""
if [ -n "$ip" ] && [ -n "$HCA" ]; then
  hex=$(printf "%02x%02x%02x%02x" $(echo "$ip" | tr . " ")); P=/sys/class/infiniband/$HCA/ports/1
  for i in $(seq 0 15); do
    t=$(cat "$P/gid_attrs/types/$i" 2>/dev/null); g=$(cat "$P/gids/$i" 2>/dev/null)
    case "$t" in *v2*) ;; *) continue ;; esac
    case "$g" in *ffff:${hex:0:4}:${hex:4:4}*) gid=$i; break ;; esac
  done
fi
echo "$ip $HCA $gid"'
read -r hip hhca hgid < <(bash -c "$fabric" _ "$FAB_IF" "$FAB_HCA") || true
read -r wip whca wgid < <(wssh "bash -c $(printf '%q' "$fabric") _ $(printf '%q' "$FAB_IF") $(printf '%q' "$FAB_HCA")") || true
hgid="${FAB_GID:-$hgid}" wgid="${FAB_GID:-$wgid}"
[ -n "$hip" ] && [ -n "$wip" ] && [ -n "$hhca" ] && [ -n "$whca" ] && [ -n "$hgid" ] && [ -n "$wgid" ] || {
  echo "fabric not ready on $FAB_IF: head ${hip:-no IPv4} ${hhca:-no RoCE device} gid ${hgid:-none}," \
    "worker ${wip:-no IPv4} ${whca:-no RoCE device} gid ${wgid:-none} (FAB_IF / FAB_HCA / FAB_GID;" \
    "../tensorfold-dsv41/docs/NETWORK.md)" >&2
  exit 1; }

# Caches on both nodes, the Markov token list the drafter's bias rows come from, the depth file.
depth_env=("VLLM_DSPARK_DEPTH_POLICY=$DEPTH_POLICY" "VLLM_DSPARK_ROUND_MS=$ROUND_MS")
mkdir -p "$CACHE_DIR" "$VD_ROOT/logs"
# (once: the containers run as root, so what they write in CACHE_DIR is root's)
[ -f "$CACHE_DIR/markov_top_tokens.json" ] || cp "$VD_ROOT/data/markov_top_tokens.json" "$CACHE_DIR/"
wssh "mkdir -p $(printf '%q' "$CACHE_DIR")"
wssh "test -f $(printf '%q' "$CACHE_DIR/markov_top_tokens.json")" ||
  scp -q -o BatchMode=yes "$VD_ROOT/data/markov_top_tokens.json" "$WORKER_SSH:$CACHE_DIR/"
if [ "$DEPTH_PROBE" = 1 ]; then
  depth_env=(VLLM_DSPARK_DEPTH_POLICY=fixed:5 VLLM_DSPARK_DEPTH_FILE=/root/.cache/dspark_depth VLLM_DSPARK_DEPTH_LOG=200
             "VLLM_DSPARK_ROUND_MS=$ROUND_MS")
  echo fixed:5 > "$CACHE_DIR/dspark_depth" 2>/dev/null || sudo -n tee "$CACHE_DIR/dspark_depth" <<< fixed:5 >/dev/null || true
fi
# GB10: CUDA allocations do not reclaim the page cache, and vLLM sizes its pools to "free".
if [ "$DROP_CACHES" = 1 ]; then
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || echo "note: drop_caches needs sudo -n here" >&2
  wssh "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1" || echo "note: drop_caches needs sudo -n on the worker" >&2
fi

# shellcheck disable=SC2206  # RECIPE_ENV / EXTRA_ENV are NAME=value lists, split on purpose
envs=($RECIPE_ENV "${depth_env[@]}" VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      VLLM_DISABLED_KERNELS=FlashInferCutlassMxfp8LinearKernel HF_HUB_OFFLINE=1 NCCL_IB_DISABLE=0 NCCL_NET=IB
      "NCCL_SOCKET_IFNAME=$FAB_IF" "GLOO_SOCKET_IFNAME=$FAB_IF" "TP_SOCKET_IFNAME=$FAB_IF")
# shellcheck disable=SC2206
extra_env=($EXTRA_ENV)
for e in "${extra_env[@]}"; do   # replace a value above of the same name (docker would keep the first)
  for i in "${!envs[@]}"; do [ "${envs[$i]%%=*}" = "${e%%=*}" ] && unset 'envs[i]'; done
  envs+=("$e")
done

run_args() {  # $1 = rank, $2 = this node's fabric IPv4, $3 its RoCE device, $4 its GID index
  local r=$1 e spec
  spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_parallel_config\":{\"tensor_parallel_size\":$DRAFT_TP}}"
  printf '%s\n' -d --name "$CONTAINER_NAME" --restart no --network host --ipc host --shm-size 64g --gpus all \
    --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$SERVE_DIR:$SERVE_DIR:ro" -v "$CACHE_DIR:/root/.cache" \
    -e "NCCL_IB_HCA=$3" -e "NCCL_IB_GID_INDEX=$4" -e "VLLM_HOST_IP=$2"
  [ -n "$CPUSET" ] && printf '%s\n' --cpuset-cpus "$CPUSET"
  for e in "${envs[@]}"; do printf '%s\n' -e "$e"; done
  printf '%s\n' --entrypoint vllm "$ref" serve "$SERVE_DIR" \
    --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 2 --nnodes 2 --node-rank "$r" --master-addr "$hip" --master-port 25000 \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --engram-config "$ENGRAM_CONFIG" --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --kernel-config '{"enable_flashinfer_autotune":false}' \
    --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  [ "$DRAFT_TOKENS" -gt 0 ] && printf '%s\n' --speculative-config "$spec"
  [ -n "$KV_BYTES" ] && printf '%s\n' --kv-cache-memory-bytes "$KV_BYTES"
  [ "$VISION" = 1 ] || printf '%s\n' --limit-mm-per-prompt '{"image":0}'
  [ "$r" != 0 ] && printf '%s\n' --headless
  [ -n "$API_KEY" ] && [ "$r" = 0 ] && printf '%s\n' --api-key "$API_KEY"
  # shellcheck disable=SC2086  # EXTRA_ARGS is a flag list, split on purpose
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

echo "image   $ref"
echo "model   $SERVE_DIR (from $MODEL_DIR + $ENGRAM_DIR), on both nodes"
echo "fabric  head $hip $hhca gid $hgid | worker $WORKER_SSH $wip $whca gid $wgid | $FAB_IF"
echo "serve   ctx $MAX_MODEL_LEN, $MAX_NUM_SEQS requests, batch $MAX_BATCHED_TOKENS, mem $GPU_UTIL, KV ${KV_BYTES:-from mem}," \
  "DSpark $DRAFT_TOKENS (drafter TP$DRAFT_TP, $([ "$DEPTH_PROBE" = 1 ] && echo "probe: fixed:5 from the depth file" || echo "$DEPTH_POLICY")), vision $([ "$VISION" = 1 ] && echo on || echo off), CPUs ${CPUSET:-all}"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\""
echo

log "starting rank 1 (headless) on the worker"
wssh "docker rm -f $CONTAINER_NAME >/dev/null 2>&1 || true"
mapfile -t a1 < <(run_args 1 "$wip" "$whca" "$wgid")
wssh "docker run $(printf '%q ' "${a1[@]}")" >/dev/null
log "starting rank 0 here"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
mapfile -t a0 < <(run_args 0 "$hip" "$hhca" "$hgid")
docker run "${a0[@]}" >/dev/null

log "waiting for :$PORT/v1/models (~13 min: JIT warmup, graph capture)"
t0=$(date +%s)
until curl -fsS -m 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "\"$SERVED_MODEL_NAME\""; do
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" = true ] || {
    log "rank 0 stopped"; docker logs --tail 60 "$CONTAINER_NAME" 2>&1; exit 1; }
  [ "$(wssh "docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null")" = true ] || {
    log "rank 1 stopped"; wssh "docker logs --tail 60 $CONTAINER_NAME 2>&1"; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt 3600 ] || { log "not ready after 60 min"; exit 1; }
  sleep 15
done
log "ready after $(( $(date +%s) - t0 )) s"
free -g | awk '/Mem:/ {print "head:   used", $3, "GiB, available", $7, "GiB"}'
wssh "free -g" | awk '/Mem:/ {print "worker: used", $3, "GiB, available", $7, "GiB"}'
