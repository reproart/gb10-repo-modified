#!/usr/bin/env bash
# Serve ERNIE 5.5 Flash Preview, TP=2 + EP=2 across both Sparks over RoCE: rank 1
# on the worker (headless), rank 0 here, then wait until the API answers. Both
# run detached; ./stop.sh stops them, ./status.sh shows them.
#
#   ./build.sh                       once: the image here and on the worker
#   ./serve.sh                       the runtime's 2x-1m-fast (PROFILE=spark MODE=tp2-dspark)
#   MODE=tp2-ar ./serve.sh           the runtime's pure-AR profile
#   PROFILE=stock ./serve.sh         vLLM-ERNIE as released, no runtime patches (the A/B)
#   WEIGHTS=baidu ./serve.sh         Baidu's official weights
#
# Then, as for the other servers (port 8888):
#   python3 ../bench/consistency.py && python3 ../bench/perf.py --levels 1 2 4 8
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
case "$PROFILE" in spark|stock) ;; *) echo "PROFILE must be spark or stock, not '$PROFILE'" >&2; exit 2 ;; esac
case "$MODE" in tp2-dspark|tp2-ar) ;; *) echo "MODE must be tp2-dspark or tp2-ar, not '$MODE'" >&2; exit 2 ;; esac
wssh true 2>/dev/null || { echo "the worker $WORKER_SSH does not answer over ssh (key-based, BatchMode)" >&2; exit 1; }

# One server at a time: each wants port 8888 and most of the memory, on both nodes for the pair ones.
pgrep -f "sglang.launch_server" >/dev/null 2>&1 && {
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2; exit 1; }
for c in gb10-vllm-ling gb10-vllm-inkling gb10-sglang-inkling gb10-vllm-b12x gb10-vllm-dsv41 tfv41-r0 glm53-flash-tf qwen38-flash qwen38-flash-v034; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
wssh "docker ps --format '{{.Names}}'" | grep -qx -E 'tfv41-r1|gb10-vllm-dsv41|glm53-flash-tf' && {
  echo "a pair server runs on the worker: stop it first" >&2; exit 1; }
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" = 000 ] || {
  echo "something already serves :$PORT" >&2; exit 1; }

# The image and the weights, on both nodes.
docker image inspect "$ref" >/dev/null 2>&1 || { echo "no $ref here: ./build.sh" >&2; exit 1; }
wssh "docker image inspect $(printf '%q' "$ref") >/dev/null 2>&1" || { echo "no $ref on the worker: ./build.sh copy" >&2; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "no checkpoint at $MODEL_DIR (WEIGHTS=$WEIGHTS; README, Weights)" >&2; exit 1; }
wssh "test -f $(printf '%q' "$MODEL_DIR/config.json")" || { echo "no checkpoint at $MODEL_DIR on the worker" >&2; exit 1; }

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

mkdir -p "$CACHE_DIR" "$VE_ROOT/logs"
wssh "mkdir -p $(printf '%q' "$CACHE_DIR")"
if [ "$MEM_PREP" = 1 ]; then "$VE_ROOT/../scripts/mem-prep.sh" prep "$WORKER_SSH" || true; fi

envs=(HF_HUB_OFFLINE=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
      NCCL_NET=IB NCCL_IB_DISABLE=0 WORLD_SIZE=2 "MASTER_ADDR=$hip"
      "NCCL_SOCKET_IFNAME=$FAB_IF" "GLOO_SOCKET_IFNAME=$FAB_IF" "TP_SOCKET_IFNAME=$FAB_IF")
if [ "$PROFILE" = spark ]; then
  envs+=("SPARK_USE_SPARKNET=$SPARKNET" "SPARK_SERVED_MODEL_NAME=$SERVED_MODEL_NAME" SPARK_BIND_HOST="$HOST"
         SPARKNET_ROCE_TOPOLOGY=direct SPARKNET_ROCE_CACHE_DIR=/root/.cache/sparknet_roce)
  [ "$SPARKNET" = 1 ] && envs+=("SPARKNET_ROCE_PEER_HCAS=$SPARKNET_PEER_HCAS" "SPARKNET_ROCE_SUBNETS=$SPARKNET_SUBNETS")
  [ -n "$API_KEY" ] && envs+=("VLLM_API_KEY=$API_KEY")
fi
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list, split on purpose
extra_env=($EXTRA_ENV)
for e in "${extra_env[@]}"; do   # replace a value above of the same name (docker would keep the first)
  for i in "${!envs[@]}"; do [ "${envs[$i]%%=*}" = "${e%%=*}" ] && unset 'envs[i]'; done
  envs+=("$e")
done

run_args() {  # $1 = rank, $2 = this node's fabric IPv4, $3 its RoCE device, $4 its GID index
  local r=$1 e spec
  printf '%s\n' -d --name "$CONTAINER_NAME" --restart no --network host --ipc host --shm-size 64g --gpus all \
    --device /dev/infiniband --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$MODEL_DIR:$MODEL_DIR:ro" -v "$CACHE_DIR:/root/.cache" \
    -e "NCCL_IB_HCA=$3" -e "NCCL_IB_GID_INDEX=$4" -e "VLLM_HOST_IP=$2"
  for e in "${envs[@]}"; do printf '%s\n' -e "$e"; done
  if [ "$PROFILE" = spark ]; then
    # The runtime's launcher: its env, its flags, its patches (via sitecustomize).
    printf '%s\n' --entrypoint bash "$ref" /opt/spark-ernie/scripts/serve_spark_ernie.sh \
      "$MODE" direct "$r" "$MODEL_DIR" "$PORT" 29801
    return 0
  fi
  spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"probabilistic\"}"
  printf '%s\n' --entrypoint vllm "$ref" serve "$MODEL_DIR" --trust-remote-code \
    --served-model-name "$SERVED_MODEL_NAME" --host "$HOST" --port "$PORT" \
    --tensor-parallel-size 2 --enable-expert-parallel --nnodes 2 --node-rank "$r" \
    --master-addr "$hip" --master-port 29801 --distributed-executor-backend mp \
    --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype fp8 --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --generation-config auto \
    --reasoning-parser ernie5 --tool-call-parser ernie5 --enable-auto-tool-choice \
    --enable-prompt-tokens-details
  [ "$DRAFT_TOKENS" -gt 0 ] && [ "$MODE" = tp2-dspark ] && printf '%s\n' --speculative-config "$spec"
  [ "$r" != 0 ] && printf '%s\n' --headless
  [ -n "$API_KEY" ] && [ "$r" = 0 ] && printf '%s\n' --api-key "$API_KEY"
  # shellcheck disable=SC2086  # EXTRA_ARGS is a flag list, split on purpose
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

echo "image   $ref"
echo "model   $MODEL_DIR ($WEIGHTS), on both nodes"
echo "fabric  head $hip $hhca gid $hgid | worker $WORKER_SSH $wip $whca gid $wgid | $FAB_IF"
echo "serve   $PROFILE $MODE$([ "$PROFILE" = spark ] && echo ", sparknet $SPARKNET" || echo ", ctx $MAX_MODEL_LEN, mem $GPU_UTIL, DSpark $DRAFT_TOKENS")"
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

log "waiting for :$PORT/v1/models (the runtime's note: a cold TP=2 load takes 8-10 min)"
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
