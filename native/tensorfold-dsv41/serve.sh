#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, 1M context) on TensorFold,
# TP=2 across both Sparks: rank 0 here (the head), wait for its rendezvous
# port, then rank 1 on the worker, then wait until the API answers. Both
# ranks run detached; ./stop.sh stops them, ./status.sh shows them. First
# start ~15 min (each rank writes its weight cache), later ones ~2-4 min.
#
#   ./serve.sh
#   MAX_RUNNING=2 ./serve.sh                 fewer concurrent requests
#   CONTEXT_LENGTH=262144 ./serve.sh         shorter window (measured: no faster)
#   PREFLIGHT=0 ./serve.sh                   skip ./preflight.sh
#   EXTRA_ENV='TF_DS_TIMING=1' ./serve.sh    per-section timings in the rank logs
#   CPUSET= ./serve.sh                       no CPU pinning (the recipe as measured)
#
# Then, as for the other servers here (same port, bench/ unchanged):
#   ./verify.sh
#   python3 ../bench/perf.py --levels 1 2 4
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
# One server at a time: each one wants port 8888 and ~100 GB a node.
if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx gb10-vllm-b12x; then
  echo "the vllm-b12x container runs: stop it first (docker stop gb10-vllm-b12x)" >&2
  exit 1
fi
if [ "$PREFLIGHT" = 1 ]; then
  "$ROOT/preflight.sh" || exit 1
fi
mkdir -p "$CACHE_DIR" "$LOGS"
wssh "mkdir -p '$CACHE_DIR'"

# Every TF_DS_* setting below is explained in docs/WHAT-I-CHANGED.md,
# "Launcher settings".
# shellcheck disable=SC2054  # the commas are inside values (TF_DS_WARM_LENGTHS, RDMA device lists)
envs=(
  TF_DS_REPLAY=1 TF_DS_REPLAY_FLOOR=1
  "TF_DS_PREFILL_CHUNK=$CHUNKED_PREFILL"
  TF_DS_RANK_CACHE=/cache/tf-ds-rank TF_DS_RANK_CACHE_READERS=32
  TF_DS_WARM_LENGTHS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,131,514,1024,2113
  TF_DS_ENGRAM=/engram
  "TF_DS_SPILL_GIB=$SPILL_GIB" TF_DS_SPILL_DIR=/cache/tf-ds-spill
  TF_DS_ENGRAM_DONTNEED=1
  TRITON_CACHE_DIR=/cache/triton CUDA_CACHE_PATH=/cache/nv
  "TF_RDMA_DEVICES=$RDMA_DEVICES" "TF_RDMA_GID_INDEX=$GID_INDEX"
  "NCCL_IB_HCA=$NCCL_IB_HCA" "NCCL_IB_GID_INDEX=$GID_INDEX" "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  TORCH_EXTENSIONS_DIR=/cache/torch_extensions HF_HUB_OFFLINE=1
)
# EXTRA_ENV replaces a value above of the same name instead of adding a second
# one (docker passes both, and the first wins).
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list, split on purpose
extra_env=($EXTRA_ENV)
for e in "${extra_env[@]}"; do
  for i in "${!envs[@]}"; do
    [ "${envs[$i]%%=*}" = "${e%%=*}" ] && unset 'envs[i]'
  done
  envs+=("$e")
done

run_args() {  # $1 = rank: the docker run words, one per line, for the head or the worker
  local r=$1 e
  printf '%s\n' -d --name "tfv41-r$r" --restart "$RESTART_POLICY"
  [ -n "$CGROUP_PARENT" ] && printf '%s\n' --cgroup-parent "$CGROUP_PARENT"
  [ -n "$CPUSET" ] && printf '%s\n' --cpuset-cpus "$CPUSET"
  printf '%s\n' --gpus all --network host --ipc host --shm-size 32g --oom-score-adj 1000 \
    --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_PTRACE \
    -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$CACHE_DIR:/cache"
  # rank 0 serves HTTP, so only it needs the keys
  [ -n "$API_KEY_FILE" ] && [ "$r" = 0 ] && printf '%s\n' -v "$API_KEY_FILE:/run/keys:ro"
  for e in "${envs[@]}"; do printf '%s\n' -e "$e"; done
  printf '%s\n' --entrypoint tensorfold "$IMAGE" serve /model --tp 2 --rank "$r" --master "$HEAD_IP" \
    --host "$HOST" --port "$PORT" --name "$SERVED_MODEL_NAME"
  [ -n "$SERVED_ALIAS" ] && printf '%s\n' --alias "$SERVED_ALIAS"
  [ -n "$API_KEY_FILE" ] && [ "$r" = 0 ] && printf '%s\n' --api-key-file /run/keys
  printf '%s\n' --context "$CONTEXT_LENGTH"
  [ "$VISION" = 1 ] && printf '%s\n' --vision
  printf '%s\n' --parallel "$MAX_RUNNING" --mtp-drafts "$DRAFT_TOKENS" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --no-thinking --no-update-check
  # shellcheck disable=SC2086  # EXTRA_ARGS is a flag list, split on purpose
  [ -n "$EXTRA_ARGS" ] && printf '%s\n' $EXTRA_ARGS
  return 0
}

echo "image   $IMAGE ($TF_REPO @ $TF_COMMIT)"
echo "model   $MODEL_DIR + Engram $ENGRAM_DIR, on both nodes"
echo "nodes   head $HEAD_IP (rank 0), worker $WORKER_SSH / $WORKER_IP (rank 1), RDMA $RDMA_DEVICES, GID $GID_INDEX"
echo "serve   ctx $CONTEXT_LENGTH, $MAX_RUNNING requests, DSpark $DRAFT_TOKENS drafts, chunk $CHUNKED_PREFILL," \
  "spill $SPILL_GIB GiB, temperature $TEMPERATURE / top-p $TOP_P$([ "$VISION" = 1 ] && echo ", vision"), CPUs ${CPUSET:-all}"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\"$([ -n "$API_KEY_FILE" ] && echo " (key file $API_KEY_FILE)")"
echo

log "starting rank 0 on the head"
docker rm -f "$RANK0" >/dev/null 2>&1 || true
mapfile -t a0 < <(run_args 0)
docker run "${a0[@]}" >/dev/null
for _ in $(seq 1 120); do ss -ltn | grep -q ":$STORE_PORT " && break; sleep 2; done
ss -ltn | grep -q ":$STORE_PORT " || { log "rank 0 never opened port $STORE_PORT"; docker logs --tail 40 "$RANK0"; exit 1; }

log "starting rank 1 on the worker ($WORKER_SSH)"
wssh "docker rm -f $RANK1 >/dev/null 2>&1 || true"
mapfile -t a1 < <(run_args 1)
wssh "docker run $(printf '%q ' "${a1[@]}")" >/dev/null

log "waiting for $API/models (the first start writes the rank caches: slow)"
for _ in $(seq 1 360); do
  healthy && { log "serving $SERVED_MODEL_NAME at $API; ./verify.sh checks both ranks with real replies"; exit 0; }
  docker ps --format '{{.Names}}' | grep -qx "$RANK0" || { log "$RANK0 exited"; docker logs --tail 60 "$RANK0"; exit 1; }
  wssh "docker ps --format '{{.Names}}'" | grep -qx "$RANK1" || { log "$RANK1 exited"; wssh "docker logs --tail 60 $RANK1"; exit 1; }
  sleep 10
done
log "not ready after 60 min"; exit 1
