#!/usr/bin/env bash
# Serve thinkingmachines/Inkling-2-Mini on one DGX Spark with vLLM in the
# vllm/vllm-openai:inkling-2-mini container, in the foreground (Ctrl-C stops
# it). The vLLM recipe's dgx_spark_gb10 command, with its values as knobs
# (config.sh); the SGLang twin is ../sglang-inkling/ (same weights, port and
# model id, for an engine A/B):
#
#   ./pull.sh                           once: the container, pinned by image.lock
#   ./serve.sh                          DSpark, 7 tokens, probabilistic draft sampling
#   SPEC_TOKENS=0 ./serve.sh            plain decode
#   DRAFT_SAMPLE_METHOD= ./serve.sh     vLLM's default draft sampling
#   GPU_UTIL=0.9 ./serve.sh             the recipe's (vLLM's default) memory share
#   EXTRA_ARGS='--max-num-seqs 8' ./serve.sh
#
# Then, as for every server here:
#   python3 ../bench/perf.py
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
if ! docker image inspect "$ref" >/dev/null 2>&1; then
  echo "container $ref not here: run $VINK_ROOT/pull.sh first" >&2
  exit 1
fi
[ -n "$IMAGE_DIGEST" ] || echo "warning: no image.lock, running whatever $IMAGE:$IMAGE_TAG is locally (./pull.sh pins it)" >&2
[ -f "$MODEL_DIR/config.json" ] || {
  echo "no checkpoint at $MODEL_DIR (config.json missing); download: see MODEL_DIR in $VINK_ROOT/config.sh" >&2
  exit 1; }
[[ "$SPEC_TOKENS" =~ ^[0-9]+$ ]] || { echo "SPEC_TOKENS must be a number (0 = off), not '$SPEC_TOKENS'" >&2; exit 2; }

# One server at a time: the native SGLang service, a llama.cpp server or
# another container would hold the memory, and any HTTP answer on the port
# means it is taken.
if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2
  exit 1
fi
for c in gb10-sglang-inkling gb10-vllm-b12x qwen38-flash tfv41-r0 gb10-vllm-dsv41 glm53-flash-tf; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c (another model server) runs: stop it first" >&2; exit 1; }
done
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container $CONTAINER_NAME already runs (docker stop $CONTAINER_NAME)" >&2
  exit 1
fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT" >&2
  exit 1
fi
# GB10 memory before the start (../scripts/mem-prep.sh): page cache dropped, free memory compacted.
mem_line=$("$VINK_ROOT/../scripts/mem-prep.sh" "$([ "$MEM_PREP" = 1 ] && echo prep || echo status)" 2>&1 || true)
if [ -z "$MIN_FREE_GIB" ]; then
  MIN_FREE_GIB=$(awk -v u="$GPU_UTIL" '/^MemTotal:/ { printf "%d", u * $2 / 1048576 + 2 }' /proc/meminfo)
fi
avail_gib=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
  echo "only $avail_gib GiB available (MIN_FREE_GIB=$MIN_FREE_GIB): something else holds memory" >&2
  ps -eo pid,rss,args --sort=-rss | awk 'NR > 1 && NR <= 6 {
    printf "  pid %-8s %6.1f GiB  %s\n", $1, $2 / 1048576, substr($0, index($0, $3), 90) }' >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"/{cache,triton}

envs=(HF_HUB_OFFLINE=1)
# EXTRA_ENV replaces a value above of the same name instead of adding a second
# one: docker passes both, and Python's os.environ keeps the first.
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list, split on purpose
extra_env=($EXTRA_ENV)
for e in "${extra_env[@]}"; do
  for i in "${!envs[@]}"; do
    [ "${envs[$i]%%=*}" = "${e%%=*}" ] && unset 'envs[i]'
  done
  envs+=("$e")
done

# The recipe runs --privileged; the narrower options below are what
# ../vllm-b12x/ runs vLLM with on a Spark.
docker_args=(
  --rm --init --name "$CONTAINER_NAME"
  --gpus all --ipc host --network host --shm-size 16g
  --ulimit memlock=-1 --ulimit stack=67108864
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined
  --stop-timeout 30
  -v "$MODEL_DIR:$MODEL_DIR$([ "$MODEL_RO" = 1 ] && echo :ro)"
  -v "$CACHE_DIR/cache:/root/.cache"
  -v "$CACHE_DIR/triton:/root/.triton"
  --entrypoint vllm
)
[ -t 0 ] && [ -t 1 ] && docker_args+=(-it)
for e in "${envs[@]}"; do docker_args+=(-e "$e"); done

# The recipe's command; --served-model-name, --host/--port and an explicit
# --gpu-memory-utilization added.
args=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
)
if [ "$SPEC_TOKENS" -gt 0 ]; then
  spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_TOKENS"
  [ -n "$DRAFT_SAMPLE_METHOD" ] && spec+=",\"draft_sample_method\":\"$DRAFT_SAMPLE_METHOD\""
  args+=(--speculative-config "$spec}")
fi
[ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$TOOL_CALL_PARSER" ] && args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
[ -n "$MAX_NUM_SEQS" ] && args+=(--max-num-seqs "$MAX_NUM_SEQS")
[ -n "$MAX_BATCHED_TOKENS" ] && args+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")
[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list, split on purpose
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

echo "image   $ref"
echo "model   $MODEL_DIR (revision $(head -1 "$MODEL_DIR/.cache/huggingface/download/config.json.metadata" 2>/dev/null || echo unknown))"
echo "serve   DSpark $([ "$SPEC_TOKENS" -gt 0 ] && echo "$SPEC_TOKENS tokens${DRAFT_SAMPLE_METHOD:+, $DRAFT_SAMPLE_METHOD}" || echo off)," \
  "ctx $MAX_MODEL_LEN, mem $GPU_UTIL, seqs ${MAX_NUM_SEQS:-vLLM default}"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\""
[ -n "$mem_line" ] && echo "${mem_line#\[*\] }"
echo
exec docker run "${docker_args[@]}" "$ref" "${args[@]}"
