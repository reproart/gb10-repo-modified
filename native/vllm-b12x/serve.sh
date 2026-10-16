#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-FP4-Local (NVIDIA's QAD NVFP4 build of
# Qwen3.8-Flash-Next) on one DGX Spark with vLLM in the b12x container, in
# the foreground (Ctrl-C stops it). The card's DGX Spark command, with its
# values as knobs (config.sh):
#
#   ./pull.sh                          once: the container, pinned
#   ./serve.sh
#   MAX_NUM_SEQS=4 MTP_TOKENS=3 ./serve.sh
#   EXTRA_ARGS='--max-cudagraph-capture-size 16' ./serve.sh
#
# Then, as for the SGLang server (same port, bench/ unchanged):
#   python3 ../bench/perf.py
#   python3 ../bench/answers.py collect questions.jsonl --out runs/fp4-local.jsonl
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
if ! docker image inspect "$ref" >/dev/null 2>&1; then
  echo "container $ref not here: run $B12X_ROOT/pull.sh first" >&2
  exit 1
fi
[ -n "$IMAGE_DIGEST" ] || echo "warning: no image.lock, running whatever $IMAGE:latest is locally (./pull.sh pins it)" >&2
[ -f "$MODEL_DIR/config.json" ] || {
  echo "no checkpoint at $MODEL_DIR (config.json missing); download: see MODEL_DIR in $B12X_ROOT/config.sh" >&2
  exit 1; }

# One server at a time: the SGLang service or a llama.cpp server would hold
# the memory, and any HTTP answer on the port means it is taken.
if pgrep -f "sglang.launch_server" >/dev/null 2>&1; then
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container $CONTAINER_NAME already runs (docker stop $CONTAINER_NAME)" >&2
  exit 1
fi
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT" >&2
  exit 1
fi
avail_gib=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
  echo "only $avail_gib GiB available (MIN_FREE_GIB=$MIN_FREE_GIB): something else holds memory" >&2
  ps -eo pid,rss,args --sort=-rss | awk 'NR > 1 && NR <= 6 {
    printf "  pid %-8s %6.1f GiB  %s\n", $1, $2 / 1048576, substr($0, index($0, $3), 90) }' >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"/{vllm,flashinfer,triton,b12x}

# The card's environment for DGX Spark.
envs=(
  VLLM_PLE_TABLE_MEMORY="$PLE_MEMORY"
  B12X_DISK_BACKEND=io_uring
  CUTE_DSL_ARCH=sm_121a
  SAFETENSORS_FAST_GPU=1
  VLLM_WORKER_MULTIPROC_METHOD=spawn
  VLLM_SSM_CONV_STATE_LAYOUT=DS
  VLLM_USE_AOT_COMPILE=1
  VLLM_USE_MEGA_AOT_ARTIFACT=1
  VLLM_USE_V2_MODEL_RUNNER=1
  B12X_POLICY_MODE=auto
  VLLM_MXFP8_LM_HEAD=1
  VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13
  VLLM_B12X_MOE_FP4_FORCE_A16=1
  B12X_W4A16_A4_PREFILL=1
  HF_HUB_OFFLINE=1
)
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list, split on purpose
[ -n "$EXTRA_ENV" ] && envs+=($EXTRA_ENV)

docker_args=(
  --rm --init --name "$CONTAINER_NAME"
  --gpus all --ipc host --network host --shm-size 16g
  --ulimit memlock=-1 --ulimit stack=67108864
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined
  --stop-timeout 30
  -v "$MODEL_DIR:$MODEL_DIR$([ "$MODEL_RO" = 1 ] && echo :ro)"
  -v "$CACHE_DIR/vllm:/root/.cache/vllm"
  -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer"
  -v "$CACHE_DIR/triton:/root/.triton"
  -v "$CACHE_DIR/b12x:/root/.cache/b12x"
  --entrypoint vllm
)
[ -t 0 ] && [ -t 1 ] && docker_args+=(-it)
for e in "${envs[@]}"; do docker_args+=(-e "$e"); done

args=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size 1 --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --quantization modelopt_mixed
  --block-size 16
  --load-format b12x
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
  --gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}'
  --gpu-memory-utilization "$GPU_UTIL"
)
[ "$MTP_TOKENS" -gt 0 ] && args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_TOKENS}")
[ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER" --enable-auto-tool-choice)
[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list, split on purpose
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

echo "image   $ref"
echo "model   $MODEL_DIR (revision $(head -1 "$MODEL_DIR/.cache/huggingface/download/config.json.metadata" 2>/dev/null || echo unknown))"
echo "serve   ctx $MAX_MODEL_LEN, $MAX_NUM_SEQS seqs, chunk $MAX_BATCHED_TOKENS, mem $GPU_UTIL, KV $KV_CACHE_DTYPE," \
  "MTP $([ "$MTP_TOKENS" -gt 0 ] && echo "$MTP_TOKENS tokens" || echo off), PLE table $PLE_MEMORY"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\""
echo
exec docker run "${docker_args[@]}" "$ref" "${args[@]}"
