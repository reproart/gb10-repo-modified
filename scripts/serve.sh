#!/usr/bin/env bash
# Run SGLang natively in the foreground: target + DFlash2 draft, tuned flags.
#
#   ./scripts/serve.sh                          # defaults below
#   DRAFT_TOKENS=16 MAX_RUNNING=16 ./scripts/serve.sh   # single-stream tuning
#   TARGET=nvfp4 ./scripts/serve.sh
#
# Settings come from the environment, then serve.env in the repo root (copy
# serve.env.sample), then the defaults here. The systemd unit from
# install-service.sh runs this same script.
#
# Serves an OpenAI- and Anthropic-compatible API on :$PORT (default 8888).
# Ctrl-C stops it. First boot takes a few minutes longer than later ones:
# kernels are JIT-compiled and cached under ~/.cache/flashinfer and ~/.triton.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"

# Defaults: the max-aggregate config from results/RESULTS.md ("FP8 target"):
# draft 10, 32 concurrent requests. For interactive single-stream use,
# DRAFT_TOKENS=16 MAX_RUNNING=16 (README, "Tune draft tokens").
DRAFT_TOKENS="${DRAFT_TOKENS:-10}"
MAX_RUNNING="${MAX_RUNNING:-32}"
# The GDN state pool is what bounds concurrency: 5 slots per running request
# under --mamba-radix-cache-strategy extra_buffer (4, plus one for the DFlash2
# verify). SGLang clamps max_running_requests to pool / 5.
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 5))}"
# Without this, batches above the captured size fall back to eager decode.
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-$MAX_RUNNING}"
# 0.80, not 0.85: memory is unified, and at 0.85 DGX OS's earlyoom can kill the
# scheduler during boot or a long prefill (SGLang cookbook, DGX Spark notes).
MEM_FRACTION="${MEM_FRACTION:-0.80}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
PREFILL_CUDA_GRAPH="${PREFILL_CUDA_GRAPH:-0}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# GB10's Cortex-X5 cores (the A725 efficiency cores are 0-4, 10-14): keeps the
# scheduler and tokenizer off the slow cores. Set it empty to disable.
CPUSET="${CPUSET-5-9,15-19}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV - run ./scripts/01-install.sh" >&2; exit 1; }
# Any HTTP answer (a 401 from a server with an API key too) means the port is taken.
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT (systemctl status gb10-sglang?)" >&2
  exit 1
fi

# JIT kernels (FlashInfer, TileLang) need nvcc. Prefer the host's CUDA 13.0
# toolkit, as the official image does (CUDA_HOME=/usr/local/cuda there too).
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1

args=(
  --model-path "$TARGET_PATH" ${TARGET_REV:+--revision "$TARGET_REV"}
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT_PATH" ${DRAFT_REV:+--speculative-draft-model-revision "$DRAFT_REV"}
  --speculative-num-draft-tokens "$DRAFT_TOKENS"
  --mamba-radix-cache-strategy extra_buffer
  --mamba-ssm-dtype bfloat16
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static "$MEM_FRACTION"
  --max-mamba-cache-size "$MAMBA_CACHE"
  --max-running-requests "$MAX_RUNNING"
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
  --attention-backend flashinfer
  --chunked-prefill-size "$CHUNKED_PREFILL"
  --context-length "$CONTEXT_LENGTH"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --sampling-defaults model
  --enable-metrics
  --enable-cache-report
  --host "$HOST"
  --port "$PORT"
)
[ "$PREFILL_CUDA_GRAPH" = 1 ] || args+=(--disable-prefill-cuda-graph)
[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
# Appended last: argparse is last-wins, so these override anything above.
read -ra extra <<< "$EXTRA_ARGS"
args+=("${extra[@]}")

pin=()
if [ -n "$CPUSET" ] && command -v taskset >/dev/null; then pin=(taskset -c "$CPUSET"); fi

echo "target  $TARGET_PATH${TARGET_REV:+ @ ${TARGET_REV:0:8}}"
echo "draft   $DRAFT_PATH${DRAFT_REV:+ @ ${DRAFT_REV:0:8}}, $DRAFT_TOKENS draft tokens"
echo "cap     $MAX_RUNNING requests (GDN pool $MAMBA_CACHE), mem-fraction $MEM_FRACTION"
echo "nvcc    $(command -v nvcc >/dev/null && nvcc --version | grep -oE 'release [0-9.]+' || echo 'not on PATH; set CUDA_HOME if kernel JIT fails')"
echo "listen  http://$HOST:$PORT/v1   CPUs ${CPUSET:-all}"
echo

exec "${pin[@]}" "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
