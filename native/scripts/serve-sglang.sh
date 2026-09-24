#!/usr/bin/env bash
# Run SGLang natively in the foreground for one model profile.
#
# Normally started through ./serve.sh in the repo root, which sets the machine
# knobs and picks the profile (models/<name>.sh: weights, SGLang version and
# the model's own flags). Direct calls work too; anything unset gets the
# profile's value, then the default below:
#   PROFILE=qwen3.8-27b DRAFT_TOKENS=16 ./scripts/serve-sglang.sh
#
# Serves an OpenAI- and Anthropic-compatible API on :$PORT (default 8888).
# Ctrl-C stops it. The first boot compiles kernels and takes longer than
# later ones, which load them from ~/.cache.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"

# Machine settings (../serve.sh explains each); the profile may set its own.
MEM_FRACTION="${MEM_FRACTION:-0.80}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$PROFILE}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CPUSET="${CPUSET-5-9,15-19}"
API_KEY="${API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
HF_OFFLINE="${HF_OFFLINE:-1}"

[ -x "$VENV/bin/python" ] || {
  echo "no venv for SGLang $SGLANG_VERSION at $VENV - run ./serve.sh $PROFILE install" >&2; exit 1; }
if ! missing="$(check_build_deps "$VENV/bin/python")"; then
  echo "missing, needed at boot:" >&2; echo "$missing" >&2; exit 1
fi
for d in "$MODEL_DIR" ${DRAFT_DIR:+"$DRAFT_DIR"}; do
  [ -f "$d/config.json" ] || {
    echo "no checkpoint at $d (config.json missing) - see models/$PROFILE.sh" >&2; exit 1; }
done
# Any HTTP answer (a 401 from a server with an API key too) means the port is
# taken, e.g. by the service or another profile: one model at a time.
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT (systemctl status gb10-sglang?)" >&2
  exit 1
fi

# The venv's bin/ goes on PATH, as `activate` would do: FlashInfer's JIT runs
# a bare `ninja`, which the venv carries (the Docker image had its venv active).
export PATH="$VENV/bin:$PATH"
# JIT kernels (FlashInfer, TileLang) need nvcc. Prefer the host's CUDA 13.0
# toolkit, as the official image does (CUDA_HOME=/usr/local/cuda there too).
if [ -z "${CUDA_HOME:-}" ] && [ -x /usr/local/cuda/bin/nvcc ]; then
  export CUDA_HOME=/usr/local/cuda
fi
[ -n "${CUDA_HOME:-}" ] && export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONUNBUFFERED=1
# FlashInfer's JIT passes MAX_JOBS to ninja as -j (see JIT_JOBS in serve.sh).
export MAX_JOBS="${JIT_JOBS:-${MAX_JOBS:-2}}"
[ "$HF_OFFLINE" = 1 ] && export HF_HUB_OFFLINE=1

# Flags every model gets; the profile's model_args adds its own.
args=(
  --model-path "$MODEL_DIR"
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --mem-fraction-static "$MEM_FRACTION"
  --sampling-defaults model
  --enable-metrics
  --enable-cache-report
  --host "$HOST"
  --port "$PORT"
)
[ -n "$CONTEXT_LENGTH" ] && args+=(--context-length "$CONTEXT_LENGTH")
if declare -F model_args >/dev/null; then model_args; fi
[ -n "$API_KEY" ] && args+=(--api-key "$API_KEY")
# Appended last: argparse is last-wins, so these override anything above.
read -ra extra <<< "$EXTRA_ARGS"
args+=("${extra[@]}")

pin=()
if [ -n "$CPUSET" ] && command -v taskset >/dev/null; then pin=(taskset -c "$CPUSET"); fi

# The boot log does not say which revision a directory holds; these lines do.
echo "profile $PROFILE"
echo "target  $MODEL_DIR (revision $(revision_of "$MODEL_DIR"))"
[ -n "$DRAFT_DIR" ] && echo "draft   $DRAFT_DIR (revision $(revision_of "$DRAFT_DIR"))"
if declare -F model_summary >/dev/null; then echo "model   $(model_summary)"; fi
echo "sglang  $SGLANG_VERSION ($VENV), mem-fraction $MEM_FRACTION"
echo "nvcc    $(command -v nvcc >/dev/null && nvcc --version | grep -oE 'release [0-9.]+' || echo 'not on PATH; set CUDA_HOME if kernel JIT fails')"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\"   CPUs ${CPUSET:-all}"
echo

exec "${pin[@]}" "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
