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
# No swap for the server: 1 = in a user scope with MemorySwapMax=0, 0 = as is,
# auto = 1 once scripts/host-tuning.sh is applied (its A/B ran that way). The
# systemd unit of scripts/install-service.sh sets MemorySwapMax=0 itself.
NOSWAP="${NOSWAP:-auto}"

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

# Unified memory: the GPU's pool is the machine's RAM. A leftover process
# holding most of it (a server that did not exit, a llama.cpp test, a merge
# script) shows up inside SGLang as "CUDA error: out of memory" before any
# weight loads, even from torch.cuda.mem_get_info. Say who it is instead.
MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
avail_gib=$(awk '/^MemAvailable:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
  echo "only $avail_gib GiB of memory available (MIN_FREE_GIB=$MIN_FREE_GIB): something else holds it." >&2
  { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true; } |
    sed 's/^/  on the GPU: /' >&2
  ps -eo pid,rss,args --sort=-rss | awk 'NR > 1 && NR <= 6 {
    printf "  pid %-8s %6.1f GiB  %s\n", $1, $2 / 1048576, substr($0, index($0, $3), 90) }' >&2
  echo "Stop it (kill <pid>; sudo systemctl stop gb10-sglang for the service), or" \
    "MIN_FREE_GIB=0 to start anyway." >&2
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
# The profile's own environment for the server (e.g. patches/ on PYTHONPATH).
if declare -F model_env >/dev/null; then model_env; fi

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

# A transient scope of the user's systemd with swap off. Probed first: the
# memory controller has to be delegated to the user manager (systemd 252+
# does by default), else the limit would be accepted and do nothing.
scope=() swap_note=""
swap_max_in_scope() {  # memory.swap.max as a user scope with MemorySwapMax=0 sees it
  command -v systemd-run >/dev/null || return 0
  # shellcheck disable=SC2016  # expanded by the sh inside the scope
  systemd-run --user --scope --quiet -p MemorySwapMax=0 \
    sh -c 'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.swap.max"' 2>/dev/null || true
}
[ "$NOSWAP" = auto ] && { [ -f /etc/sysctl.d/99-gb10.conf ] && NOSWAP=1 || NOSWAP=0; }
if [ "$NOSWAP" = 1 ]; then
  if [ -n "${INVOCATION_ID:-}" ]; then
    swap_note="swap per the systemd unit (MemorySwapMax=0 from install-service.sh)"
  elif [ "$(swap_max_in_scope)" = 0 ]; then
    scope=(systemd-run --user --scope --quiet -p MemorySwapMax=0 --)
    swap_note="no swap (user scope, MemorySwapMax=0)"
  else
    swap_note="NOSWAP=1, but a user scope cannot limit swap here (no systemd user manager, or memory not delegated): swap allowed"
  fi
fi

# The boot log does not say which revision a directory holds; these lines do.
echo "profile $PROFILE"
echo "target  $MODEL_DIR (revision $(revision_of "$MODEL_DIR"))"
[ -n "$DRAFT_DIR" ] && echo "draft   $DRAFT_DIR (revision $(revision_of "$DRAFT_DIR"))"
if declare -F model_summary >/dev/null; then echo "model   $(model_summary)"; fi
echo "sglang  $SGLANG_VERSION ($VENV), mem-fraction $MEM_FRACTION"
echo "nvcc    $(command -v nvcc >/dev/null && nvcc --version | grep -oE 'release [0-9.]+' || echo 'not on PATH; set CUDA_HOME if kernel JIT fails')"
echo "listen  http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\"   CPUs ${CPUSET:-all}"
[ -n "$swap_note" ] && echo "memory  $swap_note"
echo

exec "${scope[@]}" "${pin[@]}" "$VENV/bin/python" -m sglang.launch_server "${args[@]}"
