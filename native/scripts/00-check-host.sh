#!/usr/bin/env bash
# Check the host before installing anything: driver, GPU, Python, free memory
# and earlyoom. SGLang runs natively here (no Docker), so these are the only
# preconditions.
#
# After ./serve.sh install, run it again: it then also checks that the
# venv's torch sees the GPU.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"
# The venv last installed (a link to venv-sglang-<version>).
VENV="$GB10_WORKDIR/venv"
fail=0

echo "== driver / GPU =="
if ! nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; then
  echo "nvidia-smi failed - fix the driver before continuing"; exit 1
fi
cuda="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | cut -d' ' -f3)"
echo "driver CUDA: ${cuda:-unknown}"
case "$cuda" in
  13.*) ;;
  *) echo "!! SGLang 0.5.20 wheels need a CUDA 13 driver (580.x or newer)"; fail=1 ;;
esac

echo
echo "== Python =="
if command -v python3.12 >/dev/null; then
  python3.12 -c 'import venv, ensurepip' 2>/dev/null \
    && echo "python3.12 with venv: ok" \
    || { echo "!! python3.12 lacks venv/ensurepip: sudo apt install python3.12-venv"; fail=1; }
  if missing="$(check_build_deps python3.12)"; then
    echo "headers and C compiler for Triton: ok"
  else
    echo "!! missing:"; echo "$missing"; fail=1
  fi
else
  echo "!! python3.12 not found (the lock file in requirements/ is for 3.12)"; fail=1
fi
[ "$(uname -m)" = aarch64 ] || echo "note: $(uname -m), not aarch64 - no committed lock; install will resolve one"

echo
echo "== memory =="
# Unified memory: nvidia-smi reports "Not Supported" for memory, so read the
# host's view instead. Anything else holding GPU memory shrinks what SGLang
# can take at a given --mem-fraction-static.
awk '/MemTotal|MemAvailable/ {printf "%-13s %6.1f GB\n", $1, $2/1048576}' /proc/meminfo
if systemctl is-active --quiet earlyoom 2>/dev/null; then
  echo "earlyoom: active - it SIGTERMs the largest process when free memory runs"
  echo "          low (exit code -15, no traceback). See the README trap on 0.80."
fi

if [ -x "$VENV/bin/python" ]; then
  echo
  echo "== venv torch ($VENV) =="
  "$VENV/bin/python" - <<'EOF' || fail=1
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  CUDA {torch.version.cuda}  cuda available: {ok}")
if not ok:
    raise SystemExit("!! torch cannot see the GPU")
print(f"device {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
EOF
fi

echo
[ "$fail" = 0 ] && echo "OK" || { echo "host check found problems (see !! above)"; exit 1; }
