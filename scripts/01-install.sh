#!/usr/bin/env bash
# Install stock SGLang natively (no Docker). The weights are downloaded
# separately, once (README, "Weights").
#
#   ./serve.sh install
#
# Everything goes into one venv, $GB10_WORKDIR/venv (default ~/spark/venv):
# SGLang, the hf CLI, and pandas/pyarrow for HumanEval. Re-running is safe;
# an existing venv is updated in place.
#
# On aarch64 + Python 3.12 (DGX OS) it installs the exact versions in
# requirements/sglang-<ver>-aarch64-py312.txt. Elsewhere it resolves
# sglang==$SGLANG_VERSION fresh, with the same CUDA 13.0 constraint.
#
# ~4 GB of wheels; the venv unpacks to roughly twice that, plus uv's cache.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"

SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
LOCK="$ROOT/requirements/sglang-${SGLANG_VERSION}-aarch64-py312.txt"
CONSTRAINTS="$ROOT/requirements/constraints-cuda130.txt"
PY="${GB10_PYTHON_BASE:-python3.12}"

mkdir -p "$GB10_WORKDIR"

# ---- venv ------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating $VENV ($PY) =="
  "$PY" -m venv "$VENV"
fi
pyver="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
"$VENV/bin/python" -m pip install -q -U pip uv
UV=("$VENV/bin/uv" pip install --python "$VENV/bin/python")

# ---- SGLang ----------------------------------------------------------------
if [ "$(uname -m)" = aarch64 ] && [ "$pyver" = 3.12 ] && [ -f "$LOCK" ]; then
  echo "== installing SGLang $SGLANG_VERSION from ${LOCK#$ROOT/} =="
  "${UV[@]}" -r "$LOCK"
else
  echo "== installing SGLang $SGLANG_VERSION (no lock for $(uname -m) / Python $pyver) =="
  "${UV[@]}" --prerelease=allow -c "$CONSTRAINTS" "sglang==$SGLANG_VERSION"
fi

# Precompiled FlashInfer kernels, as the official image installs them. Not on
# PyPI, so best-effort: without it the kernels are JIT-compiled on first use
# (slower first boot, cached afterwards).
fi_ver="$("$VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')"
if ! "${UV[@]}" --no-deps --index-url https://flashinfer.ai/whl "flashinfer-cubin==$fi_ver"; then
  echo "WARNING: flashinfer-cubin $fi_ver not installed; FlashInfer will JIT-compile on first boot" >&2
fi

# ---- check -----------------------------------------------------------------
echo
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as m
import torch
print("sglang", m.version("sglang"), "| torch", torch.__version__, "CUDA", torch.version.cuda,
      "| flashinfer", m.version("flashinfer-python"))
if not torch.cuda.is_available():
    raise SystemExit("!! torch cannot see the GPU - run ./scripts/00-check-host.sh")
cap = "".join(map(str, torch.cuda.get_device_capability(0)))
print(f"GPU {torch.cuda.get_device_name(0)} (sm_{cap})")
EOF

cat <<EOF

Installed. The hf CLI is $VENV/bin/hf, for the weights (README, "Weights").
Point MODEL_DIR / DRAFT_DIR in serve.sh at them, then start the server:

    ./serve.sh

or as a service: ./scripts/install-service.sh

Record what got installed next to your measurements:

    ./serve.sh manifest > results/BUILD-MANIFEST-native.md
EOF
