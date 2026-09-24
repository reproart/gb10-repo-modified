#!/usr/bin/env bash
# Install stock SGLang natively (no Docker). The weights are downloaded
# separately, once (README, "Weights").
#
#   ./serve.sh install                # the SGLang version of the default profile
#   ./serve.sh gemma4-31b install     # ... of another profile
#
# SGLANG_VERSION (set in the profile, models/<name>.sh) picks the version;
# each one gets its own venv, $GB10_WORKDIR/venv-sglang-<version>, and
# $GB10_WORKDIR/venv is pointed at the one installed last. The venv also carries the hf CLI. Re-running is
# safe: an existing venv is brought in line with the lock.
#
# Every install goes through a lock, requirements/sglang-<version>-<arch>-
# py<python>.txt. If there is none for this version and machine yet, one is
# resolved first (sglang==<version>, CUDA held at 13.0) and written there:
# commit it once that version has served and benchmarked well. SGLANG_INDEX
# adds a package index, e.g. SGLang's nightly wheels.
#
# ~4 GB of wheels per version; each venv unpacks to roughly twice that.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"

# Empty SGLANG_CONSTRAINTS drops the CUDA 13.0 hold, for a newer driver.
# Relative to the project root, which is where the lock is resolved.
CONSTRAINTS="${SGLANG_CONSTRAINTS-requirements/constraints-cuda130.txt}"
PY="${GB10_PYTHON_BASE:-python3.12}"

[[ "$SGLANG_VERSION" =~ ^[0-9]+(\.[0-9]+)*([a-z0-9.+-]*)$ ]] || {
  echo "SGLANG_VERSION must be an exact version (e.g. 0.5.20), got '$SGLANG_VERSION'." >&2
  echo "Releases: https://pypi.org/project/sglang/#history" >&2
  exit 2; }

mkdir -p "$GB10_WORKDIR"

# ---- venv ------------------------------------------------------------------
new_venv=0
if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating $VENV ($PY) =="
  "$PY" -m venv "$VENV"
  new_venv=1
fi
pyver="$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
if ! missing="$(check_build_deps "$VENV/bin/python")"; then
  echo "The server will not start without these (install them, then re-run):" >&2
  echo "$missing" >&2
  exit 1
fi
"$VENV/bin/python" -m pip install -q -U pip uv
UV=("$VENV/bin/uv" pip install --python "$VENV/bin/python")
index=() cons=()
[ -n "$SGLANG_INDEX" ] && index=(--extra-index-url "$SGLANG_INDEX" --index-strategy unsafe-best-match)
[ -n "$CONSTRAINTS" ] && cons=(-c "$CONSTRAINTS")

# ---- lock ------------------------------------------------------------------
LOCK="$ROOT/requirements/sglang-${SGLANG_VERSION}-$(uname -m)-py${pyver}.txt"
new_lock=0
if [ ! -f "$LOCK" ]; then
  echo "== no ${LOCK#$ROOT/} yet: resolving SGLang $SGLANG_VERSION =="
  # From the project root with relative paths, so the lock's annotations
  # carry no machine-specific paths.
  if ! ( cd "$ROOT" && echo "sglang==$SGLANG_VERSION" | "$VENV/bin/uv" pip compile - \
      "${cons[@]}" "${index[@]}" --prerelease=allow --python "$VENV/bin/python" \
      --annotation-style line \
      --custom-compile-command "SGLANG_VERSION=$SGLANG_VERSION${SGLANG_INDEX:+ SGLANG_INDEX=$SGLANG_INDEX} ./serve.sh install" \
      -o "${LOCK#$ROOT/}" ); then
    rm -f "$LOCK"
    [ "$new_venv" = 1 ] && rm -rf "$VENV"   # nothing to keep for a version that doesn't resolve
    exit 1
  fi
  new_lock=1
fi

# ---- SGLang ----------------------------------------------------------------
echo "== installing SGLang $SGLANG_VERSION from ${LOCK#$ROOT/} =="
if ! "${UV[@]}" "${index[@]}" -r "$LOCK"; then
  # A lock that does not install is not worth keeping.
  [ "$new_lock" = 1 ] && rm -f "$LOCK"
  exit 1
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

# ---- $GB10_WORKDIR/venv -> this version ------------------------------------
link="$GB10_WORKDIR/venv"
if [ -L "$link" ] || [ ! -e "$link" ]; then
  ln -sfn "$(basename "$VENV")" "$link"
else
  echo "note: $link is a directory from an earlier install, not a link to this" >&2
  echo "      version's venv; remove it and re-run to have it follow SGLANG_VERSION." >&2
fi
if [ "$new_lock" = 1 ]; then
  echo
  echo "New lock: ${LOCK#$ROOT/}. Commit it once SGLang $SGLANG_VERSION has served and"
  echo "benchmarked well; until then, deleting it re-resolves on the next install."
fi

cat <<EOF

Installed. The hf CLI is $link/bin/hf, for the weights (README, "Weights").
Point MODEL_DIR / DRAFT_DIR in serve.sh at them, then start the server:

    ./serve.sh

or as a service: ./scripts/install-service.sh

Record what got installed next to your measurements:

    ./serve.sh manifest > results/BUILD-MANIFEST-native.md
EOF
