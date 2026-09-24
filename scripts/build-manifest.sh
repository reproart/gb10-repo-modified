#!/usr/bin/env bash
# Emit the resolved install. Run it next to a benchmark so a later comparison
# has the actual inputs rather than the intended ones.
#
#   ./scripts/build-manifest.sh > results/BUILD-MANIFEST-native.md
#
# Branches and version ranges move; commits and exact versions do not.
set -uo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/lib/config.sh"
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

snap() {  # resolved snapshot(s) for a cached repo
  local d="$HF_HUB/models--${1//\//--}/snapshots"
  [ -d "$d" ] && ls "$d" 2>/dev/null | paste -sd, || echo "(not cached)"
}

cat <<EOF
# Build manifest (native)

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) on \`$(hostname)\`, venv \`$VENV\`.

| Input | Resolved value |
|---|---|
| Target | \`$TARGET_PATH\` |
| Target snapshots cached | \`$(snap "$TARGET_PATH")\` |
| Draft | \`$DRAFT_PATH\` |
| Draft snapshots cached | \`$(snap "$DRAFT_PATH")\` |
EOF

"$VENV/bin/python" - "$ROOT"/requirements/sglang-*-aarch64-py312.txt <<'EOF' 2>/dev/null || echo "| venv | (missing or broken: $VENV) |"
import importlib.metadata as m
import re
import sys

import torch

def v(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return "(absent)"

for name in ("sglang", "sglang-kernel", "flashinfer-python", "flashinfer-cubin",
             "transformers", "triton", "nvidia-cuda-nvcc"):
    print(f"| {name} | `{v(name)}` |")
print(f"| torch | `{torch.__version__}` (CUDA {torch.version.cuda}) |")

# Compare the venv with the lock file for the installed SGLang version.
lock = [p for p in sys.argv[1:] if f"sglang-{v('sglang')}-" in p]
if lock:
    pins = dict(re.match(r"([A-Za-z0-9_.-]+)==(\S+)", line).groups()
                for line in open(lock[0]) if re.match(r"[A-Za-z0-9_.-]+==", line))
    off = [f"{n} {v(n)} (lock {want})" for n, want in pins.items() if v(n) != want]
    name = lock[0].rsplit("/", 1)[-1]
    print(f"| matches `{name}` | " + ("yes |" if not off else f"no: {'; '.join(off)} |"))
else:
    print("| lock file | none for this SGLang version |")
EOF

pid="$(pgrep -f -o 'sglang.launch_server' || true)"
cat <<EOF

## Host

| | |
|---|---|
| Kernel | \`$(uname -sr)\` |
| Driver / CUDA | $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) / $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | cut -d' ' -f3) |
| System nvcc | $( { /usr/local/cuda/bin/nvcc --version 2>/dev/null || echo 'none'; } | grep -oE 'release [0-9.]+|none' | head -1) |
| Python | $("$VENV/bin/python" -V 2>&1) |

## Serving flags

\`\`\`
$([ -n "$pid" ] && tr '\0' ' ' < "/proc/$pid/cmdline" || echo "(no server running)")
\`\`\`
EOF
