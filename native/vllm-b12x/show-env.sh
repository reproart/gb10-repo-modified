#!/usr/bin/env bash
# Show the environment variables the container's vLLM defines (vllm/envs.py),
# filtered by a pattern, with their defaults: what PLE_OFFLOAD=none leaves to
# the container, and what else EXTRA_ENV could set.
#
#   ./show-env.sh            PLE|ENGRAM (the n-gram table settings; vLLM logs
#                            them as "Resolved Engram configuration")
#   ./show-env.sh B12X       any other pattern (grep -E, case-insensitive); also lists
#                            the B12X_* variables the b12x library reads
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
docker image inspect "$ref" >/dev/null 2>&1 || {
  echo "container $ref not here: run $B12X_ROOT/pull.sh first" >&2; exit 1; }
pattern="${1:-PLE|ENGRAM}"
# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --entrypoint bash -e PATTERN="$pattern" "$ref" -c '
  d=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))") &&
  echo "== $d/envs.py"; grep -n -i -E -A3 "$PATTERN" "$d/envs.py" || echo "(no match)"
  echo "== where vLLM resolves them (config/)"
  grep -rn -i -E -B2 -A6 "table_memory|EngramConfig\(" "$d/config" | head -80
  b=$(python3 -c "import os, b12x; print(os.path.dirname(b12x.__file__))" 2>/dev/null) &&
  { echo "== B12X_* variables the b12x library reads ($b), matching $PATTERN"
    grep -rhoE "B12X_[A-Z0-9_]+" "$b" | sort -u | grep -i -E "$PATTERN" || echo "(no match)"; }'
