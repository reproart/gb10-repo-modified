#!/usr/bin/env bash
# Show the environment variables the container's vLLM defines (vllm/envs.py),
# filtered by a pattern, with their defaults: what PLE_OFFLOAD=none leaves to
# the container, and what else EXTRA_ENV could set.
#
#   ./show-env.sh            PLE (the n-gram table settings)
#   ./show-env.sh B12X       any other pattern (grep -E, case-insensitive)
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

need_docker
ref="$(image_ref)"
docker image inspect "$ref" >/dev/null 2>&1 || {
  echo "container $ref not here: run $B12X_ROOT/pull.sh first" >&2; exit 1; }
pattern="${1:-PLE}"
# shellcheck disable=SC2016  # expanded inside the container
docker run --rm --entrypoint bash -e PATTERN="$pattern" "$ref" -c '
  d=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))") &&
  grep -n -i -E -A3 "$PATTERN" "$d/envs.py" || echo "no match for $PATTERN in $d/envs.py"'
