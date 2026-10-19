#!/usr/bin/env bash
# Build the serving tree SERVE_DIR from MODEL_DIR and ENGRAM_DIR on both nodes (repack.py: hard
# links, a union index, the config / plan rewrites; seconds, no copy). Run it once after the weights
# are there, and again after either folder changes. The two source folders are left as they are.
#   ./repack.sh            head and worker
#   ./repack.sh --check    only check both trees
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

args=(--pack "$MODEL_DIR" --engram "$ENGRAM_DIR" --out "$SERVE_DIR")
[ "${1:-}" = --check ] && args=(--check --out "$SERVE_DIR")
for d in "$MODEL_DIR" "$ENGRAM_DIR"; do
  [ "${1:-}" = --check ] || [ -d "$d" ] || {
    echo "no $d here: download it first (../tensorfold-dsv41/README.md, \"Weights\")" >&2; exit 1; }
done

log "head: $SERVE_DIR"
python3 -I "$VD_ROOT/repack.py" "${args[@]}"
log "worker ($WORKER_SSH): $SERVE_DIR"
# The script goes over ssh on stdin, so the worker needs no checkout of this repository.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "python3 -I - $(printf '%q ' "${args[@]}")" < "$VD_ROOT/repack.py"
log "done; ./serve.sh starts both ranks on it"
