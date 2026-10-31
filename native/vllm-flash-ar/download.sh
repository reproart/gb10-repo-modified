#!/usr/bin/env bash
# Download the prepared checkpoint and the fp8 n-gram table at the pinned
# revisions, as plain `hf download --local-dir` folders (easy to copy to another
# machine): ~67 + ~49 GiB, resumable. Or build them from Intel's release with
# ./prepare.sh (README, "Modify the weights yourself").
#   ./download.sh            both
#   ./download.sh model      or table
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
HF="${HF:-$(command -v hf || echo "$HOME/spark/venv/bin/hf")}"
[ -x "$HF" ] || command -v "$HF" >/dev/null || { echo "no hf CLI (pip install -U huggingface_hub, or HF=<path>)" >&2; exit 1; }
# ~116 GiB through the page cache: never beside a GPU job (a Spark has hung so).
"$NATIVE_DIR/scripts/mem-prep.sh" idle || exit 1
what=${1:-all}
case "$what" in all|model|table) ;; *) echo "usage: $0 [all|model|table]" >&2; exit 2 ;; esac
if [ "$what" != table ]; then
  log "$MODEL_REPO @ ${MODEL_REVISION:0:12} -> $MODEL_DIR"
  "$HF" download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
fi
if [ "$what" != model ]; then
  log "$TABLE_REPO @ ${TABLE_REVISION:0:12} -> $TABLE_DIR"
  "$HF" download "$TABLE_REPO" --revision "$TABLE_REVISION" --local-dir "$TABLE_DIR"
fi
log "done; ./serve.sh"
