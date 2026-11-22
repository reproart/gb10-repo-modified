#!/usr/bin/env bash
# One-time: the lean n-gram table vLLM 0.34 reads from disk, then the checkpoint view.
# TABLE_SRC_DIR (../vllm-flash-ar's table: Qwen's FP8 shards 5-37 whole) -> TABLE_DIR (only
# the ngram_embedding tensors, ~48 GB, same file names), with the built image's Python
# (../vllm-flash-ar/tools/strip_ple_table.py). ../vllm-flash-ar reads the rows by name, so it
# can serve from the lean copy too (TABLE_DIR=/models/ple-table-fp8-lean there); once it
# does, the original can go.
#   ./prepare.sh
set -euo pipefail
trap 'echo "prepare.sh: stopped at line $LINENO (exit $?)" >&2' ERR
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE: ./build.sh first" >&2; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "no $MODEL_DIR: download the weights with ../vllm-flash-ar/download.sh" >&2; exit 1; }

n_lean=0
[ -d "$TABLE_DIR" ] && n_lean=$(find "$TABLE_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
if [ "$n_lean" -ge 33 ]; then
  log "lean table: $TABLE_DIR has $n_lean shards, kept"
else
  [ -d "$TABLE_SRC_DIR" ] || { echo "no $TABLE_SRC_DIR: ../vllm-flash-ar/download.sh fetches it" >&2; exit 1; }
  mkdir -p "$TABLE_DIR"
  need=$(( 50 * 1024 * 1024 ))   # KiB, ~48 GB of rows and some room
  free=$(df -Pk "$TABLE_DIR" | awk 'NR == 2 { print $4 }')
  if [ "$free" -lt "$need" ]; then
    echo "only $(( free / 1024 / 1024 )) GiB free on $(df -P "$TABLE_DIR" | awk 'NR == 2 { print $6 }'): the lean table needs ~48 GB" >&2
    exit 1
  fi
  # Copying 48 GB through the page cache: never beside a GPU job.
  "$NATIVE_DIR/scripts/mem-prep.sh" idle || exit 1
  log "stripping $TABLE_SRC_DIR -> $TABLE_DIR (~48 GB; minutes)"
  docker run --rm --entrypoint python3 --user "$(id -u):$(id -g)" \
    -v "$TABLE_SRC_DIR:/src:ro" -v "$TABLE_DIR:/dst" \
    -v "$NATIVE_DIR/vllm-flash-ar/tools/strip_ple_table.py:/strip_ple_table.py:ro" \
    "$IMAGE" /strip_ple_table.py /src /dst
fi
python3 "$V034_ROOT/src/make_model_view.py" "$MODEL_DIR" "$TABLE_DIR" "$VIEW_DIR"
log "ready: ./serve.sh"
