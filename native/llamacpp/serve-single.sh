#!/usr/bin/env bash
# GLM-5.3-Flash GSQ-RCO on ONE Spark: a proof that the build and the GGUF
# work, not a setup to live with. Only the 3.0-bit file can fit:
#
#   3.0-bit  117.48 GB = 109.4 GiB   against ~115 GiB free on a booted Spark
#   3.5-bit  137.07 GB = 127.7 GiB   more than the machine has (serve-dual.sh)
#
# So: text only (no vision projector), one request at a time, a short
# context, no host-side prompt cache (--cache-ram 0: on the Spark the "host"
# RAM is the same memory), no mmap (--load-mode none; with mmap the file
# would sit in the page cache next to its copy in CUDA buffers while
# loading). What is left for the KV cache, compute buffers and the OS is a
# few GiB: if earlyoom kills the server (journalctl -u earlyoom), this model
# does not fit one Spark, and that is the answer.
#
#   ./serve-single.sh
#   CTX=4096 ./serve-single.sh            less KV
#   MMPROJ=1 ./serve-single.sh            + the vision projector (1.08 GiB more)
#
# Then, from the same machine:
#   curl -s localhost:8889/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"messages":[{"role":"user","content":"19*23?"}],"max_tokens":64}'
#   GB10_BASE_URL=http://127.0.0.1:8889/v1 python3 ../bench/perf.py --only decode
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

QUANT="${QUANT:-3.0bit}"
CTX="${CTX:-8192}"
MMPROJ="${MMPROJ:-0}"
FA="${FA:-off}"              # the card's setting
# GiB that must stay free for the KV cache, compute buffers and the OS.
HEADROOM_GIB="${HEADROOM_GIB:-6}"
FORCE="${FORCE:-0}"

need_build llama-server
MODEL="$(gguf_for "$QUANT")"
need_file "$MODEL"
[ "$MMPROJ" = 1 ] && need_file "$MMPROJ_FILE"
no_sglang

# The memory budget, before loading 110 GiB into it.
weights=$(stat -c %s "$MODEL")
[ "$MMPROJ" = 1 ] && weights=$((weights + $(stat -c %s "$MMPROJ_FILE")))
avail=$(( $(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo) * 1024 ))
left=$((avail - weights))
echo "weights $(gib "$weights") GiB, available $(gib "$avail") GiB -> $(gib "$left") GiB left" \
  "for KV ($CTX tokens), buffers and the OS"
if [ "$left" -lt $((HEADROOM_GIB * 1073741824)) ]; then
  if [ "$FORCE" != 1 ]; then
    echo "less than HEADROOM_GIB=$HEADROOM_GIB GiB would be left; refusing." >&2
    [ "$QUANT" != 3.0bit ] && echo "QUANT=$QUANT does not fit one Spark: use serve-dual.sh." >&2
    echo "Free memory (stop other GPU work; the desktop: sudo systemctl isolate" \
      "multi-user.target) or FORCE=1 to try anyway." >&2
    exit 1
  fi
  echo "FORCE=1: trying anyway; watch journalctl -u earlyoom" >&2
fi

strict_f32_env
args=(-m "$MODEL")
common_server_args
args+=(-c "$CTX" -np 1 --cache-ram 0 -fa "$FA")
[ "$MMPROJ" = 1 ] && args+=(--mmproj "$MMPROJ_FILE")
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list, split on purpose
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

echo "GLM-5.3-Flash $QUANT on one Spark: ctx $CTX, 1 slot, vision $([ "$MMPROJ" = 1 ] && echo on || echo off)," \
  "strict F32 $STRICT_F32 -> http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\""
exec "$LLAMA_BIN/llama-server" "${args[@]}"
