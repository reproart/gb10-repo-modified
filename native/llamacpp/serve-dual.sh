#!/usr/bin/env bash
# GLM-5.3-Flash GSQ-RCO across TWO Sparks: llama-server here (the head, with
# the GGUFs), the other Spark's GPU through llama.cpp RPC (rpc-worker.sh there,
# started first). The layers are split between the two GPUs by free memory
# (--split-mode layer), so a token goes through the head's half, crosses the
# link once as a hidden-state vector (kilobytes), and comes back after the
# worker's half.
#
#   WORKER=10.10.10.2 ./serve-dual.sh                 3.5-bit + vision, 32K, 4 slots
#   WORKER=10.10.10.2 QUANT=3.0bit ./serve-dual.sh    the smaller file
#   WORKER=10.10.10.2 TENSOR_SPLIT=55,45 ./serve-dual.sh   head share first
#
# What two Sparks buy is memory, not speed: ~230 GiB for the 3.5-bit file
# (127.7 GiB), the vision projector, a long context and a host-side prompt
# cache. One token still reads every active weight once, half on each
# machine, one after the other, plus two hops over the link; decode is about
# what one Spark would do if the model fit, minus the hops. Prefill splits
# into micro-batches that the two halves can work on at the same time.
#
# The first start sends the worker's share of the weights over the link
# (~65 GB: about a minute at 100 Gb/s, longer over TCP in practice); with the
# worker's RPC cache on, later starts send only hashes.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

WORKER="${WORKER:?set WORKER to the address of the other Spark on the direct link, e.g. WORKER=10.10.10.2}"
RPC_PORT="${RPC_PORT:-50052}"
QUANT="${QUANT:-3.5bit}"
CTX="${CTX:-32768}"
NP="${NP:-4}"
MMPROJ="${MMPROJ:-1}"
FA="${FA:-off}"                 # the card's setting
# Host-side prompt cache (MiB), for agent loops that resend a long prefix.
CACHE_RAM="${CACHE_RAM:-8192}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"
# layer: each Spark holds whole layers, one after the other (the default, as
# measured by the card's author on three GPUs). tensor: every layer split
# across both GPUs, which then read their halves at the same time: the one
# mode that could make decode faster than one Spark, at the price of two
# exchanges per layer over the link. EXPERIMENTAL in llama.cpp and untested
# over RPC: an experiment, compare with perf.py.
SPLIT_MODE="${SPLIT_MODE:-layer}"

need_build llama-server
MODEL="$(gguf_for "$QUANT")"
need_file "$MODEL"
[ "$MMPROJ" = 1 ] && need_file "$MMPROJ_FILE"
no_sglang

# Is the worker there, and how far away?
if ! timeout 3 bash -c "exec 3<>/dev/tcp/$WORKER/$RPC_PORT" 2>/dev/null; then
  echo "no RPC server at $WORKER:$RPC_PORT: start rpc-worker.sh on the other Spark" \
    "(BIND=$WORKER), and check the link (ping $WORKER)" >&2
  exit 1
fi
# Informational only: neither may stop the start (ICMP can be filtered).
rtt=$(ping -c 5 -i 0.2 -q "$WORKER" 2>/dev/null | awk -F/ '/^rtt|^round-trip/ { print $5 }' || true)
dev=$(ip -o route get "$WORKER" 2>/dev/null | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") print $(i + 1) }' || true)
echo "worker $WORKER:$RPC_PORT via ${dev:-?} (MTU $(cat "/sys/class/net/$dev/mtu" 2>/dev/null || echo "?")," \
  "$(cat "/sys/class/net/$dev/speed" 2>/dev/null || echo "?") Mb/s), ping ${rtt:-?} ms"

strict_f32_env
args=(-m "$MODEL" --rpc "$WORKER:$RPC_PORT")
common_server_args
args+=(-c "$CTX" -np "$NP" --kv-unified --cache-ram "$CACHE_RAM" -fa "$FA" --split-mode "$SPLIT_MODE")
[ -n "$TENSOR_SPLIT" ] && args+=(--tensor-split "$TENSOR_SPLIT")
[ "$MMPROJ" = 1 ] && args+=(--mmproj "$MMPROJ_FILE")
# shellcheck disable=SC2206  # EXTRA_ARGS is a flag list, split on purpose
[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

echo "GLM-5.3-Flash $QUANT on two Sparks: ctx $CTX shared by $NP slots, vision" \
  "$([ "$MMPROJ" = 1 ] && echo on || echo off), split $SPLIT_MODE ${TENSOR_SPLIT:-by free memory}," \
  "strict F32 $STRICT_F32 -> http://$HOST:$PORT/v1 as \"$SERVED_MODEL_NAME\""
exec "$LLAMA_BIN/llama-server" "${args[@]}"
