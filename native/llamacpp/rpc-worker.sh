#!/usr/bin/env bash
# The second Spark of serve-dual.sh: llama.cpp's RPC server, which offers this
# machine's GPU to llama-server on the first one. Start it first.
#
#   BIND=10.10.10.2 ./rpc-worker.sh
#
# BIND is this machine's address on the direct cable between the two Sparks
# (README.md here, "Two Sparks"). The RPC server has no authentication: anyone
# who reaches the port can allocate memory and run graphs on this GPU, so it
# listens on the point-to-point link only; it refuses 0.0.0.0 unless
# ALLOW_ANY_ADDRESS=1.
#
# The worker needs the same build as the head (./build.sh here), not the
# GGUF: llama-server sends this machine's share of the weights over the
# link at startup. With RPC_CACHE=1 (default) they are also kept in
# ~/.cache/llama.cpp/rpc (~65 GB for the 3.5-bit file's half), so the next
# start sends only hashes; rm -rf that directory to get the disk back.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

BIND="${BIND:?set BIND to the address of this Spark on the link to the other one, e.g. BIND=10.10.10.2}"
RPC_PORT="${RPC_PORT:-50052}"
RPC_CACHE="${RPC_CACHE:-1}"

need_build ggml-rpc-server
no_sglang

if [ "$BIND" = 0.0.0.0 ] || [ "$BIND" = "::" ]; then
  if [ "${ALLOW_ANY_ADDRESS:-0}" != 1 ]; then
    echo "BIND=$BIND would expose an unauthenticated RPC server on every network;" \
      "use the direct link's address (ALLOW_ANY_ADDRESS=1 to override)" >&2
    exit 1
  fi
elif ! command -v ip >/dev/null; then
  echo "ip (iproute2) not found: cannot check that BIND=$BIND is on this machine" >&2
elif ! ip -o addr show | awk '{ print $4 }' | cut -d/ -f1 | grep -qx "$BIND"; then
  echo "BIND=$BIND is not an address of this machine:" >&2
  ip -o -4 addr show | awk '{ print "  " $2 "  " $4 }' >&2
  exit 1
else
  dev=$(ip -o addr show | awk -v a="$BIND" '{ split($4, x, "/"); if (x[1] == a) print $2 }' | head -n 1)
  echo "listening on $dev ($BIND), MTU $(cat "/sys/class/net/$dev/mtu" 2>/dev/null || echo "?")," \
    "link $(cat "/sys/class/net/$dev/speed" 2>/dev/null || echo "?") Mb/s"
fi

# The kernels run here, so the strict-F32 switch has to be set here too.
strict_f32_env
args=(-H "$BIND" -p "$RPC_PORT")
[ "$RPC_CACHE" = 1 ] && args+=(-c)
echo "ggml-rpc-server on $BIND:$RPC_PORT, cache $([ "$RPC_CACHE" = 1 ] && echo on || echo off), strict F32 $STRICT_F32"
exec "$LLAMA_BIN/ggml-rpc-server" "${args[@]}"
