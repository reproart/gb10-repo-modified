# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for GLM-5.3-Flash (EXL3 4 bpw routed experts, BF16 elsewhere,
# 1M context) on TensorFold 0.6.0, TP=2 across two DGX Spark: MiaAI-Lab's
# recipe (github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold,
# Apache-2.0), run from a checkout pinned at KIT_COMMIT. Her scripts do the
# work (image, download, copy to the worker, start, stop); this directory gives
# them this project's pair, paths and defaults, and the one-server-at-a-time
# checks. Source it, don't run it. Every value is "${VAR:-default}":
#   PARALLEL=8 ./serve.sh restart
#
# The pair's own values (CX7 addresses, ssh) are shared with ../tensorfold-dsv41:
# its config.local.sh is read, then ours (config.local.sh next to this file, not
# committed) for anything that differs. The shell wins over both.

GLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "$GLM_ROOT/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$GLM_ROOT/config.local.sh" ] && . "$GLM_ROOT/config.local.sh"
# shellcheck disable=SC1091
[ -f "$NATIVE_DIR/tensorfold-dsv41/config.local.sh" ] && . "$NATIVE_DIR/tensorfold-dsv41/config.local.sh"

# --- The two nodes: the head (this machine, rank 0, serves the API) and the worker ---
HEAD_IP="${HEAD_IP:-10.100.80.1}"
WORKER_IP="${WORKER_IP:-10.100.80.2}"
WORKER_SSH="${WORKER_SSH:-$WORKER_IP}"      # key-based ssh from the head

# --- Mia's recipe: where its checkout lives, and the commit it is pinned at ---
# bf5f945 (2026-10-25): 123 patches on TensorFold 0.6.0, the published image
# v0.6.0-789e8f3bdfa2 pinned by digest in its scripts/config.sh, up to 8 requests
# at once (patch 0069) and <|assistant|> as an end token (0075): the two changes
# LimeChain ran in production on the older 4bbf2f3.
KIT_REPO="${KIT_REPO:-https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git}"
KIT_COMMIT="${KIT_COMMIT:-bf5f94523a8251029cedae86bf61f8ce37fac218}"
KIT_DIR="${KIT_DIR:-$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold}"

# --- Weights: a Hugging Face cache folder, the same path on both nodes ---
# Mia's prepare.sh downloads the checkpoint (~164 GiB) and DFlash2 (~2.3 GB) into
# it at their pinned revisions and rsyncs them to the worker. Copying it to
# another machine works the same way: rsync -a HF_CACHE/hub/models--Mia-AiLab--*
# (README, "Weights"). ~205 GB a node with the image.
HF_CACHE="${HF_CACHE:-/models/huggingface}"

# --- The API on the head: http://0.0.0.0:8888/v1, as every server here ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"

# --- Serving ---
# Requests at once: Mia's default 4. 8 is LimeChain's production setting
# (short-prompt aggregate 300 -> 528 tok/s, per stream 76 -> 66) at ~7 GiB of
# MemAvailable left on the head at the worst moment; the recipe grows its
# memory reserve by ~1 GiB a stream past 4, so the shared KV pool shrinks
# (~2.0-2.6M tokens at 4, ~1.5M at 8).
PARALLEL="${PARALLEL:-4}"
# dflash2: incoai/GLM-5.3-Flash-DFlash2 drafts (CC BY-NC-ND 4.0: non-commercial
# use only); mtp: the checkpoint's own MTP head, slower and one request at a time.
DRAFTER="${DRAFTER:-dflash2}"
# Reasoning effort when a request does not say: high (LimeChain: tool harness
# 40/42 at high, 36/42 at max, Mia's default). Empty = Mia's default.
EFFORT="${EFFORT-high}"
# Drop the page cache on both nodes before a start (sudo -n; skipped with a note
# without passwordless sudo): TensorFold budgets from free memory, and a node
# that just copied ~170 GB of weights holds most of it as cache.
DROP_CACHES="${DROP_CACHES:-1}"

# Anything else of Mia's (CONTEXT, KV, DENSE, VISION, KV_POOL_GIB, ... her
# scripts/config.sh lists them with what each does) passes through from the
# shell: CONTEXT=524288 ./serve.sh restart

CONTAINER_NAME=glm53-flash-tf             # Mia's name, the same on both nodes

# Mia's scripts read these from the environment before their own defaults.
kit_env() {
  export WORKER="$WORKER_SSH" FABRIC_PEER="$WORKER_IP" HF_CACHE WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}" \
    HOST PORT PARALLEL DRAFTER
}

log() { echo "[glm53 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
need_kit() {
  [ -x "$KIT_DIR/start.sh" ] || { echo "no recipe checkout in $KIT_DIR: ./setup.sh first" >&2; exit 1; }
  local at; at=$(git -C "$KIT_DIR" rev-parse HEAD 2>/dev/null)
  [ "$at" = "$KIT_COMMIT" ] || echo "note: $KIT_DIR is at ${at:0:12}, not the pinned ${KIT_COMMIT:0:12} (./setup.sh checks it out)" >&2
}
