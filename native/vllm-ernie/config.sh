# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for ERNIE 5.5 Flash Preview (280B MoE, 14B active, MXFP4
# experts + FP8, bundled DSpark draft, 1M context) on vLLM 0.32.0 + Baidu's
# vLLM-ERNIE, TP=2 + EP=2 across two DGX Spark over the CX7 link, with
# jjang-ai's ernie-5.5-dgx-spark runtime (PROFILE=spark) or without it
# (PROFILE=stock). Source it, don't run it. Every value is "${VAR:-default}":
#   PROFILE=stock ./serve.sh
#
# The pair's values (addresses, ssh, the CX7 netdev) are shared with
# ../tensorfold-dsv41: its config.local.sh is read too, then ours.

VE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$VE_ROOT/../tensorfold-dsv41" 2>/dev/null && pwd || true)"
# shellcheck disable=SC1091
[ -f "$VE_ROOT/config.local.sh" ] && . "$VE_ROOT/config.local.sh"
# shellcheck disable=SC1091
[ -n "$TF_DIR" ] && [ -f "$TF_DIR/config.local.sh" ] && . "$TF_DIR/config.local.sh"

# --- The two nodes: the head (this machine, rank 0, serves the API) and the worker ---
HEAD_IP="${HEAD_IP:-10.100.80.1}"
WORKER_IP="${WORKER_IP:-10.100.80.2}"
WORKER_SSH="${WORKER_SSH:-$WORKER_IP}"
# The CX7 netdev holding HEAD_IP / WORKER_IP; its RoCE device and RoCE v2 GID
# index are found from sysfs on each node at every start (FAB_HCA, FAB_GID force them).
FAB_IF="${FAB_IF:-${NCCL_SOCKET_IFNAME:-enp1s0f1np1}}"
FAB_HCA="${FAB_HCA:-}"
FAB_GID="${FAB_GID:-}"

# --- Weights: the same path on both nodes, mounted read-only ---
#   dealign  dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4 (default): an
#            abliterated edit of Baidu's release, the one the runtime was
#            measured with. Its licence (DealignAI Community v0.3) forbids
#            serving it to anyone but yourself (README, "Weights").
#   baidu    baidu/ERNIE-5.5-Flash-Preview: the official release, Apache-2.0,
#            the same MXFP4 experts and DSpark head, the same size.
# 154.5 GiB either way. MODEL_DIR, if set, wins.
WEIGHTS="${WEIGHTS:-dealign}"
case "$WEIGHTS" in
  dealign) _dir=/models/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4 ;;
  baidu)   _dir=/models/baidu/ERNIE-5.5-Flash-Preview ;;
  *) echo "WEIGHTS must be dealign or baidu, not '$WEIGHTS'" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$_dir}"
unset _dir
# Kernel and compile caches on each node (TileLang, Triton, FlashInfer, sparknet), root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-vllm-ernie}"

# --- The image, built here by ./build.sh and copied to the worker ---
IMAGE="${IMAGE:-gb10-vllm-ernie}"
IMAGE_TAG="${IMAGE_TAG:-5b9f1c2-ba322a5}"   # runtime commit - vLLM-ERNIE commit (docker/Dockerfile)
image_ref() { echo "$IMAGE:$IMAGE_TAG"; }
CONTAINER_NAME="${CONTAINER_NAME:-gb10-vllm-ernie}"

# --- The API on the head ---
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ernie-5.5-flash}"
API_KEY="${API_KEY:-}"

# --- Serving ---
#   spark  the runtime's own launcher (scripts/serve_spark_ernie.sh) with its
#          18 patches: MODE tp2-dspark is its 2x-1m-fast profile (1M context,
#          DSpark 7, KV pinned at 9 GiB), tp2-ar its pure-AR one.
#   stock  vLLM-ERNIE's README command for FP4 on this layout: no runtime,
#          FP8 indexer (its MXFP4 indexer is for data-centre Blackwell only).
PROFILE="${PROFILE:-spark}"
MODE="${MODE:-tp2-dspark}"
# The runtime's launcher reads its knobs as SPARK_* (SPARK_MAX_MODEL_LEN,
# SPARK_GPU_MEM_UTIL, SPARK_DSPARK_NUM_TOKENS, ...): pass any of them in EXTRA_ENV.
# sparknet one-shot collectives need the pair's peer-HCA and subnet maps
# (what its `spark-ernie discover` writes); off until they are set here:
#   SPARKNET=1 SPARKNET_PEER_HCAS='{"1":["rocep1s0f1"]}' SPARKNET_SUBNETS='{"rocep1s0f1":"10.100.80.0/24"}'
SPARKNET="${SPARKNET:-0}"
SPARKNET_PEER_HCAS="${SPARKNET_PEER_HCAS:-}"
SPARKNET_SUBNETS="${SPARKNET_SUBNETS:-}"
# stock profile values.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.8}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
DRAFT_TOKENS="${DRAFT_TOKENS:-7}"          # DSpark; 0 = off

# 1: before the start, on both nodes with the GPU idle, drop the page cache and
# compact free memory (../scripts/mem-prep.sh).
MEM_PREP="${MEM_PREP:-1}"

# Anything else: vllm serve flags (stock) and NAME=value pairs for both containers.
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

log() { echo "[vllm-ernie $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
need_docker() {
  command -v docker >/dev/null || { echo "docker not found (DGX OS ships it)" >&2; exit 1; }
  docker info >/dev/null 2>&1 || {
    echo "docker is not usable by $(id -un): sudo usermod -aG docker $(id -un), then log in again" >&2; exit 1; }
}
