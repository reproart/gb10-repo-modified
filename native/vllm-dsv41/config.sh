# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for DeepSeek-V4.1-Flash-1017 (the same EXL3 2.9 bpw weights and
# Engram tables as ../tensorfold-dsv41) on sudoingX's tuned vLLM with styles01's
# 1017 overlay, TP=2 across two DGX Spark over RoCE: styles01's
# deepseek-v41-1017-exl3-2x-dgx-sparks (runbook in styles01/sparkrun-recipes),
# reshaped like the rest of this project. Source it, don't run it. Every value
# is "${VAR:-default}", so a one-off override from the shell works:
#   MAX_NUM_SEQS=1 ./serve.sh
#
# The pair's own values (addresses, ssh, the CX7 netdev) are shared with
# ../tensorfold-dsv41: its config.local.sh is read too, then ours
# (config.local.sh next to this file, not committed) for anything that
# differs. The shell wins over both.

VD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(cd "$VD_ROOT/../tensorfold-dsv41" 2>/dev/null && pwd || true)"
# shellcheck disable=SC1091
[ -f "$VD_ROOT/config.local.sh" ] && . "$VD_ROOT/config.local.sh"
# shellcheck disable=SC1091
[ -n "$TF_DIR" ] && [ -f "$TF_DIR/config.local.sh" ] && . "$TF_DIR/config.local.sh"

# --- The two nodes: the head (this machine, rank 0, serves the API) and the worker ---
HEAD_IP="${HEAD_IP:-10.100.80.1}"
WORKER_IP="${WORKER_IP:-10.100.80.2}"
WORKER_SSH="${WORKER_SSH:-$WORKER_IP}"
# The CX7 netdev that holds HEAD_IP / WORKER_IP (tensorfold's NCCL_SOCKET_IFNAME),
# and its RoCE device, found from sysfs when empty. NCCL is pinned to that one
# device: auto-detection hands NCCL the port without an IP and it dies
# (styles01's launcher note). The RoCE v2 GID index is derived on each node
# from that node's IPv4 (FAB_GID to force one), so a link flap that moves the
# GID table does not hang the start.
FAB_IF="${FAB_IF:-${NCCL_SOCKET_IFNAME:-enp1s0f1np1}}"
FAB_HCA="${FAB_HCA:-}"
FAB_GID="${FAB_GID:-}"

# --- Weights ---
# The two hf download folders ../tensorfold-dsv41 uses (its README, "Weights"),
# read only, and the serving tree ./repack.sh builds from them on each node:
# hard links (no copy; it has to be on the same filesystem as both), a union
# weight index, and the config / plan rewrites the engine reads.
MODEL_DIR="${MODEL_DIR:-${MODEL_VIEW:+$MODEL_VIEW/model}}"
MODEL_DIR="${MODEL_DIR:-/models/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}"
ENGRAM_DIR="${ENGRAM_DIR:-${MODEL_VIEW:+$MODEL_VIEW/engram}}"
ENGRAM_DIR="${ENGRAM_DIR:-/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-engram}"
SERVE_DIR="${SERVE_DIR:-/models/styles01/DeepSeek-V4.1-Flash-1017-serving}"
# Compiled kernels, the Markov token list, the depth file: the containers'
# /root/.cache, kept between runs, on each node. Root-owned.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-vllm-dsv41}"

# --- The engine image: published on GHCR by styles01 ---
# v1 is what the runbook and one-button.sh run; v2 is what the sparkrun recipe
# names. image.lock (written by ./pull.sh, committed) pins the digest it got.
IMAGE="${IMAGE:-ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
if [ -z "${IMAGE_DIGEST:-}" ] && [ -f "$VD_ROOT/image.lock" ]; then
  IMAGE_DIGEST="$(tr -d '[:space:]' < "$VD_ROOT/image.lock")"
fi
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
image_ref() {
  if [ -n "$IMAGE_DIGEST" ]; then echo "$IMAGE@$IMAGE_DIGEST"; else echo "$IMAGE:$IMAGE_TAG"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-gb10-vllm-dsv41}"

# --- The API on the head: http://HOST:PORT/v1 ---
# 8888 and the model id tensorfold-dsv41 serves, so bench/ and clients work
# unchanged (styles01 uses :8000, "dsv41"); one server at a time.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash-1017}"
API_KEY="${API_KEY:-}"

# --- Serving: the runbook's production settings (2026-10-18) ---
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"
# 0.85: their KV pool of 711,931 tokens (2.7 requests of 262K). A clean boot
# (~117 GB free) takes 0.88. The pool itself is pinned per rank by KV_BYTES
# (2.5 GiB, as the MiaAI kit ships); KV_BYTES= sizes it from GPU_UTIL.
GPU_UTIL="${GPU_UTIL:-0.85}"
KV_BYTES="${KV_BYTES-2684354560}"
# DSpark speculative decoding: up to DRAFT_TOKENS a round, the drafter split
# across both nodes (TP2 drafter: +20-30 tok/s at a fixed depth), the depth
# picked each round by the drafter's confidence against ROUND_MS, the cost of
# a round at depth 0, 1, ... (ms, drafter included). The table is per boot
# and hardware: a stale one under-speculates (structured 106 instead of 127
# tok/s). Re-measure it with DEPTH_PROBE=1 (README, "Depth table").
DRAFT_TOKENS="${DRAFT_TOKENS:-6}"
DRAFT_TP="${DRAFT_TP:-2}"
DEPTH_POLICY="${DEPTH_POLICY:-conf}"
ROUND_MS="${ROUND_MS:-31.9,35.2,38.6,42.6,45.3,48.1,51.0}"
# 1: boot for a depth sweep (fixed:5, the policy read from CACHE_DIR/dspark_depth
# every 64 verify steps, logged every 200).
DEPTH_PROBE="${DEPTH_PROBE:-0}"
VISION="${VISION:-1}"                      # 0 stubs the vision tower (~0.9 GiB a rank back to KV)
CPUSET="${CPUSET-}"                        # docker --cpuset-cpus on both; empty = as measured (5-9,15-19: the X925 cores)
DROP_CACHES="${DROP_CACHES:-1}"            # drop the page cache before start (sudo -n): vLLM sizes its pools to "free"

# The engine's tuned knobs (styles01's launcher RECIPE_ENV, as measured):
# EXL3 MoE kernels, the RDMA all-reduce ring, the paced L2 prefetch, the
# Engram spin reader, the local draft argmax, mHC overlap, TensorFold's
# grouped EXL3 prompt kernels. RECIPE_ENV= runs without them.
RECIPE_ENV="${RECIPE_ENV-VLLM_EXL3_MOE_PARALLEL_GROUPS=1 EXL3_MOE_COOP_WIDE=1 EXL3_INT8_GEMV=0 VLLM_DSPARK_DRAFT_MS=0 VLLM_DSPARK_MARKOV_CACHE=/root/.cache/markov_top_tokens.json EXL3_L2PF=attn,moe,next EXL3_RDMA_AR=1 VLLM_SHORT_CTX_BOUNDS=512,1025 EXL3_ROUTER=1 EXL3_L2PF_HC=1 VLLM_DSPARK_LOCAL_ARGMAX=1 EXL3_SMOE_FLAGS=-DSMOE_RUNFLAGS=1 VLLM_DSV41_MHC_OVERLAP=1 VLLM_ENGRAM_DISK_SPIN_MS=1000 EXL3_MROW_GROUPED=1 VLLM_EXL3_MOE_TF_PROMPT=1 TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions}"
# 256 Engram reader threads (32 stalled a 512-token prefill step ~74 ms).
engram_default='{"cpu_offload":true,"dp_shared_memory":false,"disk_offload":true,"disk_threads":256}'
ENGRAM_CONFIG="${ENGRAM_CONFIG:-$engram_default}"

# Anything else: vllm serve flags (appended last) and NAME=value pairs for both
# containers (replacing one above of the same name).
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

log() { echo "[vllm-dsv41 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
need_docker() {
  command -v docker >/dev/null || { echo "docker not found (DGX OS ships it)" >&2; exit 1; }
  docker info >/dev/null 2>&1 || {
    echo "docker is not usable by $(id -un): sudo usermod -aG docker $(id -un), then log in again" >&2; exit 1; }
}
