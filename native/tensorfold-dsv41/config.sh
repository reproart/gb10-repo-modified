# shellcheck shell=bash disable=SC2034  # read by the scripts in this directory
# Shared settings for DeepSeek-V4.1-Flash-1017 (EXL3 2.9 bpw, 1M context) on
# TensorFold, TP=2 across two DGX Spark over the ConnectX-7 cable. Source it,
# don't run it. Every value is "${VAR:-default}", so a one-off override from
# the shell works:
#   MAX_RUNNING=2 ./serve.sh
#
# The values that belong to your pair of machines (CX7 addresses, RDMA
# devices, ssh to the worker) go in config.local.sh, next to this file and
# not committed: cp config.local.sh.example config.local.sh, then edit it.
# Shell variables still win over both files.
#
# The head is the machine you type on (rank 0, serves the API); the worker is
# the second Spark (rank 1), reached with key-based ssh from the head.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Site values first, so the defaults below only fill what they leave unset.
# They are written the same way (NAME="${NAME:-value}"), so the shell wins.
# shellcheck disable=SC1091
[ -f "$ROOT/config.local.sh" ] && . "$ROOT/config.local.sh"

# --- The two nodes on the CX7 link (docs/NETWORK.md) ---
HEAD_IP="${HEAD_IP:-10.100.80.1}"          # head's address on the CX7 link (rank 0, rendezvous)
WORKER_IP="${WORKER_IP:-10.100.80.2}"      # worker's address on the CX7 link (rank 1)
WORKER_SSH="${WORKER_SSH:-$WORKER_IP}"     # how the head reaches the worker (user@host or an ssh alias)

# --- RoCE and NCCL (`ibdev2netdev`, `ip -br addr`, `show_gids`) ---
RDMA_DEVICES="${RDMA_DEVICES:-rocep1s0f1,roceP2p1s0f1}"   # TensorFold's own RDMA gathers: both CX7 ports
NCCL_IB_HCA="${NCCL_IB_HCA:-$RDMA_DEVICES}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"   # the netdev that holds HEAD_IP / WORKER_IP
# RoCE v2 IPv4 GID. After a link flap (cable reseat, one node rebooting) the
# index can move and NCCL hangs at start: -1 lets NCCL choose.
GID_INDEX="${GID_INDEX:-3}"

# --- Weights (the same paths on both nodes) ---
# ./download.sh fills model/ (EXL3 weights, 196 GiB) and engram/ (DeepSeek's
# Engram tables, 189 GiB, read from NVMe at random offsets while serving).
MODEL_VIEW="${MODEL_VIEW:-/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw}"
# Downloads are staged here first, then hard-linked into MODEL_VIEW: keep it
# on the same disk.
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$MODEL_VIEW.src}"
# Rank weight caches, spilled prompts and compiled kernels, kept between runs
# (~100 GiB per node; the first start writes it). Root-owned: the containers
# run as root.
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/gb10-tensorfold-dsv41}"

# --- The engine image (./build.sh builds it on the head, copies it to the worker) ---
TF_REPO="${TF_REPO:-https://github.com/Elevatormusic/TensorFold.git}"
TF_COMMIT="${TF_COMMIT:-92aa710b533d}"     # fork PRs #1-#4 with their review fixes (docs/WHAT-I-CHANGED.md)
IMAGE="${IMAGE:-tensorfold-dsv41:$TF_COMMIT}"

# --- The API: OpenAI-compatible at http://HOST:PORT/v1, on the head ---
# 8888 like ../serve.sh, so bench/ works unchanged; the two never run together
# (each wants ~100 GB a node). 0.0.0.0 as for the other servers here; with
# HOST=0.0.0.0 on a shared network set API_KEY_FILE (a file with one key per
# line, mounted read-only). The tests in tests/ send no key.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
API_KEY_FILE="${API_KEY_FILE:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash-1017}"
SERVED_ALIAS="${SERVED_ALIAS:-}"           # optional second model id (e.g. the one a router already uses)

# --- Serving (docs/WHAT-I-CHANGED.md, "Launcher settings") ---
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1048576}" # the full 1M window; the KV pool fits on 2 x 128 GB
MAX_RUNNING="${MAX_RUNNING:-4}"            # concurrent requests on the lane (--parallel)
DRAFT_TOKENS="${DRAFT_TOKENS:-5}"          # DSpark draft depth (--mtp-drafts); 4 measured slower
# Prompt chunk (TF_DS_PREFILL_CHUNK; engine default 512). 4096 measured 6-15%
# slower (runs/2026-10-18), and the prefix cache keeps prompts at these
# boundaries.
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
SPILL_GIB="${SPILL_GIB:-64}"               # kept prompts also go to disk (survive a restart); 0 = off
TEMPERATURE="${TEMPERATURE:-1.0}"          # DeepSeek's recommendation for V4.1
TOP_P="${TOP_P:-1.0}"
VISION="${VISION:-1}"                      # 1: accept image input (--vision; not measured here)
# The no-swap slice from ./host-tuning.sh; empty = Docker's default (swap allowed).
CGROUP_PARENT="${CGROUP_PARENT-tensorfold.slice}"
RESTART_POLICY="${RESTART_POLICY:-no}"     # docker's; see docs/OPERATIONS.md before changing it
# CPU pinning for both rank containers (docker --cpuset-cpus), as ../serve.sh
# pins SGLang: GB10's ten Cortex-X925 cores (3.9 GHz); 0-4 and 10-14 are the
# Cortex-A725 (2.8 GHz). The recipe left open whether its 10-20% slow runs
# (1 of 3 code replies at 69-71 tok/s instead of 85, same text, same steps)
# come from the engine's main thread landing on a slow core: this rules it
# out, unmeasured. ./preflight.sh checks the list against cpufreq on both
# nodes. Empty = no pinning (the recipe as measured).
CPUSET="${CPUSET-5-9,15-19}"
PREFLIGHT="${PREFLIGHT:-1}"                # 0: ./serve.sh skips ./preflight.sh

# Anything else: `tensorfold serve` flags, appended last; extra environment
# for both containers, as space-separated NAME=value pairs (replacing a value
# above of the same name).
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

# --- Derived, and the helpers the scripts share ---
API="http://${HOST/0.0.0.0/127.0.0.1}:${PORT}/v1"
STORE_PORT=29551                           # TensorFold's rank-0 rendezvous port (--master-port default)
LOGS="$ROOT/logs"
RANK0=tfv41-r0                             # container names (docs and earlyoom refer to them)
RANK1=tfv41-r1

log() { echo "[tfv41 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
auth() { [ -n "${API_KEY_FILE:-}" ] && printf 'Authorization: Bearer %s' "$(grep -v '^#' "$API_KEY_FILE" | head -1 | sed 's/^.*: *//')"; }
healthy() { curl -fsS -m 10 -H "$(auth)" "$API/models" 2>/dev/null | grep -q "\"$SERVED_MODEL_NAME\""; }

need_docker() {
  if ! command -v docker >/dev/null; then
    echo "docker not found (DGX OS ships it; sudo apt install docker.io otherwise)" >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker is installed but not usable by $(id -un): sudo usermod -aG docker $(id -un)," \
      "then log in again (or run with sudo)" >&2
    exit 1
  fi
}
