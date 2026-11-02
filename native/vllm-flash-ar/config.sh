# shellcheck shell=bash disable=SC2034  # NATIVE_DIR, VFA_ROOT: read by the scripts here
# Shared settings for Qwen3.8-Flash-Next as Intel's AutoRound int4 (+ an int8
# lm_head and fp8 side layers) on vLLM in Docker, one Spark: the scripts in this
# directory source it, scripts/serve-intel-ar.sh reads the exported values.
# Source it, don't run it. Every value is "${VAR:-default}", so the shell wins:
#   MTP=2 SEQS=16 ./serve.sh
# This machine's own values (paths, port, API key, restart policy) go in
# config.local.sh next to this file (not committed): cp config.local.sh.example
# config.local.sh, then edit it. It is read first.

VFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "$VFA_ROOT/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$VFA_ROOT/config.local.sh" ] && . "$VFA_ROOT/config.local.sh"

# --- Weights: the prepared checkpoint (int4 experts + int8 lm_head + fp8 side
# layers) and the fp8 n-gram ("PLE") table, two `hf download --local-dir`
# folders (./download.sh; or built from Intel's release with ./prepare.sh).
# The -MTP_int4RTN variant (int4 draft experts, -3.5 GiB) is the default; the
# plain -hybrid one (bf16 draft) works too: MODEL_REPO / MODEL_DIR.
export MODEL_REPO="${MODEL_REPO:-Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN}"
export MODEL_REVISION="${MODEL_REVISION:-19f9710c8f6600a32e15fb98bc77613ee8ec369b}"
export MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN}"
export TABLE_REPO="${TABLE_REPO:-Saren/Qwen3.8-Flash-Next-ple-table-fp8}"
export TABLE_REVISION="${TABLE_REVISION:-50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14}"
export TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"

# --- The image (./build.sh: the official Flash-Next image + this recipe's
# patches) and the container. IMAGE is a tag: point it at a backup tag to roll
# back.
export IMAGE="${IMAGE:-qwen38-flash-dgx}"
export NAME="${NAME:-qwen38-flash}"

# --- The API: http://<host>:PORT/v1 ---
# 8888, as every server here (the recipe's own default was 8000: PORT=8000 in
# config.local.sh keeps old clients working). One server at a time.
export PORT="${PORT:-8888}"
# Model ids the API answers to: the first is the one /v1/models lists first and
# replies carry; "qwen" is the recipe's earlier id, kept for existing clients.
export SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next qwen}"
# qwen3_xml or qwen3_coder: both run vLLM's qwen3 parser configuration, which the
# image's parser patches (12, 13) fix.
export TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"

# API key (Bearer token) for the OpenAI-compatible endpoint:
#   non-empty -> clients must send "Authorization: Bearer <key>" (vLLM --api-key)
#   empty     -> no auth (closed-network deployment)
# Generate: openssl rand -hex 32. Keep it in config.local.sh, or read it from a
# file outside the repo:  API_KEY="$(cat "$HOME/.qwen-api-key")"
# /health stays open (container HEALTHCHECK unaffected); bench/ then needs
# GB10_API_KEY set to the same value. The key is visible in `docker inspect`.
export API_KEY="${API_KEY:-}"
# Host address the API/metrics ports are published on. Empty = all
# interfaces; 127.0.0.1 = this machine only (e.g. behind a reverse proxy).
# The metrics sidecar (METRICS_PORT) has no auth even when API_KEY is set.
export BIND_ADDR="${BIND_ADDR:-}"

# --- Serving ---
export SEQS="${SEQS:-8}"            # concurrent requests (don't benchmark with 1-2: excess requests queue)
export MTP="${MTP:-3}"              # MTP speculative tokens; 0 = off
export PREFIX_CACHE="${PREFIX_CACHE:-1}"
# Deterministic memory sizing for unified-memory boxes (GB10 / DGX Spark):
# near-zero utilization fraction plus an explicit KV pool, so the driver
# never oversubscribes the unified pool (NV_ERR_NO_MEMORY / Xid 31 crashes).
export GPU_MEM="${GPU_MEM:-0.01}"
# vLLM reads "g" as 10^9 bytes: 30g = 27.9 GiB, held ~4 concurrent 127k
# contexts (~0.55M tokens, ~53 KB/token measured) on a GX10; the boot log's
# "~966k tokens" overstates it. 35g / 40g fit ~5 / ~6 such contexts but cost
# 2-7% long-prefill speed: every GiB of KV is a GiB less page cache for the
# ~48 GiB PLE table. README, "Sizing the KV pool".
export KV_BYTES="${KV_BYTES:-30g}"
# Context: 262144 native. 500k via YaRN: CTX=500000 YARN=1 (the upstream-
# validated ceiling); the launcher pins the MTP draft's length to CTX so
# speculative decoding still boots.
export CTX="${CTX:-262144}"
export YARN="${YARN:-0}"
# Prefill chunk (--max-num-batched-tokens). 8192 as the recipe measured; a
# smaller chunk keeps other streams decoding during a long prefill at a cost
# in prefill speed (blazux, NVFP4 base: a decoding client at 0.2 / 0.4-0.6 /
# 1.0 / 1.6 tok/s during a big prefill at 8192 / 2048 / 1024 / 512, prefill
# -0 / ? / -17-36% / -32-49%). ../bench/perf.py --only interference measures it.
export BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
# Marlin's split-K reduction with atomic adds (VLLM_MARLIN_USE_ATOMIC_ADD): the
# recipe's setting, for speed. Dense Marlin GEMMs only: the MoE kernel always
# reduces in order (marlin_moe.py passes use_atomic_add=False), so 0 left the
# run-to-run differences below as they were (../bench/consistency.py).
export MARLIN_ATOMIC_ADD="${MARLIN_ATOMIC_ADD:-1}"
# 1: deterministic MoE token alignment (src/patch_det_moe_align.py): the routes of an
# expert in a fixed order, so the Marlin MoE sums run in the same order every time. On
# this recipe three cold runs of one greedy prompt gave first-token logprobs 0.16-0.45
# nats apart (../bench/consistency.py), with MARLIN_ATOMIC_ADD 1 or 0; with 1 here, cold,
# cold again and cache hits matched exactly (gap 0.0000, 4K and 32K), at no measurable
# cost with the auto floor below (decode 66.4 / 65.4 tok/s with 1 / 0; README patch 15).
# On by default since 2026-11-01; 0 = vLLM's kernel. ./build.sh checks it against the
# image's CUDA kernel on the GPU; an image built before it warns and runs the kernel.
export DET_MOE_ALIGN="${DET_MOE_ALIGN:-1}"
# The steps it applies to: from this many tokens up. auto = SEQS x (1 + MTP) + 1 (33 as
# set here), one more than the largest decode-only step: prefill and steps that carry a
# prefill chunk take it, decode keeps the kernel. Decode loses nothing by that: one
# stream (1 + MTP tokens) fits one block, where the kernel's order is harmless, and a
# step of several streams depends on what else is in the batch whatever the alignment.
# 9 = every step where an expert can fill more than one block of 8 (the patch's own
# default; single-stream decode is still below it).
export DET_MOE_ALIGN_MIN_TOKENS="${DET_MOE_ALIGN_MIN_TOKENS:-auto}"
# docker --cpuset-cpus; empty = all cores (as the recipe measured). 5-9,15-19 are
# GB10's Cortex-X925 cores (blazux's concurrent profile pins to them).
export CPUSET="${CPUSET:-}"

# Restart policy of the container. no (default here) = it runs until stopped or
# the machine restarts, like every other server in this project; unless-stopped
# = it also comes back after a reboot or a crash (the recipe's default). Only
# with this as the machine's one server: next to the SGLang service or another
# recipe, two servers would start at boot and share 128 GB.
export RESTART="${RESTART:-no}"

# madvise(MADV_RANDOM) the PLE mmap: no readahead around 160-byte row faults.
# Upstream (blazux 0c6df7e) measured 4-8% faster cold prefill and a cleaner
# page cache; on by default. 0 = kernel readahead (worth trying when the
# table sits on remote RAM with no page-cache headroom).
export PLE_MADV_RANDOM="${PLE_MADV_RANDOM:-1}"

# QSA top-k (sparse-attention block selection): the stock kernel is
# non-deterministic on GB10 and can drop candidates (vllm#51782).
# DET_TOPK=1 = @jschmied's deterministic kernel at full speed (default);
# EXACT_TOPK=1 = exact torch.topk fallback (slower long prefill, wins when set).
export DET_TOPK="${DET_TOPK:-1}"
export EXACT_TOPK="${EXACT_TOPK:-0}"

# Per-step prefill metrics on vLLM's /metrics (vllm:scheduled_ctx_tokens_total;
# watch live prefill tok/s with bench/ppwatch.sh). 0 = off.
export ITER_DETAILS="${ITER_DETAILS:-0}"

# Engine-side metrics sidecar port (PLE gather counters, the mamba guard
# tripwire, the never-evict pin gauges — vllm_custom_metrics): served from the
# engine process on its own endpoint, because vLLM's /metrics in the API
# process cannot see them. 0 = off. ../bench reads it with GB10_METRICS_PORT.
export METRICS_PORT="${METRICS_PORT:-18400}"

# Prefix-cache diagnosis logging (VLLM_HIT_DEBUG=1 in the container):
# per-group hit breakdown, mamba boundary publication, evictions, chunk stops.
export HIT_DEBUG="${HIT_DEBUG:-0}"

# Never-evict pin: any request whose prompt contains this exact substring has
# its prompt-prefix KV blocks pinned (held out of eviction) — meant for a
# long fixed system prompt. Empty disables it. Needs PREFIX_CACHE=1; use a
# distinctive substring (a few dozen characters), not a word or two.
export PIN_PROMPT="${PIN_PROMPT:-}"

# Anything else: vLLM flags (appended to the command line) and NAME=value pairs
# for the container (EXTRA_ENV replaces a value the launcher sets under the
# same name: docker would keep the first of two).
export EXTRA="${EXTRA:-}"
export EXTRA_ENV="${EXTRA_ENV:-}"

# After this many seconds without a request the engine gives torch's cached GPU blocks
# back to the system (torch.cuda.empty_cache(), src/vllm_custom_metrics.py; CUDA graph
# pools stay). On GB10 they are host memory, held at the peak of the last large request:
# 8 GiB on a GX10 after a multi-stream long-context run (2026-11-02), with ~2 GiB left
# available and earlyoom at 1. 0 = keep them. The next large request allocates again.
export IDLE_RELEASE_SEC="${IDLE_RELEASE_SEC:-60}"

# The watchdog (./watchdog.sh): restart the server when available memory is under
# WATCHDOG_MIN_AVAIL_MB AND no request has run or waited for WATCHDOG_IDLE_MIN
# minutes, before earlyoom stops it at a worse moment. 1 = ./serve.sh starts it in
# the background, ./stop.sh stops it; 0 (default) = off (./watchdog.sh still runs
# by hand). A restart drops the prefix cache and takes ~5 min. At most one per
# WATCHDOG_COOLDOWN_MIN. The threshold must sit above earlyoom's SIGTERM line
# (1 GiB after ../scripts/host-tuning.sh; ~8 GB with DGX OS's own: then ~10240).
export WATCHDOG="${WATCHDOG:-0}"
export WATCHDOG_MIN_AVAIL_MB="${WATCHDOG_MIN_AVAIL_MB:-2048}"
export WATCHDOG_IDLE_MIN="${WATCHDOG_IDLE_MIN:-5}"
export WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-60}"
export WATCHDOG_COOLDOWN_MIN="${WATCHDOG_COOLDOWN_MIN:-60}"

# Before a start, with the GPU idle: drop the page cache and compact free memory
# (../scripts/mem-prep.sh; passwordless sudo for the two writes:
# ../scripts/mem-prep.sh sudoers). The PLE table reads through the page cache,
# so this costs the first requests some cold reads (PREWARM streams it once at
# boot). 0 = memory as it is.
export MEM_PREP="${MEM_PREP:-1}"
# ./serve.sh waits until the API answers (up to WAIT_MIN minutes); 0 = return
# right after the container starts.
export WAIT_MIN="${WAIT_MIN:-15}"

log() { echo "[vllm-flash-ar $(date -u +%H:%M:%SZ)] $*"; }
need_docker() {
  command -v docker >/dev/null || { echo "docker not found (DGX OS ships it)" >&2; exit 1; }
  docker info >/dev/null 2>&1 || {
    echo "docker is not usable by $(id -un): sudo usermod -aG docker $(id -un), then log in again" >&2; exit 1; }
}
