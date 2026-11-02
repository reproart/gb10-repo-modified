#!/usr/bin/env bash
# Serve the int4+int8+fp8 hybrid build of Qwen3.8-Flash-Next on a DGX Spark:
# Intel/Qwen3.8-Flash-Next-W4A16-AutoRound experts via GPTQ-Marlin int4,
# int8 GPTQ lm_head, blockwise-fp8 side layers, PLE n-gram table mmapped from
# a separate directory (fp8 table recommended — see tools/fetch-ple-table-fp8.sh).
#
# The checkpoint must be prepared first: download the prebuilt one (README
# Quickstart) or build it with ./prepare.sh (README "Modify the weights
# yourself").
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN \
#   TABLE_DIR=/models/ple-table-fp8 scripts/serve-intel-ar.sh
#
#   MTP=0 ... scripts/serve-intel-ar.sh     # no speculation (first-boot sanity)
#
# In this project ../serve.sh runs it with config.sh's values (port 8888, the
# one-server checks, memory preparation); its own defaults below stay the
# recipe's for a direct call.
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
# -MTP_int4RTN (int4 draft experts, the default variant; prepare.sh step 8
# builds it) or the plain -hybrid dir (bf16 draft, +3.5 GiB resident).
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN}"
TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
PREWARM="${PREWARM:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
EXTRA="${EXTRA:-}"
# KV_BYTES: size the KV cache explicitly (e.g. 13.2g) instead of by
# gpu-memory-utilization fraction — deterministic footprint on unified-memory
# boxes where "free memory" profiling is unreliable. Pair with a tiny GPU_MEM.
[ -n "${KV_BYTES:-}" ] && EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# API_KEY: when non-empty, require "Authorization: Bearer <key>" on the API
# (vLLM --api-key). Empty/unset = open endpoint, as before. /health stays
# unauthenticated (container HEALTHCHECK unaffected); the key ends up in the
# container's command line (visible via docker inspect), and /metrics may
# require the key too depending on the vLLM build.
API_KEY="${API_KEY:-}"
[ -n "$API_KEY" ] && EXTRA="--api-key $API_KEY $EXTRA"

# Docker log rotation (json-file driver): PLE mmap stats emits a log line
# every 30s, so an unbounded container log grows to GBs over months.
LOG_MAX_SIZE="${LOG_MAX_SIZE:-10m}"
LOG_MAX_FILE="${LOG_MAX_FILE:-3}"

# --enable-prompt-tokens-details (below): usage.prompt_tokens_details.cached_tokens
# in every reply, the prefix-cache hit a client sees (blazux #43 on this base).
# Model ids: SERVED_NAME may list several, space-separated (vLLM takes them all;
# the first is the one replies carry).
read -r -a SERVED_NAMES <<< "${SERVED_NAME:-qwen3.8-flash-next}"

# HEALTHCHECK probes vLLM's /health (503 while loading, 200 when serving);
# python3 because the base image does not guarantee curl. start-period covers
# the ~5-min weight load; 10 misses (~10 min) mark the container unhealthy in
# `docker ps`. Note --restart only reacts to the process exiting — restarting
# on "unhealthy" needs an external watcher (e.g. willfarrell/autoheal).
HEALTH_CMD="python3 -c 'import urllib.request;urllib.request.urlopen(\"http://localhost:8000/health\",timeout=10)'"

# YARN=1 with CTX>262144: extend context via YaRN rope scaling (Qwen's
# published recipe, upstream-validated to 500k on this model family). The
# override targets text_config.rope_parameters (Qwen4ExpForConditionalGeneration
# layout — same in the Intel AutoRound checkpoint), and going past the native
# max_position_embeddings needs VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 in the container.
YARN="${YARN:-0}"
OVR_ARGS=()
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then
  YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
  OVR_ARGS=(--hf-overrides "$YARN_OVR")
  ALLOW_LONG=1
fi

# PLE gather = CPU op + H2D copy: must run OUTSIDE CUDA graphs.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# FLASHINFER_AUTOTUNE=1 drops --no-enable-flashinfer-autotune (longer warmup,
# possibly faster kernels; default off as inherited from the NVFP4 recipe).
AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

SPEC=()
if [ "$MTP" != 0 ]; then
  # MTP + YaRN: dict hf_overrides are not propagated to the draft model, so
  # the draft keeps max_model_len=262144 while sharing the cache_config whose
  # mamba_block_size follows the target's length — vLLM aborts at boot with
  # "--mamba-block-size can only be set with --enable-prefix-caching".
  # Forcing the draft's max_model_len through the speculative config fixes it.
  DRAFT_LEN=""
  if [ "$YARN" != 0 ]; then DRAFT_LEN=",\"max_model_len\":${CTX}"; fi
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}${DRAFT_LEN}}")
fi

# Private MTP draft head (image patch, from upstream blazux 0c6df7e): vLLM
# shares the target's int8 lm_head with the drafter, so every draft step reads
# the whole head (616 MiB); the knobs below shrink that read. Outputs never
# change (the target verifies every drafted token), only acceptance can.
# NOTE: inert unless the image has the patch-mtp-draft-vocab section of the
# Dockerfile baked in (rebuild required; vLLM ignores the env otherwise).
# DRAFT_VOCAB: 1 = the drafter scores only the 65,536 most frequent tokens
# (+3-5% decode; the shipped id set is English/code-weighted — CJK/RU-heavy
# traffic may prefer 0 or a custom set via tools/build_draft_vocab.py);
# 0 = full vocabulary; a path = your own ids.npy.
DRAFT_VOCAB="${DRAFT_VOCAB:-1}"
# DRAFT_HEAD: int8 (default) = share the target's head; int4 = a private
# int4 g128 RTN Marlin copy of the full-vocabulary head built at first use
# (~320 MB, half the bytes per draft step, no vocabulary restriction).
# Measured a wash upstream (acceptance 1-8 points lower) — default int8.
DRAFT_HEAD="${DRAFT_HEAD:-int8}"
MTENV=()
[ "$DRAFT_HEAD" = int4 ] && MTENV+=(-e VLLM_MTP_DRAFT_HEAD=int4)
case "$DRAFT_VOCAB" in
  0) ;;
  1) MTENV+=(-e VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft_vocab_65536.npy) ;;
  *) MTENV+=(-e VLLM_MTP_DRAFT_VOCAB="$DRAFT_VOCAB") ;;
esac

# QSA top-k (the sparse-attention block selection). The stock persistent_topk
# kernel is non-deterministic on GB10 and can drop real candidates (vllm#51782).
# DET_TOPK=1 (default): @jschmied's deterministic kernel (Dockerfile, vllm#55122)
#   — deterministic at full prefill speed.
# EXACT_TOPK=1: exact torch.topk fallback (deterministic, -20-40% long prefill);
#   wins over DET_TOPK. EXACT_TOPK=fill: -inf-fill unwritten columns, then the
#   stock kernel (a diagnostic). Both 0 = stock kernel.
DET_TOPK="${DET_TOPK:-1}"
EXACT_TOPK="${EXACT_TOPK:-0}"
[ "$DET_TOPK" = 1 ] && MTENV+=(-e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so)
[ "$EXACT_TOPK" != 0 ] && MTENV+=(-e VLLM_QSA_EXACT_TOPK="$EXACT_TOPK")

# ITER_DETAILS=1: per-step prefill metrics (vllm:scheduled_ctx_tokens_total on
# vLLM's /metrics; live prefill tok/s via bench/ppwatch.sh). Needs the image's
# prefill-metrics patch — on an older image the stock flag logs one INFO line
# per engine step instead, so the launcher refuses to pass it there.
ITER_DETAILS="${ITER_DETAILS:-0}"

# Engine-side metrics sidecar (vllm_custom_metrics, image module): PLE gather
# counters, the mamba state-copy guard tripwire and the never-evict pin gauges.
# vLLM's own /metrics lives in the API-server process and cannot see
# EngineCore counters, so they are served from the engine process on a
# separate port. METRICS_PORT = host port; 0 = off. With metrics on, the PLE
# stats LOG lines default off (PLE_STATS_SEC=0). The port is wired up after
# the image pre-flight below, which checks that the module is actually there
# (publishing a mapping to nothing would just look healthy and lie).
METRICS_PORT="${METRICS_PORT:-18400}"

# PREFIX_CACHE=1: upstream disabled prefix caching over a CUBLAS error in the
# GDN in_proj GEMM on the cached-block path; the fp8-hybrid in_proj bypasses
# that kernel, and with FP8_HYBRID=1 prefix caching has been stable here.
PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-0}" = 1 ] && PC_ARG=--enable-prefix-caching

# Never-evict pin: PIN_PROMPT="some exact substring of your system prompt"
# keeps that prompt's KV blocks resident across other traffic (needs
# PREFIX_CACHE=1). See the README, patch 6. Use a distinctive substring of a
# few dozen characters: the first/last token are dropped before matching, so
# a one- or two-word marker matches nearly every prompt and the pin churns.
PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ]; then
  if [ "${PREFIX_CACHE:-0}" = 1 ]; then
    PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
             --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
  else
    echo "WARNING: PIN_PROMPT is set but PREFIX_CACHE != 1 — the never-evict pin needs prefix caching and is DISABLED." >&2
  fi
  if [ "${#PIN_PROMPT}" -lt 20 ]; then
    echo "WARNING: PIN_PROMPT is only ${#PIN_PROMPT} chars — a short marker matches unrelated prompts; use a longer, distinctive substring." >&2
  fi
fi

# BIND_ADDR: host address the API (and metrics) ports are published on.
# Empty = all interfaces (Docker's default). 127.0.0.1 keeps both reachable
# only from this machine (e.g. behind a reverse proxy) — note the metrics
# sidecar has no auth even when API_KEY is set.
BIND_ADDR="${BIND_ADDR:-}"
BIND_PFX="${BIND_ADDR:+$BIND_ADDR:}"

# Fail fast on unedited paths before touching the running container: Docker
# would silently auto-create a missing bind source (the classic /path/to
# placeholder) and the server would die at load.
case "$MODEL_DIR $TABLE_DIR" in
  *"/path/to"*)
    echo "ERROR: MODEL_DIR/TABLE_DIR contain /path/to placeholders — edit serve.sh." >&2
    exit 1 ;;
esac
for d in "$MODEL_DIR" "$TABLE_DIR"; do
  [ -d "$d" ] || { echo "ERROR: $d is not a directory — check MODEL_DIR/TABLE_DIR." >&2; exit 1; }
done

# Image pre-flight, before the running container is touched. Two knobs depend
# on image-side patches that older images lack, and vLLM ignores the env vars
# silently, so probe the image once and say so loudly instead. Serving stays
# correct either way — this is a loud note, not an error.
#   draft   — /opt/llm/draft_vocab_65536.npy: the private-MTP-draft-head section
#             (DRAFT_VOCAB / DRAFT_HEAD; without it the drafter just scores the
#             whole shared head).
#   metrics — the vllm_custom_metrics module: the engine-side metrics sidecar.
#             Without it PLE stats are neither served nor logged unless
#             PLE_STATS_SEC brings the log lines back (handled below).
IMAGE_OK=1 DRAFT_OK=0 METRICS_OK=0 DET_OK=0 EXACT_OK=0 ITER_OK=0 FENCES_OK=0 DETALIGN_OK=0 IDLE_OK=0
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  IMAGE_OK=0  # the main `docker run` fails loudly on its own; don't add noise
else
  probe=$(docker run --rm --entrypoint python3 "$IMAGE" -c '
import importlib.util as u, os
print("draft", int(os.path.exists("/opt/llm/draft_vocab_65536.npy")))
# The sidecar needs the module AND both start hooks (the PLE module kicks it on
# the first stats push; the scheduler init is the one that always runs). A
# module without hooks starts nothing and says nothing — probe for the pair.
def has(path, needle):
    try:
        return needle in open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return False
spec = u.find_spec("vllm")
pkg = os.path.dirname(spec.origin) if spec is not None else ""
ok = (spec is not None and u.find_spec("vllm_custom_metrics") is not None
      and has(os.path.join(pkg, "v1/core/sched/scheduler.py"), "set_pin_source")
      and has(os.path.join(os.path.dirname(pkg), "vllm_ple_mmap.py"), "_metrics_start"))
print("metrics", int(ok))
print("idlerelease", int(has(os.path.join(os.path.dirname(pkg), "vllm_custom_metrics.py"), "VLLM_IDLE_EMPTY_CACHE_SEC")))
if spec is not None:
    qsa = os.path.join(pkg, "models/qwen3_8_flash_next/nvidia/ops/qsa.py")
    print("det", int(os.path.exists("/opt/llm/kernel-det/_C_det.so") and has(qsa, "QSADET")))
    print("exact", int(has(qsa, "_qsa_exact_topk")))
    print("iter", int(has(os.path.join(pkg, "v1/metrics/loggers.py"), "_q38_prefill_metrics")))
    print("fences", int(has(os.path.join(pkg, "parser/engine/parser_engine_config.py"), "guard_literal_tool_markers")))
    print("detalign", int(has(os.path.join(pkg, "model_executor/layers/fused_moe/moe_align_block_size.py"), "_deterministic_align")))
' 2>/dev/null) || probe=""
  case "$probe" in *"draft 1"*) DRAFT_OK=1 ;; esac
  case "$probe" in *"metrics 1"*) METRICS_OK=1 ;; esac
  case "$probe" in *"det 1"*) DET_OK=1 ;; esac
  case "$probe" in *"exact 1"*) EXACT_OK=1 ;; esac
  case "$probe" in *"iter 1"*) ITER_OK=1 ;; esac
  case "$probe" in *"fences 1"*) FENCES_OK=1 ;; esac
  case "$probe" in *"detalign 1"*) DETALIGN_OK=1 ;; esac
  case "$probe" in *"idlerelease 1"*) IDLE_OK=1 ;; esac
  if [ -z "$probe" ]; then
    IMAGE_OK=0
    echo "WARNING: could not probe '$IMAGE' (python3/entrypoint missing?) — skipping patch checks." >&2
  fi
fi

if [ "$IMAGE_OK" = 1 ] && [ "$DRAFT_OK" = 0 ] \
   && { [ "$DRAFT_VOCAB" != 0 ] || [ "$DRAFT_HEAD" = int4 ]; }; then
  echo "WARNING: DRAFT_VOCAB=$DRAFT_VOCAB / DRAFT_HEAD=$DRAFT_HEAD will be IGNORED:" >&2
  echo "  image '$IMAGE' predates the Dockerfile draft-head section (no /opt/llm/draft_vocab_65536.npy)." >&2
  echo "  The server still runs correctly (full-vocabulary int8 draft head), just without the speedup." >&2
  echo "  To enable: docker build -t '$IMAGE' .  (layer cache keeps it quick; the ~20 GiB base is not re-downloaded)." >&2
fi

if [ "$IMAGE_OK" = 1 ] && [ "$DET_TOPK" = 1 ] && [ "$DET_OK" = 0 ] && [ "$EXACT_TOPK" = 0 ]; then
  echo "WARNING: DET_TOPK=1 will be IGNORED: image '$IMAGE' has no deterministic QSA top-k kernel" >&2
  echo "  (/opt/llm/kernel-det/_C_det.so + qsa.py wiring). The stock, non-deterministic kernel runs" >&2
  echo "  (vllm#51782). Rebuild: docker build -t '$IMAGE' .  — or EXACT_TOPK=1 on an image that has it." >&2
fi
if [ "$IMAGE_OK" = 1 ] && [ "$FENCES_OK" = 0 ]; then
  echo "WARNING: image '$IMAGE' predates the tool-call parser fixes (patches 12-13): a tool call the model" >&2
  echo "  shows as an example in a code block is parsed as a real call and the rest of the answer is lost." >&2
  echo "  Rebuild: docker build -t '$IMAGE' .  (../build.sh in this project)." >&2
fi
if [ "$IMAGE_OK" = 1 ] && [ "${IDLE_RELEASE_SEC:-0}" != 0 ] && [ "$IDLE_OK" = 0 ]; then
  echo "WARNING: IDLE_RELEASE_SEC=$IDLE_RELEASE_SEC will be IGNORED: image '$IMAGE' predates the idle release" >&2
  echo "  (src/vllm_custom_metrics.py). Rebuild: ./build.sh" >&2
fi
if [ "$IMAGE_OK" = 1 ] && [ "${DET_MOE_ALIGN:-0}" = 1 ] && [ "$DETALIGN_OK" = 0 ]; then
  echo "WARNING: DET_MOE_ALIGN=1 will be IGNORED: image '$IMAGE' predates the deterministic MoE alignment patch." >&2
  echo "  Rebuild: docker build -t '$IMAGE' .  (../build.sh in this project)." >&2
fi
if [ "$IMAGE_OK" = 1 ] && [ "$EXACT_TOPK" != 0 ] && [ "$EXACT_OK" = 0 ]; then
  echo "WARNING: EXACT_TOPK=$EXACT_TOPK will be IGNORED: image '$IMAGE' predates the exact QSA top-k patch. Rebuild." >&2
fi
if [ "$ITER_DETAILS" = 1 ]; then
  if [ "$ITER_OK" = 1 ] || [ "$IMAGE_OK" = 0 ]; then
    EXTRA="--enable-logging-iteration-details $EXTRA"
  else
    echo "WARNING: ITER_DETAILS=1 not applied: image '$IMAGE' lacks the prefill-metrics patch (the stock" >&2
    echo "  flag would log a line per engine step). Rebuild: docker build -t '$IMAGE' ." >&2
  fi
fi

# Wire up the metrics sidecar now that the image has been probed: publish the
# port and pass the in-container port only when the module is really there.
if [ "$METRICS_PORT" != 0 ]; then
  if [ "$METRICS_OK" = 1 ] || [ "$IMAGE_OK" = 0 ]; then
    # IMAGE_OK=0 means "could not tell" — keep the user's intent, the docker
    # run below fails loudly on its own if the image is broken.
    METRICS_PUBLISH=(-p "${BIND_PFX}${METRICS_PORT}:18400")
    MTENV+=(-e VLLM_CUSTOM_METRICS_PORT=18400)
  else
    echo "WARNING: METRICS_PORT=$METRICS_PORT will be IGNORED: image '$IMAGE' predates the" >&2
    echo "  engine-side metrics sidecar (module or its start hooks are missing). Rebuild:" >&2
    echo "  docker build -t '$IMAGE' .   — PLE stats stay in the log meanwhile." >&2
  fi
fi

# PLE stats go to the metrics sidecar by default (PLE_STATS_SEC=0) — but only
# when that sidecar is actually being served. If the module is absent, the
# probe was inconclusive, or the user turned the endpoint off (METRICS_PORT=0),
# keep the log lines (the pre-sidecar behaviour) rather than losing the numbers
# in both places. An explicit PLE_STATS_SEC always wins.
PLE_STATS_DEFAULT=30
{ [ "$METRICS_OK" = 1 ] && [ "$METRICS_PORT" != 0 ]; } && PLE_STATS_DEFAULT=0

# A custom ids.npy path must be visible INSIDE the container — only /model and
# /ple-table are mounted (read-only). Anywhere else the drafter dies at its
# first step with FileNotFoundError; put the file in the checkpoint dir and
# pass /model/<name>.npy.
case "$DRAFT_VOCAB" in
  0|1|/model/*|/ple-table/*) ;;
  *) echo "WARNING: DRAFT_VOCAB='$DRAFT_VOCAB' — the container only sees /model and /ple-table. Put the ids.npy in the checkpoint dir and pass /model/<name>.npy (or bake it into a custom image)." >&2 ;;
esac

# The deterministic MoE alignment from this many tokens a step up (DET_MOE_ALIGN=1);
# auto = one more than the largest decode-only step, SEQS x (1 + MTP) tokens.
DET_MIN="${DET_MOE_ALIGN_MIN_TOKENS:-auto}"
[ "$DET_MIN" = auto ] && DET_MIN=$(( SEQS * (1 + MTP) + 1 ))
case "$DET_MIN" in ''|*[!0-9]*) echo "DET_MOE_ALIGN_MIN_TOKENS='$DET_MIN': a number of tokens or auto" >&2; exit 1 ;; esac

# The container's environment, as NAME=value: EXTRA_ENV (space-separated
# NAME=value pairs) replaces a value of the same name instead of adding a second
# one (docker keeps the first of two).
ENVS=(
  VLLM_PLE_MMAP=1 "VLLM_PLE_MMAP_WORKERS=${WORKERS:-32}" "VLLM_PLE_MMAP_PREWARM=$PREWARM"
  "VLLM_PLE_MMAP_PREFETCH=${PLE_PREFETCH:-0}" "VLLM_PLE_MMAP_MADV_RANDOM=${PLE_MADV_RANDOM:-1}"
  "VLLM_PLE_MMAP_CHUNK=${PLE_CHUNK:-2048}" "VLLM_PLE_MMAP_FAST_ROWS=${PLE_FAST_ROWS:-512}"
  "VLLM_PLE_MMAP_STATS_SEC=${PLE_STATS_SEC:-$PLE_STATS_DEFAULT}"
  "VLLM_HIT_DEBUG=${HIT_DEBUG:-0}" "VLLM_HIT_DEBUG_N=${HIT_DEBUG_N:-0}" "VLLM_STEP_PROFILE=${STEP_PROFILE:-0}"
  VLLM_PLE_MMAP_DIR=/ple-table
)
for e in ${MTENV[@]+"${MTENV[@]}"}; do [ "$e" = -e ] || ENVS+=("$e"); done
ENVS+=(
  "VLLM_MARLIN_USE_ATOMIC_ADD=${MARLIN_ATOMIC_ADD:-1}" "VLLM_DET_MOE_ALIGN=${DET_MOE_ALIGN:-0}"
  "VLLM_DET_MOE_ALIGN_MIN_TOKENS=$DET_MIN" "VLLM_IDLE_EMPTY_CACHE_SEC=${IDLE_RELEASE_SEC:-0}"
  "VLLM_FP8_HYBRID=${FP8_HYBRID:-1}" VLLM_USE_DEEP_GEMM=0
  VLLM_USE_FLASHINFER_SAMPLER=1 "VLLM_ALLOW_LONG_MAX_MODEL_LEN=$ALLOW_LONG" "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
)
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list, split on purpose
EXTRA_ENVS=(${EXTRA_ENV:-})
for e in ${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}; do
  for i in "${!ENVS[@]}"; do [ "${ENVS[$i]%%=*}" = "${e%%=*}" ] && unset 'ENVS[i]'; done
  ENVS+=("$e")
done
ENV_ARGS=()
for e in "${ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done
CPU_ARGS=()
[ -n "${CPUSET:-}" ] && CPU_ARGS=(--cpuset-cpus "$CPUSET")

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart "${RESTART:-unless-stopped}" ${CPU_ARGS[@]+"${CPU_ARGS[@]}"} \
  --gpus all --ipc=host --shm-size 16g -p "${BIND_PFX}${PORT}:8000" ${METRICS_PUBLISH[@]+"${METRICS_PUBLISH[@]}"} \
  --log-opt max-size="$LOG_MAX_SIZE" --log-opt max-file="$LOG_MAX_FILE" \
  --health-cmd "$HEALTH_CMD" --health-start-period=10m \
  --health-interval=60s --health-timeout=15s --health-retries=10 \
  -v "$MODEL_DIR:/model:ro" -v "$TABLE_DIR:/ple-table:ro" \
  "${ENV_ARGS[@]}" \
  "$IMAGE" \
  /model --served-model-name "${SERVED_NAMES[@]}" \
    --host 0.0.0.0 --port 8000 --load-format "${LOAD_FORMAT:-fastsafetensors}" \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $PC_ARG --enable-prompt-tokens-details --enable-chunked-prefill --max-num-batched-tokens "${BATCHED_TOKENS:-8192}" \
    $CC \
    $AT_ARG \
    --kv-cache-dtype auto \
    ${OVR_ARGS[@]+"${OVR_ARGS[@]}"} \
    $EXTRA \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3 \
    ${PIN_ARG[@]+"${PIN_ARG[@]}"} ${SPEC[@]+"${SPEC[@]}"}

echo ">> $NAME starting on ${BIND_ADDR:-0.0.0.0}:$PORT as ${SERVED_NAMES[*]} (ctx $CTX, yarn=$YARN, mtp=$MTP, seqs=$SEQS, chunk=${BATCHED_TOKENS:-8192}, gpu_mem=$GPU_MEM, kv=${KV_BYTES:-auto}, draft_vocab=$DRAFT_VOCAB, draft_head=$DRAFT_HEAD, det_topk=$DET_TOPK, exact_topk=$EXACT_TOPK, det_moe_align=${DET_MOE_ALIGN:-0} (from $DET_MIN tokens), idle_release=${IDLE_RELEASE_SEC:-0}s, metrics=$METRICS_PORT, cpus=${CPUSET:-all}, restart=${RESTART:-unless-stopped})"
echo ">> follow with: docker logs -f $NAME"
