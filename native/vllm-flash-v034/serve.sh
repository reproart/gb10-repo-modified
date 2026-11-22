#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on vLLM 0.34 (./build.sh and ./prepare.sh once): the one-server
# checks, the checkpoint view refreshed, memory prepared, the container, a wait until the API
# answers. Rerun to restart with new settings.
#   ./serve.sh
#   MTP=3 SEQS=16 ./serve.sh            any setting from config.sh
# Then, as for every server here (the benches find :8000 by themselves):
#   python3 ../bench/perf.py --only decode decode-ru prefill
set -euo pipefail
trap 'echo "serve.sh: stopped at line $LINENO (exit $?)" >&2' ERR
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_docker

# One server at a time: each wants most of the 128 GB.
pgrep -f "sglang.launch_server" >/dev/null 2>&1 && {
  echo "an SGLang server is running: stop it first (sudo systemctl stop gb10-sglang, or Ctrl-C it)" >&2; exit 1; }
for c in qwen38-flash gb10-vllm-inkling gb10-sglang-inkling gb10-vllm-b12x tfv41-r0 gb10-vllm-dsv41 glm53-flash-tf; do
  docker ps --format '{{.Names}}' | grep -qx "$c" && { echo "container $c runs: stop it first" >&2; exit 1; }
done
if ! docker ps --format '{{.Names}}' | grep -qx "$NAME" &&
   [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null)" != 000 ]; then
  echo "something already serves :$PORT" >&2; exit 1
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE: ./build.sh first" >&2; exit 1; }
for d in "$MODEL_DIR" "$TABLE_DIR"; do
  [ -d "$d" ] || { echo "no $d: ./prepare.sh first" >&2; exit 1; }
done
# The view follows the folders as they are now (cheap: links and two JSON files).
python3 "$V034_ROOT/src/make_model_view.py" "$MODEL_DIR" "$TABLE_DIR" "$VIEW_DIR"

# The drafter's own head: the id set mounted read-only at /opt/llm/ (the image has none).
DRAFT_ENVS=() VOCAB_MOUNT=()
case "$DRAFT_VOCAB" in
  0|"") ;;
  1) vocab_src="$NATIVE_DIR/vllm-flash-ar/src/draft_vocab_65536.npy" ;;
  ru) vocab_src="$NATIVE_DIR/vllm-flash-ar/src/draft_vocab_ru_65536.npy" ;;
  ru-wide) vocab_src="$NATIVE_DIR/vllm-flash-ar/src/draft_vocab_ru_81920.npy" ;;
  *) vocab_src="$DRAFT_VOCAB" ;;
esac
if [ "${DRAFT_VOCAB:-0}" != 0 ]; then
  [ -f "$vocab_src" ] || { echo "DRAFT_VOCAB=$DRAFT_VOCAB: no file $vocab_src" >&2; exit 1; }
  vocab_src="$(cd "$(dirname "$vocab_src")" && pwd)/$(basename "$vocab_src")"
  VOCAB_MOUNT=(-v "$vocab_src:/opt/llm/$(basename "$vocab_src"):ro")
  DRAFT_ENVS+=("VLLM_MTP_DRAFT_VOCAB=/opt/llm/$(basename "$vocab_src")")
fi
case "$DRAFT_HEAD" in
  int8) ;;
  int4|bf16) DRAFT_ENVS+=("VLLM_MTP_DRAFT_HEAD=$DRAFT_HEAD") ;;
  *) echo "DRAFT_HEAD=$DRAFT_HEAD: int8, int4 or bf16" >&2; exit 1 ;;
esac
[ "$DRAFT_HEAD" = bf16 ] && [ "${DRAFT_VOCAB:-0}" = 0 ] && {
  echo "DRAFT_HEAD=bf16 is a DRAFT_VOCAB slice in bf16: set DRAFT_VOCAB too" >&2; exit 1; }
DRAFT_ON=0
[ "$MTP" != 0 ] && [ ${#DRAFT_ENVS[@]} -gt 0 ] && DRAFT_ON=1
image_label() { docker image inspect -f "{{index .Config.Labels \"$1\"}}" "$IMAGE" 2>/dev/null || true; }
if [ "$DRAFT_ON" = 1 ] && [ "$(image_label gb10.draft-head)" != 1 ]; then
  echo "image $IMAGE has no draft-head hook (built before stage 1a): ./build.sh first, or DRAFT_VOCAB=0" >&2
  exit 1
fi
case "$HC_FP8" in
  0) ;;
  1|up)
    [ "$(image_label gb10.hc-fp8)" = 1 ] || {
      echo "image $IMAGE has no FP8 HC hook (built before stage 1b): ./build.sh first, or HC_FP8=0" >&2; exit 1; } ;;
  *) echo "HC_FP8=$HC_FP8: 0, 1 or up" >&2; exit 1 ;;
esac
DET_MIN="$DET_MOE_ALIGN_MIN_TOKENS"
[ "$DET_MIN" = auto ] && DET_MIN=$(( SEQS * (1 + MTP) + 1 ))
case "$DET_MIN" in ''|*[!0-9]*) echo "DET_MOE_ALIGN_MIN_TOKENS=$DET_MIN: a number of tokens or auto" >&2; exit 1 ;; esac
case "$DET_MOE_ALIGN$QSA_DET_TOPK$QSA_EXACT_TOPK" in
  000) ;;
  *[!01]*|????*) echo "DET_MOE_ALIGN=$DET_MOE_ALIGN QSA_DET_TOPK=$QSA_DET_TOPK QSA_EXACT_TOPK=$QSA_EXACT_TOPK:" \
      "each 0 or 1" >&2; exit 1 ;;
  *) det_label="$(image_label gb10.det)"
     if [ "$QSA_DET_TOPK" = 1 ] && [ "$det_label" != 2 ]; then
       echo "image $IMAGE has no deterministic top-k kernel wiring (built before stage 3c): ./build.sh first," \
         "or QSA_DET_TOPK=0" >&2; exit 1
     fi
     case "$det_label" in 1|2) ;; *)
       echo "image $IMAGE has no determinism patches (built before stage 3b): ./build.sh first," \
         "or DET_MOE_ALIGN=0 QSA_DET_TOPK=0 QSA_EXACT_TOPK=0" >&2; exit 1 ;; esac ;;
esac
[ "$(image_label gb10.tool-parser)" = 1 ] ||
  log "NOTE: image $IMAGE predates the tool-call parser fixes (stage 3a): a tool-call example in a reply" \
    "can still turn into a call. ./build.sh"
# The ablation: 0 = off, 1 = ABLATE_VECTORS, a path = that file; its directory read-only at
# /ablate. Asked for but not possible stops the start: the server would answer as the stock model.
ABL_FILE="" ABL_MOUNT=() ABL_ENVS=()
case "$ABLATE" in
  0|"") ;;
  1) ABL_FILE="$ABLATE_VECTORS" ;;
  *) ABL_FILE="$ABLATE" ;;
esac
if [ -n "$ABL_FILE" ]; then
  if [ ! -f "$ABL_FILE" ]; then
    echo "ABLATE=$ABLATE: no vectors at $ABL_FILE" >&2
    [ "$ABLATE" = 1 ] && echo "  download them once: hf download Mambavtt/qwen3.8-flash-next-refusal-ablation-vectors" \
      "--revision 7b009deb2d3f21c5bb199366cb39e3ed129b4207 --local-dir $(dirname "$ABL_FILE")" >&2
    exit 1
  fi
  [ "$(image_label gb10.ablate)" = 1 ] || {
    echo "image $IMAGE has no ablation hook (built before stage 2): ./build.sh first, or ABLATE=0" >&2; exit 1; }
  ABL_MOUNT=(-v "$(cd "$(dirname "$ABL_FILE")" && pwd):/ablate:ro")
  ABL_ENVS=("VLLM_ABLATE=/ablate/$(basename "$ABL_FILE")" "VLLM_ABLATE_SOURCE=$ABLATE_SOURCE"
            "VLLM_ABLATE_ALPHA=$ABLATE_ALPHA" "VLLM_ABLATE_LAYERS=$ABLATE_LAYERS" "VLLM_ABLATE_MTP=$ABLATE_MTP")
fi

if [ "$MEM_PREP" = 1 ]; then "$NATIVE_DIR/scripts/mem-prep.sh" prep || true; fi

read -r -a SERVED_NAMES <<< "$SERVED_NAME"
ARGS=(/model-view --served-model-name "${SERVED_NAMES[@]}" --host 0.0.0.0 --port 8000
  --load-format "$LOAD_FORMAT" --engram-config '{"disk_offload": true}'
  --max-model-len "$CTX" --max-num-seqs "$SEQS" --kv-cache-memory-bytes "$KV_BYTES"
  --enable-chunked-prefill --max-num-batched-tokens "$BATCHED_TOKENS"
  --enable-prompt-tokens-details --kv-cache-dtype auto
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3)
if [ "$PREFIX_CACHE" = 1 ]; then ARGS+=(--enable-prefix-caching); else ARGS+=(--no-enable-prefix-caching); fi
[ "$MTP" != 0 ] && ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
[ -n "$MAMBA_SSM_DTYPE" ] && ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_DTYPE")
# shellcheck disable=SC2206  # EXTRA is a flag list, split on purpose
ARGS+=(${EXTRA:-})

QSA_TOPK_DESC=stock
[ "$QSA_DET_TOPK" = 1 ] && QSA_TOPK_DESC=deterministic
[ "$QSA_EXACT_TOPK" = 1 ] && QSA_TOPK_DESC=exact
ABL_DESC=off
[ -n "$ABL_FILE" ] && ABL_DESC="$(basename "$ABL_FILE") (source $ABLATE_SOURCE, alpha $ABLATE_ALPHA, layers $ABLATE_LAYERS, mtp $ABLATE_MTP)"
ENVS=(VLLM_FP8_HYBRID=1 "VLLM_HC_FP8=$HC_FP8" "VLLM_DET_MOE_ALIGN=$DET_MOE_ALIGN"
  "VLLM_DET_MOE_ALIGN_MIN_TOKENS=$DET_MIN" "VLLM_QSA_DET_TOPK=$QSA_DET_TOPK" "VLLM_QSA_EXACT_TOPK=$QSA_EXACT_TOPK" ${DRAFT_ENVS[@]+"${DRAFT_ENVS[@]}"} ${ABL_ENVS[@]+"${ABL_ENVS[@]}"})
[ -n "$API_KEY" ] && ENVS+=("VLLM_API_KEY=$API_KEY")
# shellcheck disable=SC2206  # EXTRA_ENV is a NAME=value list
EXTRA_ENVS=(${EXTRA_ENV:-})
for e in ${EXTRA_ENVS[@]+"${EXTRA_ENVS[@]}"}; do
  for i in "${!ENVS[@]}"; do [ "${ENVS[$i]%%=*}" = "${e%%=*}" ] && unset 'ENVS[i]'; done
  ENVS+=("$e")
done
ENV_ARGS=()
for e in "${ENVS[@]}"; do ENV_ARGS+=(-e "$e"); done
BIND_PFX="${BIND_ADDR:+$BIND_ADDR:}"
HEALTH_CMD="python3 -c 'import urllib.request;urllib.request.urlopen(\"http://localhost:8000/health\",timeout=10)'"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart "$RESTART" \
  --gpus all --ipc=host --shm-size 16g -p "${BIND_PFX}${PORT}:8000" \
  --log-opt max-size=10m --log-opt max-file=3 \
  --health-cmd "$HEALTH_CMD" --health-start-period=15m --health-interval=60s --health-timeout=15s --health-retries=10 \
  -v "$MODEL_DIR:/model:ro" -v "$TABLE_DIR:/ple-table:ro" -v "$VIEW_DIR:/model-view:ro" \
  ${VOCAB_MOUNT[@]+"${VOCAB_MOUNT[@]}"} ${ABL_MOUNT[@]+"${ABL_MOUNT[@]}"} \
  "${ENV_ARGS[@]}" "$IMAGE" "${ARGS[@]}" >/dev/null
log "$NAME starting on ${BIND_ADDR:-0.0.0.0}:$PORT as ${SERVED_NAMES[*]} (vLLM $BASE_TAG, ctx $CTX, mtp $MTP, seqs $SEQS," \
  "chunk $BATCHED_TOKENS, kv $KV_BYTES, prefix cache $PREFIX_CACHE, ssm ${MAMBA_SSM_DTYPE:-checkpoint}," \
  "draft vocab ${DRAFT_VOCAB:-0}, draft head $DRAFT_HEAD, hc fp8 $HC_FP8, det moe align $DET_MOE_ALIGN" \
  "(from $DET_MIN tokens), qsa top-k $QSA_TOPK_DESC, ablate $ABL_DESC, table from disk)"

[ "$WAIT_MIN" = 0 ] && exit 0
log "waiting for http://127.0.0.1:$PORT/v1/models (the first boot of an image compiles kernels)"
auth=()
[ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
for _ in $(seq 1 $((WAIT_MIN * 6))); do
  if curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "serving ${SERVED_NAMES[0]} on :$PORT; logs: docker logs -f $NAME"
    # both are set up in the profiling run, before the API answers
    boot_log="$(docker logs "$NAME" 2>&1 || true)"
    if [ "$DRAFT_ON" = 1 ]; then
      head_line="$(grep "MTP private draft head:" <<< "$boot_log" | head -1 || true)"
      if [ -n "$head_line" ]; then
        log "draft head: ${head_line##*MTP private draft head: }"
      else
        log "WARNING: no 'MTP private draft head:' line in the boot log, so the drafter scores the full head;" \
          "docker logs $NAME 2>&1 | grep -i 'draft head'"
      fi
    fi
    if [ "$HC_FP8" != 0 ]; then
      hc_line="$(grep "FP8 HC: [0-9]" <<< "$boot_log" | head -1 || true)"
      if [ -n "$hc_line" ]; then
        log "hc fp8: ${hc_line##*FP8 HC: }"
      else
        log "WARNING: no 'FP8 HC:' line in the boot log, so the hyper-connection projections may be BF16;" \
          "docker logs $NAME 2>&1 | grep 'FP8 HC'"
      fi
      grep -q "TORCH PATH\|kernel failed" <<< "$boot_log" &&
        log "WARNING: an FP8 HC kernel fell back to torch (slower than BF16): docker logs $NAME 2>&1 | grep 'FP8 HC'"
    fi
    if [ "$QSA_TOPK_DESC" != stock ]; then
      topk_line="$(grep "QSA top-k:" <<< "$boot_log" | tail -1 || true)"
      if [ -n "$topk_line" ]; then
        log "qsa top-k: ${topk_line##*QSA top-k: }"
        grep -q "deterministic kernel did not load" <<< "$boot_log" &&
          log "WARNING: the deterministic top-k kernel did not load: the exact top-k runs (deterministic, slower" \
            "long prefill); docker logs $NAME 2>&1 | grep 'QSA top-k'"
      else
        log "WARNING: no 'QSA top-k:' line in the boot log; docker logs $NAME 2>&1 | grep -i 'top-k'"
      fi
    fi
    if [ -n "$ABL_FILE" ]; then
      abl_line="$(grep "ABLATION ON" <<< "$boot_log" | head -1 || true)"
      if [ -n "$abl_line" ]; then
        log "ablation: ${abl_line##*ABLATION ON (VLLM_ABLATE): }"
        grep -q "MTP draft layer [0-9]* projects" <<< "$boot_log" && log "ablation: the MTP draft projects its inputs too"
      else
        log "WARNING: ABLATE=$ABLATE but no 'ABLATION ON' line in the boot log: the answers may be the stock" \
          "model's; docker logs $NAME 2>&1 | grep ABLATION"
      fi
    fi
    exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]; then
    log "$NAME stopped during boot:"; docker logs --tail 80 "$NAME" 2>&1; exit 1
  fi
  sleep 10
done
log "not answering after $WAIT_MIN min; still loading? docker logs -f $NAME"
exit 1
