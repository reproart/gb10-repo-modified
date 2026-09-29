# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Ornith-1.5-35B-A3B (a Qwen3.5-35B-A3B finetune: hybrid GDN +
# softmax attention, 256-expert MoE, ~3B active, multimodal, a reasoning
# model). Ornith AI's FP8 checkpoint by default (WEIGHTS), with Ornith AI's
# DFlash draft (SPEC), which doubles single-stream decode. r0b0tlab's NVFP4
# W4A16 checkpoint answers garbage on SGLang 0.5.20 (see "Measured here").
#
#   ./serve.sh ornith-1.5-35b                    # FP8 + DFlash: 83.8 tok/s
#   SPEC=off ./serve.sh ornith-1.5-35b           # no draft: 39.8
#
# Weights, once:
#   hf download ornith-ai/Ornith-1.5-35B-A3B \
#     --local-dir /models/Ornith-1.5-35B-A3B                   # 67 GB, BF16
#   (r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16, 23 GB: WEIGHTS=w4a16)
#   hf download ornith-ai/Ornith-1.5-35B-A3B-DFlash \
#     --local-dir /models/Ornith-1.5-35B-A3B-DFlash           # the draft
# Both MIT. The W4A16 checkpoint is r0b0tlab's community quantization, not
# Ornith AI's; the DFlash draft is Ornith AI's own, trained for the BF16
# target (the verify keeps the target's output either way).
#
# The target's card (r0b0tlab), validated on a DGX Spark with SGLang
# 0.5.6.post3: W4A16 (FP4 weights, BF16 activations) on every Linear;
# lm_head, embeddings, routers, conv1d and the MTP head in BF16; FP8 KV cast.
# Its command: --moe-runner-backend marlin, --attention-backend triton,
# --kv-cache-dtype fp8_e4m3, --mem-fraction-static 0.80, EAGLE with 1 step /
# topk 1 / 2 draft tokens (accept length 1.74), 63-77 tok/s single-request.
# The DFlash card: SGLang 0.5.18, --speculative-algorithm DFLASH with block
# size 8.
#
# SGLang 0.5.20 has what it needs: Qwen3_5MoeForConditionalGeneration and its
# MTP draft (rewritten to Qwen3_5ForCausalLMMTP), ModelOpt W4A16_NVFP4 (dense
# layers on the FP4 Marlin GEMM), DFLASH. Each value is "${VAR:-default}":
#   DRAFT_TOKENS=12 ./serve.sh ornith-1.5-35b
#
# Measured here (2026-09-29), WEIGHTS=fp8 (Ornith AI's FP8 checkpoint), greedy:
#   decode tok/s   accept_len   aggregate at 16 streams
#   SPEC=off              39.8       -            274
#   SPEC=dflash, 8        83.8   4.5-5.5          345   (the default)
#   SPEC=dflash, 12       76.7   4.3-5.4          306
#   SPEC=dflash, 16       75.7   5.7-6.6          266
# Prefill 3.4-6.0K tok/s either way; the GPU reached 83-84 C at 101K.
# The r0b0tlab W4A16 checkpoint (WEIGHTS=w4a16) answered one token repeated
# ("òòòò...") on SGLang 0.5.20, with or without a draft (72 tok/s of
# garbage); every draft looked rejected (accept_len 1.00) because of it.
# NVFP4_SCALES did not change that.

SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Which weights:
#   fp8    ornith-ai/Ornith-1.5-35B-A3B-FP8, Ornith AI's own FP8 checkpoint;
#          its quantization config is read from the checkpoint (the FP8 path
#          Qwen3.8-27B-FP8 runs on this GB10). The default.
#   bf16   ornith-ai/Ornith-1.5-35B-A3B, the original (67 GB), quantized to
#          FP8 at load (--quantization fp8: per-tensor FP8 weights, ~35 GB
#          resident; SGLang 0.5.20 has online FP8 for dense layers and MoE).
#          The DFlash draft was trained against this target.
#   w4a16  r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16. Broken on SGLang 0.5.20:
#          every answer is one token repeated, with or without a draft, and
#          NVFP4_SCALES did not change that. Kept to test a newer SGLang.
#   <path> another checkpoint directory (set QUANTIZATION / MOE_RUNNER_BACKEND)
#   hf download ornith-ai/Ornith-1.5-35B-A3B-FP8 --local-dir /models/Ornith-1.5-35B-A3B-FP8
WEIGHTS="${WEIGHTS:-fp8}"
case "$WEIGHTS" in
  fp8)
    MODEL_DIR="${MODEL_DIR:-/models/Ornith-1.5-35B-A3B-FP8}"
    MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND-}" ;;
  bf16)
    MODEL_DIR="${MODEL_DIR:-/models/Ornith-1.5-35B-A3B}"
    QUANTIZATION="${QUANTIZATION-fp8}"
    MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND-}" ;;
  w4a16)
    MODEL_DIR="${MODEL_DIR:-/models/Ornith-1.5-35B-A3B-NVFP4-W4A16}" ;;
  /*)
    MODEL_DIR="${MODEL_DIR:-$WEIGHTS}" ;;
  *)
    echo "WEIGHTS must be fp8, bf16, w4a16 or an absolute path, not '$WEIGHTS'" >&2
    exit 2 ;;
esac
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ornith-1.5-35b}"

# Speculative decoding:
#   dflash  Ornith AI's DFlash draft (block diffusion: proposes DRAFT_TOKENS - 1
#           tokens at once, the target verifies DRAFT_TOKENS). The 27B
#           profile runs the same algorithm with z-lab's DFlash2.
#   mtp     the checkpoint's BF16 MTP head (NEXTN, served as EAGLE)
#   off     none
if [ "$WEIGHTS" = w4a16 ]; then
  SPEC="${SPEC:-off}"
else
  SPEC="${SPEC:-dflash}"
fi
case "$SPEC" in
  dflash) DRAFT_DIR="${DRAFT_DIR:-/models/Ornith-1.5-35B-A3B-DFlash}" ;;
  mtp|off) DRAFT_DIR="" ;;
  *) echo "SPEC must be dflash, mtp or off, not '$SPEC'" >&2; exit 2 ;;
esac
# DFlash verify width (--speculative-num-draft-tokens; the card's block size
# is 8). On the 27B with DFlash2, 11-16 beat the block size single-stream
# (models/qwen3.8-27b.sh); worth the same sweep here. A value other than the
# draft's block size logs "DFLASH block size mismatch" at boot; harmless.
# Measured on FP8: 8 wins at every level (83.8 single, 345 at 16 streams),
# 12 and 16 accept more tokens per step but lose more to the longer verify.
DRAFT_TOKENS="${DRAFT_TOKENS:-8}"

# Quantization: empty = read from the checkpoint (w4a16: hf_quant_config.json,
# ModelOpt W4A16_NVFP4); bf16 sets fp8 above.
QUANTIZATION="${QUANTIZATION:-}"
# MoE expert kernels. Empty = SGLang's choice (bf16 + online FP8). For w4a16:
# SGLang's auto picks flashinfer_trtllm on Blackwell, which has no NVFP4
# W4A16 MoE path (the card's note, and 0.5.20's ModelOptNvFp4FusedMoEMethod);
# marlin is the W4A16 kernel.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND-marlin}"
# Attention. dflash: flashinfer, what the Qwen3.8-27B profile (same Qwen3.5
# hybrid family) runs with DFlash2 on this GB10. mtp: triton, the card's;
# with flashinfer every MTP proposal was rejected here (accept_len 1.00).
if [ "$SPEC" = mtp ]; then
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
else
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
fi
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

# SPEC=mtp: the checkpoint's BF16 MTP head (NEXTN: SGLang serves it as EAGLE
# with the target as its own draft). The card: 1 step, 2 draft tokens,
# accept length 1.74 (accept rate 0.74).
# More steps are worth a sweep (MTP_STEPS=3 MTP_DRAFT_TOKENS=4, as the
# Flash-Next profile runs); each step also runs the draft's lm_head over
# the whole vocabulary, in BF16, unless DRAFT_VOCAB narrows it.
MTP_STEPS="${MTP_STEPS:-1}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-2}"
# The draft's vocabulary (--speculative-token-map): the 65,536-token map the
# Flash-Next profile uses cut its draft head to a quarter. It holds token ids
# of Qwen3.8's tokenizer, so it only fits if this model's tokenizer is the
# same (Qwen3.5 family, 248K vocab; check before using it):
#   cmp /models/RadixArk/Qwen3.8-Flash-Next-NVFP4/tokenizer.json \
#       /models/Ornith-1.5-35B-A3B-NVFP4-W4A16/tokenizer.json
# A wrong map cannot change answers (the target verifies every token), only
# lower the accept length. Empty = the full vocabulary.
DRAFT_VOCAB="${DRAFT_VOCAB:-}"
# FP8 output heads (patches/gb10_fp8_side.py, converted after the EAGLE
# worker's init_lm_head: SPEC=mtp only; DFlash reads the target's BF16 head
# itself). The target head is read in BF16
# once per verify, a large share of a ~3B-active step:
#   FP8_DRAFT_HEAD=1  the draft's head, when it has its own (DRAFT_VOCAB set);
#                     without a token map the draft calls the target's head
#                     module and this does nothing
#   FP8_HEAD=1        the target head (and with it a shared draft head).
#                     Lossy: measured on Flash-Next without loss (HumanEval
#                     97.0% / 100%); check this model before keeping it.
FP8_DRAFT_HEAD="${FP8_DRAFT_HEAD:-1}"
FP8_HEAD="${FP8_HEAD:-0}"

# Per-shard NVFP4 global scales on the Marlin paths (patches/gb10_nvfp4_scales.py).
# SGLang fuses q/k/v, GDN in_proj_qkv + in_proj_z, and gate + up (shared and
# routed experts), which ModelOpt quantized one by one, each with its own
# global scale; 0.5.20's W4A16 Marlin path keeps one per fused layer (the
# max for dense layers, the gate's for experts) and the rest come out
# scaled wrong. The patch corrects the outputs exactly (column factors for
# dense layers, the down projection's scale for experts). Measured: the
# w4a16 answers stayed garbage with it, so that was not the fault; off.
NVFP4_SCALES="${NVFP4_SCALES:-0}"

# FP8 for the layers the FP8 checkpoint keeps in BF16. Its ignore list:
# every linear_attn.* (the GDN projections in_proj_qkvz, in_proj_ba,
# out_proj), routers, lm_head, MTP, vision. The GDN projections are 37% of
# a decode step on sm80 WMMA BF16 kernels (RESULTS). FP8_SIDE=1 converts
# them at load to FP8 weight-only, one scale per output channel, on SGLang's
# FP8 Marlin GEMM (patches/gb10_fp8_side.py, as the Flash-Next profile does:
# there ~39 -> ~16 ms a step, HumanEval unchanged). Activations stay BF16
# (the checkpoint's own FP8 layers quantize them per token). Measured: 117
# layers, 2.08 -> 1.04 GiB; 83.8 -> 97.6 tok/s single-stream, 71 -> 98 at
# the sweep's first level, 345 -> 353 at 16 streams; "437" right. Lossy:
# off until HumanEval says otherwise.
FP8_SIDE="${FP8_SIDE:-0}"

# Tuned Triton MoE kernel configs. SGLang 0.5.20 ships none for this MoE on
# GB10 ("Using default MoE kernel config ... E=256,N=512,device_name=
# NVIDIA_GB10,dtype=fp8_w8a8,per_channel_quant=True.json" in the boot log),
# and the MoE is ~47% of a decode step. Files tuned on this machine go in
# $ROOT/moe-configs/configs/triton_<version>/ (README, "Ornith"); when that
# directory exists it is SGLANG_MOE_CONFIG_DIR, which replaces SGLang's own
# directory (fine here: this model has one MoE shape).
MOE_CONFIG_DIR="${MOE_CONFIG_DIR:-$ROOT/moe-configs}"

# Concurrency: GDN state slots, 5 per running request with a draft (extra_buffer)
# plus one to keep a finished turn's state for the next one, as in the 27B
# profile. The boot log's mamba pool line says what the cap costs; the KV
# pool is the rest of the memory fraction.
MAX_RUNNING="${MAX_RUNNING:-16}"
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 6))}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-$MAX_RUNNING}"
# GDN state dtype. Empty = SGLang's default (fp32), as the card ran.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"

# 0.80 as the card, and the machine's stable value (qwen3.8-27b.sh explains
# earlyoom at 0.85).
MEM_FRACTION="${MEM_FRACTION:-0.80}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
PREFILL_CUDA_GRAPH="${PREFILL_CUDA_GRAPH:-0}"
# Context: empty = the model's own maximum (config.json: 262,144).
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
# YaRN RoPE scaling past 262K (Ornith's card: validated, factor 4 = ~1M).
# Static in SGLang, so it applies to every request and can cost a little
# quality on ordinary lengths: set it only for workloads that need it, with
# the factor sized to them (2 for ~512K). Empty = off. Sets CONTEXT_LENGTH to
# factor x 262144 unless given.
YARN_FACTOR="${YARN_FACTOR:-}"
if [ -n "$YARN_FACTOR" ] && [ -z "$CONTEXT_LENGTH" ]; then
  CONTEXT_LENGTH=$(awk -v f="$YARN_FACTOR" 'BEGIN { printf "%d", f * 262144 }')
fi

# Parsers as Ornith's card (reasoning in reasoning_content, <tool_call> blocks
# as tool_calls). Sampling: --sampling-defaults model takes the checkpoint's
# generation_config; the card recommends temperature 0.6, top_p 0.95,
# top_k 20 for general use (1.0 to reproduce its benchmarks).
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Environment of the server process, set just before it starts.
model_env() {
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  [ -n "$YARN_FACTOR" ] && export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  if [ -n "$DRAFT_VOCAB" ] && [ ! -f "$DRAFT_VOCAB" ]; then
    echo "DRAFT_VOCAB=$DRAFT_VOCAB: no such file - see models/vocab/README.md" >&2
    exit 1
  fi
  unset GB10_FP8_DRAFT_HEAD GB10_FP8_TARGET_HEAD
  if [ "$SPEC" = mtp ]; then
    [ "$FP8_DRAFT_HEAD" = 1 ] && export GB10_FP8_DRAFT_HEAD=1
    [ "$FP8_HEAD" = 1 ] && export GB10_FP8_TARGET_HEAD=1
  elif [ "$FP8_HEAD" = 1 ]; then
    # the load-time conversion hooks Qwen4-Exp only; DFlash needs a dense head
    echo "FP8_HEAD=1 needs SPEC=mtp in this profile" >&2
    exit 1
  fi
  if [ -d "$MOE_CONFIG_DIR/configs" ]; then
    export SGLANG_MOE_CONFIG_DIR="$MOE_CONFIG_DIR"
  fi
  if [ "$FP8_SIDE" = 1 ]; then
    export GB10_FP8_SIDE=1
    export GB10_FP8_SIDE_TARGET=sglang.srt.models.qwen3_5:Qwen3_5MoeForConditionalGeneration
  else
    unset GB10_FP8_SIDE GB10_FP8_SIDE_TARGET
  fi
  if [ "$NVFP4_SCALES" = 1 ]; then
    export GB10_NVFP4_SCALES=1
  else
    unset GB10_NVFP4_SCALES
  fi
  if [ -n "${GB10_FP8_DRAFT_HEAD:-}${GB10_FP8_TARGET_HEAD:-}${GB10_NVFP4_SCALES:-}${GB10_FP8_SIDE:-}" ]; then
    export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
model_args() {
  args+=(
    --attention-backend "$ATTENTION_BACKEND"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --mamba-radix-cache-strategy extra_buffer
    --max-mamba-cache-size "$MAMBA_CACHE"
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
    --chunked-prefill-size "$CHUNKED_PREFILL"
  )
  [ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
  [ -n "$MOE_RUNNER_BACKEND" ] && args+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  [ -n "$MAMBA_SSM_DTYPE" ] && args+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
  [ "$PREFILL_CUDA_GRAPH" = 1 ] || args+=(--disable-prefill-cuda-graph)
  if [ "$SPEC" = dflash ]; then
    args+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFT_DIR"
      --speculative-num-draft-tokens "$DRAFT_TOKENS"
    )
  elif [ "$SPEC" = mtp ]; then
    args+=(
      --speculative-algorithm NEXTN
      --speculative-num-steps "$MTP_STEPS"
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
    )
    [ -n "$DRAFT_VOCAB" ] && args+=(--speculative-token-map "$DRAFT_VOCAB")
  fi
  if [ -n "$YARN_FACTOR" ]; then
    args+=(--json-model-override-args "{\"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": $YARN_FACTOR, \"original_max_position_embeddings\": 262144}}")
  fi
  [ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  return 0
}

# One line for the startup summary.
model_summary() {
  local vocab=full spec=off
  [ -n "$DRAFT_VOCAB" ] && vocab="$(basename "$DRAFT_VOCAB")"
  case "$SPEC" in
    dflash) spec="DFlash, $DRAFT_TOKENS draft tokens" ;;
    mtp) spec="MTP $MTP_STEPS/1/$MTP_DRAFT_TOKENS, draft vocab $vocab, heads target $([ "$FP8_HEAD" = 1 ] && echo FP8 || echo BF16)" ;;
  esac
  echo "weights $WEIGHTS (quantization ${QUANTIZATION:-from checkpoint}$([ "$NVFP4_SCALES" = 1 ] && echo ", NVFP4 per-shard scales kept")$([ "$FP8_SIDE" = 1 ] && echo ", BF16 GDN projections -> FP8 Marlin")), MoE ${MOE_RUNNER_BACKEND:-auto}, attention $ATTENTION_BACKEND, KV $KV_CACHE_DTYPE$([ -n "$YARN_FACTOR" ] && echo ", YaRN x$YARN_FACTOR")$([ -d "$MOE_CONFIG_DIR/configs" ] && echo ", MoE configs from $MOE_CONFIG_DIR"); $spec; cap $MAX_RUNNING (GDN pool $MAMBA_CACHE)"
}
