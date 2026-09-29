# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Ornith-1.5-35B-A3B (a Qwen3.5-35B-A3B finetune: hybrid GDN +
# softmax attention, 256-expert MoE, ~3B active, multimodal, a reasoning
# model) in NVFP4 W4A16, with speculative decoding by Ornith's DFlash draft
# (default) or the checkpoint's own MTP head.
#
#   ./serve.sh ornith-1.5-35b
#   SPEC=mtp ./serve.sh ornith-1.5-35b
#
# Weights, once:
#   hf download r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 \
#     --local-dir /models/Ornith-1.5-35B-A3B-NVFP4-W4A16      # 23 GB
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
# Measured here (2026-09-29), SPEC=mtp 1/1/2 with flashinfer attention:
# accept_len 1.00 on every request (no proposal ever accepted), so 46.3 tok/s
# single-stream is plain decoding; 326 tok/s aggregate at 16 streams; prefill
# 3.3-7.3K tok/s. The card's run (triton attention) accepted 1.74: triton is
# now the MTP default, and DFlash the default accelerator.

SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

MODEL_DIR="${MODEL_DIR:-/models/Ornith-1.5-35B-A3B-NVFP4-W4A16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ornith-1.5-35b}"

# Speculative decoding:
#   dflash  Ornith AI's DFlash draft (block diffusion: proposes DRAFT_TOKENS - 1
#           tokens at once, the target verifies DRAFT_TOKENS). The 27B
#           profile runs the same algorithm with z-lab's DFlash2.
#   mtp     the checkpoint's BF16 MTP head (NEXTN, served as EAGLE)
#   off     none
SPEC="${SPEC:-dflash}"
case "$SPEC" in
  dflash) DRAFT_DIR="${DRAFT_DIR:-/models/Ornith-1.5-35B-A3B-DFlash}" ;;
  mtp|off) DRAFT_DIR="" ;;
  *) echo "SPEC must be dflash, mtp or off, not '$SPEC'" >&2; exit 2 ;;
esac
# DFlash verify width (--speculative-num-draft-tokens; the card's block size
# is 8). On the 27B with DFlash2, 11-16 beat the block size single-stream
# (models/qwen3.8-27b.sh); worth the same sweep here. A value other than the
# draft's block size logs "DFLASH block size mismatch" at boot; harmless.
DRAFT_TOKENS="${DRAFT_TOKENS:-8}"

# Quantization: empty = read from the checkpoint (hf_quant_config.json,
# ModelOpt W4A16_NVFP4), as the card does.
QUANTIZATION="${QUANTIZATION:-}"
# MoE expert kernels. SGLang's auto picks flashinfer_trtllm on Blackwell,
# which has no NVFP4 W4A16 MoE path (the card's note, and 0.5.20's
# ModelOptNvFp4FusedMoEMethod); marlin is the W4A16 kernel.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
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
# Context: empty = the model's own maximum (config.json). The card tested 32K.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"

REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Environment of the server process, set just before it starts.
model_env() {
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
  if [ -n "${GB10_FP8_DRAFT_HEAD:-}${GB10_FP8_TARGET_HEAD:-}" ]; then
    export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
model_args() {
  args+=(
    --moe-runner-backend "$MOE_RUNNER_BACKEND"
    --attention-backend "$ATTENTION_BACKEND"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --mamba-radix-cache-strategy extra_buffer
    --max-mamba-cache-size "$MAMBA_CACHE"
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
    --chunked-prefill-size "$CHUNKED_PREFILL"
  )
  [ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
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
  echo "NVFP4 W4A16 (${QUANTIZATION:-from checkpoint}), MoE $MOE_RUNNER_BACKEND, attention $ATTENTION_BACKEND, KV $KV_CACHE_DTYPE; $spec; cap $MAX_RUNNING (GDN pool $MAMBA_CACHE)"
}
