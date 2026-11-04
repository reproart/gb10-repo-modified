# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: inclusionAI Ling-3.0-flash on one DGX Spark: 124B MoE (5.1B active;
# 512 routed experts, top-8, plus a shared one) over a hybrid backbone (35 Kimi
# Delta Attention linear layers + 7 gated MLA layers), with its built-in MTP
# layer as the draft (NEXTN). A thinking model with tool calls; text only (the
# VL sibling, models/ling-3.0-flash-vl.sh, has no MTP layer).
#
#   scripts/mem-prep.sh idle &&   # no 65 GB download beside a running server (earlyoom kills the server first)
#     hf download inclusionAI/Ling-3.0-flash-fp4 --local-dir /models/inclusionAI/Ling-3.0-flash-fp4
#   ./serve.sh ling-3.0-flash install     # SGLang 0.5.21, the same venv as the VL and Qwen profiles
#   ./serve.sh ling-3.0-flash
#
# The repository has no generation_config.json, so --sampling-defaults model
# finds nothing and an OpenAI request without sampling fields runs at
# temperature 1.0. The model card's values (temperature 0.6, top_p 0.95,
# top_k 20) become the server's defaults with one file next to the weights:
#   printf '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}\n' \
#     > /models/inclusionAI/Ling-3.0-flash-fp4/generation_config.json
#
# Sources:
# - NVIDIA, "3 Ways to Boost MoE Inference on NVIDIA DGX Spark" (2026-10),
#   Ling-3.0-flash-fp4 on one Spark, one stream: 36.1 tok/s for the cookbook's
#   recipe, 39.5 with the Humming MoE kernels, 43.9 with an FP8 lm_head on top,
#   102.9 with MTP (3 steps, top-k 1, 4 draft tokens) on top: a repetitive
#   greedy 32K-in / 2K-out test, 3.90 tokens accepted a step. On GSM8K
#   (sampled, chain of thought): 37.8 -> 53.9 tok/s, 2.68 accepted, accuracy
#   96.89% -> 96.82%.
# - inclusionAI's DGX Spark notebooks for those runs (ling-cookbook,
#   guide/local-deploy/ling-3.0-flash/dgx-spark-sglang-*.ipynb): their own
#   SGLang branches (ling_v3_support_mxfp4[_humming]) for the Humming adapter,
#   the FP8 head and the draft sharing it; the int4 one runs upstream SGLang
#   with NEXTN, 8 requests, a 32-slot mamba cache, mem-fraction 0.75.
# - The SGLang cookbook page (docs.sglang.io, InclusionAI/Ling-3.0-flash):
#   native context 256K, read from the checkpoint; no Spark cells.
#
# What this profile runs on stock SGLang 0.5.21 (checked in the wheel):
#   MTP     bailing_moe_nextn.py drafts with the checkpoint's layer 42. 0.5.21
#           does not point the draft at the target checkpoint for this
#           architecture (BailingMoeV3ForCausalLM is missing from its NEXTN
#           list), so the profile passes --speculative-draft-model-path itself,
#           as inclusionAI's notebook does. The KDA state cache resolves to
#           extra_buffer on its own (the strategy speculative decoding needs).
#   FP8 head the lm_head is BF16 [157,184 x 2560] (the checkpoint keeps it out
#           of FP8), read once per verify and once per draft step. This
#           project's patches/gb10_fp8_side.py converts it to FP8 weight-only
#           (Marlin) right after the draft worker's init_lm_head, the step
#           NVIDIA's write-up says needs care: the Ling draft holds the target's
#           head tensor in a module of its own (set_embed_and_head), and the
#           patch gives that module the converted weights and kernel too
#           ("shared" in the log), so the draft cannot read FP8 bytes as BF16
#           (their symptom: accept length 1.00, ~24 tok/s).
#   Humming not with this model: Ling clamps the experts' SwiGLU after SiLU in
#           layers 35-41 (expert_swiglu_limit_list: 4; gemm1_clamp_limit in
#           SGLang), and the Humming runner of 0.5.21 and 0.5.22 only knows the
#           other clamp (swiglu_limit), so those layers would run unclamped.
#           The same holds for Marlin (it takes gemm1_clamp_limit only with a
#           gemm1_alpha). flashinfer_mxfp4 (SwigluStep) and triton apply it.
#           The profile refuses humming and marlin for that reason.
#
# Measured here (2026-11-04, GX10, perf.py: code prompt, greedy, thinking off):
#   MTP=0, FP8 head      decode 44.5 / 44.7 tok/s (two boots; NVIDIA: 43.9 for
#                        the cookbook recipe + FP8 head)
#   MTP=1, FP8 heads     decode 23.6 tok/s, accept length 1.02-1.07
#   MTP=1, BF16 heads    decode 19.3 tok/s, accept length 1.02-1.12
#   So MTP drafts are rejected on stock 0.5.21 whatever the head: the log said
#   the draft shared the target's FP8 head ("shares the FP8 copy"), the
#   checkpoint's layer 42 is complete and loads by name, and BF16 heads (the
#   patch out of the way) changed nothing. NVIDIA's 102.9 tok/s with MTP ran
#   inclusionAI's SGLang branch. MTP is off by default until a release drafts
#   with this model; MTP=1 still runs it (and costs ~half the speed).
#   With MTP=1 (triton attention, the default here): prefill 3019 / 3479 / 1222
#   / 442 tok/s at 2K / 8K / 32K / 101K; 22.6 / 33.6 / 46.4 / 63.3 tok/s
#   aggregate at 1 / 2 / 4 / 8 streams.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Which weights (MODEL_DIR, if set, wins):
#   fp4   inclusionAI/Ling-3.0-flash-fp4 (65.6 GiB): MXFP4 experts (e8m0, group
#         32) + FP8 e4m3 128x128 elsewhere; MoE on flashinfer_mxfp4, SGLang's own
#         choice for this mixed checkpoint on SM12x (the SM121 CUTLASS W4A8 path).
#         The default. Revision 3bae1cf4 (2026-08-20) when this was written.
#   int4  inclusionAI/Ling-3.0-flash-int4 (71.8 GiB): GPTQ W4A16 group 32
#         (compressed-tensors); MoE on triton, see MOE_RUNNER_BACKEND.
WEIGHTS="${WEIGHTS:-fp4}"
case "$WEIGHTS" in
  fp4)  _dir=/models/inclusionAI/Ling-3.0-flash-fp4 ;;
  int4) _dir=/models/inclusionAI/Ling-3.0-flash-int4 ;;
  *) echo "WEIGHTS must be fp4 or int4, not '$WEIGHTS'" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$_dir}"
DRAFT_DIR=""   # the MTP layer is in MODEL_DIR
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ling-3.0-flash}"
unset _dir

# MoE expert kernels. Empty = the profile's choice: SGLang's own for fp4
# (flashinfer_mxfp4), triton for int4 (SGLang's auto pick on GB10 is Marlin,
# which drops the clamp above; on B200/B300 auto is triton already). humming and
# marlin are refused (see the header).
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"
if [ -z "$MOE_RUNNER_BACKEND" ] && [ "$WEIGHTS" = int4 ]; then MOE_RUNNER_BACKEND=triton; fi

# Speculative decoding with the checkpoint's MTP layer (NEXTN): 1 = on, 0 = off.
# Off by default: on stock SGLang 0.5.21 the drafts are rejected (accept length
# ~1.05, decode 19-24 tok/s against 44.5 without; see "Measured here" above).
# NVIDIA's and inclusionAI's Spark setting, for MTP=1: 3 steps, top-k 1, 4 draft tokens.
MTP="${MTP:-0}"
MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-4}"

# FP8 weight-only for the output heads (patches/gb10_fp8_side.py):
# FP8_HEAD=1        the target head, which decides every emitted token. Lossy in
#                   principle; NVIDIA measured GSM8K 96.89% -> 96.82% with their
#                   dynamic-FP8 head. With MTP the draft shares it. On by default.
# FP8_DRAFT_HEAD=1  with FP8_HEAD=0 and MTP: the draft's own FP8 copy of the head
#                   (+0.4 GiB); a worse draft costs acceptance, never an answer.
# Unmeasured on this model here: check answers, not only tok/s.
FP8_HEAD="${FP8_HEAD:-1}"
FP8_DRAFT_HEAD="${FP8_DRAFT_HEAD:-1}"
# --enable-fp32-lm-head for a head left in BF16 (logits accumulated in FP32), as
# every inclusionAI recipe runs it. An FP8 head goes through its own kernel and
# gives BF16 logits either way.
FP32_LM_HEAD="${FP32_LM_HEAD:-1}"

# Context: native 262144 (max_position_embeddings, no rope_scaling; trained
# 8K -> 32K -> 256K). inclusionAI's Spark notebooks add the VL model's YaRN x2
# override on top; the SGLang cookbook and the model card's recipe do not, and
# neither does this profile.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

# Memory: 0.75 as inclusionAI's int4 notebook (their FP8-head notebook goes to
# 0.68, against lockups while kernels compile beside the weights loading; the
# launcher's JIT_JOBS=2 caps those compiles). KV is small (7 MLA layers, ~8 KiB
# a token); each request holds a fixed KDA state, so the cap is what
# concurrency costs. Empty MAMBA_CACHE = SGLang's sizing.
MEM_FRACTION="${MEM_FRACTION:-0.75}"
if [ "$MTP" = 1 ]; then
  MAX_RUNNING="${MAX_RUNNING:-8}"
else
  MAX_RUNNING="${MAX_RUNNING:-16}"
fi
MAMBA_CACHE="${MAMBA_CACHE:-}"
# Empty = SGLang's choice (inclusionAI's notebooks: 8192 / 64 / flashinfer).
CHUNKED_PREFILL="${CHUNKED_PREFILL:-}"
PAGE_SIZE="${PAGE_SIZE:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

# Thinking is on by default in the chat template; per request:
# "chat_template_kwargs": {"enable_thinking": false}.
REASONING_PARSER="${REASONING_PARSER:-ling3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-ling3}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

model_env() {
  case "$MOE_RUNNER_BACKEND" in
    humming|marlin)
      echo "MOE_RUNNER_BACKEND=$MOE_RUNNER_BACKEND would run Ling's experts in layers 35-41 without" \
        "their SwiGLU clamp (SGLang $SGLANG_VERSION passes it to flashinfer_mxfp4 and triton only):" \
        "see models/$PROFILE.sh" >&2
      exit 1 ;;
  esac
  # patches/sitecustomize.py installs the head conversion in every Python
  # process the server starts. With MTP it runs after the draft worker's
  # init_lm_head (model-independent); without MTP at the target's load, which
  # needs the target class named.
  unset GB10_FP8_DRAFT_HEAD GB10_FP8_TARGET_HEAD GB10_FP8_SIDE_TARGET
  if [ "$MTP" = 1 ]; then
    [ "$FP8_DRAFT_HEAD" = 1 ] && export GB10_FP8_DRAFT_HEAD=1
    [ "$FP8_HEAD" = 1 ] && export GB10_FP8_TARGET_HEAD=1
  elif [ "$FP8_HEAD" = 1 ]; then
    export GB10_FP8_TARGET_HEAD=load
    export GB10_FP8_SIDE_TARGET=sglang.srt.models.bailing_moe_v3:BailingMoeV3ForCausalLM
  fi
  if [ -n "${GB10_FP8_DRAFT_HEAD:-}${GB10_FP8_TARGET_HEAD:-}" ]; then
    export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

model_args() {
  args+=(
    --tp 1
    --max-running-requests "$MAX_RUNNING"
  )
  [ -n "$MOE_RUNNER_BACKEND" ] && args+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  [ -n "$MAMBA_CACHE" ] && args+=(--max-mamba-cache-size "$MAMBA_CACHE")
  [ -n "$CHUNKED_PREFILL" ] && args+=(--chunked-prefill-size "$CHUNKED_PREFILL")
  [ -n "$PAGE_SIZE" ] && args+=(--page-size "$PAGE_SIZE")
  [ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")
  [ "$FP32_LM_HEAD" = 1 ] && args+=(--enable-fp32-lm-head)
  if [ "$MTP" = 1 ]; then
    args+=(
      --speculative-algorithm NEXTN
      --speculative-draft-model-path "$MODEL_DIR"
      --speculative-num-steps "$MTP_STEPS"
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
    )
  fi
  [ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  return 0
}

model_summary() {
  local head=BF16
  [ "$FP8_HEAD" = 1 ] && head=FP8
  [ "$FP8_HEAD" != 1 ] && [ "$MTP" = 1 ] && [ "$FP8_DRAFT_HEAD" = 1 ] && head="BF16 (draft FP8)"
  echo "Ling-3.0-flash $WEIGHTS ($([ "$WEIGHTS" = fp4 ] && echo "MXFP4 experts + FP8" || echo "GPTQ INT4 W4A16")), MoE ${MOE_RUNNER_BACKEND:-auto}; $([ "$MTP" = 1 ] && echo "MTP $MTP_STEPS steps / $MTP_DRAFT_TOKENS tokens" || echo "no MTP"); lm_head $head; context $CONTEXT_LENGTH; cap $MAX_RUNNING"
}
