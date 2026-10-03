# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-Flash-Next (176B: 125B MoE + 51B n-gram table, 6B active)
# on one DGX Spark, NVFP4, with the in-checkpoint MTP head.
#
#   ./serve.sh qwen3.8-flash-next
#
# From the SGLang cookbook's single-Spark cell (docs/cookbook/autoregressive/
# Qwen/Qwen3.8-Flash-Next.mdx, "DGX Spark notes"): 27.5 tok/s single-stream
# with MTP, 8 concurrent requests, a ~93K-token KV pool. Not yet measured here.
#
# The checkpoint is 126 GiB on 121.6 GiB of usable memory. It fits because the
# 47.7 GiB FP8 n-gram ("PLE") table never enters it: the GPU reads table rows
# straight from storage through the host page tables. Stock SGLang does that
# from a sparse copy of the table it writes on every boot (see PLE_TABLE);
# this profile reads the checkpoint's own shards in place (patches/).
# Each value is "${VAR:-default}", so a one-off override from the shell works:
#   MOE_RUNNER_BACKEND=marlin ./serve.sh qwen3.8-flash-next

# SGLang 0.5.20 ships the model, the file-backed table and the NVFP4 loaders.
# The cookbook's Spark cells ran a qwen4-main-squashed build (4ccff141db) for
# three fixes; all are in 0.5.20: the file backend (#37068), the ModelOpt
# MIXED_PRECISION loader (#38121), and the router PDL ordering behind the "GB10
# NEXTN collapse" (NaN routing, output stuck on token 0: #36811, and #38290 in
# both moe_fused_gate and route_radix; checked in the 0.5.20 wheel).
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Weights, downloaded once (README, "Qwen3.8-Flash-Next"):
#   RadixArk/Qwen3.8-Flash-Next-NVFP4  -> MODEL_DIR (126 GiB; the table and
#     the BF16 MTP head are inside)
# nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION) loads too: set
# QUANTIZATION='' and MOE_RUNNER_BACKEND=flashinfer_cutlass (cookbook).
#
# Which weights, by name (MODEL_DIR, if set, wins over this):
#   radixark     RadixArk/Qwen3.8-Flash-Next-NVFP4, the default
#   abliterated  edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4:
#                the RadixArk checkpoint with Huihui's abliteration deltas
#                (recovered from Q8 GGUF; experts of layers 2, 4, 30, 46, 47
#                re-quantized, RadixArk's activation scales, table, MTP,
#                vision and tokenizer kept). Same layout, so every patch here
#                applies unchanged. Also: ./serve.sh qwen3.8-flash-next-abliterated
#   <a path>     any other checkpoint directory with this layout
# The served name follows (clients see which one answers).
WEIGHTS="${WEIGHTS:-radixark}"
case "$WEIGHTS" in
  radixark)
    _dir=/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
    _name=qwen3.8-flash-next ;;
  abliterated)
    _dir=/models/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4
    _name=qwen3.8-flash-next-abliterated ;;
  /*)
    _dir="$WEIGHTS"
    _name=qwen3.8-flash-next ;;
  *)
    echo "WEIGHTS must be radixark, abliterated or an absolute path, not '$WEIGHTS'" >&2
    exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$_dir}"
DRAFT_DIR=""
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$_name}"
unset _dir _name

# Where the n-gram table is read from:
#   mmap  the checkpoint's own safetensors shards, mapped read-only
#         (patches/gb10_ple_mmap.py). Nothing is written; boot does not copy
#         the table.
#   file  stock SGLang: a sparse 47.7 GiB copy under $SGLANG_CACHE_DIR/ple
#         (or PLE_FILE_DIR), rewritten through the mapping on every boot:
#         ~10 min into a fresh file, ~55 min into a filled one, so the
#         cookbook deletes it before each start. For comparison only.
# Both read rows the same way afterwards (GPU through the host page tables,
# page-cache hints before prefill, resident set capped at
# SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB, default 8).
PLE_TABLE="${PLE_TABLE:-mmap}"
# mmap: the directory holding the table shards. Any directory with the same
# ...ngram_embedding.shard_<k>.weight tensors works (e.g. the FP8 table shards
# of Qwen/Qwen3.8-Flash-Next-FP8), as long as their dtype matches what the
# checkpoint declares.
PLE_TABLE_DIR="${PLE_TABLE_DIR:-$MODEL_DIR}"
# file: where the sparse copy goes (empty = $SGLANG_CACHE_DIR/ple/<model>).
PLE_FILE_DIR="${PLE_FILE_DIR:-}"

# Checkpoint quantization and its kernels (cookbook: RadixArk export).
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-flashinfer_cutlass}"
# MoE expert kernels. Empty = SGLang's choice (the cookbook's). marlin runs
# the NVFP4 experts as W4A16 (weights dequantized in the kernel, activations
# stay bf16): the kind of kernel behind the ~60 tok/s int4 AutoRound recipe
# on vLLM, against cutlass's W4A4. Unmeasured on GB10; that is step 2 of the
# plan (README, "Qwen3.8-Flash-Next"). FP4_GEMM_BACKEND=marlin does the same
# for the dense NVFP4 layers.
# marlin also turns on patches/gb10_marlin_lean.py (see model_env): the stock
# load-time repack ran out of memory half way through the layers here.
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"

# Speculative decoding with the checkpoint's MTP head (NEXTN): 1 = on,
# 0 = off. The cookbook's low-latency cell: 3 steps, top-k 1, 4 draft tokens.
MTP="${MTP:-1}"
MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-4}"

# The drafter's vocabulary (--speculative-token-map): the 65,536-token set of
# the vLLM Flash-Next recipe (models/vocab/README.md) instead of the full
# 248,320-row lm_head, which the drafter reads on every draft step. Measured:
# 46.9 tok/s single-stream against 39.5, accept_len unchanged (3.6-3.8 of 4).
# Outputs cannot change (the target verifies every token); acceptance can, on
# CJK-heavy text especially: DRAFT_VOCAB='' goes back to the full vocabulary.
DRAFT_VOCAB="${DRAFT_VOCAB-$ROOT/models/vocab/qwen3.8-flash-next-draft-vocab-65536.pt}"

# BF16 GEMM library for the dense layers left in BF16 (attention, GDN and
# shared-expert projections). SGLang's own BF16 backends are SM90/SM100 only;
# on GB10 cuBLAS picks sm80 WMMA 16x16 kernels for them, ~40% of the decode
# step. cublaslt = TORCH_BLAS_PREFER_CUBLASLT=1: measured 40.1 tok/s against
# 39.5, no change. Empty = torch's default (cuBLAS).
BLAS="${BLAS:-}"

# FP8 for the layers the checkpoint keeps in BF16 (GDN and attention
# projections, the PLE key/value projections, the shared expert): quantized
# per output channel at load and run on SGLang's FP8 Marlin GEMM
# (patches/gb10_fp8_side.py). The decode profile puts ~40-48% of a step in
# those BF16 GEMMs on sm80 WMMA kernels; the vLLM recipe carries them in FP8.
# Measured: 56.7 tok/s single-stream against 46.9, and HumanEval (thinking
# off) 96.3% (158/164), the same as the int4 AutoRound recipe on vLLM. Lossy
# like any FP8 weight quantization. 0 = keep them in BF16.
FP8_SIDE="${FP8_SIDE:-1}"

# FP8 for the output heads (lm_head, BF16 [vocab x 2560]), same kernel
# (patches/gb10_fp8_side.py). After FP8_SIDE the profile has ~11 ms of a step
# in them: the target head once per verify (~5.4 ms) and the draft head once
# per MTP step (~1.2 ms x ~5).
# FP8_DRAFT_HEAD=1: the MTP draft's head (with DRAFT_VOCAB, a 65,536-row
#   slice of the target head). Only proposes tokens: a worse draft can lower
#   acceptance, never change an answer. On by default.
# FP8_HEAD=1: the target head, which decides every emitted token. Lossy in
#   principle; measured HumanEval 97.0% without thinking and 100% with medium
#   thinking (96.3% / 98.8% with it in BF16). On by default.
# Measured: 56.3 tok/s with both heads BF16, 60.9 with the draft head on FP8,
# 63.4 with both.
FP8_DRAFT_HEAD="${FP8_DRAFT_HEAD:-1}"
FP8_HEAD="${FP8_HEAD:-1}"

# A Triton GEMM for the small layers that stay BF16 because they select
# something: the MoE routers (mlp.gate [512 x 2560]) and the QSA indexer
# projections (index_qk_proj [640 x 2560]), ~64 calls a decode step at ~46 us
# on cuBLAS's sm80 WMMA kernels (patches/gb10_skinny.py). Same BF16 weights,
# FP32 accumulation, only the kernel changes; prefill keeps cuBLAS.
# Measured: no gain (62.0 tok/s against 63.4; the Triton kernel 45.0 us a
# call against cuBLAS's 45.8). SGLang runs both on a second stream under
# CUDA graphs (the router beside the shared expert, the indexer beside the
# qkv projection), so they share the GPU and are mostly off the critical
# path. Kept as an option; 0 = cuBLAS.
SKINNY_BF16="${SKINNY_BF16:-0}"

# FP8 for the hyper-connection mix weights (input_mix_weight_down/up,
# [320 x 10240] + [10240 x 320] BF16 per mix, 97 mixes in the target): SGLang's
# persistent _hc_mix kernel is bound by reading them, ~67 us a call, ~103 calls
# a decode step (~7.4 ms of ~57, not overlapped). patches/gb10_fp8_hc.py keeps
# them in FP8 with a per-row scale and runs a copy of the kernel that reads
# FP8: half the bytes. Measured 65.7-65.9 tok/s against 63.4 (the kernel
# 43 us a call against 67), 212-224 tok/s aggregate at 8 streams; HumanEval
# 97.6% without thinking, 99.4% with medium thinking (only 145, which fails
# on every build). Lossy in principle; on by default. 0 = BF16.
FP8_HC="${FP8_HC:-1}"

# Directional ablation at inference (patches/gb10_ablate.py): per-layer unit
# vectors from an .npz (keys = layer indices), e.g. the refusal directions
# built for RadixArk/Qwen3.8-Flash-Next-NVFP4:
#   hf download Mambavtt/qwen3.8-flash-next-refusal-ablation-vectors \
#     --local-dir /models/qwen3.8-flash-next-refusal-ablation-vectors
#   ABLATE=/models/qwen3.8-flash-next-refusal-ablation-vectors/refusal_directions-v4-late.npz ./serve.sh qwen3.8-flash-next
# Each listed layer's residual stream loses ABLATE_ALPHA times its
# component along the layer's vector: h - alpha d (d . h); no weight changes,
# empty ABLATE = the stock model. ALPHA 1 is the vectors' validated default,
# 1.5 stronger (watch for incoherence). ABLATE_AT output|input (which side
# of each layer; the boot log prints how alike neighbouring vectors are),
# ABLATE_STREAMS each|mean (all 4 hyper-connection streams, or only their
# mean, which the vectors were taken from). For the RadixArk weights only:
# vectors are model- and quantization-specific, and pointless on the
# abliterated checkpoint (WEIGHTS=abliterated), whose weights already have it.
ABLATE="${ABLATE:-}"
ABLATE_ALPHA="${ABLATE_ALPHA:-1.0}"
ABLATE_AT="${ABLATE_AT:-output}"
ABLATE_STREAMS="${ABLATE_STREAMS:-each}"

# Concurrency is bought with GDN state slots (~113 MB each in fp32 at TP=1),
# out of the ~12-18 GB the weights leave. The cookbook's pins: with MTP,
# 8 requests x 5 slots (extra_buffer); without, 24 x 4 (extra_buffer_lazy).
if [ "$MTP" = 1 ]; then
  MAX_RUNNING="${MAX_RUNNING:-8}"
  MAMBA_STRATEGY="${MAMBA_STRATEGY:-extra_buffer}"
  MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 5))}"
else
  MAX_RUNNING="${MAX_RUNNING:-24}"
  MAMBA_STRATEGY="${MAMBA_STRATEGY:-extra_buffer_lazy}"
  MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 4))}"
fi
# GDN state dtype. Empty = SGLang's default (fp32), as in the cookbook's Spark
# cells. bfloat16 halves every slot (the cookbook's RTX PRO 6000 cells, which
# kept GSM8K there): room for twice the requests, or a bigger KV pool.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"

# 0.85, as in the cookbook's cells (host memory never went below 10 GiB there;
# they also advise a watchdog on MemAvailable, which DGX OS's earlyoom is).
# The scheduler clamps the cap to what the GDN pool admits: the effective cap
# is in the boot log ("max_running_requests"), not always /get_server_info.
# That is DGX OS earlyoom's default margin: if the scheduler dies with exit
# code -15, see the Qwen 27B profile's note, or go to 0.82.
MEM_FRACTION="${MEM_FRACTION:-0.85}"

CHUNKED_PREFILL="${CHUNKED_PREFILL:-4096}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

# The model always reasons. Tool calls: auto = the detector matching the
# checkpoint's chat template (named in the boot log); empty = no tool parser.
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-auto}"

# Any other SGLang flags, appended last, so they override everything above.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Environment of the server process, set just before it starts.
model_env() {
  # The cookbook's advice for 200K+ contexts: variable-shape prefill buffers
  # otherwise fragment the caching allocator.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  case "$BLAS" in
    cublaslt) export TORCH_BLAS_PREFER_CUBLASLT=1 ;;
    '') ;;
    *) echo "BLAS must be empty or cublaslt, not '$BLAS'" >&2; exit 1 ;;
  esac
  if [ -n "$DRAFT_VOCAB" ] && [ ! -f "$DRAFT_VOCAB" ]; then
    echo "DRAFT_VOCAB=$DRAFT_VOCAB: no such file - see models/vocab/README.md" >&2
    exit 1
  fi
  case "$PLE_TABLE" in
    mmap)
      if ! compgen -G "$PLE_TABLE_DIR/*.safetensors" >/dev/null; then
        echo "no *.safetensors in PLE_TABLE_DIR=$PLE_TABLE_DIR - see models/$PROFILE.sh" >&2
        exit 1
      fi
      export GB10_PLE_MMAP=1 GB10_PLE_TABLE_DIR="$PLE_TABLE_DIR"
      ;;
    file) unset GB10_PLE_MMAP ;;
    *) echo "PLE_TABLE must be mmap or file, not '$PLE_TABLE'" >&2; exit 1 ;;
  esac
  # Marlin repacks all 48 MoE layers after loading; stock SGLang ran out of
  # memory half way on this box. patches/gb10_marlin_lean.py repacks with one
  # copy less and logs memory per layer ("Marlin repack, MoE layer N").
  if [ "$MOE_RUNNER_BACKEND" = marlin ]; then
    export GB10_MARLIN_LEAN=1
  else
    unset GB10_MARLIN_LEAN
  fi
  # patches/sitecustomize.py installs the enabled patches in every Python
  # process the server starts (SGLang spawns its scheduler).
  if [ "$FP8_SIDE" = 1 ]; then
    export GB10_FP8_SIDE=1
  else
    unset GB10_FP8_SIDE
  fi
  # Heads: with MTP they are converted after the draft takes its copy of
  # the target head; without MTP the target head is converted at load.
  unset GB10_FP8_DRAFT_HEAD GB10_FP8_TARGET_HEAD
  if [ "$MTP" = 1 ] && [ "$FP8_DRAFT_HEAD" = 1 ]; then
    export GB10_FP8_DRAFT_HEAD=1
  fi
  if [ "$FP8_HEAD" = 1 ]; then
    if [ "$MTP" = 1 ]; then export GB10_FP8_TARGET_HEAD=1; else export GB10_FP8_TARGET_HEAD=load; fi
  fi
  if [ "$SKINNY_BF16" = 1 ]; then
    export GB10_SKINNY_BF16=1
  else
    unset GB10_SKINNY_BF16
  fi
  if [ "$FP8_HC" = 1 ]; then
    export GB10_FP8_HC=1
  else
    unset GB10_FP8_HC
  fi
  if [ -n "$ABLATE" ]; then
    if [ ! -f "$ABLATE" ]; then
      echo "ABLATE=$ABLATE: no such file" >&2
      exit 1
    fi
    export GB10_ABLATE="$ABLATE" GB10_ABLATE_ALPHA="$ABLATE_ALPHA" \
      GB10_ABLATE_AT="$ABLATE_AT" GB10_ABLATE_STREAMS="$ABLATE_STREAMS"
  else
    unset GB10_ABLATE GB10_ABLATE_ALPHA GB10_ABLATE_AT GB10_ABLATE_STREAMS
  fi
  if [ -n "${GB10_PLE_MMAP:-}${GB10_MARLIN_LEAN:-}${GB10_FP8_SIDE:-}${GB10_FP8_DRAFT_HEAD:-}${GB10_FP8_TARGET_HEAD:-}${GB10_SKINNY_BF16:-}${GB10_FP8_HC:-}${GB10_ABLATE:-}" ]; then
    export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
model_args() {
  args+=(
    --tp 1
    --page-size "$PAGE_SIZE"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --max-running-requests "$MAX_RUNNING"
    --max-mamba-cache-size "$MAMBA_CACHE"
    --mamba-radix-cache-strategy "$MAMBA_STRATEGY"
    --ple-offload-embedding
    --ple-offload-backend file
  )
  [ -n "$QUANTIZATION" ] && args+=(--quantization "$QUANTIZATION")
  [ -n "$FP4_GEMM_BACKEND" ] && args+=(--fp4-gemm-backend "$FP4_GEMM_BACKEND")
  [ -n "$MOE_RUNNER_BACKEND" ] && args+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
  [ -n "$MAMBA_SSM_DTYPE" ] && args+=(--mamba-ssm-dtype "$MAMBA_SSM_DTYPE")
  [ "$PLE_TABLE" = file ] && [ -n "$PLE_FILE_DIR" ] && args+=(--ple-offload-dir "$PLE_FILE_DIR")
  if [ "$MTP" = 1 ]; then
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
  local ple="read in place from $PLE_TABLE_DIR"
  [ "$PLE_TABLE" = file ] && ple="stock sparse copy (rewritten at boot)"
  local vocab=full
  [ -n "$DRAFT_VOCAB" ] && vocab="$(basename "$DRAFT_VOCAB")"
  local heads="target BF16"
  [ "$FP8_HEAD" = 1 ] && heads="target FP8"
  [ "$MTP" = 1 ] && heads="$heads, draft $([ "$FP8_DRAFT_HEAD" = 1 ] && echo FP8 || echo BF16)"
  echo "NVFP4 ($QUANTIZATION), MoE ${MOE_RUNNER_BACKEND:-auto}, BF16 side layers $([ "$FP8_SIDE" = 1 ] && echo "-> FP8 (Marlin)" || echo "BF16 (${BLAS:-cublas})"); heads $heads; HC mix $([ "$FP8_HC" = 1 ] && echo FP8 || echo BF16); routers/indexer $([ "$SKINNY_BF16" = 1 ] && echo "Triton skinny GEMM" || echo cuBLAS); draft vocab $vocab; MTP $([ "$MTP" = 1 ] && echo "on ($MTP_STEPS/1/$MTP_DRAFT_TOKENS)" || echo off); cap $MAX_RUNNING (GDN pool $MAMBA_CACHE); PLE table $ple$([ -n "$ABLATE" ] && echo "; ablation $(basename "$ABLATE") alpha $ABLATE_ALPHA at layer $ABLATE_AT, $ABLATE_STREAMS stream(s)")"
}
