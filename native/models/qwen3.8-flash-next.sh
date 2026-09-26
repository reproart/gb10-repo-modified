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

# SGLang 0.5.20 ships the model, the file-backed table and the NVFP4 loaders
# (the cookbook's Spark cells ran a qwen4-main-squashed build of the same code).
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Weights, downloaded once (README, "Qwen3.8-Flash-Next"):
#   RadixArk/Qwen3.8-Flash-Next-NVFP4  -> MODEL_DIR (126 GiB; the table and
#     the BF16 MTP head are inside)
# nvidia/Qwen3.8-Flash-Next-NVFP4 (ModelOpt MIXED_PRECISION) loads too: set
# QUANTIZATION='' and MOE_RUNNER_BACKEND=flashinfer_cutlass (cookbook).
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-NVFP4}"
DRAFT_DIR=""
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next}"

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
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"

# Speculative decoding with the checkpoint's MTP head (NEXTN): 1 = on,
# 0 = off. The cookbook's low-latency cell: 3 steps, top-k 1, 4 draft tokens.
MTP="${MTP:-1}"
MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-4}"

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

# 0.85, as in the cookbook's cells (host memory never went below 10 GiB there).
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
  case "$PLE_TABLE" in
    mmap)
      if ! compgen -G "$PLE_TABLE_DIR/*.safetensors" >/dev/null; then
        echo "no *.safetensors in PLE_TABLE_DIR=$PLE_TABLE_DIR - see models/$PROFILE.sh" >&2
        exit 1
      fi
      # patches/sitecustomize.py installs the patch in every Python process
      # the server starts (SGLang spawns its scheduler).
      export GB10_PLE_MMAP=1 GB10_PLE_TABLE_DIR="$PLE_TABLE_DIR"
      export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
      ;;
    file) unset GB10_PLE_MMAP ;;
    *) echo "PLE_TABLE must be mmap or file, not '$PLE_TABLE'" >&2; exit 1 ;;
  esac
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
  fi
  [ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  return 0
}

# One line for the startup summary.
model_summary() {
  local ple="read in place from $PLE_TABLE_DIR"
  [ "$PLE_TABLE" = file ] && ple="stock sparse copy (rewritten at boot)"
  echo "NVFP4 ($QUANTIZATION), MoE ${MOE_RUNNER_BACKEND:-auto}; MTP $([ "$MTP" = 1 ] && echo "on ($MTP_STEPS/1/$MTP_DRAFT_TOKENS)" || echo off); cap $MAX_RUNNING (GDN pool $MAMBA_CACHE); PLE table $ple"
}
