# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B + DFlash2 speculative decoding. The recipe this project
# was built around; every number in results/ is from it.
#
#   ./serve.sh qwen3.8-27b          (the default profile, so also just ./serve.sh)
#
# A profile holds everything that belongs to one model: where its weights
# are, which SGLang it runs on, and its own flags (model_args below). Machine
# settings (port, API key, CPU pinning, JIT jobs) stay in ../serve.sh.
# Each value is "${VAR:-default}", so a one-off override from the shell works:
#   DRAFT_TOKENS=16 MAX_RUNNING=16 ./serve.sh

# SGLang version. 0.5.20 is the one this recipe was built against: its lock is
# in requirements/ and every flag below exists there. To try a newer one: set
# it, ./serve.sh install, ./serve.sh. Its own venv leaves the working one
# untouched, so rolling back is setting this back. A version with no lock yet
# is resolved on install and its lock written to requirements/; commit it once
# that version has served and benchmarked well. If a newer SGLang renames a
# flag, the boot fails with "unrecognized arguments": adjust model_args.
# Releases: https://pypi.org/project/sglang/#history
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
# Extra package index, for nightly builds (then use the exact nightly version
# string above): https://docs.sglang.ai/whl/cu130/  Empty = PyPI only.
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Required: the target checkpoint and the DFlash2 draft, downloaded once with
# a pinned --revision (README, "Weights"). Targets measured here:
#   Qwen/Qwen3.8-27B-FP8 @ 017b9c7a (default): Qwen's own checkpoint and the
#     more accurate one (HumanEval 97.6% with thinking off, vs 93.9%).
#   RadixArk/Qwen3.8-27B-NVFP4 @ 554ebba9: ~40% faster single-stream, 2.5x
#     faster prefill (results/RESULTS.md, "FP8 target").
# Draft: z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c (incoai/Qwen3.8-27B-DFlash2, the
# SGLang cookbook's name, mirrors the same weights).
# Already in an HF cache? Point at the snapshot directory instead of copying,
# e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7a...
# Either way the server prints the revision it found at startup.
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
DRAFT_DIR="${DRAFT_DIR:-/models/Qwen3.8-27B-DFlash2}"

# Speculative decoding:
#   dflash  z-lab's DFlash2 draft (DRAFT_DIR), DRAFT_TOKENS per step: the
#           default, 94.2 tok/s single-stream at 15 on RadixArk NVFP4.
#   mtp     the checkpoint's own MTP head (NEXTN; no draft download), as the
#           infercrane Qwen3.8-27B-FP8 recipe serves it on an H200: 3 steps,
#           top-1, 4 tokens verified. At most 4 tokens a step against DFlash2's
#           ~9 accepted, so expect less on one stream here (a GB10 step is
#           bound by reading the weights); unmeasured on GB10. The checkpoint
#           must carry the head (config.json: mtp_num_hidden_layers).
SPEC="${SPEC:-dflash}"
MTP_STEPS="${MTP_STEPS:-3}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-4}"
case "$SPEC" in
  dflash) ;;
  mtp) DRAFT_DIR="" ;;
  *) echo "SPEC must be dflash or mtp, not '$SPEC'" >&2; exit 2 ;;
esac
# flashinfer with DFlash2, as measured. MTP takes triton, as on Ornith, where
# flashinfer + MTP rejected every proposal (accept_len 1.00; on a checkpoint
# that was broken anyway, so try ATTENTION_BACKEND=flashinfer too).
if [ "$SPEC" = mtp ]; then
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
else
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
fi
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

# Draft tokens per step, the largest single-stream lever. The optima diverge:
# 10 wins aggregate throughput (435 vs 385 tok/s at 16 streams on NVFP4),
# 16 wins a single stream (78.6 vs 65.2 tok/s, +28% over the draft's 8).
# 15 is the default since the draft runs in FP8 (FP8_DRAFT): a cheaper draft
# step pays for the longer block. RadixArk NVFP4, cap 12, against 11: 94.2
# vs 77.1 tok/s single-stream (+22%), the same or better at 1-8 streams
# (75 / 138 / 225 / 339 vs 74 / 127 / 221 / 292), -14% at 12 (340 vs 396).
# With the BF16 draft 15 lost on short varied answers and 11 was the default
# (results/RESULTS.md, "Draft 10 / 11 / 12 at cap 12", "draft 11 vs 15").
# Mostly 10-12 requests at once: DRAFT_TOKENS=11. Each extra draft token
# costs ~70 MB of verify buffer per running request (~3.4 GB at cap 12 for
# 15 vs 11), taken from the KV pool.
# Past 16 accept_len falls and both get worse. Any value other than the
# draft's block size (8) logs "DFLASH block size mismatch" at boot; harmless.
# Change it together with MAX_RUNNING: the verify buffer
# (intermediate_ssm_state_cache in the boot log) holds ~70 MB per running
# request per draft token. 32 x 10 measured 23.2 GB natively; 32 x 16 would be
# ~37 GB, taken out of the KV pool, for less aggregate. For single-stream use:
# DRAFT_TOKENS=16 MAX_RUNNING=16 (~18 GB).
DRAFT_TOKENS="${DRAFT_TOKENS:-15}"

# Concurrent requests: set it to the most you actually run at once. Concurrency
# on this hybrid model is bought with GDN state, not KV, and the cap reserves
# that state up front whether or not it is used: ~0.0735 GiB per slot, 5 slots
# per request (4, plus 1 for the DFlash2 verify; SGLang clamps the cap to
# pool / 5), plus the verify buffer (requests x draft tokens). Cap 32 holds
# ~35 GiB of it, cap 12 ~14 GiB; the difference goes to the KV pool (~+520K
# tokens, room for about five 262K sessions at 0.85). Below the cap, speed does
# not depend on it; the cap only limits how many requests run at once.
# 12 fits "up to ~12 concurrent requests"; qwen3.8-27b-throughput keeps 32
# (578 tok/s peak natively), qwen3.8-27b-longctx trades down to 6 x 262K.
# 48 (~47 GB of state) failed to boot on 128 GB.
MAX_RUNNING="${MAX_RUNNING:-12}"
# 5 slots per running request, plus one per request so a finished turn's GDN
# state can stay cached for the next turn of the same conversation.
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 6))}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-$MAX_RUNNING}"

# Memory fraction of the unified 128 GB that SGLang may take for weights, GDN
# state and KV. Not a speed lever: 0.82 / 0.85 / 0.90 measure the same. It is
# a stability one: at 0.85 the host keeps ~8 GB, exactly DGX OS earlyoom's
# threshold, and earlyoom SIGTERMs the scheduler during boot or a long
# prefill (exit code -15, no traceback; journalctl -u earlyoom). The SGLang
# cookbook lost 15 of 48 GB10 configs at 0.85, none at 0.80. 0.85 ran fine
# here under Docker. Above 0.85 add --max-total-tokens 1048576 to EXTRA_ARGS:
# a bigger KV pool goes unused and the lost headroom cost 18% at 32 streams.
MEM_FRACTION="${MEM_FRACTION:-0.80}"

# Prefill chunk. 8192 is what everything in results/ was measured with and
# favours prefill throughput; the cookbook's 2048 keeps decode smoother
# while long prompts are being prefilled alongside it.
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
# Prefill CUDA graphs: 0 = off, as measured here; 1 = on, as in the cookbook's
# GB10 cell. Not benchmarked here.
PREFILL_CUDA_GRAPH="${PREFILL_CUDA_GRAPH:-0}"

# Context: 262144 is the model's native length.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

# Where a decode step of RadixArk/Qwen3.8-27B-NVFP4 + DFlash2 goes
# (bench/profile_decode.py, results/RESULTS.md "Decode profile of the 27B"):
# the MLPs are NVFP4 (CUTLASS, ~38%), the GDN / attention projections FP8
# W8A8 (cuBLASLt nvjet_*_qq*, ~32%), the lm_head NVFP4; BF16 is left in the
# DFlash2 draft (~13%), GDN's small in_proj_ba (~2%) and the vision tower.

# FP8 for the target's linear layers the checkpoint keeps in BF16
# (patches/gb10_fp8_side.py, as on Flash-Next and Ornith: GDN in_proj_qkvz,
# in_proj_ba, out_proj; attention qkv_proj / o_proj; shared experts) to FP8
# weight-only on SGLang's FP8 Marlin GEMM; the boot log counts them ("FP8 side
# (target): ..."). The model class comes from the checkpoint's config.json.
# On RadixArk's NVFP4 it finds GDN's 48 in_proj_ba (96 x 5120, padded to
# the tile) and 27 vision qkv: 75 layers, 0.24 -> 0.13 GiB, ~2% of a step.
FP8_SIDE="${FP8_SIDE:-0}"

# FP8 for the draft: the DFlash2 draft's MLPs and o_proj (~3 GB of BF16 read
# every step, the only BF16 GEMMs left in a RadixArk NVFP4 decode step) to
# FP8 weight-only on Marlin; qkv_proj stays BF16 for the fused context-KV
# GEMM (patches/gb10_fp8_side.py, GB10_FP8_DFLASH). The boot log has
# "FP8 side (DFlash draft): N linear layers ...". The draft only proposes: a
# worse draft costs accept_len, never the answers. RadixArk NVFP4, draft 11
# (with FP8_SIDE=1): 77.1 tok/s single-stream against 74.1, accept_len
# unchanged (8.2-8.9), +4-12% at 1-4 streams, the same at 8-12. The DSpark
# variant (a DFlash backbone too) keeps it off until measured there.
FP8_DRAFT="${FP8_DRAFT:-1}"

# The dense NVFP4 GEMM kernel (--fp4-gemm-backend): empty = SGLang's auto
# (CUTLASS here: ~339 us a call at 12 rows, ~40% of the weight-read speed).
# marlin (W4A16: BF16 activations, FP4 weights; SGLang's own default for
# some models on SM120), flashinfer_cudnn, flashinfer_trtllm,
# flashinfer_cutlass, flashinfer_cutedsl. flashinfer_cudnn: 74.9 tok/s, no
# gain. marlin: 78.5 single-stream (+2%), but -18% / -23% at 8 / 12 streams
# (W4A16 at 132 verify rows is slower than FP4 tensor cores). marlin needs SGLang's fused SiLU + FP4-quant MLP path off (it hands
# down_proj a packed FP4 tuple whatever the backend: "apply_fp4_marlin_linear()
# Expected a value of type 'Tensor' ... found type 'tuple'"); set below.
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-}"

# Any other SGLang flags, appended last, so they override everything above.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# The server environment for FP8_SIDE, FP8_DRAFT and FP4_GEMM_BACKEND;
# variants that define their own model_env call this too.
fp8_side_env() {
  if [ "$FP8_SIDE" = 1 ]; then
    local arch
    arch="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["architectures"][0])' \
      "$MODEL_DIR/config.json")" || { echo "FP8_SIDE: cannot read $MODEL_DIR/config.json" >&2; exit 1; }
    export GB10_FP8_SIDE=1 GB10_FP8_SIDE_TARGET="sglang.srt.models.qwen3_5:$arch"
  else
    unset GB10_FP8_SIDE GB10_FP8_SIDE_TARGET
  fi
  if [ "$FP8_DRAFT" = 1 ]; then
    export GB10_FP8_DFLASH=1
  else
    unset GB10_FP8_DFLASH
  fi
  if [ "$FP4_GEMM_BACKEND" = marlin ]; then
    export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
  fi
  if [ "$FP8_SIDE" = 1 ] || [ "$FP8_DRAFT" = 1 ]; then
    export PYTHONPATH="$ROOT/patches${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

model_env() {
  fp8_side_env
}

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
model_args() {
  spec_args
  args+=(
    --mamba-radix-cache-strategy extra_buffer
    --mamba-ssm-dtype bfloat16
    --kv-cache-dtype fp8_e4m3
    --max-mamba-cache-size "$MAMBA_CACHE"
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
    --attention-backend "$ATTENTION_BACKEND"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
  )
  [ "$PREFILL_CUDA_GRAPH" = 1 ] || args+=(--disable-prefill-cuda-graph)
  [ -n "$FP4_GEMM_BACKEND" ] && args+=(--fp4-gemm-backend "$FP4_GEMM_BACKEND")
  return 0
}

# The speculative-decoding flags, on their own so that a variant with another
# draft (qwen3.8-27b-dspark.sh) replaces only these.
spec_args() {
  if [ "$SPEC" = mtp ]; then
    args+=(
      --speculative-algorithm NEXTN
      --speculative-num-steps "$MTP_STEPS"
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
    )
    return 0
  fi
  args+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_DIR"
    --speculative-num-draft-tokens "$DRAFT_TOKENS"
  )
}

# The FP8 / FP4 knobs for the startup summary.
fp8_summary() {
  [ "$FP8_SIDE" = 1 ] && printf '; BF16 layers -> FP8 (Marlin)'
  [ "$FP8_DRAFT" = 1 ] && [ "$SPEC" != mtp ] && printf '; draft -> FP8 (Marlin)'
  [ -n "$FP4_GEMM_BACKEND" ] && printf '; FP4 GEMM %s' "$FP4_GEMM_BACKEND"
  return 0
}

# One line for the startup summary.
model_summary() {
  echo "$([ "$SPEC" = mtp ] && echo "MTP (NEXTN $MTP_STEPS/1/$MTP_DRAFT_TOKENS), attention $ATTENTION_BACKEND" || echo "DFlash2, $DRAFT_TOKENS draft tokens"); cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)$(fp8_summary)"
}
