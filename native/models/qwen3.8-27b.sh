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

# SGLang version. 0.5.21 since 2026-10-17: decode and prefill the same as
# 0.5.20 or better, HumanEval unchanged (results/RESULTS.md, "Cap 16 on
# 0.5.21"); this profile's variants (-single, -throughput, -longctx, -dspark,
# zen6, meerkat-triz) follow. SGLANG_VERSION=0.5.20 goes back: the recipe was
# built against it and its lock is in requirements/. To try a newer one: set
# it, ./serve.sh install, ./serve.sh. Its own venv leaves the working one
# untouched, so rolling back is setting this back. A version with no lock yet
# is resolved on install and its lock written to requirements/; commit it once
# that version has served and benchmarked well. If a newer SGLang renames a
# flag, the boot fails with "unrecognized arguments": adjust model_args.
# Releases: https://pypi.org/project/sglang/#history
SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
# Extra package index, for nightly builds (then use the exact nightly version
# string above): https://docs.sglang.ai/whl/cu130/  Empty = PyPI only.
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Required: the target checkpoint and the DFlash2 draft, downloaded once with
# a pinned --revision (README, "Weights"). Targets measured here:
#   RadixArk/Qwen3.8-27B-NVFP4 @ 554ebba9 (default since 2026-10-17): ~40%
#     faster single-stream, 2.5x faster prefill (results/RESULTS.md, "FP8
#     target"); every recent table in results/ is on it.
#   Qwen/Qwen3.8-27B-FP8 @ 017b9c7a: Qwen's own checkpoint and the more
#     accurate one with thinking off (HumanEval 97.6% vs 93.9%; with thinking
#     on both make at most one error). MODEL_DIR=/models/Qwen3.8-27B-FP8.
# Draft: z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c (incoai/Qwen3.8-27B-DFlash2, the
# SGLang cookbook's name, mirrors the same weights).
# Already in an HF cache? Point at the snapshot directory instead of copying,
# e.g. ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/554ebba9...
# Either way the server prints the revision it found at startup.
MODEL_DIR="${MODEL_DIR:-/models/RadixArk/Qwen3.8-27B-NVFP4}"
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
# pool / 5), plus the verify buffer (requests x draft tokens). Cap 32 at draft
# 10 holds ~35 GiB of it, cap 16 at draft 15 ~25 GiB (~995K tokens of KV left
# at 0.80, three full 262K sessions; README, "Long sessions"). Below the cap, speed does
# not depend on it; the cap only limits how many requests run at once.
# 16 since 2026-10-17: +10% at the peak against 12 on 0.5.21 (410-436 tok/s
# at 16 streams against 370-383 at 12), the rest the same, and still ~985K
# tokens of fp8 KV at 0.80. qwen3.8-27b-throughput keeps 32 (578 tok/s peak
# natively), qwen3.8-27b-longctx trades down to 6 x 262K. 48 (~47 GB of
# state) failed to boot on 128 GB.
MAX_RUNNING="${MAX_RUNNING:-16}"
# 5 slots per running request, plus one per request so a finished turn's GDN
# state can stay cached for the next turn of the same conversation.
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 6))}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-$MAX_RUNNING}"

# Memory fraction of the unified 128 GB that SGLang may take for weights, GDN
# state and KV. Not a speed lever: 0.82 / 0.85 / 0.90 measure the same. It is
# a stability one: at 0.85 the host keeps ~8 GB, exactly DGX OS earlyoom's
# threshold, and earlyoom SIGTERMs the scheduler during boot or a long
# prefill (exit code -15, no traceback; journalctl -u earlyoom;
# scripts/host-tuning.sh lowers that threshold to 1 GiB). The SGLang
# cookbook lost 15 of 48 GB10 configs at 0.85, none at 0.80. 0.85 ran fine
# here under Docker. Above 0.85 add --max-total-tokens 1048576 to EXTRA_ARGS:
# a bigger KV pool goes unused and the lost headroom cost 18% at 32 streams.
# With scripts/host-tuning.sh applied (earlyoom at 1 GiB, not ~8 GB) the
# trap is gone and the default is 0.85: ~1.15M tokens of fp8 KV against ~995K,
# four 262K sessions against three; six full perf.py suites ran at 0.85 so
# (results/RESULTS.md, "Host tuning"). Without it, 0.80.
if [ -z "${MEM_FRACTION:-}" ] && { [ -f /etc/sysctl.d/99-gb10.conf ] || [ -f /etc/sysctl.d/99-tensorfold.conf ]; }; then
  MEM_FRACTION=0.85
fi
MEM_FRACTION="${MEM_FRACTION:-0.80}"

# Prefill chunk and how decode gets in between (results/RESULTS.md, "Decode
# during a long prefill"). SGLang runs a ready prefill batch before decode
# and, with speculative decoding, never mixes the two, so with no interval
# the chunks of a long prompt run back to back and every stream already
# generating stops for the whole prefill (14 s for 32K at 8192, ~80 s for
# 101K). PREFILL_DECODE_INTERVAL=N (--prefill-decode-interval) runs N decode
# rounds after each prefill batch: at 2048 / 1 the longest pause is 1.3-2.1 s
# and decode keeps 5-16 tok/s through it. 2048 also prefills faster alone
# than 8192 (2,052 against 1,765-1,821 tok/s at 32K, 1,334 against
# 1,198-1,224 at 101K; 2K prompts ~10% slower, two chunks), and the sweep is
# the same. A larger N gives the running streams more of the time and the
# long prompt a longer TTFT (each round is one decode step, ~0.1 s, per
# chunk). 8192 / 0 was everything in results/ before 2026-10-17.
CHUNKED_PREFILL="${CHUNKED_PREFILL:-2048}"
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-1}"
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
# (CUTLASS here: at the read roof, ~240 of 264 GB/s; results/RESULTS.md,
# "Decode GEMMs of the 27B").
# marlin (W4A16: BF16 activations, FP4 weights; SGLang's own default for
# some models on SM120), flashinfer_cudnn, flashinfer_trtllm,
# flashinfer_cutlass, flashinfer_cutedsl. flashinfer_cudnn: 74.9 tok/s, no
# gain. marlin: 78.5 single-stream (+2%), but -18% / -23% at 8 / 12 streams
# (W4A16 at 132 verify rows is slower than FP4 tensor cores). marlin needs SGLang's fused SiLU + FP4-quant MLP path off (it hands
# down_proj a packed FP4 tuple whatever the backend: "apply_fp4_marlin_linear()
# Expected a value of type 'Tensor' ... found type 'tuple'"); set below.
FP4_GEMM_BACKEND="${FP4_GEMM_BACKEND:-}"

# How the checkpoint's FP8 GEMMs (GDN / attention projections, cuBLASLt through
# FlashInfer) get their kernels. SGLang autotunes them at boot at 1, 2, 4, ...,
# 256 rows; a DFlash2 verify runs DRAFT_TOKENS (15) rows a request. A tactic is
# an index into cuBLASLt's algorithm list for the tuned size, applied to the
# list for the real one: tuned at 16, used at 15, it can be another, slower
# kernel (GDN out_proj 166-190 us instead of 129: up to ~4 ms of a ~104 ms
# step). At 16 rows several kernels tie, so each boot's timing noise picks, and
# the tuning cache that would keep a pick was refused (written under an older
# cuDNN frontend): the two single-stream modes, 88-89.5 or 93.8-94.2 tok/s
# (results/RESULTS.md, "Decode GEMMs of the 27B"; bench sections 5-7).
#   exact      also tune at the verify sizes themselves (DRAFT_TOKENS x 1 ..
#              CUDA_GRAPH_BS) and look those sizes up there
#              (patches/gb10_fp8_tune.py); other sizes as before
#   heuristic  no autotune for the FP8 GEMMs (--flashinfer-autotune-skip-ops
#              fp8_gemm): cuBLASLt's own pick, the same every boot; in the bench
#              as fast at 15 / 60 / 240 rows, slower at 120 (out_proj 170-178 us
#              against 146)
#   stock      SGLang as shipped
FP8_GEMM_TUNE="${FP8_GEMM_TUNE:-exact}"

# Directional ablation at inference (patches/gb10_ablate.py), as on Flash-Next:
# each listed layer's post-layer residual stream loses ABLATE_ALPHA times its
# component along that layer's unit vector, h - alpha d (d . h); no weight
# changes. The refusal directions msuiche derived on Qwen/Qwen3.8-27B (BF16) as
# projective GGUF control vectors (glp.mode = project), once:
#   hf download msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49 \
#     --revision 5f1be31cfde653f24fe74c71047433012b8ede9e \
#     --local-dir /models/qwen3.8-27b-refusal-vectors
# then
#   ABLATE=1 ./serve.sh qwen3.8-27b         the vectors at ABLATE_VECTORS
#   ./serve.sh qwen3.8-27b-ablated          the same, as a profile (and a served
#                                           name) of its own
#   ABLATE=/path/other.gguf ./serve.sh qwen3.8-27b   another file (.gguf or .npz)
# ABLATE empty or 0 = the stock model. The download has two files:
# GLP-63-L1-63 (one direction pooled over layers 1-63, the same as the repo's
# LoRA: the default) and GLP-49-L10-58 (a direction per layer, layers 10-58).
# Their card, on BF16: 81-84% delivery on its refusal suite against 3% stock,
# harmless prompts 100%, capability 11-12 of 12. alpha 1 is the validated
# value: 2 reflects the component (harmless prompts refused), 4 collapses the
# model; for less, use fewer layers, not a smaller alpha. RadixArk's NVFP4 is
# that model quantized; the card expects the projection to survive it (the
# weight error enters suppressed ~1/sqrt(5120)).
# ABLATE_AT output|input (after layer l, as the files say, or before it).
# Measured (2026-11-10), single-stream decode against 93.8 tok/s stock: GLP-63
# 83.6 (-11%), GLP-49 75.2 (-20%); with GLP-63, 8 / 16 streams as stock (319 /
# 422 against 322 / 430), prefill -2-5%. The hooks cost next to nothing inside
# the CUDA graph (GLP-63 has more of them); what is lost is the DFlash2 draft's
# acceptance: it reads the ablated states, which it was not trained on. A
# question GLP-49 refused mid-answer got its answer with GLP-63. For the
# occasional question it is for; serve the plain profile otherwise. A refusal
# can still come back late in a long answer (msuiche saw the vector hold the
# first paragraph on another model) or follow an earlier one in the chat: start
# a new chat, or try the other file.
ABLATE="${ABLATE:-}"
ABLATE_VECTORS="${ABLATE_VECTORS:-/models/qwen3.8-27b-refusal-vectors/Qwen3.8-27B-abliterated-cyber-GLP-63-L1-63-a1.gguf}"
case "$ABLATE" in
  ''|0) ABLATE_FILE="" ;;
  1) ABLATE_FILE="$ABLATE_VECTORS" ;;
  *) ABLATE_FILE="$ABLATE" ;;
esac
ABLATE_ALPHA="${ABLATE_ALPHA:-1.0}"
ABLATE_AT="${ABLATE_AT:-output}"

# Any other SGLang flags, appended last, so they override everything above.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# The server environment for FP8_SIDE, FP8_DRAFT, FP4_GEMM_BACKEND,
# FP8_GEMM_TUNE and ABLATE; variants that define their own model_env call this too.
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
  case "$FP8_GEMM_TUNE" in
    exact)
      # rows one request adds to a verify, and the most requests in a decode graph
      GB10_FP8_TUNE_ROWS="$DRAFT_TOKENS"
      [ "$SPEC" = mtp ] && GB10_FP8_TUNE_ROWS="$MTP_DRAFT_TOKENS"
      export GB10_FP8_TUNE_ROWS GB10_FP8_TUNE_MAX_BS="$CUDA_GRAPH_BS" ;;
    heuristic|stock) unset GB10_FP8_TUNE_ROWS GB10_FP8_TUNE_MAX_BS ;;
    *) echo "FP8_GEMM_TUNE must be exact, heuristic or stock, not '$FP8_GEMM_TUNE'" >&2; exit 1 ;;
  esac
  if [ -n "$ABLATE_FILE" ]; then
    if [ ! -f "$ABLATE_FILE" ]; then
      if [ "$ABLATE" = 1 ]; then
        echo "ABLATE=1: no vectors at ABLATE_VECTORS=$ABLATE_VECTORS - download them once:" >&2
        echo "  hf download msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49 --revision 5f1be31cfde653f24fe74c71047433012b8ede9e --local-dir $(dirname "$ABLATE_VECTORS")" >&2
      else
        echo "ABLATE=$ABLATE: no such file" >&2
      fi
      exit 1
    fi
    export GB10_ABLATE="$ABLATE_FILE" GB10_ABLATE_ALPHA="$ABLATE_ALPHA" GB10_ABLATE_AT="$ABLATE_AT"
  else
    unset GB10_ABLATE GB10_ABLATE_ALPHA GB10_ABLATE_AT
  fi
  if [ "$FP8_SIDE" = 1 ] || [ "$FP8_DRAFT" = 1 ] || [ "$FP8_GEMM_TUNE" = exact ] || [ -n "$ABLATE_FILE" ]; then
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
    --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
  )
  [ "$PREFILL_CUDA_GRAPH" = 1 ] || args+=(--disable-prefill-cuda-graph)
  [ -n "$FP4_GEMM_BACKEND" ] && args+=(--fp4-gemm-backend "$FP4_GEMM_BACKEND")
  [ "$FP8_GEMM_TUNE" = heuristic ] && args+=(--flashinfer-autotune-skip-ops fp8_gemm)
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
  [ "$FP8_GEMM_TUNE" != stock ] && printf '; FP8 GEMM tactics %s' "$FP8_GEMM_TUNE"
  [ -n "$ABLATE_FILE" ] && printf '; ablation %s alpha %s at layer %s' "$(basename "$ABLATE_FILE")" "$ABLATE_ALPHA" "$ABLATE_AT"
  return 0
}

# One line for the startup summary.
model_summary() {
  echo "$([ "$SPEC" = mtp ] && echo "MTP (NEXTN $MTP_STEPS/1/$MTP_DRAFT_TOKENS), attention $ATTENTION_BACKEND" || echo "DFlash2, $DRAFT_TOKENS draft tokens"); cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)$(fp8_summary)"
}
