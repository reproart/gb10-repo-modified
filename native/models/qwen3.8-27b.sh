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
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-sglang}"

# Draft tokens per step, the largest single-stream lever. The optima diverge:
# 10 wins aggregate throughput (435 vs 385 tok/s at 16 streams on NVFP4),
# 16 wins a single stream (78.6 vs 65.2 tok/s, +28% over the default 8).
# Past 16 accept_len falls and both get worse. Any value other than the
# draft's block size (8) logs "DFLASH block size mismatch" at boot; harmless.
# Change it together with MAX_RUNNING: the verify buffer
# (intermediate_ssm_state_cache in the boot log) holds ~70 MB per running
# request per draft token. 32 x 10 measured 23.2 GB natively; 32 x 16 would be
# ~37 GB, taken out of the KV pool, for less aggregate. For single-stream use:
# DRAFT_TOKENS=16 MAX_RUNNING=16 (~18 GB).
DRAFT_TOKENS="${DRAFT_TOKENS:-10}"

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

# Any other SGLang flags, appended last, so they override everything above.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
model_args() {
  args+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_DIR"
    --speculative-num-draft-tokens "$DRAFT_TOKENS"
    --mamba-radix-cache-strategy extra_buffer
    --mamba-ssm-dtype bfloat16
    --kv-cache-dtype fp8_e4m3
    --max-mamba-cache-size "$MAMBA_CACHE"
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_BS"
    --attention-backend flashinfer
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
  )
  [ "$PREFILL_CUDA_GRAPH" = 1 ] || args+=(--disable-prefill-cuda-graph)
}

# One line for the startup summary.
model_summary() {
  echo "DFlash2, $DRAFT_TOKENS draft tokens; cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)"
}
