# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Gemma 4 31B (dense, BF16) + its MTP "assistant" draft.
#
#   ./serve.sh gemma4-31b
#
# From the SGLang cookbook's Gemma 4 recipe (docs/cookbook/autoregressive/
# Google/Gemma4.mdx). The cookbook has no DGX Spark cell for Gemma 4, so
# nothing here is measured on GB10 yet: treat every value as a starting point
# and run bench/perf.py before trusting it.
#
# Expect it to be slow. 31B in BF16 is ~62 GB of weights, and decode on GB10
# is bound by memory bandwidth (~273 GB/s): that is roughly 4 tok/s per stream
# before speculative decoding. MTP is on below for that reason.
# Each value is "${VAR:-default}", so a one-off override from the shell works:
#   MTP=0 ./serve.sh gemma4-31b

# SGLang 0.5.20 ships Gemma 4 (models, MTP draft, gemma4 parsers).
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# Weights, downloaded once (README, "Other models"). Gemma repos on the Hub have
# been gated: if hf download answers 401/403, accept the license on the model
# page and `hf auth login`. No revision is pinned here yet: pass the commit you
# download with --revision, and the server prints it at startup.
#   google/gemma-4-31B-it            -> MODEL_DIR   (~62 GB, BF16)
#   google/gemma-4-31B-it-assistant  -> DRAFT_DIR   (the MTP draft)
# The QAT release (-qat-q4_0-unquantized, with its own -assistant) keeps BF16
# weights too, so it is no smaller or faster to serve.
MODEL_DIR="${MODEL_DIR:-/models/gemma-4-31B-it}"
DRAFT_DIR="${DRAFT_DIR:-/models/gemma-4-31B-it-assistant}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-31b}"

# Speculative decoding with the paired assistant model (NEXTN): 1 = on,
# 0 = off. The cookbook's values: 5 steps, 6 draft tokens, top-k 1.
MTP="${MTP:-1}"
MTP_STEPS="${MTP_STEPS:-5}"
MTP_DRAFT_TOKENS="${MTP_DRAFT_TOKENS:-6}"

# Same reasoning as the Qwen profile: 0.80 keeps DGX OS earlyoom off the
# server. The weights take ~62 GB of the ~100 GB this allows; the rest is KV.
MEM_FRACTION="${MEM_FRACTION:-0.80}"

# Concurrent requests. Empty = SGLang's own default. Set it (e.g. 8) to cap
# the batch and capture decode CUDA graphs up to that size.
MAX_RUNNING="${MAX_RUNNING:-}"

# Empty = the model's own context length.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"

# Any other SGLang flags, appended last, so they override everything above.
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Without MTP the assistant model is not loaded, so its directory is not needed.
[ "$MTP" = 1 ] || DRAFT_DIR=""

# This model's flags, appended to the common ones in scripts/serve-sglang.sh.
# No --attention-backend: SGLang picks triton for Gemma 4 on this GPU itself,
# which image tokens need (bidirectional attention during prefill).
model_args() {
  args+=(
    --reasoning-parser gemma4
    --tool-call-parser gemma4
  )
  if [ "$MTP" = 1 ]; then
    args+=(
      --speculative-algorithm NEXTN
      --speculative-draft-model-path "$DRAFT_DIR"
      --speculative-num-steps "$MTP_STEPS"
      --speculative-num-draft-tokens "$MTP_DRAFT_TOKENS"
      --speculative-eagle-topk 1
    )
  fi
  if [ -n "$MAX_RUNNING" ]; then
    args+=(--max-running-requests "$MAX_RUNNING" --cuda-graph-max-bs-decode "$MAX_RUNNING")
  fi
}

# One line for the startup summary.
model_summary() {
  if [ "$MTP" = 1 ]; then
    echo "MTP (NEXTN) $MTP_STEPS steps / $MTP_DRAFT_TOKENS draft tokens; cap ${MAX_RUNNING:-SGLang default}"
  else
    echo "no speculative decoding; cap ${MAX_RUNNING:-SGLang default}"
  fi
}
