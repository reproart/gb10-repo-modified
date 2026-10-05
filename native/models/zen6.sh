# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: zenlm/zen6, a repackaging of the 27B this project serves:
# RadixArk/Qwen3.8-27B-NVFP4 (NVFP4 MLPs and lm_head, FP8 attention /
# GDN projections, static FP8 KV scheme; ModelOpt 0.47 MIXED_PRECISION;
# Qwen3_5ForConditionalGeneration) with YaRN factor 4 in its config.json
# (1,048,576 tokens) and the DFlash2 draft bundled in dflash2/.
#
# How SGLang 0.5.20 reads that config: the YaRN sits in
# text_config.rope_parameters next to the interleaved mrope sections, and
# get_rope builds YaRNScalingMRotaryEmbedding from it (original 262144 x 4)
# for the 16 full-attention layers; the 48 GDN layers have no RoPE.
# max_position_embeddings stays 262144 and the YaRN block names
# original_max_position_embeddings, so SGLang derives a 262144 context and
# refuses a longer --context-length unless
# SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1, which model_env sets when the
# config has YaRN.
#
#   hf download zenlm/zen6 --local-dir /models/zenlm/zen6
#   ./serve.sh zen6
#
# Everything comes from qwen3.8-27b.sh (SGLang 0.5.20, DFlash2 at 15 draft
# tokens, FP8 draft, cap 12, fp8_e4m3 KV, parsers) except the paths, the
# name and the context length. Not the card's flags: --speculative-num-steps
# does nothing for DFLASH, and fp8_e5m2 KV keeps 2 mantissa bits where e4m3,
# the base profile's, keeps 3.
#
# The card's decode figures (141 tok/s for code with the drafter, 62 alone)
# are about twice what this box measures for the same weights and draft
# (77 tok/s with the drafter, results/RESULTS.md); read them as a claim.
#
# YaRN here is static: SGLang scales RoPE the same way for every request, a
# 2K prompt included, which can cost a little quality at ordinary lengths.
# For work that stays under 262K the base profile on RadixArk's checkpoint
# (the same weights without YaRN) is the cleaner choice:
#   MODEL_DIR=/models/RadixArk/Qwen3.8-27B-NVFP4 ./serve.sh qwen3.8-27b
#
# Memory for 1M: ~42 KiB of fp8 KV per token (32 target + 10 draft), so one
# 1,048,576-token session needs ~42 GiB of the KV pool. The base split (0.80,
# cap 12) leaves roughly that; check the boot line "KV Cache is allocated ...
# #tokens": at or above 1048576 a full-length request fits. Below it, SGLang
# caps a request at the pool; MEM_FRACTION=0.85 and a lower MAX_RUNNING
# (README, "Long sessions") buy more.

MODEL_DIR="${MODEL_DIR:-/models/zenlm/zen6}"
DRAFT_DIR="${DRAFT_DIR:-$MODEL_DIR/dflash2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zen6}"

# The YaRN factor and original length in the checkpoint's config.json
# (wherever it nests: rope_scaling, rope_parameters, text_config), "" if none.
zen6_yarn() {
  python3 - "$MODEL_DIR/config.json" <<'EOF' 2>/dev/null
import json, sys

def walk(node):
    if isinstance(node, dict):
        if "yarn" in (node.get("rope_type"), node.get("type")) and node.get("factor"):
            yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)

for rope in walk(json.load(open(sys.argv[1]))):
    print(rope["factor"], rope.get("original_max_position_embeddings") or 262144)
    break
EOF
}

# Context: what the checkpoint's YaRN covers (factor x original), else the
# native 262144. A CONTEXT_LENGTH set in the shell wins (e.g. 524288).
read -r ZEN6_YARN_FACTOR ZEN6_YARN_ORIG <<<"$(zen6_yarn)" || true
if [ -z "${CONTEXT_LENGTH:-}" ]; then
  if [ -n "${ZEN6_YARN_FACTOR:-}" ]; then
    CONTEXT_LENGTH=$(awk -v f="$ZEN6_YARN_FACTOR" -v o="$ZEN6_YARN_ORIG" 'BEGIN { printf "%d", f * o }')
  else
    CONTEXT_LENGTH=262144
  fi
fi

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"

model_env() {
  if [ "$SPEC" != mtp ] && [ ! -f "$DRAFT_DIR/config.json" ]; then
    echo "zen6: no draft at $DRAFT_DIR (the card bundles it in dflash2/; set DRAFT_DIR)" >&2
    exit 1
  fi
  if [ -z "${ZEN6_YARN_FACTOR:-}" ] && [ "$CONTEXT_LENGTH" -gt 262144 ]; then
    echo "zen6: CONTEXT_LENGTH=$CONTEXT_LENGTH but $MODEL_DIR/config.json has no YaRN" \
      "rope scaling; past 262144 the model would read positions it was never trained on" >&2
    exit 1
  fi
  # YaRN is in the checkpoint, but SGLang derives the context from
  # max_position_embeddings (262144 here): allow the longer one
  [ -n "${ZEN6_YARN_FACTOR:-}" ] && export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  fp8_side_env
}

model_summary() {
  echo "zen6: YaRN $([ -n "${ZEN6_YARN_FACTOR:-}" ] && echo "x$ZEN6_YARN_FACTOR" || echo none), context $CONTEXT_LENGTH; $([ "$SPEC" = mtp ] && echo "MTP (NEXTN $MTP_STEPS/1/$MTP_DRAFT_TOKENS)" || echo "DFlash2 (bundled), $DRAFT_TOKENS draft tokens"); cap $MAX_RUNNING requests (GDN pool $MAMBA_CACHE)$(fp8_summary)"
}
