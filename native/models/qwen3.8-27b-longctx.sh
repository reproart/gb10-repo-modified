# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B for a few full-length (262K) sessions held at once.
#
#   ./serve.sh qwen3.8-27b-longctx
#
# Everything comes from qwen3.8-27b.sh (weights, SGLang version, flags; edit
# MODEL_DIR there and all profiles follow) except the memory split.
#
# The budget (README, "Long sessions"), from the native boot logs:
#   pools = free at start - weights - (1 - MEM_FRACTION) x 121.6 GiB
#   one 262K session = 10.5 GiB of KV (32 KiB/token target + 10 KiB/token
#     draft, fp8) + ~0.0735 GiB per GDN slot x (slots + draft tokens)
# With the FP4-head RadixArk checkpoint (~25.9 GiB with the draft) and ~115 GiB
# free at start, 0.85 gives ~71 GiB of pools: 6 sessions at draft 10, 5 at
# draft 16. A 7th needs ~0.94, which the machine cannot give. The throughput
# profile's cap of 32 spends ~35 GiB on GDN state; 6 requests need ~7.
#
# Untested as a whole: check the "KV Cache is allocated ... #tokens" boot line
# (tokens / 262144 = sessions that fit), then bench/longctx.py (README).

MAX_RUNNING="${MAX_RUNNING:-6}"
# 10, not 16: the verify buffer grows with requests x draft tokens, and at 16
# the 6th session no longer fits at 0.85.
DRAFT_TOKENS="${DRAFT_TOKENS:-10}"
# 5 slots per running request, plus one per session so a finished turn's state
# can stay cached for the next (otherwise the next turn re-prefills up to 262K,
# ~4 minutes). ~0.44 GiB for the six extra slots.
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 6))}"
# 0.85: what 6 sessions need. Above it gains less than a session and eats into
# earlyoom's margin: if the scheduler dies with exit code -15
# (journalctl -u earlyoom), lower earlyoom's -m threshold or go back to 0.82.
# No --max-total-tokens cap here: the KV pool should take everything left.
MEM_FRACTION="${MEM_FRACTION:-0.85}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"
