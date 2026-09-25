# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B for many short concurrent requests (up to 32).
#
#   ./serve.sh qwen3.8-27b-throughput
#
# Everything comes from qwen3.8-27b.sh (weights, SGLang version, flags; edit
# MODEL_DIR there and all profiles follow) except the cap. This was the
# default before the base profile moved to 12: the max-aggregate config every
# "@ 32" figure in results/ was measured at. Natively, RadixArk NVFP4:
# 70.5 tok/s single-stream, 597.8 tok/s peak at 32 streams.
#
# The cap reserves ~35 GiB of GDN state and verify buffer whether or not 32
# requests ever arrive; the KV pool is ~520K tokens smaller than at cap 12.
# Only worth it if more than ~16 requests really run at once.

MAX_RUNNING="${MAX_RUNNING:-32}"
# 5 per request, as measured (no extra cached-state slot at this size).
MAMBA_CACHE="${MAMBA_CACHE:-$((MAX_RUNNING * 5))}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"
