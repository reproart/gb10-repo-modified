# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B tuned for one stream at a time (one user or agent).
#
#   ./serve.sh qwen3.8-27b-single
#
# Everything comes from qwen3.8-27b.sh (weights, SGLang version, flags; edit
# MODEL_DIR there and both profiles follow) except two defaults:
#   DRAFT_TOKENS 16, the single-stream optimum of the draft sweep
#   MAX_RUNNING  16, because the DFlash2 verify buffer grows with
#                requests x draft tokens (16 x 16 ~ 18 GB, 32 x 16 ~ 37 GB)
#
# Measured natively on the Uncensored NVFP4 finetune (results/RESULTS.md,
# "Native build"): 70.9 tok/s single-stream against 56.4 for the default
# 10 / 32 profile, +26%; peak aggregate 375 tok/s at 16 streams against 558 at
# 32. More than a few concurrent requests: use qwen3.8-27b.

DRAFT_TOKENS="${DRAFT_TOKENS:-16}"
MAX_RUNNING="${MAX_RUNNING:-16}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"
