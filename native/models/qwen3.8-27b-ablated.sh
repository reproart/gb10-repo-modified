# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-27B (RadixArk NVFP4 + DFlash2) with the refusal directions
# ablated at inference (patches/gb10_ablate.py).
#
#   ./serve.sh qwen3.8-27b-ablated
#
# The same as `ABLATE=1 ./serve.sh qwen3.8-27b`, as a profile of its own so the
# systemd unit (scripts/install-service.sh) can serve it, under its own served
# name so clients see which one answers (and, like every profile, also as
# SERVED_ALIASES, "qwen" by default). The vectors file is ABLATE_VECTORS in
# qwen3.8-27b.sh (download it once, see there); ABLATE_ALPHA, ABLATE_AT and
# every other knob come from there too. ABLATE=<path> here picks another file.

ABLATE="${ABLATE:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-27b-ablated}"

# shellcheck source=qwen3.8-27b.sh
. "$ROOT/models/qwen3.8-27b.sh"
