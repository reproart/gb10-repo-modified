# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Qwen3.8-Flash-Next with the abliterated weights
# (edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4).
#
#   ./serve.sh qwen3.8-flash-next-abliterated
#
# The same as `WEIGHTS=abliterated ./serve.sh qwen3.8-flash-next`, as a profile
# of its own so the systemd unit (scripts/install-service.sh) can serve it.
# Everything else, and every knob, comes from qwen3.8-flash-next.sh; the
# weights directory and the served name are chosen there.
#
# Weights, once (the card: TP1 on one Spark checked with text, code, tool
# calls and images; changes recovered from quantized GGUF weights, so not an
# exact BF16 reconstruction, and not benchmarked by its author):
#   hf download edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4 \
#     --local-dir /models/edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4
# The MTP head is RadixArk's own while the trunk changed: accept_len may be a
# little below the RadixArk build's 3.5-3.8.

WEIGHTS=abliterated

# shellcheck source=qwen3.8-flash-next.sh
. "$ROOT/models/qwen3.8-flash-next.sh"
