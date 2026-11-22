# Qwen3.8-Flash-Next (Intel AutoRound int4 hybrid) on stock vLLM 0.34, one DGX Spark.
#
# Built by ./build.sh from a context it assembles (this file, ../vllm-flash-ar/src/
# vllm_fp8_hybrid.py and patch_mtp_draft_vocab.py, src/patch_lm_head_quant.py); BASE is
# vllm/vllm-openai@<image.lock>.
# vLLM 0.34 serves this model (Qwen4Exp) with its n-gram table read from disk
# (--engram-config disk_offload) natively; what this checkpoint needs on top:
#   1. the int4+fp8 hybrid dispatch: the GPTQ config's excluded layers (GDN / attention
#      projections, shared experts) are stored as blockwise FP8 (weight + weight_scale_inv,
#      128x128), which the stock AutoGPTQConfig would build as BF16 and fail to load.
#      ../vllm-flash-ar's patch, unchanged: 0.34's AutoGPTQConfig has the same three methods
#      (maybe_update_config, apply_vllm_mapper, get_quant_method). Off unless VLLM_FP8_HYBRID=1.
#   2. the int8 GPTQ lm_head: 0.34 builds Qwen4Exp's heads without a quant_config.
#   3. the private MTP draft head (DRAFT_VOCAB / DRAFT_HEAD in config.sh): ../vllm-flash-ar's
#      patch 10, the same file; it finds Qwen4ExpMTP and 0.34's Marlin helper signatures
#      itself. Inert unless VLLM_MTP_DRAFT_VOCAB / VLLM_MTP_DRAFT_HEAD is set.
ARG BASE=vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${BASE}

COPY vllm_fp8_hybrid.py patch_lm_head_quant.py patch_mtp_draft_vocab.py /tmp/gb10/
RUN set -eu; \
    python3 -c 'import vllm; print("vLLM", vllm.__version__)'; \
    SP="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
    cp /tmp/gb10/vllm_fp8_hybrid.py "$SP/vllm_fp8_hybrid.py"; \
    GPTQ="$SP/vllm/model_executor/layers/quantization/auto_gptq.py"; \
    grep -q "def maybe_update_config" "$GPTQ"; grep -q "def get_quant_method" "$GPTQ"; \
    cp "$GPTQ" "$GPTQ.orig"; \
    printf '\n\n# --- gb10: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1; vllm-flash-ar/src/vllm_fp8_hybrid.py)\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> "$GPTQ"; \
    python3 -c 'import ast, sys; ast.parse(open(sys.argv[1]).read())' "$GPTQ"; \
    echo "auto_gptq.py: fp8 hybrid hook OK"; \
    Q="$SP/vllm/models/qwen4_exp/nvidia"; \
    cp "$Q/model.py" "$Q/model.py.orig"; cp "$Q/mtp.py" "$Q/mtp.py.orig"; \
    python3 /tmp/gb10/patch_lm_head_quant.py "$Q/model.py" "$Q/mtp.py"; \
    echo "lm_head quant_config OK"; \
    python3 /tmp/gb10/patch_mtp_draft_vocab.py "$Q/mtp.py"; \
    grep -q "^    Qwen4ExpMTP.compute_logits = _dv_compute_logits" "$Q/mtp.py"; \
    echo "mtp.py: private draft head hook OK"; \
    rm -rf /tmp/gb10
