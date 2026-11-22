# Qwen3.8-Flash-Next (Intel AutoRound int4 hybrid) on stock vLLM 0.34, one DGX Spark.
#
# Built by ./build.sh from a context it assembles (this file, ../vllm-flash-ar/src/
# vllm_fp8_hybrid.py, patch_mtp_draft_vocab.py, gb10_fp8_rowwise.py, patch_hc_fp8.py,
# gb10_abliterate.py and patch_abliterate.py, src/patch_lm_head_quant.py, check_tool_parser.py,
# patches/*.v034.patch); BASE is vllm/vllm-openai@<image.lock>.
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
#   4. FP8 hyper-connection projections (HC_FP8 in config.sh): ../vllm-flash-ar's patch 16,
#      the same files; on 0.34 a converted layer also turns off GatedResidual's fused kernel
#      that reads its weight as BF16 (down+SiLU). Inert unless VLLM_HC_FP8=1|up.
#   5. the refusal-direction ablation (ABLATE in config.sh): ../vllm-flash-ar's patch 17, the
#      same files, with 0.34's own insertion points (picked by class name). Inert unless
#      VLLM_ABLATE is set.
#   6. the tool-call parser fixes (blazux's 12 and 13, ../vllm-flash-ar's patch 14), ported to
#      0.34's parser engine (src/patches/*.v034.patch): a quoted or fenced tool-call example stays
#      text instead of becoming a call that drops the answer. On for the qwen3 configuration
#      (--tool-call-parser qwen3_xml); check_tool_parser.py runs the cases at build time.
ARG BASE=vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${BASE}

COPY vllm_fp8_hybrid.py patch_lm_head_quant.py patch_mtp_draft_vocab.py gb10_fp8_rowwise.py patch_hc_fp8.py \
     gb10_abliterate.py patch_abliterate.py check_tool_parser.py \
     qwen-tool-preamble.v034.patch qwen-tool-marker-guard.v034.patch /tmp/gb10/
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
    cp /tmp/gb10/gb10_fp8_rowwise.py "$SP/gb10_fp8_rowwise.py"; \
    cp "$Q/hyperconnection.py" "$Q/hyperconnection.py.orig"; \
    python3 /tmp/gb10/patch_hc_fp8.py "$Q/hyperconnection.py"; \
    grep -q "_gb10_fp8_rowwise.install_hc(" "$Q/hyperconnection.py"; \
    grep -q "_use_hc_down_silu" "$Q/hyperconnection.py"; \
    python3 -c "import gb10_fp8_rowwise"; \
    echo "hyperconnection.py: FP8 HC hook OK"; \
    cp /tmp/gb10/gb10_abliterate.py "$SP/gb10_abliterate.py"; \
    python3 /tmp/gb10/patch_abliterate.py "$Q/model.py" "$Q/mtp.py"; \
    grep -q "^Qwen4ExpDecoderLayer._gb10_abl_on = False" "$Q/model.py"; \
    grep -q "prev_block_output = _gb10_abl_project(" "$Q/mtp.py"; \
    python3 -c "import gb10_abliterate"; \
    echo "model.py, mtp.py: ablation hook OK"; \
    command -v patch >/dev/null || { apt-get update -qq && apt-get install -y -qq --no-install-recommends patch >/dev/null; }; \
    (cd "$SP" && patch --batch --forward --fuzz=0 -p1 < /tmp/gb10/qwen-tool-preamble.v034.patch \
              && patch --batch --forward --fuzz=0 -p1 < /tmp/gb10/qwen-tool-marker-guard.v034.patch); \
    python3 /tmp/gb10/check_tool_parser.py; \
    echo "tool-call parser: preamble + marker guard OK"; \
    rm -rf /tmp/gb10
# serve.sh checks these before it passes DRAFT_VOCAB / DRAFT_HEAD / HC_FP8 / ABLATE (an older image lacks the hook)
LABEL gb10.draft-head="1" gb10.hc-fp8="1" gb10.ablate="1" gb10.tool-parser="1"
