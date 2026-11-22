# Qwen3.8-Flash-Next (Intel AutoRound int4 hybrid) on stock vLLM 0.34, one DGX Spark.
#
# Built by ./build.sh from a context it assembles (this file, ../vllm-flash-ar/src/
# vllm_fp8_hybrid.py, patch_mtp_draft_vocab.py, gb10_fp8_rowwise.py, patch_hc_fp8.py,
# gb10_abliterate.py and patch_abliterate.py, src/patch_lm_head_quant.py, check_tool_parser.py,
# patches/*.v034.patch, ../vllm-flash-ar/src/patch_det_moe_align.py, src/patch_qsa_topk.py);
# BASE is vllm/vllm-openai@<image.lock>.
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
#   7. determinism: the MoE token alignment in a fixed order (../vllm-flash-ar's patch 15, the
#      same file, which finds 0.34's implementation behind its @overload stubs; on with
#      VLLM_DET_MOE_ALIGN=1) and a deterministic QSA block top-k (src/patch_qsa_topk.py, the
#      dev image's patch 9 ported to 0.34's _topk: @jschmied's kernel, built below, with
#      VLLM_QSA_DET_TOPK=1, the exact torch.topk with VLLM_QSA_EXACT_TOPK=1 or as its fallback).
ARG BASE=vllm/vllm-openai@sha256:20fdb089454afbb324efa4110a2c025ff7b27cc1a086e07aa580b18df8ad1ce1
FROM ${BASE}

# @jschmied's deterministic persistent_topk (vllm#55122), built as a standalone extension
# (_C_det.so) with the base image's nvcc: the same sources, commit and checksums as
# ../vllm-flash-ar's Dockerfile (Apache-2.0; attribution: @jschmied). Its own layer, so a
# change to the patches below does not rebuild it. If it cannot be built here (no nvcc, a
# compile error) the image still builds: VLLM_QSA_DET_TOPK=1 then logs a warning and takes
# the exact top-k, deterministic but slower. DET_ARCH=120a for x86 Blackwell (RTX 5090).
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ARG DET_ARCH=121a
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
RUN set -u; cd /opt/llm/kernel-det/src; export PATH="/usr/local/cuda/bin:$PATH"; \
    if ! command -v nvcc >/dev/null; then \
      echo "DET_TOPK kernel NOT built: no nvcc in the base image (VLLM_QSA_DET_TOPK=1 falls back to the exact top-k)"; \
    elif DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py > /opt/llm/kernel-det/build.log 2>&1 \
         && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so; then \
      echo "DET_TOPK kernel built OK ($(nvcc --version | tail -1))"; \
    else \
      tail -25 /opt/llm/kernel-det/build.log; \
      echo "DET_TOPK kernel NOT built (log: /opt/llm/kernel-det/build.log; VLLM_QSA_DET_TOPK=1 falls back to the exact top-k)"; \
    fi

COPY vllm_fp8_hybrid.py patch_lm_head_quant.py patch_mtp_draft_vocab.py gb10_fp8_rowwise.py patch_hc_fp8.py \
     gb10_abliterate.py patch_abliterate.py check_tool_parser.py \
     qwen-tool-preamble.v034.patch qwen-tool-marker-guard.v034.patch \
     patch_det_moe_align.py patch_qsa_topk.py /tmp/gb10/
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
    python3 /tmp/gb10/patch_det_moe_align.py "$SP/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"; \
    grep -q "and expert_map is None and scatter_idx is None:" "$SP/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"; \
    python3 /tmp/gb10/patch_qsa_topk.py "$Q/ops/qsa_indexer.py"; \
    grep -q "topk_op = _GB10_QSA_DET_OP" "$Q/ops/qsa_indexer.py"; \
    echo "determinism: MoE align + QSA top-k OK ($([ -f /opt/llm/kernel-det/_C_det.so ] && echo 'deterministic kernel' || echo 'exact top-k only'))"; \
    rm -rf /tmp/gb10
# serve.sh checks these before it passes DRAFT_VOCAB / DRAFT_HEAD / HC_FP8 / ABLATE (an older image lacks the hook)
LABEL gb10.draft-head="1" gb10.hc-fp8="1" gb10.ablate="1" gb10.tool-parser="1" gb10.det="2"
