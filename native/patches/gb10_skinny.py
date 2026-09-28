"""A faster BF16 GEMM for the small layers of Qwen4-Exp that must stay BF16.

After gb10_fp8_side the decode profile of Qwen3.8-Flash-Next on GB10 still
had ~64 BF16 GEMMs a step at ~46 us each (~2.9 ms of ~59): cuBLAS runs them
on sm80 WMMA kernels (cutlass_80_wmma ... 128x2) that read the 2.6-3.3 MB
weight at ~60 GB/s. The boot log's "still in BF16" list names them: the MoE
routers (mlp.gate [512 x 2560], 48 + 1 MTP) and the QSA indexer projections
(indexer.index_qk_proj [640 x 2560], 12 + 1 MTP). Both pick something
(experts, attended tokens), so they are not quantized: the weights stay
BF16 and only the kernel changes. gb10_skinny_kernel's Triton GEMM is FP32
accumulation like cuBLAS; results can differ from it in the last bit, as
cuBLAS's own differ between batch sizes.

Enabled by GB10_SKINNY_BF16=1 (models/qwen3.8-flash-next.sh: SKINNY_BF16=1).
Layers: GB10_SKINNY_LAYERS, a regex on the module name (default below). Used
for inputs of up to GB10_SKINNY_MAX_M rows (default 64: decode and verify
batches); longer inputs (prefill) keep the original path. Written against
SGLang 0.5.20.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_skinny")

TARGET_MODULE = "sglang.srt.models.qwen4_exp"
MTP_MODULE = "sglang.srt.models.qwen4_exp_mtp"
DEFAULT_LAYERS = r"\.(mlp\.gate|indexer\.index_qk_proj)$"
MIN_K = 16  # tl.dot's smallest K


def layer_pattern() -> re.Pattern:
    return re.compile(os.environ.get("GB10_SKINNY_LAYERS") or DEFAULT_LAYERS)


def max_m() -> int:
    return int(os.environ.get("GB10_SKINNY_MAX_M", "64"))


def _kernel(x, w):
    from gb10_skinny_kernel import skinny_gemm

    return skinny_gemm(x, w)


class SkinnyBf16Method:
    """quant_method wrapper: the Triton GEMM for few-row inputs, the layer's
    original method otherwise (and for anything unusual: bias, 3-D input)."""

    def __init__(self, orig, limit, gemm=_kernel):
        self.orig = orig
        self.limit = limit
        self._gemm = gemm

    def __getattr__(self, name):  # anything else the model asks the method
        orig = self.__dict__.get("orig")
        if orig is None:
            raise AttributeError(name)
        return getattr(orig, name)

    def process_weights_after_loading(self, layer) -> None:
        return

    def apply(self, layer, x, bias=None):
        import torch

        w = layer.weight
        if (bias is not None or x.dim() != 2 or x.shape[0] > self.limit
                or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16):
            return self.orig.apply(layer, x, bias)
        if x.stride(1) != 1:
            x = x.contiguous()
        return self._gemm(x, w)


def convert_model(model, *, is_linear, is_unquantized, pattern=None, label="target",
                  gemm=_kernel) -> list:
    """Route matching BF16 linear layers of `model` through SkinnyBf16Method.
    Returns the converted names."""
    import torch

    pattern = pattern or layer_pattern()
    limit = max_m()
    done = []
    for name, module in model.named_modules():
        if not (is_linear(module) and pattern.search(name)):
            continue
        w = getattr(module, "weight", None)
        qm = getattr(module, "quant_method", None)
        if (not is_unquantized(qm) or w is None or w.dim() != 2 or w.dtype != torch.bfloat16
                or w.shape[1] % MIN_K or getattr(module, "bias", None) is not None):
            continue
        if not w.is_contiguous():
            module.weight.data = w.data.contiguous()
        module.quant_method = SkinnyBf16Method(qm, limit, gemm)
        done.append(name)
    shapes, mods = {}, dict(model.named_modules())
    for name in done:
        key = (re.sub(r"\.\d+\.", ".N.", name), tuple(mods[name].weight.shape))
        shapes[key] = shapes.get(key, 0) + 1
    logger.info("BF16 skinny GEMM (%s): %d layers up to %d rows%s", label, len(done), limit,
                "".join(f"; {n} [{s[0]} x {s[1]}] x{c}" for (n, s), c in shapes.items()))
    return done


def _sglang_parts():
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return dict(is_linear=lambda m: isinstance(m, LinearBase),
                is_unquantized=lambda qm: isinstance(qm, UnquantizedLinearMethod))


def _wrap_load_weights(cls, label):
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        convert_model(self, label=label, **_sglang_parts())
        return result

    cls.load_weights = load_weights


def apply_target(mod) -> None:
    if not hasattr(mod, "Qwen4ExpForConditionalGeneration"):
        raise RuntimeError("GB10_SKINNY_BF16: Qwen4ExpForConditionalGeneration not found "
                           "(written for SGLang 0.5.20); unset GB10_SKINNY_BF16.")
    _wrap_load_weights(mod.Qwen4ExpForConditionalGeneration, "target")


def apply_mtp(mod) -> None:
    if not hasattr(mod, "Qwen4ExpForCausalLMMTP"):
        raise RuntimeError("GB10_SKINNY_BF16: Qwen4ExpForCausalLMMTP not found "
                           "(written for SGLang 0.5.20); unset GB10_SKINNY_BF16.")
    _wrap_load_weights(mod.Qwen4ExpForCausalLMMTP, "MTP draft")
