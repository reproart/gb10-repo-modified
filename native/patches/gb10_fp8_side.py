"""FP8 weight-only for the dense layers a NVFP4 Qwen4-Exp checkpoint keeps in BF16.

A decode profile of Qwen3.8-Flash-Next on one GB10 (bench/profile_decode.py)
put ~40-48% of every step in BF16 GEMMs: the checkpoint quantizes only the
routed experts, and the rest (GDN in/out projections, attention qkv/o, the PLE
key/value projections, the shared expert) runs on cuBLAS, which picks sm80
WMMA 16x16 kernels on this GPU. SGLang's faster BF16 backends are SM90/SM100
only. The int4 vLLM recipe for this model carries exactly these layers in FP8.

This patch does that at load time, without a new checkpoint: after the
weights are loaded, each matching BF16 linear layer is quantized to FP8 E4M3
with one scale per output channel (amax / 448, weights only; activations stay
BF16) and served by SGLang's own FP8 Marlin GEMM (marlin_utils_fp8), the
kernel it uses for FP8 checkpoints on GPUs without FP8 tensor cores. Half the
bytes per step for those layers, a kernel built for small batches, and the
freed BF16 memory goes to the KV pool.

Enabled by GB10_FP8_SIDE=1 (models/qwen3.8-flash-next.sh: FP8_SIDE=1). Only
layers named in GB10_FP8_SIDE_LAYERS (a regex on the module name; default
below) are converted, and only if their shape fits Marlin (N % 64, K % 128);
every conversion and skip is logged. GDN's fused BF16 in_proj buffer is
dropped for converted layers, so both projections go through the FP8 path.
The target and, unless GB10_FP8_SIDE_MTP=0, the MTP draft are converted.

Lossy, like any FP8 quantization (per-channel FP8 of BF16 weights typically
moves logits very little; the vLLM recipe uses 128x128 blocks). Check
answers, not only tok/s. Written against SGLang 0.5.20.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_fp8_side")

TARGET_MODULE = "sglang.srt.models.qwen4_exp"
MTP_MODULE = "sglang.srt.models.qwen4_exp_mtp"

DEFAULT_LAYERS = (
    r"\.(in_proj_qkvz|in_proj_ba|out_proj|qkv_proj|o_proj|key_proj|value_proj"
    r"|shared_expert\.gate_up_proj|shared_expert\.down_proj)$"
)
FP8_MAX = 448.0
MIN_N, MIN_K = 64, 128  # Marlin's thread tile (marlin_utils.GPTQ_MARLIN_MIN_THREAD_*)


def enabled() -> bool:
    return os.environ.get("GB10_FP8_SIDE") == "1"


def layer_pattern() -> re.Pattern:
    return re.compile(os.environ.get("GB10_FP8_SIDE_LAYERS") or DEFAULT_LAYERS)


def quantize_per_channel(weight):
    """BF16 [N, K] -> (FP8 E4M3 [N, K], float32 scale [N]); w ~= q * scale[:, None]."""
    import torch

    w = weight.float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX
    q = (w / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


class Fp8MarlinSideMethod:
    """quant_method for a converted layer: FP8 Marlin weight-only GEMM."""

    def __init__(self, apply_fn):
        self._apply_fn = apply_fn

    def process_weights_after_loading(self, layer) -> None:  # already prepared
        return

    def apply(self, layer, x, bias=None):
        return self._apply_fn(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )


def convert_module(module, prepare_fn, apply_fn) -> None:
    """Quantize `module.weight` in place and route the layer through Marlin FP8."""
    import torch

    n, k = module.weight.shape
    q, scale = quantize_per_channel(module.weight.data)
    module.weight = torch.nn.Parameter(q, requires_grad=False)
    module.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
    module.orig_dtype = torch.bfloat16
    module.output_size_per_partition = n
    module.input_size_per_partition = k
    if getattr(module, "weight_block_size", None) is not None:
        module.weight_block_size = None
    prepare_fn(module, size_k_first=False)
    module.quant_method = Fp8MarlinSideMethod(apply_fn)


def convert_model(model, *, is_linear, is_unquantized, prepare_fn, apply_fn,
                  pattern=None, label="target") -> dict:
    """Convert every matching BF16 linear layer of `model`. Returns counters."""
    import torch

    pattern = pattern or layer_pattern()
    stats = {"converted": 0, "skipped": 0, "bytes_before": 0, "bytes_after": 0}
    touched_gdn = []
    for name, module in model.named_modules():
        if not (is_linear(module) and pattern.search(name)):
            continue
        qm = getattr(module, "quant_method", None)
        w = getattr(module, "weight", None)
        if not is_unquantized(qm) or w is None or w.dtype != torch.bfloat16 or w.dim() != 2:
            continue
        if getattr(module, "bias", None) is not None and module.bias.dtype != torch.bfloat16:
            continue
        n, k = w.shape
        if n % MIN_N or k % MIN_K:
            stats["skipped"] += 1
            logger.info("FP8 side (%s): %s [%d x %d] left in BF16 (Marlin needs N%%%d, K%%%d)",
                        label, name, n, k, MIN_N, MIN_K)
            continue
        before = w.numel() * w.element_size()
        convert_module(module, prepare_fn, apply_fn)
        stats["converted"] += 1
        stats["bytes_before"] += before
        stats["bytes_after"] += module.weight.numel() * module.weight.element_size()
        if name.endswith((".in_proj_qkvz", ".in_proj_ba")):
            touched_gdn.append(name.rsplit(".", 1)[0])
    # GDN's fused BF16 in_proj buffer bypasses quant_method; drop it where
    # its halves were converted, so both go through the FP8 layers.
    # A half left in BF16 is a view of that buffer: give it its own storage,
    # or the whole buffer (with the converted half's old BF16 rows) stays.
    mods = dict(model.named_modules())
    for parent in set(touched_gdn):
        gdn = mods.get(parent)
        if gdn is None or getattr(gdn, "_fused_in_proj_weight", None) is None:
            continue
        gdn._fused_in_proj_weight = None
        for half in ("in_proj_qkvz", "in_proj_ba"):
            lin = getattr(gdn, half, None)
            if lin is not None and lin.weight.dtype == torch.bfloat16:
                lin.weight.data = lin.weight.data.clone()
    torch.cuda.empty_cache()
    logger.info(
        "FP8 side (%s): %d linear layers to FP8 weight-only (Marlin), %d left in BF16; "
        "%.2f GiB -> %.2f GiB", label, stats["converted"], stats["skipped"],
        stats["bytes_before"] / 2**30, stats["bytes_after"] / 2**30,
    )
    return stats


def _sglang_parts():
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        apply_fp8_marlin_linear,
        prepare_fp8_layer_for_marlin,
    )
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return dict(
        is_linear=lambda m: isinstance(m, LinearBase),
        is_unquantized=lambda qm: isinstance(qm, UnquantizedLinearMethod),
        prepare_fn=prepare_fp8_layer_for_marlin,
        apply_fn=apply_fp8_marlin_linear,
    )


def _wrap_load_weights(cls, label):
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        convert_model(self, label=label, **_sglang_parts())
        return result

    cls.load_weights = load_weights


def apply_target(mod) -> None:
    if not hasattr(mod, "Qwen4ExpForConditionalGeneration"):
        raise RuntimeError("GB10_FP8_SIDE: Qwen4ExpForConditionalGeneration not found "
                           "(written for SGLang 0.5.20); unset GB10_FP8_SIDE.")
    _wrap_load_weights(mod.Qwen4ExpForConditionalGeneration, "target")
    logger.info("GB10_FP8_SIDE: target BF16 side layers will load as FP8 (Marlin)")


def apply_mtp(mod) -> None:
    if os.environ.get("GB10_FP8_SIDE_MTP", "1") != "1":
        return
    if not hasattr(mod, "Qwen4ExpForCausalLMMTP"):
        raise RuntimeError("GB10_FP8_SIDE: Qwen4ExpForCausalLMMTP not found "
                           "(written for SGLang 0.5.20); set GB10_FP8_SIDE_MTP=0.")
    _wrap_load_weights(mod.Qwen4ExpForCausalLMMTP, "MTP draft")
