"""FP8 weights for Qwen4-Exp's hyper-connection mix.

Every decoder layer mixes its 4 residual streams twice (before attention and
before the MLP) through a low-rank gate: input_mix_weight_down [320 x 10240]
and input_mix_weight_up [10240 x 320], BF16, 13 MB a mix. SGLang runs it in
one persistent Triton kernel (layers/hc_mix_triton.py) that is bound by
reading those weights: on GB10 ~67 us a call, ~103 calls a decode step
(96 in the target, the model's final mixer, the MTP layer per draft step),
~7 ms of ~57, and no other kernel overlaps it.

This patch stores those two weights in FP8 E4M3 with one float32 scale per
row (amax / 448) after loading, and routes the mix through a copy of the
kernel that reads FP8 (gb10_fp8_hc_kernel): half the bytes. The copy takes
up to 64 rows (SGLang's stops at 16, and runs more through torch.compile):
a verify at 8 requests is 32. Longer inputs (prefill) take a torch path
that converts each weight to BF16 once per call and applies the row scales
to the outputs. block_inject_weight [4 x 10240] is left alone.

Enabled by GB10_FP8_HC=1 (models/qwen3.8-flash-next.sh: FP8_HC=1). Lossy:
the gate that mixes the residual streams moves by per-channel FP8 error;
check answers. Written against SGLang 0.5.20.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_fp8_hc")

HC_MODULE = "sglang.srt.layers.hyperconnection"
TARGET_MODULE = "sglang.srt.models.qwen4_exp"
MTP_MODULE = "sglang.srt.models.qwen4_exp_mtp"
MAX_ROWS = 64      # gb10_fp8_hc_kernel.MAX_ROWS (tiles of 16, 32 or 64 rows)
K_TILE = 256       # its BLOCK_K; the down projection's K is not masked
WEIGHTS = ("input_mix_weight_down", "input_mix_weight_up")

_patched = {"hc": False}
_wide = {"ok": True}   # the 32/64-row tiles built and ran


def scale_of(weight):
    return getattr(weight, "_gb10_scale", None)


def _kernel(*args, **kwargs):
    from gb10_fp8_hc_kernel import fused_hc_mix_fp8

    return fused_hc_mix_fp8(*args, **kwargs)


def mix_reference(x, w_down, s_down, w_up, s_up, hc, hs):
    """SGLang's _mix_compute on the FP8 weights: the path for prefill-sized
    inputs. Each weight is converted to x's dtype once (exact: every E4M3
    value is a BF16 value) and its row scales multiply the GEMM's output
    columns, so no FP32 copy of a weight or of the [rows, 10240] gate is made."""
    import torch
    import torch.nn.functional as F

    t = F.linear(x, w_down.to(x.dtype))
    t.mul_(s_down.to(t.dtype)).div_(hc)
    t = F.silu(t)
    gate = F.linear(t, w_up.to(x.dtype))
    gate.mul_(s_up.to(gate.dtype))
    torch.sigmoid_(gate)
    return (gate.unflatten(-1, (hc, hs)) * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def make_wrappers(orig_supported, orig_mix, kernel=_kernel):
    import torch

    def fused_hc_mix_supported(x, w_down, w_up):
        if scale_of(w_down) is None:
            return orig_supported(x, w_down, w_up)
        # FP8 weights: always ours (the torch.compile fallback cannot read them)
        return x.dtype in (torch.bfloat16, torch.float16)

    def fused_hc_mix(x, w_down, w_up, hc, hs):
        s_down, s_up = scale_of(w_down), scale_of(w_up)
        if s_down is None:
            return orig_mix(x, w_down, w_up, hc, hs)
        rows = x.shape[0] if x.dim() == 2 else MAX_ROWS + 1
        if (x.is_cuda and rows <= MAX_ROWS and x.is_contiguous() and x.shape[1] % K_TILE == 0
                and (rows <= 16 or _wide["ok"])):
            if rows <= 16:
                return kernel(x, w_down, s_down, w_up, s_up, hc, hs)
            try:
                return kernel(x, w_down, s_down, w_up, s_up, hc, hs)
            except Exception as e:  # noqa: BLE001 - a tile that does not build on this GPU
                _wide["ok"] = False
                logger.warning("FP8 HC: the %d-row kernel failed (%s: %s); inputs over 16 "
                               "rows use the torch path", rows, type(e).__name__, e)
        return mix_reference(x, w_down, s_down, w_up, s_up, hc, hs)

    fused_hc_mix_supported._gb10_fp8_hc = True
    fused_hc_mix._gb10_fp8_hc = True
    return fused_hc_mix_supported, fused_hc_mix


def apply_hc(mod) -> None:
    """Hook for sglang.srt.layers.hyperconnection: it imported fused_hc_mix and
    fused_hc_mix_supported by name, so they are replaced in its namespace."""
    for name in ("fused_hc_mix", "fused_hc_mix_supported"):
        if not callable(getattr(mod, name, None)):
            raise RuntimeError(f"GB10_FP8_HC: {HC_MODULE}.{name} not found (written for "
                               "SGLang 0.5.20); unset GB10_FP8_HC.")
    if getattr(mod.fused_hc_mix, "_gb10_fp8_hc", False):
        return
    mod.fused_hc_mix_supported, mod.fused_hc_mix = make_wrappers(
        mod.fused_hc_mix_supported, mod.fused_hc_mix)
    _patched["hc"] = True
    logger.info("GB10_FP8_HC: hyper-connection mix reads FP8 weights where converted")


def convert_model(model, label="target") -> dict:
    """Quantize every hyper-connection mix's two weights in `model` to FP8."""
    import torch

    from gb10_fp8_side import quantize_per_channel

    if not _patched["hc"]:
        raise RuntimeError("GB10_FP8_HC: the hyper-connection module was not patched; "
                           "FP8 mix weights would reach code that cannot read them")
    stats = {"mixes": 0, "bytes_before": 0, "bytes_after": 0}
    for _, module in model.named_modules():
        lins = [getattr(module, w, None) for w in WEIGHTS]
        if not all(isinstance(lin, torch.nn.Linear) for lin in lins):
            continue
        if any(lin.weight.dtype != torch.bfloat16 or lin.bias is not None for lin in lins):
            continue
        for lin in lins:
            w = lin.weight
            stats["bytes_before"] += w.numel() * w.element_size()
            q, s = quantize_per_channel(w.data)
            lin.weight = torch.nn.Parameter(q, requires_grad=False)
            lin.weight._gb10_scale = s.contiguous()
            stats["bytes_after"] += q.numel() * q.element_size() + s.numel() * s.element_size()
        stats["mixes"] += 1
    torch.cuda.empty_cache()
    logger.info("FP8 HC (%s): %d hyper-connection mixes to FP8 (per-row scale); "
                "%.2f GiB -> %.2f GiB", label, stats["mixes"],
                stats["bytes_before"] / 2**30, stats["bytes_after"] / 2**30)
    return stats


def _wrap_load_weights(cls, label):
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        convert_model(self, label)
        return result

    cls.load_weights = load_weights


def apply_target(mod) -> None:
    if not hasattr(mod, "Qwen4ExpForConditionalGeneration"):
        raise RuntimeError("GB10_FP8_HC: Qwen4ExpForConditionalGeneration not found "
                           "(written for SGLang 0.5.20); unset GB10_FP8_HC.")
    _wrap_load_weights(mod.Qwen4ExpForConditionalGeneration, "target")


def apply_mtp(mod) -> None:
    if not hasattr(mod, "Qwen4ExpForCausalLMMTP"):
        raise RuntimeError("GB10_FP8_HC: Qwen4ExpForCausalLMMTP not found "
                           "(written for SGLang 0.5.20); unset GB10_FP8_HC.")
    _wrap_load_weights(mod.Qwen4ExpForCausalLMMTP, "MTP draft")
