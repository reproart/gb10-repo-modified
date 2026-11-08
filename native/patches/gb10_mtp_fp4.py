"""A torch NVFP4 quantizer for SGLang's load-time NVFP4 conversion, as a
fallback for the MTP draft's experts.

RadixArk/Qwen3.8-Flash-Next-NVFP4 keeps the MTP layer's 512 routed experts in
BF16 (the checkpoint's exclude list has mtp.*): ~5 GiB, and the draft runs
them on flashinfer's BF16 CUTLASS MoE, ~0.77 ms a call, 3 calls a decode step
(one draft extend + two draft steps at MTP 3/1/4): ~2.3 ms of ~54. SGLang
0.5.21 can quantize them while loading: an explicit
--speculative-draft-model-quantization modelopt_fp4 on a serialized NVFP4
checkpoint whose MTP experts are excluded gives the draft
ModelOptFp4Config.for_online_weight_quantization() (model_loader/
weight_utils.py, _resolve_explicit_draft_quant_config), and each expert
shard is converted to NVFP4 as it loads (layers/quantization/nvfp4_online.py).
With --speculative-moe-runner-backend marlin those experts run as W4A16 on
SGLang's NVFP4 Marlin MoE (the kernel MOE_RUNNER_BACKEND=marlin runs on GB10),
so no activation scale is needed: the online path has no calibrated one and
falls back to 1.0, which a W4A4 kernel would use.
models/qwen3.8-flash-next.sh: MTP_EXPERTS=fp4 passes both flags.

The conversion calls flashinfer's nvfp4_quantize with backend="cute-dsl",
which may not build for GB10 (SM121). This patch wraps
ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4: GB10_MTP_FP4_QUANT
  auto   flashinfer first; on an exception, this file's torch quantizer for
         that call and every later one (logged once). The default.
  torch  the torch quantizer only.
The torch quantizer does what the loader asks of flashinfer during the
conversion (FLASHINFER_NVFP4_4OVER6=1, ERR_MODE=MSE, E4M3 max 448): per
16-element block, the scale that maps the block's max to 6 or the one that
maps it to 4, whichever leaves the smaller squared error; E2M1 values rounded
to nearest, ties to even; two values a byte, the first in the low nibble;
block scales FP8 E4M3 in the linear [rows, cols / 16] layout; the per-tensor
scale amax / (448 * 6).

The draft only proposes tokens: a worse draft costs acceptance, never an
answer. Enabled by GB10_MTP_FP4=1. Written against SGLang 0.5.21.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("sglang.srt.layers.quantization.gb10_mtp_fp4")

TARGET_MODULE = "sglang.srt.layers.quantization.nvfp4_online"
E4M3_MAX = 448.0
E2M1_MAX = 6.0
BLOCK = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_state = {"torch": False, "calls": 0}


def mode() -> str:
    m = os.environ.get("GB10_MTP_FP4_QUANT", "auto")
    if m not in ("auto", "torch"):
        raise RuntimeError(f"GB10_MTP_FP4_QUANT must be auto or torch, not {m!r}")
    return m


def e2m1_codes(a):
    """|values| already scaled to the E2M1 range -> codes 0..7 (round to
    nearest, ties to the even code, saturating at 6)."""
    import torch

    a = a.abs()
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75).to(torch.uint8) + (a > 1.25).to(torch.uint8)
            + (a >= 1.75).to(torch.uint8) + (a > 2.5).to(torch.uint8) + (a >= 3.5).to(torch.uint8)
            + (a > 5.0).to(torch.uint8))
    return code


def _quantize_blocks(blocks, weight_scale_2, target_max):
    """blocks [rows, nb, 16] float32 -> (codes with sign [rows, nb, 16] uint8,
    block scales E4M3 [rows, nb], the blocks dequantized as SGLang will)."""
    import torch

    gs = 1.0 / weight_scale_2   # the global encode scale
    bmax = blocks.abs().amax(dim=-1)
    sf = (bmax / target_max * gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    inv = torch.where(sf_f > 0, gs / sf_f, torch.zeros_like(sf_f))
    scaled = blocks * inv[..., None]
    code = e2m1_codes(scaled)
    values = torch.tensor(E2M1_VALUES, device=blocks.device)[code.long()]
    sign = scaled < 0
    deq = torch.where(sign, -values, values) * (sf_f * weight_scale_2)[..., None]
    code = code | (sign.to(torch.uint8) << 3)
    return code, sf, deq


def quantize_nvfp4_torch(weight, weight_scale_2=None):
    """BF16/FP16/FP32 [rows, cols] -> (packed uint8 [rows, cols / 2], block scales
    float8_e4m3fn [rows, cols / 16], weight_scale_2 float32 scalar): the return of
    ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4. Dequantized:
    value(code) * scale / (1 / weight_scale_2)."""
    import torch

    if weight.ndim != 2:
        raise ValueError(f"NVFP4: expects a 2D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % BLOCK:
        raise ValueError(f"NVFP4: K must be a multiple of {BLOCK}, got shape {tuple(weight.shape)}")
    w = weight.float().nan_to_num()
    if weight_scale_2 is None:
        amax = w.abs().amax()
        weight_scale_2 = torch.where(amax > 0, amax / (E4M3_MAX * E2M1_MAX), torch.ones_like(amax))
    else:
        weight_scale_2 = weight_scale_2.to(device=w.device, dtype=torch.float32)
    blocks = w.view(rows, cols // BLOCK, BLOCK)
    code6, sf6, deq6 = _quantize_blocks(blocks, weight_scale_2, 6.0)
    code4, sf4, deq4 = _quantize_blocks(blocks, weight_scale_2, 4.0)
    use4 = ((deq4 - blocks) ** 2).sum(-1) < ((deq6 - blocks) ** 2).sum(-1)
    code = torch.where(use4[..., None], code4, code6)
    sf = torch.where(use4, sf4.view(torch.uint8), sf6.view(torch.uint8)).view(torch.float8_e4m3fn)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.reshape(rows, cols // 2).contiguous(), sf.contiguous(), weight_scale_2


def dequantize_nvfp4(packed, sf, weight_scale_2, dtype=None):
    """The inverse, for tests and checks: [rows, cols] float32 (or dtype)."""
    import torch

    rows, half = packed.shape
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    code = torch.stack((low, high), dim=-1).reshape(rows, half * 2)
    values = torch.tensor(E2M1_VALUES, device=packed.device)[(code & 0x07).long()]
    values = torch.where((code & 0x08) > 0, -values, values)
    scale = sf.float() * weight_scale_2.float()
    out = (values.view(rows, -1, BLOCK) * scale[..., None]).reshape(rows, half * 2)
    return out if dtype is None else out.to(dtype)


def make_wrapper(orig):
    def _quantize_weight_nvfp4(weight, weight_scale_2=None):
        _state["calls"] += 1
        if _state["torch"] or mode() == "torch":
            if not _state["torch"]:
                _state["torch"] = True
                logger.info("GB10_MTP_FP4: load-time NVFP4 conversion on the torch quantizer "
                            "(GB10_MTP_FP4_QUANT=torch)")
            return quantize_nvfp4_torch(weight, weight_scale_2)
        try:
            return orig(weight, weight_scale_2)
        except Exception as e:  # noqa: BLE001 - flashinfer's cute-dsl kernel may not build here
            _state["torch"] = True
            logger.warning("GB10_MTP_FP4: flashinfer's NVFP4 quantizer failed (%s: %s); "
                           "converting on the torch quantizer instead",
                           type(e).__name__, str(e).splitlines()[0][:200] if str(e) else "")
            return quantize_nvfp4_torch(weight, weight_scale_2)

    _quantize_weight_nvfp4._gb10_mtp_fp4 = True
    return staticmethod(_quantize_weight_nvfp4)


def apply(mod) -> None:
    cls = getattr(mod, "ModelOptNvFp4OnlineFusedMoEMethod", None)
    orig = getattr(cls, "_quantize_weight_nvfp4", None) if cls is not None else None
    if not callable(orig):
        raise RuntimeError(f"GB10_MTP_FP4: {TARGET_MODULE}.ModelOptNvFp4OnlineFusedMoEMethod."
                           "_quantize_weight_nvfp4 not found (written for SGLang 0.5.21); "
                           "unset GB10_MTP_FP4 or set MTP_EXPERTS=bf16.")
    if getattr(orig, "_gb10_mtp_fp4", False):
        return
    mode()   # refuse a bad setting at import, not at the first expert
    cls._quantize_weight_nvfp4 = make_wrapper(orig)
    logger.info("GB10_MTP_FP4: load-time NVFP4 conversion wrapped (GB10_MTP_FP4_QUANT=%s)", mode())
