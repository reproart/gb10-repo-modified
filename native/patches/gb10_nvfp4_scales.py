"""Per-shard global scales for NVFP4 layers that SGLang runs fused on Marlin.

ModelOpt quantizes each Linear of a Hugging Face checkpoint on its own, with
its own FP32 global scale (weight_scale_2). SGLang then fuses some of them
into one GEMM: q/k/v, a GDN block's in_proj_qkv + in_proj_z, a shared
expert's gate + up, and every routed expert's gate (w1) + up (w3). On the
Marlin path (W4A16 checkpoints; SGLang 0.5.20) one global scale serves the
whole fused layer:

  * dense (ModelOptNvFp4A16LinearMethod): the max over the shards, with only
    a warning ("weight_scale_2 differs across fused parallel layers"), so
    shard i comes out g_max / g_i times too large;
  * MoE (ModelOptNvFp4FusedMoEMethod, marlin): the gate's scale for both
    halves ("w1_weight_scale_2 must match w3_weight_scale_2"), so up, and
    with it the expert's output, is off by g_gate / g_up.

When the shards' scales differ by a lot, the model's output is garbage (a
W4A16 Ornith-1.5-35B-A3B checkpoint answered one token repeated). Both are
corrected exactly, without touching the FP4 weights or their block scales:

  * dense: the GEMM runs as SGLang set it up (g_max), and output column
    block i is multiplied by g_i / g_max (bias added after);
  * MoE: silu(gate) * up is linear in up and the down projection is
    linear, so the expert's down global scale takes the factor
    g_up / g_gate.

Layers whose shards share one scale are left alone. Enabled by
GB10_NVFP4_SCALES=1. Written against SGLang 0.5.20.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sglang.srt.layers.quantization.gb10_nvfp4_scales")

TARGET_MODULE = "sglang.srt.layers.quantization.modelopt_quant"

_counts = {"dense": 0, "moe": 0}


def _note(kind: str, detail: str) -> None:
    _counts[kind] += 1
    n = _counts[kind]
    if n & (n - 1) == 0:  # 1, 2, 4, 8, ...: a few lines, the last one near the total
        logger.info("NVFP4 scales: %d %s layer(s) with per-shard global scales corrected "
                    "so far (latest: %s)", n, kind, detail)


def dense_column_scale(weight_scale_2, logical_widths):
    """(per-output-column factor g_i / g_max, g_max), or None when the shards
    share one scale or the widths do not match the scales."""
    import torch

    s = weight_scale_2.detach().float().reshape(-1)
    widths = [int(w) for w in logical_widths]
    if s.numel() <= 1 or s.numel() != len(widths):
        return None
    g = s.max()
    if bool(torch.all(s == g)):
        return None
    col = torch.cat([torch.full((w,), float(s[i] / g), dtype=torch.float32)
                     for i, w in enumerate(widths)])
    return col, g


def fix_dense(layer) -> bool:
    """Before SGLang's process_weights_after_loading: record the column
    factors and give every shard the max scale (which it would use anyway)."""
    s2 = getattr(layer, "weight_scale_2", None)
    widths = getattr(layer, "logical_widths", None)
    if s2 is None or widths is None:
        return False
    res = dense_column_scale(s2, widths)
    if res is None:
        return False
    col, g = res
    dtype = getattr(layer, "params_dtype", None) or col.dtype
    layer._gb10_col_scale = col.to(device=s2.device, dtype=dtype)
    s2.data.fill_(g.item())
    _note("dense", f"shard scales / max {col.min().item():.3g}..1, widths {list(widths)}")
    return True


def dense_apply(orig_apply, method, layer, x, bias=None):
    cs = getattr(layer, "_gb10_col_scale", None)
    if cs is None:
        return orig_apply(method, layer, x, bias)
    out = orig_apply(method, layer, x, None) * cs.to(x.dtype)
    if bias is not None:
        out = out + bias
    return out


def fix_moe(layer) -> bool:
    """Before SGLang's marlin collapse of w13's scale to the gate's: move the
    up/gate ratio into the down projection's per-expert global scale."""
    import torch

    s13 = getattr(layer, "w13_weight_scale_2", None)
    s2 = getattr(layer, "w2_weight_scale_2", None)
    if s13 is None or s2 is None or s13.dim() != 2 or s13.shape[1] < 2:
        return False
    g_gate, g_up = s13[:, 0].float(), s13[:, 1].float()
    if torch.allclose(g_gate, g_up):
        return False
    ratio = g_up / g_gate
    if s2.numel() != ratio.numel():
        logger.warning("NVFP4 scales: w2_weight_scale_2 %s does not match %d experts; "
                       "MoE layer left as SGLang loads it", tuple(s2.shape), ratio.numel())
        return False
    s2.data.copy_((s2.detach().float().reshape(-1) * ratio).reshape(s2.shape).to(s2.dtype))
    s13.data[:, 1] = s13.data[:, 0]
    _note("moe", f"up/gate ratio {ratio.min().item():.3g}..{ratio.max().item():.3g}")
    return True


def apply(mod) -> None:
    dense_cls = getattr(mod, "ModelOptNvFp4A16LinearMethod", None)
    moe_cls = getattr(mod, "ModelOptNvFp4FusedMoEMethod", None)
    if dense_cls is None or moe_cls is None:
        raise RuntimeError("GB10_NVFP4_SCALES: ModelOptNvFp4A16LinearMethod / "
                           "ModelOptNvFp4FusedMoEMethod not found (written for SGLang "
                           "0.5.20); unset GB10_NVFP4_SCALES.")
    if getattr(dense_cls, "_gb10_nvfp4_scales", False):
        return

    orig_dense_process = dense_cls.process_weights_after_loading
    orig_dense_apply = dense_cls.apply

    def dense_process(self, layer):
        fix_dense(layer)
        return orig_dense_process(self, layer)

    def dense_apply_(self, layer, x, bias=None):
        return dense_apply(orig_dense_apply, self, layer, x, bias)

    dense_cls.process_weights_after_loading = dense_process
    dense_cls.apply = dense_apply_

    orig_moe_process = moe_cls.process_weights_after_loading
    get_backend = getattr(mod, "get_moe_runner_backend", None)

    def moe_process(self, layer):
        backend = getattr(self, "_moe_runner_backend", None)
        if backend is None and get_backend is not None:
            backend = get_backend()
        cfg = getattr(layer, "moe_runner_config", None)
        if backend is not None and backend.is_marlin() and getattr(cfg, "is_gated", False):
            fix_moe(layer)
        return orig_moe_process(self, layer)

    moe_cls.process_weights_after_loading = moe_process
    dense_cls._gb10_nvfp4_scales = True
    logger.info("GB10_NVFP4_SCALES: per-shard NVFP4 global scales kept on the Marlin paths")
