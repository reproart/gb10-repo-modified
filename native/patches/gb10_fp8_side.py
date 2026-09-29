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
below) are converted, and only if K fits Marlin's tile (K % 128); every
conversion and skip is logged. An output width off Marlin's 64-column tile
(GDN's in_proj_ba is 96) is zero-padded to it and sliced back after the GEMM.
GDN's fused BF16 in_proj buffer is dropped for converted layers, so both
projections go through the FP8 path.
The target and, unless GB10_FP8_SIDE_MTP=0, the MTP draft are converted.
Another model with the same kind of BF16 layers (e.g. a Qwen3.5 MoE FP8
checkpoint that keeps its GDN projections in BF16): GB10_FP8_SIDE_TARGET=
<module>:<class>; its MTP draft is then left alone.

The output heads (lm_head, BF16 [vocab x hidden]) are separate switches, since
the target head decides every emitted token and the draft head only proposes:
  GB10_FP8_DRAFT_HEAD=1   the MTP draft's head (with --speculative-token-map a
                          [65536 x hidden] slice of the target head) to FP8;
                          a worse draft costs acceptance, never correctness.
  GB10_FP8_TARGET_HEAD=1  the target head to FP8 (lossy for the answers: check
                          HumanEval). With speculative decoding.
  GB10_FP8_TARGET_HEAD=load  the same without speculative decoding: converted at
                          load, so the freed memory goes to the KV pool.
With speculative decoding the heads are converted right after the EAGLE worker's
init_lm_head, which builds the draft head from the target's BF16 one (a slice
for the token map, or the target module itself); it runs after the KV pools
are sized and before any CUDA graph is captured.

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
SPEC_MODULES = (
    "sglang.srt.speculative.eagle_worker_v2",              # EagleDraftWorker
    "sglang.srt.speculative.multi_layer_eagle_worker_v2",  # MultiLayerEagleDraftWorker
)
# logits_processor._UNQUANTIZED_LM_HEAD_METHODS: heads that run as a plain matmul
UNQUANT_HEAD_METHODS = ("UnquantizedEmbeddingMethod", "UnquantizedLinearMethod")

DEFAULT_LAYERS = (
    r"\.(in_proj_qkvz|in_proj_ba|out_proj|qkv_proj|o_proj|key_proj|value_proj"
    r"|shared_expert\.gate_up_proj|shared_expert\.down_proj)$"
)
FP8_MAX = 448.0
MIN_N, MIN_K = 64, 128  # Marlin's thread tile (marlin_utils.GPTQ_MARLIN_MIN_THREAD_*)


def enabled() -> bool:
    return os.environ.get("GB10_FP8_SIDE") == "1"


def draft_head_enabled() -> bool:
    return os.environ.get("GB10_FP8_DRAFT_HEAD") == "1"


def target_head_mode() -> str:
    """'' (BF16), '1' (after init_lm_head) or 'load' (no speculative decoding)."""
    mode = os.environ.get("GB10_FP8_TARGET_HEAD", "0")
    if mode not in ("0", "1", "load"):
        raise RuntimeError(f"GB10_FP8_TARGET_HEAD must be 0, 1 or load, not {mode!r}")
    return "" if mode == "0" else mode


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
    """quant_method for a converted layer: FP8 Marlin weight-only GEMM.

    A layer whose output width was padded up to Marlin's tile (see
    convert_module) computes the padded width and returns the first
    `layer._gb10_n` columns."""

    def __init__(self, apply_fn):
        self._apply_fn = apply_fn

    def process_weights_after_loading(self, layer) -> None:  # already prepared
        return

    def apply(self, layer, x, bias=None):
        out = self._apply_fn(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer._gb10_marlin_n,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
        if layer._gb10_marlin_n != layer._gb10_n:
            out = out[..., : layer._gb10_n].contiguous()
        return out


def convert_module(module, prepare_fn, apply_fn) -> None:
    """Quantize `module.weight` in place and route the layer through Marlin FP8.

    An output width that is not a multiple of Marlin's 64-column tile (GDN's
    in_proj_ba is 96 wide) is padded with zero rows; the extra outputs are
    sliced off in apply. Left alone, such a small layer ran on a cuBLAS BF16
    kernel at ~245 us a call on GB10, 36 calls per decode step."""
    import torch

    n, k = module.weight.shape
    n_pad = -(-n // MIN_N) * MIN_N
    q, scale = quantize_per_channel(module.weight.data)
    if n_pad != n:
        q = torch.cat([q, torch.zeros(n_pad - n, k, dtype=q.dtype, device=q.device)])
        scale = torch.cat([scale, torch.ones(n_pad - n, dtype=scale.dtype, device=scale.device)])
    module.weight = torch.nn.Parameter(q, requires_grad=False)
    module.weight_scale = torch.nn.Parameter(scale.to(torch.bfloat16), requires_grad=False)
    module.orig_dtype = torch.bfloat16
    logical_n = getattr(module, "output_size_per_partition", n)
    module.output_size_per_partition = n_pad  # what the Marlin prepare checks
    module.input_size_per_partition = k
    if getattr(module, "weight_block_size", None) is not None:
        module.weight_block_size = None
    prepare_fn(module, size_k_first=False)
    module.output_size_per_partition = logical_n  # what the model sees
    module._gb10_n = n
    module._gb10_marlin_n = n_pad
    module.quant_method = Fp8MarlinSideMethod(apply_fn)


def convert_model(model, *, is_linear, is_unquantized, prepare_fn, apply_fn,
                  pattern=None, label="target") -> dict:
    """Convert every matching BF16 linear layer of `model`. Returns counters."""
    import torch

    pattern = pattern or layer_pattern()
    stats = {"converted": 0, "padded": 0, "skipped": 0, "bytes_before": 0, "bytes_after": 0}
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
        if k % MIN_K or (n % MIN_N and getattr(module, "bias", None) is not None):
            stats["skipped"] += 1
            logger.info("FP8 side (%s): %s [%d x %d] left in BF16 (Marlin needs K%%%d; "
                        "N is padded to %d only without a bias)", label, name, n, k, MIN_K, MIN_N)
            continue
        if n % MIN_N:
            stats["padded"] += 1
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
    stats["left"] = report_bf16_left(model, is_linear, label)
    logger.info(
        "FP8 side (%s): %d linear layers to FP8 weight-only (Marlin; %d with N padded to "
        "the %d-column tile), %d left in BF16; %.2f GiB -> %.2f GiB", label,
        stats["converted"], stats["padded"], MIN_N, stats["skipped"],
        stats["bytes_before"] / 2**30, stats["bytes_after"] / 2**30,
    )
    return stats


def report_bf16_left(model, is_linear, label) -> list:
    """Log the linear layers still in BF16 after the conversion, grouped by name
    with layer numbers folded ("layers.N.mlp.gate [256 x 2560] x48"): in a
    decode profile under CUDA graphs a kernel has no stack, only its name, and
    small BF16 GEMMs on GB10's sm80 WMMA kernels take ~40-250 us a call."""
    import torch

    groups = {}
    for name, module in model.named_modules():
        w = getattr(module, "weight", None)
        if not (is_linear(module) or isinstance(module, torch.nn.Linear)):
            continue
        if not isinstance(w, torch.Tensor) or w.dim() != 2 or w.dtype != torch.bfloat16:
            continue
        key = (re.sub(r"\.\d+\.", ".N.", name), tuple(w.shape))
        groups[key] = groups.get(key, 0) + 1
    left = sorted(((n, shape, c) for (n, shape), c in groups.items()), key=lambda g: -g[2])
    if left:
        logger.info("FP8 side (%s): linear layers still in BF16: %s", label, "; ".join(
            f"{n} [{shape[0]} x {shape[1]}] x{c}" for n, shape, c in left))
    return left


def _is_unquant_head(qm) -> bool:
    return qm is None or type(qm).__name__ in UNQUANT_HEAD_METHODS


def _ptr(t) -> int:
    return t.untyped_storage().data_ptr()


def _share_converted(dst, src) -> None:
    """Point another head module at an already converted head's FP8 weights."""
    for attr in ("weight", "weight_scale", "workspace", "orig_dtype", "input_size_per_partition",
                 "output_size_per_partition", "_gb10_n", "_gb10_marlin_n", "quant_method"):
        setattr(dst, attr, getattr(src, attr))


def convert_heads(heads, *, protected=(), prepare_fn, apply_fn) -> list:
    """Convert output heads to FP8 Marlin. `heads` is [(label, module)];
    `protected` holds tensors a head must not be converted out of (the input
    embedding, when the head is tied to it). A module listed twice is converted
    once; a second module holding the same BF16 tensor shares the FP8 copy.
    Returns [(label, 'converted' | 'shared' | reason)]."""
    import torch

    protected_ptrs = {_ptr(t) for t in protected if t is not None}
    by_module, by_tensor, report = {}, {}, []
    for label, head in heads:
        if head is None:
            continue
        if id(head) in by_module:
            report.append((label, f"same module as the {by_module[id(head)]} head"))
            continue
        w = getattr(head, "weight", None)
        if isinstance(getattr(head, "quant_method", None), Fp8MarlinSideMethod):
            report.append((label, "already FP8"))
            continue
        if w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or not _is_unquant_head(
                getattr(head, "quant_method", None)):
            why = f"not a BF16 matmul head ({getattr(w, 'dtype', None)}, " \
                  f"{type(getattr(head, 'quant_method', None)).__name__})"
            report.append((label, why))
            logger.info("FP8 head (%s): left as is, %s", label, why)
            continue
        if _ptr(w) in protected_ptrs:
            report.append((label, "tied to the input embedding"))
            logger.info("FP8 head (%s): left in BF16, tied to the input embedding", label)
            continue
        n, k = w.shape
        if k % MIN_K or getattr(head, "bias", None) is not None:
            report.append((label, "shape/bias"))
            logger.info("FP8 head (%s): [%d x %d] left in BF16 (Marlin needs K%%%d, no bias)",
                        label, n, k, MIN_K)
            continue
        by_module[id(head)] = label
        src = by_tensor.get(_ptr(w))
        if src is not None:
            _share_converted(head, src)
            report.append((label, "shared"))
            logger.info("FP8 head (%s): shares the FP8 copy of the same BF16 tensor", label)
            continue
        by_tensor[_ptr(w)] = head
        before = w.numel() * w.element_size()
        convert_module(head, prepare_fn, apply_fn)
        report.append((label, "converted"))
        logger.info("FP8 head (%s): lm_head [%d x %d] BF16 -> FP8 weight-only (Marlin); "
                    "%.0f MiB -> %.0f MiB", label, n, k, before / 2**20,
                    head.weight.numel() * head.weight.element_size() / 2**20)
    torch.cuda.empty_cache()
    return report


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


def _embed_of(model):
    try:
        return model.get_embed_and_head()[0]
    except Exception:  # noqa: BLE001 - only used to protect a tied head
        return getattr(getattr(getattr(model, "model", None), "embed_tokens", None), "weight", None)


def _wrap_load_weights(cls, label, *, side=True, target_head=False):
    orig = cls.load_weights

    def load_weights(self, weights, *args, **kwargs):
        result = orig(self, weights, *args, **kwargs)
        parts = _sglang_parts()
        if side:
            convert_model(self, label=label, **parts)
        if target_head:
            convert_heads([("target", getattr(self, "lm_head", None))], protected=[_embed_of(self)],
                          prepare_fn=parts["prepare_fn"], apply_fn=parts["apply_fn"])
        return result

    cls.load_weights = load_weights


def target_spec() -> tuple:
    """(module, class) of the target model to convert: Qwen4-Exp by default,
    another model with GB10_FP8_SIDE_TARGET=<module>:<class> (e.g.
    sglang.srt.models.qwen3_5:Qwen3_5MoeForConditionalGeneration)."""
    spec = os.environ.get("GB10_FP8_SIDE_TARGET") or f"{TARGET_MODULE}:Qwen4ExpForConditionalGeneration"
    module, _, cls = spec.partition(":")
    if not module or not cls:
        raise RuntimeError(f"GB10_FP8_SIDE_TARGET must be <module>:<class>, not {spec!r}")
    return module, cls


def apply_target(mod) -> None:
    _, cls_name = target_spec()
    if not hasattr(mod, cls_name):
        raise RuntimeError(f"GB10_FP8_SIDE: {mod.__name__}.{cls_name} not found "
                           "(written for SGLang 0.5.20); unset GB10_FP8_SIDE.")
    head = target_head_mode() == "load"
    _wrap_load_weights(getattr(mod, cls_name), "target", side=enabled(), target_head=head)
    logger.info("GB10_FP8_SIDE: target %s will load as FP8 (Marlin)",
                " and ".join(x for x, on in (("BF16 side layers", enabled()), ("lm_head", head)) if on))


def apply_mtp(mod) -> None:
    if os.environ.get("GB10_FP8_SIDE_MTP", "1") != "1":
        return
    if not hasattr(mod, "Qwen4ExpForCausalLMMTP"):
        raise RuntimeError("GB10_FP8_SIDE: Qwen4ExpForCausalLMMTP not found "
                           "(written for SGLang 0.5.20); set GB10_FP8_SIDE_MTP=0.")
    _wrap_load_weights(mod.Qwen4ExpForCausalLMMTP, "MTP draft")


def heads_after_init_lm_head(worker) -> list:
    """Called with an EAGLE draft worker once init_lm_head has shared the target's
    embedding and head with the draft model(s)."""
    target = worker.target_worker.model_runner.model
    runners = getattr(worker, "draft_runner_list", None) or [worker.draft_runner]
    drafts = [r.model for r in runners]
    target_head = getattr(target, "lm_head", None)
    with_target = target_head_mode() == "1"
    heads = [("target", target_head)] if with_target else []
    if draft_head_enabled():
        for d in drafts:
            head = getattr(d, "lm_head", None)
            if head is target_head and not with_target:
                # no --speculative-token-map: the draft calls the target's own
                # head module; converting it would convert the target head
                logger.info("FP8 head (draft): left in BF16, it is the target's head module "
                            "(no --speculative-token-map); GB10_FP8_TARGET_HEAD=1 converts both")
                continue
            heads.append(("draft", head))
    if not heads:
        return []
    parts = _sglang_parts()
    return convert_heads(heads, protected=[_embed_of(target)] + [_embed_of(d) for d in drafts],
                         prepare_fn=parts["prepare_fn"], apply_fn=parts["apply_fn"])


def _wrap_init_lm_head(cls) -> None:
    orig = cls.__dict__["init_lm_head"]

    def init_lm_head(self, *args, **kwargs):
        if target_head_mode() == "load":
            # the draft head is built from the target's BF16 head right here
            raise RuntimeError("GB10_FP8_TARGET_HEAD=load converts the target head at load and "
                               "is for serving without speculative decoding; use 1 with MTP.")
        result = orig(self, *args, **kwargs)
        heads_after_init_lm_head(self)
        return result

    init_lm_head._gb10_fp8_heads = True
    cls.init_lm_head = init_lm_head


def apply_spec(mod) -> None:
    """Hook for SPEC_MODULES: wrap the draft workers' own init_lm_head (a
    subclass that overrides it, like the standalone worker, is left alone)."""
    wrapped = []
    for name in ("EagleDraftWorker", "MultiLayerEagleDraftWorker"):
        cls = getattr(mod, name, None)
        if cls is not None and cls.__module__ == mod.__name__ and "init_lm_head" in cls.__dict__:
            if not getattr(cls.__dict__["init_lm_head"], "_gb10_fp8_heads", False):
                _wrap_init_lm_head(cls)
            wrapped.append(name)
    if mod.__name__ == SPEC_MODULES[0] and "EagleDraftWorker" not in wrapped:
        raise RuntimeError("GB10_FP8 heads: EagleDraftWorker.init_lm_head not found (written "
                           "for SGLang 0.5.20); set GB10_FP8_DRAFT_HEAD=0 GB10_FP8_TARGET_HEAD=0.")
    if wrapped:
        logger.info("GB10_FP8 heads: %s converted to FP8 after %s.init_lm_head",
                    " and ".join(x for x, on in (("draft head", draft_head_enabled()),
                                                 ("target head", target_head_mode() == "1")) if on),
                    "/".join(wrapped))
