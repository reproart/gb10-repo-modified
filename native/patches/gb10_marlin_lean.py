"""Leaner NVFP4 -> Marlin MoE repack at load time, with per-layer memory logs.

`--moe-runner-backend marlin` repacks every MoE layer's NVFP4 expert weights
after loading (sglang.srt.layers.quantization.marlin_utils_fp4,
prepare_moe_nvfp4_layer_for_marlin). On one GB10 with Qwen3.8-Flash-Next
(~78 GiB of weights on 121.6 GiB) that ran out of memory half way through the
48 layers: 114.65 GiB allocated at the failure, ~35 GiB above the loaded
weights, about one layer's original experts per layer done.

Two changes, enabled by GB10_MARLIN_LEAN=1 (models/qwen3.8-flash-next.sh sets
it with MOE_RUNNER_BACKEND=marlin):

  * the repack writes each expert into one preallocated output instead of a
    list of per-expert tensors joined by torch.stack, which held the weights
    three times at once (original, list, stacked) instead of twice;
  * after each layer: gc.collect() and torch.cuda.empty_cache(), then one log
    line with the allocated memory and its change. If the layer's original
    weights are still alive afterwards (weak reference), the first time it
    happens the objects holding them are logged, which is what a fix needs.

A no-op for every other backend. Written against SGLang 0.5.20; refuses to
patch a version without the functions it wraps.
"""

from __future__ import annotations

import gc
import logging
import weakref

logger = logging.getLogger("sglang.srt.layers.quantization.gb10_marlin_lean")

TARGET_MODULE = "sglang.srt.layers.quantization.modelopt_quant"
REPACK_MODULE = "sglang.srt.layers.quantization.marlin_utils_fp4"

_state = {"layer": 0, "reported": False}


def repack_into_one(weight, *, num_experts, size_n, size_k, perm, repack):
    """_repack_moe_fp4_weight_for_marlin without the list + stack copy: the
    same per-expert repack, written into one preallocated output."""
    import torch

    assert weight.shape == (num_experts, size_n, size_k // 2)
    out = None
    for i in range(num_experts):
        q = repack(
            b_q_weight=weight[i].view(torch.int32).T.contiguous(),
            perm=perm,
            size_k=size_k,
            size_n=size_n,
            num_bits=4,
        )
        if out is None:
            out = q.new_empty((num_experts, *q.shape))
        out[i].copy_(q)
        del q
    return out


def _gib(n: int) -> float:
    return n / 2**30


def _holders(dead_ptrs: set) -> list[str]:
    """Describe what still references tensors on the given storages."""
    import torch

    found = []
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor) or obj.untyped_storage().data_ptr() not in dead_ptrs:
                continue
        except Exception:  # noqa: BLE001 - meta/fake tensors
            continue
        for ref in gc.get_referrers(obj):
            if isinstance(ref, dict):
                keys = [k for k, v in ref.items() if v is obj][:3]
                owners = [type(o).__name__ for o in gc.get_referrers(ref)
                          if not isinstance(o, (list, dict))][:3]
                found.append(f"dict key {keys} of {owners}")
            elif isinstance(ref, (list, tuple)):
                found.append(f"{type(ref).__name__} of len {len(ref)}")
            else:
                found.append(type(ref).__name__)
        if len(found) > 12:
            break
    return found


def apply(mod) -> None:
    import sys

    import torch

    mfp4 = sys.modules.get(REPACK_MODULE)
    if (
        mfp4 is None
        or not hasattr(mfp4, "_repack_moe_fp4_weight_for_marlin")
        or not hasattr(mod, "prepare_moe_nvfp4_layer_for_marlin")
    ):
        raise RuntimeError(
            "GB10_MARLIN_LEAN: the NVFP4 Marlin MoE repack functions are not where "
            "SGLang 0.5.20 has them; unset GB10_MARLIN_LEAN."
        )

    def _repack(weight, *, num_experts, size_n, size_k, perm):
        return repack_into_one(
            weight, num_experts=num_experts, size_n=size_n, size_k=size_k,
            perm=perm, repack=mfp4.gptq_marlin_repack,
        )

    mfp4._repack_moe_fp4_weight_for_marlin = _repack

    orig_prepare = mod.prepare_moe_nvfp4_layer_for_marlin

    def prepare(layer):
        _state["layer"] += 1
        n = _state["layer"]
        old = [layer.w13_weight, layer.w2_weight]
        refs = [weakref.ref(p) for p in old]
        ptrs = {p.untyped_storage().data_ptr() for p in old}
        size = sum(p.numel() * p.element_size() for p in old)
        del old
        before = torch.cuda.memory_allocated()
        orig_prepare(layer)
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated()
        alive = any(r() is not None for r in refs)
        logger.info(
            "Marlin repack, MoE layer %d: allocated %.2f GiB (%+.2f GiB; experts "
            "%.2f GiB)%s", n, _gib(after), _gib(after - before), _gib(size),
            ", ORIGINAL WEIGHTS STILL ALIVE" if alive else "",
        )
        # The repacked weights replace the originals one for one, so growth of
        # more than a quarter of the layer's experts means something kept them
        # (a Parameter, or just a tensor on their storage, which only the
        # memory figure shows).
        if (alive or after - before > size / 4) and not _state["reported"]:
            _state["reported"] = True
            logger.warning(
                "Marlin repack: layer %d kept ~%.2f GiB after the repack; holders "
                "of the original expert weights: %s", n, _gib(after - before),
                _holders(ptrs) or "none found (the growth is elsewhere)",
            )

    mod.prepare_moe_nvfp4_layer_for_marlin = prepare
    logger.info("GB10_MARLIN_LEAN: NVFP4 Marlin MoE repack patched (lean, logged)")
