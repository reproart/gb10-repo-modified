"""Exact autotune sizes for FlashInfer's FP8 GEMM on cuBLASLt (the FP8 GDN and
attention projections of RadixArk/Qwen3.8-27B-NVFP4).

SGLang 0.5.21 autotunes FlashInfer's GEMMs at boot. For bmm_fp8 on cuBLASLt a
tactic is an index into the list of algorithms cuBLASLt returns for one problem
shape. The tuner measures at bucket sizes (powers of two: 1, 2, 4, 8, 16, ...,
256 rows) and maps a run-time size to its bucket, so it tunes at 16 rows for a
DFlash2 verify of 15 and stores, say, "#5". At run time FlashInfer enumerates
the list for the real 15 rows and takes its #5, which can be another algorithm
(FlashInfer only guards against an index past the end). Measured on GB10 with
the server's own cache files (bench_sglang_gb10_gpu.py, section 6): GDN out_proj
[5120 x 6144], tactic #5 tuned at 16 rows is 64x64x128 there (134 us) and
64x104x128 at 15 rows (166 us; 190 us in the server's decode profile), against
129 us for the best; in_proj_qkvz, #3 is 96x128x128 at 15 rows (374-379 us)
against 346. Several algorithms tie within ~1% at 16 rows, so which index wins
is the timing noise of that boot: the 27B's "two modes per boot" (single-stream
decode 88-89.5 or 93.8-94.2 tok/s, results/RESULTS.md).

GB10_FP8_TUNE_ROWS=R (the rows one request adds to a verify: the draft tokens)
and GB10_FP8_TUNE_MAX_BS=B (the most requests in a decode CUDA graph): the FP8
GEMM's tuning config also profiles R, 2R, ..., B x R rows (up to the largest
bucket the tuning pass reaches) and maps exactly those sizes to themselves, so a
tactic is found on the same list it is used on. Every other size keeps
FlashInfer's buckets. Only this config changes: the NVFP4 GEMM's tactics are
kernel configurations, the same at any size. The boot log has "GB10_FP8_TUNE:
FlashInfer's FP8 GEMM tuned at exact verify sizes ..." when the tuning pass
uses it. Written against FlashInfer 0.6.18 (SGLang 0.5.21).
"""

from __future__ import annotations

import dataclasses
import logging
import os

logger = logging.getLogger("sglang.srt.model_executor.gb10_fp8_tune")

TARGET_MODULE = "flashinfer.gemm.gemm_base"
CONFIG_NAME = "_FP8_GEMM_SM100_TUNING_CONFIG"
_MARK = "_gb10_fp8_tune_sizes"


def exact_sizes() -> tuple:
    """R, 2R, ..., B x R from the environment; () when the patch is off."""
    rows = int(os.environ.get("GB10_FP8_TUNE_ROWS", "0") or 0)
    max_bs = int(os.environ.get("GB10_FP8_TUNE_MAX_BS", "16") or 16)
    if rows <= 0 or max_bs <= 0:
        return ()
    return tuple(rows * b for b in range(1, max_bs + 1))


def exact_spec(spec, sizes):
    """A copy of a DynamicTensorSpec that also tunes at `sizes` and maps them to
    themselves; other sizes go through the spec's own buckets."""
    gen0, map0 = spec.gen_tuning_buckets, spec.map_to_tuning_buckets
    exact = frozenset(sizes)
    logged = []

    def gen(m):
        base = tuple(gen0(m)) if callable(gen0) else tuple(gen0)
        top = max(base) if base else m
        extra = sorted(s for s in exact if s <= top and s not in base)
        if extra and not logged:
            logged.append(True)
            logger.info("GB10_FP8_TUNE: FlashInfer's FP8 GEMM tuned at exact verify sizes %s rows "
                        "besides its buckets (patches/gb10_fp8_tune.py)",
                        ", ".join(map(str, extra)))
        return tuple(sorted(set(base) | set(extra)))

    def map_exact(m):
        return m if m in exact else map0(m)

    return dataclasses.replace(spec, gen_tuning_buckets=gen, map_to_tuning_buckets=map_exact)


def apply(mod) -> None:
    sizes = exact_sizes()
    if not sizes:
        return
    if getattr(mod, _MARK, None) == sizes:
        return
    cfg = getattr(mod, CONFIG_NAME, None)
    specs = getattr(cfg, "dynamic_tensor_specs", None)
    if not specs:
        logger.warning("GB10_FP8_TUNE: %s.%s not found; FlashInfer's FP8 GEMM keeps its tuning "
                       "buckets", TARGET_MODULE, CONFIG_NAME)
        return
    if getattr(mod, _MARK, None):   # applied with other sizes: start from the original
        cfg = mod._gb10_fp8_tune_orig
        specs = cfg.dynamic_tensor_specs
    else:
        mod._gb10_fp8_tune_orig = cfg
    new = dataclasses.replace(cfg, dynamic_tensor_specs=(exact_spec(specs[0], sizes),) + tuple(specs[1:]))
    setattr(mod, CONFIG_NAME, new)
    setattr(mod, _MARK, sizes)
