#!/usr/bin/env python3
"""CPU tests for gb10_fp8_tune (no GPU, no SGLang):

    python3 patches/test_gb10_fp8_tune.py

A stand-in for flashinfer.gemm.gemm_base with FlashInfer's own dataclass shapes
checks the config swap (buckets plus the exact verify sizes, those sizes mapped
to themselves, everything else as before, idempotent, off without the
environment). Where FlashInfer is importable (the SGLang venv on the Spark), the
same spec also goes through FlashInfer's own AutoTuner: the profiles one tuning
pass generates, and the bucket a run-time 15 / 16 / 240 / 2048 rows look up.
"""

import dataclasses
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gb10_fp8_tune as tune  # noqa: E402


def next_pow2(x):
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def pow2_buckets(m):
    """FlashInfer's hybrid buckets below 256: powers of two up to next_pow2(m)."""
    out, b = [], 1
    while b <= next_pow2(m):
        out.append(b)
        b *= 2
    return tuple(out)


@dataclasses.dataclass(slots=True, unsafe_hash=True)
class DynamicTensorSpec:
    input_idx: tuple
    dim_idx: tuple
    gen_tuning_buckets: object
    map_to_tuning_buckets: object


@dataclasses.dataclass(slots=True, unsafe_hash=True)
class ConstraintSpec:
    input_idx: int
    dim_idx: int
    infer_shape: object


@dataclasses.dataclass(kw_only=True, unsafe_hash=True)
class TuningConfig:
    dynamic_tensor_specs: tuple = ()
    constraint_specs: tuple = ()
    use_cold_l2_cache: bool = False


def fake_gemm_base():
    mod = types.ModuleType("fake_gemm_base")
    mod._FP8_GEMM_SM100_TUNING_CONFIG = TuningConfig(
        dynamic_tensor_specs=(DynamicTensorSpec((0,), (-2,), pow2_buckets, next_pow2),),
        constraint_specs=(ConstraintSpec(4, -2, lambda shapes: shapes[0][-2]),))
    return mod


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.get(k) for k in ("GB10_FP8_TUNE_ROWS", "GB10_FP8_TUNE_MAX_BS")}
        self.addCleanup(self.restore)

    def restore(self):
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def set(self, rows, max_bs="16"):
        os.environ["GB10_FP8_TUNE_ROWS"] = rows
        os.environ["GB10_FP8_TUNE_MAX_BS"] = max_bs


class ConfigSwapTest(EnvCase):
    def test_exact_sizes(self):
        self.set("15")
        self.assertEqual(tune.exact_sizes(), tuple(range(15, 241, 15)))
        self.set("4", "8")
        self.assertEqual(tune.exact_sizes(), (4, 8, 12, 16, 20, 24, 28, 32))
        self.set("0")
        self.assertEqual(tune.exact_sizes(), ())
        os.environ.pop("GB10_FP8_TUNE_ROWS")
        self.assertEqual(tune.exact_sizes(), ())

    def test_swap_tunes_and_maps_the_verify_sizes(self):
        self.set("15")
        mod = fake_gemm_base()
        orig = mod._FP8_GEMM_SM100_TUNING_CONFIG
        tune.apply(mod)
        cfg = mod._FP8_GEMM_SM100_TUNING_CONFIG
        self.assertIsNot(cfg, orig)
        self.assertEqual(cfg.constraint_specs, orig.constraint_specs)
        spec = cfg.dynamic_tensor_specs[0]
        self.assertEqual((spec.input_idx, spec.dim_idx), ((0,), (-2,)))
        with self.assertLogs("sglang.srt.model_executor.gb10_fp8_tune", "INFO"):
            got = spec.gen_tuning_buckets(240)   # the decode-shaped pass at 16 x 15 rows
        self.assertEqual(got, tuple(sorted(set(pow2_buckets(240)) | set(range(15, 241, 15)))))
        self.assertEqual(spec.gen_tuning_buckets(15), (1, 2, 4, 8, 15, 16))   # up to its top bucket
        for m, want in ((15, 15), (60, 60), (240, 240), (16, 16), (17, 32), (2048, 2048), (1, 1)):
            self.assertEqual(spec.map_to_tuning_buckets(m), want, m)
        hash(cfg)   # FlashInfer keys an lru_cache on the specs

    def test_idempotent_and_resizable(self):
        self.set("15")
        mod = fake_gemm_base()
        tune.apply(mod)
        first = mod._FP8_GEMM_SM100_TUNING_CONFIG
        tune.apply(mod)
        self.assertIs(mod._FP8_GEMM_SM100_TUNING_CONFIG, first)
        self.set("10", "32")
        tune.apply(mod)
        spec = mod._FP8_GEMM_SM100_TUNING_CONFIG.dynamic_tensor_specs[0]
        self.assertEqual(spec.map_to_tuning_buckets(15), 16)   # the 15s are gone, not stacked
        self.assertEqual(spec.map_to_tuning_buckets(320), 320)

    def test_off_and_missing(self):
        os.environ.pop("GB10_FP8_TUNE_ROWS", None)
        mod = fake_gemm_base()
        orig = mod._FP8_GEMM_SM100_TUNING_CONFIG
        tune.apply(mod)
        self.assertIs(mod._FP8_GEMM_SM100_TUNING_CONFIG, orig)
        self.set("15")
        empty = types.ModuleType("no_config")
        with self.assertLogs("sglang.srt.model_executor.gb10_fp8_tune", "WARNING"):
            tune.apply(empty)


def real_autotuner():
    try:
        from flashinfer.autotuner import AutoTuner, TuningConfig  # noqa: F401
        from flashinfer.fused_moe.utils import (  # noqa: F401
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        )
    except Exception:  # noqa: BLE001
        return None
    import flashinfer.autotuner as fa
    import flashinfer.fused_moe.utils as fu
    return fa, fu


@unittest.skipIf(real_autotuner() is None, "flashinfer not importable (run in the SGLang venv)")
class RealAutoTunerTest(EnvCase):
    """FlashInfer's own profile generation and bucket lookup with the swapped spec,
    on a config built like gemm_base's _FP8_GEMM_SM100_TUNING_CONFIG."""

    def test_profiles_and_lookup(self):
        fa, fu = real_autotuner()
        self.set("15")
        mod = types.ModuleType("gemm_base_copy")
        mod._FP8_GEMM_SM100_TUNING_CONFIG = fa.TuningConfig(
            dynamic_tensor_specs=(fa.DynamicTensorSpec(
                (0,), (-2,), fu.get_hybrid_num_tokens_buckets, fu.map_to_hybrid_bucket_uncapped),),
            constraint_specs=(fa.ConstraintSpec(4, -2, lambda shapes: shapes[0][-2]),))
        tune.apply(mod)
        cfg = mod._FP8_GEMM_SM100_TUNING_CONFIG
        tuner = fa.AutoTuner.get()
        import torch

        a = torch.empty(1, 240, 64, dtype=torch.float8_e4m3fn)
        b = torch.empty(1, 64, 32, dtype=torch.float8_e4m3fn).transpose(1, 2).contiguous().transpose(1, 2)
        s = torch.ones(1)
        out = torch.empty(1, 240, 32, dtype=torch.bfloat16)
        ws = torch.empty(1024, dtype=torch.uint8)
        profiles = tuner._generate_optimization_profiles(cfg, [a, b, s, s, out, ws])
        tuned = sorted({p.get_opt_shapes()[0][1] for p in profiles})
        self.assertEqual(tuned, sorted(set(fu.get_hybrid_num_tokens_buckets(240)) | set(range(15, 241, 15))))

        def nearest(m):
            shapes = ((1, m, 64), (1, 64, 32), (1,), (1,), (1, m, 32), (1024,))
            return tuner._find_nearest_profile(shapes, cfg)[0][1]
        self.assertEqual([nearest(m) for m in (15, 60, 240, 16, 17, 2048)], [15, 60, 240, 16, 32, 2048])


if __name__ == "__main__":
    unittest.main(verbosity=2)
