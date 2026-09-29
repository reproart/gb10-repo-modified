#!/usr/bin/env python3
"""CPU tests for gb10_nvfp4_scales (torch only, no GPU, no SGLang):

    python3 patches/test_gb10_nvfp4_scales.py

A fake Marlin path that does what SGLang 0.5.20 does with a fused layer's
global scales (dense: the max over the shards; MoE: the gate's for both
halves) is compared with the math of the unfused checkpoint, with and
without the patch. On the Spark the boot log has "NVFP4 scales: N dense
layer(s) ... corrected" and the answers must be right again.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


class Backend:
    def __init__(self, marlin):
        self._m = marlin

    def is_marlin(self):
        return self._m


def fake_module(marlin=True):
    """Stand-ins for SGLang's two methods, reduced to the scale handling."""

    class ModelOptNvFp4A16LinearMethod:
        def process_weights_after_loading(self, layer):
            layer.weight_global_scale = layer.weight_scale_2.max()   # SGLang: max over shards

        def apply(self, layer, x, bias=None):
            w = layer.w_deq * layer.weight_global_scale
            out = x @ w.T
            return out if bias is None else out + bias

    class ModelOptNvFp4FusedMoEMethod:
        def process_weights_after_loading(self, layer):
            layer.w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0].clone()  # gate's

        def run(self, layer, x, e):
            g = layer.w13_weight_scale_2[e]
            gate = x @ (layer.w1_deq[e] * g).T
            up = x @ (layer.w3_deq[e] * g).T
            h = torch.nn.functional.silu(gate) * up
            return h @ (layer.w2_deq[e] * layer.w2_weight_scale_2[e]).T

    mod = types.ModuleType("sglang.srt.layers.quantization.modelopt_quant")
    mod.ModelOptNvFp4A16LinearMethod = ModelOptNvFp4A16LinearMethod
    mod.ModelOptNvFp4FusedMoEMethod = ModelOptNvFp4FusedMoEMethod
    mod.get_moe_runner_backend = lambda: Backend(marlin)
    return mod


def dense_layer(widths, scales, k=32):
    torch.manual_seed(0)
    layer = types.SimpleNamespace()
    layer.logical_widths = list(widths)
    layer.params_dtype = torch.float32
    layer.w_deq = torch.randn(sum(widths), k)       # FP4 values x block scales
    layer.weight_scale_2 = torch.tensor(scales, dtype=torch.float32)
    rows = torch.cat([torch.full((w,), s) for w, s in zip(widths, scales)])
    layer.true_w = layer.w_deq * rows[:, None]       # what the checkpoint means
    return layer


def moe_layer(g_gate, g_up, g_down, e=3, i=16, h=32):
    torch.manual_seed(1)
    layer = types.SimpleNamespace()
    layer.w1_deq, layer.w3_deq = torch.randn(e, i, h), torch.randn(e, i, h)
    layer.w2_deq = torch.randn(e, h, i)
    layer.w13_weight_scale_2 = torch.stack([torch.tensor(g_gate), torch.tensor(g_up)], dim=1)
    layer.w2_weight_scale_2 = torch.tensor(g_down)
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    layer.truth = (torch.tensor(g_gate), torch.tensor(g_up), torch.tensor(g_down))
    return layer


def moe_true(layer, x, e):
    g1, g3, g2 = (t[e] for t in layer.truth)
    gate, up = x @ (layer.w1_deq[e] * g1).T, x @ (layer.w3_deq[e] * g3).T
    return (torch.nn.functional.silu(gate) * up) @ (layer.w2_deq[e] * g2).T


@unittest.skipIf(torch is None, "torch not installed")
class ScalesTest(unittest.TestCase):
    def setUp(self):
        import gb10_nvfp4_scales as s

        self.s = s
        s._counts.update(dense=0, moe=0)

    def test_dense_fused_shards(self):
        mod = fake_module()
        m = mod.ModelOptNvFp4A16LinearMethod()
        x = torch.randn(4, 32)
        for patched in (False, True):
            layer = dense_layer([48, 16, 16], [0.02, 0.005, 0.0004])
            if patched:
                self.s.apply(mod)
            meth = mod.ModelOptNvFp4A16LinearMethod()
            meth.process_weights_after_loading(layer)
            bias = torch.randn(80) * 1e-3   # small next to the output, so errors show
            out = meth.apply(layer, x, bias)
            ref = x @ layer.true_w.T + bias
            err = ((out - ref).norm() / ref.norm()).item()
            if patched:
                self.assertLess(err, 1e-5)
            else:
                self.assertGreater(err, 0.5)   # what SGLang does alone: far off
        self.assertEqual(self.s._counts["dense"], 1)
        self.assertIsNotNone(m)

    def test_dense_equal_scales_untouched(self):
        mod = fake_module()
        self.s.apply(mod)
        layer = dense_layer([16, 16], [0.01, 0.01])
        meth = mod.ModelOptNvFp4A16LinearMethod()
        meth.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "_gb10_col_scale"))
        x = torch.randn(2, 32)
        torch.testing.assert_close(meth.apply(layer, x), x @ layer.true_w.T)

    def test_moe_gate_up_scales(self):
        g_gate, g_up, g_down = [0.01, 0.02, 0.005], [0.003, 0.02, 0.05], [0.1, 0.2, 0.3]
        x = torch.randn(5, 32)
        for patched in (False, True):
            mod = fake_module()
            if patched:
                self.s.apply(mod)
            layer = moe_layer(g_gate, g_up, g_down)
            meth = mod.ModelOptNvFp4FusedMoEMethod()
            meth.process_weights_after_loading(layer)
            errs = []
            for e in range(3):
                ref = moe_true(layer, x, e)
                errs.append(((meth.run(layer, x, e) - ref).norm() / ref.norm()).item())
            if patched:
                self.assertLess(max(errs), 1e-5)
            else:
                self.assertGreater(max(errs), 0.5)
        self.assertEqual(self.s._counts["moe"], 1)

    def test_moe_not_marlin_untouched(self):
        mod = fake_module(marlin=False)
        self.s.apply(mod)
        layer = moe_layer([0.01], [0.02], [0.1], e=1)
        before = layer.w2_weight_scale_2.clone()
        mod.ModelOptNvFp4FusedMoEMethod().process_weights_after_loading(layer)
        torch.testing.assert_close(layer.w2_weight_scale_2, before)

    def test_apply_twice_and_missing(self):
        mod = fake_module()
        self.s.apply(mod)
        first = mod.ModelOptNvFp4A16LinearMethod.apply
        self.s.apply(mod)
        self.assertIs(mod.ModelOptNvFp4A16LinearMethod.apply, first)
        with self.assertRaises(RuntimeError):
            self.s.apply(types.ModuleType("empty"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
