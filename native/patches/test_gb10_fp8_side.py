#!/usr/bin/env python3
"""CPU tests for gb10_fp8_side (torch only, no GPU, no SGLang):

    python3 patches/test_gb10_fp8_side.py

Covers the per-channel FP8 quantization, which layers get converted (name,
dtype, Marlin shape limits), the GDN fused-buffer drop and the quant_method
swap. The Marlin repack and GEMM themselves need the GPU: on the Spark the
boot log has "FP8 side (target): N linear layers to FP8 ..." and the answers
must stay coherent.
"""

import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class QuantizeTest(unittest.TestCase):
    def test_per_channel_error_is_small(self):
        import gb10_fp8_side as f

        torch.manual_seed(0)
        w = (torch.randn(256, 512) * torch.logspace(-3, 0, 256)[:, None]).to(torch.bfloat16)
        q, s = f.quantize_per_channel(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.shape, (256,))
        back = q.float() * s[:, None]
        rel = ((back - w.float()).norm(dim=1) / w.float().norm(dim=1)).max().item()
        self.assertLess(rel, 0.05)  # e4m3: ~2-3% RMS per row
        self.assertTrue(torch.isfinite(back).all())

    def test_zero_row_does_not_divide_by_zero(self):
        import gb10_fp8_side as f

        w = torch.zeros(64, 128, dtype=torch.bfloat16)
        q, s = f.quantize_per_channel(w)
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertTrue((q.float() == 0).all())


class Unquant:
    pass


class Linear(torch.nn.Module if torch else object):
    def __init__(self, n, k, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k).to(dtype or torch.bfloat16), requires_grad=False)
        self.bias = None
        self.quant_method = Unquant()


class GDN(torch.nn.Module if torch else object):
    def __init__(self, a=256, b=64):
        super().__init__()
        self.in_proj_qkvz = Linear(a, 256)
        self.in_proj_ba = Linear(b, 256)
        self.out_proj = Linear(256, 128)
        self._fused_in_proj_weight = torch.cat([self.in_proj_qkvz.weight, self.in_proj_ba.weight])


@unittest.skipIf(torch is None, "torch not installed")
class ConvertTest(unittest.TestCase):
    def model(self, ba_rows=64):
        m = torch.nn.Module()
        m.layers = torch.nn.ModuleList()
        layer = torch.nn.Module()
        layer.linear_attn = GDN(b=ba_rows)
        layer.mlp = torch.nn.Module()
        layer.mlp.shared_expert = torch.nn.Module()
        layer.mlp.shared_expert.gate_up_proj = Linear(512, 256)
        layer.mlp.shared_expert.down_proj = Linear(256, 256)
        layer.mlp.gate = Linear(64, 256)                       # router: name not matched
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = Linear(384, 256)
        layer.self_attn.o_proj = Linear(256, 100)              # K % 128 != 0 -> skipped
        m.layers.append(layer)
        m.lm_head = Linear(1024, 256)                          # not matched
        return m

    def convert(self, m):
        import gb10_fp8_side as f

        prepared = []

        def prepare(module, size_k_first):
            self.assertFalse(size_k_first)
            self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(module.weight_scale.shape, (module.weight.shape[0],))
            module.workspace = "ws"
            prepared.append(module)

        calls = []

        def apply(**kw):
            calls.append(kw)
            return "out"

        stats = f.convert_model(
            m, is_linear=lambda mod: isinstance(mod, Linear),
            is_unquantized=lambda qm: isinstance(qm, Unquant),
            prepare_fn=prepare, apply_fn=apply)
        return f, stats, prepared, calls

    def test_selection_and_swap(self):
        m = self.model()
        f, stats, prepared, calls = self.convert(m)
        layer = m.layers[0]
        converted = [layer.linear_attn.in_proj_qkvz, layer.linear_attn.in_proj_ba,
                     layer.linear_attn.out_proj, layer.mlp.shared_expert.gate_up_proj,
                     layer.mlp.shared_expert.down_proj, layer.self_attn.qkv_proj]
        self.assertEqual(stats["converted"], len(converted))
        self.assertEqual(stats["skipped"], 1)                  # o_proj, K = 100
        self.assertEqual({id(x) for x in prepared}, {id(x) for x in converted})
        for mod in (layer.mlp.gate, m.lm_head, layer.self_attn.o_proj):
            self.assertEqual(mod.weight.dtype, torch.bfloat16)
            self.assertIsInstance(mod.quant_method, Unquant)
        self.assertIsNone(layer.linear_attn._fused_in_proj_weight)
        self.assertEqual(stats["bytes_after"] * 2, stats["bytes_before"])
        qkv = layer.self_attn.qkv_proj
        self.assertIsInstance(qkv.quant_method, f.Fp8MarlinSideMethod)
        self.assertEqual(qkv.quant_method.apply(qkv, "x"), "out")
        self.assertEqual((calls[0]["size_n"], calls[0]["size_k"]), (384, 256))

    def test_fused_buffer_kept_when_a_half_stays_bf16(self):
        m = self.model()
        gdn = m.layers[0].linear_attn
        os.environ["GB10_FP8_SIDE_LAYERS"] = r"\.in_proj_qkvz$"   # in_proj_ba stays BF16
        self.addCleanup(os.environ.pop, "GB10_FP8_SIDE_LAYERS")
        # as finalize_fused_in_proj leaves it: both halves are views of the buffer
        gdn.in_proj_qkvz.weight.data = gdn._fused_in_proj_weight[:256]
        gdn.in_proj_ba.weight.data = gdn._fused_in_proj_weight[256:]
        fused_ptr = gdn._fused_in_proj_weight.untyped_storage().data_ptr()
        self.convert(m)
        # qkvz converted: the buffer must not be used, and the BF16 half must
        # not keep it (and qkvz's old BF16 rows) alive
        self.assertIsNone(gdn._fused_in_proj_weight)
        self.assertEqual(gdn.in_proj_ba.weight.dtype, torch.bfloat16)
        self.assertNotEqual(gdn.in_proj_ba.weight.untyped_storage().data_ptr(), fused_ptr)

    def test_narrow_output_is_padded_and_sliced(self):
        import gb10_fp8_side as f

        m = self.model(ba_rows=96)                              # GDN in_proj_ba on the real model
        gdn = m.layers[0].linear_attn
        ba_bf16 = gdn.in_proj_ba.weight.detach().float().clone()

        def prepare(module, size_k_first):
            if module is gdn.in_proj_ba:
                self.assertEqual(module.output_size_per_partition, 128)  # the padded width
                self.assertEqual(module.weight.shape, (128, 256))
            module.workspace = "ws"

        seen = {}

        def apply(*, input, weight, weight_scale, workspace, size_n, size_k, bias):
            seen["size_n"] = size_n
            # dequantize and multiply, like the kernel: [.., K] x [N_pad, K]^T
            w = weight.float() * weight_scale.float()[:, None]
            return (input.float() @ w.T).to(torch.bfloat16)

        stats = f.convert_model(m, is_linear=lambda mod: isinstance(mod, Linear),
                                is_unquantized=lambda qm: isinstance(qm, Unquant),
                                prepare_fn=prepare, apply_fn=apply)
        self.assertEqual(stats["padded"], 1)
        ba = gdn.in_proj_ba
        self.assertEqual(ba.output_size_per_partition, 96)      # what the model reads
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        out = ba.quant_method.apply(ba, x)
        self.assertEqual(seen["size_n"], 128)
        self.assertEqual(tuple(out.shape), (4, 96))
        self.assertTrue(out.is_contiguous())
        ref = x.float() @ ba_bf16.T
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 0.05)

    def test_padding_needs_no_bias(self):
        import gb10_fp8_side as f

        m = self.model(ba_rows=96)
        ba = m.layers[0].linear_attn.in_proj_ba
        ba.bias = torch.nn.Parameter(torch.zeros(96, dtype=torch.bfloat16), requires_grad=False)
        _, stats, _, _ = self.convert(m)
        self.assertIsInstance(ba.quant_method, Unquant)
        self.assertEqual(stats["skipped"], 2)                   # this and o_proj (K = 100)
        self.assertIsNotNone(f)

    def test_custom_pattern(self):
        import gb10_fp8_side as f

        os.environ["GB10_FP8_SIDE_LAYERS"] = r"\.out_proj$"
        try:
            m = self.model()
            _, stats, _, _ = self.convert(m)
            self.assertEqual(stats["converted"], 1)
            self.assertIsInstance(m.layers[0].linear_attn.out_proj.quant_method, f.Fp8MarlinSideMethod)
        finally:
            del os.environ["GB10_FP8_SIDE_LAYERS"]


class HookOrderTest(unittest.TestCase):
    def test_two_hooks_on_one_module_both_run_in_order(self):
        import gb10_ple_mmap as g

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "gb10twohooks"))
            open(os.path.join(d, "gb10twohooks", "__init__.py"), "w").close()
            with open(os.path.join(d, "gb10twohooks", "m.py"), "w") as fh:
                fh.write("X = 1\n")
            seen = []
            sys.path.insert(0, d)
            try:
                g.install_import_hook("gb10twohooks.m", lambda mod: seen.append("first"))
                g.install_import_hook("gb10twohooks.m", lambda mod: seen.append("second"))
                importlib.import_module("gb10twohooks.m")
                self.assertEqual(seen, ["first", "second"])
            finally:
                sys.path.remove(d)
                sys.meta_path[:] = [x for x in sys.meta_path if not isinstance(x, g._Finder)]
                for name in ("gb10twohooks.m", "gb10twohooks"):
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
