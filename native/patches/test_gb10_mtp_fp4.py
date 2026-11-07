#!/usr/bin/env python3
"""CPU tests for gb10_mtp_fp4 (torch only; no GPU, no SGLang, no flashinfer):

    python3 patches/test_gb10_mtp_fp4.py

Covers the torch NVFP4 quantizer against an independent nearest-value search
and SGLang's own decoding of the format (test/quant_ref_utils.py: E2M1 table,
low nibble first, linear block scales divided by the global encode scale),
the 4-or-6 block scale choice, and the wrapper around SGLang's load-time
quantizer (flashinfer first, the torch quantizer after a failure, or always).
On the Spark the boot log has "Running online NVFP4 quantization for MoE expert
weights in layer 0 (mtp.layers.0.mlp.experts)" and, if flashinfer's cute-dsl
quantizer does not build there, one "GB10_MTP_FP4: flashinfer's NVFP4
quantizer failed" line.
"""

import logging
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def sglang_dequantize(packed, sf_linear, global_scale):
    """SGLang's reference decode (test/quant_ref_utils.py: break_fp4_bytes and
    dequantize_nvfp4_to_dtype, without the swizzle), written out here."""
    m, n = packed.shape
    flat = packed.flatten()
    high = (flat & 0xF0) >> 4
    low = flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    vals = torch.tensor(E2M1)[(combined & 0x07).long()] * torch.where(signs, -1.0, 1.0)
    vals = vals.reshape(m, n * 2).reshape(m, -1, 16)
    return (vals * (sf_linear.float() / global_scale).unsqueeze(-1)).reshape(m, n * 2)


@unittest.skipIf(torch is None, "torch not installed")
class QuantizerTest(unittest.TestCase):
    def setUp(self):
        import gb10_mtp_fp4 as q

        self.q = q
        torch.manual_seed(0)

    def test_e2m1_rounding_and_ties(self):
        a = torch.tensor([0.0, 0.2, 0.25, 0.26, 0.75, 1.0, 1.25, 1.26, 1.75, 2.5, 2.6,
                          3.5, 4.9, 5.0, 5.1, 6.0, 9.0])
        want = [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 5, 6, 6, 6, 7, 7, 7]
        self.assertEqual(self.q.e2m1_codes(a).tolist(), want)
        # away from ties: the nearest value, by brute force
        x = torch.rand(10000) * 7
        nearest = (x[:, None] - torch.tensor(E2M1)[None, :]).abs().argmin(dim=1)
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        off_tie = ((x[:, None] - ties[None, :]).abs() > 1e-4).all(dim=1)
        self.assertTrue(torch.equal(self.q.e2m1_codes(x)[off_tie].long(), nearest[off_tie]))

    def test_format_matches_sglang_decoding(self):
        w = (torch.randn(64, 256) * 0.02).to(torch.bfloat16)
        packed, sf, ws2 = self.q.quantize_nvfp4_torch(w)
        self.assertEqual((packed.dtype, tuple(packed.shape)), (torch.uint8, (64, 128)))
        self.assertEqual((sf.dtype, tuple(sf.shape)), (torch.float8_e4m3fn, (64, 16)))
        self.assertEqual(ws2.dtype, torch.float32)
        self.assertAlmostEqual(ws2.item(), w.float().abs().max().item() / (448 * 6), places=9)
        ours = self.q.dequantize_nvfp4(packed, sf, ws2)
        ref = sglang_dequantize(packed, sf, 1.0 / ws2)
        self.assertTrue(torch.allclose(ours, ref, rtol=1e-6, atol=0))
        err = ((ours - w.float()).norm() / w.float().norm()).item()
        self.assertLess(err, 0.12)        # NVFP4 on Gaussian weights: ~8-10%

    def test_low_nibble_first(self):
        # one block whose max maps to 6 exactly: 6, then 1, then zeros
        w = torch.zeros(1, 16)
        w[0, 0], w[0, 1] = 6.0, 1.0
        packed, sf, ws2 = self.q.quantize_nvfp4_torch(w)
        self.assertEqual(packed[0, 0].item() & 0x0F, 7)          # 6.0 in the low nibble
        self.assertEqual(packed[0, 0].item() >> 4, 2)            # 1.0 in the high one
        back = self.q.dequantize_nvfp4(packed, sf, ws2)
        self.assertTrue(torch.allclose(back, w))
        neg = self.q.quantize_nvfp4_torch(-w)[0]
        self.assertEqual(neg[0, 0].item(), 0xAF)                  # sign bits set: 0xF, 0xA

    def test_four_over_six_never_worse_than_six(self):
        w = torch.randn(32, 512) * torch.rand(32, 1)
        packed, sf, ws2 = self.q.quantize_nvfp4_torch(w)
        mixed = self.q.dequantize_nvfp4(packed, sf, ws2)
        _, _, six = self.q._quantize_blocks(w.view(32, -1, 16), ws2, 6.0)
        e_mixed = ((mixed.view(32, -1, 16) - w.view(32, -1, 16)) ** 2).sum(-1)
        e_six = ((six - w.view(32, -1, 16)) ** 2).sum(-1)
        self.assertTrue(bool((e_mixed <= e_six + 1e-12).all()))
        self.assertLess(e_mixed.sum().item(), e_six.sum().item())   # and better overall

    def test_shared_scale_and_bad_shapes(self):
        w = torch.randn(8, 64)
        given = torch.tensor(0.01)
        self.assertEqual(self.q.quantize_nvfp4_torch(w, given)[2].item(), given.item())
        with self.assertRaises(ValueError):
            self.q.quantize_nvfp4_torch(torch.randn(8, 40))
        with self.assertRaises(ValueError):
            self.q.quantize_nvfp4_torch(torch.randn(2, 8, 64))
        z = self.q.quantize_nvfp4_torch(torch.zeros(4, 32))
        self.assertEqual(z[2].item(), 1.0)
        self.assertEqual(int(z[0].sum()), 0)


@unittest.skipIf(torch is None, "torch not installed")
class WrapperTest(unittest.TestCase):
    def setUp(self):
        import gb10_mtp_fp4 as q

        self.q = q
        q._state.update(torch=False, calls=0)
        self.addCleanup(q._state.update, torch=False, calls=0)
        self.addCleanup(os.environ.pop, "GB10_MTP_FP4_QUANT", None)

    def module(self, orig):
        mod = types.ModuleType(self.q.TARGET_MODULE)

        class ModelOptNvFp4OnlineFusedMoEMethod:
            _quantize_weight_nvfp4 = staticmethod(orig)

        mod.ModelOptNvFp4OnlineFusedMoEMethod = ModelOptNvFp4OnlineFusedMoEMethod
        return mod

    def test_flashinfer_first(self):
        calls = []

        def orig(w, s=None):
            calls.append(w.shape)
            return "flashinfer"

        mod = self.module(orig)
        self.q.apply(mod)
        quant = mod.ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4
        self.assertEqual(quant(torch.randn(4, 32)), "flashinfer")
        self.assertEqual(len(calls), 1)
        self.q.apply(mod)                               # idempotent
        self.assertIs(mod.ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4, quant)

    def test_failure_switches_to_torch_once(self):
        calls = []

        def orig(w, s=None):
            calls.append(1)
            raise RuntimeError("cute-dsl: unsupported arch sm_121")

        mod = self.module(orig)
        self.q.apply(mod)
        quant = mod.ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4
        with self.assertLogs("sglang.srt.layers.quantization.gb10_mtp_fp4", logging.WARNING):
            packed = quant(torch.randn(4, 32))[0]
        self.assertEqual(tuple(packed.shape), (4, 16))
        quant(torch.randn(4, 32))
        self.assertEqual(len(calls), 1)                 # not retried

    def test_torch_mode_never_calls_flashinfer(self):
        os.environ["GB10_MTP_FP4_QUANT"] = "torch"

        def orig(w, s=None):
            raise AssertionError("called")

        mod = self.module(orig)
        self.q.apply(mod)
        out = mod.ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4(torch.randn(2, 16))
        self.assertEqual(out[0].dtype, torch.uint8)

    def test_refusals(self):
        with self.assertRaises(RuntimeError):
            self.q.apply(types.ModuleType(self.q.TARGET_MODULE))
        os.environ["GB10_MTP_FP4_QUANT"] = "fast"
        with self.assertRaises(RuntimeError):
            self.q.apply(self.module(lambda w, s=None: None))


class SitecustomizeTest(unittest.TestCase):
    def test_hook_registered(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sitecustomize.py")) as f:
            src = f.read()
        self.assertIn('os.environ.get("GB10_MTP_FP4") == "1"', src)
        self.assertIn("gb10_mtp_fp4.TARGET_MODULE, gb10_mtp_fp4.apply", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
