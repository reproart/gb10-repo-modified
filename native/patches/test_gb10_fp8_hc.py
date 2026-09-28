#!/usr/bin/env python3
"""CPU tests for gb10_fp8_hc (torch, triton for the kernel check; no GPU, no SGLang):

    python3 patches/test_gb10_fp8_hc.py

Covers which weights are converted, the routing of fused_hc_mix calls (FP8
kernel, the torch path for prefill-sized inputs, untouched BF16 weights),
and the FP8 kernel against SGLang's mix math under the Triton interpreter
(TRITON_INTERPRET=1). The interpreter runs one program at a time, so the
kernel's grid barrier is exercised with a single CTA, and it mishandles
BF16 tl.dot, so the activations are FP16 there (the kernel follows x's
dtype). On the Spark the boot log has "FP8 HC (target): 97 hyper-connection
mixes to FP8" and answers must stay right.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

HC, HS, LOWRANK = 4, 64, 40   # K = 256; LOWRANK not a multiple of the 32-wide tiles


def mix_bf16(x, w_down, w_up, hc, hs):
    """SGLang's _mix_compute (layers/hyperconnection.py)."""
    t = F.silu(F.linear(x, w_down) / hc)
    gate = torch.sigmoid(F.linear(t, w_up)).unflatten(-1, (hc, hs))
    return (gate * x.unflatten(-1, (hc, hs))).mean(dim=-2)


def hc_module():
    m = torch.nn.Module()
    m.input_mix_weight_down = torch.nn.Linear(HC * HS, LOWRANK, bias=False, dtype=torch.bfloat16)
    m.input_mix_weight_up = torch.nn.Linear(LOWRANK, HC * HS, bias=False, dtype=torch.bfloat16)
    m.block_inject_weight = torch.nn.Linear(HC * HS, HC, bias=False, dtype=torch.bfloat16)
    return m


@unittest.skipIf(torch is None, "torch not installed")
class ConvertTest(unittest.TestCase):
    def setUp(self):
        import gb10_fp8_hc as h

        self.h = h
        h._patched["hc"] = True
        self.addCleanup(h._patched.__setitem__, "hc", False)
        torch.manual_seed(0)

    def model(self):
        m = torch.nn.Module()
        m.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.attn_hyper_connection = hc_module()
            layer.mlp_hyper_connection = hc_module()
            m.layers.append(layer)
        m.hyper_connection_mixer = hc_module()
        return m

    def test_convert(self):
        m = self.model()
        ref = m.layers[0].attn_hyper_connection.input_mix_weight_up.weight.detach().float().clone()
        stats = self.h.convert_model(m)
        self.assertEqual(stats["mixes"], 5)
        hc = m.layers[0].attn_hyper_connection
        w = hc.input_mix_weight_up.weight
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(self.h.scale_of(w).shape, (HC * HS,))
        back = w.float() * self.h.scale_of(w)[:, None]
        self.assertLess(((back - ref).norm() / ref.norm()).item(), 0.05)
        self.assertEqual(hc.block_inject_weight.weight.dtype, torch.bfloat16)   # left alone
        # FP8 plus a float32 scale per row (a sizable share only at these toy sizes)
        self.assertLess(stats["bytes_after"], 0.55 * stats["bytes_before"])
        self.assertEqual(self.h.convert_model(m)["mixes"], 0)                    # idempotent

    def test_refuses_without_the_hc_patch(self):
        self.h._patched["hc"] = False
        with self.assertRaises(RuntimeError):
            self.h.convert_model(self.model())

    def test_routing(self):
        calls = {"orig_mix": 0, "orig_sup": 0, "kernel": []}

        def orig_supported(x, wd, wu):
            calls["orig_sup"] += 1
            return True

        def orig_mix(x, wd, wu, hc, hs):
            calls["orig_mix"] += 1
            return mix_bf16(x, wd, wu, hc, hs)

        def kernel(x, wd, sd, wu, su, hc, hs):
            calls["kernel"].append(tuple(x.shape))
            return self.h.mix_reference(x, wd, sd, wu, su, hc, hs)

        supported, mix = self.h.make_wrappers(orig_supported, orig_mix, kernel)
        plain = hc_module()
        conv = hc_module()
        wd_bf16 = conv.input_mix_weight_down.weight.detach().clone()
        wu_bf16 = conv.input_mix_weight_up.weight.detach().clone()
        holder = torch.nn.Module()
        holder.hc = conv
        self.h.convert_model(holder)
        wd, wu = conv.input_mix_weight_down.weight, conv.input_mix_weight_up.weight

        x = torch.randn(4, HC * HS, dtype=torch.bfloat16)
        # BF16 weights: SGLang's own functions
        supported(x, plain.input_mix_weight_down.weight, plain.input_mix_weight_up.weight)
        mix(x, plain.input_mix_weight_down.weight, plain.input_mix_weight_up.weight, HC, HS)
        self.assertEqual((calls["orig_sup"], calls["orig_mix"]), (1, 1))
        # FP8 weights, any size: ours
        self.assertTrue(supported(torch.randn(100, HC * HS, dtype=torch.bfloat16), wd, wu))
        self.assertEqual(calls["orig_sup"], 1)
        # a CPU tensor is never the kernel's (it needs CUDA): the torch path, checked
        # against the BF16 math
        big = torch.randn(40, HC * HS, dtype=torch.bfloat16)
        out = mix(big, wd, wu, HC, HS)
        ref = mix_bf16(big.float(), wd_bf16.float(), wu_bf16.float(), HC, HS)
        self.assertEqual(tuple(out.shape), (40, HS))
        self.assertLess(((out.float() - ref).norm() / ref.norm()).item(), 0.03)
        self.assertEqual(calls["kernel"], [])
        # a CUDA-looking small input goes to the kernel
        class FakeCuda(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        fake = x.as_subclass(FakeCuda)
        mix(fake, wd, wu, HC, HS)
        self.assertEqual(calls["kernel"], [(4, HC * HS)])
        mix(torch.randn(32, HC * HS, dtype=torch.bfloat16).as_subclass(FakeCuda), wd, wu, HC, HS)
        self.assertEqual(calls["kernel"][-1], (32, HC * HS))            # a verify at 8 requests
        mix(torch.randn(65, HC * HS, dtype=torch.bfloat16).as_subclass(FakeCuda), wd, wu, HC, HS)
        self.assertEqual(len(calls["kernel"]), 2)                       # 65 rows: torch path

    def test_wide_tile_failure_falls_back_once(self):
        state = {"n": 0}

        def kernel(x, *a):
            state["n"] += 1
            if x.shape[0] > 16:
                raise RuntimeError("out of resources: shared memory")
            return torch.zeros(x.shape[0], HS, dtype=x.dtype)

        class FakeCuda(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        self.addCleanup(self.h._wide.__setitem__, "ok", True)
        _, mix = self.h.make_wrappers(None, None, kernel)
        conv = hc_module()
        holder = torch.nn.Module()
        holder.hc = conv
        self.h.convert_model(holder)
        wd, wu = conv.input_mix_weight_down.weight, conv.input_mix_weight_up.weight
        wide = torch.randn(32, HC * HS, dtype=torch.bfloat16).as_subclass(FakeCuda)
        self.assertEqual(tuple(mix(wide, wd, wu, HC, HS).shape), (32, HS))    # torch path
        mix(wide, wd, wu, HC, HS)                                            # not retried
        self.assertEqual(state["n"], 1)
        mix(torch.randn(8, HC * HS, dtype=torch.bfloat16).as_subclass(FakeCuda), wd, wu, HC, HS)
        self.assertEqual(state["n"], 2)                                      # 16-row kernel still used

    def test_apply_hc_is_idempotent(self):
        import types

        mod = types.ModuleType(self.h.HC_MODULE)
        mod.fused_hc_mix = lambda *a: "orig"
        mod.fused_hc_mix_supported = lambda *a: True
        self.h.apply_hc(mod)
        first = mod.fused_hc_mix
        self.h.apply_hc(mod)
        self.assertIs(mod.fused_hc_mix, first)
        with self.assertRaises(RuntimeError):
            self.h.apply_hc(types.ModuleType(self.h.HC_MODULE))


def kernel_selfcheck():
    """Run in a child with TRITON_INTERPRET=1. Prints the worst relative error
    of the FP8 kernel against the same math in torch on the same FP8 weights."""
    from gb10_fp8_hc_kernel import fused_hc_mix_fp8
    from gb10_fp8_side import quantize_per_channel

    torch.manual_seed(0)
    worst = 0.0
    for rows in (1, 4, 16, 17, 32, 33, 64):
        x = torch.randn(rows, HC * HS).to(torch.float16)
        wd, sd = quantize_per_channel(torch.randn(LOWRANK, HC * HS) * 0.05)
        wu, su = quantize_per_channel(torch.randn(HC * HS, LOWRANK) * 0.2)
        out = fused_hc_mix_fp8(x, wd, sd, wu, su, HC, HS, num_ctas=1)
        # the same math in FP32 with the weights dequantized
        wdf, wuf = wd.float() * sd[:, None], wu.float() * su[:, None]
        t = torch.nn.functional.silu(x.float() @ wdf.T / HC)
        gate = torch.sigmoid(t @ wuf.T).unflatten(-1, (HC, HS))
        ref = (gate * x.float().unflatten(-1, (HC, HS))).mean(dim=-2)
        assert out.shape == (rows, HS) and out.dtype == torch.float16
        worst = max(worst, ((out.float() - ref).norm() / ref.norm()).item())
    try:
        fused_hc_mix_fp8(torch.randn(65, HC * HS).half(), wd, sd, wu, su, HC, HS, num_ctas=1)
        raise AssertionError("65 rows accepted")
    except ValueError:
        pass
    print(f"worst relative error: {worst:.2e}")


@unittest.skipIf(torch is None, "torch not installed")
class KernelTest(unittest.TestCase):
    def test_kernel_under_the_interpreter(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton not installed")
        env = dict(os.environ, TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--kernel-selfcheck"],
                           env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        worst = float(r.stdout.strip().rsplit(" ", 1)[-1])
        self.assertLess(worst, 5e-3, r.stdout)   # FP16 activations, same FP8 weights


if __name__ == "__main__":
    if sys.argv[1:] == ["--kernel-selfcheck"]:
        kernel_selfcheck()
        sys.exit(0)
    unittest.main(verbosity=2)
