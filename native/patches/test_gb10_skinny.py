#!/usr/bin/env python3
"""CPU tests for gb10_skinny (torch, triton for the kernel check; no GPU, no SGLang):

    python3 patches/test_gb10_skinny.py

Covers which layers are routed to the skinny GEMM, when a call falls back to
the layer's original method, and the Triton kernel's indexing and masking
under the interpreter (TRITON_INTERPRET=1). The interpreter mishandles BF16
in tl.dot, so the kernel is checked in FP16 and FP32 there (the code is
dtype-generic); on the Spark the boot log has "BF16 skinny GEMM (target): N
layers" and answers must stay the same.
"""

import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


class Unquant:
    def __init__(self):
        self.calls = 0
        self.flag = "orig attr"

    def apply(self, layer, x, bias=None):
        self.calls += 1
        return x.float() @ layer.weight.float().T


class Other:
    pass


class Linear(torch.nn.Module if torch else object):
    def __init__(self, n, k, dtype=None, qm=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k).to(dtype or torch.bfloat16),
                                         requires_grad=False)
        self.bias = None
        self.quant_method = qm or Unquant()


@unittest.skipIf(torch is None, "torch not installed")
class ConvertTest(unittest.TestCase):
    def setUp(self):
        import gb10_skinny as s

        self.s = s
        self.gemm_calls = []

    def gemm(self, x, w):
        self.gemm_calls.append((tuple(x.shape), x.is_contiguous()))
        return (x.float() @ w.float().T).to(x.dtype)

    def model(self):
        m = torch.nn.Module()
        m.layers = torch.nn.ModuleList()
        for i in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.gate = Linear(512, 256)
            layer.mlp.shared_expert_gate = Linear(1, 256)       # not in the pattern
            layer.self_attn = torch.nn.Module()
            layer.self_attn.indexer = torch.nn.Module()
            layer.self_attn.indexer.index_qk_proj = Linear(640, 256)
            layer.self_attn.qkv_proj = Linear(384, 256, qm=Other())  # already FP8 etc.
            m.layers.append(layer)
        m.layers[1].mlp.gate = Linear(512, 250)                  # K % 16: left alone
        return m

    def convert(self, m):
        return self.s.convert_model(m, is_linear=lambda mod: isinstance(mod, Linear),
                                    is_unquantized=lambda qm: isinstance(qm, Unquant),
                                    gemm=self.gemm)

    def test_selection(self):
        m = self.model()
        done = self.convert(m)
        self.assertEqual(sorted(done), ["layers.0.mlp.gate", "layers.0.self_attn.indexer.index_qk_proj",
                                        "layers.1.self_attn.indexer.index_qk_proj"])
        self.assertIsInstance(m.layers[1].mlp.gate.quant_method, Unquant)
        self.assertIsInstance(m.layers[0].mlp.shared_expert_gate.quant_method, Unquant)
        self.assertIsInstance(m.layers[0].self_attn.qkv_proj.quant_method, Other)

    def test_routing_and_fallbacks(self):
        m = self.model()
        self.convert(m)
        gate = m.layers[0].mlp.gate
        qm = gate.quant_method
        self.assertEqual(qm.flag, "orig attr")                   # other attributes pass through
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        out = qm.apply(gate, x)
        self.assertEqual(self.gemm_calls, [((4, 256), True)])
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out.float(), x.float() @ gate.weight.float().T, rtol=2e-2, atol=2e-1)
        # prefill-sized input, bias, 3-D input, fp32 input: the original method
        qm.apply(gate, torch.randn(65, 256, dtype=torch.bfloat16))
        qm.apply(gate, x, bias=torch.zeros(512, dtype=torch.bfloat16))
        qm.apply(gate, x.view(2, 2, 256))
        qm.apply(gate, x.float())
        self.assertEqual(qm.orig.calls, 4)
        self.assertEqual(len(self.gemm_calls), 1)
        # a column slice of a wider tensor: made contiguous in the last dim
        wide = torch.randn(4, 512, dtype=torch.bfloat16)[:, ::2]
        qm.apply(gate, wide)
        self.assertEqual(self.gemm_calls[-1], ((4, 256), True))

    def test_max_m_env(self):
        os.environ["GB10_SKINNY_MAX_M"] = "8"
        self.addCleanup(os.environ.pop, "GB10_SKINNY_MAX_M")
        m = self.model()
        self.convert(m)
        gate = m.layers[0].mlp.gate
        gate.quant_method.apply(gate, torch.randn(9, 256, dtype=torch.bfloat16))
        self.assertEqual(self.gemm_calls, [])
        self.assertEqual(gate.quant_method.orig.calls, 1)


def kernel_selfcheck():
    """Run in a child with TRITON_INTERPRET=1. Prints the worst relative error."""
    from gb10_skinny_kernel import skinny_gemm

    torch.manual_seed(0)
    worst = 0.0
    for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 2e-3)):
        for m, n, k in ((4, 512, 256), (1, 40, 272), (20, 48, 512), (17, 16, 16)):
            x = torch.randn(m, k).to(dtype)
            w = torch.randn(n, k).to(dtype)
            wide = torch.randn(m, k + 32).to(dtype)[:, :k]      # row stride != K
            for xi in (x, wide):
                y = skinny_gemm(xi, w)
                ref = xi.float() @ w.float().T
                assert y.shape == (m, n) and y.dtype == dtype
                err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                worst = max(worst, err / tol)
    print(f"worst error / tolerance: {worst:.3f}")


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
        self.assertLess(worst, 1.0, r.stdout)


if __name__ == "__main__":
    if sys.argv[1:] == ["--kernel-selfcheck"]:
        kernel_selfcheck()
        sys.exit(0)
    unittest.main(verbosity=2)
