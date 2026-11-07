#!/usr/bin/env python3
"""CPU tests for gb10_fp8_prefill (torch, triton for the gather; no GPU, no SGLang):

    python3 patches/test_gb10_fp8_prefill.py

A fake prepare_fp8_layer_for_marlin packs bytes the way Marlin's 8-bit repack
does (vLLM's marlin_utils_test reference: 16 x 16 tiles, four of them a
1024-element block permuted by get_weight_perm(8), four bytes an int32). The
tests learn that order from probes, restore random weights from it in torch and
in the Triton gather (under TRITON_INTERPRET=1), refuse a layout without the
tile structure, and route a converted layer's inputs: prefill-sized ones to
W8A8 (fake fp8_scaled_mm / per-token quant in torch), decode-sized ones and
everything after a failure to Marlin. On the Spark the boot log has "FP8 side
prefill: W8A8 for inputs of 2048+ rows (Marlin byte order learned and checked
...)" or the reason it stayed off.
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


def marlin_perm8():
    """vLLM marlin_utils_test.get_weight_perm(num_bits=8)."""
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend(p + 256 * j for p in perm1)
    perm = torch.tensor(perm).view(-1, 4)[:, [0, 2, 1, 3]].flatten()
    return perm


def marlin_pack(w_nk):
    """uint8 [N, K] -> Marlin's packed int32 [K / 16, 4 N] (marlin_weights, 8 bits)."""
    n, k = w_nk.shape
    q = w_nk.T.contiguous().view(k // 16, 16, n // 16, 16).permute(0, 2, 1, 3).reshape(k // 16, n * 16)
    q = q.reshape(-1, 1024)[:, marlin_perm8()].reshape(k // 16, n * 16)
    return q.contiguous().view(torch.int32)


def fake_prepare(m, size_k_first=True):
    assert not size_k_first
    w = m.weight.data.view(torch.uint8)
    assert tuple(w.shape) == (m.output_size_per_partition, m.input_size_per_partition)
    m.weight = torch.nn.Parameter(marlin_pack(w), requires_grad=False)
    m.workspace = None


def fake_prepare_plain(m, size_k_first=True):
    """A layout without Marlin's tiles: K-major bytes."""
    m.weight = torch.nn.Parameter(m.weight.data.view(torch.uint8).T.contiguous(), requires_grad=False)
    m.workspace = None


_REF = {}


def fake_marlin_apply(input, weight, weight_scale, workspace, size_n, size_k, bias):  # noqa: A002
    wd = _REF[weight.data_ptr()]
    out = input.float() @ wd.T
    return (out if bias is None else out + bias).to(input.dtype)


def fake_quant(x):
    s = x.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
    return (x.float() / s).to(torch.float8_e4m3fn), s


def fake_mm(xq, wt, xs, ws, dtype, bias=None):
    out = (xq.float() * xs) @ wt.float() * ws.view(1, -1)
    return (out if bias is None else out + bias).to(dtype)


@unittest.skipIf(torch is None, "torch not installed")
class LayoutTest(unittest.TestCase):
    def setUp(self):
        import gb10_fp8_prefill as p

        self.p = p

    def test_learn_and_round_trip(self):
        bytes_of = self.p._marlin_bytes_via(fake_prepare, "cpu")
        inv = self.p.learn_layout(bytes_of)
        self.assertEqual(sorted(inv.tolist()), list(range(1024)))
        g = torch.Generator().manual_seed(1)
        for n, k in ((64, 16), (192, 384), (128, 2560)):
            w = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
            packed = marlin_pack(w)
            self.assertTrue(torch.equal(self.p.unpack_torch(packed, n, k, inv), w), (n, k))

    def test_refuses_a_layout_without_tiles(self):
        with self.assertRaises(ValueError):
            self.p.learn_layout(self.p._marlin_bytes_via(fake_prepare_plain, "cpu"))
        with self.assertRaises(ValueError):   # not a permutation at all
            self.p.learn_layout(lambda w: torch.zeros(w.numel(), dtype=torch.uint8))


def gather_selfcheck():
    """Run in a child with TRITON_INTERPRET=1: the Triton gather against torch."""
    import gb10_fp8_prefill as p

    inv = p.learn_layout(p._marlin_bytes_via(fake_prepare, "cpu"))
    g = torch.Generator().manual_seed(2)
    for n, k in ((64, 128), (192, 256)):
        w = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
        out = p.unpack(marlin_pack(w), n, k, inv)
        assert torch.equal(out, w), (n, k)
    print("gather ok")


@unittest.skipIf(torch is None, "torch not installed")
class GatherTest(unittest.TestCase):
    def test_triton_gather_under_the_interpreter(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton not installed")
        env = dict(os.environ, TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--gather-selfcheck"],
                           env=env, capture_output=True, text=True, timeout=600, check=False)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("gather ok", r.stdout)


@unittest.skipIf(torch is None, "torch not installed")
class RoutingTest(unittest.TestCase):
    def setUp(self):
        import gb10_fp8_prefill as p
        import gb10_fp8_side as side

        self.p, self.side = p, side
        self.reset()
        self.addCleanup(self.reset)
        os.environ["GB10_FP8_SIDE_PREFILL"] = "w8a8"
        os.environ["GB10_FP8_SIDE_PREFILL_ROWS"] = "8"
        self.addCleanup(os.environ.pop, "GB10_FP8_SIDE_PREFILL", None)
        self.addCleanup(os.environ.pop, "GB10_FP8_SIDE_PREFILL_ROWS", None)
        self.addCleanup(setattr, p, "MIN_WEIGHT", p.MIN_WEIGHT)
        p.MIN_WEIGHT = 64 * 256
        torch.manual_seed(0)

    def reset(self):
        self.p._state.update(ready=None, inv=None, why="", runtime_ok=True, used=0)
        self.p._ops.clear()
        self.p._ops.update(mm=fake_mm, quant=fake_quant)

    def layer(self, n=96, k=256):
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter((torch.randn(n, k) * 0.02).to(torch.bfloat16), requires_grad=False)
        q, s = self.side.quantize_per_channel(m.weight.data)
        wd = q.float() * s[:, None]
        self.side.convert_module(m, fake_prepare, fake_marlin_apply)
        n_pad = m._gb10_marlin_n
        _REF[m.weight.data_ptr()] = torch.cat([wd, torch.zeros(n_pad - n, k)])
        return m, wd

    def test_setup_checks_the_gemm(self):
        self.assertTrue(self.p.setup(fake_prepare, "cpu"))
        self.assertTrue(self.p.setup(None, "cpu"))          # cached
        self.reset()
        self.p._ops["mm"] = lambda xq, wt, xs, ws, dtype, bias=None: torch.zeros(
            xq.shape[0], wt.shape[1], dtype=dtype)
        self.assertFalse(self.p.setup(fake_prepare, "cpu"))
        self.assertIn("off by", self.p._state["why"])

    def test_prefill_rows_take_w8a8_decode_rows_marlin(self):
        m, wd = self.layer()
        self.assertTrue(self.p.mark(m, fake_prepare))
        self.assertEqual(m._gb10_marlin_n, 128)          # 96 padded to Marlin's tile
        big = torch.randn(16, 256).to(torch.bfloat16)
        out = m.quant_method.apply(m, big)
        self.assertEqual(tuple(out.shape), (16, 96))
        self.assertEqual(self.p._state["used"], 1)
        ref = big.float() @ wd.T
        self.assertLess(((out.float() - ref).norm() / ref.norm()).item(), 0.08)
        small = torch.randn(4, 256).to(torch.bfloat16)
        out = m.quant_method.apply(m, small)
        self.assertEqual(self.p._state["used"], 1)       # Marlin
        self.assertLess(((out.float() - small.float() @ wd.T).norm()
                         / (small.float() @ wd.T).norm()).item(), 0.01)

    def test_chunks_long_inputs(self):
        m, _ = self.layer()
        self.p.mark(m, fake_prepare)
        calls = []
        mm = self.p._ops["mm"]
        self.p._ops["mm"] = lambda xq, *a: calls.append(xq.shape[0]) or mm(xq, *a)
        self.addCleanup(setattr, self.p, "CHUNK_ROWS", self.p.CHUNK_ROWS)
        self.p.CHUNK_ROWS = 16
        out = m.quant_method.apply(m, torch.randn(2, 20, 256).to(torch.bfloat16))
        self.assertEqual(tuple(out.shape), (2, 20, 96))
        self.assertEqual(calls, [16, 16, 8])

    def test_failure_goes_back_to_marlin(self):
        m, wd = self.layer()
        self.p.mark(m, fake_prepare)

        def broken(*a):
            raise RuntimeError("no kernel image for sm_121")
        self.p._ops["mm"] = broken
        x = torch.randn(16, 256).to(torch.bfloat16)
        with self.assertLogs("sglang.srt.models.qwen4_exp.gb10_fp8_prefill", "WARNING"):
            out = m.quant_method.apply(m, x)
        ref = x.float() @ wd.T
        self.assertLess(((out.float() - ref).norm() / ref.norm()).item(), 0.01)
        self.assertFalse(self.p._state["runtime_ok"])
        self.assertFalse(self.p.wants(m, x))

    def test_not_marked_when_off_small_or_unready(self):
        m, _ = self.layer()
        os.environ["GB10_FP8_SIDE_PREFILL"] = "marlin"
        self.assertFalse(self.p.mark(m, fake_prepare))
        os.environ["GB10_FP8_SIDE_PREFILL"] = "w8a8"
        small, _ = self.layer(n=64, k=128)
        self.assertFalse(self.p.mark(small, fake_prepare))            # under MIN_WEIGHT
        self.p._state["ready"] = False
        self.assertFalse(self.p.mark(m, fake_prepare))
        self.assertFalse(getattr(m, "_gb10_w8a8", False))
        os.environ["GB10_FP8_SIDE_PREFILL"] = "fast"
        with self.assertRaises(RuntimeError):
            self.p.mode()

    def test_convert_model_marks_layers(self):
        model = torch.nn.Module()
        model.big = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
        model.tiny = torch.nn.Linear(128, 64, bias=False, dtype=torch.bfloat16)
        stats = self.side.convert_model(
            model, is_linear=lambda mod: isinstance(mod, torch.nn.Linear),
            is_unquantized=lambda qm: qm is None, prepare_fn=fake_prepare,
            apply_fn=fake_marlin_apply, pattern=__import__("re").compile(r"big|tiny"))
        self.assertEqual((stats["converted"], stats["w8a8"]), (2, 1))
        self.assertTrue(model.big._gb10_w8a8)
        self.assertFalse(getattr(model.tiny, "_gb10_w8a8", False))


if __name__ == "__main__":
    if sys.argv[1:] == ["--gather-selfcheck"]:
        gather_selfcheck()
        sys.exit(0)
    unittest.main(verbosity=2)
