#!/usr/bin/env python3
"""CPU tests for gb10_marlin_lean (torch only, no GPU, no SGLang):

    python3 patches/test_gb10_marlin_lean.py

The repack must produce exactly what SGLang 0.5.20's list + torch.stack
version does, and the per-layer check must name what keeps a layer's
original weights alive. The real kernels run only on the Spark: the boot log
then has one "Marlin repack, MoE layer N" line per layer.
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


def fake_repack(*, b_q_weight, perm, size_k, size_n, num_bits):
    """Stands in for gptq_marlin_repack: a layout change with another shape."""
    assert num_bits == 4 and b_q_weight.dtype == torch.int32
    return (b_q_weight.reshape(-1, 2).flip(1).reshape(size_k // 16, -1) + 1).contiguous()


def stock_repack(weight, *, num_experts, size_n, size_k, perm):
    """SGLang 0.5.20's _repack_moe_fp4_weight_for_marlin, with the fake kernel."""
    tensor_list = []
    for i in range(num_experts):
        qweight = weight[i].view(torch.int32).T.contiguous()
        tensor_list.append(fake_repack(b_q_weight=qweight, perm=perm, size_k=size_k,
                                       size_n=size_n, num_bits=4))
    return torch.stack(tensor_list)


@unittest.skipIf(torch is None, "torch not installed")
class RepackTest(unittest.TestCase):
    def test_same_result_as_list_and_stack(self):
        import gb10_marlin_lean as m

        e, n, k = 5, 32, 64
        w = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8)
        perm = torch.empty(0, dtype=torch.int)
        want = stock_repack(w, num_experts=e, size_n=n, size_k=k, perm=perm)
        got = m.repack_into_one(w, num_experts=e, size_n=n, size_k=k, perm=perm, repack=fake_repack)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.dtype, want.dtype)
        self.assertTrue(torch.equal(got, want))


@unittest.skipIf(torch is None, "torch not installed")
class ApplyTest(unittest.TestCase):
    def setUp(self):
        import gb10_marlin_lean as m

        self.m = m
        m._state.update(layer=0, reported=False)
        self.leak = []
        mfp4 = types.ModuleType(m.REPACK_MODULE)
        mfp4.gptq_marlin_repack = fake_repack
        mfp4._repack_moe_fp4_weight_for_marlin = stock_repack
        self.saved = sys.modules.get(m.REPACK_MODULE)
        sys.modules[m.REPACK_MODULE] = mfp4
        self.mfp4 = mfp4

        def prepare(layer, leak=self.leak, mfp4=mfp4):
            # the real function's shape: new Parameters replace the old ones
            w = layer.w13_weight.data
            if layer.leaky:
                leak.append(w)
            e, n, k2 = w.shape
            layer.w13_weight = torch.nn.Parameter(
                mfp4._repack_moe_fp4_weight_for_marlin(
                    w, num_experts=e, size_n=n, size_k=k2 * 2, perm=None),
                requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(layer.w2_weight.data.clone(), requires_grad=False)

        self.mod = types.ModuleType(m.TARGET_MODULE)
        self.mod.prepare_moe_nvfp4_layer_for_marlin = prepare
        m.apply(self.mod)

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop(self.m.REPACK_MODULE, None)
        else:
            sys.modules[self.m.REPACK_MODULE] = self.saved

    def layer(self, leaky=False):
        layer = torch.nn.Module()
        layer.leaky = leaky
        layer.w13_weight = torch.nn.Parameter(
            torch.randint(0, 256, (4, 32, 32), dtype=torch.uint8), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(
            torch.randint(0, 256, (4, 64, 16), dtype=torch.uint8), requires_grad=False)
        return layer

    def test_lean_repack_is_used_and_each_layer_logged(self):
        with self.assertLogs(self.m.logger, level="INFO") as logs:
            for _ in range(3):
                layer = self.layer()
                self.mod.prepare_moe_nvfp4_layer_for_marlin(layer)
        self.assertIsNot(self.mfp4._repack_moe_fp4_weight_for_marlin, stock_repack)
        lines = [r for r in logs.output if "Marlin repack, MoE layer" in r]
        self.assertEqual(len(lines), 3)
        self.assertFalse(any("STILL ALIVE" in r or "holders" in r for r in logs.output))

    def test_a_kept_reference_is_named(self):
        # CPU has no CUDA allocator: count what the "leak" holds as allocated.
        real = torch.cuda.memory_allocated
        torch.cuda.memory_allocated = lambda *a: sum(t.numel() for t in self.leak)
        self.addCleanup(setattr, torch.cuda, "memory_allocated", real)
        with self.assertLogs(self.m.logger, level="INFO") as logs:
            self.mod.prepare_moe_nvfp4_layer_for_marlin(self.layer(leaky=True))
            self.mod.prepare_moe_nvfp4_layer_for_marlin(self.layer(leaky=True))
        warnings = [r for r in logs.output if r.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)  # reported once
        self.assertIn("list of len", warnings[0])

    def test_refuses_other_versions(self):
        with self.assertRaisesRegex(RuntimeError, "0.5.20"):
            self.m.apply(types.ModuleType("empty"))


class HookTest(unittest.TestCase):
    def test_two_targets_both_fire(self):
        import importlib
        import tempfile

        import gb10_ple_mmap as g

        with tempfile.TemporaryDirectory() as d:
            for name in ("gb10hookpkg", "gb10hookpkg/a.py", "gb10hookpkg/b.py"):
                p = os.path.join(d, name)
                if name.endswith(".py"):
                    open(p, "w").write("X = 1\n")
                else:
                    os.mkdir(p)
                    open(os.path.join(p, "__init__.py"), "w").close()
            seen = []
            sys.path.insert(0, d)
            try:
                g.install_import_hook("gb10hookpkg.a", lambda mod: seen.append("a"))
                g.install_import_hook("gb10hookpkg.b", lambda mod: seen.append("b"))
                importlib.import_module("gb10hookpkg.a")
                importlib.import_module("gb10hookpkg.b")
                self.assertEqual(seen, ["a", "b"])
            finally:
                sys.path.remove(d)
                sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, g._Finder)]
                for mname in ("gb10hookpkg.a", "gb10hookpkg.b", "gb10hookpkg"):
                    sys.modules.pop(mname, None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
