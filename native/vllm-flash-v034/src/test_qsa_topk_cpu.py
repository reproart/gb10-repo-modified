#!/usr/bin/env python3
"""CPU checks for patch_qsa_topk.py (torch; no GPU, no vLLM):

    python3 src/test_qsa_topk_cpu.py

1. The patch on vLLM 0.34.0's own qsa_indexer.py (src/testdata, from the wheel): the choice
   goes in front of the persistent_topk call in _topk, once; the file parses; a second run
   changes nothing; a file without that call is refused.
2. The settings: off by default; VLLM_QSA_EXACT_TOPK=1 wins over VLLM_QSA_DET_TOPK=1; a
   deterministic kernel that does not load falls back to the exact path with a warning.
3. 0.34's patched _topk itself (exec'd from the patched file with a stand-in platform and
   stand-in _C ops): stock kernel when nothing is set; the deterministic op, with the stock
   call's arguments, when it is loaded; the exact path when it is asked for, and when the
   deterministic op is loaded but block_topk is not one it is built for.
4. The exact top-k against a reference: for every row the first min(visible, block_topk)
   slots (all that 0.34's expansion kernel reads) hold exactly the ids of the row's highest
   logits among its visible columns, in ascending order; the garbage past a row's visible
   count never gets in; fewer columns than block_topk leaves -1 after them; two runs give the
   same ids; nothing outside the rows' slots is written.
"""
import ast
import os
import shutil
import sys
import tempfile
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_qsa_topk as pq

V034 = os.path.join(HERE, "testdata", "qsa_indexer.py.v034")
ENV = ("VLLM_QSA_EXACT_TOPK", "VLLM_QSA_DET_TOPK", "VLLM_QSA_DET_LIB")


def hook_ns(env, torch_mod=torch):
    for k in ENV:
        os.environ.pop(k, None)
    os.environ.update(env)
    ns = {"torch": torch_mod}
    try:
        exec(compile(pq.HOOK, "hook", "exec"), ns)  # noqa: S102 - the hook as the patch appends it
    finally:
        for k in ENV:
            os.environ.pop(k, None)
    return ns


def exact_topk():
    ns = hook_ns({"VLLM_QSA_EXACT_TOPK": "1"})
    assert ns["_GB10_QSA_EXACT_TOPK"] is True
    return ns["_gb10_qsa_exact_topk"]


def patched_source():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "qsa_indexer.py")
        shutil.copy(V034, p)
        pq.patch(p)
        with open(p) as f:
            return f.read()


class PatchTest(unittest.TestCase):
    def test_v034_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "qsa_indexer.py")
            shutil.copy(V034, p)
            self.assertEqual(pq.patch(p), "INSTALLED")
            self.assertEqual(pq.patch(p), "already installed")
            with open(p) as f:
                src = f.read()
            body = src[src.index("def _topk("):src.index("def qsa_select_paged_decode(")]
            self.assertLess(body.index("_gb10_qsa_exact_topk("), body.index("    topk_op(\n"))
            self.assertLess(body.index("topk_op = _GB10_QSA_DET_OP"), body.index("    topk_op(\n"))
            with open(p, "w") as f:
                f.write("import torch\n")
            with self.assertRaises(SystemExit):
                pq.patch(p)


class SettingsTest(unittest.TestCase):
    def test_off_by_default(self):
        ns = hook_ns({})
        self.assertFalse(ns["_GB10_QSA_EXACT_TOPK"])
        self.assertIsNone(ns["_GB10_QSA_DET_OP"])

    def test_exact_wins(self):
        ns = hook_ns({"VLLM_QSA_EXACT_TOPK": "1", "VLLM_QSA_DET_TOPK": "1"})
        self.assertTrue(ns["_GB10_QSA_EXACT_TOPK"])
        self.assertIsNone(ns["_GB10_QSA_DET_OP"])

    def test_missing_kernel_falls_back_to_exact(self):
        with self.assertLogs("vllm.gb10_qsa_topk", "WARNING") as logs:
            ns = hook_ns({"VLLM_QSA_DET_TOPK": "1", "VLLM_QSA_DET_LIB": "/nonexistent/_C_det.so"})
        self.assertTrue(ns["_GB10_QSA_EXACT_TOPK"])
        self.assertIsNone(ns["_GB10_QSA_DET_OP"])
        self.assertTrue(any("did not load" in r and "exact torch.topk" in r for r in logs.output))

    def test_loaded_kernel(self):
        det = object()
        fake_ops = types.SimpleNamespace(load_library=lambda path: None,
                                         _C_det=types.SimpleNamespace(persistent_topk=det))
        fake_torch = types.SimpleNamespace(ops=fake_ops)
        with self.assertLogs("vllm.gb10_qsa_topk", "INFO") as logs:
            ns = hook_ns({"VLLM_QSA_DET_TOPK": "1"}, torch_mod=fake_torch)
        self.assertIs(ns["_GB10_QSA_DET_OP"], det)
        self.assertFalse(ns["_GB10_QSA_EXACT_TOPK"])
        self.assertTrue(any("deterministic persistent_topk (/opt/llm/kernel-det/_C_det.so" in r for r in logs.output))


class RoutingTest(unittest.TestCase):
    """0.34's _topk as patched, with a stand-in GB10 platform and stand-in _C ops."""

    def topk_fn(self, exact=False, det=None):
        src = patched_source()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_topk")
        calls = []
        stock = lambda *a: calls.append(("stock", a))
        coop = lambda *a: calls.append(("cooperative", a))

        class TorchProxy:
            ops = types.SimpleNamespace(_C=types.SimpleNamespace(persistent_topk=stock, cooperative_topk=coop))

            def __getattr__(self, name):
                return getattr(torch, name)

        ns = hook_ns({})
        ns.update(torch=TorchProxy(), _GB10_QSA_EXACT_TOPK=exact, _GB10_QSA_DET_OP=det,
                  current_platform=types.SimpleNamespace(
                      has_device_capability=lambda c: True,
                      is_device_capability_family=lambda f: f == 120))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "_topk", "exec"), ns)  # noqa: S102
        return ns["_topk"], calls

    def args(self, block_topk=512, columns=1024):
        logits = torch.randn(3, columns, generator=torch.Generator().manual_seed(5))
        visible = torch.tensor([columns, 700, 100], dtype=torch.int32)
        out = torch.full((3, block_topk), -7, dtype=torch.int32)
        ws = torch.empty(16, dtype=torch.uint8)
        return logits, visible, block_topk * 4, 4, out, ws

    def test_stock(self):
        topk, calls = self.topk_fn()
        topk(*self.args())
        self.assertEqual([c[0] for c in calls], ["stock"])   # GB10: never the cooperative kernel

    def test_deterministic_op_gets_the_stock_arguments(self):
        det_calls = []
        topk, calls = self.topk_fn(det=lambda *a: det_calls.append(a))
        logits, visible, token_topk, ratio, out, ws = self.args()
        topk(logits, visible, token_topk, ratio, out, ws)
        self.assertEqual(calls, [])
        (a,) = det_calls
        self.assertTrue(a[0] is logits and a[1] is visible and a[2] is out and a[3] is ws)
        self.assertEqual(a[4:], (512, 1024))

    def test_exact(self):
        for exact, det in ((True, None), (False, lambda *a: self.fail("det op called for k=256"))):
            topk, calls = self.topk_fn(exact=exact, det=det)
            logits, visible, token_topk, ratio, out, ws = self.args(block_topk=256 if det else 512)
            topk(logits, visible, token_topk, ratio, out, ws)
            self.assertEqual(calls, [])
            want = sorted(torch.topk(logits[2, :100], 100).indices.tolist())
            self.assertEqual(out[2, :100].tolist(), want)


class ExactTopkTest(unittest.TestCase):
    def run_case(self, rows, columns, block_topk, seed, width=None):
        fn = exact_topk()
        g = torch.Generator().manual_seed(seed)
        width = width or columns
        logits = torch.randn(rows, width, generator=g)
        visible = torch.randint(0, columns + 1, (rows,), generator=g, dtype=torch.int32)
        visible[0], visible[-1] = columns, min(3, columns)
        # garbage past each row's visible count, larger than any real logit
        cols = torch.arange(width)
        logits = torch.where(cols[None, :] >= visible[:, None], torch.full_like(logits, 1e9), logits)
        out = torch.full((rows + 2, block_topk), -7, dtype=torch.int32)   # a guard row before and after
        fn(logits, visible, out[1:-1], block_topk, columns)
        again = torch.full_like(out, -7)
        fn(logits, visible, again[1:-1], block_topk, columns)
        self.assertTrue(torch.equal(out, again))
        self.assertTrue(bool((out[0] == -7).all()) and bool((out[-1] == -7).all()))
        for r in range(rows):
            v = int(visible[r])
            complete = min(v, block_topk)
            want = sorted(torch.topk(logits[r, :v], complete).indices.tolist()) if complete else []
            got = out[r + 1, :complete].tolist()
            self.assertEqual(got, want, f"row {r}, visible {v}")
            if columns < block_topk:
                self.assertTrue(bool((out[r + 1, columns:] == -1).all()))

    def test_prefill_shapes(self):
        self.run_case(rows=40, columns=256, block_topk=64, seed=1)     # visible spans 0..256
        self.run_case(rows=33, columns=192, block_topk=64, seed=2, width=256)  # a wider logits buffer

    def test_fewer_columns_than_topk(self):
        self.run_case(rows=10, columns=40, block_topk=64, seed=3)

    def test_decode_like(self):
        self.run_case(rows=5, columns=1024, block_topk=512, seed=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
