#!/usr/bin/env python3
"""CPU checks for patch_qsa_exact_topk.py (torch; no GPU, no vLLM):

    python3 src/test_qsa_exact_topk_cpu.py

1. The patch on vLLM 0.34.0's own qsa_indexer.py (src/testdata, from the wheel): the exact
   path goes in front of the persistent_topk call in _topk, once; the file parses; a second
   run changes nothing; a file without that call is refused.
2. The exact top-k itself, against a reference: for every row the first
   min(visible, block_topk) slots (all that 0.34's expansion kernel reads) hold exactly the
   ids of the row's highest logits among its visible columns, in ascending order; the garbage
   past a row's visible count never gets in (it is masked to -inf); fewer columns than
   block_topk leaves -1 after them; two runs give the same ids; nothing outside the rows'
   slots is written.
"""
import os
import shutil
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import patch_qsa_exact_topk as pq

V034 = os.path.join(HERE, "testdata", "qsa_indexer.py.v034")


def exact_topk():
    os.environ["VLLM_QSA_EXACT_TOPK"] = "1"
    ns = {"torch": torch}
    exec(compile(pq.HOOK, "hook", "exec"), ns)  # noqa: S102 - the hook as the patch appends it
    os.environ.pop("VLLM_QSA_EXACT_TOPK")
    assert ns["_GB10_QSA_EXACT_TOPK"] is True
    return ns["_gb10_qsa_exact_topk"]


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
            self.assertEqual(src.count(pq.MARK), 2)
            with open(p, "w") as f:
                f.write("import torch\n")
            with self.assertRaises(SystemExit):
                pq.patch(p)

    def test_off_by_default(self):
        ns = {"torch": torch}
        os.environ.pop("VLLM_QSA_EXACT_TOPK", None)
        exec(compile(pq.HOOK, "hook", "exec"), ns)  # noqa: S102 - the hook as the patch appends it
        self.assertFalse(ns["_GB10_QSA_EXACT_TOPK"])


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
