#!/usr/bin/env python3
"""CPU checks for vllm-flash-v034's build and view tools (standard library only):

    python3 src/test_v034_cpu.py

1. patch_lm_head_quant.py on vLLM 0.34.0's own qwen4_exp/nvidia/model.py and mtp.py
   (src/testdata/*.v034, from the vllm-0.34.0 wheel): one quant_config line in each head,
   the same indentation, idempotent; a file without the head stops it.
2. the Dockerfile's mtp.py steps in order: the lm_head patch, then ../vllm-flash-ar's
   draft-head patch on Qwen4ExpMTP; the line the Dockerfile greps for is there.
3. make_model_view.py: links to the container mounts, the table's tensors in the merged
   index, ple_embedding_dtype in config.json, a non-lean table refused, a rerun replaces the
   view, a foreign file in the view directory refused.
"""
import ast
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_model_view as mv
import patch_lm_head_quant as lh

sys.path.insert(0, os.path.join(HERE, "..", "..", "vllm-flash-ar", "src"))
import patch_mtp_draft_vocab as dv


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape)}; zero bytes of the right size."""
    sizes = {"F8_E4M3": 1, "BF16": 2, "I32": 4, "I64": 8}
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for s in shape:
            n *= s
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + bytes(off))


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)


class LmHeadTest(Tmp):
    def test_v034_files(self):
        for name in ("model.py.v034", "mtp.py.v034"):
            p = os.path.join(self.d, name)
            shutil.copy(os.path.join(HERE, "testdata", name), p)
            self.assertEqual(lh.patch(p), "INSTALLED")
            with open(p) as f:
                lines = f.read().splitlines()
            i = next(k for k, line in enumerate(lines) if lh.MARK in line)
            self.assertEqual(lines[i].strip().split("  #")[0], "quant_config=vllm_config.quant_config,")
            self.assertEqual(lines[i + 1].strip(), lh.ANCHOR)
            self.assertEqual(lines[i][: len(lines[i]) - len(lines[i].lstrip())],
                             lines[i + 1][: len(lines[i + 1]) - len(lines[i + 1].lstrip())])
            self.assertEqual(lh.patch(p), "already installed")

    def test_refuses_other_file(self):
        p = os.path.join(self.d, "x.py")
        with open(p, "w") as f:
            f.write("x = 1\n")
        with self.assertRaises(SystemExit):
            lh.patch(p)


class DraftHeadStepTest(Tmp):
    def test_dockerfile_mtp_steps(self):
        p = os.path.join(self.d, "mtp.py")
        shutil.copy(os.path.join(HERE, "testdata", "mtp.py.v034"), p)
        self.assertEqual(lh.patch(p), "INSTALLED")
        self.assertEqual(dv.patch(p), "INSTALLED on Qwen4ExpMTP")
        with open(p) as f:
            src = f.read()
        ast.parse(src)
        self.assertIn(lh.MARK, src)
        with open(os.path.join(HERE, "..", "Dockerfile")) as f:
            grep = re.search(r'grep -q "(\^    Qwen4ExpMTP[^"]+)" "\$Q/mtp.py"', f.read()).group(1)
        self.assertTrue(re.search(grep, src, re.MULTILINE))
        self.assertEqual(dv.patch(p), "already installed")


class ViewTest(Tmp):
    def make(self, lean=True):
        model, table = os.path.join(self.d, "model"), os.path.join(self.d, "table")
        os.makedirs(model)
        os.makedirs(table)
        with open(os.path.join(model, "config.json"), "w") as f:
            json.dump({"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}, f)
        write_safetensors(os.path.join(model, "model-00001-of-00001.safetensors"),
                          {"model.language_model.embed_tokens.weight": ("BF16", (4, 8)),
                           "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": ("I64", (3,))})
        with open(os.path.join(model, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": 1}, "weight_map": {
                "model.language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers":
                    "model-00001-of-00001.safetensors"}}, f)
        with open(os.path.join(model, "tokenizer.json"), "w") as f:
            f.write("{}")
        t = {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": ("F8_E4M3", (10, 4)),
             "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale": ("BF16", (1,))}
        if not lean:
            t["model.language_model.layers.1.mlp.experts.0.down_proj.weight_scale_inv"] = ("BF16", (2, 2))
        write_safetensors(os.path.join(table, "model-00005-of-00131.safetensors"), t)
        return model, table, os.path.join(self.d, "view")

    def test_view(self):
        model, table, view = self.make()
        self.assertEqual(mv.build(model, table, view), (4, 1, 2))
        self.assertEqual(os.readlink(os.path.join(view, "model-00001-of-00001.safetensors")),
                         "/model/model-00001-of-00001.safetensors")
        self.assertEqual(os.readlink(os.path.join(view, "model-00005-of-00131.safetensors")),
                         "/ple-table/model-00005-of-00131.safetensors")
        self.assertEqual(os.readlink(os.path.join(view, "tokenizer.json")), "/model/tokenizer.json")
        self.assertFalse(os.path.islink(os.path.join(view, "config.json")))
        with open(os.path.join(view, "config.json")) as f:
            self.assertEqual(json.load(f)["text_config"]["ple_embedding_dtype"], "float8_e4m3fn")
        with open(os.path.join(view, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        self.assertEqual(wm["model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"],
                         "model-00005-of-00131.safetensors")
        self.assertEqual(len(wm), 4)
        with open(os.path.join(model, "config.json")) as f:     # the model's own config is untouched
            self.assertNotIn("ple_embedding_dtype", json.load(f)["text_config"])
        self.assertEqual(mv.build(model, table, view), (4, 1, 2))   # a rerun replaces it

    def test_refusals(self):
        model, table, view = self.make(lean=False)
        with self.assertRaisesRegex(SystemExit, "not a lean table"):
            mv.build(model, table, view)
        model2, view2 = os.path.join(self.d, "m2"), os.path.join(self.d, "v2")
        shutil.copytree(model, model2)
        os.makedirs(view2)
        with open(os.path.join(view2, "notes.txt"), "w") as f:
            f.write("mine")
        lean = os.path.join(self.d, "lean")
        os.makedirs(lean)
        write_safetensors(os.path.join(lean, "model-00005-of-00131.safetensors"),
                          {"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight":
                           ("F8_E4M3", (10, 4))})
        with self.assertRaisesRegex(SystemExit, "not part of a view"):
            mv.build(model2, lean, view2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
