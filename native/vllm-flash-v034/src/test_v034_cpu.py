#!/usr/bin/env python3
"""CPU checks for vllm-flash-v034's build and view tools (standard library only):

    python3 src/test_v034_cpu.py

1. patch_lm_head_quant.py on vLLM 0.34.0's own qwen4_exp/nvidia/model.py and mtp.py
   (src/testdata/*.v034, from the vllm-0.34.0 wheel): one quant_config line in each head,
   the same indentation, idempotent; a file without the head stops it.
2. the Dockerfile's mtp.py steps in order: the lm_head patch, then ../vllm-flash-ar's
   draft-head patch on Qwen4ExpMTP; the line the Dockerfile greps for is there. Its
   hyperconnection.py step: ../vllm-flash-ar's FP8 HC hook on 0.34's file, and every
   string the Dockerfile greps for in it. The ablation step, on model.py and mtp.py as the
   steps before leave them: 0.34's insertion points, the strings the Dockerfile greps for.
   The determinism step: ../vllm-flash-ar's MoE alignment patch and the exact QSA top-k on
   0.34's moe_align_block_size.py and qsa_indexer.py, the strings the Dockerfile greps for.
3. the tool-call parser patches (src/patches/*.v034.patch): both apply to vLLM 0.34.0's own
   parser files (src/testdata/parser/, from the wheel) with --fuzz=0, in order, and the
   result parses and switches both guards on for the qwen3 configuration only. (Their
   behaviour: check_tool_parser.py at build time; vLLM 0.34.0's parser tests plus blazux's,
   224 of 224, were run against the patched files when they were ported.)
4. make_model_view.py: links to the container mounts, the table's tensors in the merged
   index, ple_embedding_dtype in config.json, a non-lean table refused, a rerun replaces the
   view, a foreign file in the view directory refused.
"""
import ast
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import typing
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_model_view as mv
import patch_lm_head_quant as lh
import patch_qsa_exact_topk as pq

sys.path.insert(0, os.path.join(HERE, "..", "..", "vllm-flash-ar", "src"))
import patch_abliterate as pa
import patch_det_moe_align as pd
import patch_hc_fp8 as ph
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


def bre(pattern):
    """grep's basic regular expression as Python's: ( ) { } + ? | are literal there."""
    return re.sub(r"([(){}+?|])", r"\\\1", pattern)


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

    def test_dockerfile_ablation_step(self):
        m, t = os.path.join(self.d, "model.py"), os.path.join(self.d, "mtp.py")
        shutil.copy(os.path.join(HERE, "testdata", "model.py.v034"), m)
        shutil.copy(os.path.join(HERE, "testdata", "mtp.py.v034"), t)
        lh.patch(m)
        lh.patch(t)
        dv.patch(t)
        with open(m) as f:
            model_edits, model_hook, mtp_edits = pa.engine_edits(f.read())
        self.assertIs(model_edits, pa.V034_MODEL_EDITS)
        self.assertTrue(pa.patch(m, model_edits, model_hook).startswith("INSTALLED"))
        self.assertTrue(pa.patch(t, mtp_edits, pa.MTP_HOOK).startswith("INSTALLED"))
        with open(os.path.join(HERE, "..", "Dockerfile")) as f:
            docker = f.read()
        for name, path in (("model.py", m), ("mtp.py", t)):
            with open(path) as f:
                src = f.read()
            ast.parse(src)
            for g in re.findall(rf'grep -q "([^"]+)" "\$Q/{re.escape(name)}"', docker):
                self.assertTrue(re.search(bre(g), src, re.MULTILINE), (name, g))
        self.assertIn(lh.MARK, src)
        self.assertIn(dv.MARK, src)

    def test_dockerfile_determinism_step(self):
        with open(os.path.join(HERE, "..", "Dockerfile")) as f:
            docker = f.read()
        with open(os.path.join(HERE, "testdata", "moe_align_block_size.py.v034")) as f:
            moe = pd.patch(f.read())
        ast.parse(moe)
        self.assertEqual(pd.patch(moe), moe)
        for g in re.findall(r'grep -q "([^"]+)" "\$SP/vllm/model_executor/layers/fused_moe/moe_align_block_size.py"',
                            docker):
            self.assertTrue(re.search(bre(g), moe, re.MULTILINE), g)
        q = os.path.join(self.d, "qsa_indexer.py")
        shutil.copy(os.path.join(HERE, "testdata", "qsa_indexer.py.v034"), q)
        self.assertEqual(pq.patch(q), "INSTALLED")
        with open(q) as f:
            qsa = f.read()
        ast.parse(qsa)
        greps = re.findall(r'grep -q "([^"]+)" "\$Q/ops/qsa_indexer.py"', docker)
        self.assertEqual(len(greps), 1)
        self.assertTrue(re.search(bre(greps[0]), qsa, re.MULTILINE))

    def test_dockerfile_hc_step(self):
        p = os.path.join(self.d, "hyperconnection.py")
        shutil.copy(os.path.join(HERE, "testdata", "hyperconnection.py.v034"), p)
        self.assertEqual(ph.patch(p), "INSTALLED")
        with open(p) as f:
            src = f.read()
        ast.parse(src)
        with open(os.path.join(HERE, "..", "Dockerfile")) as f:
            greps = re.findall(r'grep -q "([^"]+)" "\$Q/hyperconnection.py"', f.read())
        self.assertEqual(len(greps), 2)
        for g in greps:
            self.assertIn(g, src)
        self.assertEqual(ph.patch(p), "already installed")


class ToolParserPatchTest(Tmp):
    FILES: typing.ClassVar[dict[str, str]] = {"engine/parser_engine_config.py": "parser_engine_config.py.v034",
             "engine/streaming_parser_engine.py": "streaming_parser_engine.py.v034",
             "qwen3.py": "qwen3.py.v034", "nemotron_v3.py": "nemotron_v3.py.v034"}

    @unittest.skipUnless(shutil.which("patch"), "needs the patch tool")
    def test_patches_apply_to_v034(self):
        root = os.path.join(self.d, "vllm", "parser")
        for rel, src in self.FILES.items():
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            shutil.copy(os.path.join(HERE, "testdata", "parser", src), os.path.join(root, rel))
        for name in ("qwen-tool-preamble.v034.patch", "qwen-tool-marker-guard.v034.patch"):
            with open(os.path.join(HERE, "patches", name)) as f:
                r = subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1"], cwd=self.d,
                                   stdin=f, capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("offset", r.stdout)
        srcs = {}
        for rel in self.FILES:
            with open(os.path.join(root, rel)) as f:
                srcs[rel] = f.read()
            ast.parse(srcs[rel])
        self.assertIn("guard_literal_tool_markers: bool = False", srcs["engine/parser_engine_config.py"])
        self.assertIn("validate_tool_preamble: bool = False", srcs["engine/parser_engine_config.py"])
        self.assertIn('validate_tool_preamble=name == "qwen3",\n        guard_literal_tool_markers=name == "qwen3",',
                      srcs["qwen3.py"])
        self.assertIn("validate_tool_preamble=False,\n        guard_literal_tool_markers=False,", srcs["nemotron_v3.py"])
        engine = srcs["engine/streaming_parser_engine.py"]
        for name in ("def _confirm_tool_preamble", "def _flush_tool_preamble", "def _marker_is_literal",
                     "def _track_literal_text", "self._reset_marker_guard()", "confirmed_tool_start: bool = False"):
            self.assertIn(name, engine)


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
