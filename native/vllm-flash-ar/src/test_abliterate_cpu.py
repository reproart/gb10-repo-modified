#!/usr/bin/env python3
"""CPU checks for the refusal-direction ablation (src/patch_abliterate.py + src/gb10_abliterate.py):
torch and numpy, no GPU, no vLLM.

    python3 src/test_abliterate_cpu.py
    docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_abliterate_cpu.py

src/testdata/{model,mtp}.py.image are the image's own models/qwen3_8_flash_next/nvidia/model.py
and mtp.py as the Dockerfile leaves them before this patch (int8 lm_head sed, draft-head hook).
1. The patch inserts its six projections there, once each, parses, is idempotent, and stops on
   a file it was not written for.
2. Loading: Mambavtt's layout (a [n, hidden] matrix + "layers"), one array per layer, .npy and
   .pt; the mean / one layer / top-k bases are orthonormal; bad input fails with a message.
3. project() removes alpha times each stream's component in the basis span.
4. The patched decoder layer, model and MTP draft predictor, exec'd from those files with
   small CPU stand-ins for vLLM's pieces (the hyper-connection mix / combine as vLLM's
   reference implementation computes them: scalar injection weight per stream, elementwise
   gate on the read side): with VLLM_ABLATE every stream of the final multi-stream state, and
   the draft layer's input, has no component left along the direction; with some layers left
   out it has; without VLLM_ABLATE the outputs equal the unpatched files' bit for bit.
"""
import ast
import builtins
import contextlib
import itertools
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gb10_abliterate as ga  # noqa: E402
import patch_abliterate as pa  # noqa: E402

MODEL_IMG = os.path.join(HERE, "testdata", "model.py.image")
MTP_IMG = os.path.join(HERE, "testdata", "mtp.py.image")
H, NS, LAYERS, T = 16, 4, 4, 6          # hidden, HC streams, decoder layers, tokens
ENV_KEYS = ("VLLM_ABLATE", "VLLM_ABLATE_ALPHA", "VLLM_ABLATE_SOURCE", "VLLM_ABLATE_LAYERS", "VLLM_ABLATE_MTP")


# --------------------------------------------------------------------------- stand-ins
class FakeGatedResidual(nn.Module):
    """vLLM's GatedResidual (common/hyperconnection.py, the reference the NVIDIA kernels
    implement): per-stream RMSNorm with a per-element weight, a low-rank sigmoid gate per
    element, the gated mean over streams as the block input (read); the block output added
    to each stream with 2 sigmoid(injection / n), one scalar per stream (write)."""

    def __init__(self, cfg, use_combine=True, prefix=""):
        super().__init__()
        n, h = cfg.hc_count, cfg.hidden_size
        self.n, self.h = n, h
        self.norm_w = nn.Parameter(torch.randn(n * h) * 0.2)
        self.down = nn.Linear(n * h, cfg.hc_lowrank, bias=False)
        self.up = nn.Linear(cfg.hc_lowrank, n * h, bias=False)
        self.inject = nn.Linear(n * h, n, bias=False) if use_combine else None

    def _norm(self, x):
        xs = x.unflatten(-1, (self.n, self.h))
        xs = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + 1e-6)
        return xs.flatten(-2) * (1.0 + self.norm_w)

    def mix(self, hidden):
        xn = self._norm(hidden)
        gate = torch.sigmoid(self.up(F.silu(self.down(xn) / self.n))).unflatten(-1, (self.n, self.h))
        block_input = (gate * xn.unflatten(-1, (self.n, self.h))).mean(-2)
        return hidden, block_input, (self.inject(xn) if self.inject is not None else None)

    def combine(self, hidden, block_output, injection):
        w = 2.0 * torch.sigmoid(injection / self.n)
        return (hidden.unflatten(-1, (self.n, self.h)) + block_output.unsqueeze(-2) * w.unsqueeze(-1)).flatten(-2)

    def combine_and_mix(self, hidden, block_output, injection):
        return self.mix(self.combine(hidden, block_output, injection))


class Block(nn.Module):
    """attention / GDN / MLP stand-in: a dense map with a bias, so its output has a part
    along any direction."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lin = nn.Linear(H, H)

    def forward(self, hidden_states, positions=None):
        return torch.tanh(self.lin(hidden_states)) * 3


class FakePLE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.lin = nn.Linear(NS * H, NS * H)

    def forward(self, hidden_states, input_ids, query_start_loc, ngram_context):
        return self.lin(hidden_states)


class FakeRMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.randn(size) * 0.2)

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + self.w)


def fakes():
    def make_layers(n, get_layer, prefix):
        return 0, n, nn.ModuleList(get_layer(f"{prefix}.{i}") for i in range(n))

    class Qwen3_5Model:
        hf_to_vllm_mapper = frozenset()

    return {
        "torch": torch, "nn": nn, "islice": itertools.islice, "Iterable": Iterable, "re": re,
        "support_torch_compile": lambda **kw: (lambda cls: cls),
        "get_pp_group": lambda: types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
        "extract_layer_index": lambda prefix: int(re.findall(r"\d+", prefix)[-1]),
        "make_layers": make_layers,
        "make_empty_intermediate_tensors_factory": lambda keys, size: (lambda *a, **k: None),
        "maybe_prefix": lambda prefix, name: f"{prefix}.{name}" if prefix else name,
        "VocabParallelEmbedding": lambda vocab, hidden: nn.Embedding(vocab, hidden),
        "QwenGatedDeltaNetAttention": Block, "Qwen3NextAttention": Block, "Qwen3NextMLP": Block,
        "Qwen3_8FlashNextPLELayer": FakePLE,
        "HyperConnectionConfig": lambda **kw: types.SimpleNamespace(**kw),
        "GatedResidual": FakeGatedResidual,
        "WeightsMapper": lambda **kw: frozenset(),
        "Qwen3_5Model": Qwen3_5Model,
        # mtp.py
        "get_draft_quant_config": lambda cfg: None,
        "replace": lambda cfg, **kw: types.SimpleNamespace(**{**vars(cfg), **kw}),
        "set_current_vllm_config": lambda cfg, prefix="": contextlib.nullcontext(),
        "ColumnParallelLinear": lambda i, o, **kw: nn.Linear(i, o, bias=False),
        "GemmaRMSNorm": FakeRMSNorm,
    }


def load(text: str, keep_classes, extra=None) -> dict:
    """Exec a model file's functions, its constants before the first class, the classes named
    and everything from this patch's hook on, with fakes for the names vLLM would provide."""
    tree = ast.parse(text)
    lines = text.splitlines()
    hook_line = next((i + 1 for i, s in enumerate(lines) if pa.MARK in s and s.startswith("#")), 10 ** 9)
    first_class = min((n.lineno for n in tree.body if isinstance(n, ast.ClassDef)), default=10 ** 9)
    body = []
    for node in tree.body:
        if node.lineno > hook_line:
            body.append(node)
        elif isinstance(node, ast.FunctionDef):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in keep_classes:
            body.append(node)
        elif isinstance(node, ast.Assign) and node.lineno < first_class:
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ns = {"__name__": "fake_model", **fakes(), **(extra or {})}
    for name in {n.id for n in ast.walk(module) if isinstance(n, ast.Name)}:
        if name not in ns and not hasattr(builtins, name):
            ns[name] = type(name, (), {})           # annotations and unused bases
    exec(compile(module, "<patched model>", "exec"), ns)
    return ns


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def patched_texts():
    d = tempfile.mkdtemp()
    try:
        m, t = os.path.join(d, "model.py"), os.path.join(d, "mtp.py")
        shutil.copy(MODEL_IMG, m)
        shutil.copy(MTP_IMG, t)
        pa.patch(m, pa.MODEL_EDITS, pa.MODEL_HOOK)
        pa.patch(t, pa.MTP_EDITS, pa.MTP_HOOK)
        return read(m), read(t)
    finally:
        shutil.rmtree(d)


def text_config():
    return types.SimpleNamespace(
        hidden_size=H, hc_count=NS, hc_lowrank=8, rms_norm_eps=1e-6, num_hidden_layers=LAYERS,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        ple_layer_ids=[2], mlp_only_layers=[], num_experts=0, decoder_sparse_step=1,
        intermediate_size=2 * H, hidden_act="silu", vocab_size=50, indexer_n_heads=None,
        shared_expert_intermediate_size=0, mtp_num_hidden_layers=1, tie_word_embeddings=False)


def vllm_config(cfg):
    mc = types.SimpleNamespace(hf_text_config=cfg, dtype=torch.float32)
    return types.SimpleNamespace(
        model_config=mc, cache_config=types.SimpleNamespace(), quant_config=None,
        parallel_config=types.SimpleNamespace(use_sequence_parallel_moe=False,
                                              eplb_config=types.SimpleNamespace(num_redundant_experts=0)),
        speculative_config=types.SimpleNamespace(method="mtp", draft_model_config=mc),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=64))


def directions(seed=0, n=LAYERS + 2):
    """Correlated unit vectors, as refusal directions of neighbouring layers are."""
    g = np.random.default_rng(seed)
    base = g.normal(size=H)
    V = np.stack([base + 0.6 * g.normal(size=H) for _ in range(n)])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(self.restore)

    def restore(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def npz(self, V, layers=None, name="v.npz"):
        path = os.path.join(self.tmp, name)
        np.savez(path, layers=np.asarray(layers if layers is not None else range(len(V)), dtype=np.int32),
                 directions=V.astype(np.float32))
        return path


# --------------------------------------------------------------------------- 1. the patch
class PatchTest(unittest.TestCase):
    def test_inserts_once_and_is_idempotent(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        m, t = os.path.join(d, "model.py"), os.path.join(d, "mtp.py")
        shutil.copy(MODEL_IMG, m)
        shutil.copy(MTP_IMG, t)
        self.assertTrue(pa.patch(m, pa.MODEL_EDITS, pa.MODEL_HOOK).startswith("INSTALLED"))
        self.assertTrue(pa.patch(t, pa.MTP_EDITS, pa.MTP_HOOK).startswith("INSTALLED"))
        model, mtp = read(m), read(t)
        self.assertEqual(model.count("_gb10_abl_project("), 5)        # the five sites
        self.assertEqual(mtp.count("_gb10_abl_project("), 1)
        self.assertEqual(model.count("if self._gb10_abl_on:"), 4)
        self.assertIn("MTP private draft head", mtp)                   # the earlier hook stays
        self.assertEqual(pa.patch(m, pa.MODEL_EDITS, pa.MODEL_HOOK), "already installed")
        self.assertEqual(read(m), model)

    def test_refuses_another_file(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "model.py")
        write(p, read(MODEL_IMG).replace("mlp_out = self.mlp(block_input)", "mlp_out = self.mlp(x)"))
        with self.assertRaises(SystemExit) as cm:
            pa.patch(p, pa.MODEL_EDITS, pa.MODEL_HOOK)
        self.assertIn("0 matches", str(cm.exception))
        self.assertNotIn(pa.MARK, read(p))


# --------------------------------------------------------------------------- 2. loading
class LoadTest(EnvCase):
    def test_settings(self):
        self.assertIsNone(ga.settings({}))
        s = ga.settings({"VLLM_ABLATE": "/a/v.npz", "VLLM_ABLATE_LAYERS": "4-6, 9", "VLLM_ABLATE_MTP": "0",
                         "VLLM_ABLATE_ALPHA": "0.5", "VLLM_ABLATE_SOURCE": "TOP2"})
        self.assertEqual((s["layers"], s["mtp"], s["alpha"], s["source"]), ({4, 5, 6, 9}, False, 0.5, "top2"))
        self.assertIsNone(ga.settings({"VLLM_ABLATE": "x"})["layers"])
        for bad in ({"VLLM_ABLATE_LAYERS": "5-3"}, {"VLLM_ABLATE_LAYERS": "a"}, {"VLLM_ABLATE_MTP": "yes"},
                    {"VLLM_ABLATE_ALPHA": "one"}):
            with self.assertRaises(RuntimeError):
                ga.settings({"VLLM_ABLATE": "x", **bad})

    def test_layouts_and_sources(self):
        V = directions(1, 6)
        layers = [4, 5, 6, 7, 8, 9]
        mambavtt = self.npz(V * 3.0, layers)                       # unnormalized on purpose
        keyed = os.path.join(self.tmp, "k.npz")
        np.savez(keyed, **{str(i): v for i, v in zip(layers, V)}, note=np.zeros(2))
        for path in (mambavtt, keyed):
            B, what, cov = ga.basis(path, "mean", H)
            m = V.mean(0)
            self.assertEqual(B.shape, (1, H))
            self.assertAlmostEqual(abs(float(B[0] @ m / np.linalg.norm(m))), 1.0, places=10)
            self.assertEqual(sorted(cov), layers)
            self.assertIn("mean of layers 4-9", what)
        B, _, cov = ga.basis(mambavtt, "6", H)
        self.assertAlmostEqual(abs(float(B[0] @ V[2])), 1.0, places=10)
        self.assertAlmostEqual(cov[6], 1.0, places=10)
        B, what, cov = ga.basis(mambavtt, "top3", H)
        np.testing.assert_allclose(B @ B.T, np.eye(3), atol=1e-12)
        self.assertTrue(all(0 <= c <= 1 + 1e-12 for c in cov.values()))
        self.assertGreater(min(cov.values()), min(ga.basis(mambavtt, "mean", H)[2].values()))
        one = os.path.join(self.tmp, "one.npy")
        np.save(one, V[0] * 7)
        B, what, cov = ga.basis(one, "mean", H)
        self.assertAlmostEqual(abs(float(B[0] @ V[0])), 1.0, places=10)
        self.assertEqual(cov, {})
        pt = os.path.join(self.tmp, "d.pt")
        torch.save({5: torch.tensor(V[1]), 6: torch.tensor(V[2])}, pt)
        self.assertAlmostEqual(abs(float(ga.basis(pt, "5", H)[0][0] @ V[1])), 1.0, places=6)

    def test_bad_input(self):
        V = directions(2, 3)
        path = self.npz(V)
        for source, msg in (("7", "no such layer"), ("top9", "mean, a layer index"), ("first", "mean, a layer")):
            with self.assertRaisesRegex(RuntimeError, msg):
                ga.basis(path, source, H)
        with self.assertRaisesRegex(RuntimeError, "hidden size is 32"):
            ga.basis(path, "mean", 32)
        with self.assertRaisesRegex(RuntimeError, "no such file"):
            ga.basis(os.path.join(self.tmp, "missing.npz"), "mean", H)
        zero = self.npz(np.zeros((2, H)), name="z.npz")
        with self.assertRaisesRegex(RuntimeError, "zero"):
            ga.basis(zero, "mean", H)


# --------------------------------------------------------------------------- 3. the projection
class ProjectTest(unittest.TestCase):
    def test_streams_lose_the_component(self):
        g = torch.Generator().manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(H, 2, generator=g, dtype=torch.float64))
        B = Q.T                                                  # [2, H] orthonormal
        x = torch.randn(T, NS * H, generator=g, dtype=torch.float64)
        out = ga.project(x, B)
        self.assertEqual(out.shape, x.shape)
        c = out.view(T, NS, H) @ B.T
        self.assertLess(c.abs().max().item(), 1e-12)
        # outside the span nothing moves
        P = torch.eye(H, dtype=torch.float64) - B.T @ B
        torch.testing.assert_close(out.view(T, NS, H) @ P, x.view(T, NS, H) @ P)
        # alpha: u = sqrt(alpha) B removes alpha of it; a single stream [T, H] too
        y = torch.randn(T, H, generator=g, dtype=torch.float64)
        half = ga.project(y, (0.25 ** 0.5) * B[:1])
        torch.testing.assert_close(half @ B[0], 0.75 * (y @ B[0]))


# --------------------------------------------------------------------------- 4. the model
class ModelTest(EnvCase):
    @classmethod
    def setUpClass(cls):
        cls.patched_model, cls.patched_mtp = patched_texts()
        cls.orig_model = read(MODEL_IMG)

    def build(self, model_text, env=None, mtp_text=None):
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(env or {})
        keep = ("Qwen3_8FlashNextDecoderLayer", "Qwen3_8FlashNextModel", "Qwen3_8FlashNextSparseMoeBlock")
        ns = load(model_text, keep)
        cfg = text_config()
        torch.manual_seed(1234)
        model = ns["Qwen3_8FlashNextModel"](vllm_config=vllm_config(cfg), prefix="model")
        mtp = None
        if mtp_text is not None:
            tns = load(mtp_text, ("Qwen3_8FlashNextMultiTokenPredictor",),
                       {"Qwen3_8FlashNextDecoderLayer": ns["Qwen3_8FlashNextDecoderLayer"],
                        "_HC_WEIGHTS_MAPPER": frozenset(), "_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES": []})
            torch.manual_seed(99)
            mtp = tns["Qwen3_8FlashNextMultiTokenPredictor"](vllm_config=vllm_config(cfg), prefix="mtp")
        return model, mtp

    def run_model(self, model):
        g = torch.Generator().manual_seed(7)
        ids = torch.randint(0, 50, (T,), generator=g)
        with torch.no_grad():
            sample = model(ids, torch.arange(T), query_start_loc=torch.tensor([0, T]),
                           ngram_context=torch.zeros(T, 2), deepstack_input_embeds={
                               "deepstack_input_embeds_0": torch.randn(T, H, generator=g)})
        return ids, sample, model._mtp_hidden_buffer[:T].clone()

    @staticmethod
    def along(state, u):
        """largest |component| along u's rows of any stream, relative to the stream norms"""
        xs = state.view(-1, NS, H)
        return ((xs @ u.t()).abs().max() / xs.norm(dim=-1).max()).item()

    def env(self, V, **kw):
        return {"VLLM_ABLATE": self.npz(V), **{f"VLLM_ABLATE_{k.upper()}": str(v) for k, v in kw.items()}}

    def test_off_is_bit_identical(self):
        orig, _ = self.build(self.orig_model)
        off, _ = self.build(self.patched_model)
        self.assertFalse(any(getattr(m, "_gb10_abl_on", False) for m in off.modules()))
        self.assertFalse(any(n.endswith("_gb10_abl_u") for n, _ in off.named_buffers()))
        _, s0, m0 = self.run_model(orig)
        _, s1, m1 = self.run_model(off)
        self.assertTrue(torch.equal(s0, s1) and torch.equal(m0, m1))

    def test_every_stream_loses_the_direction(self):
        V = directions(3)
        stock, _ = self.build(self.patched_model)
        _, _, m_stock = self.run_model(stock)
        with self.assertLogs("vllm.gb10_abliterate", "WARNING") as logs:
            model, _ = self.build(self.patched_model, self.env(V))
        self.assertTrue(any("ABLATION ON" in r and f"{LAYERS} of {LAYERS} layers" in r and "the embedding" in r
                            for r in logs.output), logs.output)
        on = [m for m in model.modules() if getattr(m, "_gb10_abl_on", False)]
        self.assertEqual(len(on), LAYERS + 1)                   # the layers + the embedding
        self.assertEqual(len({id(m._gb10_abl_u) for m in on}), 1)  # one shared tensor
        u = on[0]._gb10_abl_u
        _, _, multi = self.run_model(model)
        self.assertGreater(self.along(m_stock, u), 1e-2)
        self.assertLess(self.along(multi, u), 1e-5)

    def test_rank_and_alpha(self):
        V = directions(4)
        model, _ = self.build(self.patched_model, self.env(V, source="top2"))
        u = model.layers[0]._gb10_abl_u
        self.assertEqual(tuple(u.shape), (2, H))
        _, _, multi = self.run_model(model)
        self.assertLess(self.along(multi, u), 1e-5)
        with self.assertLogs("vllm.gb10_abliterate", "WARNING") as logs:
            stock, _ = self.build(self.patched_model, self.env(V, alpha=0))
        self.assertTrue(any("ALPHA=0" in r for r in logs.output))
        self.assertFalse(any(getattr(m, "_gb10_abl_on", False) for m in stock.modules()))

    def test_layers_left_out_keep_it(self):
        V = directions(5)
        model, _ = self.build(self.patched_model, self.env(V, layers="2-3"))
        self.assertEqual([bool(layer._gb10_abl_on) for layer in model.layers], [False, False, True, True])
        self.assertFalse(model._gb10_abl_on)                    # the embedding goes with layer 0
        u = model.layers[2]._gb10_abl_u
        _, _, multi = self.run_model(model)
        self.assertGreater(self.along(multi, u), 1e-3)

    def test_mtp_draft(self):
        V = directions(6)
        captured = {}

        def grab(mod, args, kwargs):
            captured["input"] = kwargs["hidden_states"].detach().clone()

        for flag, want_on in (("1", True), ("0", False)):
            model, mtp = self.build(self.patched_model, self.env(V, mtp=flag), self.patched_mtp)
            layer = mtp.layers[0]
            self.assertEqual(bool(layer._gb10_abl_on), want_on)
            u = model.layers[0]._gb10_abl_u
            ids, _, multi = self.run_model(model)
            handle = layer.register_forward_pre_hook(grab, with_kwargs=True)
            with torch.no_grad():
                _, draft_multi = mtp(ids, torch.arange(T), hidden_states=multi)
            handle.remove()
            if want_on:
                self.assertLess(self.along(captured["input"], u), 1e-5)
                self.assertLess(self.along(draft_multi, u), 1e-5)
            else:
                self.assertGreater(self.along(captured["input"], u), 1e-3)

    def test_compiles_without_graph_breaks(self):
        """vLLM compiles the model's forward with torch.compile(fullgraph=True): the inserted
        flag checks and projections must trace as one graph (aot_eager: no C compiler needed)."""
        V = directions(8)
        model, _ = self.build(self.patched_model, self.env(V))
        ids, eager, _ = self.run_model(model)
        g = torch.Generator().manual_seed(7)
        ids2 = torch.randint(0, 50, (T,), generator=g)
        deepstack = {"deepstack_input_embeds_0": torch.randn(T, H, generator=g)}
        compiled = torch.compile(model.forward, fullgraph=True, backend="aot_eager", dynamic=True)
        with torch.no_grad():
            out = compiled(ids2, torch.arange(T), query_start_loc=torch.tensor([0, T]),
                           ngram_context=torch.zeros(T, 2), deepstack_input_embeds=deepstack)
        torch.testing.assert_close(out, eager)

    def test_mtp_off_is_bit_identical(self):
        orig_mtp_text = read(MTP_IMG)
        _, m0 = self.build(self.orig_model, mtp_text=orig_mtp_text)
        _, m1 = self.build(self.patched_model, mtp_text=self.patched_mtp)
        g = torch.Generator().manual_seed(3)
        ids, hid = torch.randint(0, 50, (T,), generator=g), torch.randn(T, NS * H, generator=g)
        with torch.no_grad():
            a, b = m0(ids, torch.arange(T), hidden_states=hid), m1(ids, torch.arange(T), hidden_states=hid)
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
