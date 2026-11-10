#!/usr/bin/env python3
"""CPU tests for gb10_ablate (torch + numpy, no GPU, no SGLang):

    python3 patches/test_gb10_ablate.py

The projection itself (each stream / the streams' mean, alpha), which layers
get hooks, output vs input, and the loading checks. On the Spark the boot log
has "GB10_ABLATE: N layers, alpha ..." and the answers must stay coherent.

Qwen3.5 (the 27B): a stand-in stack with SGLang 0.5.21's residual stream (a
layer returns its MLP contribution, the batch's stream holds the residual, the
next layer adds them): the post-layer stream loses d, the projection matches
h - alpha d (d . h) on the summed stream, input mode, the refusals (layer 0 in
input mode, an owed contribution); projective GGUF control vectors load, an
additive one is refused. GB10_ABLATE_TEST_GGUF=<file> also loads a real one.
"""

import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

H, K, LAYERS = 16, 4, 8


class Layer(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.seen = None

    def forward(self, positions=None, hidden_states=None, residual=None, **kw):
        self.seen = hidden_states.clone()
        return hidden_states * 1.0, None


class Text(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer() for _ in range(LAYERS)])
        self.start_layer, self.end_layer, self.hc_count, self.hidden_size = 0, LAYERS, K, H

    def forward(self, h):
        for layer in self.layers:
            h, _ = layer(positions=None, hidden_states=h, residual=None)
        return h


class Target(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.model = Text()

    def load_weights(self, weights):
        return "loaded"


@unittest.skipIf(torch is None, "torch/numpy not installed")
class AblateTest(unittest.TestCase):
    def setUp(self):
        import gb10_ablate

        self.f = gb10_ablate
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        g = np.random.default_rng(0)
        self.dirs = {i: g.standard_normal(H).astype(np.float32) for i in (2, 3, 4)}
        self.path = os.path.join(self.tmp.name, "dirs.npz")
        np.savez(self.path, **{str(i): v * 3 for i, v in self.dirs.items()})  # not unit: normalized
        for k in ("GB10_ABLATE", "GB10_ABLATE_ALPHA", "GB10_ABLATE_AT", "GB10_ABLATE_STREAMS"):
            self.addCleanup(os.environ.pop, k, None)
        os.environ["GB10_ABLATE"] = self.path

    def unit(self, i):
        v = torch.from_numpy(self.dirs[i])
        return v / v.norm()

    def run_model(self, **env):
        os.environ.update(env)
        mod = types.ModuleType(self.f.TARGET_MODULE)
        cls = type(self.f.TARGET_CLASS, (Target,), {})
        mod.__dict__[self.f.TARGET_CLASS] = cls
        self.f.apply_target(mod)
        m = cls()
        self.assertEqual(m.load_weights([]), "loaded")
        return m, torch.randn(5, K * H)

    def test_projection_each_stream(self):
        d = self.unit(2)
        h = torch.randn(3, K * H)
        out = self.f.project(h, d, 1.0, H, "each")
        per = out.view(3, K, H) @ d
        self.assertLess(per.abs().max().item(), 1e-5)          # d gone from every stream
        orth = h.view(3, K, H) - (h.view(3, K, H) @ d)[..., None] * d
        self.assertTrue(torch.allclose(out.view(3, K, H), orth, atol=1e-5))
        self.assertTrue(torch.equal(self.f.project(h, d, 0.0, H, "each"), h))

    def test_projection_mean_only(self):
        d = self.unit(2)
        h = torch.randn(3, K * H)
        out = self.f.project(h, d, 1.0, H, "mean").view(3, K, H)
        self.assertLess((out.mean(1) @ d).abs().max().item(), 1e-5)   # the mean loses d
        before = h.view(3, K, H) @ d
        after = out @ d
        diff_before = before - before.mean(1, keepdim=True)
        self.assertTrue(torch.allclose(after, diff_before, atol=1e-5))  # differences stay

    def test_hooks_on_listed_layers_output(self):
        m, x = self.run_model(GB10_ABLATE_ALPHA="1")
        self.assertEqual(len(m._gb10_ablate_hooks), 3)
        y = m.model(x.clone())
        layers = m.model.layers
        # layer 5 sees layer 4's ablated output: no d_4 component in any stream
        self.assertLess((layers[5].seen.view(-1, K, H) @ self.unit(4)).abs().max().item(), 1e-4)
        # layer 2 sees the untouched input (its own output is ablated, not its input)
        self.assertTrue(torch.allclose(layers[2].seen, x))
        self.assertEqual(y.shape, x.shape)

    def test_input_mode(self):
        m, x = self.run_model(GB10_ABLATE_AT="input")
        m.model(x.clone())
        self.assertLess((m.model.layers[2].seen.view(-1, K, H) @ self.unit(2)).abs().max().item(), 1e-4)
        self.assertTrue(torch.allclose(m.model.layers[1].seen, x))

    def test_alpha_zero_installs_nothing(self):
        m, x = self.run_model(GB10_ABLATE_ALPHA="0")
        self.assertEqual(m._gb10_ablate_hooks, [])

    def test_wrong_width_and_layer_fail(self):
        np.savez(self.path, **{"2": np.ones(H + 1, np.float32)})
        with self.assertRaises(RuntimeError):
            self.run_model()
        np.savez(self.path, **{str(LAYERS + 3): np.ones(H, np.float32)})
        with self.assertRaises(RuntimeError):
            self.run_model()

    def test_metadata_keys_are_skipped(self):
        # the published file: per-layer keys plus a "layers" list
        np.savez(self.path, layers=np.array([2, 3, 4]), **{str(i): v for i, v in self.dirs.items()})
        dirs = self.f.load_directions(self.path, H)
        self.assertEqual(sorted(dirs), [2, 3, 4])

    def test_matrix_layout(self):
        mat = np.stack([self.dirs[i] for i in (2, 3, 4)])
        np.savez(self.path, layers=np.array([2, 3, 4]), directions=mat)
        dirs = self.f.load_directions(self.path, H)
        self.assertEqual(sorted(dirs), [2, 3, 4])
        self.assertTrue(np.allclose(dirs[3], self.dirs[3] / np.linalg.norm(self.dirs[3])))

    def test_no_vectors_lists_entries(self):
        np.savez(self.path, layers=np.array([2]), note=np.array([1.0]))
        with self.assertRaisesRegex(RuntimeError, "layers"):
            self.f.load_directions(self.path, H)

    def test_bad_settings(self):
        os.environ["GB10_ABLATE_AT"] = "middle"
        with self.assertRaises(RuntimeError):
            self.f.settings()

    def test_separate_residual_fails_loudly(self):
        m, x = self.run_model()
        m.model.layers[2].forward = lambda **kw: (kw["hidden_states"], torch.zeros(1))
        with self.assertRaises(RuntimeError):
            m.model(x)


def write_gguf(path, meta, tensors):
    """A GGUF v3 file with string / float / bool / int metadata and F32 tensors."""
    import struct

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += s(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + s(v)
        elif isinstance(v, bool):
            out += struct.pack("<I?", 7, v)
        elif isinstance(v, float):
            out += struct.pack("<If", 6, v)
        else:
            out += struct.pack("<Ii", 5, v)
    off, blobs = 0, []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype="<f4")
        out += s(name) + struct.pack("<I", arr.ndim)
        out += b"".join(struct.pack("<Q", d) for d in arr.shape[::-1]) + struct.pack("<IQ", 0, off)
        blobs.append(arr.tobytes())
        off += -(-arr.nbytes // 32) * 32
    out += b"\0" * (-len(out) % 32)
    for b in blobs:
        out += b + b"\0" * (-len(b) % 32)
    with open(path, "wb") as f:
        f.write(out)


class Stream:
    def __init__(self):
        self.residual, self.pending = None, None


class StreamLayer(torch.nn.Module if torch else object):
    """SGLang 0.5.21's Qwen3.5 layer, as the residual stream sees it."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.wa = torch.nn.Parameter(torch.randn(H, H, generator=g) * 0.3, requires_grad=False)
        self.wm = torch.nn.Parameter(torch.randn(H, H, generator=g) * 0.3, requires_grad=False)
        self.seen = None

    def forward(self, positions=None, hidden_states=None, forward_batch=None,
                captured_last_layer_outputs=None, **kw):
        st = forward_batch.residual_stream
        if st.pending is None:
            st.residual = hidden_states                        # stack entry: the embedding
        else:
            assert hidden_states is st.pending.value           # SGLang checks the identity
            st.residual = st.residual + st.pending.value
        self.seen = st.residual.clone()                        # the stream entering this layer
        st.residual = st.residual + torch.tanh(st.residual @ self.wa)   # attention's add
        out = torch.tanh(st.residual @ self.wm)                # the MLP's contribution
        st.pending = types.SimpleNamespace(value=out, owed=None)
        return out


class StreamStack(torch.nn.Module if torch else object):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([StreamLayer(i) for i in range(LAYERS)])
        self.start_layer, self.end_layer = 0, LAYERS
        self.config = types.SimpleNamespace(hidden_size=H)

    def set_dflash_layers_to_capture(self, ids):
        pass

    def forward(self, x):
        fb = types.SimpleNamespace(residual_stream=Stream())
        h = x
        for layer in self.layers:
            h = layer(positions=None, hidden_states=h, forward_batch=fb, captured_last_layer_outputs=None)
        st = fb.residual_stream
        return st.residual + st.pending.value


@unittest.skipIf(torch is None, "torch/numpy not installed")
class Qwen35Test(unittest.TestCase):
    def setUp(self):
        import gb10_ablate

        self.f = gb10_ablate
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        g = np.random.default_rng(1)
        self.dirs = {i: g.standard_normal(H).astype(np.float32) for i in (2, 3, 4)}
        self.path = os.path.join(self.tmp.name, "dirs.gguf")
        write_gguf(self.path, {"general.architecture": "controlvector", "glp.mode": "project",
                               "glp.hook_point": "residual_stream_post_layer",
                               "glp.layer_ids_zero_based": "2,3,4", "glp.alpha_default": 1.0},
                   {f"direction.{i}": v / np.linalg.norm(v) for i, v in self.dirs.items()})
        for k in ("GB10_ABLATE", "GB10_ABLATE_ALPHA", "GB10_ABLATE_AT"):
            self.addCleanup(os.environ.pop, k, None)
        os.environ["GB10_ABLATE"] = self.path

    def unit(self, i):
        v = torch.from_numpy(self.dirs[i])
        return v / v.norm()

    def model(self, **env):
        os.environ.update(env)
        mod = types.ModuleType(self.f.QWEN3_5_MODULE)
        outer = type(self.f.QWEN3_5_CLASSES[0], (torch.nn.Module,), {
            "__init__": lambda self_: (torch.nn.Module.__init__(self_), setattr(self_, "model", StreamStack()))[0],
            "load_weights": lambda self_, w: "loaded"})
        inner = type(self.f.QWEN3_5_CLASSES[1], (StreamStack,), {"load_weights": lambda self_, w: "inner"})
        mod.__dict__.update({outer.__name__: outer, inner.__name__: inner})
        self.f.apply_qwen3_5(mod)
        m = outer()
        self.assertEqual(m.load_weights([]), "loaded")
        return m

    def test_gguf_loads_and_additive_is_refused(self):
        dirs = self.f.load_directions(self.path, H)
        self.assertEqual(sorted(dirs), [2, 3, 4])
        self.assertTrue(np.allclose(dirs[3], self.dirs[3] / np.linalg.norm(self.dirs[3]), atol=1e-6))
        add = os.path.join(self.tmp.name, "additive.gguf")
        write_gguf(add, {"general.architecture": "controlvector"},
                   {"direction.2": np.ones(H, np.float32)})
        with self.assertRaisesRegex(RuntimeError, "additive"):
            self.f.load_directions(add, H)
        bad = os.path.join(self.tmp.name, "ids.gguf")
        write_gguf(bad, {"glp.mode": "project", "glp.layer_ids_zero_based": "2,3"},
                   {f"direction.{i}": np.ones(H, np.float32) for i in (2, 3, 4)})
        with self.assertRaisesRegex(RuntimeError, "layer_ids"):
            self.f.load_directions(bad, H)

    @unittest.skipUnless(os.environ.get("GB10_ABLATE_TEST_GGUF"), "GB10_ABLATE_TEST_GGUF not set")
    def test_a_real_file(self):
        path = os.environ["GB10_ABLATE_TEST_GGUF"]
        meta, tensors = self.f.read_gguf(path)
        hidden = next(iter(tensors.values())).shape[-1]
        dirs = self.f.load_directions(path, hidden)
        ids = sorted(int(x) for x in meta["glp.layer_ids_zero_based"].split(","))
        self.assertEqual(sorted(dirs), ids)
        for v in dirs.values():
            self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)

    def test_post_layer_stream_loses_d(self):
        m = self.model(GB10_ABLATE_ALPHA="1")
        stack = m.model
        self.assertEqual(len(stack._gb10_ablate_hooks), 3)
        x = torch.randn(5, H)
        y = stack(x)
        for i in (2, 3, 4):   # the stream entering layer i + 1 is layer i's output stream
            self.assertLess((stack.layers[i + 1].seen @ self.unit(i)).abs().max().item(), 1e-4, i)
        self.assertTrue(torch.allclose(stack.layers[2].seen, self.reference(x, upto=2)))
        self.assertEqual(y.shape, x.shape)

    def reference(self, x, upto, alpha=1.0):
        """The stream entering layer `upto`, with h - alpha d (d . h) after each listed layer."""
        h = x
        for i, layer in enumerate(StreamStack().layers[:upto]):
            h = h + torch.tanh(h @ layer.wa)
            h = h + torch.tanh(h @ layer.wm)
            if i in self.dirs:
                d = self.unit(i)
                h = h - alpha * torch.outer(h @ d, d)
        return h

    def test_matches_the_projection_of_the_summed_stream(self):
        m = self.model(GB10_ABLATE_ALPHA="0.5")
        x = torch.randn(4, H)
        m.model(x)
        for upto in (3, 4, 5, 7):
            self.assertTrue(torch.allclose(m.model.layers[upto].seen, self.reference(x, upto, 0.5),
                                           atol=1e-5), upto)

    def test_alpha_zero_and_input_mode(self):
        m = self.model(GB10_ABLATE_ALPHA="0")
        self.assertEqual(m._gb10_ablate_hooks, [])
        m = self.model(GB10_ABLATE_ALPHA="1", GB10_ABLATE_AT="input")
        m.model(torch.randn(3, H))
        self.assertLess((m.model.layers[2].seen @ self.unit(2)).abs().max().item(), 1e-4)
        self.assertGreater((m.model.layers[1].seen @ self.unit(2)).abs().max().item(), 1e-3)

    def test_refusals(self):
        write_gguf(self.path, {"glp.mode": "project"}, {"direction.0": np.ones(H, np.float32)})
        with self.assertRaisesRegex(RuntimeError, "layer 0"):
            self.model(GB10_ABLATE_AT="input")
        write_gguf(self.path, {"glp.mode": "project"}, {"direction.2": np.ones(H, np.float32)})
        m = self.model(GB10_ABLATE_AT="output")
        layer = m.model.layers[2]
        orig = layer.forward

        def owed(**kw):
            out = orig(**kw)
            kw["forward_batch"].residual_stream.pending.owed = "all-reduce"
            return out
        layer.forward = owed
        with self.assertRaisesRegex(RuntimeError, "pending contribution"):
            m.model(torch.randn(2, H))


if __name__ == "__main__":
    unittest.main(verbosity=2)
