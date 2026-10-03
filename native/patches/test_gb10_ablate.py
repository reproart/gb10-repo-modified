#!/usr/bin/env python3
"""CPU tests for gb10_ablate (torch + numpy, no GPU, no SGLang):

    python3 patches/test_gb10_ablate.py

The projection itself (each stream / the streams' mean, alpha), which layers
get hooks, output vs input, and the loading checks. On the Spark the boot log
has "GB10_ABLATE: N layers, alpha ..." and the answers must stay coherent.
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

    def test_bad_settings(self):
        os.environ["GB10_ABLATE_AT"] = "middle"
        with self.assertRaises(RuntimeError):
            self.f.settings()

    def test_separate_residual_fails_loudly(self):
        m, x = self.run_model()
        m.model.layers[2].forward = lambda **kw: (kw["hidden_states"], torch.zeros(1))
        with self.assertRaises(RuntimeError):
            m.model(x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
