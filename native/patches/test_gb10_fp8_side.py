#!/usr/bin/env python3
"""CPU tests for gb10_fp8_side (torch only, no GPU, no SGLang):

    python3 patches/test_gb10_fp8_side.py

Covers the per-channel FP8 quantization, which layers get converted (name,
dtype, Marlin shape limits), the GDN fused-buffer drop, the quant_method
swap, and the output heads after the EAGLE worker's init_lm_head (token-map
slice, shared target module, multi-layer drafts, tied embedding). The Marlin repack and GEMM themselves need the GPU: on the Spark the
boot log has "FP8 side (target): N linear layers to FP8 ..." and the answers
must stay coherent.
"""

import importlib
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class QuantizeTest(unittest.TestCase):
    def test_per_channel_error_is_small(self):
        import gb10_fp8_side as f

        torch.manual_seed(0)
        w = (torch.randn(256, 512) * torch.logspace(-3, 0, 256)[:, None]).to(torch.bfloat16)
        q, s = f.quantize_per_channel(w)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.shape, (256,))
        back = q.float() * s[:, None]
        rel = ((back - w.float()).norm(dim=1) / w.float().norm(dim=1)).max().item()
        self.assertLess(rel, 0.05)  # e4m3: ~2-3% RMS per row
        self.assertTrue(torch.isfinite(back).all())

    def test_zero_row_does_not_divide_by_zero(self):
        import gb10_fp8_side as f

        w = torch.zeros(64, 128, dtype=torch.bfloat16)
        q, s = f.quantize_per_channel(w)
        self.assertTrue(torch.isfinite(q.float()).all())
        self.assertTrue((q.float() == 0).all())


class Unquant:
    pass


class Linear(torch.nn.Module if torch else object):
    def __init__(self, n, k, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k).to(dtype or torch.bfloat16), requires_grad=False)
        self.bias = None
        self.quant_method = Unquant()


class GDN(torch.nn.Module if torch else object):
    def __init__(self, a=256, b=64):
        super().__init__()
        self.in_proj_qkvz = Linear(a, 256)
        self.in_proj_ba = Linear(b, 256)
        self.out_proj = Linear(256, 128)
        self._fused_in_proj_weight = torch.cat([self.in_proj_qkvz.weight, self.in_proj_ba.weight])


@unittest.skipIf(torch is None, "torch not installed")
class ConvertTest(unittest.TestCase):
    def model(self, ba_rows=64):
        m = torch.nn.Module()
        m.layers = torch.nn.ModuleList()
        layer = torch.nn.Module()
        layer.linear_attn = GDN(b=ba_rows)
        layer.mlp = torch.nn.Module()
        layer.mlp.shared_expert = torch.nn.Module()
        layer.mlp.shared_expert.gate_up_proj = Linear(512, 256)
        layer.mlp.shared_expert.down_proj = Linear(256, 256)
        layer.mlp.gate = Linear(64, 256)                       # router: name not matched
        layer.self_attn = torch.nn.Module()
        layer.self_attn.qkv_proj = Linear(384, 256)
        layer.self_attn.o_proj = Linear(256, 100)              # K % 128 != 0 -> skipped
        m.layers.append(layer)
        m.lm_head = Linear(1024, 256)                          # not matched
        return m

    def convert(self, m):
        import gb10_fp8_side as f

        prepared = []

        def prepare(module, size_k_first):
            self.assertFalse(size_k_first)
            self.assertEqual(module.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(module.weight_scale.shape, (module.weight.shape[0],))
            module.workspace = "ws"
            prepared.append(module)

        calls = []

        def apply(**kw):
            calls.append(kw)
            return "out"

        stats = f.convert_model(
            m, is_linear=lambda mod: isinstance(mod, Linear),
            is_unquantized=lambda qm: isinstance(qm, Unquant),
            prepare_fn=prepare, apply_fn=apply)
        return f, stats, prepared, calls

    def test_selection_and_swap(self):
        m = self.model()
        f, stats, prepared, calls = self.convert(m)
        layer = m.layers[0]
        converted = [layer.linear_attn.in_proj_qkvz, layer.linear_attn.in_proj_ba,
                     layer.linear_attn.out_proj, layer.mlp.shared_expert.gate_up_proj,
                     layer.mlp.shared_expert.down_proj, layer.self_attn.qkv_proj]
        self.assertEqual(stats["converted"], len(converted))
        self.assertEqual(stats["skipped"], 1)                  # o_proj, K = 100
        self.assertEqual({id(x) for x in prepared}, {id(x) for x in converted})
        for mod in (layer.mlp.gate, m.lm_head, layer.self_attn.o_proj):
            self.assertEqual(mod.weight.dtype, torch.bfloat16)
            self.assertIsInstance(mod.quant_method, Unquant)
        self.assertIsNone(layer.linear_attn._fused_in_proj_weight)
        self.assertEqual(stats["bytes_after"] * 2, stats["bytes_before"])
        qkv = layer.self_attn.qkv_proj
        self.assertIsInstance(qkv.quant_method, f.Fp8MarlinSideMethod)
        self.assertEqual(qkv.quant_method.apply(qkv, "x"), "out")
        self.assertEqual((calls[0]["size_n"], calls[0]["size_k"]), (384, 256))

    def test_fused_buffer_kept_when_a_half_stays_bf16(self):
        m = self.model()
        gdn = m.layers[0].linear_attn
        os.environ["GB10_FP8_SIDE_LAYERS"] = r"\.in_proj_qkvz$"   # in_proj_ba stays BF16
        self.addCleanup(os.environ.pop, "GB10_FP8_SIDE_LAYERS")
        # as finalize_fused_in_proj leaves it: both halves are views of the buffer
        gdn.in_proj_qkvz.weight.data = gdn._fused_in_proj_weight[:256]
        gdn.in_proj_ba.weight.data = gdn._fused_in_proj_weight[256:]
        fused_ptr = gdn._fused_in_proj_weight.untyped_storage().data_ptr()
        self.convert(m)
        # qkvz converted: the buffer must not be used, and the BF16 half must
        # not keep it (and qkvz's old BF16 rows) alive
        self.assertIsNone(gdn._fused_in_proj_weight)
        self.assertEqual(gdn.in_proj_ba.weight.dtype, torch.bfloat16)
        self.assertNotEqual(gdn.in_proj_ba.weight.untyped_storage().data_ptr(), fused_ptr)

    def test_narrow_output_is_padded_and_sliced(self):
        import gb10_fp8_side as f

        m = self.model(ba_rows=96)                              # GDN in_proj_ba on the real model
        gdn = m.layers[0].linear_attn
        ba_bf16 = gdn.in_proj_ba.weight.detach().float().clone()

        def prepare(module, size_k_first):
            if module is gdn.in_proj_ba:
                self.assertEqual(module.output_size_per_partition, 128)  # the padded width
                self.assertEqual(module.weight.shape, (128, 256))
            module.workspace = "ws"

        seen = {}

        def apply(*, input, weight, weight_scale, workspace, size_n, size_k, bias):
            seen["size_n"] = size_n
            # dequantize and multiply, like the kernel: [.., K] x [N_pad, K]^T
            w = weight.float() * weight_scale.float()[:, None]
            return (input.float() @ w.T).to(torch.bfloat16)

        stats = f.convert_model(m, is_linear=lambda mod: isinstance(mod, Linear),
                                is_unquantized=lambda qm: isinstance(qm, Unquant),
                                prepare_fn=prepare, apply_fn=apply)
        self.assertEqual(stats["padded"], 1)
        ba = gdn.in_proj_ba
        self.assertEqual(ba.output_size_per_partition, 96)      # what the model reads
        x = torch.randn(4, 256, dtype=torch.bfloat16)
        out = ba.quant_method.apply(ba, x)
        self.assertEqual(seen["size_n"], 128)
        self.assertEqual(tuple(out.shape), (4, 96))
        self.assertTrue(out.is_contiguous())
        ref = x.float() @ ba_bf16.T
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(rel, 0.05)

    def test_padding_needs_no_bias(self):
        import gb10_fp8_side as f

        m = self.model(ba_rows=96)
        ba = m.layers[0].linear_attn.in_proj_ba
        ba.bias = torch.nn.Parameter(torch.zeros(96, dtype=torch.bfloat16), requires_grad=False)
        _, stats, _, _ = self.convert(m)
        self.assertIsInstance(ba.quant_method, Unquant)
        self.assertEqual(stats["skipped"], 2)                   # this and o_proj (K = 100)
        self.assertIsNotNone(f)

    def test_custom_pattern(self):
        import gb10_fp8_side as f

        os.environ["GB10_FP8_SIDE_LAYERS"] = r"\.out_proj$"
        try:
            m = self.model()
            _, stats, _, _ = self.convert(m)
            self.assertEqual(stats["converted"], 1)
            self.assertIsInstance(m.layers[0].linear_attn.out_proj.quant_method, f.Fp8MarlinSideMethod)
        finally:
            del os.environ["GB10_FP8_SIDE_LAYERS"]


class UnquantizedEmbeddingMethod:  # the name logits_processor checks
    pass


class Head(torch.nn.Module if torch else object):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class Model(torch.nn.Module if torch else object):
    def __init__(self, head, embed):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Module()
        self.model.embed_tokens.weight = embed
        self.lm_head = head

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


def ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def fake_parts(prepared):
    def prepare(module, size_k_first):
        module.workspace = "ws"
        prepared.append(module)

    def apply(*, input, weight, weight_scale, workspace, size_n, size_k, bias):
        w = weight.float() * weight_scale.float()[:, None]
        return (input.float() @ w.T).to(torch.bfloat16)

    return dict(is_linear=None, is_unquantized=None, prepare_fn=prepare, apply_fn=apply)


@unittest.skipIf(torch is None, "torch not installed")
class HeadTest(unittest.TestCase):
    VOCAB, HIDDEN, HOT = 512, 256, 128

    def setUp(self):
        import gb10_fp8_side as f

        self.f = f
        self.prepared = []
        self._parts = f._sglang_parts
        f._sglang_parts = lambda: fake_parts(self.prepared)
        self.env = {k: os.environ.pop(k, None) for k in ("GB10_FP8_DRAFT_HEAD", "GB10_FP8_TARGET_HEAD")}
        torch.manual_seed(0)
        self.head_w = torch.randn(self.VOCAB, self.HIDDEN).to(torch.bfloat16)
        self.embed = torch.nn.Parameter(torch.randn(self.VOCAB, self.HIDDEN).to(torch.bfloat16),
                                        requires_grad=False)

    def tearDown(self):
        self.f._sglang_parts = self._parts
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def worker(self, *, token_map=True, layers=1, tied=False):
        """What EagleDraftWorker.init_lm_head leaves behind."""
        target_head = Head(self.embed.data if tied else self.head_w.clone())
        target = Model(target_head, self.embed)
        drafts = []
        for _ in range(layers):
            if token_map:               # head.clone(); head.data = head.data[hot_token_id]
                hot = torch.arange(0, self.VOCAB, self.VOCAB // self.HOT)
                d_head = Head(target_head.weight.data[hot].clone())
            elif layers > 1:            # multi-layer: set_embed_and_head(embed, head)
                d_head = Head(target_head.weight.data)
            else:                       # set_lm_head_from_target(target_lm_head)
                d_head = target_head
            drafts.append(Model(d_head, self.embed))
        w = ns(target_worker=ns(model_runner=ns(model=target)))
        if layers > 1:
            w.draft_runner_list = [ns(model=d) for d in drafts]
        else:
            w.draft_runner = ns(model=drafts[0])
        return w, target, drafts

    def env_set(self, draft="0", target="0"):
        os.environ["GB10_FP8_DRAFT_HEAD"] = draft
        os.environ["GB10_FP8_TARGET_HEAD"] = target

    def assert_close(self, head, ref_weight):
        x = torch.randn(3, self.HIDDEN, dtype=torch.bfloat16)
        out = head.quant_method.apply(head, x)
        ref = x.float() @ ref_weight.float().T
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        self.assertLess(((out.float() - ref).norm() / ref.norm()).item(), 0.05)

    def test_draft_head_with_token_map(self):
        self.env_set(draft="1")
        w, target, (draft,) = self.worker()
        ref = draft.lm_head.weight.detach().clone()
        report = self.f.heads_after_init_lm_head(w)
        self.assertEqual(report, [("draft", "converted")])
        self.assertIsInstance(draft.lm_head.quant_method, self.f.Fp8MarlinSideMethod)
        self.assertEqual(draft.lm_head.weight.shape, (self.HOT, self.HIDDEN))
        self.assert_close(draft.lm_head, ref)
        self.assertEqual(target.lm_head.weight.dtype, torch.bfloat16)   # target untouched
        self.assertIsInstance(target.lm_head.quant_method, UnquantizedEmbeddingMethod)

    def test_draft_only_leaves_a_shared_target_module_alone(self):
        self.env_set(draft="1")
        w, target, (draft,) = self.worker(token_map=False)
        self.assertIs(draft.lm_head, target.lm_head)
        self.assertEqual(self.f.heads_after_init_lm_head(w), [])
        self.assertEqual(target.lm_head.weight.dtype, torch.bfloat16)
        self.assertEqual(self.prepared, [])

    def test_target_head_converts_a_shared_module_once(self):
        self.env_set(draft="1", target="1")
        w, target, (draft,) = self.worker(token_map=False)
        report = self.f.heads_after_init_lm_head(w)
        self.assertEqual(report[0], ("target", "converted"))
        self.assertEqual(report[1][0], "draft")
        self.assertIn("same module", report[1][1])
        self.assertEqual(len(self.prepared), 1)
        self.assert_close(target.lm_head, self.head_w)

    def test_both_heads_with_token_map(self):
        self.env_set(draft="1", target="1")
        w, target, (draft,) = self.worker()
        report = self.f.heads_after_init_lm_head(w)
        self.assertEqual(report, [("target", "converted"), ("draft", "converted")])
        self.assertEqual(target.lm_head.weight.shape, (self.VOCAB, self.HIDDEN))  # fake prepare: no repack

    def test_multi_layer_drafts_share_one_fp8_copy(self):
        self.env_set(draft="1")
        w, target, drafts = self.worker(token_map=False, layers=3)
        report = self.f.heads_after_init_lm_head(w)
        self.assertEqual([r for _, r in report], ["converted", "shared", "shared"])
        self.assertEqual(len(self.prepared), 1)
        self.assertIs(drafts[1].lm_head.weight, drafts[0].lm_head.weight)
        self.assertEqual(target.lm_head.weight.dtype, torch.bfloat16)   # its own module stays
        self.assert_close(drafts[2].lm_head, self.head_w)

    def test_tied_head_is_left_alone(self):
        self.env_set(target="1")
        w, target, _ = self.worker(tied=True)
        report = self.f.heads_after_init_lm_head(w)
        self.assertEqual(report, [("target", "tied to the input embedding")])
        self.assertEqual(target.lm_head.weight.dtype, torch.bfloat16)

    def test_off_by_default(self):
        w, target, (draft,) = self.worker()
        self.assertEqual(self.f.heads_after_init_lm_head(w), [])
        self.assertEqual(self.prepared, [])

    def test_bad_mode(self):
        os.environ["GB10_FP8_TARGET_HEAD"] = "yes"
        with self.assertRaises(RuntimeError):
            self.f.target_head_mode()

    def test_apply_spec_wraps_init_lm_head(self):
        import types

        mod = types.ModuleType(self.f.SPEC_MODULES[0])
        test = self
        calls = []

        class EagleDraftWorker:
            def init_lm_head(self):
                calls.append("orig")
                test.assertEqual(test.prepared, [])     # converted only after the original
                w, _, _ = test.worker()
                self.target_worker, self.draft_runner = w.target_worker, w.draft_runner

        class StandaloneDraftWorker(EagleDraftWorker):   # overrides: left alone
            def init_lm_head(self):
                calls.append("standalone")

        EagleDraftWorker.__module__ = StandaloneDraftWorker.__module__ = mod.__name__
        mod.EagleDraftWorker = EagleDraftWorker
        mod.StandaloneDraftWorker = StandaloneDraftWorker
        self.env_set(draft="1")
        self.f.apply_spec(mod)
        self.f.apply_spec(mod)                             # a second hook does not double-wrap
        worker = EagleDraftWorker()
        worker.init_lm_head()
        self.assertEqual(calls, ["orig"])
        self.assertEqual(len(self.prepared), 1)
        self.assertIsInstance(worker.draft_runner.model.lm_head.quant_method, self.f.Fp8MarlinSideMethod)
        StandaloneDraftWorker().init_lm_head()
        self.assertEqual(calls, ["orig", "standalone"])
        os.environ["GB10_FP8_TARGET_HEAD"] = "load"        # target converted at load: refuse
        with self.assertRaises(RuntimeError):
            EagleDraftWorker().init_lm_head()

    def test_apply_spec_without_the_class_fails_loudly(self):
        import types

        with self.assertRaises(RuntimeError):
            self.f.apply_spec(types.ModuleType(self.f.SPEC_MODULES[0]))
        self.f.apply_spec(types.ModuleType(self.f.SPEC_MODULES[1]))   # optional module: fine

    def test_target_head_at_load(self):
        import types

        class Qwen4ExpForConditionalGeneration(Model):
            def load_weights(self, weights):
                self.loaded = weights

        mod = types.ModuleType(self.f.TARGET_MODULE)
        mod.Qwen4ExpForConditionalGeneration = Qwen4ExpForConditionalGeneration
        os.environ["GB10_FP8_TARGET_HEAD"] = "load"
        side = os.environ.pop("GB10_FP8_SIDE", None)
        self.addCleanup(lambda: side is not None and os.environ.__setitem__("GB10_FP8_SIDE", side))
        self.f.apply_target(mod)
        m = Qwen4ExpForConditionalGeneration(Head(self.head_w.clone()), self.embed)
        m.load_weights("w")
        self.assertEqual(m.loaded, "w")
        self.assertEqual(self.prepared, [m.lm_head])       # side layers off: only the head
        self.assert_close(m.lm_head, self.head_w)


class HookOrderTest(unittest.TestCase):
    def test_two_hooks_on_one_module_both_run_in_order(self):
        import gb10_ple_mmap as g

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "gb10twohooks"))
            open(os.path.join(d, "gb10twohooks", "__init__.py"), "w").close()
            with open(os.path.join(d, "gb10twohooks", "m.py"), "w") as fh:
                fh.write("X = 1\n")
            seen = []
            sys.path.insert(0, d)
            try:
                g.install_import_hook("gb10twohooks.m", lambda mod: seen.append("first"))
                g.install_import_hook("gb10twohooks.m", lambda mod: seen.append("second"))
                importlib.import_module("gb10twohooks.m")
                self.assertEqual(seen, ["first", "second"])
            finally:
                sys.path.remove(d)
                sys.meta_path[:] = [x for x in sys.meta_path if not isinstance(x, g._Finder)]
                for name in ("gb10twohooks.m", "gb10twohooks"):
                    sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
