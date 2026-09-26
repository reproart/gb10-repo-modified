#!/usr/bin/env python3
"""CPU tests for gb10_ple_mmap: checkpoint layout, the read-only mapping, the
address arithmetic the gather kernel uses, page-cache hints and the import
hook. No GPU, torch or SGLang needed (numpy only):

    python3 patches/test_gb10_ple_mmap.py

With torch installed, the patch is also applied to a stand-in for SGLang
0.5.20's qwen4_exp module (SGLangIntegrationTest); with triton too, the gather
kernel itself runs under the Triton interpreter (TRITON_INTERPRET=1, CPU) on
the mapped shards (KernelTest). What only the Spark can show is the real
module and the GPU reading the mapping: the boot log ("PLE table: layer ...
read in place") and bench/perf.py (README, "Qwen3.8-Flash-Next").
"""

import hashlib
import json
import os
import random
import struct
import sys
import tempfile
import textwrap
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb10_ple_mmap as g  # noqa: E402

PREFIX = "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding"


def write_safetensors(path, tensors):
    """tensors: {name: (dtype str, shape, raw bytes)}; returns nothing."""
    header, blobs, off = {}, [], 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


class Checkpoint:
    """A fake checkpoint: one PLE layer (id 1) of `vocab` rows split like the
    real one into `parts` shards over several files, mixed with other tensors,
    plus an MTP tensor that must be ignored. Shard `drop` is left out."""

    def __init__(self, d, vocab=1000, parts=8, dim=12, dtype="F8_E4M3", files=3, drop=None, layer=1):
        self.dir, self.vocab, self.parts, self.dim, self.dtype = d, vocab, parts, dim, dtype
        self.shard_size = -(-vocab // parts)
        self.row_bytes = dim * g.DTYPE_BYTES[dtype]
        rng = np.random.default_rng(0)
        self.table = rng.integers(0, 256, size=(vocab, self.row_bytes), dtype=np.uint8)
        self.present = np.zeros(vocab, dtype=bool)
        per_file = [dict() for _ in range(files)]
        for k in range(parts):
            lo, hi = k * self.shard_size, min((k + 1) * self.shard_size, vocab)
            if k == drop or lo >= hi:
                continue
            self.present[lo:hi] = True
            f = per_file[k % files]
            f[f"other.{k}.weight"] = ("BF16", (3, 5), rng.integers(0, 256, 30, dtype=np.uint8).tobytes())
            f[f"{PREFIX.format(layer)}.shard_{k}.weight"] = (dtype, (hi - lo, dim), self.table[lo:hi].tobytes())
        per_file[0]["mtp.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"] = (dtype, (1, dim), bytes(self.row_bytes))
        per_file[0][f"{PREFIX.format(layer)}.weight_scale"] = ("BF16", (1,), b"\x80\x3f")
        self.paths = []
        for i, t in enumerate(per_file):
            p = os.path.join(d, f"model-{i:05d}-of-{files:05d}.safetensors")
            write_safetensors(p, t)
            self.paths.append(p)
        with open(os.path.join(d, "config.json"), "w") as f:
            f.write("{}")

    def digest(self):
        h = hashlib.sha256()
        for p in self.paths:
            with open(p, "rb") as f:
                h.update(f.read())
        return h.hexdigest()


def read_row(table, row):
    addr = table.row_address(row)
    return None if addr == 0 else np.frombuffer(
        __import__("ctypes").string_at(addr, table.row_bytes), dtype=np.uint8)


class LayoutTest(unittest.TestCase):
    def test_scan_groups_shards_and_ignores_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d)
            layers = g.scan_table_dir(d)
            self.assertEqual(list(layers), [1])
            self.assertEqual(sorted(layers[1]), list(range(ck.parts)))
            s = layers[1][3]
            self.assertEqual((s.rows, s.dim, s.dtype), (ck.shard_size, ck.dim, "F8_E4M3"))

    def test_rejects_wrong_split(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d)
            shards = g.scan_table_dir(d)[1]
            with self.assertRaisesRegex(ValueError, "split_ngram_parts"):
                g.map_table(shards, d, ck.shard_size - 1, ck.vocab)  # shards too big
            with self.assertRaisesRegex(ValueError, "split_ngram_parts"):
                g.map_table(shards, d, ck.shard_size, ck.shard_size * 4)  # too few slots

    def test_rejects_unknown_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            write_safetensors(os.path.join(d, "a.safetensors"),
                              {f"{PREFIX.format(1)}.shard_0.weight": ("F32", (2, 2), bytes(16))})
            with self.assertRaisesRegex(ValueError, "F32"):
                g.scan_table_dir(d)


class MappingTest(unittest.TestCase):
    def check_rows(self, ck, table):
        rows = list(range(ck.vocab)) + [ck.vocab + 5]
        for r in rows:
            got = read_row(table, r)
            if r < ck.vocab and ck.present[r]:
                self.assertIsNotNone(got, r)
                np.testing.assert_array_equal(got, ck.table[r], err_msg=f"row {r}")
            else:
                self.assertIsNone(got, r)

    def test_every_row_reads_back_fp8(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d)
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab)
            self.assertEqual(t.mapped_rows, ck.vocab)
            self.check_rows(ck, t)

    def test_bf16_missing_shard_and_short_last_shard(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d, vocab=997, parts=7, dim=6, dtype="BF16", drop=2)
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab)
            self.assertEqual(t.rows[2], 0)
            self.assertEqual(t.bases[2], 0)
            self.assertLess(t.rows[-1], ck.shard_size)
            self.check_rows(ck, t)

    def test_mapping_is_read_only_shared_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d)
            before = ck.digest()
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab)
            self.check_rows(ck, t)
            perms = set()
            with open("/proc/self/maps") as f:
                for line in f:
                    fields = line.split()
                    if len(fields) >= 6 and fields[5] in ck.paths:
                        lo, hi = (int(x, 16) for x in fields[0].split("-"))
                        self.assertTrue(t.addr <= lo and hi <= t.addr + t.nbytes)
                        perms.add(fields[1])
            self.assertEqual(perms, {"r--s"})
            self.assertEqual(ck.digest(), before)
            self.assertEqual(sorted(os.listdir(d)), sorted([os.path.basename(p) for p in ck.paths] + ["config.json"]))

    def test_copy_on_write_mode_maps_private(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d)
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab, writable=True)
            self.check_rows(ck, t)
            with open("/proc/self/maps") as f:
                perms = {ln.split()[1] for ln in f if len(ln.split()) >= 6 and ln.split()[5] in ck.paths}
            self.assertEqual(perms, {"rw-p"})


class PrefetchTest(unittest.TestCase):
    def test_pages_match_brute_force(self):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d, vocab=5000, parts=9, dim=160, drop=4)
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab)
            page = 4096
            ids = np.array(random.Random(1).sample(range(ck.vocab + 50), 700), dtype=np.int64)
            want = {}
            for r in ids.tolist():
                k, local = divmod(r, ck.shard_size)
                if k >= t.num_shards or local >= t.rows[k]:
                    continue
                f = int(t.shard_file[k])
                start = int(t.shard_file_offset[k]) + local * t.row_bytes
                want.setdefault(f, set()).update({start // page, (start + t.row_bytes - 1) // page})
            got = g.pages_for_rows(t, ids, page)
            self.assertEqual({f: set(p.tolist()) for f, p in got.items()}, want)


class HookTest(unittest.TestCase):
    def test_hook_runs_after_import_and_sources_stay_readable(self):
        import importlib
        import inspect

        with tempfile.TemporaryDirectory() as d:
            pkg = os.path.join(d, "gb10fakepkg")
            os.mkdir(pkg)
            open(os.path.join(pkg, "__init__.py"), "w").close()
            with open(os.path.join(pkg, "mod.py"), "w") as f:
                f.write(textwrap.dedent("""
                    VALUE = 1
                    def kernel():
                        return VALUE
                """))
            seen = []

            def hook(module):
                seen.append(module.__name__)
                module.VALUE = 2

            sys.path.insert(0, d)
            try:
                g.install_import_hook("gb10fakepkg.mod", hook)
                mod = importlib.import_module("gb10fakepkg.mod")
                self.assertEqual(seen, ["gb10fakepkg.mod"])
                self.assertEqual(mod.kernel(), 2)
                self.assertIn("return VALUE", inspect.getsource(mod.kernel))
            finally:
                sys.path.remove(d)
                sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, g._Finder)]
                for m in ("gb10fakepkg.mod", "gb10fakepkg"):
                    sys.modules.pop(m, None)


try:
    import torch
except ImportError:  # the layout tests above need numpy only
    torch = None


def fake_sglang_module(allocations):
    """A stand-in for sglang.srt.models.qwen4_exp (0.5.20) with the pieces the
    patch touches, built the way the real ones are: the embedding calls the
    module-global allocate_ple_host_table and wraps the result in a new
    nn.Parameter, the PLE layer has the real __init__ signature, and
    load_weights sees (name, tensor) pairs."""
    import types

    from torch import nn

    mod = types.ModuleType("fake_qwen4_exp")

    def allocate_ple_host_table(shape, dtype, backend="pinned", table_dir=None, tag=None):
        allocations.append(backend)
        if backend == "file":
            raise AssertionError("stock file backend reached: it would write the table")
        return torch.empty(tuple(shape), dtype=dtype)

    class ShardIndices:
        def __init__(self, n):
            self.org_vocab_start_index, self.org_vocab_end_index = 0, n

    class Source:
        def __init__(self, vocab, dim, dtype):
            self.org_vocab_size, self.embedding_dim = vocab, dim
            self.shard_indices = ShardIndices(vocab)
            self.weight = torch.empty(0, dtype=dtype)
            self.weight.weight_loader = "stock"
            self.weight_scale = torch.ones(1)

    class Qwen4ExpPinnedHostEmbedding(nn.Module):
        def __init__(self, embedding, *, backend="pinned", table_dir=None):
            super().__init__()
            self.org_vocab_size = embedding.org_vocab_size
            self.embedding_dim = embedding.embedding_dim
            self.shard_indices = embedding.shard_indices
            self.quant_method = None
            host = mod.allocate_ple_host_table(
                shape=(embedding.org_vocab_size, embedding.embedding_dim),
                dtype=embedding.weight.dtype, backend=backend, table_dir=table_dir, tag="rows")
            self._file_prefetcher = getattr(host, "_sglang_ple_file_path", None)
            self._file_rss_trimmer = None
            w = nn.Parameter(host, requires_grad=False)
            for k, v in vars(embedding.weight).items():
                setattr(w, k, v)
            self.register_parameter("weight", w)
            self.register_buffer("weight_scale", embedding.weight_scale)
            self._block_d = 1 << (embedding.embedding_dim - 1).bit_length()

        def allocate_output(self, shape, device):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        def gather(self, input_ids, out=None):
            raise AssertionError("stock gather reached")

    class NGram(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.ngram_embedding = emb

    class Qwen4ExpPLELayer(nn.Module):
        def __init__(self, config, quant_config=None, prefix="", layer_id=None, ple_layer_index=0):
            super().__init__()
            src = Source(config.vocab, config.dim, torch.float8_e4m3fn)
            self.ple_embedding = NGram(src)
            if config.ple_offload_embedding:
                self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                    src, backend=config.ple_offload_backend, table_dir="/unused")

    class Qwen4ExpForConditionalGeneration(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ple = Qwen4ExpPLELayer(config, None, "model.layers.1.ple", layer_id=1)
            self.seen = []

        def load_weights(self, weights):
            self.seen = [n for n, _ in weights]
            return set(self.seen)

    for obj in (allocate_ple_host_table, Qwen4ExpPinnedHostEmbedding, Qwen4ExpPLELayer,
                Qwen4ExpForConditionalGeneration):
        setattr(mod, obj.__name__, obj)
    return mod


@unittest.skipIf(torch is None, "torch not installed")
class SGLangIntegrationTest(unittest.TestCase):
    def setUp(self):
        import types

        class Env:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

        envs = types.SimpleNamespace(
            SGLANG_QWEN4_PLE_FILE_PREFETCH=Env(True),
            SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=Env(8.0),
            SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S=Env(3600.0),
        )
        self.trimmers = []
        test = self

        class Trimmer:
            def __init__(self, **kw):
                test.trimmers.append(kw)

            def start(self):
                pass

        self.saved = {k: sys.modules.get(k) for k in (
            "sglang", "sglang.srt", "sglang.srt.environ",
            "sglang.srt.models", "sglang.srt.models.qwen4_exp_ple_table", "gb10_ple_kernel")}
        for name in ("sglang", "sglang.srt", "sglang.srt.models"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["sglang.srt.environ"] = types.SimpleNamespace(envs=envs)
        sys.modules["sglang.srt.models.qwen4_exp_ple_table"] = types.SimpleNamespace(PleFileRssTrimmer=Trimmer)
        self.kernel_calls = []
        sys.modules["gb10_ple_kernel"] = types.SimpleNamespace(gather_rows=self.emulate_kernel)
        self.saved_device = g._device
        g._device = lambda torch: torch.device("cpu")
        self.env = {k: os.environ.get(k) for k in (g.ENV_TABLE_DIR, "GB10_PLE_MMAP_WRITABLE")}

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        g._device = self.saved_device
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        g._TABLES.clear()

    def emulate_kernel(self, bases, rows, flat_ids, output, *, embedding_dim, shard_size,
                       num_shards, tp_vocab_start, tp_vocab_end, is_fp8, block_d):
        """The kernel's arithmetic, row by row, reading the same addresses."""
        import ctypes

        self.kernel_calls.append(dict(shard_size=shard_size, num_shards=num_shards, is_fp8=is_fp8))
        width = embedding_dim * (1 if is_fp8 else 2)
        for i, gid in enumerate(flat_ids.tolist()):
            in_tp = tp_vocab_start <= gid < tp_vocab_end
            gid = gid if in_tp else 0
            k = min(gid // shard_size, num_shards - 1)
            local = gid - k * shard_size
            if not (in_tp and local < int(rows[k])):
                output.view(-1, embedding_dim)[i] = 0
                continue
            raw = ctypes.string_at(int(bases[k]) + local * width, width)
            row = torch.frombuffer(bytearray(raw), dtype=torch.float8_e4m3fn if is_fp8 else torch.bfloat16)
            output.view(-1, embedding_dim)[i] = row.to(torch.bfloat16)

    def build(self, d, vocab=1000, parts=8, dim=12, drop=None, backend="file"):
        import types

        ck = Checkpoint(d, vocab=vocab, parts=parts, dim=dim, drop=drop)
        allocations = []
        mod = fake_sglang_module(allocations)
        g.apply(mod)
        os.environ[g.ENV_TABLE_DIR] = d
        cfg = types.SimpleNamespace(vocab=vocab, dim=dim, split_ngram_parts=parts,
                                    ple_offload_embedding=True, ple_offload_backend=backend)
        model = mod.Qwen4ExpForConditionalGeneration(cfg)
        return ck, allocations, model, model.ple.ple_embedding.ngram_embedding

    def test_file_backend_is_read_in_place(self):
        with tempfile.TemporaryDirectory() as d:
            ck, allocations, model, emb = self.build(d, drop=3)
            self.assertEqual(allocations, [])  # the stock allocator never ran
            self.assertEqual(tuple(emb.weight.shape), (ck.vocab, ck.dim))
            self.assertEqual(emb.weight.untyped_storage().nbytes(), 1)
            self.assertEqual(emb.weight.weight_loader, "stock")  # attributes carried over
            table = emb._gb10_table
            np.testing.assert_array_equal(emb._gb10_dev["bases"].numpy(), table.bases)
            self.assertEqual(len(self.trimmers), 1)
            self.assertEqual((self.trimmers[0]["addr"], self.trimmers[0]["nbytes"]), (table.addr, table.nbytes))

            ids = torch.tensor([[0, 5, ck.shard_size * 3 + 1], [ck.vocab - 1, ck.vocab + 7, 42]])
            out = emb.gather(ids)
            self.assertEqual(tuple(out.shape), (2, 3, ck.dim))
            self.assertEqual(self.kernel_calls[-1], dict(shard_size=ck.shard_size, num_shards=ck.parts, is_fp8=True))
            ref = torch.frombuffer(bytearray(ck.table.tobytes()), dtype=torch.float8_e4m3fn)
            ref = ref.view(ck.vocab, ck.dim).to(torch.bfloat16)
            for (r, c), gid in np.ndenumerate(ids.numpy()):
                want = ref[gid] if gid < ck.vocab and ck.present[gid] else torch.zeros(ck.dim, dtype=torch.bfloat16)
                torch.testing.assert_close(out[r, c], want, rtol=0, atol=0, equal_nan=True, msg=f"id {gid}")

            weights = [(f"{PREFIX.format(1)}.shard_{k}.weight", None) for k in range(3)] + [
                (f"{PREFIX.format(1)}.weight_scale", None),
                ("mtp.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight", None),
                ("model.language_model.layers.0.mlp.gate.weight", None)]
            model.load_weights(iter(weights))
            self.assertEqual(model.seen, [n for n, _ in weights[3:]])

    def test_other_backends_stay_stock(self):
        with tempfile.TemporaryDirectory() as d:
            _, allocations, model, emb = self.build(d, backend="pinned")
            self.assertEqual(allocations, ["pinned"])
            self.assertFalse(hasattr(emb, "_gb10_table"))
            model.load_weights(iter([(f"{PREFIX.format(1)}.shard_0.weight", None)]))
            self.assertEqual(len(model.seen), 1)  # shards reach the stock loader

    def test_refuses_a_version_without_the_pieces(self):
        import types

        with self.assertRaisesRegex(RuntimeError, "0.5.20"):
            g.apply(types.ModuleType("empty"))


def kernel_selfcheck():
    """Run in a child with TRITON_INTERPRET=1 (it must be set before triton
    is imported). Prints the number of rows that differ from the reference."""
    from gb10_ple_kernel import gather_rows

    bad = 0
    for dtype, dim in (("F8_E4M3", 12), ("BF16", 6)):
        with tempfile.TemporaryDirectory() as d:
            ck = Checkpoint(d, vocab=997, parts=7, dim=dim, dtype=dtype, drop=2)
            t = g.map_table(g.scan_table_dir(d)[1], d, ck.shard_size, ck.vocab)
            tdt = torch.float8_e4m3fn if dtype == "F8_E4M3" else torch.bfloat16
            ref = torch.frombuffer(bytearray(ck.table.tobytes()), dtype=tdt).view(ck.vocab, dim)
            ref = ref.to(torch.bfloat16)
            # 0x7f / 0xff are e4m3fn NaN; the interpreter converts them to
            # +-480 (so does nothing on the GPU path: a real table has no NaN).
            nan_bytes = ck.table.reshape(ck.vocab, -1)
            ids = torch.tensor(list(range(ck.vocab)) + [ck.vocab + 3, 5000], dtype=torch.long)
            for lo, hi in ((0, ck.vocab), (100, 600)):  # whole table; one TP rank
                out = torch.empty(len(ids), dim, dtype=torch.bfloat16)
                gather_rows(torch.from_numpy(t.bases), torch.from_numpy(t.rows), ids, out,
                            embedding_dim=dim, shard_size=t.shard_size, num_shards=t.num_shards,
                            tp_vocab_start=lo, tp_vocab_end=hi, is_fp8=dtype == "F8_E4M3",
                            block_d=1 << (dim - 1).bit_length())
                for i, gid in enumerate(ids.tolist()):
                    ok = lo <= gid < hi and gid < ck.vocab and ck.present[gid]
                    want = ref[gid] if ok else torch.zeros(dim, dtype=torch.bfloat16)
                    keep = torch.ones(dim, dtype=torch.bool)
                    if ok and dtype == "F8_E4M3":
                        keep = torch.from_numpy((nan_bytes[gid] & 0x7F) != 0x7F)
                    bad += int(not torch.allclose(out[i][keep], want[keep], rtol=0, atol=0, equal_nan=True))
    print(f"mismatched rows: {bad}")


@unittest.skipIf(torch is None, "torch not installed")
class KernelTest(unittest.TestCase):
    def test_kernel_reads_the_mapped_shards(self):
        try:
            import triton  # noqa: F401
        except ImportError:
            self.skipTest("triton not installed")
        import subprocess

        env = dict(os.environ, TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--kernel-selfcheck"],
                           env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("mismatched rows: 0", r.stdout)


if __name__ == "__main__":
    if sys.argv[1:] == ["--kernel-selfcheck"]:
        kernel_selfcheck()
        sys.exit(0)
    unittest.main(verbosity=2)
