"""Serve the Qwen3.8-Flash-Next n-gram (PLE) table straight from its
safetensors shards: read-only mmap, nothing written to disk.

Stock SGLang 0.5.20 has two places for the 47.7 GiB table: pinned host memory
(which on GB10 comes out of the same 128 GB as the weights, so the model does
not fit) and ``--ple-offload-backend file``: a sparse file under
$SGLANG_CACHE_DIR/ple that the weight loader fills on every boot. That is a
second 47.7 GiB copy of data already on disk, written ~10 minutes on each
boot into a fresh file and ~55 into a filled one (the cookbook's advice is to
delete it before every start).

The table is already on disk in exactly the layout the gather kernel needs:
``...ple.ple_embedding.ngram_embedding.shard_<k>.weight`` tensors, row-major,
in the checkpoint's safetensors files. This patch maps those files read-only
and points the gather kernel at the shards where they are:

  * ``allocate_ple_host_table`` (file backend) returns a placeholder instead
    of creating the sparse file;
  * each PLE layer gets a table of per-shard base addresses (shard k holds
    rows [k * shard_size, k * shard_size + rows_k), the loader's own layout);
  * ``Qwen4ExpPinnedHostEmbedding.gather`` runs a kernel that picks the shard
    by row, like the stock one reads a single base pointer;
  * the loader skips the shard tensors (the table is already where it will be
    read from).

What stays stock: the file backend's device check (the GPU must read pageable
memory through the host page tables, as GB10 does), MADV_RANDOM, the
page-cache hints before prefill-sized gathers and the resident-set trimmer
(SGLANG_QWEN4_PLE_FILE_* settings apply unchanged).

Enabled by GB10_PLE_MMAP=1 (models/qwen3.8-flash-next.sh sets it) together
with --ple-offload-embedding --ple-offload-backend file; the table is read
from GB10_PLE_TABLE_DIR (default: the model directory). sitecustomize.py in
this directory installs the import hook that applies it when SGLang imports
the model. Written against SGLang 0.5.20; a version whose code no longer has
the pieces above fails at load time with a message, instead of silently
falling back to writing the table.

This module imports neither torch nor triton at import time (it is imported
by every Python process started with this directory on PYTHONPATH), and its
file layout and address arithmetic run without them: see
test_gb10_ple_mmap.py.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.abc
import json
import logging
import os
import re
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_ple_mmap")

ENV_ENABLE = "GB10_PLE_MMAP"
ENV_TABLE_DIR = "GB10_PLE_TABLE_DIR"
TARGET_MODULE = "sglang.srt.models.qwen4_exp"

SHARD_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
)
DTYPE_BYTES = {"F8_E4M3": 1, "BF16": 2}

# mmap(2) / madvise(2) constants, identical on x86_64 and aarch64 Linux.
_PROT_NONE, _PROT_READ = 0x0, 0x1
_MAP_SHARED, _MAP_PRIVATE, _MAP_FIXED = 0x01, 0x02, 0x10
_MAP_ANONYMOUS, _MAP_NORESERVE = 0x20, 0x4000
_MADV_RANDOM = 1
_MAP_FAILED = ctypes.c_void_p(-1).value

# Below this many rows a gather is decode-sized: the faults are cheap and the
# page-cache hint would cost more than it saves (SGLang's own threshold).
PREFETCH_MIN_ROWS = 2048


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE) == "1"


# ---- checkpoint layout -------------------------------------------------------


@dataclass
class ShardInfo:
    index: int  # k in shard_<k>
    path: str
    file_offset: int  # absolute offset of the shard's first row in `path`
    rows: int
    dim: int
    dtype: str  # safetensors dtype string


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """(tensor entries, absolute offset where tensor data starts)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def scan_table_dir(table_dir: str) -> dict[int, dict[int, ShardInfo]]:
    """{checkpoint layer id: {shard index: ShardInfo}} over *.safetensors."""
    layers: dict[int, dict[int, ShardInfo]] = {}
    names = sorted(n for n in os.listdir(table_dir) if n.endswith(".safetensors"))
    for name in names:
        path = os.path.join(table_dir, name)
        header, data_start = read_safetensors_header(path)
        for tensor, meta in header.items():
            if "mtp" in tensor:
                continue
            m = SHARD_RE.search(tensor)
            if m is None:
                continue
            layer, shard = int(m.group(1)), int(m.group(2))
            shape = meta["shape"]
            if len(shape) != 2:
                raise ValueError(f"{path}: {tensor} has shape {shape}, expected 2-D")
            if meta["dtype"] not in DTYPE_BYTES:
                raise ValueError(
                    f"{path}: {tensor} is {meta['dtype']}; the gather kernel reads "
                    "F8_E4M3 or BF16"
                )
            begin, end = meta["data_offsets"]
            rows, dim = int(shape[0]), int(shape[1])
            if end - begin != rows * dim * DTYPE_BYTES[meta["dtype"]]:
                raise ValueError(f"{path}: {tensor} size does not match its shape")
            if shard in layers.setdefault(layer, {}):
                raise ValueError(f"layer {layer} shard {shard} appears twice ({path})")
            layers[layer][shard] = ShardInfo(
                shard, path, data_start + begin, rows, dim, meta["dtype"]
            )
    return layers


# ---- the mapping -------------------------------------------------------------


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_long,
    ]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return libc


@dataclass
class MappedTable:
    """One PLE layer's shards, mapped read-only in one contiguous address range.

    Every file that holds a shard is mapped whole, back to back inside a
    single reserved range, so the resident-set trimmer can watch one
    [addr, addr + nbytes) span. Headers and unrelated tensors in the same
    files are mapped too but never touched, so they never become resident.
    """

    table_dir: str
    shard_size: int  # rows per shard slot, the loader's ceil(vocab / parts)
    dim: int
    dtype: str
    bases: np.ndarray  # int64 address of row 0 per shard slot, 0 = absent
    rows: np.ndarray  # int64 rows per shard slot, 0 = absent
    shard_file: np.ndarray  # int64 index into `paths` per slot, -1 = absent
    shard_file_offset: np.ndarray  # int64 file offset of row 0 per slot
    paths: list[str]
    addr: int  # the whole span, for the resident-set trimmer
    nbytes: int
    mapped_rows: int
    table_bytes: int
    _prefetcher: Optional["Prefetcher"] = field(default=None, repr=False)

    @property
    def row_bytes(self) -> int:
        return self.dim * DTYPE_BYTES[self.dtype]

    @property
    def num_shards(self) -> int:
        return len(self.bases)

    def row_address(self, row: int) -> int:
        """Address of global row `row` (0 if the checkpoint has no such row)."""
        k = row // self.shard_size
        local = row - k * self.shard_size
        if k >= self.num_shards or local >= self.rows[k]:
            return 0
        return int(self.bases[k]) + local * self.row_bytes


def map_table(
    shards: dict[int, ShardInfo],
    table_dir: str,
    shard_size: int,
    vocab_rows: int,
    writable: bool = False,
) -> MappedTable:
    """Map the files holding `shards` and lay the shards out like the loader.

    The loader copies shard k to rows [k * shard_size, ...) of a vocab_rows
    table (qwen4_exp.load_weights); rows no shard covers stay zero in its
    sparse file. The kernel reproduces both: slots with no shard, and rows
    past a short shard, read as zero.

    writable=True maps copy-on-write (PROT_READ|PROT_WRITE, MAP_PRIVATE,
    MAP_NORESERVE) instead of PROT_READ/MAP_SHARED, for a device that turns
    out to need writable page-table entries. Nothing is ever written through
    it, and a private mapping could not reach the file anyway.
    """
    if not shards:
        raise ValueError(f"no PLE table shards in {table_dir}")
    dims = {s.dim for s in shards.values()}
    dtypes = {s.dtype for s in shards.values()}
    if len(dims) != 1 or len(dtypes) != 1:
        raise ValueError(f"PLE shards in {table_dir} disagree on dim/dtype: {dims} {dtypes}")
    dim, dtype = dims.pop(), dtypes.pop()
    num_slots = -(-vocab_rows // shard_size)
    for s in shards.values():
        if s.index >= num_slots:
            raise ValueError(
                f"PLE shard {s.index} ({s.path}) is past the table's {num_slots} "
                f"slots of {shard_size} rows: split_ngram_parts differs from the "
                "checkpoint's"
            )
        if s.rows > shard_size:
            raise ValueError(
                f"PLE shard {s.index} has {s.rows} rows, more than the loader's "
                f"{shard_size} per shard: split_ngram_parts differs from the checkpoint's"
            )

    page = os.sysconf("SC_PAGE_SIZE")
    paths = sorted({s.path for s in shards.values()})
    sizes = [os.path.getsize(p) for p in paths]
    slots = [(sz + page - 1) // page * page for sz in sizes]
    total = sum(slots)

    libc = _libc()
    base = libc.mmap(
        None, total, _PROT_NONE, _MAP_PRIVATE | _MAP_ANONYMOUS | _MAP_NORESERVE, -1, 0
    )
    if base in (None, _MAP_FAILED):
        raise OSError(ctypes.get_errno(), f"reserving {total} bytes for the PLE table")
    prot = _PROT_READ | (0x2 if writable else 0)
    flags = (_MAP_PRIVATE | _MAP_NORESERVE if writable else _MAP_SHARED) | _MAP_FIXED
    file_base = {}
    cursor = base
    for path, size, slot in zip(paths, sizes, slots):
        fd = os.open(path, os.O_RDONLY)
        try:
            got = libc.mmap(cursor, size, prot, flags, fd, 0)
        finally:
            os.close(fd)  # the mapping keeps the file referenced
        if got != cursor:
            raise OSError(ctypes.get_errno(), f"mapping {path} for the PLE table")
        file_base[path] = cursor
        cursor += slot
    # Pure random access (16 rows per token): no readahead around each fault.
    if libc.madvise(base, total, _MADV_RANDOM) != 0:
        logger.warning("PLE table: madvise(MADV_RANDOM) failed (errno %d)", ctypes.get_errno())

    bases = np.zeros(num_slots, dtype=np.int64)
    rows = np.zeros(num_slots, dtype=np.int64)
    shard_file = np.full(num_slots, -1, dtype=np.int64)
    shard_off = np.zeros(num_slots, dtype=np.int64)
    for s in shards.values():
        bases[s.index] = file_base[s.path] + s.file_offset
        rows[s.index] = s.rows
        shard_file[s.index] = paths.index(s.path)
        shard_off[s.index] = s.file_offset
    return MappedTable(
        table_dir=table_dir,
        shard_size=shard_size,
        dim=dim,
        dtype=dtype,
        bases=bases,
        rows=rows,
        shard_file=shard_file,
        shard_file_offset=shard_off,
        paths=paths,
        addr=base,
        nbytes=total,
        mapped_rows=int(rows.sum()),
        table_bytes=int(rows.sum()) * dim * DTYPE_BYTES[dtype],
    )


# ---- page-cache hints for prefill-sized gathers ------------------------------


def pages_for_rows(table: MappedTable, ids: np.ndarray, page: int) -> dict[int, np.ndarray]:
    """{file index: sorted unique page numbers} the rows `ids` live on."""
    ids = np.asarray(ids, dtype=np.int64)
    k = ids // table.shard_size
    ok = k < table.num_shards
    k, ids = k[ok], ids[ok]
    local = ids - k * table.shard_size
    ok = local < table.rows[k]
    k, local = k[ok], local[ok]
    start = table.shard_file_offset[k] + local * table.row_bytes
    end = start + table.row_bytes - 1
    files = table.shard_file[k]
    out = {}
    for f in np.unique(files):
        sel = files == f
        out[int(f)] = np.unique(np.concatenate([start[sel] // page, end[sel] // page]))
    return out


class Prefetcher:
    """posix_fadvise(WILLNEED) the pages a prefill-sized gather will fault.

    The same idea as SGLang's PleFilePrefetcher (which only knows its single
    table file): a cold prefill chunk faults tens of thousands of pages one at
    a time from inside the kernel; hinting them first lets the block layer
    serve them concurrently. One background thread; decode-sized gathers and
    CUDA-graph capture are skipped.
    """

    def __init__(self, table: MappedTable, min_rows: int = PREFETCH_MIN_ROWS) -> None:
        self._table = table
        self._fds = [os.open(p, os.O_RDONLY) for p in table.paths]
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._min_rows = min_rows
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-prefetch")

    def _advise(self, ids: np.ndarray) -> None:
        for f, pages in pages_for_rows(self._table, ids, self._page).items():
            fd = self._fds[f]
            for p in pages.tolist():
                try:
                    os.posix_fadvise(fd, p * self._page, self._page, os.POSIX_FADV_WILLNEED)
                except OSError:
                    return

    def enqueue(self, flat_ids) -> bool:
        if flat_ids.numel() < self._min_rows:
            return False
        import torch

        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # .cpu() syncs the stream, as in SGLang's own prefetcher: acceptable
        # for prefill chunks, which is all that reaches this line.
        ids = flat_ids.cpu().numpy()
        self._pool.submit(self._advise, ids)
        return True


# ---- the SGLang side -----------------------------------------------------------


# Placeholders handed out, by storage address. The embedding wraps the tensor
# in a new nn.Parameter (attributes set on it do not carry over), which shares
# its storage; the list keeps them alive so an address is never reused.
_PLACEHOLDERS: list = []


def _placeholder(torch, shape, dtype):
    """A full-shaped CPU tensor over one element: what the embedding module
    wraps as its `weight` (never read: gather is replaced)."""
    t = torch.empty(1, dtype=dtype).view(*([1] * len(shape))).expand(*[int(d) for d in shape])
    _PLACEHOLDERS.append(t)
    return t


def _is_placeholder(tensor) -> bool:
    return any(tensor.data_ptr() == t.data_ptr() for t in _PLACEHOLDERS)


_TABLES: dict[str, dict[int, dict[int, ShardInfo]]] = {}
_TABLES_LOCK = threading.Lock()


def _scan_cached(table_dir: str) -> dict[int, dict[int, ShardInfo]]:
    with _TABLES_LOCK:
        if table_dir not in _TABLES:
            _TABLES[table_dir] = scan_table_dir(table_dir)
        return _TABLES[table_dir]


def apply(mod) -> None:
    """Patch sglang.srt.models.qwen4_exp (the module object) in place."""
    import inspect

    import torch

    missing = [
        n for n in ("allocate_ple_host_table", "Qwen4ExpPinnedHostEmbedding",
                    "Qwen4ExpPLELayer", "Qwen4ExpForConditionalGeneration")
        if not hasattr(mod, n)
    ]
    if missing or not hasattr(mod.Qwen4ExpPinnedHostEmbedding, "gather"):
        raise RuntimeError(
            f"GB10_PLE_MMAP: {TARGET_MODULE} lacks {missing or ['gather']}; this "
            "patch was written for SGLang 0.5.20. Unset GB10_PLE_MMAP (profile: "
            "PLE_TABLE=file) to use the stock file backend."
        )
    Pinned = mod.Qwen4ExpPinnedHostEmbedding
    PLELayer = mod.Qwen4ExpPLELayer
    Model = mod.Qwen4ExpForConditionalGeneration

    orig_alloc = mod.allocate_ple_host_table

    def allocate_ple_host_table(shape, dtype, backend="pinned", table_dir=None, tag=None):
        if backend != "file":
            return orig_alloc(shape, dtype, backend=backend, table_dir=table_dir, tag=tag)
        return _placeholder(torch, shape, dtype)

    mod.allocate_ple_host_table = allocate_ple_host_table

    orig_layer_init = PLELayer.__init__
    layer_sig = inspect.signature(orig_layer_init)

    def layer_init(self, *args, **kwargs):
        orig_layer_init(self, *args, **kwargs)
        emb = self.ple_embedding.ngram_embedding
        if not isinstance(emb, Pinned) or not _is_placeholder(emb.weight):
            return
        bound = layer_sig.bind(self, *args, **kwargs)
        config = bound.arguments["config"]
        layer_id = bound.arguments.get("layer_id")
        _attach_table(torch, emb, config, layer_id)

    PLELayer.__init__ = layer_init

    orig_gather = Pinned.gather

    def gather(self, input_ids, out=None):
        dev = getattr(self, "_gb10_dev", None)
        if dev is None:
            return orig_gather(self, input_ids, out)
        from gb10_ple_kernel import gather_rows

        expected = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected, input_ids.device)
        else:
            if tuple(out.shape) != expected:
                raise ValueError(f"invalid PLE prefetch output shape: {tuple(out.shape)} != {expected}")
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError("PLE prefetch output must be bfloat16 on the id device")
            output = out
        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel():
            table = self._gb10_table
            if table._prefetcher is not None:
                table._prefetcher.enqueue(flat_ids)
            gather_rows(
                dev["bases"], dev["rows"], flat_ids, output,
                embedding_dim=self.embedding_dim,
                shard_size=table.shard_size,
                num_shards=table.num_shards,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=table.dtype == "F8_E4M3",
                block_d=self._block_d,
            )
        return output

    Pinned.gather = gather

    orig_load = Model.load_weights

    def load_weights(self, weights):
        if not any(getattr(m, "_gb10_dev", None) is not None for m in self.modules()):
            return orig_load(self, weights)
        skipped = [0]

        def without_table():
            for name, tensor in weights:
                if ".ngram_embedding.shard_" in name and "mtp" not in name:
                    skipped[0] += 1
                    continue
                yield name, tensor

        result = orig_load(self, without_table())
        logger.info("PLE table: %d shard tensors left in place (read through the mapping)", skipped[0])
        return result

    Model.load_weights = load_weights
    logger.info("GB10_PLE_MMAP: %s patched (PLE table read in place)", TARGET_MODULE)


def _device(torch):
    """Where the shard address table lives: the GPU the gather runs on."""
    return torch.device("cuda", torch.cuda.current_device())


def _attach_table(torch, emb, config, layer_id) -> None:
    table_dir = os.environ.get(ENV_TABLE_DIR) or ""
    if not table_dir:
        raise RuntimeError(f"GB10_PLE_MMAP=1 needs {ENV_TABLE_DIR} (the directory holding the table shards)")
    layers = _scan_cached(table_dir)
    if layer_id in layers:
        shards = layers[layer_id]
    elif len(layers) == 1:
        (only,) = layers.values()
        shards = only
    else:
        raise RuntimeError(
            f"no PLE shards for layer {layer_id} in {table_dir} "
            f"(found layers {sorted(layers) or 'none'})"
        )
    # The loader's layout: shard k -> rows [k * shard_size, ...), with
    # shard_size = ceil(vocab / split_ngram_parts) (qwen4_exp.load_weights).
    parts = int(getattr(config, "split_ngram_parts", 512))
    vocab_rows = int(emb.org_vocab_size)
    shard_size = -(-vocab_rows // parts)
    table = map_table(
        shards, table_dir, shard_size, vocab_rows,
        writable=os.environ.get("GB10_PLE_MMAP_WRITABLE") == "1",
    )
    want = {torch.float8_e4m3fn: "F8_E4M3", torch.bfloat16: "BF16"}.get(emb.weight.dtype)
    if table.dtype != want:
        raise RuntimeError(
            f"PLE table in {table_dir} is {table.dtype}, the model expects {want}; "
            'set text_config.ple_embedding_dtype="float8_e4m3fn" for an fp8 table'
        )
    if table.dim != emb.embedding_dim:
        raise RuntimeError(f"PLE table rows are {table.dim} wide, the model expects {emb.embedding_dim}")

    from sglang.srt.environ import envs

    if envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        table._prefetcher = Prefetcher(table)
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb > 0:
        from sglang.srt.models.qwen4_exp_ple_table import PleFileRssTrimmer

        trimmer = PleFileRssTrimmer(
            addr=table.addr,
            nbytes=table.nbytes,
            budget_bytes=int(budget_gb * 2**30),
            interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
        )
        trimmer.start()
        emb._file_rss_trimmer = trimmer
    device = _device(torch)
    # Plain attributes, not buffers: invisible to the loader and state dicts.
    emb._gb10_table = table
    emb._gb10_dev = {
        "bases": torch.from_numpy(table.bases).to(device),
        "rows": torch.from_numpy(table.rows).to(device),
    }
    logger.info(
        "PLE table: layer %s read in place from %s: %d of %d rows in %d shards "
        "(%.1f GiB %s) across %d files, %s mapping, nothing written",
        layer_id, table_dir, table.mapped_rows, vocab_rows,
        int((table.rows > 0).sum()), table.table_bytes / 2**30, table.dtype,
        len(table.paths), "copy-on-write" if os.environ.get("GB10_PLE_MMAP_WRITABLE") == "1" else "read-only",
    )
    if table.mapped_rows < vocab_rows:
        logger.info(
            "PLE table: %d rows past the checkpoint's shards read as zero, as in "
            "the stock file backend", vocab_rows - table.mapped_rows,
        )


# ---- import hook ---------------------------------------------------------------


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, hook):
        self._inner = inner
        self._hook = hook

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._hook(module)

    def __getattr__(self, name):  # get_source, get_filename, ... (triton reads sources)
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target: str, hook):
        self._target = target
        self._hook = hook

    def find_spec(self, fullname, path, target=None):
        if fullname != self._target:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self._hook)
        return spec


def install_import_hook(target: str = TARGET_MODULE, hook=apply) -> None:
    """Run `hook(module)` right after `target` is imported, in this process
    (and, through sitecustomize, in every process SGLang spawns)."""
    if target in sys.modules:
        hook(sys.modules[target])
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(target, hook))
