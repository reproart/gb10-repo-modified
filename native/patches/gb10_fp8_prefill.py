"""FP8 W8A8 for prefill-sized inputs of the layers gb10_fp8_side put on FP8 Marlin.

Marlin is a weight-only kernel made for a few rows: it converts each FP8 weight
to BF16 inside the GEMM and runs BF16 tensor-core math, ~78 TFLOPS at best on
GB10, and as low as ~47 for the K=6144 projections (patches/bench_sglang_gb10_gpu.py,
section 4). Over 512 rows SGLang's own FP8 GEMM (sgl_kernel fp8_scaled_mm,
CUTLASS on GB10's FP8 tensor cores, with one activation scale per token from
sgl_per_token_quant_fp8) runs the same layers at 105-160 TFLOPS: a 4096-token
chunk's side layers took 186 ms that way against 364 on Marlin.

The weights exist only in Marlin's layout (keeping a second, plain copy would
cost ~2.8 GiB of KV pool). Marlin's FP8 repack only moves bytes (the exponent
bias goes into the scales, not the weights), so each call restores the plain
[N, K] FP8 weight with one gather pass (a Triton kernel; ~0.35 ms for the
largest layer) and runs fp8_scaled_mm on it. The byte order is not hard-coded:
setup() learns it once at load from SGLang's own prepare_fp8_layer_for_marlin
on probe weights whose bytes encode their position, checks the tile structure
the gather relies on, round-trips random probes of other shapes, and checks one
fp8_scaled_mm against a reference. Anything off and the path stays disabled
(logged), every layer staying on Marlin.

Used for layers with at least MIN_WEIGHT elements (GDN in_proj_qkvz/out_proj,
attention qkv/o; the small ones gain little) and inputs of at least
GB10_FP8_SIDE_PREFILL_ROWS rows (default 1024; a decode verify is at most 32).
Inputs over CHUNK_ROWS run in slices of that many rows: one 8192-row GEMM ran
at 97 TFLOPS against 160 for 4096 rows in the bench.

Lossy beyond the weight-only path: the activations go to FP8 per token
(relative error ~3e-2 per GEMM output against ~3e-3 for weight-only), so a
prompt's KV and GDN state are computed with them; decode is unchanged.
Enabled by GB10_FP8_SIDE_PREFILL=w8a8 (models/qwen3.8-flash-next.sh:
FP8_SIDE_PREFILL=w8a8). Written against SGLang 0.5.21.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("sglang.srt.models.qwen4_exp.gb10_fp8_prefill")

MIN_WEIGHT = 1 << 22    # N * K: GDN in_proj_qkvz / out_proj, attention qkv / o
CHUNK_ROWS = 4096
TILE_K, TILE_N, BLOCK_BYTES = 16, 64, 1024   # Marlin's 16 x 64 tile, one 1024-byte block

_state = {"ready": None, "inv": None, "why": "", "runtime_ok": True, "used": 0}
_ops = {}   # fp8_scaled_mm / per-token quant, filled on first use (tests put fakes here)


def mode() -> str:
    m = os.environ.get("GB10_FP8_SIDE_PREFILL", "marlin")
    if m not in ("marlin", "w8a8"):
        raise RuntimeError(f"GB10_FP8_SIDE_PREFILL must be marlin or w8a8, not {m!r}")
    return m


def min_rows() -> int:
    return int(os.environ.get("GB10_FP8_SIDE_PREFILL_ROWS", "1024"))


# ---- Marlin's byte order -----------------------------------------------------------

def _q_index(n, k):
    """Position of weight element (n, k) inside its 1024-byte Marlin block, before
    the block's own permutation: (n // 16 % 4, k % 16, n % 16)."""
    return ((n // 16) % 4) * 256 + (k % 16) * 16 + (n % 16)


def _block_base(n, k, size_n):
    """Byte offset of the 1024-byte block holding (n, k): blocks run along N
    (64 columns each) within a 16-row K tile."""
    return (k // TILE_K) * (TILE_K * size_n) + (n // TILE_N) * BLOCK_BYTES


def learn_layout(marlin_bytes_of, size_n=128, size_k=128):
    """marlin_bytes_of(weight uint8 [N, K]) -> the Marlin-packed bytes (flat uint8),
    produced by the real pipeline. Returns the 1024-entry table: inv[q] = the byte
    offset inside a block of the element whose _q_index is q. Raises ValueError if
    the layout does not have the tile structure the gather assumes."""
    import torch

    idx = torch.arange(size_n * size_k, dtype=torch.int64).view(size_n, size_k)
    lo = marlin_bytes_of((idx & 0xFF).to(torch.uint8)).long().cpu()
    hi = marlin_bytes_of((idx >> 8).to(torch.uint8)).long().cpu()
    if lo.numel() != size_n * size_k or hi.numel() != lo.numel():
        raise ValueError(f"packed size {lo.numel()}, expected {size_n * size_k}")
    src = lo | (hi << 8)                       # which element sits at each byte
    if not torch.equal(torch.sort(src).values, torch.arange(src.numel())):
        raise ValueError("the packed bytes are not a permutation of the weight")
    pos = torch.arange(src.numel())
    n, k = src // size_k, src % size_k
    off = pos - _block_base(n, k, size_n)
    if bool(((off < 0) | (off >= BLOCK_BYTES)).any()):
        raise ValueError("an element landed outside its 16 x 64 tile's block")
    q = _q_index(n, k)
    inv = torch.full((BLOCK_BYTES,), -1, dtype=torch.int64)
    inv[q] = off
    if bool((inv < 0).any()) or not torch.equal(inv[q], off):
        raise ValueError("the blocks are not permuted alike")
    return inv.to(torch.int32)


def unpack_torch(packed, size_n, size_k, inv):
    """Marlin-packed FP8 weight (any dtype, contiguous) -> plain [N, K] uint8,
    in torch: the reference for the Triton gather."""
    import torch

    b = packed.contiguous().view(torch.uint8).view(size_k // TILE_K, size_n // TILE_N, BLOCK_BYTES)
    b = b.index_select(2, inv.to(device=b.device, dtype=torch.int64))   # q order
    b = b.view(size_k // TILE_K, size_n // TILE_N, 4, TILE_K, 16)       # (kt, nb, n4, kk, nn)
    return b.permute(1, 2, 4, 0, 3).reshape(size_n, size_k).contiguous()


def unpack(packed, size_n, size_k, inv):
    """The same with one Triton pass on CUDA (torch elsewhere)."""
    if not packed.is_cuda and os.environ.get("TRITON_INTERPRET") != "1":
        return unpack_torch(packed, size_n, size_k, inv)
    from gb10_fp8_prefill_kernel import marlin_fp8_unpack

    return marlin_fp8_unpack(packed, size_n, size_k, inv)


# ---- setup at load --------------------------------------------------------------------

def _marlin_bytes_via(prepare_fn, device):
    """The pipeline gb10_fp8_side.convert_module runs (prepare_fp8_layer_for_marlin
    with size_k_first=False), on a probe module."""
    import torch

    def bytes_of(w_u8):
        n, k = w_u8.shape
        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(w_u8.to(device).view(torch.float8_e4m3fn), requires_grad=False)
        m.weight_scale = torch.nn.Parameter(torch.ones(n, device=device, dtype=torch.bfloat16),
                                            requires_grad=False)
        m.orig_dtype = torch.bfloat16
        m.output_size_per_partition = n
        m.input_size_per_partition = k
        prepare_fn(m, size_k_first=False)
        return m.weight.data.contiguous().view(torch.uint8).flatten()
    return bytes_of


def _load_ops():
    if not _ops:
        from sglang.kernels.ops.gemm import fp8_scaled_mm
        from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

        _ops["mm"] = fp8_scaled_mm
        _ops["quant"] = sglang_per_token_quant_fp8
    return _ops


def setup(prepare_fn, device) -> bool:
    """Learn and check Marlin's byte order and the FP8 GEMM, once. True if the
    W8A8 prefill path can be used."""
    import torch

    if _state["ready"] is not None:
        return _state["ready"]
    try:
        bytes_of = _marlin_bytes_via(prepare_fn, device)
        inv = learn_layout(bytes_of)
        inv_dev = inv.to(device)
        g = torch.Generator().manual_seed(0)
        for n, k in ((192, 384), (1280, 2560)):
            w = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
            packed = bytes_of(w)
            if not torch.equal(unpack(packed, n, k, inv_dev).cpu(), w):
                raise ValueError(f"a random [{n} x {k}] weight does not round-trip")
        ops = _load_ops()
        x = torch.randn(256, 512, generator=g).to(device=device, dtype=torch.bfloat16)
        wf = torch.randn(128, 512, generator=g) * 0.02
        s = wf.abs().amax(dim=1).clamp_min(1e-12) / 448.0
        wq = (wf / s[:, None]).to(torch.float8_e4m3fn).to(device)
        xq, xs = ops["quant"](x)
        out = ops["mm"](xq, wq.t(), xs, s.to(device).view(-1, 1), torch.bfloat16)
        ref = x.float() @ (wq.float() * s.to(device)[:, None]).T
        err = ((out.float() - ref).norm() / ref.norm()).item()
        if not err < 0.08:
            raise ValueError(f"fp8_scaled_mm is off by {err:.3f} against FP32")
        _state.update(ready=True, inv=inv_dev)
        logger.info("FP8 side prefill: W8A8 for inputs of %d+ rows (Marlin byte order learned and "
                    "checked; fp8_scaled_mm rel %.3f)", min_rows(), err)
    except Exception as e:  # noqa: BLE001 - stay on Marlin
        _state.update(ready=False, why=f"{type(e).__name__}: {e}")
        logger.warning("FP8 side prefill: W8A8 path disabled, every layer stays on Marlin (%s)",
                       _state["why"])
    return _state["ready"]


def mark(module, prepare_fn) -> bool:
    """Called for each layer gb10_fp8_side converted: enables the W8A8 prefill path
    on it if the mode asks for it, the layer is big enough and setup() passed."""
    import torch

    n_pad, k = module._gb10_marlin_n, module.input_size_per_partition
    if mode() != "w8a8" or n_pad * k < MIN_WEIGHT or getattr(module, "bias", None) is not None:
        return False
    if not setup(prepare_fn, module.weight.device):
        return False
    module._gb10_w8a8 = True
    module._gb10_wscale_col = module._gb10_wscale.to(torch.float32).view(-1, 1).contiguous()
    return True


# ---- the forward ----------------------------------------------------------------------

def wants(layer, x) -> bool:
    return (getattr(layer, "_gb10_w8a8", False) and _state["runtime_ok"]
            and x.numel() >= min_rows() * x.shape[-1])


def apply_w8a8(layer, x, bias=None):
    """layer: a converted module (Marlin weight, _gb10_* attributes). Returns
    [..., n_pad] in x's dtype, or None if the path failed (logged once; the
    caller then runs Marlin)."""
    import torch

    try:
        ops = _load_ops()
        k = x.shape[-1]
        n_pad = layer._gb10_marlin_n
        w = unpack(layer.weight, n_pad, k, _state["inv"]).view(torch.float8_e4m3fn)
        x2 = x.reshape(-1, k)
        rows = x2.shape[0]
        outs = []
        for i in range(0, rows, CHUNK_ROWS):
            xq, xs = ops["quant"](x2[i:i + CHUNK_ROWS].contiguous())
            outs.append(ops["mm"](xq, w.t(), xs, layer._gb10_wscale_col, x.dtype, bias))
        out = outs[0] if len(outs) == 1 else torch.cat(outs)
        _state["used"] += 1
        return out.view(*x.shape[:-1], n_pad)
    except Exception as e:  # noqa: BLE001 - keep serving on Marlin
        _state["runtime_ok"] = False
        logger.warning("FP8 side prefill: the W8A8 path failed (%s: %s); back to Marlin",
                       type(e).__name__, e)
        return None
