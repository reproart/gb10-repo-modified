"""gb10_fp8_rowwise — FP8 weights with one scale per output row for small BF16
linear layers, and a Triton GEMM for decode-sized inputs that reads them.

Used by patch_hc_fp8.py (VLLM_HC_FP8=1) for Qwen3.8-Flash-Next's hyper-connection
projections: every decoder layer mixes its 4 residual streams twice through a
low-rank gate, input_mix_weight_down(+block_inject) [336 x 10240] and
input_mix_weight_up [10240 x 320], BF16, ~13.5 MB a mix, ~98 mixes a decode step.
A decode profile (2026-11-03, bench/profile_decode.py) had them at ~7 ms of a
~50 ms step as cuBLAS GEMMs at 170-210 GB/s: bound by reading BF16 weights.
Stored as FP8 E4M3 with a float32 scale per row (amax / 448, the recipe of
../../patches/gb10_fp8_hc.py, FP8_HC on SGLang: +3-3.5% decode there, HumanEval
unchanged), they are half the bytes.

  * process_weights_after_loading quantizes the weight; rows from `keep_from` on
    stay BF16 (the 16 block_inject + padding rows merged into the down projection:
    SGLang kept block_inject in BF16, so this does too);
  * inputs of up to MAX_ROWS rows (decode and MTP verify: 4 a sequence) run a
    Triton kernel: FP8 rows converted to BF16 in registers (exact), FP32
    accumulation, the row scale applied to the FP32 sum. Long K is split over
    programs and the partial sums added by a second kernel in a fixed order:
    no atomics, the same bits every run;
  * longer inputs (prefill) take cuBLAS on a BF16 copy of the weight made for that
    call by one Triton pass (FP8 rows times their scales, then the tail rows):
    ~10 MB of traffic a call, against a GEMM that reads a [8192 x 10240] input.

The GEMM is a torch custom op (gb10::fp8_row_linear), so torch.compile treats
it as one opaque call and CUDA graphs capture its kernels. Lossy: the gate that
mixes the residual streams moves by per-row FP8 rounding; check answers.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("vllm.gb10_fp8_rowwise")

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MAX_ROWS = 64        # inputs up to this many rows take the Triton kernel
ROW_ALIGN = 16       # FP8 rows end on a multiple of this (the kernel's BLOCK_N for a BF16 tail)
STATS = {"layers": 0, "bytes_before": 0, "bytes_after": 0, "logged": False}
_state = {"bad_tiles": set(), "sms": None}  # row tiles whose kernel did not build here


def quantize_per_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] -> (q [N, K] E4M3, s [N] float32) with w ~= q * s[:, None]."""
    wf = w.float()
    s = (wf.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    q = (wf / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.contiguous(), s.contiguous()


def dequantize(q: torch.Tensor, s: torch.Tensor, tail: torch.Tensor | None = None) -> torch.Tensor:
    """The float32 weight the kernels compute with: [N8 + tail rows, K]."""
    w = q.float() * s[:, None]
    return w if tail is None or tail.shape[0] == 0 else torch.cat([w, tail.float()])


# --------------------------------------------------------------------------- kernels

_kernels = {}


def _get_kernels():
    if _kernels:
        return _kernels["mm"], _kernels["reduce"]
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_row_mm(x_ptr, wq_ptr, s_ptr, wt_ptr, y_ptr, p_ptr,
                    M, N, N8, K, K_SPLIT, stride_xm, stride_ym,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    SPLIT: tl.constexpr):
        # program (n block, k split): BLOCK_N output columns over K_SPLIT of K.
        # Blocks below N8 read FP8 rows (and apply their scales), the rest BF16 rows.
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        k_lo = pid_k * K_SPLIT
        k_hi = tl.minimum(k_lo + K_SPLIT, K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        m_ok = rm[:, None] < M
        if pid_n * BLOCK_N < N8:
            n_ok = rn[:, None] < N8
            for k0 in range(k_lo, k_hi, BLOCK_K):
                kk = k0 + rk
                k_ok = kk[None, :] < k_hi
                x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :], mask=m_ok & k_ok, other=0.0)
                w = tl.load(wq_ptr + rn[:, None] * K + kk[None, :], mask=n_ok & k_ok, other=0.0)
                acc += tl.dot(x, tl.trans(w.to(x.dtype)))
            s = tl.load(s_ptr + rn, mask=rn < N8, other=0.0)
            acc = acc * s[None, :]
        else:
            rt = rn - N8
            n_ok = rn[:, None] < N
            for k0 in range(k_lo, k_hi, BLOCK_K):
                kk = k0 + rk
                k_ok = kk[None, :] < k_hi
                x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :], mask=m_ok & k_ok, other=0.0)
                w = tl.load(wt_ptr + rt[:, None] * K + kk[None, :], mask=n_ok & k_ok, other=0.0)
                acc += tl.dot(x, tl.trans(w.to(x.dtype)))
        out_ok = m_ok & (rn[None, :] < N)
        if SPLIT == 1:
            tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=out_ok)
        else:
            tl.store(p_ptr + (pid_k * BLOCK_M + rm[:, None]) * N + rn[None, :], acc, mask=out_ok)

    @triton.jit
    def _reduce(p_ptr, y_ptr, M, N, stride_ym,
                SPLIT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK: tl.constexpr):
        # the K splits' partial sums, added in split order: the same bits every run
        rn = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        rm = tl.arange(0, BLOCK_M)
        ok = (rm[:, None] < M) & (rn[None, :] < N)
        acc = tl.zeros((BLOCK_M, BLOCK), dtype=tl.float32)
        for i in tl.static_range(SPLIT):
            acc += tl.load(p_ptr + (i * BLOCK_M + rm[:, None]) * N + rn[None, :], mask=ok, other=0.0)
        tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=ok)

    _kernels["mm"], _kernels["reduce"] = _fp8_row_mm, _reduce
    return _fp8_row_mm, _reduce


def _sms(device) -> int:
    if _state["sms"] is None:
        _state["sms"] = (torch.cuda.get_device_properties(device).multi_processor_count
                         if device.type == "cuda" else 48)
    return _state["sms"]


def config(n: int, k: int, n8: int, sms: int) -> tuple[int, int, int, int]:
    """(BLOCK_N, BLOCK_K, SPLIT, K_SPLIT) for an [N, K] weight: ~4 programs an SM,
    from output blocks first, then from splitting K. Depends on the weight only,
    never on the rows, so a sequence's sums do not change with the batch."""
    block_n = 16 if (n <= 1024 or n8 < n) else 64
    block_k = 256 if k >= 2048 else 64
    n_blocks = -(-n // block_n)
    k_steps = -(-k // block_k)
    split = 1
    if n_blocks < 4 * sms:
        split = max(1, min(k_steps, round(4 * sms / n_blocks)))
    steps = -(-k_steps // split)
    split = -(-k_steps // steps)
    return block_n, block_k, split, steps * block_k


def row_tile(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def triton_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, wt: torch.Tensor,
              cfg: tuple[int, int, int, int] | None = None) -> torch.Tensor:
    """x [M <= MAX_ROWS, K] (BF16/FP16) times the FP8 rows (scaled) and the BF16 tail rows."""
    import triton

    mm, reduce = _get_kernels()
    m, k = x.shape
    n8, n = wq.shape[0], wq.shape[0] + wt.shape[0]
    if x.stride(1) != 1:
        x = x.contiguous()
    block_n, block_k, split, k_split = cfg or config(n, k, n8, _sms(x.device))
    assert n8 % block_n == 0 or n8 == n, "FP8 rows must end on a block"
    block_m = row_tile(m)
    if block_m == 64 and block_k > 128 and k_split % 128 == 0:
        block_k = 128  # 64 x 256 BF16 + 16 x 256 FP8 x 3 stages would not fit a GB10 SM's shared memory
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    p = torch.empty((split, block_m, n), dtype=torch.float32, device=x.device) if split > 1 else y
    tail = wt if wt.shape[0] else wq  # any valid pointer when there is no tail
    mm[(triton.cdiv(n, block_n), split)](
        x, wq, s, tail, y, p, m, n, n8, k, k_split, x.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT=split, num_warps=4, num_stages=3)
    if split > 1:
        reduce[(triton.cdiv(n, 128),)](p, y, m, n, y.stride(0),
                                       SPLIT=split, BLOCK_M=block_m, BLOCK=128, num_warps=4)
    return y


def _get_dequant():
    if "dequant" in _kernels:
        return _kernels["dequant"]
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant(wq_ptr, s_ptr, wt_ptr, out_ptr, N8, K, BLOCK_K: tl.constexpr):
        # one output row, BLOCK_K of its columns: FP8 row x its scale in FP32, or a tail row
        row = tl.program_id(0)
        rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        ok = rk < K
        if row < N8:
            v = tl.load(wq_ptr + row * K + rk, mask=ok, other=0.0).to(tl.float32) * tl.load(s_ptr + row)
        else:
            v = tl.load(wt_ptr + (row - N8) * K + rk, mask=ok, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * K + rk, v.to(out_ptr.dtype.element_ty), mask=ok)

    _kernels["dequant"] = (_dequant, triton)
    return _kernels["dequant"]


def dequant_weight(wq: torch.Tensor, s: torch.Tensor, wt: torch.Tensor, dtype: torch.dtype,
                   force_triton: bool = False) -> torch.Tensor:
    """The whole weight [N8 + tail rows, K] in `dtype`: each FP8 row times its scale in FP32,
    rounded once, then the tail rows. On the GPU one Triton pass (reads ~1 byte, writes 2 a
    weight); elsewhere, or if that kernel does not build, torch (the same values)."""
    n8, k = wq.shape
    if (wq.is_cuda or force_triton) and "dequant" not in _state["bad_tiles"]:
        try:
            kern, triton = _get_dequant()
            out = torch.empty((n8 + wt.shape[0], k), dtype=dtype, device=wq.device)
            bk = min(1024, triton.next_power_of_2(k))
            kern[(out.shape[0], triton.cdiv(k, bk))](wq, s, wt if wt.shape[0] else wq, out, n8, k,
                                                     BLOCK_K=bk, num_warps=4)
            return out
        except Exception as e:  # noqa: BLE001 - torch from now on
            _state["bad_tiles"].add("dequant")
            logger.warning("FP8 HC: the dequantize kernel failed (%s: %s); prefill converts with torch",
                           type(e).__name__, e)
    w = (wq.float() * s[:, None]).to(dtype)
    return torch.cat([w, wt.to(dtype)]) if wt.shape[0] else w


def torch_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    """The path for longer inputs (prefill): the weight dequantized to x's dtype for this
    call (dequant_weight), then one cuBLAS GEMM over the input, as without FP8. Earlier
    versions scaled the output or ran the BF16 tail as a second GEMM, which read the
    [8192 x 10240] input of the down projection twice (+0.9 ms a call at 8192 rows)."""
    import torch.nn.functional as F

    return F.linear(x, dequant_weight(wq, s, wt, x.dtype))


def _impl(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    k = x.shape[-1]
    n = wq.shape[0] + wt.shape[0]
    x2 = x.reshape(-1, k)
    if x2.shape[0] == 0:
        return x.new_empty((*x.shape[:-1], n))
    y = None
    tile = row_tile(x2.shape[0])
    if x2.is_cuda and x2.shape[0] <= MAX_ROWS and tile not in _state["bad_tiles"]:
        try:
            y = triton_mm(x2, wq, s, wt)
        except Exception as e:  # noqa: BLE001 - a tile that does not build here: torch path for its sizes
            _state["bad_tiles"].add(tile)
            logger.warning("FP8 HC: the Triton kernel for %d-row tiles failed (%s: %s); inputs of that "
                           "size take the torch path", tile, type(e).__name__, e)
    if y is None:
        y = torch_mm(x2, wq, s, wt)
    if not STATS["logged"]:
        STATS["logged"] = True
        logger.info("FP8 HC: %d hyper-connection projections in FP8 with per-row scales, "
                    "%.2f -> %.2f GiB; inputs up to %d rows on a Triton kernel, longer ones on torch",
                    STATS["layers"], STATS["bytes_before"] / 2**30, STATS["bytes_after"] / 2**30, MAX_ROWS)
    return y.reshape(*x.shape[:-1], n)


if hasattr(torch.ops, "gb10") and hasattr(torch.ops.gb10, "fp8_row_linear"):
    fp8_row_linear = torch.ops.gb10.fp8_row_linear
else:
    @torch.library.custom_op("gb10::fp8_row_linear", mutates_args=())
    def fp8_row_linear(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
        return _impl(x, wq, s, wt)

    @fp8_row_linear.register_fake
    def _(x, wq, s, wt):
        return x.new_empty((*x.shape[:-1], wq.shape[0] + wt.shape[0]))


# --------------------------------------------------------------------------- vLLM wiring

HC_LAYERS = ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up")
_method_cls = {}


def _linear_method_cls(base):
    if base in _method_cls:
        return _method_cls[base]

    class Gb10Fp8RowLinearMethod(base):
        """An unquantized linear method whose weight becomes FP8 rows after loading."""

        def process_weights_after_loading(self, layer):
            super().process_weights_after_loading(layer)
            w = layer.weight
            if w.dim() != 2 or w.dtype not in (torch.bfloat16, torch.float16):
                logger.warning("FP8 HC: %s left as is (%s %s)", type(layer).__name__, w.dtype, tuple(w.shape))
                return
            keep = getattr(self, "_gb10_keep_from", None) or w.shape[0]
            n8 = keep - keep % ROW_ALIGN if keep < w.shape[0] else w.shape[0]
            q, s = quantize_per_row(w.data[:n8])
            tail = w.data[n8:].contiguous()
            STATS["layers"] += 1
            STATS["bytes_before"] += w.numel() * w.element_size()
            STATS["bytes_after"] += q.numel() + s.numel() * 4 + tail.numel() * tail.element_size()
            layer.weight = torch.nn.Parameter(q, requires_grad=False)
            layer.register_buffer("gb10_scale", s, persistent=False)
            layer.register_buffer("gb10_tail", tail, persistent=False)
            layer._gb10_fp8 = True

        def apply(self, layer, x, bias=None):
            if not getattr(layer, "_gb10_fp8", False):
                return super().apply(layer, x, bias)
            y = torch.ops.gb10.fp8_row_linear(x, layer.weight, layer.gb10_scale, layer.gb10_tail)
            return y if bias is None else y + bias

    _method_cls[base] = Gb10Fp8RowLinearMethod
    return Gb10Fp8RowLinearMethod


def convert_layer(lin, base, keep_from=None) -> bool:
    """Give `lin` (a vLLM LinearBase with exactly `base` as its quant method) the FP8
    method: a new instance carrying the old one's state, so a shared method is untouched."""
    qm = getattr(lin, "quant_method", None)
    if type(qm) is not base:
        return False
    cls = _linear_method_cls(base)
    new = cls.__new__(cls)
    new.__dict__.update(qm.__dict__)
    new._gb10_keep_from = keep_from
    lin.quant_method = new
    return True


def install_hc(ns: dict, linear_base=None, unquantized=None) -> int:
    """Wrap __init__ of every nn.Module class defined in the module whose globals are
    `ns` (vLLM's qwen3_8_flash_next/nvidia/hyperconnection.py): after it runs, its
    HC_LAYERS get the FP8 method. block_inject rows (after the first output part of
    the merged down projection) stay BF16. Returns the number of classes wrapped."""
    if linear_base is None or unquantized is None:
        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
        linear_base, unquantized = LinearBase, UnquantizedLinearMethod
    wrapped = 0
    for obj in list(ns.values()):
        if not (isinstance(obj, type) and issubclass(obj, torch.nn.Module)
                and obj.__module__ == ns.get("__name__") and "__init__" in obj.__dict__):
            continue
        if getattr(obj.__init__, "_gb10_fp8", False):
            continue
        orig = obj.__init__

        def __init__(self, *args, _orig=orig, **kwargs):
            _orig(self, *args, **kwargs)
            for name in HC_LAYERS:
                lin = self._modules.get(name)
                if not isinstance(lin, linear_base):
                    continue
                keep = None
                if name.endswith("_block_inject"):
                    sizes = getattr(lin, "output_sizes", None)
                    keep = sizes[0] if sizes else None
                convert_layer(lin, unquantized, keep)

        __init__._gb10_fp8 = True
        obj.__init__ = __init__
        wrapped += 1
    if not wrapped:
        raise RuntimeError("VLLM_HC_FP8=1: no module class found to wrap in " + str(ns.get("__name__")))
    logger.info("FP8 HC on: %d hyper-connection classes wrapped (VLLM_HC_FP8=1)", wrapped)
    return wrapped
