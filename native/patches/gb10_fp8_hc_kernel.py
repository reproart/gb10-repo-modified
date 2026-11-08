"""SGLang 0.5.20's _hc_mix_persistent_kernel (layers/hc_mix_triton.py) reading
FP8 E4M3 weights with one float32 scale per weight row. Both GEMMs reduce
along a row of their weight, so each row's scale multiplies that output
column after the dot: the rest of the kernel is unchanged. Also the end of
the mix for prefill-sized inputs (hc_gate_apply: row scale, sigmoid, product
with x and mean over the streams in one pass). Imported on the first call (it
needs triton, which gb10_fp8_hc must not import at startup)."""

import triton
import triton.language as tl


@triton.jit
def _grid_barrier(counter_ptr, num_ctas):
    tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
        pass


@triton.jit
def _hc_mix_fp8_kernel(
    x_ptr,
    w_down_ptr,
    s_down_ptr,
    w_up_ptr,
    s_up_ptr,
    t_raw_ptr,
    out_ptr,
    counters_ptr,
    K,
    LOWRANK,
    HS,
    num_rows,
    num_ctas,
    inv_hc,
    ROWS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, ROWS)
    mask_m = offs_m < num_rows

    zero_span = ROWS * LOWRANK
    offs_z = tl.arange(0, 256)
    for z0 in range(pid * 256, zero_span, num_ctas * 256):
        idx = z0 + offs_z
        tl.store(t_raw_ptr + idx, 0.0, mask=idx < zero_span)
    _grid_barrier(counters_ptr + 0, num_ctas)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    n_blocks = tl.cdiv(LOWRANK, BLOCK_N)
    k_chunks = tl.cdiv(K, BLOCK_K)
    for tile in range(pid, n_blocks * k_chunks, num_ctas):
        nb = tile % n_blocks
        kc = tile // n_blocks
        n = nb * BLOCK_N + offs_n
        k = kc * BLOCK_K + offs_k
        mask_n = n < LOWRANK
        xt = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        w = tl.load(
            w_down_ptr + n[:, None] * K + k[None, :],
            mask=mask_n[:, None],
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        s = tl.load(s_down_ptr + n, mask=mask_n, other=0.0)
        acc = tl.dot(xt, tl.trans(w)) * s[None, :]
        tl.atomic_add(
            t_raw_ptr + offs_m[:, None] * LOWRANK + n[None, :],
            acc,
            mask=mask_n[None, :],
            sem="relaxed",
            scope="gpu",
        )
    _grid_barrier(counters_ptr + 1, num_ctas)

    offs_j = tl.arange(0, BLOCK_J)
    offs_r = tl.arange(0, BLOCK_R)
    offs_g = tl.arange(0, HC)
    j_blocks = tl.cdiv(HS, BLOCK_J)
    for jb in range(pid, j_blocks, num_ctas):
        j = jb * BLOCK_J + offs_j
        mask_j = j < HS
        gj = offs_g[:, None] * HS + j[None, :]
        gj_flat = tl.reshape(gj, (HC * BLOCK_J,))
        mask_gj = tl.reshape(
            tl.broadcast_to(mask_j[None, :], (HC, BLOCK_J)), (HC * BLOCK_J,)
        )
        acc = tl.zeros((ROWS, HC * BLOCK_J), dtype=tl.float32)
        for r0 in range(0, LOWRANK, BLOCK_R):
            r = r0 + offs_r
            mask_r = r < LOWRANK
            a = tl.load(
                t_raw_ptr + offs_m[:, None] * LOWRANK + r[None, :],
                mask=mask_r[None, :],
                other=0.0,
            )
            a = a * inv_hc
            t = (a * tl.sigmoid(a)).to(x_ptr.dtype.element_ty)
            w = tl.load(
                w_up_ptr + gj_flat[:, None] * LOWRANK + r[None, :],
                mask=mask_gj[:, None] & mask_r[None, :],
                other=0.0,
            ).to(x_ptr.dtype.element_ty)
            acc = tl.dot(t, tl.trans(w), acc)
        s = tl.load(s_up_ptr + gj_flat, mask=mask_gj, other=0.0)
        acc = acc * s[None, :]
        gate = tl.sigmoid(tl.reshape(acc, (ROWS, HC, BLOCK_J)))
        xg = tl.load(
            x_ptr
            + offs_m[:, None, None] * (HC * HS)
            + offs_g[None, :, None] * HS
            + j[None, None, :],
            mask=mask_m[:, None, None] & mask_j[None, None, :],
            other=0.0,
        ).to(tl.float32)
        out = tl.sum(gate * xg, axis=1) * inv_hc
        tl.store(
            out_ptr + offs_m[:, None] * HS + j[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_j[None, :],
        )

    ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
    if ticket == num_ctas - 1:
        tl.store(counters_ptr + 0, 0)
        tl.store(counters_ptr + 1, 0)
        tl.store(counters_ptr + 2, 0)


@triton.jit
def _hc_gate_apply_kernel(
    gate_ptr,
    s_ptr,
    x_ptr,
    out_ptr,
    rows,
    HS,
    stride_g,
    stride_x,
    stride_o,
    inv_hc,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """out[m, j] = mean_g sigmoid(gate[m, g*HS + j] * s[g*HS + j]) * x[m, g*HS + j]:
    the end of the mix for prefill-sized inputs, one pass over the gate and x."""
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    j = tl.program_id(1) * BLOCK_J + tl.arange(0, BLOCK_J)
    mask_m = m < rows
    mask_j = j < HS
    mask = mask_m[:, None] & mask_j[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    for g in tl.static_range(HC):
        col = g * HS + j
        s = tl.load(s_ptr + col, mask=mask_j, other=0.0)
        a = tl.load(gate_ptr + m[:, None] * stride_g + col[None, :], mask=mask, other=0.0)
        xv = tl.load(x_ptr + m[:, None] * stride_x + col[None, :], mask=mask, other=0.0)
        acc += tl.sigmoid(a.to(tl.float32) * s[None, :]) * xv.to(tl.float32)
    tl.store(out_ptr + m[:, None] * stride_o + j[None, :],
             (acc * inv_hc).to(out_ptr.dtype.element_ty), mask=mask)


def hc_gate_apply(gate, s_up, x, hc, hs):
    """gate [rows, hc*hs] (the up projection before its row scales), s_up [hc*hs]
    float32, x [rows, hc*hs] -> [rows, hs] in x's dtype. Both inputs need a unit
    stride along the last dimension."""
    import torch

    rows = x.shape[0]
    out = torch.empty((rows, hs), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    block_m, block_j = 8, 256
    grid = (triton.cdiv(rows, block_m), triton.cdiv(hs, block_j))
    _hc_gate_apply_kernel[grid](
        gate, s_up, x, out, rows, hs, gate.stride(0), x.stride(0), out.stride(0), 1.0 / hc,
        HC=hc, BLOCK_M=block_m, BLOCK_J=block_j, num_warps=4,
    )
    return out


MAX_ROWS = 64
_counters_cache = {}


def _get_counters(device):
    import torch

    buf = _counters_cache.get(device)
    if buf is None:
        buf = torch.zeros(3, dtype=torch.int32, device=device)
        _counters_cache[device] = buf
    return buf


def default_tiles(rows_pad):
    """(BLOCK_N, BLOCK_K, BLOCK_J, BLOCK_R, num_warps, num_stages) for a row tile.
    16 rows: SGLang's tiles. 32/64: a shorter K tile and two stages, so the
    pipelined x tile stays well inside GB10's ~100 KB shared memory."""
    if rows_pad == 16:
        return (32, 256, 32, 64, 8, 3)
    return (32, 128, 32, 64, 8, 2)


def fused_hc_mix_fp8(x, w_down, s_down, w_up, s_up, hc, hs, num_ctas=None, tiles=None):
    """SGLang's fused_hc_mix (x [rows, hc*hs] contiguous) with FP8 w_down
    [lowrank, hc*hs] / w_up [hc*hs, lowrank] and their float32 per-row
    scales. Up to 64 rows: SGLang's kernel stops at 16, one decode step's
    verify at 8 requests x 4 draft tokens is 32. tiles: default_tiles()'s
    tuple, for a sweep (BLOCK_K must divide hc*hs: K is not masked)."""
    import torch

    rows, k = x.shape
    lowrank = w_down.shape[0]
    if rows > MAX_ROWS:
        raise ValueError(f"fused_hc_mix_fp8: {rows} rows, at most {MAX_ROWS}")
    rows_pad = 16 if rows <= 16 else 32 if rows <= 32 else 64
    device = x.device
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    t_raw = torch.empty((rows_pad, lowrank), dtype=torch.float32, device=device)
    out = torch.empty((rows, hs), dtype=x.dtype, device=device)
    if rows == 0:
        return out
    bn, bk, bj, br, warps, stages = tiles or default_tiles(rows_pad)
    _hc_mix_fp8_kernel[(num_ctas,)](
        x, w_down, s_down, w_up, s_up, t_raw, out, _get_counters(device),
        k, lowrank, hs, rows, num_ctas, 1.0 / hc,
        ROWS=rows_pad, HC=hc, BLOCK_N=bn, BLOCK_K=bk,
        BLOCK_J=bj, BLOCK_R=br, num_warps=warps, num_stages=stages,
    )
    return out
