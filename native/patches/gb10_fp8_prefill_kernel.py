"""Triton gather for gb10_fp8_prefill: a Marlin-packed FP8 weight back to plain
[N, K] bytes in one pass. Imported on first use (it needs triton)."""

import triton
import triton.language as tl


@triton.jit
def _marlin_fp8_unpack_kernel(src_ptr, inv_ptr, dst_ptr, N, K,
                              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (n < N)[:, None] & (k < K)[None, :]
    # gb10_fp8_prefill._q_index and _block_base
    q = (((n // 16) % 4) * 256 + (n % 16))[:, None] + ((k % 16) * 16)[None, :]
    off = tl.load(inv_ptr + q, mask=mask, other=0)
    base = ((k // 16) * (16 * N))[None, :] + ((n // 64) * 1024)[:, None]
    v = tl.load(src_ptr + base + off, mask=mask, other=0)
    tl.store(dst_ptr + n[:, None] * K + k[None, :], v, mask=mask)


def marlin_fp8_unpack(packed, size_n, size_k, inv):
    import torch

    src = packed.contiguous().view(torch.uint8)
    out = torch.empty((size_n, size_k), dtype=torch.uint8, device=src.device)
    block_n, block_k = 64, 128
    grid = (triton.cdiv(size_n, block_n), triton.cdiv(size_k, block_k))
    _marlin_fp8_unpack_kernel[grid](src, inv, out, size_n, size_k,
                                    BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4)
    return out
