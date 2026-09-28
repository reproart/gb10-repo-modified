"""The GEMM kernel for gb10_skinny: y = x @ w.T for a few rows of x (decode and
verify batches), BF16 in and out, FP32 accumulation. Imported on the first
call (it needs triton, which gb10_skinny must not import at startup)."""

import triton
import triton.language as tl

BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 256


@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per 16 output columns (and per 16 rows of x): N = 512 or
    # 640 gives 32-40 programs, each streaming its 16 weight rows once.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * stride_xm + kk[None, :],
                    mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + kk[None, :],
                    mask=(rn[:, None] < N) & (kk[None, :] < K), other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def skinny_gemm(x, w):
    """x [M, K] (last dim contiguous), w [N, K] contiguous -> [M, N], x's dtype."""
    import torch

    m, k = x.shape
    n = w.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, BLOCK_N), triton.cdiv(m, BLOCK_M))
    _skinny_gemm_kernel[grid](x, w, y, m, n, k, x.stride(0), w.stride(0), y.stride(0),
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                              num_warps=4, num_stages=3)
    return y
