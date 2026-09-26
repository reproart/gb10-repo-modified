"""The gather kernel for gb10_ple_mmap: SGLang 0.5.20's
_gather_ple_embedding_from_pinned_kernel, reading from per-shard base
addresses instead of one table pointer. Imported only when a gather runs
(it needs triton, which gb10_ple_mmap must not import at startup)."""

import triton
import triton.language as tl


@triton.jit
def _gather_ple_rows_kernel(
    bases_ptr,
    rows_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    shard_size,
    num_shards,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_tp = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    global_idx = tl.where(in_tp, global_idx, 0)
    # Shard k holds global rows [k * shard_size, k * shard_size + rows[k]),
    # the layout SGLang's loader writes into its table file.
    shard = tl.minimum(global_idx // shard_size, num_shards - 1)
    local_idx = global_idx - shard * shard_size
    shard_rows = tl.load(rows_ptr + shard)
    # Rows no shard covers are zero in the stock sparse file; zero here too.
    present = in_tp & (local_idx < shard_rows)
    local_idx = tl.where(present, local_idx, 0)
    base = tl.load(bases_ptr + shard)
    offsets = tl.arange(0, BLOCK_D)
    in_row = offsets < embedding_dim
    if is_fp8:
        row_ptr = base.to(tl.pointer_type(tl.float8e4nv))
    else:
        row_ptr = base.to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        row_ptr + local_idx * embedding_dim + offsets,
        mask=in_row & present,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(output_ptr + row_id * embedding_dim + offsets, values, mask=in_row)


def gather_rows(
    bases, rows, flat_ids, output, *, embedding_dim, shard_size, num_shards,
    tp_vocab_start, tp_vocab_end, is_fp8, block_d,
):
    _gather_ple_rows_kernel[(flat_ids.numel(),)](
        bases,
        rows,
        flat_ids,
        output,
        embedding_dim=embedding_dim,
        shard_size=shard_size,
        num_shards=num_shards,
        tp_vocab_start=tp_vocab_start,
        tp_vocab_end=tp_vocab_end,
        is_fp8=is_fp8,
        BLOCK_D=block_d,
    )
