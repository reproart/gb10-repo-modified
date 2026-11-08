# [Perf] Qwen4-Exp PLE file backend: GPU gather through host page tables makes prefill depend on page-cache state (GB10); a CPU-side gather fixes it

## Summary

With `--ple-offload-embedding --ple-offload-backend file` (Qwen3.8-Flash-Next
on a DGX Spark / GB10), prefill throughput varies from run to run by up to
~40% on the same server and prompts. The cause is
`_gather_ple_embedding_from_pinned_kernel` reading the file-backed table
through the host page tables. Every row whose page is not in the page cache
costs the GPU a fault served from storage. These faults take ~35 µs each and
appear to be served one at a time. `PleFilePrefetcher`'s `WILLNEED` hints keep
them rare in some machine states and not in others.

Reading prefill-sized gathers on CPU threads into a pinned buffer, then one
host-to-device copy, removed the variation. We measured 2.1-2.3K tok/s in
every run, against 1.2-2.0K with the GPU gather. Decode-sized gathers and
CUDA-graph capture can keep the kernel.

## Environment

- SGLang 0.5.21, TP 1, `RadixArk/Qwen3.8-Flash-Next-NVFP4`
- NEXTN MTP 3/1/4, chunked prefill 4096
- DGX Spark (GB10, 128 GB unified memory, local NVMe), DGX OS
- The table is 47.7 GiB of FP8, read in place (it does not fit next to the
  weights). `cudaDevAttrPageableMemoryAccessUsesHostPageTables` = 1.

One caveat about our setup. We measured with a patch that maps the
checkpoint's own shard files instead of SGLang's sparse copy. The read path is
the same: a gather kernel dereferencing the mapping through host page tables,
with `WILLNEED` hints sent for the rows of prefill-sized gathers. We have not
measured the stock file backend separately. We expect the same behaviour, and
a check on your side would be welcome.

## What we measured

`/proc/vmstat` deltas per prompt: `pgpgin`, the bytes read from storage, and
`pgmajfault`, the faults that waited for storage. Unique random-word prompts,
three runs back to back on one server:

| Prompt | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 2K | 1774 tok/s, 309 major faults | 1595, 8.6K | 1259, 15K |
| 8K | 1910, 1.1K | 1703, 28K | 1161, 80K |
| 32K | 1974, 7.6K | 1619, 148K | 1339, 193K |
| 101K | 1848, 72K (2.24 GiB read) | 1337, 597K (3.67 GiB) | 1532, 378K (2.96 GiB) |

About the same data comes from storage in fast and slow runs. What differs is
whether it arrived before the kernel touched it. In a slow run the gather
faulted on it itself. Run 2's 101K prompt took 21 s longer with 525K more
faults, which is ~35 µs a fault: one NVMe read at queue depth 1. GPU clocks
and temperatures were the same across runs. Decode was unaffected.

This matches the numbers in `PleFilePrefetcher`'s own docstring: cold
650-750 tok/s, 1,000-2,100 with the hints, warm 2,200-2,600. The hints help
by a different amount depending on the state of the machine.

## What did not explain it

Each was tested with the same counters, and each changed nothing:

- **Hint timing.** The stock hints are one `posix_fadvise(WILLNEED)` per page
  on a background Python thread, which needs the GIL between calls. We batched
  them through `process_madvise(MADV_WILLNEED)`: up to 1024 page ranges a call,
  ~100 calls a 4096-token chunk. Then we sent them synchronously before
  launching the gather (~47 ms a chunk).
- **The multi-gen LRU** evicting unaccessed read-ahead pages first. Disabled:
  `/sys/kernel/mm/lru_gen/enabled` = n.
- **kswapd reclaim and low free memory.** Slow runs also occurred with kswapd
  freeing nothing and 8-12 GiB free.
- **The watermark boost:** `vm.watermark_boost_factor=0`.
- **A cgroup memory limit.** There was none, and the cgroup reclaimed nothing.

Why the hinted pages were absent when the GPU reached them in some states
remains open. `workingset_refault_file` is not evidence either way: it also
counts pages evicted long before, e.g. while the weights loaded.

## What fixed it

For gathers of ≥ `PLE_FILE_PREFETCH_MIN_ROWS` rows that are not under CUDA
graph capture:

1. Take the ids to the host. This is the same sync the prefetcher already
   does.
2. Copy the valid rows with `index_select` from the mapping into a reused
   pinned buffer, split over a pool of CPU threads. `index_select` releases
   the GIL, so each thread takes its own page faults and storage serves them
   in parallel.
3. Do one non-blocking copy to the GPU, then convert to BF16 and scatter into
   place (`index_copy_`). Ids outside the TP window read as zero, as in the
   kernel.
4. Record a CUDA event, and wait on it before reusing the pinned buffer.

Same server, same prompts, three runs back to back:

| Prompt | GPU gather (range over many runs) | CPU gather, runs 1 / 2 / 3 |
|---|---:|---:|
| 2K | 1188-1888 tok/s | 2240 / 2283 / 2151 |
| 8K | 1110-1959 | 2259 / 2312 / 2295 |
| 32K | 1239-1974 | 2271 / 2302 / 2283 |
| 101K | 1209-1872 | 2122 / 2137 / 2114 |

The CPU gather takes a few hundred major faults a prompt, against tens or
hundreds of thousands, for the same bytes read. It is faster than even the GPU
gather's best runs, because the faults that remain are taken in parallel.
From the page cache it costs ~16 ms per 4096-token chunk (65K rows × 2560 B).

A sketch for `Qwen4ExpPinnedHostEmbedding.gather` with the file backend, where
`self.weight` is the file-backed host tensor holding this rank's vocabulary
shard:

```python
flat_ids = input_ids.reshape(-1).long()
if (self._cpu_gather is not None
        and flat_ids.numel() >= PLE_FILE_PREFETCH_MIN_ROWS
        and not torch.cuda.is_current_stream_capturing()):
    ids = flat_ids.cpu()                                   # the prefetcher syncs here too
    start = self.shard_indices.org_vocab_start_index
    end = self.shard_indices.org_vocab_end_index
    pos = ((ids >= start) & (ids < end)).nonzero().squeeze(1)
    # thread pool: index_select(self.weight.view(torch.uint8), 0, ids[pos] - start)
    # in slices, each into its part of a reused pinned buffer
    rows = self._cpu_gather.take(self.weight.view(torch.uint8), ids[pos] - start)
    dev = rows.to(output.device, non_blocking=True)
    self._cpu_gather.record_copy()                         # event: buffer reusable after it
    flat = output.view(-1, self.embedding_dim)
    if pos.numel() < ids.numel():
        flat.zero_()
    flat.index_copy_(0, pos.to(output.device, non_blocking=True),
                     dev.view(self.weight.dtype).view(-1, self.embedding_dim).to(flat.dtype))
else:
    ...  # the existing hint + kernel path
```

Two refinements from another implementation of the same table (a vLLM recipe
that gathers it on CPU threads from the start):

- Deduplicate the ids before gathering. N-gram rows repeat within a chunk, so
  gather only the unique ones and expand on the GPU.
- Optionally hash the ids at batch assembly and gather while the GPU runs the
  layers before the PLE layer.

## Why it matters beyond this model

Lookup-table memories are spreading: DeepSeek's Engram in DeepSeek V4
(`layers/engram.py`) and Qwen's PLE n-gram table. They are tens of GiB, read a
few rows a token, and on memory-limited machines the natural design is to
leave them on storage and read them in place. The Engram host tables are
resident and registered today. A file-backed mode for small machines would
meet the same trap if it lets a kernel take the faults.

## Offer

We can open a PR with the CPU-side gather for the file backend. It would be
off by default or behind an env var at first if you prefer, with CPU tests
against the kernel's semantics. Our implementation (for the mapped-checkpoint
variant) and the full write-up, including how to measure it:
<!-- if the repository is public, link it here; otherwise drop this line -->
https://github.com/reproart/gb10-repo-modified/blob/main/native/docs/gpu-file-reads.md
