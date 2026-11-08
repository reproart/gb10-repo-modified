# When the GPU reads a file through the page tables (GB10)

On a DGX Spark / GX10 the GPU can dereference ordinary host pointers: a CUDA
kernel may read a `mmap`ed file directly, the GPU walking the CPU's page tables
(`cudaDevAttrPageableMemoryAccessUsesHostPageTables` = 1). That is how a table
bigger than what memory can spare is served without loading it: Qwen3.8-Flash-
Next's 47.7 GiB n-gram (PLE) table stays on NVMe and the page cache, and the
gather kernel reads the rows it needs. It works, and it is correct. It is also
slow in a way that comes and goes, and this note is about that: what it looks
like, how to measure it, what was ruled out, and the design that fixed it.

**Why it will come up again.** Lookup-table memories are new and spreading:
DeepSeek's Engram (DeepSeek V4; in SGLang `layers/engram.py`) and the n-gram
PLE table of Qwen3.8-Flash-Next (`models/qwen4_exp*.py`). They are large (tens
of GiB) and touched sparsely (a few rows a token), so on a machine short of
memory the obvious move is to leave them on storage and read them in place.
SGLang's own file backend for the PLE table does exactly that with a GPU
gather, and has this problem; SGLang's Engram tables are resident and
registered today, but a file-backed mode for small machines would invite the
same design. Whoever writes the next one should read this first.

**In short:** do not let the GPU take page faults on file-backed memory on the
prefill path. Read the rows on CPU threads into a pinned buffer and hand the GPU
the buffer. On Flash-Next this took prefill from 1.2-2.0K tok/s (depending on
the machine's state) to 2.1-2.3K tok/s in every run.

## Who is affected

The pattern is narrow: a **GPU kernel dereferencing pageable, file-backed
memory** (an `mmap` of a file, not registered with `cudaHostRegister`), on a
path that touches many cold pages at once (prefill).

| Where | How the rows are read | Affected |
|---|---|---|
| SGLang, `models/qwen3.8-flash-next.sh`, `PLE_TABLE=mmap` (patches/gb10_ple_mmap.py) | `PLE_GATHER=gpu`: the gather kernel on the mapping | **yes** (was the default until 2026-11-07) |
| same, `PLE_GATHER=cpu` (the default now) | CPU threads, pinned buffer, one copy to the GPU | no |
| stock SGLang file backend (`PLE_TABLE=file`, `--ple-offload-backend file`) | the gather kernel on a sparse file it maps | **yes**, same mechanism (not measured separately) |
| vllm-flash-ar (`vllm-flash-ar/src/vllm_ple_mmap.py`) | 32 CPU threads over `np.memmap`, unique rows only, pinned buffer | no: designed this way from the start |
| SGLang engram host tables (DeepSeek V4) | anonymous / memfd memory, pre-faulted and `cudaHostRegister`ed | no: nothing is file-backed or faulted at run time |
| llama.cpp (`llamacpp/`), every recipe that loads its weights | weights copied into device allocations at load | no |

A future integration is at risk when it offloads something large to storage
and lets a kernel read it in place. The usual candidates: embedding tables
(n-gram / PLE / "engram"-style), expert weights kept on disk, KV or prefix
caches spilled to files.

## The symptom

Prefill throughput that depends on the run, not on the prompt. Three
`bench/perf.py` runs back to back on the same server (GPU gather, 2026-11-07):

| Prompt | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| 2K | 1774 tok/s | 1595 | 1259 |
| 8K | 1910 | 1703 | 1161 |
| 32K | 1974 | 1619 | 1339 |
| 101K | 1848 | 1337 | 1532 |

Decode is unaffected (it reads 16 rows a token, and the hot ones stay cached).
GPU clocks and temperatures are the same in fast and slow runs.

## How to see it

`bench/perf.py` prints, for every prefill prompt, what came from storage and
how many page faults had to wait for it (from `/proc/vmstat`: `pgpgin`,
`pgmajfault`):

```
prompt  101148 tok -> TTFT 54.75s  prefill ~1848 tok/s
        vm: ...; read from disk 2.24 GiB, 72K major faults; ...
prompt  101415 tok -> TTFT 75.85s  prefill ~1337 tok/s
        vm: ...; read from disk 3.67 GiB, 597K major faults; ...
```

**The tell is the ratio, not the bytes.** About the same data comes from
storage in fast and slow runs. In a fast run most of it was read ahead (by
page-cache hints) before the kernel touched it; in a slow run the kernel
faulted on it and waited, one page at a time. The cost works out to **~35 us
per major fault** (run 2 above took 21 s longer with 525K more faults), which
is one NVMe read at queue depth 1: the GPU's faults are served essentially
serially. Hundreds of thousands of them on a 101K prompt is seconds.

`bench/profile_decode.py --prefill 16384` shows where a prefill's GPU time
goes; with this problem the gather's group grows from milliseconds to a large
share of the chunk.

## What did not explain it

Each was tested with the counters above on the Spark, and each was ruled out.
Kept here so the next integration does not spend a day on them again.

1. **The hints arrive too late** (one `posix_fadvise(WILLNEED)` per page from a
   background Python thread that needs the GIL between calls). Batched them
   (`process_madvise` over 1024 page ranges a call, ~100 calls a chunk), then
   sent them synchronously *before* launching the gather (~47 ms a 4096-token
   chunk). Same pattern.
2. **The multi-gen LRU evicts read-ahead pages first** (it puts unaccessed clean
   pages in its oldest generations). `echo n > /sys/kernel/mm/lru_gen/enabled`:
   same pattern.
3. **kswapd reclaim / free memory at the watermarks.** Slow runs often coincided
   with low `MemFree`, and `drop_caches` made the next run fast. But a run with
   kswapd freeing nothing and 8-12 GiB free was slow too.
4. **The watermark boost** (`vm.watermark_boost_factor`, kswapd reclaiming
   extra after fragmentation events from CUDA allocations). Set to 0: same.
5. **A cgroup memory limit** (reclaim inside a cgroup is invisible in
   `/proc/vmstat`). The server ran in the login session's scope with no limit
   and reclaimed nothing itself. `perf.py` now prints the server's cgroup, its
   limits and what it reclaimed, so this is one line to check.

A trap on the way: `workingset_refault_file` ("pages read again after
reclaim") counts a page evicted at *any* earlier time, for instance while the
weights loaded at boot. It does not mean something is evicting pages now.

Why the hinted pages were still absent when the GPU got to them in some
machine states was **not** found. The fix below does not depend on it.

## The fix: gather on the CPU, hand the GPU a buffer

`PLE_GATHER=cpu`, `CpuGather` in `patches/gb10_ple_mmap.py`:

- one zero-copy `torch.frombuffer` view per table shard over the mapping;
- for a prefill-sized gather (2048+ rows; a 4096-token chunk is 65K rows),
  group the ids by shard and copy the rows with `index_select` straight into a
  reused **pinned** buffer, split over a pool of CPU threads (`index_select`
  releases the GIL). Each thread takes its own page faults, so storage serves
  them **in parallel** instead of one at a time;
- one host-to-device copy, then convert to BF16 and scatter into place on the
  GPU (`index_copy_`); ids outside the TP window or absent from the checkpoint
  are zeros, as in the kernel;
- decode-sized gathers and CUDA-graph capture keep the GPU kernel (a few rows,
  mostly hot, and nothing CPU-side can run inside a capture);
- a CUDA event guards the pinned buffer until the copy has read it.

~16 ms a 4096-token chunk from the page cache (4-core VM). On the Spark,
three runs back to back:

| Prompt | GPU gather (range over runs) | CPU gather, runs 1 / 2 / 3 |
|---|---:|---:|
| 2K | 1188-1888 | 2240 / 2283 / 2151 |
| 8K | 1110-1959 | 2259 / 2312 / 2295 |
| 32K | 1239-1974 | 2271 / 2302 / 2283 |
| 101K | 1209-1872 | 2122 / 2137 / 2114 |

Major faults went from tens or hundreds of thousands a prompt to a few
hundred; the bytes read stayed the same. It beats even the GPU gather's best
runs, because the faults that remain are taken in parallel.

vllm-flash-ar arrived at the same design independently (`VLLM_PLE_MMAP_*`):
CPU worker threads over `np.memmap`, a pinned buffer, and two refinements worth
copying: **deduplicate the ids** before gathering (n-gram rows repeat inside a
chunk; gather the unique ones, expand on the GPU) and optionally **prefetch at
batch assembly** (hash the ids when the batch is built and gather in a worker
thread while the GPU runs the layers before the PLE layer).

## Checklist for a new integration

- If a kernel reads file-backed memory, assume prefill will be bound by
  serialized GPU faults on cold pages. Measure before trusting it:
  `perf.py`'s `read from disk ... major faults` per prompt, several runs back
  to back, without dropping caches in between.
- Prefer a CPU-side gather into a pinned buffer for anything prefill-sized.
  Keep the GPU path only for decode-sized reads and CUDA-graph replays.
- Use `MADV_RANDOM` on the mapping (no readahead around each fault: the reads
  are random rows, not sequential).
- Bound the mapping's resident set (SGLang's `PleFileRssTrimmer`,
  `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB`): mapped pages accumulate in the same
  pool the KV cache and the weights live in.
- Do not read a page-cache counter without knowing what it counts
  (`workingset_refault_file` above).
- Page-cache hints (`WILLNEED`) are cheap when batched (`process_madvise`), but
  were not enough on their own here. With the CPU gather they may be
  redundant; measured, they are not: without them (`SGLANG_QWEN4_PLE_FILE_PREFETCH=0`)
  the CPU gather ran 3-8% slower (101K 2008-2081 tok/s against 2114-2137), the
  threads then taking every fault themselves (~300-500K a 101K prompt) instead of
  finding pages already on their way. Keep both: hints first (async I/O at a
  high queue depth), then the threads.

## Open questions

- Why hinted pages were missing at GPU-touch time in some states. A reproducer
  outside the server (map the table, hint a chunk's pages, check residency with
  `mincore` over time, then touch them from a kernel) would separate "the hints
  did not insert the pages" from "something removed them".
- Whether the stock SGLang file backend (`PLE_TABLE=file`) shows the same
  spread; it uses the same GPU gather and the per-page hints.
- Whether the GPU's fault servicing on GB10 is serialized by design
  (one outstanding fault per context?) or by configuration; NVIDIA would know.

Measurements and the day's sequence: `results/RESULTS.md`, "Kernel bench on
SGLang 0.5.21 (2026-11-07)" and the notes after it.
