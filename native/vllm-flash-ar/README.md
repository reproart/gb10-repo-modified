# Qwen3.8-Flash-Next on a single DGX Spark (GB10) — int4 + int8 + fp8 hybrid, vLLM

> **In this project** (`native/vllm-flash-ar/`, moved here from `_separate/` on
> 2026-10-31): the recipe below, run like every other server here.
>
> ```bash
> cd native/vllm-flash-ar
> ./build.sh                 # the image: the official Flash-Next vLLM image + this recipe's patches (~20 GiB base once)
> ./download.sh              # the prepared checkpoint (~67 GiB) + the fp8 n-gram table (~49 GiB), pinned revisions
> ./serve.sh                 # one-server checks, memory prepared, the container, waits until the API answers
> ./status.sh / ./stop.sh
> GB10_METRICS_PORT=18400 python3 ../bench/perf.py   # + the engine sidecar's PLE gather numbers
> python3 ../bench/consistency.py                    # tool-call examples, greedy determinism, prefix-cache hits
> ```
>
> Settings: `config.sh` (every knob of the recipe, with what it does), this
> machine's own values in `config.local.sh` (`config.local.sh.example`), the
> shell over both: `MTP=2 SEQS=16 ./serve.sh`.
>
> **What changed against the copy** (the recipe's numbers below were measured
> before these; none of them changes the model's math except where noted):
>
> - **Tool-call parser fixes**, blazux's patches 12 and 13 at `2ba2d77`, their
>   last commit on this same base image, where they applied with `--fuzz=0`:
>   a tool call the model shows as an example inside a fenced code block, or a
>   `<function=` header mid-line, stays text instead of becoming a call that
>   drops the rest of the answer; a quoted marker in reasoning no longer
>   swallows the final answer ("Tool-call parser" below). Needs `./build.sh`;
>   the launcher warns when the image predates them.
> - **Cached tokens in every reply** (`--enable-prompt-tokens-details`, as
>   blazux #43 on this base), and a **deterministic MoE token alignment**
>   (`DET_MOE_ALIGN=1`, default, patch 15 below): `../bench/consistency.py`
>   found three cold runs of one greedy prompt apart from the first token's
>   logprobs (0.16-0.45 nats), with `MARLIN_ATOMIC_ADD` on or off; with
>   `DET_MOE_ALIGN=1` cold, cold again and cache hits gave the same text and
>   the same first-token logprobs (gap 0.0000, 4K and 32K), at no
>   measurable cost in decode.
> - **Port 8888** and model ids `qwen3.8-flash-next` plus `qwen` (the old id,
>   still answered); `PORT=8000` in `config.local.sh` keeps old clients.
> - **`RESTART=no`** by default, like every server here; `unless-stopped`
>   (the recipe's default: back after a reboot) only where this is the
>   machine's one server.
> - **New knobs**: `BATCHED_TOKENS` (the prefill chunk, was fixed at 8192),
>   `MARLIN_ATOMIC_ADD` (was fixed on), `CPUSET`, `EXTRA_ENV` (replaces a
>   container variable of the same name: docker keeps the first of two).
> - **`prepare.sh`**: relative paths now work (it changed directory before
>   using them); Intel's and Qwen's repos are pinned by revision
>   (`SRC_REVISION`, `FP8_REVISION`), as `./download.sh` pins the published
>   artifacts.
> - **This project's guards**: the one-server-at-a-time checks both ways,
>   page cache dropped and memory compacted before a start
>   (`../scripts/mem-prep.sh`, `MEM_PREP`), and `build.sh`, `download.sh` and
>   `prepare.sh` refuse while anything runs on the GPU (a Spark has hung on a
>   large copy beside a GPU job).
> - **Benchmarks**: the recipe's `bench/perf.py`, `longctx.py` and `common.py`
>   were an earlier copy of this project's; `../bench/` has the current ones
>   (they read the sidecar with `GB10_METRICS_PORT=18400`).
>   `bench/decode_bench.py`, `concurrency_bench.py` and `ppwatch.sh` stay here.
> - The GitHub CI workflow is gone (it does not run from a subfolder);
>   `ruff.toml` and the CPU tests stay ("What's in here").
>
> Not taken from blazux's later work on this base: the fp8 KV cache on the
> QSA path (a 1M context on one box; it patches files of the model this
> container image carries, which could not be checked here), the M%4 padding
> for the fp8 GEMM (a no-op with prefix caching on) and the load-time patches
> for NVIDIA's NVFP4 checkpoint. blazux's later bases (official vLLM 0.31 /
> 0.32) need a port of every patch here.

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**: **~50–60 tok/s
single-stream decode with MTP=3** (~2,100 tok/s prefill), working **prefix
caching**, and a **never-evict pin** that keeps your system prompt's KV resident
through arbitrary traffic.

Forked from **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**,
which established the foundation this fork stands on: the ~49 GiB fp8 n-gram ("PLE")
table is a pure lookup that a token only touches 16 rows of, so it is served
**from NVMe via `mmap`** instead of living in the 128 GB unified pool
(full story: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)). Upstream serves the
official **NVFP4** checkpoint at 25–28 tok/s; for that path and its tuning
guide, use upstream. This fork replaces the checkpoint with
**[Intel's W4A16 AutoRound int4](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)**
plus an int8 GPTQ lm_head and blockwise-fp8 side layers (all prepared by
CPU-only tools in `tools/`), an **fp8** PLE table, and a set of GB10/vLLM
patches — roughly **1.8× faster decode** than the NVFP4 recipe on the same box.

> **This repository** is a modified copy of
> [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
> (all of the above and every patch below). On top of it: an engine-side
> Prometheus sidecar (PLE gather cost, mamba guard, pin gauges), API-key auth,
> `BIND_ADDR`, a container healthcheck and log rotation, a resumable
> `prepare.sh`, chat-API benchmarks (since moved to this project's `../bench/`) and CI.

> **Upstream's NVFP4 recipe independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10), which also
> contributed the concurrency findings below.

| | **this fork (int4/int8/fp8)** |
|---|---|
| Prefill | **~2,100–2,200 tok/s** |
| Decode, single stream |**~50–60 tok/s (MTP=3, thinking on); 74.5 tok/s on code with thinking off** (`perf.py`, 2026-11-03) |
| Prefix caching | **on** (+ never-evict pin) |
| Context | 262k native / 500k YaRN |
| Weights resident | **~68 GiB** |

## Throughput and concurrency

Measured on this stack with the default `serve.sh` config (DGX Spark, default GPU
clocks; int4 MTP draft experts, `DRAFT_VOCAB=1`, MTP=3, prefix caching on, `SEQS=16`,
8192-token prefill chunks). One caveat: on this box the PLE table is served from
another machine's RAM over RDMA (the `magi` branch), the fastest row source there is —
with the table on a regular local NVMe, as in the Quickstart, expect decode about
2–3% slower than the numbers below. Single-stream decode by workload, **thinking enabled** —
reproduce with
[bench_qwen35.sh](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/blob/master/bench_qwen35.sh)
(from albond's 122B recipe) pointed at your endpoint, on a freshly started container:

| Task | Prompt Tokens | Gen Tokens | Time (s) | Speed (tok/s) |
| --- | --- | --- | --- | --- |
| **[Q&A]** | 65 | 57.5 ± 5.2 | 1.18 ± 0.17 | 49.1 ± 2.6 |
| **[Code]** | 72 | 275.8 ± 2.2 | 4.89 ± 0.24 | 56.5 ± 3.1 |
| **[JSON]** | 90 | 841.8 ± 20.8 | 13.54 ± 0.46 | 62.1 ± 0.7 |
| **[Math]** | 71 | 64.0 ± 0.0 | 1.15 ± 0.01 | 55.5 ± 0.2 |
| **[LongCode]** | 79 | 2048.0 ± 0.0 | 42.33 ± 2.53 | 48.5 ± 2.8 |

*Note: arithmetic mean ± sample standard deviation over 4 runs (2 scripts × 2 runs, 2026-09-09). Draft acceptance with thinking on is ~70% (2.1 of 3); with `enable_thinking: false` it is ~87% and decode runs 5–10% faster than the table.*


### Measured here with `bench/perf.py` (local NVMe table)

DGX Spark / GX10, the recipe's `serve.sh` of the time (MTP=3, `SEQS=8`, `KV_BYTES=30g`,
`DRAFT_VOCAB=1`, table on local NVMe), second run after a restart (see
*Warmup* below), code prompt with thinking off, 2026-09-23:

| | |
|---|---|
| TTFT, short prompt | 129 ms |
| Single-stream decode | **65.8 tok/s** (MTP 3.5 tok/step, 84% of drafts accepted) |
| Aggregate, 8 streams × 300 tokens | **259 tok/s** (35 tok/s per stream, TTFT 0.47 s) |
| 16 streams | requests queue above `SEQS=8` (4.7 s/req): aggregate measures admission |
| Prefill, 2k / 8k / 33k / 105k unique tokens | 2,103 / 2,302 / 2,244 / 2,115 tok/s |

Prefill prompts are unique from their first token: shared-prefix filler lets
the prefix cache serve part of every longer prompt and inflated this same box
to "2,860 tok/s at 120k". Run `python3 ../bench/perf.py` and
`python3 ../bench/longctx.py` (N concurrent long prompts; flags queueing and KV
preemption) after any change.

Under concurrency, TTFT grows with MTP's prefill cost (see the next
section), not with the paged table. On upstream's NVFP4 path
[@jschmied](https://github.com/jschmied) measured aggregate throughput
scaling to ~267 tok/s at 48 streams
([load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md));
two portable takeaways: per-token page-fault cost *falls* with concurrency
(batched tokens share n-gram rows), and a low `--max-num-seqs` silently
queues requests — check `vllm:request_queue_time_seconds_sum` before quoting
an aggregate number.

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint + fp8 PLE table, on reasonably fast
  storage (the table is read at runtime — NVMe strongly recommended).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120) for testing, though this is tuned for the Spark.

## Quickstart

The commands at the top of this page. By hand, without the scripts here:

```bash
docker build -t qwen38-flash-dgx .
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN --local-dir /models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 --local-dir /models/ple-table-fp8
# option: the same checkpoint with the MTP draft experts left in bf16 (+3.5 GiB, ~3% slower decode):
#   hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid --local-dir /models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
# (or build them yourself from Intel's release: ./prepare.sh — see below)
. ./config.sh && scripts/serve-intel-ar.sh   # boots on :8888 (~5 min with fastsafetensors)
docker logs -f qwen38-flash                  # wait for "Application startup complete"
python3 ../bench/perf.py --only warmup       # optional: compile first-use kernels now (see Warmup)
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

`./serve.sh` runs `scripts/serve-intel-ar.sh` with `config.sh`'s values —
every knob is an env var. The defaults below are `config.sh`'s; where the bare
`scripts/serve-intel-ar.sh` falls back to something else, the note says so.
Keep your machine's real settings in `config.local.sh`.

## Modify the weights yourself

The quickstart's two `hf download` repos are the finished artifacts — hashes
verified against the local originals. If you'd rather build (or audit) them
yourself from
[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound),
one script runs the whole pipeline (CPU-only — a NAS box is fine):

```bash
./prepare.sh /models/Qwen3.8-Flash-Next-W4A16-AutoRound /models/ple-table-fp8
```

Each step is explained in `prepare.sh`'s header comments: int8 lm_head repack,
fp8 side-layer conversion, n-gram index strip, fp8 table fetch, the
`quantization_config` rewrite, and the int4 MTP-draft-experts variant. On that last one: this vLLM build has no
auto-round loader, but its GPTQ config (`AutoGPTQConfig` → Marlin kernels)
reads the same packed tensors — the GPTQModel-style `dynamic` rules exclude
the families that are not int4-packed and flip the head to 8-bit. The original
AutoRound config is kept as `config.json.autoround`.

The script is resumable: if a step fails (network, disk), fix the cause and
run the same command again — finished steps are skipped, and no tool ever
re-quantizes its own output or overwrites a backup (`tools/test_tools_cpu.py`
checks exactly that). The download also skips shards that hold only the
n-gram table (dropped from the index anyway) when the `huggingface_hub`
Python package is importable.

## Serving

```bash
docker build -t qwen38-flash-dgx .
MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN \
TABLE_DIR=/models/ple-table-fp8 \
PREFIX_CACHE=1 PIN_PROMPT="You are HomeBot, the household assistant." \
scripts/serve-intel-ar.sh
```

or set them in `config.local.sh` and run `./serve.sh`.

| Var | `config.sh` default | Notes |
|---|---|---|
| `MODEL_DIR` / `TABLE_DIR` | `/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN` / `/models/ple-table-fp8` — edit to your paths | Prepared checkpoint / fp8 PLE table dirs (must exist; checked before the running container is touched) |
| `PORT` | `8888` | API port, as every server in this project (the recipe's earlier default: `8000`; bare script: `18300`) |
| `CTX` | `262144` | Max context |
| `YARN` | `0` | `1` = Qwen's YaRN rope scaling (factor 4) past the native 262144 — set `CTX` too (500k was upstream's validated ceiling). Also forces the MTP draft's `max_model_len`, which `--hf-overrides` alone doesn't reach |
| `SEQS` | `8` | Max concurrent sequences (don't benchmark with 1–2, see below) |
| `GPU_MEM` | `0.01` | Near-zero pool fraction, paired with `KV_BYTES`: deterministic sizing, so the driver never oversubscribes the unified pool (`NV_ERR_NO_MEMORY` / Xid 31 freezes). Bare script: a `0.85` fraction — avoid on unified-memory boxes. |
| `KV_BYTES` | `30g` | Explicit KV pool size, passed as `--kv-cache-memory-bytes` (bare script: unset). vLLM reads `g` as 10⁹ bytes: `30g` = 27.9 GiB, measured to hold ~4 concurrent 127k contexts. When to raise it to `35g`/`40g`: see [Sizing the KV pool](#sizing-the-kv-pool-30g-35g-or-40g) |
| `MTP` | `3` | Speculative tokens from the MTP head (`0` = off; bare script: `2`) |
| `PREFIX_CACHE` | `1` | Prefix caching — fixed and recommended on this fork (bare script: `0`) |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (patch 9; @jschmied, vllm#55122): identical output at T=0 at full prefill speed. `0` = stock kernel (non-deterministic, may drop attention candidates) |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (patch 9; deterministic, −20–40% on long prefill). Wins over `DET_TOPK` when set |
| `DRAFT_VOCAB` | `1` | The MTP drafter scores only the 65,536 most frequent tokens (patch 10; from upstream blazux): +3–5% decode on English/code, draft acceptance unchanged there (thinking on or off). The shipped id set is English/code-weighted — **CJK-heavy output loses acceptance and decode speed with it**, so set `0` (full vocabulary) or build your own set with `tools/build_draft_vocab.py` over your traffic |
| `DRAFT_HEAD` | `int8` | The drafter's head as the target has it: int8 GPTQ-Marlin, shared (`DRAFT_VOCAB=0`) or, with `DRAFT_VOCAB`, its own rows for those ids (160 MiB per draft step; images built before 2026-11-03 made that slice bf16, 320 MiB). `bf16` = that bf16 slice, for comparison. `int4` = a private int4 g128 RTN GPTQ-Marlin copy of the full-vocabulary head (built at first use, ~313 MiB vs 616 MiB read per draft step): measured a wash, tg unchanged within noise, draft acceptance 1–8 points lower (see the MTP options section) |
| `PIN_PROMPT` / `PIN_MAX_FRACTION` | unset / `0.25` | Never-evict pin (patch 6); needs `PREFIX_CACHE=1` (the launcher warns otherwise). Use a distinctive substring of a few dozen characters — a word or two matches unrelated prompts |
| `API_KEY` | unset | Non-empty → Bearer auth on the API (vLLM `--api-key`; generate with `openssl rand -hex 32`). `/health` stays open; the key is visible in `docker inspect` |
| `BIND_ADDR` | unset | Host address for the published ports. Unset = all interfaces; `127.0.0.1` = this machine only (e.g. behind a reverse proxy). The metrics sidecar has no auth even with `API_KEY` |
| `FP8_HYBRID` | `1` | int4+fp8 hybrid dispatch (patch 4) |
| `PLE_MADV_RANDOM` | `1` | `MADV_RANDOM` on the table mmap (patch 1): no readahead around 160-byte row faults — upstream (blazux `0c6df7e`) measured 4–8% faster cold prefill and a cleaner page cache, now the default |
| `METRICS_PORT` | `18400` | Engine-side Prometheus sidecar (`src/vllm_custom_metrics.py`): `vllm:ple_mmap_*` counters (ops, op/gather ms, rows, bytes, prefetch hit/miss), `vllm:mamba_state_copy_guard_total` (tripwire, expect 0) and `vllm:never_evict_{blocks_reserved,pin_queue_blocks,pin_bytes}` gauges. vLLM's own `/metrics` runs in the API-server process and cannot see EngineCore counters, so these are served from the engine process on this separate port; `0` = off. The launcher probes the image for both the module and its start hooks, and warns when the sidecar cannot come up (the port would just stay closed) |
| `PLE_PREFETCH` | `0` | Batch-assembly prefetch — measured not worth enabling (see appendix) |
| `PLE_CHUNK` / `PLE_FAST_ROWS` / `PLE_STATS_SEC` | `2048` / `512` / `0`* | Gather rows per task / decode fast-path threshold / stats-log period. The log lines are off by default since the numbers moved to the metrics sidecar (`METRICS_PORT`); set `30` to bring them back. *On an image without the sidecar the launcher keeps them at `30` |
| `HIT_DEBUG` | `0` | Prefix-cache tracing (patch 8) |
| `HIT_DEBUG_N` | `0` | Event budget for `HIT_DEBUG=1`; tracing auto-disables with a warning once spent |
| `STEP_PROFILE` | `0` | `1` — on-demand torch profiler, triggered by `touch /tmp/profile_trigger` inside the container (patch 11) |
| `PREWARM` | `1` | Stream the table once at boot to warm the page cache |
| `WORKERS` | `32` | Threads for the mmap gather |
| `LOAD_FORMAT` | `fastsafetensors` | Noticeably faster cold boots |
| `TOOL_PARSER` | `qwen3_xml` | Tool-call parser (bare script: `qwen3_coder`); both run vLLM's `qwen3` parser configuration, which patches 12–13 fix |
| `SERVED_NAME` | `qwen3.8-flash-next qwen` | Model ids on the API, space-separated; replies carry the first. `qwen` was the recipe's id (bare script: `qwen3.8-flash-next`) |
| `ITER_DETAILS` | `0` | `1` = per-step prefill metrics: `vllm:scheduled_ctx_tokens_total` updates every engine step (stock `prompt_tokens_total` only moves when a prefill finishes). Live view: `bench/ppwatch.sh` |
| `FLASHINFER_AUTOTUNE` | `0` | `1` — enable flashinfer kernel autotuning (longer warmup, possibly faster kernels) |
| `CUDA_LAUNCH_BLOCKING` | `0` | `1` — synchronous CUDA errors, for debugging Xid 31 (much slower; not for production) |
| `RESTART` | `no` | Container restart policy. `unless-stopped` (the recipe's earlier default; bare script too) survives reboots and crashes — only where this is the machine's one server, else two servers start at boot. The next `./serve.sh` re-creates the container with this value |
| `BATCHED_TOKENS` | `8192` | Prefill chunk (`--max-num-batched-tokens`). Smaller keeps other streams decoding during a long prefill, at a prefill cost (blazux's table, NVFP4 base: a decoding client at 0.2 / 0.4–0.6 / 1.0 / 1.6 tok/s during a big prefill at 8192 / 2048 / 1024 / 512, long prefill −17–36% at 1024). `../bench/perf.py --only interference` measures it here |
| `MARLIN_ATOMIC_ADD` | `1` | `VLLM_MARLIN_USE_ATOMIC_ADD`: Marlin's split-K reduction by atomic adds, for speed, in the dense Marlin GEMMs; the MoE kernel never uses it (`marlin_moe.py`: `use_atomic_add=False`). `0` changed nothing in `consistency.py` here; see patch 15 |
| `DET_MOE_ALIGN` | `1` | deterministic MoE token alignment (patch 15): the same prompt gives the same logprobs every run, cold or from the cache (measured), no measurable decode cost. `0` = vLLM's kernel |
| `DET_MOE_ALIGN_MIN_TOKENS` | `auto` | the steps it applies to, from this many tokens up; `auto` = `SEQS` × (1 + `MTP`) + 1 (33), one more than the largest decode-only step: prefill takes it, decode keeps the kernel. `9` = every step where an expert can fill more than one block |
| `CPUSET` | empty | `docker --cpuset-cpus`; `5-9,15-19` = GB10's Cortex-X925 cores (blazux's concurrent profile pins to them). Not measured on this recipe |
| `EXTRA_ENV` | | `NAME=value` pairs for the container; one with a name the launcher already sets replaces it |
| `IDLE_RELEASE_SEC` | `60` | after this many seconds without a request the engine calls `torch.cuda.empty_cache()`: torch's freed GPU blocks (host memory on GB10, kept at the last large request's peak; 8 GiB after a long-context run here) go back to the system, CUDA graph pools stay; once per rise; a line in `docker logs`. Once per idle period also `malloc_trim(0)`: the engine's freed heap back to the system instead of to swap later (a line when it gave back ≥ 64 MiB; `EXTRA_ENV=VLLM_IDLE_MALLOC_TRIM=0` turns it off). `0` = keep them |
| `NOSWAP` | `0` | `1` = the container may not use swap: `--memory` / `--memory-swap` at the host's RAM (no limit, `memory.swap.max` = 0, checked after the start), as the SGLang servers here run. Under a long prefill's pressure the kernel then drops page cache (the PLE table's clean pages) instead of writing the engine's memory to swap on the disk the table is read from. Measured 2026-11-03: no gain (swap hardly moves; the limit is available memory), see "Long prefill and the host's memory" |
| `HC_FP8` | `1` | The hyper-connection projections (`input_mix_weight_down` + block_inject, `input_mix_weight_up`; ~13.5 MB of BF16 a mix, ~98 mixes a decode step) in FP8 with a scale per row, on a Triton GEMM for decode inputs (patch 16): decode 74.5 tok/s against 69.7, HumanEval 96.3% either way, prefill within run-to-run spread. On since 2026-11-03. Lossy (answers differ a little from BF16's, deterministically); `0` = BF16 as shipped |
| `WATCHDOG` | `0` | `1` = `./serve.sh` starts `./watchdog.sh` in the background (`./stop.sh` stops it): it restarts the server when available memory is under `WATCHDOG_MIN_AVAIL_MB` (`2048`) **and** no request has run or waited for `WATCHDOG_IDLE_MIN` (`5`) minutes (vLLM's `/metrics`, read again just before the stop), at most once per `WATCHDOG_COOLDOWN_MIN` (`60`), checking every `WATCHDOG_INTERVAL` (`60`) s; never starts a stopped server, stops itself if a restart fails. Costs the prefix cache and ~5 min. Log: `~/.cache/gb10/vllm-flash-ar-watchdog.log` |
| `PROFILE` / `PROFILE_DIR` | `0` / `/tmp/gb10-profile` | `1` = vLLM's torch profiler endpoints, traces to `PROFILE_DIR` (mounted at the same path): `python3 ../bench/profile_decode.py` then shows where a decode step's GPU time goes, per kernel group. Costs nothing until a profile is started. `PROFILE_SHAPES=1` adds op input shapes and runs eagerly (no CUDA graphs), so each top kernel names its weight and a second table gives the time per layer and the rate it reads its weight at (GB/s): for finding layers and how far each is from the memory's bandwidth. Kernel times hold; the gaps between them are eager-mode ones |
| `MEM_PREP` / `WAIT_MIN` | `1` / `15` | `./serve.sh`: page cache dropped and memory compacted before the start (`../scripts/mem-prep.sh`); wait up to N minutes for the API (`0` = return at once) |
| `LOG_MAX_SIZE` / `LOG_MAX_FILE` | `10m` / `3` | Docker log rotation — `PLE mmap stats` logs a line every 30 s, so an unbounded log grows to GBs |
| `NAME` / `IMAGE` | `qwen38-flash` / `qwen38-flash-dgx` | Container / image names (`IMAGE` is not overwritten by `./build.sh` unless you build to it — point it at a backup tag to roll back) |
| `EXTRA` | | Extra vLLM flags, passed verbatim |

The launcher probes the image before touching the running container and warns
when a knob needs a patch the image lacks (`DET_TOPK`, `EXACT_TOPK`,
`ITER_DETAILS`, `DRAFT_VOCAB`/`DRAFT_HEAD`, `METRICS_PORT`) or lacks the
tool-call parser fixes — rebuild with
`docker build -t qwen38-flash-dgx .` rather than trusting a silently ignored
env var.

The launcher also fail-fasts on unedited `/path/to` placeholders or missing
`MODEL_DIR`/`TABLE_DIR` *before* it removes a running container, and creates
the container with a `/health` HEALTHCHECK (a 10-minute start period covers
the weight load; 10 consecutive misses → `unhealthy` in `docker ps` — a
restart on unhealthy needs an external watcher such as `willfarrell/autoheal`)
and json-file log rotation. The bench and smoke-test scripts read `BASE` /
`MODEL` / `PIN` / `API_KEY` from the environment (where applicable), so they
work unchanged with a non-default port or with auth on. Their defaults match
`config.sh` (`http://localhost:8888`, model `qwen`); `scripts/smoke-test.sh`
also takes `host:port` as its argument and, without `MODEL`, uses whatever
`/v1/models` reports.

For graphs, scrape **two** endpoints: vLLM's own `/metrics` on `PORT` (request
throughput, cache usage, queue time) and the engine-side sidecar on
`METRICS_PORT` (default 18400) — PLE gather cost
(`rate(vllm:ple_mmap_op_ms_total)/rate(vllm:ple_mmap_ops_total)` = ms/op), the
mamba state-copy guard tripwire and the never-evict pin footprint. With the
sidecar on, the `PLE mmap stats` log lines default off (`PLE_STATS_SEC=0`).

### Sizing the KV pool: 30g, 35g or 40g

**Keep the default `KV_BYTES=30g` unless you run five or more long (100k+
token) sessions at the same time.** Measured on a GX10 (local NVMe table,
MTP=3, `SEQS=8`; `bench/longctx.py --ctx 120000 --streams 8 --gen 1000` and
`bench/perf.py --only prefill` (now `../bench/`), warmed up, 2026-09-24):

| | `30g` (default) | `35g` (estimate) | `40g` |
|---|---|---|---|
| KV pool | 27.9 GiB | 32.6 GiB | 37.3 GiB |
| Long contexts resident at once (~127k each) | **~4** (0.55M tokens) | ~5 (≈0.65M) | **~6** (0.77M) |
| Preemptions, 8 × 127k | 0 | ? | 2 |
| Long-prefill TTFT vs 30g (32k / 101k prompt) | 14.6 s / 50.0 s | ≈ +2–3.5% | +6.7% / +4.3% |
| Single-stream decode | ~66 tok/s | ~66 tok/s | ~66 tok/s |

The 35g column is interpolated from the two measured ones, not measured —
run the two commands above if you pick it.

Why it is a trade-off:

- **What more KV buys.** vLLM admits a request only when its KV blocks fit,
  so on a full pool extra long requests *wait* (`queue` in `longctx.py`)
  rather than fail. More KV = more long sessions decoding at once (agents
  with big contexts), and more room for the prefix cache to keep several long
  conversations warm between turns. It does **not** speed up a batch of long
  prompts: prefill runs one prompt at a time either way (8 × 127k took
  548 s at 30g and 564–576 s at 40g).
- **What it costs.** KV memory is pinned by the driver, so every GiB comes
  out of the page cache that holds the ~48 GiB PLE table. At 40g ~2.5 GB of
  page cache was left, n-gram row reads per 8k prefill chunk rose from
  ~184 to ~273 ms at 101k, and long prefill slowed 4–7%. Decode is
  unaffected (its hot rows stay cached). At 40g the pool also runs at ~99%
  under load, so admitted requests occasionally get preempted and recomputed.
- **Memory safety.** 40g ran for hours of mixed load without hangs; the
  kernel moved ~6 GB of cold vLLM startup memory to swap once and did not
  page it back (`vmstat` `si` ≈ 0). Check the same on your box — `vmstat 5`
  during load (`si`/`so` should stay near 0) and `cat /proc/pressure/memory`
  (`some avg10` near 0) — especially if other programs share the machine.
- **Ignore the boot log's capacity.** "GPU KV cache size: 966,390 tokens" at
  30g assumes ~31 KB/token; on long contexts the measured cost is ~53 KB/token
  (the mamba/GDN state groups hold more than that estimate assumes), i.e.
  ~0.55M tokens. `../bench/longctx.py` prints the real figure for your settings.

So: **30g** for chat, coding assistants and up to ~4 concurrent long
contexts (fastest prefill, no preemption); **35g** if you regularly have
exactly one more long session than that; **40g** for 5–6 concurrent long
sessions, accepting ~5% slower long prefill.


## Limitations & notes

- **One big model at a time** — and on a Spark the OS and GPU share the pool;
  prefer the deterministic `GPU_MEM=0.01` + `KV_BYTES` sizing over a large
  fraction (an OOM inside the unified pool can freeze the box).
- **earlyoom stops the engine when the host runs out.** With the server up the
  host keeps only a few GiB available (3-7 in perf.py's headers here); below
  earlyoom's SIGTERM line (1 GiB once `../scripts/host-tuning.sh` is applied,
  ~8 GB by DGX OS default) it stops the engine first. In `docker logs`:
  `EngineCore: trigger received signal=SIGTERM`, ~10 s later the API server's
  `engine core exited unexpectedly` and an `EngineDeadError` traceback, the
  container exiting 0; `journalctl -u earlyoom` names it. On 2026-11-02 that
  happened with the server idle for 88 minutes after a long-context run (1015
  MiB available, 6 GiB of swap in use). `../scripts/mem-watch.sh` beside the
  next run (GX10, 30g, multi-stream long contexts) showed where the memory is:
  of 121.6 GiB, ~108.5 GiB held by the GPU driver, 105.5 of it by torch's
  allocator in the engine, which kept 8.2 GiB of freed blocks (reserved 105.5,
  allocated 97.3 GiB); ~4 GiB of page cache (of the 48 GiB table), 3.5 GiB of
  process heaps, ~2 GiB available under load. And idle, with no request, the
  available memory fell ~7 MiB a minute (~0.45 GiB an hour), all of it outside
  torch's allocator (its numbers did not move; the driver-held total grew);
  stopping the server stopped it, and neither the health check (30 runs: no
  change) nor the desktop (Xorg 18 MiB, gnome-shell 6 MiB) is it. Not found
  yet; `mem-watch.sh` now lists the driver's per-process count (`gpu:`) to say
  which process. Against both, `IDLE_RELEASE_SEC=60` (default) hands the
  allocator's freed blocks back after a minute without requests (8 GiB there:
  ~19 h of that leak instead of ~2), `KV_BYTES=25g` gives 4.7 GiB more for
  good, and `WATCHDOG=1` restarts the server itself, idle and before earlyoom's
  line, instead of earlyoom stopping it whenever (off by default). With both
  on, about a day of real use on that GX10 (2026-11-02/03, 30g) needed no
  restart: each burst's idle release gave back 2.3-3.6 GiB, the allocator's
  hold returning to 98.74 GiB every time (97.32 GiB allocated). A large
  download, copy or build beside the server gets there for sure: `./build.sh`
  and `./download.sh` refuse beside a GPU job (`../scripts/mem-prep.sh idle`);
  a hand-run `hf download` does not, so put that check in front of it.
- **Long prefill and the host's memory** (measured 2026-11-03). The 48 GiB
  PLE table lives in the page cache, and a unique prompt reads rows of it that
  are not there yet; a long prompt also grows torch's allocator by its
  temporary buffers. `../bench/perf.py --only prefill` prints, per prompt, how
  low available memory went and what had moved at that moment (driver = CUDA
  allocations, page cache, anon), the swap written, the seconds tasks waited on
  memory (PSI), and with `GB10_METRICS_PORT=18400` torch's growth. On the GX10
  (30g, `HC_FP8=1`, one boot each, run 5 minutes after the start):

  | prompt | torch grew | available, low | `NOSWAP=0`: tok/s | `NOSWAP=1`: tok/s |
  |---|---|---|---:|---:|
  | 2K | +0.2 GiB | 3.9 / 3.6 GiB | 1891 | 2014 |
  | 8K | +1.4 GiB | 2.5 / 2.9 GiB | 1782 | 1848 |
  | 32K | +1.5 GiB | 1.3 / 1.4 GiB | 2054 | 1945 (2375 reclaim stalls) |
  | 101K | +0.0 GiB | **1.0 / 0.9 GiB**, 1.0-1.2 s stalled | 1831 | failed |

  Swap hardly moved in either run (≤ 271 pages), so `NOSWAP=1` changed
  nothing that matters and stays off. What matters is the budget: about 4 GiB
  available after the start, the allocator taking ~3 GiB of it on the first
  8K-32K prompts (100.3 -> 103.4 GiB reserved; the idle release hands it back
  after a minute, so the next long prompt takes it again), and the 101K prompt's
  page-cache churn then running at ~1 GiB, earlyoom's SIGTERM line; the
  `NOSWAP=1` run's 101K request failed there. With that little page cache the
  PLE gather also slows: ~260 ms of disk reads per 8K chunk at 101K, against
  ~184 ms measured at 30g on 2026-09-24. The perf line now flags a prompt that
  takes available memory under 1.5 GiB. More room comes from `KV_BYTES`
  (`28g` +1.9 GiB, `27g` +2.8 GiB, ~5-10% fewer long-context tokens resident)
  or from smaller prefill chunks (`BATCHED_TOKENS`, or
  `EXTRA="--long-prefill-token-threshold 4800"`: smaller buffers, more steps).
  The idle `malloc_trim(0)` logged nothing in these runs (under 64 MiB freed).
- **1M context is out of reach on one box**: the QSA layers refuse an fp8 KV
  cache, and in bf16 a single 1M request needs ~30 GiB of KV. 500k with YaRN
  was upstream's validated ceiling.
- **Weights are not included** and the checkpoint carries Qwen's license (with
  a MAU/revenue clause) — review it before production use.

## What runs in what precision

| Component | Precision | How |
|---|---|---|
| 512-expert MoE, 48 main layers | **int4** GPTQ-Marlin g128 | Intel checkpoint as-is |
| MTP draft layer's own 512 experts | **int4** GPTQ-Marlin g128 RTN (the default `-MTP_int4RTN` checkpoint) | Intel leaves layer 48 in bf16 (~4.7 GiB, `-:.*layers\.48\..*`); `tools/quantize_mtp_experts_int4.py` (patch 10) makes it int4 on the Marlin path: −3.5 GiB, +2–4% decode, acceptance unchanged. The plain `-hybrid` repo keeps them bf16 |
| lm_head (shared with MTP draft head) | **int8** GPTQ-Marlin (uint8b128) | `tools/quantize_lm_head_int8.py` + `"lm_head": true` |
| GDN in/out projections, QSA q/k/v/o, shared expert | **fp8** blockwise e4m3 (128×128) | `tools/fp8_convert.py` + `src/vllm_fp8_hybrid.py` |
| Hyper-connection projections (input_mix down + block_inject / up) | **fp8** e4m3, one scale per row (block_inject rows bf16) | converted at load (`HC_FP8=1`, default; patch 16); bf16 in the checkpoint |
| Embeddings, other hyper-connection weights, norms, MoE gates, fc_hidden | bf16 | excluded via `dynamic` rules |
| PLE n-gram table (51B params, layer 1) | **fp8** rows, mmapped from disk | `tools/fetch-ple-table-fp8.sh` + the mmap patch |
| KV cache | bf16 | QSA refuses fp8 KV |

## The patches

Everything is applied at image build time (see the `Dockerfile`); each patch is
independent and gated by an env var where it changes behavior. The build is
fail-fast — patch scripts assert their anchors — and a successful build prints
one confirmation per step: `ple_layer.py patched OK`, `fla shmem gate patched`,
`fla num_warps pinned`, `auto_gptq.py patched OK`, `never-evict pin patched OK`,
`lm_head patched OK in model.py + mtp.py`, `mamba_utils.py guarded OK`,
`patch_hit_debug.py applied OK`, `patch_mamba_align_split.py applied OK`,
`qsa.py: top-k variants (1|fill) added OK`, `qsadet INSTALLED`, `qsadet wired OK`,
`patch_step_profile.py applied OK`, `loggers.py: per-step prefill metrics added
OK`, `draft-head hook INSTALLED` and `FP8 HC hook INSTALLED`. A missing line means
that step did not run.

### 1. PLE mmap upgrades (`src/vllm_ple_mmap.py`, extends upstream's patch)

- **Any table dtype**: bf16/f16 tables and fp8 (with `weight_scale`) are all
  accepted; row size is derived from the safetensors headers. The fp8 table
  halves the bytes read per token vs bf16.
- **`VLLM_PLE_MMAP_DIR`**: the table no longer has to live inside the
  checkpoint dir — point it at any directory of safetensors shards (NFS, local
  NVMe, a RAM-backed device...). The backend matters: the ~49 GiB table
  outgrows what the page cache can keep warm next to the model, so gathers
  cost ~1.3 ms/op from a RAM-backed source vs 5–9 ms from local NVMe vs
  30–50 ms over NFS — decode impact in the appendix below.
- **Hot path**: per-step dedup of row ids on CPU (`np.unique`), gather of
  unique rows only, staging through a persistent pinned buffer with an async
  H2D copy, and GPU-side expansion via the inverse index. A decode fast path
  (`VLLM_PLE_MMAP_FAST_ROWS`, default 512) skips the thread pool entirely for
  small gathers. Net effect: ~7.2 → ~2.5–3.8 ms per lookup op on a RAM-backed table.
- **`VLLM_PLE_MMAP_MADV_RANDOM=1`**: `madvise(MADV_RANDOM)` the mmap so faults
  stay single-page — for tables on remote RAM or boxes with no page-cache
  headroom.
- **Stats**: lifetime counters go to the metrics sidecar (patch 13); with
  `VLLM_PLE_MMAP_STATS_SEC=30` the engine also logs
  `PLE mmap stats (last Ns): calls, op ms, gather ms, rows, MB` per period.
- **Fail-fast table check**: every shard `0..parts-1` must be present at load
  time — a missing one used to surface mid-serving as a `TypeError`.

### 2. FLA shared-memory gate (`Dockerfile` sed)

sm_121 reports 99 KiB of shared memory per block — the same as ADA, where the
flash-linear-attention Triton kernels use their big GDN tiles — but the gate in
`vllm/third_party/flash_linear_attention/ops/utils.py` demands 100 KiB, so all
36 GDN layers silently fell back to small tiles. Lowering the gate to 99 KiB
(101376) lets GB10 take the big-tile path. (Found the hard way in the
Qwen3.5-122B Spark recipe — ported from
[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)'s
`patch_fla_shmem.py`.)

### 3. int8 lm_head enablement (`Dockerfile` sed on `model.py` / `mtp.py`)

Upstream constructs `ParallelLMHead` without `quant_config`, forcing a bf16
head (1.27 GiB, and a bf16 GEMV per token over a 248320 vocab). One added
kwarg in both the main model and the MTP draft lets the head pick up the
checkpoint's int8 GPTQ packing. Without the `mtp.py` half, MTP ≥ 3 crashes at
load ("no module or parameter named 'lm_head.qweight'").

### 4. int4+fp8 hybrid dispatch (`src/vllm_fp8_hybrid.py`, `VLLM_FP8_HYBRID=1`)

vLLM's GPTQ config quantizes listed layers and leaves the rest to
`UnquantizedLinearMethod` — it has no notion of "this excluded layer is
actually fp8 in the checkpoint". This shim wraps `AutoGPTQConfig`: it scans the
checkpoint metadata for `F8_E4M3` weights with a `weight_scale_inv` sibling and
routes exactly those layers to vLLM's blockwise-`Fp8Config`
(`weight_block_size=[128,128]`, dynamic activation scheme) while everything
else keeps the GPTQ path. `VLLM_USE_DEEP_GEMM=0` is required on sm_121
(DeepGEMM hits `CUDA_ERROR_LAUNCH_FAILED`); the triton fallback is fine.

### 5. Prefix caching: on, and fixed (`src/patch_mamba_align_split.py`)

Upstream runs `--no-enable-prefix-caching` because of a CUBLAS error in the
GDN `in_proj` GEMM on the cached-block path. With the fp8 side layers that
GEMM runs a different kernel, and prefix caching is stable in our serving
(`PREFIX_CACHE=1`).

It also *works properly* now. On this hybrid model the reconciled cache hit is
the **minimum across all KV cache groups** (full attention + four mamba/GDN
state groups), and mamba "align" mode can only cache a state at a prefill
chunk end on a 1600-token boundary. The image's scheduler aligned those chunk
ends to `cache_config.block_size` — which the engine rewrites to the *minimum*
group block size (8, the MTP draft granularity) — so chunks ended where no
mamba state was cacheable, a cold request published **zero** reusable mamba
states, and a repeated prompt only got fast on the **3rd** try (the miss
triggers junction machinery that rebuilds the boundary one request late).
Long prompts effectively never hit. The patch makes the split use
`cache_config.mamba_block_size` (1600). Verified: an 8k-token repeat goes
10.1 s → **0.90 s on the 2nd request**.

The same rewritten `block_size` also poisoned the **worker** side: the
align-mode state-slot seed (`mamba_hybrid.py`) divided by it too, so a prefix
hit at 6400 tokens seeded state column 799 instead of 3, read past the
block-table row and restored a wrong (often all-zero / stale) mamba state —
greedy outputs visibly changed on cache hits. Root-caused upstream by
[blazux](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2#issuecomment-546252046)
(his fix: `8347e7c`); the same one-line seed fix is folded into this patch.
Verified: cold-vs-hit first-token logprobs now agree within the stack's
normal run-to-run jitter (Marlin atomic-add nondeterminism), where before the
fix greedy outputs diverged within the first few tokens.

Notes: the prefix-cache granularity is large (1600 tokens; shorter prefixes
get no reuse), and a repeat hit tops out at `round_down(P,1600) − 1600` — MTP
(eagle-style) always recomputes the last matched block.

### 6. Never-evict prompt pinning (`src/patch_never_evict.py`)

`--never-evict-kv-cache-prompt-includes "<substring of your system prompt>"`
pins the KV blocks of any prompt containing that marker: they are held in a
side queue on `BlockPool`, excluded from the free count, and thus never handed
out for eviction — your assistant's system prompt stays cached no matter what
other traffic does. `--never-evict-kv-cache-max-fraction` (default 0.25) caps
the pin. The pin set is *replaced* on each matching request, so an updated
system prompt releases the old blocks automatically.

Verified end-to-end: a pinned 8k prompt still answers in **0.94 s after 2M
tokens of unique traffic** (3.1× full KV-pool turnover).

Implementation notes: the marker is tokenized once and matched as a token-id
subsequence (first/last token dropped — BPE merges at the boundaries); the pin
is keyed on block *hashes*, not block ids, because this hybrid model frees
mamba/GDN state blocks mid-request — each freed block is re-claimed by the pin
the moment `free_blocks()` sees it. Only prefix-cacheable KV groups
participate (the MTP draft layer's group is not, and must be skipped). This is
a pin-only port of our `arc_pin2` patch from the Qwen3.5-122B Spark stack,
which built the pin on top of the ARC GPU-eviction work in
[vllm#40270](https://github.com/vllm-project/vllm/pull/40270); the ARC/2Q
policies themselves were deliberately dropped — the stock free queue is
C-speed on a path this model hits every step.

Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" --entrypoint python3
qwen38-flash-dgx /t/test_never_evict_pin.py`.

### 7. Mamba state-copy guard (`src/mamba_utils_guarded.py`)

Hardens the align-mode state-copy kernels against the "CUDA illegal memory
access / Xid 31 under load" crash class (also
[blazux#2](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2)): backports
[vllm#50729](https://github.com/vllm-project/vllm/pull/50729) (overlapping
state-copy race) and bounds-checks every block id against its state pool
before dereferencing — an out-of-range id skips the copy and bumps a counter
(logged as `mamba state-copy guard`) instead of taking down the CUDA context.

With the block-size seed fix (patch 5) the out-of-range ids the guard was
absorbing are gone at the root: the counter is expected to stay at **0**, and
a nonzero count is logged as an *error* — it now indicates a new bug worth
reporting, not a known quirk.

### 8. Prefix-cache tracing (`src/patch_hit_debug.py`, `HIT_DEBUG=1`)

Set `HIT_DEBUG=1` (→ `VLLM_HIT_DEBUG=1` in the container) to log, per request:
the per-KV-group hit reconciliation (which group truncated the hit), mamba
boundary-state publication (which slots were real/null/hashed), cached-block
evictions, and prefill chunk-stop decisions. This is what found the bug in
patch 5; costs nothing when off.

### 9. Deterministic QSA top-k (`src/patch_qsa_exact_topk.py` + a kernel built in the `Dockerfile`)

Taken from upstream [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(`8347e7c`, `4b723de`, `0022e36`). The sparse attention (QSA) picks its top-k key
blocks with vLLM's `persistent_topk` kernel, which is non-deterministic on GB10 and
can drop real candidates ([vllm#51782](https://github.com/vllm-project/vllm/issues/51782),
diagnosed by [@k3dani](https://github.com/k3dani) in
[blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3)). Two fixes, both opt-in
env vars, `DET_TOPK=1` the default:

- **`DET_TOPK=1` — deterministic kernel** by [@jschmied](https://github.com/jschmied)
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)): index-ordered output
  slots and bit-exact tie resolution (signed zero canonicalised), plus a deterministic
  low-shared-memory fallback. The sources are fetched at a pinned commit (sha256-checked)
  and compiled with the image's nvcc at build time as a standalone `_C_det.so` — no vLLM
  rebuild; `qsadet_patch.py` (also @jschmied's) wires the QSA block selection to it.
  Upstream measured the whole prefill penalty of the exact path gone (GX10, 32k:
  1,794 → 2,996 tok/s) with decode unchanged.
- **`EXACT_TOPK=1` — exact `torch.topk`** over the visible columns (blazux's first fix):
  also deterministic, −20–40% on long prefill; kept as the fallback and wins over
  `DET_TOPK` when set. Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" -w /t
  --entrypoint python3 qwen38-flash-dgx test_qsa_exact_topk_cpu.py`.

### 10. MTP drafter options: int4 draft experts + reduced draft vocabulary

Both on by default (the Quickstart checkpoint + `DRAFT_VOCAB=1`); the plain
`-hybrid` checkpoint and `DRAFT_VOCAB=0` remain options. Both leave outputs
unchanged — the target verifies every drafted token, only the draft's cost and
acceptance move. Measured on a DGX Spark (`bench_qwen35.sh` with
`enable_thinking: false`, T=0, second run):

| | bf16 draft (`-hybrid` repo) | int4 draft experts | int4 + `DRAFT_VOCAB=1` (default) |
|---|---|---|---|
| Code / JSON / LongCode tok/s | 57.3 / 61.3 / 57.9 | 59.1 / 62.6 / 60.3 | **62.0 / 66.0 / 62.4** |
| draft acceptance (of 3) | 88.0% (2.64) | 88.5% (2.65) | 87.3% (2.62) |
| weights resident | ~71.4 GiB | 67.9 GiB | 67.9 GiB |

With thinking **on** (albond's original script) draft acceptance is ~70% in all three
configs — only 1.0% of reasoning tokens fall outside the 65k set — and the default
config is still the fastest (Code 56.5, JSON 62.1, LongCode 48.5 tok/s, see the
Throughput table).

- **int4 draft experts** — Intel's checkpoint leaves the MTP layer's 512 routed
  experts in bf16 (~4.7 GiB on the unquantized MoE path). `tools/quantize_mtp_experts_int4.py`
  writes a variant checkpoint (hardlinks for the untouched shards) with those
  experts as int4 g128 RTN in Intel's exact GPTQ layout and drops the `layers.48`
  exclusion, so the draft runs on GPTQ-Marlin like the main layers. Prebuilt:
  [Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN](https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN)
  — point `MODEL_DIR` at it. Idea from upstream's `hybrid-mtp` mode
  ([blazux#11](https://github.com/blazux/qwen3.8-Flash-DGX/pull/11) by @pfy, after
  thavoc's graft write-up), done here as RTN int4 instead of an NVFP4 graft so no
  new kernel path is needed.
- **`DRAFT_VOCAB=1`** (`src/patch_mtp_draft_vocab.py`, from upstream blazux `0c6df7e`,
  idea from MiaAI-Lab's recipe reimplemented there) — vLLM shares the target's
  lm_head with the drafter, so every draft pass scores all 248,320 rows. With the
  flag the draft scores a private 65,536-row slice (`src/draft_vocab_65536.npy`:
  corpus frequency + BPE order + all special tokens; `tools/build_draft_vocab.py`
  rebuilds it) and every other id is −inf. This fork's head is int8 GPTQ-Marlin,
  so the slice takes those rows from the checkpoint's GPTQ tensors at first use
  and keeps them int8 on the same Marlin kernel (616 → 160 MiB read per draft
  step; the draft's logits on the slice are the target head's). Until
  2026-11-03 the slice was dequantized to bf16 (320 MiB): a decode profile
  (`bench/profile_decode.py`, 2026-11-03) showed that bf16 GEMV at ~1.2 ms per
  draft pass, three passes a step, on a ~45 ms step. `DRAFT_HEAD=bf16` brings
  it back for comparison; `src/test_draft_head_cpu.py` checks the slice's
  layout (no GPU). Measured 2026-11-03 (GX10, `bench/perf.py`, code prompt,
  thinking off, `DET_MOE_ALIGN=1`), against the bf16 slice on 2026-11-01:

  | | single-stream decode (median of 5) | aggregate at 1 / 2 / 4 / 8 streams |
  |---|---|---|
  | bf16 slice | 66.4 tok/s (MTP 3.50 tok/step, 84%) | 60.5 / 93.8 / 177.4 / 241.9 |
  | int8 slice | **69.7 tok/s** (3.50, 84%) | 65.6 / 102.5 / 171.7 / 252.5 |

  +5% decode, ~2.5 ms off a ~52 ms step (three draft passes of ~0.6 ms
  saved each, as the profile predicted); acceptance unchanged, as it should
  be with the same logits. The 4-stream cell is within the sweep's spread.
- **`DRAFT_HEAD=int4`** (same patch) — the other way to halve the head read
  without touching the vocabulary: a private int4 g128 RTN copy of the full head,
  quantized with vLLM's own `marlin_quantize` at first use (~23 s) from the
  checkpoint's int8 GPTQ tensors, run through the same GPTQ-Marlin kernel
  (616 → 313 MiB per draft step; +313 MB resident). Outputs cannot change (the
  target verifies every token), only acceptance can, and it does: int4 RTN puts
  ~11% relative error on the head (int8: ~0.5%). Measured 2026-09-10, thinking
  off, 600-token answers, second run of each prompt:

  | | int8 shared head (default) | `DRAFT_HEAD=int4` |
  |---|---|---|
  | Traditional Chinese prose | 34.6 tok/s, 31% accept | 35.5 tok/s, 30% |
  | Cantonese prose | 32.8 tok/s, 28% | 32.3 tok/s, 25% |
  | English prose | 40.4 tok/s, 44% | 39.6 tok/s, 39% |
  | Python + tests | 62.5 tok/s, 86% | 60.7 tok/s, 78% |

  The bandwidth saved is handed back in rejected drafts, so it stays off. A
  calibrated int4 (AutoRound/GPTQ on real MTP hidden states) might keep more
  acceptance; not tried. Context: krisbailey.com's "shortlist MTP" write-up found
  the same ~4% ceiling for head-shrinking tricks once the head kernel is
  bandwidth-bound, which Marlin already is here.

### 11. On-demand step profiling (`src/patch_step_profile.py`, `STEP_PROFILE=1`)

This vLLM build predates `VLLM_TORCH_PROFILER_DIR` and `/start_profile`. With
`STEP_PROFILE=1`, `docker exec qwen38-flash touch /tmp/profile_trigger` makes
the next 24 engine steps run under torch.profiler (CPU+CUDA) and writes a
chrome trace to `/tmp/step_profile_<n>.json` — `docker cp` it out and open at
ui.perfetto.dev. Old traces are pruned to the 3 newest; when idle the patch
costs one `os.path.exists` per step.

### 12. Per-step prefill metrics (`src/patch_prefill_metrics.py`, `ITER_DETAILS=1`)

vLLM credits `vllm:prompt_tokens_total` only when a request's prefill
*finishes*, so a 100k-token prompt shows 0 tok/s for a minute and then a spike.
With `ITER_DETAILS=1` (→ `--enable-logging-iteration-details`) this patch feeds
`vllm:scheduled_ctx_tokens_total` and `vllm:scheduled_iterations_total` on
vLLM's own `/metrics` every engine step and mutes the stock one-line-per-step
log. `bench/ppwatch.sh` prints live prefill tok/s from them. From
Saren-Arterius/qwen3.8-Flash-DGX-AutoRound. Self-check: `docker run --rm -v
"$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_prefill_metrics_cpu.py`.

### 13. Engine-side metrics sidecar (`src/vllm_custom_metrics.py`, `METRICS_PORT`)

vLLM's own `/metrics` is served by the API-server process, which cannot see the
EngineCore-side counters this fork cares about; plumbing them through the
serialized `SchedulerStats` IPC would mean patching vLLM internals blindly. So
the engine process serves them itself with `prometheus_client` (the library
vLLM already uses) on its own port: `vllm:ple_mmap_*` (ops, op/gather ms, rows,
bytes, prefetch hit/miss — `rate(op_ms)/rate(ops)` is ms/op), the
`vllm:mamba_state_copy_guard_total` tripwire, the `vllm:never_evict_*` pin
gauges, and torch's caching allocator in the engine
(`vllm:torch_cuda_allocated_bytes`, `..._reserved_bytes`,
`..._max_reserved_bytes`: on GB10 that is host memory, and what the allocator
keeps after a large request is not given back; `../scripts/mem-watch.sh` puts
these three in its line). Its refresher thread also does the idle release
(`IDLE_RELEASE_SEC`, `vllm:torch_idle_released_bytes_total`): after that many
seconds with the scheduler empty and at least 512 MiB of freed blocks held,
`torch.cuda.empty_cache()`, again only once later work has raised the hold by
512 MiB (so CUDA graph pools, which it cannot free, do not make it repeat). It
starts lazily from the first data push (i.e. in the process that
actually has data) and every failure is swallowed — telemetry never takes the
server down. With it on, the `PLE mmap stats` log lines default off
(`PLE_STATS_SEC=0`); on an image without the module the launcher keeps them at
`30` instead of going silent.

### 14. Tool-call parser (`src/patches/qwen-tool-preamble.patch`, `qwen-tool-marker-guard.patch`)

From [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(their patches 12 and 13, Apache-2.0), at `2ba2d77`, their last commit on
this same base image (`fc120ece...`), where both applied with `--fuzz=0`; here
they apply the same way to the parser sources of vLLM PR #53896 the image comes
from. Both are on for vLLM's `qwen3` parser configuration, which
`--tool-call-parser qwen3_xml` and `qwen3_coder` run.

- **12** (Jordi Posthumus): a tool preamble is confirmed before the output
  switches channels. Before it, quoting `<tool_call>` in reasoning or in the
  answer could switch the parser into a tool call and drop what followed,
  the final answer after `</think>` included.
- **13** (@techfury90, blazux#29): when the model shows its tool-call syntax as
  an example — documenting the format inside a ```` ```xml ```` block — the
  parser confirmed a call and consumed the rest: with tools in the request,
  a call the model never meant; without, `content: null`. Now a marker
  inside a fenced code block (CommonMark fences: a run of three or more
  backticks or tildes at a line start, closed only by a run of the same
  character at least as long) stays text, and a wrapper-less `<function=`
  header opens a call only at the start of a line. Not covered, by design:
  example XML outside any fence still opens a call, and a fence the model
  never closes keeps the guard on for the rest of the turn.

blazux's tests for both are in `src/patches/*-tests.patch` (vLLM's test tree
is not in the image): on their side 30 cases, 21 failing on 12 alone and all
passing with 13. `python3 ../bench/consistency.py --only fence` checks the
served model end to end. This project's SGLang profiles carry the same guard
for SGLang's own parser (`../patches/gb10_tool_fences.py`).

### 15. Deterministic MoE token alignment (`src/patch_det_moe_align.py`, `DET_MOE_ALIGN=1`)

`moe_align_block_size` sorts a batch's (token, expert) routes into per-expert
blocks for the grouped MoE GEMM; vLLM's CUDA kernel places an expert's routes
with `atomicAdd`, in thread-arrival order. When an expert gets more routes than
one block (prefill) that order decides which m-block a route lands in, and
Marlin's split-K stripes depend on the m-block, so a prompt's sums run in
another order from one run to the next. Measured here with
`../bench/consistency.py` (2026-10-31, 4K and 32K prompts, greedy): three cold
runs of one prompt, first-token top-5 logprobs 0.16-0.45 nats apart, replies
forking after 8-117 characters, with `MARLIN_ATOMIC_ADD` 1 and 0 alike (that
switch never reaches the MoE: its Marlin call reduces in order, fp32, no
atomics, so given one block layout its sums are fixed); the QSA top-k is
deterministic already (patch 9). The first token comes from the prefill
alone, so MTP is out of it too. blazux's NVFP4 path, whose MoE
does not run on Marlin, measured deterministic first tokens.

**Measured with the patch on (2026-10-31, GX10, `DET_MOE_ALIGN=1`):**
`consistency.py --only repeat` at ~4K and ~32K tokens, cold, two cache hits
(1600 / 30400 tokens from the cache) and cold again with its own
`cache_salt`: the same text every time and a first-token top-5 gap of 0.0000
for cold vs cold, hit vs hit and cold vs hit. So the alignment was the whole
run-to-run difference on this stack, and a prefix-cache hit (Mamba state at a
1600-token boundary plus the KV after it) reproduces a cold prefill bit for
bit. The GPU check (`./build.sh`) found the kernel moving routes on every
one of 10 runs of a 100-8192-token batch; one call costs ~0.2 ms against
12-43 us for the kernel (mostly launches of small torch ops; 48 MoE layers
plus the MTP layer per step).

**Serving cost (2026-11-01, GX10, one boot each, `DET_MOE_ALIGN_MIN_TOKENS`
auto = 33):**

| `DET_MOE_ALIGN` | decode, 1 stream (median of 5) | aggregate tok/s at 1 / 2 / 4 / 8 streams |
|---|---|---|
| 1 | 66.4 tok/s (MTP 3.50 tok/step, 84%) | 60.5 / 93.8 / 177.4 / 241.9 |
| 0 | 65.4 tok/s (3.50-3.63, 83-88%) | 61.4 / 108.3 / 163.2 / 254.3 |

The streams columns change sign from one level to the next (2: -13%, 4:
+9%, 8: -5%), which is the spread of one sweep, not a cost: decode at 8
streams (32 tokens a step) runs the kernel either way, and the only steps the
torch path takes are the prompts' prefill. With the floor at 9 (every decode
step of 3+ streams on the torch path) the same sweep gave 231.5 at 8 streams
(2026-10-31). Prefill was not compared; the estimate is 49 calls x ~0.25 ms
per 8192-token chunk of ~3.8 s, ~0.3%. On by default since then.

With `VLLM_DET_MOE_ALIGN=1` steps of at least `VLLM_DET_MOE_ALIGN_MIN_TOKENS`
tokens are aligned by torch ops: the same blocks,
padding and expert ids as the kernel, each expert's routes in ascending route
order, no host sync. Routes to an id outside the experts (vLLM gives padding
tokens -1) are left out, as the kernel leaves them out: the first version
indexed with them and stopped the engine in its profile run with a
device-side assert (2026-10-31). Smaller steps keep the kernel. The patch's
own floor is 9 tokens: below it no expert can get more routes than a block of
8 holds, and the order inside a block is harmless (a block's rows are computed
alike wherever they sit). The launcher passes `DET_MOE_ALIGN_MIN_TOKENS`,
`auto` by default: `SEQS` × (1 + `MTP`) + 1, one more than the largest step
of decode alone (33 with 8 streams and MTP 3), so a step that carries a
prefill chunk takes the deterministic path and decode keeps the kernel's
~12 us. Decode gives up nothing by it: one stream's step (1 + `MTP` tokens)
is below the floor anyway, and a step of several streams depends on what
else is in the batch, whatever the alignment. With 9 every multi-stream
decode step paid ~0.2 ms a MoE layer (49 of them a step) instead. The idea and the implementation are
prestonlogan's (Flint-vLLM-Single-DGX-Spark, `fast/patch_align.py`,
Apache-2.0; ~3.7% prefill there); the patcher here locates the call by name
in this image's `moe_align_block_size.py`. `src/test_det_moe_align_cpu.py`
checks the layout against a reference of the kernel's on CPU (random batches,
512 experts top-10, padding tokens, ids past the last expert) and the patcher
on the PR #53896 file. `src/test_det_moe_align_gpu.py` checks it in the image
against the CUDA kernel itself, counts how often the kernel's own order moves
between runs of one batch, and times both; `./build.sh` runs it after a build
(a few seconds on the idle GPU; `DET_ALIGN_CHECK=0` skips it):

```bash
docker run --rm --gpus all --entrypoint python3 -v "$PWD/src:/w:ro" qwen38-flash-dgx /w/test_det_moe_align_gpu.py
```

### MoE kernels: Marlin or Humming

Tried on 2026-11-01 and dropped: `--moe-backend humming` (`EXTRA` passes it;
the recipe had a `MOE_BACKEND` switch with a host cache for the JIT kernels for
the test) runs the int4 experts (and the MTP draft's) on Humming,
vLLM's JIT quantized-GEMM library (`humming-kernels` 0.1.12 in this image):
the log says `Using 'HUMMING' WNA16 MoE backend.` and `Using HummingIndexedExperts`,
the kernels built during the normal boot time (NVRTC), and no shape fell back
to a default tile. Measured on the GX10 (2026-11-01, `DET_MOE_ALIGN=1`, one boot
each; Marlin's runs are the two of "Serving cost" above):

| | decode, 1 stream (median of 5) | aggregate at 1 / 2 / 4 / 8 streams | prefill 2K / 8K / 32K / 101K |
|---|---|---|---|
| Humming | 64.2 tok/s (3.53 tok/step) | 60.5 / 99.8 / 167.5 / 258.0 | 1,969 / 1,973 / 2,111 / 2,025 |
| Marlin | 66.4 / 65.4 tok/s | 60.5-61.4 / 93.8-108.3 / 163.2-177.4 / 241.9-254.3 | 2,000 / 2,153 / 2,134 / 1,981 (2026-10-31) |

Within one sweep's spread everywhere, decode a little under: on this GPU at these
batch sizes (4-32 tokens a decode step) Marlin already runs the experts at the
memory bandwidth, and vLLM's published gains (~2x Marlin for MoE W4A16) are at
medium and large batches on H200 and B300.

And Humming is not deterministic: `consistency.py --only repeat` with
`DET_MOE_ALIGN=1` gave cold vs cold and hit vs hit apart from the first
token (top-5 gaps 0.20-0.33 nats at 4K and 32K), where Marlin gives 0.0000.
The alignment is not the cause this time (the same patched
`moe_align_block_size` feeds both); the kernel is: for K > 1024 Humming
splits the K loop over blocks (stream-K), and a tile split more than three
ways is summed into the output with `atomicAdd` on half2, in arrival order and
in BF16 (`include/humming/epilogue/gmem_writer.cuh` in humming-kernels
0.1.12). Marlin's MoE call reduces in order in FP32. Humming's
`use_batch_invariant` kernels drop stream-K, but vLLM turns them on only with
`VLLM_BATCH_INVARIANT=1`, its engine-wide batch-invariant mode. Marlin it is.

### 16. FP8 hyper-connection projections (`src/gb10_fp8_rowwise.py`, `HC_FP8=1`)

**Where a decode step goes** (2026-11-03, GX10, `PROFILE=1 PROFILE_SHAPES=1`,
`bench/profile_decode.py`, one 160-token code answer, 45 MTP steps of 4 rows,
the bf16 draft slice of the time; per step, kernel time, eager mode; GB/s =
the weight a GEMM reads over its time, ~240 is what a GB10 delivers):

| Layer | ms / step | GB/s | |
|---|---:|---:|---|
| MoE experts, gate_up + down (int4 Marlin) | 14.3 | — | reads only the routed experts |
| GDN `in_proj` [2560 → 16384] (FP8 CUTLASS blockwise) | 6.4 | 236 | at the bandwidth |
| draft head slice [2560 → 65536], 3 a step (bf16 then) | 5.6 | 185 | int8 since (patch 10): −2.5 ms measured |
| GDN / QSA `out_proj` [6144 → 2560] (FP8) | 4.5 | 169 | |
| hyper-connection down + block_inject [10240 → 336] (BF16) | 3.9 | 172 | 98 a step |
| hyper-connection up [320 → 10240] (BF16) | 3.1 | 209 | 100 a step |
| lm_head [2560 → 248320] (int8 Marlin) | 2.6 | 242 | at the bandwidth |
| QSA qkv [2560 → 13312] (FP8) | 1.8 | 233 | |
| shared expert gate_up / down (FP8) | 1.3 / 0.6 | 118 / 137 | |
| FP8 activation quantization | ~0.8 | — | 8 us a call |
| GDN decode kernel | 1.3 | — | |
| MTP layer's dense projections at 1 row (BF16) | ~1 | 186 | qkv [2560 → 13312] alone 0.75 |
| router [2560 → 512], GDN `in_proj_ba` [2560 → 96] (BF16) | 0.8 / 0.6 | 169 / 28 | |

The MoE, the large FP8 projections and the lm_head read at the memory's
bandwidth: only fewer bytes make them faster. The largest BF16 item after the
draft head is the hyper-connection mix: two GEMMs per mix, 7 ms a step together,
on weights the checkpoint keeps in BF16.

**The patch.** With `HC_FP8=1` those two projections of every mix (target and
MTP layer) keep their weights in FP8 E4M3 with one float32 scale per row (amax /
448) after loading: ~1.2 GiB → ~0.6 GiB. The 16 rows the down projection carries
for block_inject (and padding) stay BF16. Inputs of up to 64 rows (decode and
MTP verify, 4 a sequence) run a Triton GEMM that converts the FP8 rows to BF16
in registers (exact), accumulates in FP32 and applies the row scale to the sum;
the down projection's long K is split over programs and the partial sums added
by a second kernel in a fixed order, so the result is the same bits every run
and does not depend on the other rows in the batch. Longer inputs (prefill)
take cuBLAS on a BF16 copy of the weight made for that call by one Triton pass
(FP8 rows times their scales, then the tail rows; ~10 MB of traffic against a
GEMM that reads up to an [8192 x 10240] input). The GEMM is a torch custom op
(`gb10::fp8_row_linear`), opaque to torch.compile and captured by CUDA graphs.
The recipe is that of `FP8_HC` on the SGLang profile of this model
(`../patches/gb10_fp8_hc.py`): +3-3.5% decode there, HumanEval 97.6% / 99.4%
(not worse than BF16). It is lossy; measured below and on by default since 2026-11-03 (`HC_FP8=0`
turns it off). The boot log says `FP8 HC on: N hyper-connection classes
wrapped` and, at the first forward, `FP8 HC: N hyper-connection projections in
FP8 ...`.

**First measurement** (2026-11-03, GX10, `perf.py`, against the int8 draft
slice the same day; one boot each):

| | decode, 1 stream (median of 5) | aggregate at 1 / 2 / 4 / 8 streams | HumanEval (no thinking, greedy) |
|---|---|---|---|
| `HC_FP8=0` | 69.7 tok/s (MTP 3.50 tok/step, 84%) | 65.6 / 102.5 / 171.7 / 252.5 | 96.3% (158/164) |
| `HC_FP8=1` | **74.5 tok/s** (3.56, 85%) | 68.5 / 109.9 / 170.3 / 259.6 | 96.3% (158/164) |

+7% decode, ~2.4 ms off a ~50 ms step; HumanEval the same score (one problem
traded for another: 113 failed without, 127 with). Boot log: 200 projections,
1.25 → 0.64 GiB. Prefill in that run was 1727 / 1757 / 1495 / 1854 tok/s at
2K / 8K / 32K / 101K, against 2000 / 2153 / 2134 / 1981 on 2026-10-31 (not the
same boot; the 32K prompt ran with the host swapping). The kernel
bench's first version timed each weight from the GPU's caches (BF16 "at"
570-900 GB/s, above what the memory delivers); it now cycles through ~300 MB
of copies, as the model reads ~98 different mixes a step, and agrees with the
server: a mix at 4 rows 62.0 us in BF16, 35.7 us in FP8 (down 182 GB/s, up
209 GB/s), 2.6 ms a step.

**Prefill A/B** (same image, one boot each, `perf.py --only prefill`; at that
time the prefill path scaled the up projection's weight copy in three torch
passes and ran the block_inject rows as a second GEMM):

| | 2K | 8K | 32K | 101K |
|---|---:|---:|---:|---:|
| `HC_FP8=1` | 1826 | 1748 | 1503 (host swapping: 338K pages out) | 1923 |
| `HC_FP8=0` | 1906 | 1689 (51K pages out, 1203 reclaim stalls) | 2086 | 1742 |

The spread is the host's memory: each run's slow cell is the one where the
host reclaimed or swapped (~5 GB free while serving), whichever the setting.
The kernel bench put that prefill path's own cost at +0.45 ms a mix at 2048
rows and +1.2 ms at 8192 (the block_inject GEMM read the [8192 x 10240] input
a second time), ~3-4% of a chunk; the one-pass dequantize that replaced it
reads ~3.4 MB and writes ~6.9 MB a call.

**A slow boot** (2026-11-03, evening, `HC_FP8=1` by default, image with the
one-pass dequantize): decode 64.2 tok/s, below BF16's 69.7 (3.54 tokens a step,
the same acceptance), every concurrency level ~10% down, prefill 1798 / 1892 /
1834 / 1915 tok/s. The decode code had not changed since the 74.5 run; what
the numbers fit is decode on the torch fallback (dequantize + cuBLAS) instead
of the Triton kernel, and the log said so: `the Triton kernel for 16-row tiles
failed (KeyError: 'mm')`, then 64 and 32. A bug of the one-pass dequantize: its
kernel went into the dict that cached the GEMM's, which was looked up by "the
dict is not empty"; prefill runs first at boot, so the dict held only the
dequantize and every decode tile failed over to torch, for good, with one
warning line each. Fixed (looked up by name; `src/test_hc_fp8_cpu.py` builds
the dequantize first and fails on the old code with that KeyError). Also since:
every kernel variant (16-, 32- and 64-row tiles, the dequantize) is built when
the weights are converted, before any CUDA graph capture, and says so loudly if
one does not build; the GEMM no longer specializes on the row count; the
first-forward log line names the path each tile takes (`decode kernel for
16/32/64-row tiles; prefill dequantize on Triton`, or TORCH PATH in capitals).
The bench's section 5 checks the prewarm and probes a kernel's first launch
inside a capture (it loads fine on the GX10). After the fix: decode 73.4 and
73.3 tok/s (two runs, 3.53 tokens a step, 84%), the log naming the kernel for
all three tiles. The same bench showed 16-64 rows 10-30% slower than before,
because that commit had also stopped specializing on the strides, which keeps
the loads of x from being vectorized; only the row count is unspecialized now.

Checks: `src/test_hc_fp8_cpu.py` (no GPU; with `pip install triton` it also
runs the kernel in Triton's interpreter) and, in the image, the kernel against
cuBLAS on the real shapes, with correctness, run-to-run bits, prefill sizes and
a sweep of its tile settings:

```bash
docker run --rm --gpus all --entrypoint python3 -v "$PWD/src:/w:ro" qwen38-flash-dgx /w/bench_fp8_rowwise_gpu.py
```

## Speculative decoding and TTFT

MTP's draft layer is a stateful autoregressive transformer: vLLM's v1 engine
only emits the first token after the drafter has run, and on every prefill
chunk the drafter runs a full-chunk-width forward (always eager: above the
cudagraph capture sizes) to sync its own KV/GDN state, plus k−1 sequential
single-token passes. Earlier versions of this README called that a ~0.8 s TTFT
floor. On this stack, warmed up (`../bench/perf.py`, MTP=3), it is not a floor
but part of the prefill rate:

| prompt | 1 line | 2,194 | 8,481 | 33,612 | 105,290 tokens |
|---|---|---|---|---|---|
| TTFT | 0.13 s | 1.04 s | 3.68 s | 14.98 s | 49.77 s |

— i.e. TTFT ≈ 0.1 s + prompt / ~2,200–2,400 tok/s. A cold server does show
second-long stalls, from first-use kernel compiles (next section), not from
MTP. Cross-attention drafters like DFlash skip the per-chunk drafter forward,
but Flash-Next has no such drafter; `MTP=0` removes that cost from prefill at
the price of decode speed.

### Warmup

After every restart, Triton compiles several kernels on first use *during
inference* — vLLM's `jit_monitor` logs each one as `Triton kernel JIT
compilation during inference`: the spec-decode sampling kernels
(`_rejection_kernel`, `_resample_kernel`, ...) on the first request, the QSA
split-k/merge kernels at the first few concurrent streams, and the QSA indexer
(`_qsa_pre_indexer_kernel`, `_expand_qsa_indices_kernel`) on the first ~2k-token
prompt. Each is a stall of about a second for whoever hits it first — on a
fresh container the first 4-stream wave saw TTFT 1.94 s instead of 0.39 s.
`python3 ../bench/perf.py --only warmup` touches all of those shapes in ~20 s;
`../bench/perf.py` runs it before measuring (`--no-warmup` to measure a cold
server).

## What's in here

```
Dockerfile                    official vLLM Flash-Next image + the patches above
config.sh                     every setting, with what it does (config.local.sh: this machine's)
serve.sh / stop.sh / status.sh  start (checks, memory, wait) / stop / state, as every server here
build.sh / download.sh        the image / the two weight folders at pinned revisions
prepare.sh                    build the checkpoint + table from Intel's release
src/vllm_ple_mmap.py          mmap PLE table (any dtype, relocatable dir, fast gather)
src/vllm_custom_metrics.py    engine-side Prometheus sidecar: PLE/guard/pin (METRICS_PORT)
src/vllm_fp8_hybrid.py        int4+fp8 hybrid dispatch on the GPTQ config
src/patch_never_evict.py      never-evict system-prompt KV pinning
src/patch_mamba_align_split.py  prefix-cache chunk-alignment fix
src/patch_hit_debug.py        prefix-cache tracing (VLLM_HIT_DEBUG)
src/patch_qsa_exact_topk.py   exact, deterministic QSA top-k (VLLM_QSA_EXACT_TOPK=1; from blazux)
(Dockerfile, _C_det.so)       @jschmied's deterministic persistent_topk kernel, built at docker build
src/test_qsa_exact_topk_cpu.py  CPU unit test for the exact top-k (no GPU needed)
src/patch_mtp_draft_vocab.py  private MTP draft head: reduced vocabulary (from blazux), int8 slice / int4 (VLLM_MTP_DRAFT_VOCAB, VLLM_MTP_DRAFT_HEAD)
src/test_draft_head_cpu.py    CPU test of the draft head's int8 slice (no GPU, no vLLM)
src/gb10_fp8_rowwise.py       FP8 per-row weights + Triton GEMM for the hyper-connection projections (HC_FP8=1)
src/patch_hc_fp8.py           its hook in hyperconnection.py
src/test_hc_fp8_cpu.py        CPU test of both (kernel in Triton's interpreter when triton is installed)
src/bench_fp8_rowwise_gpu.py  GPU check and timing of the kernel against cuBLAS, in the image
src/draft_vocab_65536.npy     the default 65,536-token draft id set (from blazux)
tools/build_draft_vocab.py    rebuild the draft id set (from blazux)
tools/quantize_mtp_experts_int4.py  int4 RTN the MTP draft experts -> -MTP_int4RTN checkpoint
src/mamba_utils_guarded.py    hardened align-mode state copy (vllm#50729 + guard)
src/test_ple_mmap_cpu.py      CPU unit test for the gather (no GPU needed)
src/test_never_evict_pin.py   CPU unit test for the pin (no GPU needed)
src/test_custom_metrics_cpu.py  CPU unit test for the metrics sidecar (no GPU, no vLLM)
src/test_prefill_metrics_cpu.py  check for the prefill-metrics patch (inside the image)
src/patch_step_profile.py     on-demand torch.profiler around engine steps (VLLM_STEP_PROFILE)
scripts/serve-intel-ar.sh     the recipe's launcher: image probes, the docker run behind serve.sh
src/patches/                  tool-call parser fixes 12-13 (from blazux) and their tests
src/patch_det_moe_align.py    deterministic MoE token alignment (DET_MOE_ALIGN; after prestonlogan)
src/test_det_moe_align_cpu.py its CPU check (torch, no GPU)
src/test_det_moe_align_gpu.py its GPU check against the image's kernel (./build.sh runs it)
scripts/smoke-test.sh         health + coherence + prefill/decode numbers
watchdog.sh                   restart before earlyoom: low memory + idle (WATCHDOG; scripts/test-watchdog.sh, stubs)
src/patch_prefill_metrics.py  per-step prefill counters for Prometheus (ITER_DETAILS=1)
bench/ppwatch.sh              live prefill tok/s from those counters
bench/decode_bench.py         batch-1 decode / TTFT / spec-acceptance bench
bench/concurrency_bench.py    N-stream throughput / TTFT / queue-time bench (completions API)
../bench/perf.py, longctx.py, consistency.py   this project's chat-API benchmarks (were copied here)
tools/eval_quality.py         ppl + greedy-facts quality check against the API
tools/test_tools_cpu.py       CPU test for the preparation tools (correctness, safe re-runs)
tools/                        CPU-only checkpoint preparation
docs/HOW-IT-WORKS.md          upstream's mmap-PLE story (NVFP4-era numbers)
ruff.toml                     lint config (ruff check .: syntax + pyflakes); shellcheck *.sh scripts/*.sh
                              CPU tests (torch, numpy, safetensors, prometheus_client): src/test_ple_mmap_cpu.py,
                              src/test_custom_metrics_cpu.py, tools/test_tools_cpu.py, src/test_qsa_exact_topk_cpu.py,
                              src/test_det_moe_align_cpu.py
```

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- This repository is based on
  **[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)**,
  the int4/int8/fp8 AutoRound fork this README describes.
- **This is a fork of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** —
  the original mmap-PLE idea, the GB10 serving recipe, the NVFP4 path, and
  docs/HOW-IT-WORKS.md are theirs.
- int4 checkpoint this fork builds on: **[Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)**
  (AutoRound); fp8 PLE table from **[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)**.
- Several pieces originate in the Qwen3.5-122B-A10B Spark recipes:
  the int4 AutoRound + int8 lm_head serving approach from
  **[albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)**;
  the FLA shared-memory gate fix and the int4+fp8 hybrid idea from
  **[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)**.
  The never-evict pin was built for that 122B stack on top of the ARC
  GPU-eviction work in [vllm#40270](https://github.com/vllm-project/vllm/pull/40270)
  and re-ported here.
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896).
- MTP drafter options: the reduced draft vocabulary (patch, id set, build tool) from
  upstream **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**
  (`0c6df7e`; idea from MiaAI-Lab's recipe, reimplemented there); the quantized draft
  experts follow upstream's `hybrid-mtp` mode
  ([blazux#11](https://github.com/blazux/qwen3.8-Flash-DGX/pull/11) by **@pfy**).
- Deterministic QSA top-k: the exact `torch.topk` fix and the kernel wiring/pins from
  upstream **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**;
  the deterministic `persistent_topk` kernel itself by **[@jschmied](https://github.com/jschmied)**
  ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122)); the bug diagnosed by
  **[@k3dani](https://github.com/k3dani)** ([blazux#3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3)).
- Tool-call parser fixes (patches 12-13 of upstream, at `2ba2d77`): **Jordi Posthumus** (12) and
  **[@techfury90](https://github.com/techfury90)** (13, [blazux#29](https://github.com/blazux/qwen3.8-Flash-DGX/pull/29)),
  from **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**.
- Independent reproduction of the upstream recipe, the native-offload fixes and
  the concurrency measurements: **[@jschmied](https://github.com/jschmied)**
  ([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- License: [Apache-2.0](LICENSE).

## Appendix: where to put the PLE table (and why not to bother with `PLE_PREFETCH`)

Batch-1 decode bench (`bench/decode_bench.py`, medians of 3; W1 = fresh-prompt
1000-token decode, W2 = pinned ~8k prefix hit + 256 tokens; DGX Spark GB10,
MTP=3). Only the table location and the prefetch flag change between rows (all local NVMe is Gen 4):

| table source              | prefetch | W1 tok/s | W2 tok/s | gather      |
|---------------------------|----------|----------|----------|-------------|
| RDMA row daemon           | on       | 42.4     | 47.8     | ~1.3 ms/op  |
| RDMA row daemon           | off      | 41.6     | 46.7     | ~1.3 ms/op  |
| local NVMe                | off      | 36.5     | 45.7     | 5–9 ms/op   |
| local NVMe                | on       | 34.5     | 42.6     | 9–17 ms/op  |
| local NVMe, warm cache    | on       | 33.4     | 41.5     | 9–16 ms/op  |
| NFS (RDMA mount, btrfs)   | on       | 26.2     | 36.5     | 26–56 ms/op |
| NFS (RDMA mount, btrfs)   | off      | 24.2     | 32.9     | 30–42 ms/op |

Takeaways:

- **Put the table on local NVMe.** It costs ~15% decode vs an exotic fast row
  source, and it is the simple recipe. Serving straight off a NAS works but
  costs ~40% (and that was NFS-over-RDMA to a btrfs box — a plain GbE NAS
  will be worse).
- **Don't bother with `PLE_PREFETCH=1`** (`VLLM_PLE_MMAP_PREFETCH`). The
  batch-assembly prefetch only wins where the row source is very slow (NFS,
  +2–3 tok/s, within run-to-run variance), is a wash on fast sources, and is
  **net-negative on local NVMe**: the handoff/wait in `consume()` costs more
  than the inline gather it replaces (confirmed on a warm page cache). It
  stays experimental and default-off.
- TTFT is unaffected by any of this (prefill compute dominates), and
  between-restart variance on this bench is >10% — treat small deltas above
  accordingly.

About those "RDMA row daemon" rows: that is a custom one-sided RDMA READ
server that pins the whole table in another box's RAM — the fastest row
source measured, at the price of a second machine and an ibverbs science
project. If you have a NAS with **≥64 GB of RAM** and a **≥100 Gbit RDMA
link** to your Spark — and no second DGX Spark to put to better use — the
[`magi` branch](../../tree/magi) ships the tool (`src/ple_rdma/`) and setup
notes ("PLE table over RDMA"). Everyone else: local NVMe is the recipe.
