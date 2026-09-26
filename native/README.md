# Qwen3.8-27B on DGX Spark (GB10)

A measured recipe for the fastest Qwen3.8-27B setup I could get on a DGX Spark:
**stock SGLang + DFlash2 speculative decoding**, installed natively with pip
(no Docker) and run as a systemd service.
Other models plug in as profiles: [Gemma 4 31B](#gemma-4-31b) and
[Qwen3.8-Flash-Next](#qwen38-flash-next) so far.

Throughput measured at the max-aggregate configuration (draft 10, 32
concurrent requests: the `qwen3.8-27b-throughput` profile); HumanEval does
not depend on it:

| Target | Single-stream | Peak aggregate | TTFT | HumanEval, thinking off |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` (default) | 49.5–51.3 tok/s | 494–514 tok/s @ 32 | 273–293 ms | **97.6%** |
| `RadixArk/Qwen3.8-27B-NVFP4` | **70.1 tok/s** | **572 tok/s** @ 32 | 197 ms | 93.9% |

Tuned for one stream (draft 16), NVFP4 reaches **78.6 tok/s**, and 97.0% on
HumanEval with thinking on. **Pick by what you need:** FP8 is Qwen's own
checkpoint and the more accurate one, NVFP4 is ~40% faster on single-stream
and 2.5× faster on prefill. On single-stream alone both are 1.5–2.5× the
widely-shared "FP8 on vLLM at ~32 tok/s" recipe.

**These numbers come from the earlier Docker build** (a patched SGLang
image), before SGLang shipped DFlash2 support upstream. The native install
below runs the same flags on stock SGLang 0.5.20 and has not been
benchmarked yet. Full data in [`results/RESULTS.md`](results/RESULTS.md);
the build behind it in [`results/BUILD-MANIFEST.md`](results/BUILD-MANIFEST.md).

Most of what this repo contributes is **which flags, at which values, and
why**: the GDN pool that actually bounds concurrency, the draft-token count
whose optimum depends on the workload, the memory fraction that doesn't
matter for speed but does for stability. Plus a benchmark suite that refuses
to report numbers it can't trust.

## Quick start

```bash
git clone https://github.com/reproart/gb10-repo-modified.git
cd gb10-repo-modified/native      # this directory is the whole project

# System packages: venv support, plus the headers and C compiler Triton needs at boot
sudo apt install python3.12-venv python3.12-dev build-essential
sudo mkdir -p /models && sudo chown $USER /models

./serve.sh install     # venv with stock SGLang 0.5.20 + the hf CLI, no Docker (~10 GB)

# The weights (one-time, ~35 GB), at the revisions the results were measured on:
HF=~/spark/venv/bin/hf
$HF download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /models/Qwen3.8-27B-FP8
$HF download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 \
  --local-dir /models/Qwen3.8-27B-DFlash2
# option: the NVFP4 target instead (~40% faster, less accurate; see the table above):
#   $HF download RadixArk/Qwen3.8-27B-NVFP4 --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
#     --local-dir /models/Qwen3.8-27B-NVFP4
# (already in an HF cache? skip this, see "Weights" below)

# Point MODEL_DIR / DRAFT_DIR in models/qwen3.8-27b.sh at those dirs, then:
./serve.sh                           # boots on :8888; ready when /v1/models answers
python3 bench/perf.py --only warmup  # optional: compile first-use kernels now
```

All model repos are public; no token needed.

Built for Ubuntu 24.04 (DGX OS), kernel 6.17-nvidia, driver 580.x / CUDA 13.0,
128 GB unified, Python 3.12. No Docker anywhere.

---

## 1. Check the host

```bash
./scripts/00-check-host.sh
```

Driver and CUDA version, Python 3.12 with `venv`, free memory, and whether
earlyoom is running. Run it again after `./serve.sh install`: it then also
checks that the venv's torch sees the GPU.

## 2. Install SGLang

```bash
./serve.sh install
```

Creates `~/spark/venv-sglang-0.5.20` (`GB10_WORKDIR` in `serve.sh`,
`SGLANG_VERSION` in the model profile), installs SGLang into it, and points `~/spark/venv` at it. The
venv carries the `hf` CLI too.

It installs the exact versions in
[`requirements/sglang-0.5.20-aarch64-py312.txt`](requirements/sglang-0.5.20-aarch64-py312.txt):
all 206 packages have prebuilt aarch64 wheels, so nothing compiles at install
time. (At first boot Triton compiles small C launchers, which is what
`python3.12-dev` and `build-essential` are for; install checks they are
there.) The lock
holds the pip-installed CUDA compiler at 13.0 to match the driver
([`constraints-cuda130.txt`](requirements/constraints-cuda130.txt)); left
alone, one dependency pulls nvcc 13.4.

### Trying a newer SGLang

```bash
# in models/qwen3.8-27b.sh: SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
./serve.sh install     # new venv next to the old one, plus a new lock
./serve.sh             # the startup lines name the version and venv in use
```

- **Each version has its own venv**, `~/spark/venv-sglang-<version>`, so
  the one that works stays untouched. Rolling back is setting
  `SGLANG_VERSION` back; its venv is still there. Delete the ones you're done
  with (~8 GB each).
- **Each version has its own lock.** With none in `requirements/` for this
  version, machine and Python, `install` resolves `sglang==<version>` with the
  same CUDA 13.0 hold and writes one, e.g.
  `requirements/sglang-0.5.21-aarch64-py312.txt`. Commit it once that version
  has served and benchmarked well; delete it to re-resolve. A version that
  doesn't resolve leaves nothing behind.
- **Nightly builds:** `SGLANG_INDEX=https://docs.sglang.ai/whl/cu130/` and
  the exact nightly version string.
- **What can break:** the server flags in
  [`scripts/serve-sglang.sh`](scripts/serve-sglang.sh) were checked against
  0.5.20. A newer version that renames one fails at boot with
  `unrecognized arguments`. One that needs a newer CUDA than the 13.0 hold
  fails to resolve: with a newer driver, empty `SGLANG_CONSTRAINTS` lifts it.
  And no throughput or quality number here has been measured on it: run
  `bench/perf.py` before trusting it.

It also tries to install `flashinfer-cubin` (precompiled kernels, as the
official image has) from FlashInfer's own index. If that fails, it says so,
and FlashInfer compiles those kernels on the first boot instead.

## 3. Weights

The server loads two local directories, `MODEL_DIR` and `DRAFT_DIR` in the
model profile (`models/qwen3.8-27b.sh`), and never touches the Hub (`HF_OFFLINE=1`). Two ways to fill them:

- **Download once** with `hf download --revision <sha> --local-dir <dir>`, as
  in the quick start. Always pass `--revision`: without it you get the repo's
  mutable default, which is not necessarily what was measured here.
- **Reuse an HF cache** you already have, without copying: point the variable
  at the snapshot directory, whose name is the revision.

  ```bash
  ls -d ~/.cache/huggingface/hub/models--*/snapshots/*
  # the old Docker toolkit kept its own cache:
  ls -d ~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface/hub/models--*/snapshots/*
  ```

| Checkpoint | Revision | Size |
|---|---|---:|
| `Qwen/Qwen3.8-27B-FP8` (default target) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | ~29 GB |
| `RadixArk/Qwen3.8-27B-NVFP4` | `554ebba9b5f1b79dc11246341960360e6ef05ef4` | ~22 GB |
| `z-lab/Qwen3.8-27B-DFlash2` (draft) | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | ~3 GB |

At startup the server prints each directory with the revision it holds (the
snapshot name, or the metadata `hf download --local-dir` leaves in
`<dir>/.cache/huggingface/`), because the boot log itself doesn't say.

## 4. Configure: `serve.sh` and a model profile

Two kinds of files, each knob with why it has the value it has. Keep your
settings as a local-only commit on top of them.

- [`serve.sh`](serve.sh) in the repo root: **the machine**, and which profile
  to serve by default.
- [`models/<profile>.sh`](models/): **one model**: its weights, the SGLang
  version it runs on, and its own flags (a `model_args` function). The one
  this README is about is [`models/qwen3.8-27b.sh`](models/qwen3.8-27b.sh);
  see [Other models](#other-models-profiles) for the rest.

| Knob | Where | Default | |
|---|---|---|---|
| `PROFILE` | `serve.sh` | `qwen3.8-27b` | served when `./serve.sh` names none |
| `JIT_JOBS` | `serve.sh` | `2` | parallel compiles for kernels built on first boot; see the traps |
| `API_KEY` | `serve.sh` | none | `"$(cat ~/.qwen-api-key)"` keeps the secret out of git |
| `HF_OFFLINE` | `serve.sh` | `1` | no Hub access while serving |
| `CPUSET` | `serve.sh` | `5-9,15-19` | the Cortex-X5 cores; empty disables pinning |
| `PORT`, `HOST` | `serve.sh` | `8888`, `0.0.0.0` | |
| `MODEL_DIR`, `DRAFT_DIR` | profile | `/models/Qwen3.8-27B-FP8`, `/models/Qwen3.8-27B-DFlash2` | step 3 |
| `SGLANG_VERSION` | profile | `0.5.20` | see "Trying a newer SGLang" |
| `DRAFT_TOKENS` | profile | `11` | 16 for single-stream, 10 for cap 32 (step 8) |
| `MAX_RUNNING` | profile | `12` | the most requests you run at once; sizes the GDN pool and CUDA graphs with it (step 7) |
| `MEM_FRACTION` | profile | `0.80` | see the earlyoom trap |
| `CHUNKED_PREFILL` | profile | `8192` | the cookbook uses 2048: smoother decode under mixed load |
| `PREFILL_CUDA_GRAPH` | profile | `0` | 1 turns prefill CUDA graphs on (untested here) |
| `CONTEXT_LENGTH` | profile | `262144` | |
| `EXTRA_ARGS` | profile | | any SGLang flags, appended last so they win |

Profile values are `${VAR:-default}`, so a one-off experiment needs no edit:
`DRAFT_TOKENS=16 MAX_RUNNING=16 ./serve.sh`. The machine settings in
`serve.sh` are exported as they stand, and win over the shell.

## 5. Serve

```bash
./serve.sh
```

Serves on **:8888**, OpenAI- and Anthropic-compatible, model name
`qwen3.8-27b-sglang`. Ready when `curl -s localhost:8888/v1/models` answers.
The first boot compiles kernels and can take much longer than later ones
(see Traps); they are cached under `~/.cache/flashinfer` and `~/.triton`.

It prints the profile, the checkpoint directories with their revisions and
the caps, then `exec`s `python -m sglang.launch_server`. The flags every model
gets are in [`scripts/serve-sglang.sh`](scripts/serve-sglang.sh); the model's
own in its profile's `model_args`.

Benchmark it:

```bash
python3 bench/perf.py      # defaults to http://127.0.0.1:8888/v1
```

## 6. Run it as a service

```bash
./scripts/install-service.sh               # the default profile; asks for sudo
./scripts/install-service.sh gemma4-31b    # or another one
journalctl -u gb10-sglang -f               # watch the boot
```

Installs `gb10-sglang.service`, running `./serve.sh <profile>` as you from
this checkout, enabled at boot. After editing `serve.sh` or the profile:
`sudo systemctl restart gb10-sglang`. It is one unit for one model: running
the installer with another profile switches it.

- Restarts on failure, but gives up after 3 failed starts in 20 minutes, so a
  config that can't boot doesn't loop.
- `MemoryMax=110G` (`SERVICE_MEMORY_MAX` to change) is the cgroup limit
  `docker --memory` used to provide. Whether GPU-side allocations on GB10's
  unified memory count toward it has not been tested here.
- Logs go to the journal, which rotates them. The Docker setup needed a
  daemon-wide log cap for that; this doesn't.

## 7. Concurrency: three flags, one knob

Concurrency on this hybrid (Gated DeltaNet) model is bought with **GDN state,
not KV cache**, and three flags co-limit it. Raising only the obvious one
measures nothing:

| Flag | Set to |
|---|---|
| `--max-running-requests` | `MAX_RUNNING` |
| `--max-mamba-cache-size` | `5 × MAX_RUNNING` (5 state slots per request: 4, plus 1 for the DFlash2 verify) |
| `--cuda-graph-max-bs-decode` | `MAX_RUNNING` (else larger batches fall back to eager) |

SGLang clamps `max_running_requests = max_mamba_cache_size / 5` and logs it.
Check the `max_running_requests` line after a change.

Measured on NVFP4, draft 8, short prompts:

| Streams | cap 4 | cap 16 |
|---:|---:|---:|
| 1 | 58.3 | 52.9 |
| 8 | 190.1 | 315.9 |
| **16** | — | **480.7** |
| 32 | — | 469.8 |

The peak tracks the cap, not the hardware: cap 32 (pool 160) moves NVFP4 to
**572 tok/s** at 32 streams with sub-second TTFT, i.e. nothing queued. Cap 48
(pool 240, ~47 GB of GDN state) does not fit alongside a usable KV pool on
128 GB and failed to boot here; treat 32 as the practical ceiling.

Aggregate is bought with memory, and these are short-prompt figures. Each
GDN slot costs 0.196 GB at `--mamba-ssm-dtype bfloat16` (twice that at the
float32 default), so pool 160 is ~31 GB taken from the KV pool.
[An independent run](results/REPRODUCTION.md) measured higher still at pool
160 (606.4) and 240 (640.9); we could not reproduce those magnitudes, see that
file for why.

For a single interactive user, `MAX_RUNNING=4` gives the memory back to KV at
~5% better single-stream.

### Which cap: set it to the load you actually have

The cap reserves GDN state up front, used or not, and does nothing for speed
below it. So the default profile uses **cap 12**: ~14 GiB of GDN state
instead of cap 32's ~35, with the difference (~520K tokens) in the KV pool.

| Profile | For | Cap / draft | What it gives |
|---|---|---|---|
| `qwen3.8-27b` (default) | up to ~12 concurrent requests | 12 / 11 | ~5 full-length sessions' worth of KV at 0.85; 73.6 tok/s single-stream, 401 tok/s at 12 (nvidia NVFP4) |
| `qwen3.8-27b-single` | 1–2 users | 16 / 16 | 70.9 tok/s single-stream (Uncensored finetune) |
| `qwen3.8-27b-longctx` | six 262K sessions | 6 / 10, 0.85 | see below |
| `qwen3.8-27b-throughput` | many short requests | 32 / 10 | 597.8 tok/s peak (RadixArk NVFP4) |
| `qwen3.8-27b-dspark` | A/B against DFlash2 | 12 / DSpark gamma 7 (verify 8) | RadixArk's DSpark draft instead of DFlash2; unmeasured here |

### Long sessions: how many 262K contexts fit

The other end of the trade: a few requests, each with the full 262K context.
From the native boot logs:

| Per token of context | Per GDN slot | Pools get |
|---|---|---|
| 42 KiB of KV: 32 target + 10 draft (fp8) | ~0.0735 GiB; a request holds 5, plus `DRAFT_TOKENS` in the verify buffer | free at start − weights − (1 − `MEM_FRACTION`) × 121.6 GiB |

So one 262K session is 10.5 GiB of KV plus ~1.1 GiB of GDN state (draft
10). With the FP4-head RadixArk checkpoint (~25.9 GiB with the draft) and
~115 GiB free at start:

| `MEM_FRACTION` | Pools | 262K sessions, draft 10 | draft 16 |
|---:|---:|---:|---:|
| 0.80 | 65.2 GiB | 5 | 5 |
| 0.85 | 71.3 GiB | **6** | 5 |
| 0.90 | 77.3 GiB | 6 | 6 |

Six is the ceiling; a seventh would need ~0.94. The lever is `MAX_RUNNING`
more than `MEM_FRACTION`: the throughput profile's cap of 32 spends ~35 GiB on
GDN state, six requests need ~7. `./serve.sh qwen3.8-27b-longctx` is that
setting (cap 6, draft 10, a GDN slot per session to keep its state cached
between turns, 0.85). Untested as a whole; check it:

```bash
./serve.sh qwen3.8-27b-longctx      # boot log: "KV Cache is allocated ... #tokens: N"; N / 262144 = sessions
python3 bench/longctx.py --ctx 250000 --streams 2 4 6 --gen 1000   # holds each KV resident; ~40 min
```

At 0.85 watch earlyoom (`journalctl -u earlyoom`): if it kills the scheduler
(exit code -15) during a long prefill, lower its `-m` threshold
(`systemctl cat earlyoom`) or use `MEM_FRACTION=0.82`. And budget time as well
as memory: prefill runs one sequence at a time at ~1,200 tok/s, so filling a
262K context takes ~3.5 minutes, six of them over 20.

## 8. Tune draft tokens: the largest single-stream lever

`--speculative-num-draft-tokens` (`DRAFT_TOKENS`). Measured on NVFP4:

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 8 (draft's block size) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**The optima diverge, so pick one:** 16 for interactive/single-stream (+28%
over 8; a second machine measured +41.6%), 10 for concurrent serving at cap
32 (the throughput profile). You cannot have both.

**At cap 12, the default, 11 is the middle that wins most.** Three runs each
on `nvidia/Qwen3.8-27B-NVFP4`, aggregate tok/s
([RESULTS](results/RESULTS.md#draft-10--11--12-at-cap-12-2026-092526)):

| Draft | single-stream | 2 streams | 4 | 8 | 12 |
|---:|---:|---:|---:|---:|---:|
| 10 | 69.8 | 112 | **186** | 259 | **404** |
| **11** (default) | 73.6 | 124 | 169 | **319** | 401 |
| 12 | **74.1** | **125** | 163 | 311 | 392 |

The dip at 4 streams reproduces at 11 and 12 and is not CUDA-graph padding
(every batch size up to 8 is captured); the verify batch then has 44-48 rows
instead of 40, so likely a kernel shape. If your load sits at 3-5 streams,
`DRAFT_TOKENS=10`.

**Another draft: DSpark.** `qwen3.8-27b-dspark` swaps DFlash2 for
[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
(1.86B, `--speculative-algorithm DSPARK`, in SGLang 0.5.20) with everything
else from the base profile, so `bench/perf.py` against each is a
like-for-like comparison. It verifies 8 tokens per step (gamma 7, the card's
value; `DSPARK_GAMMA` to sweep), against DFlash2's 11, so its verify buffer
is smaller. Download and details in the profile.

On the native build, with the Uncensored NVFP4 finetune: **56.4 → 70.9
tok/s** single-stream going from 10 / 32 to 16 / 16, peak aggregate 558 → 375
([RESULTS](results/RESULTS.md#draft-16--cap-16-on-the-native-build)). The
`qwen3.8-27b-single` profile is that setting: `./serve.sh qwen3.8-27b-single`.

Where the two cross, from two native runs of the Uncensored finetune
(aggregate tok/s, 300-token answers):

| Streams | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| draft 10 (cap 32) | 55.4 | 100.5 | **186.8** | **288.5** | **439.6** |
| draft 16 (cap 16) | **62.2** | **115.5** | 179.1 | 254.2 | 375.4 |

**Around 3 concurrent streams.** Below it draft 16 wins, above it draft 10.

Move `MAX_RUNNING` with it. The DFlash2 verify buffer
(`intermediate_ssm_state_cache` in the boot log) costs ~70 MB per running
request per draft token: 23.2 GB at 32 x 10 on the native build, ~37 GB at
32 x 16, which the KV pool pays for. Single-stream is
`DRAFT_TOKENS=16 MAX_RUNNING=16 ./serve.sh` (~18 GB).

Past 16, `accept_len` *falls* even though more tokens are drafted: the
drafter can't sustain longer correct runs, so you pay draft compute for tokens
that get rejected. Quality is unaffected at any value, since every draft token
is verified against the target.

## Quality

HumanEval, 164 problems at temperature 0, each executed against its real unit
tests (real pass@1, not self-judged). Measured with the Docker build; the
harness, which sandboxes the generated code in a container, is not part of
this Docker-free project. Details in [`results/RESULTS.md`](results/RESULTS.md#quality--humaneval).

| Target | Thinking off | Thinking on | Tokens/problem (off / on) |
|---|---:|---:|---:|
| FP8 | **97.6%** | 157/157 answered correctly, 7 ran out of budget | ~200 / ~1,569 |
| NVFP4 | 93.9% | **97.0%** | ~200 / ~945 |
| NVFP4, **native build** | 95.1% | **98.8%** (`reasoning_effort` medium) | ~206 / ~737 |

The native build scores within the ±2-problem noise of the Docker build
([RESULTS](results/RESULTS.md#humaneval-on-the-native-build)); `medium`
thinking effort beat `xhigh` there with fewer tokens and fewer runaways.

FP8 with thinking off already matches NVFP4 with thinking on.

---

## Other models: profiles

A profile is one file in [`models/`](models/): where the model's weights
are, the SGLang version it runs on, and a `model_args` function with the
flags only that model takes. Everything else (checks, environment, the flags
all models share, the service, the manifest) is common.

```bash
ls models/                        # gemma4-31b  qwen3.8-27b(-single,-longctx,-throughput,-dspark)  qwen3.8-flash-next
./serve.sh gemma4-31b             # serve it
./serve.sh gemma4-31b install     # its SGLang version, if it differs
./serve.sh gemma4-31b manifest
```

One model at a time: a second server on the same port refuses to start, and
the service runs one profile. Profiles on the same SGLang version share a
venv. To add a model, copy the profile closest to it and change the weights,
`SERVED_MODEL_NAME` and `model_args`.

### Gemma 4 31B

[`models/gemma4-31b.sh`](models/gemma4-31b.sh), from the SGLang cookbook's
Gemma 4 recipe: BF16 weights, the paired `-assistant` model for MTP (NEXTN,
5 steps, 6 draft tokens), the `gemma4` reasoning and tool-call parsers.

```bash
HF=~/spark/venv/bin/hf
$HF download google/gemma-4-31B-it --local-dir /models/gemma-4-31B-it
$HF download google/gemma-4-31B-it-assistant --local-dir /models/gemma-4-31B-it-assistant
./serve.sh gemma4-31b             # serves as "gemma-4-31b" on :8888
```

- **Not measured on GB10.** The cookbook has no DGX Spark cell for Gemma 4, so
  every value in the profile is a starting point. Run `bench/perf.py` (and
  `MTP=0 ./serve.sh gemma4-31b` for the baseline) before trusting it.
- **Expect it to be slow.** 31B in BF16 is ~62 GB of weights, and decode on
  GB10 is bound by memory bandwidth (~273 GB/s): roughly 4 tok/s per stream
  before speculative decoding, which is why MTP is on. The QAT release keeps
  BF16 weights, so it is no faster to serve.
- **No revision pinned yet.** Add `--revision <sha>` when you download; the
  server prints the revision it finds at startup.
- **Gemma repos on the Hub have been gated.** A 401/403 from `hf download`
  means accepting the license on the model page and `hf auth login`.
- **Attention backend:** none is passed; SGLang picks `triton` for Gemma 4 on
  this GPU, which image inputs need.
- **Thinking is off by default** in Gemma 4's template, unlike Qwen's; turn it
  on per request with `chat_template_kwargs: {"enable_thinking": true}`.

### Qwen3.8-Flash-Next

[`models/qwen3.8-flash-next.sh`](models/qwen3.8-flash-next.sh): the 176B MoE
(125B + a 51B n-gram table, 6B active) on one Spark, NVFP4, with its own MTP
head. Flags from the SGLang cookbook's single-Spark cell; the n-gram table is
read in place from the checkpoint (below).

```bash
HF=~/spark/venv/bin/hf
$HF download RadixArk/Qwen3.8-Flash-Next-NVFP4 --revision <sha> \
  --local-dir /models/Qwen3.8-Flash-Next-NVFP4   # 126 GiB, table and MTP head inside
./serve.sh qwen3.8-flash-next                    # serves as "qwen3.8-flash-next" on :8888
python3 patches/test_gb10_ple_mmap.py            # optional, CPU only: the patch below
```

**Where the n-gram table lives.** The checkpoint is 126 GiB on 121.6 GiB of
usable memory; it fits because the 47.7 GiB FP8 table stays on the NVMe and
the GPU reads its rows through the host page tables (16 rows per token).
Stock SGLang does that from a *copy*: `--ple-offload-backend file` creates a
sparse 47.7 GiB file under `$SGLANG_CACHE_DIR/ple` and the weight loader
writes the whole table into it on every boot, ~10 minutes into a fresh file
and ~55 into a filled one (the cookbook's advice: delete it before each
start). This profile skips the copy: `patches/gb10_ple_mmap.py` maps the
checkpoint's own shard tensors (`...ngram_embedding.shard_<k>.weight`)
read-only and points the gather kernel at them. Nothing is written and no
second 47.7 GiB exists; the rest of the file backend stays stock (the device
check, `MADV_RANDOM`, page-cache hints before prefill, the 8 GiB cap on the
table's resident set). `PLE_TABLE=file` switches back to stock for a
comparison. The patch loads through `patches/sitecustomize.py`, only into
this profile's server (`PYTHONPATH`, `GB10_PLE_MMAP=1`), and refuses to run
on an SGLang whose code no longer matches 0.5.20.

**What to check at the first boot:**

- `PLE table: layer <n> read in place from /models/...: <rows> of <rows>
  rows in <k> shards (47.7 GiB F8_E4M3) ..., nothing written`, and later
  `... shard tensors left in place`. No `file-backed mmap` line and nothing new under
  `~/.cache/sglang/ple`.
- The first answer is coherent. Garbage output means the rows are read
  wrong: `PLE_TABLE=file` gives the stock reference to compare with.
- A CUDA fault or `illegal memory access` on the first request: retry with
  `GB10_PLE_MMAP_WRITABLE=1`, which maps copy-on-write instead of read-only
  (in case the GPU needs writable page-table entries; the file is still never
  written). Note which of the two booted: it is the one to keep.

**The plan, and what decides it.** The cookbook measured 27.5 tok/s
single-stream with MTP on this cell; the int4 AutoRound recipe on vLLM does
~66. The difference is the expert kernel (cutlass W4A4 vs Marlin W4A16), not
the engine, so:

1. `./serve.sh qwen3.8-flash-next`, then `bench/perf.py`: boots without the
   copy, and the baseline on this box.
2. `MOE_RUNNER_BACKEND=marlin ./serve.sh qwen3.8-flash-next` (and
   `FP4_GEMM_BACKEND=marlin` for the dense NVFP4 layers): the same weights on
   W4A16 kernels. Near 60 tok/s makes one engine for everything worth it.
3. Only if 2 does not get there: port the AutoRound checkpoint.

**Memory.** 8 concurrent requests with MTP (5 fp32 GDN slots each, ~113 MB a
slot), 24 without; a ~93K-token KV pool at 0.85. `MAMBA_SSM_DTYPE=bfloat16`
halves the slots. Every knob is commented in the profile.

---

## Moving from the Docker toolkit / SparkStation

For a Spark that runs an earlier version of this recipe. The weights and the
draft are the same, so nothing needs downloading again.

1. **Stop the old server**: `./stop.sh` in the toolkit checkout, or
   `sparkstation stop`. Either holds most of the GPU memory, and the toolkit
   also holds :8888.
2. **Point the profile at the weights you have**: `MODEL_DIR` / `DRAFT_DIR`
   in `models/qwen3.8-27b.sh`
   to the snapshot directories in the toolkit's cache (step 3), e.g.
   `~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7a…`.
   Nothing is copied or downloaded. To move them somewhere tidier later, see
   [`docs/cache-transfer.md`](docs/cache-transfer.md).
3. **Install and serve**: steps 2, 5 and 6 above. `./serve.sh install` reuses
   the venv for `SGLANG_VERSION` if it exists.
4. **Carry your flags over.** Your `DF_EXTRA` maps to the knobs in
   `models/qwen3.8-27b.sh`. For example, this toolkit run

   ```bash
   DF_EXTRA="--model-path Qwen/Qwen3.8-27B-FP8 --revision 017b9c7a… \
     --mamba-ssm-dtype bfloat16 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
     --max-mamba-cache-size 160 --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
     --speculative-num-draft-tokens 10" ./start-dflash.sh
   ```

   is the profile as committed, except `MEM_FRACTION=0.80`. The toolkit's own
   flags (flashinfer, 8192-token chunks, no prefill graphs, extra_buffer,
   parsers, metrics, CPU pinning) are all carried over.
5. **Benchmarks.** `bench/` now defaults to `http://127.0.0.1:8888/v1` with no
   API key. Behind the SparkStation gateway you had to set
   `GB10_BASE_URL`/`GB10_API_KEY`/`GB10_METRICS_URL`; natively none of them
   are needed.

Once the native server has booted, the toolkit's image and SparkStation can
go. Keep the toolkit checkout for as long as the profile points into its
cache.

---

## Traps

Each of these cost real time.

- **Pin the revision when you download, not just the draft.** `hf download`
  without `--revision` takes the repo's mutable default. Upstream moved the
  NVFP4 default off the pinned `554ebba9` on 2026-08-22, and an unpinned
  launch served the newer `319f741c` for days before anyone noticed: the boot
  log does not say which revision loaded. `serve.sh` loads local directories
  only, and prints the revision each one holds at startup; `unknown` means a
  directory it cannot vouch for.
- **earlyoom kills the server at 0.85.** Memory is unified, so
  `--mem-fraction-static` prices the host's memory too. 0.85 of 128 GB leaves
  ~8 GB, which is exactly DGX OS earlyoom's SIGTERM threshold, and the first
  long prefill or graph capture dips under it: the scheduler dies with
  `exit code -15` and no traceback (`journalctl -u earlyoom` shows the kill).
  SGLang's own GB10 sweep lost 15 of 48 configs at 0.85 and none at 0.80, and
  FP8 + bfloat16 state + DFlash2 (this default) was among the most exposed.
  0.85 ran fine here under Docker; `MEM_FRACTION=0.85` is yours to try. At
  0.80 the KV pool is roughly 150K tokens smaller, which short-prompt
  concurrency never notices.
- **mem-fraction is not a perf lever.** 0.82 / 0.85 / 0.90 all measure the
  same: concurrency is bound by GDN slots, long context by prefill. Above
  0.85, also pass `--max-total-tokens 1048576` in `EXTRA_ARGS`: an uncapped KV
  pool grows but nothing uses it, and the lost headroom cost 18% at 32 streams.
- **The first boot is slow, and the very first FP8 boot is very slow.**
  Kernels are compiled and autotuned on first use; the Docker build's first
  FP8 boot took 77 minutes, later boots 5.5 (NVFP4: 3.4). Don't kill a quiet
  first boot. The caches are `~/.cache/flashinfer` and `~/.triton`; deleting
  them, or changing the FlashInfer/Triton version, pays it again.
- **`fatal error: Python.h` at boot, then "Triton is not supported on current
  platform".** Triton builds a C launcher on first use and needs the Python
  headers; the Docker image had them, DGX OS doesn't:
  `sudo apt install python3.12-dev`. `00-check-host.sh`, `install` and
  `serve.sh` now check for them (and for a C compiler) before anything else.
- **`Ninja build failed`, compilers `Killed` (code 137), during the first
  CUDA-graph capture.** FlashInfer compiles the CUTLASS FP4 GEMMs on first use,
  after the server already holds `MEM_FRACTION` of memory, and ninja defaults
  to one job per core. Twenty-odd `nvcc` on those templates exhaust the ~23 GB
  left, and the kernel kills them. `JIT_JOBS` (default 2) caps the parallelism;
  what did compile is cached, so a rerun only builds the rest, and later boots
  build nothing.
- **`FileNotFoundError: 'ninja'` while capturing CUDA graphs.** FlashInfer's
  JIT runs a bare `ninja`, which lives in the venv's `bin/`; the Docker image
  had its venv active, a plain `python` from the venv does not. The server is
  now started with the venv's `bin/` on `PATH`.
- **Keep the CUDA compiler at the driver's version.** The driver is CUDA 13.0,
  and one of SGLang's dependencies pulls nvcc 13.4 from pip, whose output a
  13.0 driver may refuse. The lock pins 13.0, and the server starts with the host's
  `/usr/local/cuda` (13.0 on DGX OS) first on PATH, as the official image does. A
  `CUDA_HOME is not set` error means neither was found.
- **"DFLASH block size mismatch" at boot is expected.** The draft was trained
  with blocks of 8; any other `DRAFT_TOKENS` logs this warning and works.
- **Offline by default.** `HF_OFFLINE=1` sets `HF_HUB_OFFLINE=1` for the
  server: with local directories nothing should reach the Hub, and on the
  Docker build SGLang's early speculative-algorithm probe did look the draft
  up there. If a boot fails asking for the network, that lookup is back; set
  `HF_OFFLINE=0` to let it through.
- **5 GDN slots per request, not 4.** DFlash2's verify needs an extra under
  `--mamba-radix-cache-strategy extra_buffer`. Sized on 4, a pool meant for 16
  lands at 12. `MAX_RUNNING` does the ×5 for you.
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one server instance. Size A/B deltas against 8%, and re-measure on a
  fresh boot before believing a small win.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  HumanEval problems between runs. Don't read a sub-2% delta as a regression.
- **Thinking is on by default**, and a small `max_tokens` returns empty
  `content` with everything in `reasoning_content`. Always pair a token cap with
  a `finish_reason` check: truncation otherwise reads as a quality regression.
  That mistake made a 97.0% run score 90.9%.
- **Count `completion_tokens`, not stream events.** DFlash2 emits ~3.75 tokens
  per event; event-counting inflates throughput ~4×.
- **The draft has two names.** The SGLang cookbook uses
  `incoai/Qwen3.8-27B-DFlash2`, a mirror of `z-lab/Qwen3.8-27B-DFlash2`. The
  revision pin is z-lab's.

## Layout

```
serve.sh       the launcher: machine knobs, profile choice; also `install` and `manifest`
models/        one profile per model: qwen3.8-27b.sh (this README), its
               variants -single, -longctx and -throughput, gemma4-31b.sh,
               qwen3.8-flash-next.sh
patches/       gb10_ple_mmap.py (+ its kernel): Flash-Next's n-gram table read
               in place · sitecustomize.py (loads it) · test_gb10_ple_mmap.py
scripts/       00-check-host · 01-install · serve-sglang · install-service
               build-manifest · lib/config.sh (shared defaults)
requirements/  one lock per SGLang version (sglang-<ver>-<arch>-py<py>.txt) · constraints-cuda130.txt
bench/         common.py · perf.py · longctx.py
results/       RESULTS.md: all measurements (Docker build)
               BUILD-MANIFEST.md: the build behind them
               REPRODUCTION.md: an independent run
               runs/: saved benchmark output (gitignored)
docs/          cache-transfer.md: move the weights to a new machine, offline
```

The benchmarks read `GB10_BASE_URL` (API root including `/v1`, default
`http://127.0.0.1:8888/v1`), `GB10_MODEL` and `GB10_API_KEY` from the
environment; nothing is baked in. The same command measures any
OpenAI-compatible server, one model and endpoint per run:

```bash
python3 bench/perf.py
GB10_BASE_URL=http://other-box:8000/v1 GB10_MODEL=default python3 bench/perf.py --levels 1 8 16
```

Without `GB10_MODEL` the model is read from `/v1/models`. Engine metrics (queue
time, KV usage, accept length, cached tokens) come from `GB10_METRICS_URL`,
default `<server>/metrics`; the server is started with them enabled. Behind a gateway, point it
at the engine.

`perf.py` takes `--levels`, `--prefill`, `--only <section>` and `--no-warmup`.
Without `--levels` the concurrency sweep follows the server's request cap:
1 2 4 8 16 24 32 up to it, the cap itself, and one burst row at 1.5x the cap
(flagged as queued, left out of the peak). Cap 12 gives 1 2 4 8 12 18.
`longctx.py` takes `--ctx`, `--streams` and `--gen N` (forced generation, a KV
capacity test). See each script's docstring.

Every run records what it measured:

- **Server settings.** The header lists the engine's effective settings from
  SGLang's `/get_server_info` (model and draft paths, draft tokens, request
  cap, ...). That catches the "which draft count" trap from the output alone,
  and "which checkpoint" by path; the revision a local directory holds is in
  the server's startup lines and in `./serve.sh manifest`.
- **Saved output.** The full output is saved under `results/runs/`
  (`GB10_RUN_DIR`; set it empty to skip saving).
- **GPU.** When the server is local, peak temperature, lowest SM clock and
  throttle reasons come from `nvidia-smi` per section or row
  (`GB10_GPU_WATCH=0|1`). A row is flagged when the GPU throttled or came
  within 2 °C of the 80 °C suspend threshold.
- **Engine restarts.** A restart mid-run, detected from
  `process_start_time_seconds`, is flagged too, and such rows are left out of
  the peak.

Record the install next to a run with
`./serve.sh manifest > results/BUILD-MANIFEST-native.md`: installed
versions, whether the venv still matches the lock, cached snapshots, driver,
and the running server's command line.

## Caveats

Throughput figures are code generation at temperature 0 on short prompts.
Long-context behaves very differently: a ~120K-token prompt costs ~95 s before
the first token on NVFP4, ~140 s on FP8 (full prefill curve in
[`results/RESULTS.md`](results/RESULTS.md)). Single machine; treat the ordering
of findings as durable and absolute numbers as indicative, doubly so until the
native build is re-measured.

MIT — see [LICENSE](LICENSE).
