# Qwen3.8-27B on DGX Spark (GB10)

A measured recipe for the fastest Qwen3.8-27B setup I could get on a DGX Spark:
**SGLang + NVFP4 + DFlash2 speculative decoding**, managed by
[SparkStation](https://github.com/kshetrajna12/sparkstation).

| Single-stream | Peak aggregate | TTFT | HumanEval pass@1 |
|---:|---:|---:|---:|
| **78.6 tok/s** | **480.7 tok/s** (16 streams) | **190 ms** | **97.0%** |

Those two throughput figures come from different settings: single-stream is at
`--speculative-num-draft-tokens 16`, peak aggregate at 8. The optima diverge —
see step 6.

**NVFP4 beats FP8, and SGLang beats vLLM, on this hardware.** The widely-shared
"FP8 on vLLM at ~32 tok/s" recipe is roughly half this speed.

Reproduce with [`bench/`](bench/). Full data in
[`results/RESULTS.md`](results/RESULTS.md). The exact
build inputs behind those numbers are in
[`results/BUILD-MANIFEST.md`](results/BUILD-MANIFEST.md).

## Ingredients

| Component | Pin |
|---|---|
| Target | `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` |
| Draft | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c` |
| Image | `lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2` |
| Toolkit | [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) @ `c90d8c34` |
| SparkStation | [kshetrajna12/sparkstation](https://github.com/kshetrajna12/sparkstation) @ `6a19736` |
| Disk | ~110 GB |

All model repos are public; no token needed. Verified on Ubuntu 24.04,
kernel 6.17-nvidia, driver 580.173.02 / CUDA 13.0, Docker 29.2.1, 128 GB unified.

---

## 1. Verify GPU passthrough

```bash
./scripts/00-verify-gpu.sh
```

DGX OS ships CDI, not a registered Docker runtime — `docker info` showing only
`runc` is normal and `--gpus all` still works.

## 2. Pull the image, fetch the weights

```bash
./scripts/01-build-and-fetch.sh     # pulls the official image + weights
```

LMSYS published official DFlash2 images on 2026-08-22: `dev-cu13-qwen38-27b-dflash2`,
`dev-qwen38-27b-dflash2`, `dev-cu12-qwen38-27b-dflash2`, arm64 included. Only the
`qwen38-27b-dflash2` tag name is absent. The official image needs no `lm_head` patch.

## 3. Run standalone, get a baseline

Get a number before adding orchestration.

```bash
cd ~/spark/Qwen3.8-27B-SGLang-DGX-Spark   # required: start.sh uses WORK_DIR="$(pwd)"
cp -n .env.sample .env
IMAGE=lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2 \
  DF_EXTRA="--mem-fraction-static 0.85" ./start-dflash.sh
```

Serves on **:8888**, OpenAI- and Anthropic-compatible. First boot ≈3 min.

`IMAGE=` is required: `start-dflash.sh` defaults to the locally built tag and
will build it if absent, which is exactly what step 2 avoids.

**Use 0.85 on a first boot.** The toolkit defaults NVFP4 to `0.90` and generic
`start.sh` to `0.95`; both this project and the toolkit record machines
unrecoverably wedged during weight load at higher fractions — GB10 unified
memory starves the OS.

```bash
GB10_BASE_URL=http://127.0.0.1:8888/v1 GB10_MODEL=qwen3.8-27b-sglang \
  python3 bench/perf.py
```

## 4. Add SparkStation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/kshetrajna12/sparkstation.git
cd sparkstation && git checkout 6a19736    # see the pin note below
uv sync
cp .env.example .env      # TOTAL_UNIFIED_MEMORY_GB=128, MEMORY_HARD_LIMIT_GB=110
```

**Pin SparkStation.** Its default branch moved on: from 2026-08-29 (`78f52e3`)
the `generic` profile runs a different model (`qwen-flash-next`), and
`qwen3.8-sglang` is kept only for rollback. `6a19736` is the last commit whose
`generic` profile starts this recipe, and both files in [`patches/`](patches/)
apply to it.

Two required `models.yaml` edits in the `qwen3.8-sglang` entry. The entry ships
pinned to a second machine, and it names the locally built image tag, which
the registry does not have:

```diff
-    host: worker1
+    host: primary
-    docker_image: "lmsysorg/sglang:qwen38-27b-dflash2"
+    docker_image: "lmsysorg/sglang:dev-cu13-qwen38-27b-dflash2"
```

Keep the old tag only if you built the image with `BUILD_LOCAL=1`.

```bash
uv run sparkstation start -d --profile generic
```

Gateway on **:8000**. The model container is bridge-networked, so its internal
:8000 maps to host **:8001** — no collision despite both reporting 8000.
Gateway overhead measured at zero.

Through the gateway, benchmark with its key and read metrics from the model
container:

```bash
GB10_BASE_URL=http://127.0.0.1:8000/v1 GB10_MODEL=default \
  GB10_API_KEY=<LITELLM_MASTER_KEY from .env> \
  GB10_METRICS_URL=http://127.0.0.1:8001/metrics python3 bench/perf.py
```

## 5. Unlock concurrency

The shipped config caps you at 4 concurrent requests. **Three flags are
co-limiting** — raising only the obvious one measures nothing.

```diff
- "--max-running-requests"      "4"  →  "16"
- "--max-mamba-cache-size"     "20"  →  "80"    # 16 requests × 5 slots
- "--cuda-graph-max-bs-decode"  "4"  →  "16"    # else batches >4 fall back to eager
```

| Streams | cap 4 | cap 16 |
|---:|---:|---:|
| 1 | 58.3 | 52.9 |
| 8 | 190.1 | 315.9 |
| **16** | — | **480.7** |
| 32 | — | 469.8 |

16 is the cap, not a hardware optimum. SGLang clamps
`max_running_requests = max_mamba_cache_size / 5`, so the peak lands wherever the
pool allows and the 24/32 rows are queueing against it.

A larger pool does raise aggregate — we measured **474.0 tok/s at 32 streams**
with pool 160, against 362.7 at pool 80. [An independent run](results/REPRODUCTION.md)
reports higher still (606.4 at pool 160, 640.9 at pool 240 with cap 48); we did
not reproduce those magnitudes, see that file for why.

Aggregate is bought with KV, so these are short-prompt figures. Pool 240 needs
~47 GB of GDN state, which on 128 GB means either a mem-fraction we measured as
harmful or a KV pool too small to be useful — it failed to boot here. Treat
pool 160 as the practical ceiling.

Leave `--max-total-tokens 1048576` in place: uncapping it grows the KV pool but
nothing uses the space, and the lost headroom cost 18% at 32 streams.

Costs ~5% single-stream and 11.8 GB of GDN state. Details in
[`patches/sparkstation-models.yaml.md`](patches/sparkstation-models.yaml.md).

## 6. Tune draft tokens — the largest single-stream lever

`--speculative-num-draft-tokens` ships at 8. The measured optimum is ~16,
worth **+28%** single-stream. A second machine measured +41.6%; our 78.6 is
probably the low end of the spread. See [RESULTS](results/RESULTS.md#draft-tokens).

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 8 (default) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**The optima diverge — pick one:** 16 for interactive/single-stream, **10 for
concurrent serving** (435 vs 385 aggregate). You cannot have both.

Past 16, `accept_len` *falls* even though more tokens are drafted — the drafter
can't sustain longer correct runs, so you pay draft compute for tokens that get
rejected. `accept_rate` collapses from 0.80 to 0.32.

## 7. Benchmark quality

```bash
./scripts/run-humaneval.sh          # thinking off, ~5 min
./scripts/run-humaneval.sh think    # thinking on,  ~20 min
```

164 problems at temperature 0, each executed against its real unit tests in a
`--network none` container. Real pass@1, not self-judged.

The script uses the venv from step 2 (`~/spark/venv`: `hf`, pandas, pyarrow);
point `GB10_PYTHON` / `GB10_HF` elsewhere if you keep them in another place.
The sandbox is capped at 2 GB (`GB10_SANDBOX_MEMORY`), 256 processes and 4 CPUs:
memory is unified, the engine already holds most of it, and a runaway candidate
must not push the host into OOM. Failed requests (a 503 while the supervisor
restarts the model) are reported as `REQUEST-ERR`, not counted as wrong answers.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% | ~200 | 3 min |
| Thinking on | **97.0%** | ~945 | 19 min |

Default to thinking off for well-specified functions — 93.9% at a fifth of the
tokens. Switch it on for hard cases.

---

## Updating an existing install

For a Spark that already runs an earlier version of this recipe. The weight,
draft, toolkit and image pins are unchanged, so nothing is re-downloaded or
rebuilt, and step 2 does not need to be run again.

### 1. Update this repo

```bash
cd /path/to/this/repo
git status                 # commit or stash local edits first
git pull
```

`separate/` is gone: its benchmarks now live in `bench/`. Anything you changed
there is still in history, e.g. `git show 26273d1:separate/bench/perf.py`.
`decode_bench.py`, `concurrency_bench.py` and `ppwatch.sh` were not ported.

### 2. Adjust how you call the benchmarks

The `GB10_*` variables work as before. If you used the `separate/` scripts,
rename their variables:

| was | now |
|---|---|
| `BASE=http://host:8000` | `GB10_BASE_URL=http://host:8000/v1` (note `/v1`) |
| `MODEL` | `GB10_MODEL` (optional now: read from `/v1/models`) |
| `API_KEY` | `GB10_API_KEY` |
| `METRICS_PORT` | `GB10_METRICS_PORT` (off unless set) |
| `VLLM_CUSTOM_METRICS_INTERVAL` | `GB10_METRICS_INTERVAL` |

Before comparing new numbers with old runs:

- Single-stream decode now streams and prints two rates. **`e2e` is the old
  figure**. `decode` excludes the time to the first token, so it reads higher.
- The concurrency peak now skips rows that queued or during which the engine
  restarted. The rows are still printed, but "peak aggregate" can land at a
  lower stream count than the old script reported.
- Runs are saved to `results/runs/`, with the server's settings in the header.
  Check `revision=554ebba9…` there before trusting a number.

`run-humaneval.sh` now takes Python and `hf` from `~/spark/venv`, the venv step
2 already created. Export `GB10_WORKDIR` if yours lives elsewhere.

### 3. SparkStation

```bash
cd /path/to/sparkstation
git log -1 --format='%h %ad' --date=short
```

- **At `6a19736` or earlier:** nothing to change. Apply the new `docker_image`
  edit from step 4 if you pulled the official image.
- **Newer than `6a19736`** (you pulled after 2026-08-29): `generic` no longer
  runs this recipe. Move back to the pin. Save your local edits first
  (`models.yaml`, a patched `cli.py` or launcher) — `.env` is untracked and
  stays:

  ```bash
  git diff > ~/sparkstation-local.diff      # a record of your edits
  git stash
  git checkout 6a19736
  git stash pop                             # resolve conflicts, if any
  uv sync
  cp cli.py .venv/lib/python3.12/site-packages/cli.py   # see Traps: uv copies it
  sparkstation stop && sparkstation start -d --profile generic
  ```

  `git apply --check patches/<file>` tells you whether a patch is still needed:
  one that is already applied fails the check, and `git apply -R --check`
  confirms it.

### 4. Cap the serving logs (new)

Apply the `daemon.json` from the log-rotation trap below. It only affects
containers created afterwards, so recreate the model container: `./stop.sh`
and `./start-dflash.sh` for the standalone toolkit, or
`sparkstation stop && sparkstation start` for SparkStation. Check:

```bash
docker inspect -f '{{.HostConfig.LogConfig}}' <container>   # {local map[max-file:5 max-size:50m]}
```

### 5. Smoke test

```bash
GB10_BASE_URL=http://127.0.0.1:8888/v1 python3 bench/perf.py --only ttft
```

The header should list the server settings, including the pinned revision. A
`note: ... unreachable` line means the engine's `/metrics` is not reachable;
set `GB10_METRICS_URL` (behind the gateway) or start SGLang with
`--enable-metrics`.

---

## Traps

Each of these cost real time.

- **Pin the target revision, not just the draft.** `hf download` without
  `--revision` takes the repo's mutable default, and SGLang needs `--revision`
  separately — downloading the right checkpoint does not make the server load
  it. SparkStation's `models.yaml` already carries it; the standalone path does
  not.
- **A fast boot proves a snapshot was cached — not which one.** Upstream moved
  the NVFP4 default off the pinned `554ebba9` on 2026-08-22, and the
  standalone leg served the newer `319f741c` for days before anyone noticed;
  the boot log does not say which revision loaded.
  `ls ~/.cache/huggingface/hub/models--*/snapshots/` (and the toolkit's
  mounted `.cache/`) is the check — see
  [`docs/cache-transfer.md`](docs/cache-transfer.md).
- **Fully-offline first start still touches the network.** SGLang's early
  speculative-algorithm probe resolves the draft config without a revision, so a
  cache holding only the pinned snapshot can still reach for that repo's default
  ref. Prime the cache while online.
- **A working CDI fallback is not automatically usable.** If `--gpus all` fails
  and `--device nvidia.com/gpu=all` works, neither launcher picks that up — both
  hardcode `--gpus all`. You have to edit them.
- **SparkStation restarts healthy models under load.** `HEALTH_CHECK_TIMEOUT_SECONDS=5`
  x 3 failures: a model saturated on long prefill cannot answer a 5s probe, so the
  supervisor kills it mid-job and clients get 503s. Raise it to 30.
- **mem-fraction is not a perf lever here.** 0.82 / 0.85 / 0.90 all measure the
  same. Concurrency is bound by mamba slots, long context by prefill — never by
  memory capacity.
- **The DFlash2 image now exists upstream** (`dev-cu13-qwen38-27b-dflash2`).
- **HF cache symlinks inside the container mount break everything.** The
  container bind-mounts `~/.cache/huggingface`; symlinks *within* it point at
  host-only paths, so it re-downloads and dies with
  `OSError: I/O error: File exists (os error 17)`. Keep real directories there.
- **`host: worker1` assumes a second Spark.** Use `primary`, and check any
  profile you use — a profile-level `host:` wins.
- **`start.sh` uses `WORK_DIR="$(pwd)"`**, not the script dir. Run it from
  inside the repo or your caches land elsewhere.
- **5 mamba slots per request, not 4.** DFlash2's verify needs an extra. SGLang
  clamps `max_running_requests = pool / 5` and logs it. Size the pool at 5×
  target.
- **uv installs `cli.py` into site-packages.** Editing the repo copy does
  nothing — the console script imports from `.venv/bin`. Copy it across too.
- **Gateway reports "not healthy" while working.** `_gateway_healthy()`
  hardcodes `Bearer dummy-key`, so a custom `LITELLM_MASTER_KEY` gets HTTP 400.
  Fixed upstream in SparkStation `33be0f1` (2026-08-31); for older checkouts
  the patch is in [`patches/`](patches/).
- **Only the FLUX launcher forwards `HF_TOKEN`.** A gated model can't
  authenticate its own download; pre-pull on the host.
- **Never `docker rm -f` a managed container.** The supervisor's state goes
  stale; `sparkstation stop && start` reconciles.
- **Nothing rotates the serving logs.** Neither the toolkit's `start.sh` nor
  SparkStation's SGLang launcher passes `--log-opt`, so the container's stdout
  grows without bound in `/var/lib/docker/containers/*/*-json.log` unless the
  Docker daemon caps it. Slow under normal logging, fast with `--log-requests`
  (every 100k-token prompt lands in the log). Cap it daemon-wide — applies to
  containers created afterwards:

  ```bash
  # /etc/docker/daemon.json — merge into the existing file, don't replace it
  { "log-driver": "local", "log-opts": { "max-size": "50m", "max-file": "5" } }
  sudo systemctl restart docker
  ```

  SparkStation's own `data/sparkstation.log` rotates, but `supervisor.log`,
  `gateway.log`, `gateway-proxy.log` (in `~/.sparkstation/logs/`) only rotate
  on restart, and `gateway/.litellm-<port>.log` in the repo is appended
  forever. A logrotate rule with `copytruncate` (the processes keep the files
  open) covers them — logrotate needs absolute paths:

  ```
  /home/YOU/.sparkstation/logs/*.log /home/YOU/sparkstation/gateway/.litellm-*.log {
      weekly
      rotate 4
      maxsize 100M
      compress
      missingok
      notifempty
      copytruncate
  }
  ```
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one server instance. Size A/B deltas against 8%, and re-measure on a
  fresh boot before believing a small win.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  HumanEval problems between runs. Don't read a sub-2% delta as a regression.
- **Thinking is on by default**, and a small `max_tokens` returns empty
  `content` with everything in `reasoning_content`. Always pair a token cap with
  a `finish_reason` check — truncation otherwise reads as a quality regression.
  That mistake made a 97.0% run score 90.9%.
- **Count `completion_tokens`, not stream events.** DFlash2 emits ~3.75 tokens
  per event; event-counting inflates throughput ~4×.

## Layout

```
bench/     common.py · perf.py · longctx.py · humaneval/{generate,execute,report}.py
scripts/   00-verify-gpu · 01-build-and-fetch · build-manifest · run-humaneval
patches/   models.yaml edits · gateway-health fix
results/   RESULTS.md — all measurements
           BUILD-MANIFEST.md — resolved build inputs for those numbers
           runs/ — saved benchmark output (gitignored)
docs/      cache-transfer.md — move the model cache to a new machine, offline
```

Scripts read `GB10_BASE_URL` (API root including `/v1`), `GB10_MODEL` and
`GB10_API_KEY` from the environment; nothing is baked in. The same command
measures any OpenAI-compatible server, one model and endpoint per run:

```bash
GB10_BASE_URL=http://127.0.0.1:8888/v1 GB10_MODEL=qwen3.8-27b-sglang python3 bench/perf.py
GB10_BASE_URL=http://127.0.0.1:8000/v1 GB10_MODEL=default python3 bench/perf.py --levels 1 8 16
```

Without `GB10_MODEL` the model is read from `/v1/models`. Optional engine
metrics (queue time, KV usage, accept length, cached tokens) come from
`GB10_METRICS_URL`, default `<server>/metrics`; behind the SparkStation gateway
point it at the model container, `http://127.0.0.1:8001/metrics`, and start
SGLang with `--enable-metrics`. Without it those columns show `n/a`.

`perf.py` takes `--levels`, `--prefill`, `--only <section>` and `--no-warmup`;
`longctx.py` takes `--ctx`, `--streams` and `--gen N` (forced generation, a KV
capacity test). See each script's docstring.

Every run records what it measured:

- **Server settings.** The header lists the engine's effective settings from
  SGLang's `/get_server_info` (model path, revision, draft tokens, request
  cap, ...). That catches the "which snapshot loaded" and "which draft count"
  traps above from the output alone.
- **Saved output.** The full output is saved under `results/runs/`
  (`GB10_RUN_DIR`; set it empty to skip saving).
- **GPU.** When the server is local, peak temperature, lowest SM clock and
  throttle reasons come from `nvidia-smi` per section or row
  (`GB10_GPU_WATCH=0|1`). A row is flagged when the GPU throttled or came
  within 2 °C of the 80 °C suspend threshold.
- **Engine restarts.** A restart mid-run, detected from
  `process_start_time_seconds`, is flagged too, and such rows are left out of
  the peak.

## Caveats

Throughput figures are code generation at temperature 0 on short prompts.
Long-context behaves very differently — a ~120K-token prompt costs ~95 s before
the first token (full prefill curve in
[`results/RESULTS.md`](results/RESULTS.md)). Single machine; treat the ordering
of findings as durable and absolute numbers as indicative.

MIT — see [LICENSE](LICENSE).
