# DeepSeek-V4.1-Flash-1017 on two Sparks: TensorFold, TP=2

DeepSeek's V4.1 Flash (the 1017 release) in LibertAIDAI's EXL3 2.9 bpw
quantization (196 GiB of weights, plus 189 GiB of DeepSeek's Engram tables
read from NVMe), served across **both** DGX Spark with tensor parallelism over
the ConnectX-7 cable (RoCE v2), 1M-token context. The engine is TensorFold's
Python line with bertholomus's V4.1 port, from
[Elevatormusic's fork](https://github.com/Elevatormusic/TensorFold) at a pinned
commit, which adds what agents need: forced tool calls, structured output
under speculative decoding, fault isolation. This directory is
[Elevatormusic's recipe](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark)
(Apache-2.0, [CREDITS](CREDITS.md)) with its launch and settings reshaped like
the rest of this project: `config.sh` with `${VAR:-default}` knobs instead of
`.env`, `./serve.sh` instead of `./start.sh`, the helpers next to it.

Their numbers (17-18 October 2026, one pair, [docs/RECIPE.md](docs/RECIPE.md),
every one backed by a file in [runs/](runs/2026-10-18/INDEX.md)); not yet
measured here:

| | vLLM kit, same pair | This engine |
|---|---:|---:|
| Decode, code / prose, 1 request | 39.9 / 27.0 tok/s | **92.2 / 56.8** |
| 4 requests at once, code, aggregate | (1 at a time) | 140 tok/s |
| Prefill, cold prompt 8K / 32K / 64K / 128K | 789 / 925 / ~800 / — | 2,179 / 2,165 / 2,037 / 1,961 |
| 1,035,020-token prompt | — | 1,266 tok/s (TTFT 13.6 min), needle found |
| Second question on a 24K document (prefix cache) | | TTFT 1.1 s instead of 12.8 |

For scale: our 27B on one Spark decodes code at 88-94 tok/s and peaks at
410-436 tok/s over 16 requests; this is a much larger model at the same
single-stream speed, holding both machines and at most 4 requests. Its
output quality is not measured by the recipe (docs/KNOWN-LIMITS.md):
HumanEval and `bench/answers.py` here would be the first numbers.

## What it takes

- **Both Sparks, cabled** (one QSFP cable between the CX7 ports, NVIDIA's
  [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks)),
  static addresses on the link and key-based ssh from the head to the
  worker; [docs/NETWORK.md](docs/NETWORK.md). The same user and paths on both.
- **~500 GB of disk and ~100 GB of free memory on each.** Nothing else runs:
  no SGLang profile, no vLLM, on either node (`serve.sh` refuses on the head).
- Docker with GPU support (DGX OS has it), `rsync`, `curl`, the `hf` CLI
  (`~/spark/venv/bin/hf` from `../serve.sh install` will do), earlyoom.
- Time: the ~400 GB download, then a first start of ~15 minutes.

## Setup and run (on the head, in this directory)

```bash
cp config.local.sh.example config.local.sh   # your CX7 addresses, RDMA devices, ssh to the worker
./download.sh          # weights + Engram tables (~400 GB, resumable), sha256 check, copy to the worker
./patch-template.py --worker   # the chat template at a pinned commit + a 1-hunk fix (AGPL: not stored here)
./build.sh             # the engine image at TF_COMMIT, copied to the worker (~10 min)
./host-tuning.sh       # once, sudo on both nodes: no swap for the engine, memory sysctls, earlyoom; --undo removes it
./serve.sh             # preflight, rank 0 here, rank 1 there, waits for the API (~15 min first, 2-4 min later)
./verify.sh            # five real requests: only a reply proves TP=2 works
./status.sh            # both containers and the API
./stop.sh              # both ranks; their logs go to logs/
```

Then, as for the other servers (same port 8888, so `../bench/` works):
`python3 ../bench/perf.py --levels 1 2 4`, model `DeepSeek-V4.1-Flash-1017`.
Thinking is off by default; per request `"reasoning_effort": "high"`
([docs/CONNECT.md](docs/CONNECT.md): thinking levels, tools, structured output).

## Knobs (config.sh, each overridable from the shell)

| Variable | Default | |
|---|---|---|
| `HEAD_IP`, `WORKER_IP`, `WORKER_SSH` | 10.100.80.1, .2, = `WORKER_IP` | the CX7 link; set in `config.local.sh` |
| `RDMA_DEVICES`, `NCCL_SOCKET_IFNAME`, `GID_INDEX` | both CX7 RoCE devices, `enp1s0f1np1`, 3 | from `ibdev2netdev`, `ip -br addr`, `show_gids`; `GID_INDEX=-1` after a link flap |
| `MODEL_VIEW` | `/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | `model/` + `engram/`, the same path on both nodes |
| `CACHE_DIR` | `~/.cache/gb10-tensorfold-dsv41` | rank weight caches, spilled prompts, kernels (~100 GiB a node) |
| `TF_COMMIT`, `IMAGE` | `92aa710b533d`, `tensorfold-dsv41:<commit>` | the engine build (`./build.sh`) |
| `MAX_RUNNING` | 4 | concurrent requests (`--parallel`) |
| `DRAFT_TOKENS` | 5 | DSpark draft depth (`--mtp-drafts`); 4 measured slower |
| `CONTEXT_LENGTH` | 1048576 | 262144 measured no faster |
| `CHUNKED_PREFILL` | 2048 | prompt chunk (`TF_DS_PREFILL_CHUNK`); 4096 measured 6-15% slower |
| `SPILL_GIB` | 64 | kept prompts also on disk, so a conversation survives a restart; 0 = off |
| `TEMPERATURE`, `TOP_P` | 1.0, 1.0 | DeepSeek's recommendation; a request's own values win |
| `HOST`, `PORT`, `API_KEY_FILE` | 0.0.0.0, 8888, none | as the other servers here; the key file has one key per line (the tests send none) |
| `SERVED_MODEL_NAME`, `SERVED_ALIAS` | `DeepSeek-V4.1-Flash-1017`, none | |
| `VISION` | 1 | image input on (not measured) |
| `CGROUP_PARENT`, `RESTART_POLICY` | `tensorfold.slice`, no | the no-swap slice from `host-tuning.sh`; docker restarts (docs/OPERATIONS.md first) |
| `PREFLIGHT` | 1 | 0 skips `./preflight.sh` |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `tensorfold serve` flags; more `NAME=value` for both containers, replacing one above of the same name (`EXTRA_ENV='TF_DS_TIMING=1'`) |

`config.local.sh` (not committed) holds the values of your pair, in the same
`NAME="${NAME:-value}"` form, so the shell still wins over it.

## What host-tuning.sh changes, and why it matters beyond this model

The engine holds ~102 of the 128 GiB a node; under load the kernel swapped,
reclaimed and compacted, and long prompts ran 10-30% slower with 1-4 s
stalls (runs/2026-10-18). `./host-tuning.sh` (sudo on both nodes, checks both
before changing either, `--undo` restores the saved values) sets:

- a `tensorfold.slice` with `MemorySwapMax=0` for the rank containers;
- `vm.swappiness=10`, `vm.compaction_proactiveness=0`, `vm.extfrag_threshold=500`,
  `vm.min_free_kbytes=1048576`, `vm.watermark_scale_factor=50`: slow prompt
  chunks went from 46 of 248 (background compaction on) to 2;
- earlyoom: SIGTERM below 1 GiB available, SIGKILL below 0.5, preferring the
  `tensorfold` process and avoiding sshd, systemd, dockerd and the display
  stack, so a full node loses the engine, not ssh.

`../scripts/host-tuning.sh`, the one-Spark version for the SGLang profiles,
sets the same sysctl values and an earlyoom that also prefers `tensorfold`,
so preflight accepts either; only this one adds the no-swap slice the rank
containers run in. These are system-wide: they stay for the SGLang profiles too (the earlyoom
thresholds become 1 / 0.5 GiB instead of DGX OS's own), until `--undo`. The
same settings are worth an A/B on one Spark with the 27B: its prefill varies
from run to run (results/RESULTS.md), which compaction stalls could explain.
Not measured yet.

## Files

| | |
|---|---|
| `config.sh`, `config.local.sh.example` | settings and the helpers the scripts share |
| `serve.sh`, `stop.sh`, `status.sh`, `verify.sh` | run it |
| `preflight.sh` | the checks `serve.sh` runs on both nodes: files, same image, RoCE GID, memory, disk, host settings, port |
| `download.sh`, `patch-template.py`, `build.sh`, `docker/Dockerfile` | weights, chat template, engine image |
| `host-tuning.sh`, `deploy-gate.sh` | host memory settings; a new engine image with automatic rollback |
| `tests/` | the 32-test agent suite, the grammar check, prompt-encoding tests, a benchmark, the Hermes gate ([tests/README.md](tests/README.md)) |
| `docs/` | the recipe page, what was changed in the engine and why, benchmarks, network, operations, known limits |
| `runs/` | every log and result behind the numbers; `weights.sha256` for `download.sh` |
| `raw/` | the scripts as they ran on the measured pair |
| `AGENTS.md` | the steps for an AI agent setting this up |

Traps from the recipe ([docs/OPERATIONS.md](docs/OPERATIONS.md)): never
`docker exec` into a rank while it serves (the exec counts against its
memory); never stop the display manager on a Spark (it powered a node off);
exit code 75 means the lane broke on a device fault: `./stop.sh && ./serve.sh`.
