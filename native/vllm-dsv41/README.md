# DeepSeek-V4.1-Flash-1017 on two Sparks: tuned vLLM, TP=2

The same model and the same two weight folders as
[`../tensorfold-dsv41/`](../tensorfold-dsv41/README.md) (LibertAIDAI's EXL3 2.9 bpw
pack + DeepSeek's 1017 Engram tables), served by another engine: sudoingX's tuned
vLLM ([dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41):
vLLM `6e517b15` + 22 patches + the `vllm_exl3_moe` plugin, exllamav3 1.6.0) with
styles01's 1017 overlay, from styles01's published image
(`ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks`, pinned in `image.lock`). Their
recipe: [styles01/deepseek-v41-1017-exl3-2x-dgx-sparks](https://github.com/styles01/deepseek-v41-1017-exl3-2x-dgx-sparks)
(`METHOD-1017.md`, `scripts/launch/dsv41_tp2.sh`; runbook in
[styles01/sparkrun-recipes](https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/deepseek-v41-1017-exl3-tp2.md)),
Apache-2.0, reshaped here like the rest of this project: `config.sh` knobs,
`./serve.sh`, the pair's addresses shared with tensorfold-dsv41, port 8888 and
the same model id.

Their numbers (bertholomus's `kit_bench.py`, greedy, one stream, their pair,
2026-10-18), against tensorfold-dsv41's with the same client on its pair; not
yet measured here:

| | This engine | tensorfold-dsv41 |
|---|---:|---:|
| Code, set a / b, tok/s | 88.3 / 93.8 | 84.7 / 95.1 |
| Prose, set a / b | 49.1 / 52.4 | **58.7 / 57.7** |
| Structured, set a / b | 105.9 / 126.8 | 103.8 / **140.6** |
| Context | 262,144 (KV 711,931 tokens at 0.85) | **1,048,576** |
| Requests at once | 2 | **4** |
| Images | on, **checked** (a colour test image) | on, not checked |
| Forced tool calls / JSON schema under spec decoding | not shown | checked (32-test suite) |
| Prefill | not published | ~2,000-2,200 tok/s to 128K |
| Boot | ~13 min | ~15 min first, 2-4 later |

Code is a tie; tensorfold-dsv41 is ahead on prose and structured output, holds
4x the context and twice the requests, and has its agent features tested. This
one is the A/B partner, and the one with images checked. One server at a time:
both hold both machines.

## What it takes

Everything tensorfold-dsv41 needs (both Sparks cabled over CX7 with static
addresses, key-based ssh from the head to the worker, Docker; its README, "What
it takes"), the two weight folders on both nodes (its README, "Weights": two `hf
download` commands or `./download.sh`), and room for the serving tree on the same
filesystem (hard links: no extra space).

## Run (on the head, in this directory)

```bash
./pull.sh          # the image on both nodes, digest pinned in image.lock (commit it)
./repack.sh        # the serving tree on both nodes, from MODEL_DIR + ENGRAM_DIR (seconds)
./serve.sh         # checks, rank 1 on the worker, rank 0 here, waits for the API (~13 min)
./status.sh        # both containers, free memory, the API
./stop.sh          # both; their logs go to logs/
```

Then `python3 ../bench/perf.py --levels 1 2` or any OpenAI client at
`http://<head>:8888/v1`, model `DeepSeek-V4.1-Flash-1017`.

`serve.sh` refuses next to an SGLang server, the vllm-b12x container or a
TensorFold rank; checks the image and the serving tree on both nodes; finds each
node's fabric (the IPv4 on `FAB_IF`, its RoCE device, and the RoCE v2 GID index
whose IPv4-mapped GID is that address: computed at every start, so a link flap
that moved the GID table does not hang NCCL); copies the drafter's Markov token
list to both caches; drops the page cache (`sudo -n`: vLLM sizes its pools to
"free", and GB10 does not reclaim page cache for CUDA); starts the worker's
headless rank, then rank 0, and waits.

## The serving tree (repack.sh)

The engine reads `quant_method: deepseek_v41_exl3`, a per-layer
`exl3_moe_plan.json` with a drafter section, and the Engram tables through the
checkpoint's own index. `repack.py` (styles01's `repack_ours.py`, adapted) builds
`SERVE_DIR` from the two folders: hard links to every shard and file, a union
index (the pack's 43 shards + the 4 Engram entries), the config and plan
rewrites. Two changes from the original, same result:

- it writes the rewritten files as new files instead of through the hard links:
  the original rewrote `ENGRAM_DIR`'s own `config.json` and
  `model.safetensors.index.json` (checked here on a mock: both sha256 changed),
  which tensorfold-dsv41 reads and `./download.sh verify` checks;
- it skips directories and dotfiles (the original stops on hf download's
  `.cache/` with `PermissionError`) and `chat_template.jinja`, which
  tensorfold-dsv41's `patch-template.py` adds to `MODEL_DIR` for its own engine
  (the pack ships none, so their tree has none).

`./repack.sh --check` checks both nodes' trees; run it again after either folder
changes.

## Knobs (config.sh, each overridable from the shell)

| Variable | Default | |
|---|---|---|
| `HEAD_IP`, `WORKER_IP`, `WORKER_SSH`, `FAB_IF` | from `../tensorfold-dsv41/config.local.sh` | the pair; `FAB_HCA`, `FAB_GID` force what serve.sh derives |
| `MODEL_DIR`, `ENGRAM_DIR` | tensorfold-dsv41's two folders | read only |
| `SERVE_DIR` | `/models/styles01/DeepSeek-V4.1-Flash-1017-serving` | the repacked tree, same filesystem as both |
| `IMAGE`, `IMAGE_TAG` | `ghcr.io/styles01/dsv41-1017-exl3-2x-dgx-sparks`, `v1` | v1 is the runbook's and one-button.sh's; the sparkrun recipe names v2 (`rm image.lock && IMAGE_TAG=v2 ./pull.sh`) |
| `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS` | 262144, 2, 2048 | |
| `GPU_UTIL`, `KV_BYTES` | 0.85, 2684354560 | KV pinned per rank at 2.5 GiB; `KV_BYTES=` sizes it from GPU_UTIL; a clean boot takes 0.88 |
| `DRAFT_TOKENS`, `DRAFT_TP` | 6, 2 | DSpark depth cap; the drafter split across both nodes (+20-30 tok/s at a fixed depth) |
| `DEPTH_POLICY`, `ROUND_MS` | `conf`, `31.9,35.2,38.6,42.6,45.3,48.1,51.0` | depth per round from the drafter's confidence against the cost of a round at depth 0..6 (ms): measured on their boots, re-measure on yours |
| `DEPTH_PROBE` | 0 | 1: boot for `./depth-sweep.py` |
| `VISION` | 1 | 0: text only, ~0.9 GiB a rank back to KV |
| `HOST`, `PORT`, `SERVED_MODEL_NAME`, `API_KEY` | 0.0.0.0, 8888, `DeepSeek-V4.1-Flash-1017`, none | |
| `CPUSET` | empty | `--cpuset-cpus` on both (`5-9,15-19`: the X925 cores); empty as measured |
| `DROP_CACHES` | 1 | needs `sudo -n` on both, else a note |
| `RECIPE_ENV`, `ENGRAM_CONFIG` | their launcher's | the tuned EXL3 / RDMA / L2-prefetch / Engram-reader knobs; `RECIPE_ENV=` runs without them |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `vllm serve` flags; more `NAME=value` for both, replacing one of the same name |

## Depth table

The adaptive depth policy is only as good as its cost table: with a table
measured on another setup it under-speculates (their structured set went from
~127 to ~106 tok/s). Their fixed-depth sweep shows why it matters: prose is best
at depth 2 (~50 tok/s; the 1017 drafter accepts ~2 tokens a round on flowing
text), structured at 5 (131.6), code still climbs at 6 (87.3), and the adaptive
policy beats every fixed depth on every set. To re-measure on your pair:

```bash
DEPTH_PROBE=1 ./serve.sh      # rank 0 reads its policy from CACHE_DIR/dspark_depth every 64 verify steps
python3 depth-sweep.py        # fixed:0..6 x code / prose / structured, then conf with the new table
./stop.sh && ROUND_MS=<printed> ./serve.sh   # or ROUND_MS="${ROUND_MS:-...}" in config.local.sh
```

`depth-sweep.py` times each streamed chunk as one verify round (ms per round,
tokens per round), as their `depth_sweep.sh` does with `kit_bench.py`.

## Files

| | |
|---|---|
| `config.sh`, `config.local.sh.example` | settings; the pair's values come from tensorfold-dsv41's `config.local.sh` |
| `pull.sh`, `image.lock` | the image, pinned, on both nodes |
| `repack.py`, `repack.sh` | the serving tree, built on both nodes |
| `serve.sh`, `stop.sh`, `status.sh` | run it |
| `depth-sweep.py` | the round-cost table for the depth policy |
| `data/markov_top_tokens.json` | the drafter's Markov bias token list (sudoingX's, counted on the pre-1017 base; theirs says regenerating it might buy 2-5% acceptance), with the source repo's LICENSE / NOTICE and commit |
