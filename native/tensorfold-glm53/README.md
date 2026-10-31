# GLM-5.3-Flash on two DGX Spark: TensorFold, EXL3 4 bpw

GLM-5.3-Flash (321B MoE) with its routed experts in EXL3 4 bpw and the rest in
BF16, on TensorFold 0.6.0 with tensor parallelism across both Sparks over the
CX7 cable, DFlash2 drafts, a 1M-token context, images. This directory runs
[MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(Apache-2.0; 123 patches on TensorFold, her published image pinned by digest)
from a checkout pinned at `KIT_COMMIT`, with this project's pair, paths and
port, and the one-server-at-a-time checks. Her scripts do the work; nothing of
hers is copied here.

Why this one: of the published 2-Spark setups it is the fastest measured.
LimeChain's matched bake-off (2026-09-30 / 10-03, same suite and hardware,
[LimeChain/glm-5.3-flash-2xdgx-spark](https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark)):

| | TensorFold, Mia's recipe, 8 streams | Mia's recipe, 4 streams | vLLM TP2, NVFP4 (LibertAIDAI) |
|---|---:|---:|---:|
| 1 request, short (tok/s) | 114 | 116 | 37 |
| 1 request, prose / code, thinking off | 50 / 70 | 50 / 70 | 23 / 29 |
| 8 requests, short, aggregate | **528** | 300 (4) | 185 |
| one 128K+ prompt, prefill (tok/s) | 1,729 | 1,711 | 1,518 |
| lowest MemAvailable head / worker (GiB) | 7.3 / 7.6 | 10.7 / 11.1 | 4.5 / 6.5 |
| context per request | 1,048,576 | 1,048,576 | 262,144 |

Mia's own (two Sparks, GPU clocks capped at 2,200 MHz): one request 60.4 tok/s
prose, 114.7 structured; prefill 1,952 / 1,943 / 1,838 / 1,642 tok/s at 8K /
64K / 131K / 262K; GSM8K 98.8% and HumanEval 95.7% (thinking off). The SGLang
path for this model on two Sparks reports 29.4 tok/s for one request
(NVIDIA forum); `../llamacpp/` runs it as a GGUF. None of this is measured
here yet.

What LimeChain changed on Mia's 2026-10-02 commit is in her current one: up
to 8 requests (patch 0069) and `<|assistant|>` as an end token (0075, a fake
next turn otherwise). Their third change, effort High by default (tool
harness 40/42 against 36/42 at Mia's Max), is `EFFORT=high` here. Their
OpenAI `logprobs` patch is not carried over (it targets the older tree).

## Licences

- Recipe and TensorFold 0.6.0: Apache-2.0.
- The DFlash2 drafter, `incoai/GLM-5.3-Flash-DFlash2`: **CC BY-NC-ND 4.0,
  non-commercial use only**. `DRAFTER=mtp` uses the checkpoint's own MTP head
  instead: slower and one request at a time.
- The checkpoint: see its model card (`Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`).

## Needs

- Two Sparks on the CX7 cable with key-based ssh from the head, as for
  `../tensorfold-dsv41/` (its `config.local.sh` holds the pair's addresses and
  is read here too).
- **~205 GB of disk on each** (checkpoint ~176 GB, DFlash2 ~2.3 GB, image
  ~25 GB); Mia's `prepare.sh` refuses below 180 GB free.
- **~110 GiB of free memory on each** at start: nothing else on either GPU
  (`serve.sh` refuses next to SGLang, `vllm-b12x`, `vllm-dsv41` or
  `tensorfold-dsv41`).
- `../scripts/host-tuning.sh` applied (earlyoom stops the engine before the
  node runs dry; no swap): `serve.sh` notes it when it is not.

## Run

```bash
cd native/tensorfold-glm53
./setup.sh          # once: Mia's recipe at KIT_COMMIT, her prepare.sh: image on both, weights on the head, rsync to the worker
./serve.sh          # rank 1 on the worker, rank 0 here, API on :8888; first start ~10 min, later under a minute
./status.sh         # both containers, memory left, /health
./stop.sh

PARALLEL=8 ./serve.sh restart              # LimeChain's production setting (config.sh: what it costs)
CONTEXT=524288 KV=bf16 ./serve.sh restart  # any of Mia's settings (her scripts/config.sh) passes through
python3 ../bench/perf.py --levels 1 2 4 8
```

## Weights

`HF_CACHE` (`/models/huggingface`) is a Hugging Face cache folder, the same
path on both nodes. `./setup.sh` fills it on the head at the pinned revisions
and copies it to the worker. To fetch it yourself, or to carry it to another
machine:

```bash
HF_HOME=/models/huggingface hf download Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold \
  --revision 078455ffe6472f9a52fbc1139f58b9db2881b25c
HF_HOME=/models/huggingface hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a
rsync -a /models/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold \
  /models/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2  other-spark:/models/huggingface/hub/
```

(`-a` keeps the cache's snapshot symlinks.) The revisions are the ones Mia's
`scripts/config.sh` pins at `KIT_COMMIT`.

## Settings

`config.sh`: `KIT_COMMIT` / `KIT_DIR` (the recipe), `HF_CACHE`, `HOST` /
`PORT`, `PARALLEL` (4), `DRAFTER` (dflash2), `EFFORT` (high), `MEM_PREP`
(page cache dropped and free memory compacted on both nodes before a start,
`../scripts/mem-prep.sh`; passwordless sudo for it: `../scripts/mem-prep.sh
sudoers <worker>`). `setup.sh` refuses while anything runs on either GPU.
Everything else is Mia's and passes through from the shell; her
`scripts/config.sh` documents each with its measured effect.

## Known limits (Mia's and LimeChain's notes)

- Eight long fresh prompts arriving together prefill mostly in turn (8 × 23K:
  last first token after 134 s); raise client timeouts.
- Long multi-agent sessions lower MemAvailable over time (Mia's issue #197;
  `TF_GLM_KEPT_BYTES_GIB` is the lever, a restart the full reset).
- Changing the CX7 addresses while it runs can hang the ranks (issue #196):
  static addresses only.
- No `logprobs`, no `n > 1`; send a `seed` per best-of-N candidate.
