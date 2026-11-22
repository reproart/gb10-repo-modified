# Qwen3.8-Flash-Next on vLLM 0.34 (`vllm-flash-v034/`)

The same checkpoint as [`../vllm-flash-ar/`](../vllm-flash-ar/README.md) (Intel's
AutoRound int4 experts, an int8 lm_head, blockwise-fp8 side layers, the fp8 n-gram
table on NVMe) on stock **vLLM 0.34.0** (`vllm/vllm-openai:v0.34.0`, pinned by digest in
`image.lock`) instead of the August dev image that recipe is built on. A second engine
next to the first, not a replacement: the two share the weights, the port (8000, as
`vllm-flash-ar` runs here) and the model ids (`qwen3.8-flash-next`, `qwen`); one runs at
a time, and every `serve.sh` in this project refuses to start beside the other.

**Why.** CipherCloak's single-Spark Flash-Next recipe (MiaAI-Lab's checkpoint,
different weights) measured, moving from the same dev image to 0.34 with the same model
and settings: cold prefill 1,849 / 2,309 / 2,237 → 2,870 / 2,991 / 2,969 tok/s at 15K /
61K / 122K, per-turn latency of a 12-turn tool session halved (prefix-cache hits land
closer to the end of the prompt on this hybrid model), boot 662 → 241 s, decode equal or
better. Our recipe's prefill is ~1,800-2,150 tok/s on the dev image. Whether 0.34 does
the same for this checkpoint is what this folder is for.

## Status: stage 0 boots and measures (2026-11-22)

The first boot worked as built: `../bench/perf.py`, GX10, MTP 4, 8 sequences, BF16 KV,
GDN state float32, against `vllm-flash-ar` with this project's patches on (int8 draft
slice of the Russian vocabulary, FP8 hyper-connection projections), each one boot:

| | `vllm-flash-ar` (dev image, tuned) | vLLM 0.34, stage 0 (full draft head, BF16 HC) |
|---|---|---|
| TTFT, short prompt | 124-129 ms | 136 ms |
| Single-stream decode, code | **79.6 tok/s** (4.19 tok/step, 80%) | 68.8 (4.35, 84%) |
| Single-stream decode, Russian prose | **50.2** (2.58, 40%) | 43.1 (2.66, 41%) |
| Aggregate at 1 / 2 / 4 / 8 streams | 64.0 / 106.9 / 163.5 / 256.6 (2026-11-19, `ABLATE=1`) | 63.5 / 91.1 / 165.8 / 242.9 |
| Prefill, 2K / 8K / 32K / 101K | 1,838 / 2,009 / 1,810 / 1,913 (2026-11-19); its best 2,000 / 2,153 / 2,134 / 1,981 | **2,488 / 2,659 / 2,742 / 2,649** |
| TTFT of the 101K prompt | 52.7 s | **38.2 s** |
| Longest pause of running streams under a 32K prefill (1 / 4 streams) | 5.68 / 11.31 s | **2.46 / 2.49 s** |

Prefill +24-34% over the dev image's best (+32-51% over its last run), a long prompt
no longer stalls the streams already generating for more than ~2.5 s, and ~4 GiB more
memory left available during a long prefill (~10 GiB against ~4-6). Decode is behind
by what this lane does not have yet: the drafter scores the shared full-vocabulary
head (616 MiB read per draft pass, four a step, against the 160 MiB slice; its
acceptance is the full vocabulary's, 84% on code against 80% with the slice), and the
hyper-connection projections run BF16. Those two are stage 1.

## Stage 1a: the private draft head (built, to measure)

`vllm-flash-ar`'s patch 10, the same file (`../vllm-flash-ar/src/patch_mtp_draft_vocab.py`),
now finds the file's MTP class (`Qwen4ExpMTP` here) and the Marlin helpers' signatures
itself (0.34 dropped act-order: no permutation in `gptq_marlin_repack`, no `g_idx` /
`is_k_full` in `apply_gptq_marlin_linear`, three arguments to `marlin_quantize`); on the
dev image it makes the calls it always made. `DRAFT_VOCAB=ru` and `DRAFT_HEAD=int8` by
default, as in `vllm-flash-ar`: the drafter scores the int8 rows of the 65,536 English /
code / Russian ids on the same Marlin kernel (160 MiB per draft pass instead of 616),
every other id -inf; the target still verifies every token. The id sets are
`vllm-flash-ar`'s (`../vllm-flash-ar/src/draft_vocab_*.npy`), mounted read-only at
`/opt/llm/` by `serve.sh`; a path in `DRAFT_VOCAB` mounts your own.

0.34 calls the drafter's `compute_logits` first in the profiling run, eager, before it
captures the draft CUDA graphs, so the slice is built there (a few seconds) and every
graph replays it. The boot log should say:

```
MTP private draft head enabled: vocab=/opt/llm/draft_vocab_ru_65536.npy head=shared
MTP private draft head self-check: top-1 agreement ...
MTP private draft head: int8 GPTQ-Marlin (the target head's own rows), 65536 of 248320 rows, 616 -> 162 MiB per draft step, built in ...
```

and never `first called inside a CUDA graph capture` (that graph would keep the full head:
correct, only slower). Expected: decode back to `vllm-flash-ar`'s 79.6 / 50.2 tok/s
(code / Russian) or near it; what remains after that is the BF16 hyper-connection
projections (stage 1b).

### Stage 0, as built

What 0.34 does natively that the dev image needs patches for: the n-gram table read
from disk (`--engram-config '{"disk_offload": true}'`: the rows stay in the
safetensors files, each step's rows are read on the host into fixed device buffers, so
CUDA graphs replay), FP8 KV on the sparse-attention path, GB10 skinny-GEMM plans for
this model, `thinking_token_budget`, `disable_eagle_block_drop`.

What this checkpoint needs on top, all in this folder:

| | Why | Where |
|---|---|---|
| int4+fp8 hybrid dispatch | the GPTQ config's excluded layers are stored as blockwise FP8 (`weight` + `weight_scale_inv`, 128x128); stock `AutoGPTQConfig` builds them BF16 and cannot load them | `../vllm-flash-ar/src/vllm_fp8_hybrid.py`, unchanged (0.34's `AutoGPTQConfig` has the same three methods), appended to `auto_gptq.py` by the `Dockerfile`; `VLLM_FP8_HYBRID=1` |
| int8 lm_head | 0.34 builds Qwen4Exp's heads (target and MTP) without a `quant_config` | `src/patch_lm_head_quant.py` (`Dockerfile`), the same change as `vllm-flash-ar`'s `sed` |
| a lean table | 0.34 loads every tensor of every file it is given, and `vllm-flash-ar`'s table folder holds Qwen's FP8 shards whole (expert scales and dense weights of other layers beside the rows) | `./prepare.sh` once: `../vllm-flash-ar/tools/strip_ple_table.py` into `TABLE_DIR` (~48 GB, same file names) |
| one model folder | the rows must come through the model's own files | `src/make_model_view.py` (each `serve.sh`): a folder of links to the model's files and the lean table's, a merged index, and `config.json` with `ple_embedding_dtype: float8_e4m3fn` (with a GPTQ config 0.34 would take the table for BF16 and refuse the fp8 rows); no weights copied, the checkpoint untouched |
| the private draft head (stage 1a) | 0.34 shares or loads the full int8 head for the drafter: 616 MiB per draft pass | `../vllm-flash-ar/src/patch_mtp_draft_vocab.py` (`Dockerfile`), `DRAFT_VOCAB` / `DRAFT_HEAD` in `config.sh` |
| `--load-format safetensors` | the disk-offloaded table keeps the loader's mmap views of the shards; `fastsafetensors` (`vllm-flash-ar`'s loader) does not hand them over | `config.sh` |

The model's `config.json` already says `qwen4_exp` (0.34's name for Flash-Next), so no
rename is needed (MiaAI-Lab's checkpoint did need one).

Not ported yet, so off in this lane: FP8 hyper-connection projections (`HC_FP8`), the refusal-direction ablation (`ABLATE`), deterministic MoE
alignment and QSA top-k, the metrics sidecar, never-evict pinning, the tool-parser fixes
12-13, the mamba state-copy guard. Each comes over as its own stage once the bare lane
boots and measures; some may be upstream already (0.34 may not need them).

## Use

```bash
./build.sh            # base image (~10 GB, pinned) + the patches; "... OK" per patch
./prepare.sh          # once: the lean table (~48 GB of disk) and the view
./serve.sh            # same port and names as vllm-flash-ar; stop that one first
python3 ../bench/perf.py --only decode decode-ru prefill
python3 ../bench/consistency.py
./stop.sh
```

`config.sh` holds every setting with `vllm-flash-ar`'s values (MTP 4, 8 sequences, a
262K context, a 30g KV pool, 8192-token prefill chunks, prefix caching on, BF16 KV, the
Russian draft vocabulary; `DRAFT_VOCAB=0` is stage 0's full head).
`MAMBA_SSM_DTYPE` follows whatever `vllm-flash-ar` ends up with (`bfloat16` once its A/B
says so). Compare the two lanes with the same `perf.py` sections on the same day.

What the boot log should say: `fp8 hybrid: N blockwise-fp8 layers detected` and `fp8
hybrid patch applied to AutoGPTQConfig`, `Initialized PLE embedding ...
quantization_method=Qwen4ExpPLEFp8EmbeddingMethod`, and no `Falling back to unquantized
PLE embedding`.

## Files

```
config.sh                 settings (config.local.sh: this machine's, not committed)
image.lock                vllm/vllm-openai:v0.34.0's manifest digest (arm64 + amd64)
Dockerfile, build.sh      the image: the pinned base + the hybrid hook + the int8 heads + the draft head
prepare.sh                the lean table and the view, once
serve.sh / stop.sh        start (checks, view, memory, wait) / stop
src/make_model_view.py    the checkpoint view (standard library)
src/patch_lm_head_quant.py  the int8 lm_head patch
src/test_v034_cpu.py      CPU tests of both and of the Dockerfile's mtp.py steps (python3 src/test_v034_cpu.py)
../vllm-flash-ar/src/patch_mtp_draft_vocab.py, draft_vocab_*.npy   the draft head and its id sets (shared)
src/testdata/*.v034       vLLM 0.34.0's qwen4_exp/nvidia/model.py and mtp.py, from the wheel
```
