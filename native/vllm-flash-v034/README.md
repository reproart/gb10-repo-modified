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

## Status: stages 1 and 2 on, the full bench (2026-11-22)

`../bench/perf.py` with this folder's defaults (the int8 draft slice of the Russian
vocabulary, FP8 hyper-connection projections) and `ABLATE=1` (the mean direction, all 48
layers, the embedding, the MTP draft), one boot, against `vllm-flash-ar` and this lane's
stage 0. The dev image's decode cells are without the ablation (79.3 / 4.26 / 82% with it,
patch 17), its sweep with it:

| | `vllm-flash-ar` (dev image) | 0.34, stage 0 | **0.34, stage 1 + `ABLATE=1`** |
|---|---|---|---|
| TTFT, short prompt | 124-129 ms | 136 ms | **120 ms** |
| Single-stream decode, code | 79.6 tok/s (4.19 tok/step, 80%) | 68.8 (4.35, 84%) | **79.8** (4.30, 83%) |
| Single-stream decode, Russian prose | 50.2 (2.58, 40%) | 43.1 (2.66, 41%) | 49.3 (2.57, 39%) |
| Aggregate at 1 / 2 / 4 / 8 streams | 64.0 / 106.9 / 163.5 / 256.6 | 63.5 / 91.1 / 165.8 / 242.9 | **70.4 / 110.4 / 169.8** / 254.4 |
| Prefill, 2K / 8K / 32K / 101K | best 2,000 / 2,153 / 2,134 / 1,981 | **2,488 / 2,659 / 2,742 / 2,649** | 2,319 / 2,505 / 2,602 / 2,556 |
| TTFT of the 101K prompt | 52.7 s | **38.2 s** | 39.5 s |
| Longest pause under a 32K prefill (1 / 4 streams) | 5.68 / 11.31 s | **2.46 / 2.49 s** | 2.53 / 2.56 s |

Decode is level with the dev image (code 79.8 against 79.6, Russian within its acceptance
noise) and with this lane's own stage 1b run without the ablation (78.5), so the ablation's
projections cost decode nothing measurable; the aggregate is 3-10% above the dev image's
(ablated too) at 1-4 streams and level at 8 (16-32 streams queue behind the 8-sequence cap:
240-244 tok/s), prefill 16-29% above its best run, and a long prompt still stalls the running
streams for ~2.5 s instead of 6-11.

Prefill is 3.5-7% below stage 0's run. Two things are added there, both small by the bytes:
the ablation's projections (two per layer over [8192, 2560] chunks plus the PLE layers', ~2% of
a chunk if each reads its input three times) and the FP8 projections' prefill path (each
weight converted back to BF16 per call, ~20 MB a mix, <1%); the rest is within this machine's
run-to-run spread (the dev image's prefill runs spread 5-15%). A plain `./serve.sh` (no
ablation) with `--only prefill` separates them. Still to run here: `../bench/consistency.py`
without the ablation, and a request the stock model refuses with it.

## Stage 0 (2026-11-22)

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

## Stage 1a: the private draft head (measured 2026-11-22)

`../bench/perf.py --only decode decode-ru`, same machine and settings as stage 0, one boot
(median of 5 code answers, of 3 Russian ones):

| | `vllm-flash-ar` (dev image, tuned) | 0.34, stage 0 (full draft head) | 0.34, stage 1a (`DRAFT_VOCAB=ru`, int8 slice) |
|---|---|---|---|
| Single-stream decode, code | **79.6 tok/s** (4.19 tok/step, 80%) | 68.8 (4.35, 84%) | 76.0 (4.22, 81%) |
| Single-stream decode, Russian prose | 50.2 (2.58, 40%) | 43.1 (2.66, 41%) | **52.2** (2.80, 45%) |

Code +10% and Russian +21% over stage 0. Russian is now ahead of the dev image (+4%, three
answers), code 4.5% behind it at the same acceptance, so the rest of that gap is time per
step, not drafting; the largest known difference left is the BF16 hyper-connection
projections (stage 1b). The boot log, as expected:
`self-check: top-1 agreement 1.000, rel err 0.0000 (T=8)`, `65536 of 248320 rows, 616 ->
162 MiB per draft step, built in 10.1s`, in the profiling run, 2 min 25 s after the
engine started; no capture warning.

### How it works

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
correct, only slower). `serve.sh` prints the third line itself once the API answers (or
warns that it is missing), and refuses a `DRAFT_VOCAB` / `DRAFT_HEAD` on an image built
before this stage (no `gb10.draft-head` label: `./build.sh`).

## Stage 1b: FP8 hyper-connection projections (measured 2026-11-22)

`../bench/perf.py --only decode decode-ru`, one boot, as for 1a:

| | `vllm-flash-ar` | 0.34, stage 1a | 0.34, stage 1b (`HC_FP8=1`) | `HC_FP8=up` |
|---|---|---|---|---|
| Single-stream decode, code | 79.6 tok/s (4.19 tok/step, 80%) | 76.0 (4.22, 81%) | **78.5** (4.17, 79%) | 75.8 (4.12, 78%) |
| Single-stream decode, Russian prose | 50.2 (2.58, 40%) | 52.2 (2.80, 45%) | 50.8 (2.70, 43%) | 51.0 (2.66, 42%) |
| Time per decode step, code (median of the 5) | 52.6 ms | 55.6 ms | 53.7 ms | 54.4 ms |

Code +3.3%, the step 1.9 ms shorter (the estimate was ~3 ms: the fused BF16 down+SiLU
kernel it gives up was already lean). Russian moved with its acceptance (43% against
45%, three answers of 281-390 tokens; per step 53.6 -> 53.1 ms), i.e. within the noise.
Both are now level with `vllm-flash-ar` (79.6 / 50.2). The boot log: `200
hyper-connection projections in FP8 with per-row scales, 1.25 -> 0.64 GiB; inputs up to
64 rows: decode kernel for 16/32/64-row tiles; prefill dequantize on Triton; fused BF16
kernels off in 98 modules`.

`HC_FP8=up` (only the 100 up projections FP8, 0.61 -> 0.31 GiB; the fused BF16 down+SiLU
kernel kept): the step 54.4 ms, between the two. So each half pays: FP8 up against cuBLAS
BF16 ~1.2 ms a step, the FP8 down GEMM + a SiLU kernel against the fused BF16 one ~0.7 ms.
`1` stays the default. To do: prefill and the concurrency sweep with it on, `consistency.py`.

### How it works

`vllm-flash-ar`'s patch 16, the same files (`../vllm-flash-ar/src/gb10_fp8_rowwise.py`,
`patch_hc_fp8.py`), `HC_FP8=1` by default as there. Each of the ~98 mixes a step reads
two BF16 projections, down+inject [336 x 10240] (6.9 MB) and up [10240 x 320] (6.6 MB),
~1.3 GB a target forward; in FP8 with a scale per row they are half that (the 16 inject
and padding rows stay BF16). On the dev image that was 69.7 -> 74.5 tok/s.

What 0.34 does with them on GB10 now (read from the wheel): the up projection is
`F.linear` (cuBLAS) for the verify step's 5 rows (the model's CuTe skinny-GEMM plan for
[10240 x 320] covers 1, 2, 4 and 8 rows: the drafter's single row uses it), followed by
the gate-mix kernel; the down+inject projection is a fused CuTe-DSL down+SiLU GEMM for
up to 48 rows. Both read BF16. A converted layer gives up the fused path (the module's
flag goes off and it runs its unfused one: the Linear, i.e. the FP8 GEMM, then the split
and the SiLU kernel), so the trade is half the bytes against one more small kernel a
mix. Estimate: ~0.65 GB less read a forward, ~3 ms of the 55.5 ms step at stage 1a, so
~+5% decode, about where `vllm-flash-ar` is. `HC_FP8=up` converts only the up
projections and keeps the fused down kernel: the fallback if `1` measures worse.

`serve.sh` prints the first-forward line once the API answers:

```
hc fp8: <N> hyper-connection projections in FP8 with per-row scales, <~1.3> -> <~0.65> GiB; inputs up to 64 rows: decode kernel for 16/32/64-row tiles; prefill dequantize on Triton; fused BF16 kernels off in <M> modules
```

(two projections a mix; the fused kernel is on in every combining mix), warns if it is missing or if a kernel fell back to
torch (`TORCH PATH`: slower than BF16), and refuses `HC_FP8` on an image without the
hook (`gb10.hc-fp8` label: `./build.sh`).

## Stage 2: the refusal-direction ablation (boots and measures, 2026-11-22)

`ABLATE=1 ./serve.sh` booted as built; `serve.sh`'s report:

```
ablation: refusal direction the mean of layers 4-46's vectors from refusal_directions-v4-late.npz (sha256 bb311e37712e...), rank 1, alpha 1; covers 37%-89% (mean 71%) of the file's per-layer directions; projected out of the writers of 48 of 48 layers (0-47), the embedding. Answers are not the stock model's.
ablation: the MTP draft projects its inputs too
```

and the full bench with it is the status table above: decode 79.8 / 49.3 tok/s, 83% of code
drafts accepted (the draft projects its inputs as the target does, so acceptance holds).
To confirm by hand: a request the stock model refuses.

### How it is wired

`vllm-flash-ar`'s patch 17, the same files (`../vllm-flash-ar/src/gb10_abliterate.py`,
`patch_abliterate.py`) and the same knobs (`ABLATE`, `ABLATE_VECTORS`, `ABLATE_SOURCE`,
`ABLATE_ALPHA`, `ABLATE_LAYERS`, `ABLATE_MTP`; off by default):

```bash
ABLATE=1 ./serve.sh        # Mambavtt's vectors at ABLATE_VECTORS (hf download, see config.sh)
```

The direction is projected out of every writer of the four residual streams: the token
embedding, the PLE addition, each attention / GDN and MLP output, the deepstack embeddings,
and the MTP draft's inputs. 0.34's insertion points differ from the dev image's in two
places: its PLE layer returns the streams with the addition already in them (the residual
add is fused into its short-convolution kernel), so the addition is taken back out
(`ple_out - hidden_states`), projected and added back (the streams' component along the
direction is kept exactly; the addition is rounded to BF16 once more); and its draft has two
writers, `fc_hidden`'s output (the draft layer's streams) and `fc_embedding`'s (a pending block
output added to every stream with unit weight), both projected.
`../vllm-flash-ar/src/test_abliterate_cpu.py` runs 0.34's own decoder layer, model and draft
on CPU stand-ins: every stream of the final state and both draft inputs lose the direction,
a layer left out keeps it, without `VLLM_ABLATE` the outputs equal the unpatched files' bit
for bit, the patched forward compiles as one graph, and dropping the PLE, attention or MLP
insertion fails it.

`serve.sh` refuses `ABLATE` without the vectors or on an image without the hook
(`gb10.ablate` label), and prints the boot log's lines (above). On the dev image the same
vectors answered a request the stock model refused, decode 79.3 tok/s (82% of drafts
accepted).

## Stage 3a: the tool-call parser fixes (built, to check)

`../bench/consistency.py` on stages 1 and 2 failed its tool-call section as the dev image
did before its patch 14: a reply that shows the tool-call syntax as an example (a call
inside a ```` ```xml ```` fence, or `<function=NAME>` in the middle of a sentence) came back as
a call, and the rest of the reply was dropped. vLLM 0.34's parser engine is the same one,
moved on: blazux's two patches (12, Jordi Posthumus: a tool wrapper is held until its
whitespace-only preamble reaches a function header; 13, @techfury90: a marker inside a fenced
code block, or a wrapper-less `<function=` header that does not start a line, stays text)
no longer apply (4 of 7 and 5 of 5 hunks off), so they are ported by hand to 0.34's lines:
`src/patches/qwen-tool-preamble.v034.patch`, `qwen-tool-marker-guard.v034.patch` (Apache-2.0,
with their attribution). The changes are the same; 0.34's new `skip_reasoning_parsing` bypass
runs before the deferral, and nemotron_v3's config, now a `dataclasses.replace` of qwen3's,
keeps both guards off. Both are on for the qwen3 configuration that
`--tool-call-parser qwen3_xml` runs.

Checked against vLLM 0.34.0's own parser tests (`tests/parser/engine` at the v0.34.0 tag)
plus blazux's for these patches, on CPU: the stock 0.34 parser passes vLLM's 173 and fails 45
of blazux's 51; the patched one passes all 224. The Dockerfile applies both with `--fuzz=0`
and runs `src/check_tool_parser.py` (a fenced example and a mid-line header stay text, a real
call parses, a quoted `<tool_call>` in the reasoning keeps the answer; chunks of 1, 7 and
all characters), so a build that does not hold stops there. To check on the server:
`python3 ../bench/consistency.py --only fence`.

## Stage 0, as built

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
| FP8 hyper-connection projections (stage 1b) | the checkpoint's BF16 HC projections, ~1.3 GB read a forward | `../vllm-flash-ar/src/gb10_fp8_rowwise.py` + `patch_hc_fp8.py` (`Dockerfile`), `HC_FP8` in `config.sh` |
| the refusal-direction ablation (stage 2) | optional, `ABLATE=1`: a direction projected out of every writer of the streams | `../vllm-flash-ar/src/gb10_abliterate.py` + `patch_abliterate.py` (`Dockerfile`), `ABLATE*` in `config.sh` |
| the tool-call parser fixes (stage 3a) | 0.34's parser turns a quoted or fenced tool-call example into a call that drops the reply | `src/patches/*.v034.patch` (blazux 12 + 13 ported, `Dockerfile`), checked by `src/check_tool_parser.py` at build time |
| `--load-format safetensors` | the disk-offloaded table keeps the loader's mmap views of the shards; `fastsafetensors` (`vllm-flash-ar`'s loader) does not hand them over | `config.sh` |

The model's `config.json` already says `qwen4_exp` (0.34's name for Flash-Next), so no
rename is needed (MiaAI-Lab's checkpoint did need one).

Not ported yet, so off in this lane: deterministic MoE alignment and QSA top-k, the metrics
sidecar, never-evict pinning, the mamba state-copy guard. Each comes over as its own stage
once the bare lane boots and measures; some may be upstream already (0.34 may not need
them).

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
src/test_v034_cpu.py      CPU tests of both, of the Dockerfile's mtp.py / hyperconnection.py / ablation steps and of the parser patches
src/patches/*.v034.patch  blazux's tool-call parser patches 12 and 13, ported to vLLM 0.34.0
src/check_tool_parser.py  the parser cases the build runs after them
../vllm-flash-ar/src/patch_mtp_draft_vocab.py, draft_vocab_*.npy   the draft head and its id sets (shared)
../vllm-flash-ar/src/gb10_fp8_rowwise.py, patch_hc_fp8.py          FP8 HC (shared; test_hc_fp8_cpu.py runs 0.34's file)
../vllm-flash-ar/src/gb10_abliterate.py, patch_abliterate.py       the ablation (shared; test_abliterate_cpu.py runs 0.34's classes)
src/testdata/*.v034       vLLM 0.34.0's qwen4_exp/nvidia/model.py, mtp.py, hyperconnection.py, from the wheel
src/testdata/parser/      vLLM 0.34.0's parser files the patches change, from the wheel
```
