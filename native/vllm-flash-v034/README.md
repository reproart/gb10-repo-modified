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
vocabulary, FP8 hyper-connection projections), without and with `ABLATE=1` (the mean
direction, all 48 layers, the embedding, the MTP draft), one boot each, against
`vllm-flash-ar` and this lane's stage 0. The dev image's decode cells are without the
ablation (79.3 / 4.26 / 82% with it, patch 17), its sweep with it:

| | `vllm-flash-ar` (dev image) | 0.34, stage 0 | **0.34, stage 1** | 0.34, stage 1 + `ABLATE=1` |
|---|---|---|---|---|
| TTFT, short prompt | 124-129 ms | 136 ms | **119 ms** | 120 ms |
| Single-stream decode, code | 79.6 tok/s (4.19 tok/step, 80%) | 68.8 (4.35, 84%) | 79.3 (4.23, 81%) | **79.8** (4.30, 83%) |
| Single-stream decode, Russian prose | 50.2 (2.58, 40%) | 43.1 (2.66, 41%) | 49.8 (2.66, 42%) | 49.3 (2.57, 39%) |
| Aggregate at 1 / 2 / 4 / 8 streams | 64.0 / 106.9 / 163.5 / 256.6 | 63.5 / 91.1 / 165.8 / 242.9 | **74.3 / 113.7 / 170.4 / 264.5** | 70.4 / 110.4 / 169.8 / 254.4 |
| Prefill, 2K / 8K / 32K / 101K | best 2,000 / 2,153 / 2,134 / 1,981 | 2,488 / 2,659 / 2,742 / **2,649** | **2,520 / 2,832 / 2,753** / 2,637 | 2,319 / 2,505 / 2,602 / 2,556 |
| TTFT of the 101K prompt | 52.7 s | **38.2 s** | 38.3 s | 39.5 s |
| Longest pause under a 32K prefill (1 / 4 streams) | 5.68 / 11.31 s | 2.46 / 2.49 s | **2.49 / 2.49 s** | 2.53 / 2.56 s |

Without the ablation: decode level with the dev image (79.3 / 49.8 against 79.6 / 50.2),
the aggregate 3-16% above it at 1-8 streams (16-32 streams queue behind the 8-sequence cap:
247-258 tok/s), prefill 26-33% above its best run and level with stage 0 (so the FP8
projections' prefill path, each weight converted back to BF16 per call, costs nothing
measurable), and a long prompt stalls the running streams for ~2.5 s instead of 6-11.

The ablation costs prefill 3-12% (8K: 2,832 -> 2,505) and the sweep 0-5%; single-stream decode
moved within its noise (79.3 -> 79.8). By the bytes it should be less: each of the ~100
projections a forward is a few separate kernels (a matrix-vector product, an outer product, a
subtraction), each reading or writing the [T, 2560] output again, plus launches at decode.
One fused kernel per projection (read once, write once) would cut most of it; not done yet.

`../bench/consistency.py` on both: the tool-call section fails (fixed in stage 3a, below) and
the determinism section fails (stage 3b, to do).

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

## Stage 3a: the tool-call parser fixes (checked 2026-11-22)

`consistency.py` section 1 on the image with them: all six cases PASS (the fenced example and
the mid-line header, streamed and not, stay text with the reply after them; the control calls
parse), where stages 1-2 failed three.


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

## Stage 4: what is left from the dev image (2026-11-22)

Four items of `vllm-flash-ar` were not here yet. Checked against 0.34's source:

- **The metrics sidecar and the idle release** (its patch 13): ported. After
  `IDLE_RELEASE_SEC` (60) seconds without a request the engine gives torch's cached GPU
  blocks back and runs `malloc_trim(0)` once; on GB10 that is host memory, and stage 1's
  bench started with 2-4 GiB free. The allocator gauges are on `METRICS_PORT` (18400). The
  same `vllm_custom_metrics.py`, started from a hook on 0.34's `Scheduler.__init__`
  (`src/patch_scheduler_metrics.py`); the dev image's other sources (its PLE mmap counters,
  mamba guard, pin gauges) do not exist here and are skipped.
- **The mamba state-copy guard** (patch 7): not ported. 0.34's `v1/worker/mamba_utils.py`
  already has the race fix it backported (vllm#50729: the overlapping-copy single-CTA path,
  a snapshot of the accepted counts); what is missing is the block-id bounds check, which on
  the dev image counted 0 once its block-size fix was in. If a CUDA illegal-memory-access /
  Xid 31 shows up under load, that is the place.
- **Never-evict pinning** (patch 6): not ported. Off by default there too (`PIN_PROMPT`
  empty); it patches the block pool and the scheduler. On request.
- **A fused ablation projection**: not done. The ablation costs prefill 3-12% here (about a
  hundred projections a forward, each a few separate kernels); one Triton kernel per
  projection would cut most of it. On request.

## Stage 3c: the deterministic top-k kernel (measured 2026-11-22)

Built in the image as planned (`DET_TOPK kernel built OK`, CUDA 13.0's nvcc in the base image,
~9 s), and the boot log says `QSA top-k: deterministic persistent_topk`. `consistency.py
--only repeat`: cold vs cold and hit vs hit the same text, first-token gap 0.0000, at ~4K and
~32K (cold vs hit: the late-fork warning, first token equal). Prefill 2,577 / 2,858 / 2,721 /
2,636 tok/s at 2K / 8K / 32K / 101K, against 2,520 / 2,832 / 2,753 / 2,637 with the stock
non-deterministic kernel: determinism at no cost. With stage 3a's parser fixes,
`consistency.py` now passes all of its checks on this lane, as on the dev image.


The A/B settled it (2026-11-22, `consistency.py`, `DET_MOE_ALIGN=1`): with `QSA_EXACT_TOPK=1`
cold vs cold and hit vs hit gave the same text with a first-token top-5 gap of 0.0000 at ~4K
and ~32K tokens, where the stock `persistent_topk` forked from the first token. So 0.34's build
of that kernel is still non-deterministic on GB10, and the exact path, which on the dev image
cost 20-40% of long prefill, is what holds it. Cold vs a cache hit still drifts late (the
first token agrees, the reply forks 17-95 characters in): a hit computes fewer tokens, in
other chunk shapes; the bench counts that as a warning, not a failure.

The fast way is @jschmied's deterministic kernel (vllm#55122), the dev image's `DET_TOPK`:
the Dockerfile fetches the same five sources at the same commit with the same sha256 checksums
and builds `_C_det.so` with the base image's nvcc in a layer of its own;
`src/patch_qsa_topk.py` (the stage 3b patch, renamed) swaps `torch.ops._C_det.persistent_topk`
into 0.34's `_topk` with `QSA_DET_TOPK=1`, now the default. It registers the same arguments in
the same order as the stock call (logits, lengths, output, workspace, k, max_seq_len); it is
built for k = 512, 1024 and 2048 (this model's block top-k is 512; another k takes the exact
path); its workspace need is a few hundred KB, within the 1 MiB 0.34 passes. If the kernel
cannot be built (no nvcc in the base image, a compile error), the image still builds and the
server logs `the deterministic kernel did not load` and takes the exact top-k: asking for
determinism never gets the stock kernel. `serve.sh` prints the boot log's `QSA top-k:` line.
`src/test_qsa_topk_cpu.py` runs 0.34's patched `_topk` with stand-in ops: the stock kernel when
nothing is set, the deterministic op with the stock call's arguments, the exact path when asked
for or for an unsupported k, and the fallback when the library does not load.

## Stage 3b: determinism (MoE alignment; the top-k found)

Measured with the image of stage 3b (2026-11-22, `DET_MOE_ALIGN=1`, `QSA_EXACT_TOPK=0`): still
not deterministic. `consistency.py` section 2: cold vs cold and hit vs hit fork 15-78
characters in, with first-token top-5 gaps 0.10-0.54 nats at ~4K and ~32K tokens, and at 32K a
cache hit gives another first token than the cold prefill. The first token comes from the
prefill alone, and the MoE alignment is now fixed there (0.34's Marlin MoE calls the patched
`moe_align_block_size`; its grouped GEMM reduces in order, FP32, no atomics; the dense
Marlin's atomic reduction is off by default), so the next suspect is the QSA top-k kernel:
`QSA_EXACT_TOPK=1` is the A/B. If that settles it, @jschmied's deterministic kernel is the fast
way: it registers `_C_det.persistent_topk(logits, lengths, output, workspace, k,
max_seq_len)`, the same arguments in the same order as 0.34's `_topk` passes to
`persistent_topk`, so its wiring is a swap of the op.

What was built:

`../bench/consistency.py` section 2 sends one greedy request cold, twice from the prefix
cache and cold again; on stages 1-2 the replies forked a few dozen characters in (top-5
first-token gaps 0.2-0.5 nats), cold vs cold and hit vs hit alike, as on the dev image
before its patches 15 and 9. Two possible sources, both ported:

- **MoE token alignment** (`DET_MOE_ALIGN=1`, the default): vLLM's `moe_align_block_size`
  kernel places an expert's routes with `atomicAdd`, so a route lands in another Marlin MoE
  m-block from run to run and the expert's sums change order. On the dev image this was the
  whole difference (gap 0.0000 after it, no measurable cost). `../vllm-flash-ar`'s patch,
  which now finds 0.34's implementation behind its three `@overload` stubs and keeps the
  kernel when a caller asks for 0.34's new `scatter_idx` (Marlin MoE does not). 0.34 runs
  int4 experts on Marlin on GB10 (Humming is preferred only on SM 9.0). From `SEQS x (1 +
  MTP) + 1` tokens a step (41): prefill and steps with a prefill chunk, decode keeps the
  kernel.
- **QSA block top-k** (`QSA_EXACT_TOPK=1`; stage 3c makes the fast kernel the default): on GB10 0.34 always selects the sparse
  attention's key blocks with `persistent_topk` (`cooperative_topk` is gated off for SM 12.x).
  That kernel was non-deterministic in the dev image's vLLM (vllm#51782); whether 0.34's build
  has @jschmied's fix (vllm#55122) cannot be seen from its Python. `src/patch_qsa_topk.py`
  is the dev image's exact fallback ported to 0.34's `_topk`: `torch.topk` over the visible
  columns, the rest masked. One change: the chosen block ids are put in ascending order,
  because 0.34's expansion kernel reads only the first `min(visible, block_topk)` slots, and
  `torch.topk(sorted=False)` does not promise the visible ones come first (the masked ids are
  the highest, so they sort last; it is also the order @jschmied's kernel writes). Off by
  default: on the dev image it cost 20-40% of long prefill, and 0.34's kernel may not need it.

The plan: `consistency.py --only repeat` with the defaults; if it still forks, again with
`QSA_EXACT_TOPK=1`. If the exact top-k is what it takes, @jschmied's deterministic kernel (the
dev image's `DET_TOPK`) is the fast way and comes next. Checks: `../vllm-flash-ar/src/
test_det_moe_align_cpu.py` (0.34's own file, a stand-in kernel: the reference layout with no
kernel call for a prefill batch; decode, `scatter_idx` and an `expert_map` keep the kernel) and
`src/test_qsa_topk_cpu.py` (the reference top-k in the slots the expansion reads,
garbage past the visible count kept out, repeatable).

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
| determinism (stages 3b, 3c) | the same greedy request forks from run to run: the QSA top-k kernel (measured), MoE token alignment by `atomicAdd` | `src/patch_qsa_topk.py` + @jschmied's kernel built in the `Dockerfile` (`QSA_DET_TOPK=1`; `QSA_EXACT_TOPK=1` the exact fallback), `../vllm-flash-ar/src/patch_det_moe_align.py` (`DET_MOE_ALIGN=1`) |
| `--load-format safetensors` | the disk-offloaded table keeps the loader's mmap views of the shards; `fastsafetensors` (`vllm-flash-ar`'s loader) does not hand them over | `config.sh` |

The model's `config.json` already says `qwen4_exp` (0.34's name for Flash-Next), so no
rename is needed (MiaAI-Lab's checkpoint did need one).

Not ported (stage 4 has why): never-evict pinning, the mamba state-copy guard's bounds check.

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
src/patch_scheduler_metrics.py  starts ../vllm-flash-ar's vllm_custom_metrics (allocator gauges, idle release); test_scheduler_metrics_cpu.py
src/patch_qsa_topk.py     the QSA block top-k: @jschmied's kernel (QSA_DET_TOPK=1) or exact (QSA_EXACT_TOPK=1); test_qsa_topk_cpu.py
../vllm-flash-ar/src/patch_det_moe_align.py  the deterministic MoE alignment (DET_MOE_ALIGN=1, shared)
../vllm-flash-ar/src/patch_mtp_draft_vocab.py, draft_vocab_*.npy   the draft head and its id sets (shared)
../vllm-flash-ar/src/gb10_fp8_rowwise.py, patch_hc_fp8.py          FP8 HC (shared; test_hc_fp8_cpu.py runs 0.34's file)
../vllm-flash-ar/src/gb10_abliterate.py, patch_abliterate.py       the ablation (shared; test_abliterate_cpu.py runs 0.34's classes)
src/testdata/*.v034       vLLM 0.34.0's qwen4_exp/nvidia/model.py, mtp.py, hyperconnection.py, from the wheel
src/testdata/parser/      vLLM 0.34.0's parser files the patches change, from the wheel
src/testdata/{moe_align_block_size,qsa_indexer}.py.v034  the files the determinism patches change
```
