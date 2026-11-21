# Inkling-2-Mini on vLLM (vLLM's container)

The vLLM twin of [`../sglang-inkling/`](../sglang-inkling/README.md): the
same `thinkingmachines/Inkling-2-Mini` weights (26B looped MoE, text and
images, bundled DSpark draft, ~18.2 GiB), the same port 8888 and model id
`inkling-2-mini`, so `../bench/` runs against the two compare unchanged.
This project measured SGLang ahead of vLLM on the 27B; this pair asks the
same question on a model both engines support from day one.

Support needs vLLM >= 0.35.0, not released on 2026-11-21 (nightly only).
The vLLM recipe names `vllm/vllm-openai:inkling-2-mini`, so this directory
runs that image, pinned by digest in `image.lock`, with the recipe's
`dgx_spark_gb10` command as knobs. Once a release carries the model, a
native install or a newer pinned image replaces it.

```bash
# The weights, once for both directories (skip if ../sglang-inkling/ has them):
../scripts/mem-prep.sh idle && hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --local-dir /models/thinkingmachines/Inkling-2-Mini
./pull.sh       # the container (~10.5 GB compressed), the digest in image.lock
./serve.sh      # foreground, Ctrl-C stops; http://<host>:8888/v1, "inkling-2-mini"
```

One server at a time: `serve.sh` refuses next to the SGLang service, the
SGLang twin or any other model container, and they refuse next to it.

## The recipe

[recipes.vllm.ai](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Mini?hardware=dgx_spark_gb10&variant=default&strategy=single_node_tp&nodes=1&features=reasoning%2Ctool_calling%2Cspec_decoding),
hardware `dgx_spark_gb10`, features reasoning, tool_calling, spec_decoding:

```bash
vllm serve thinkingmachines/Inkling-2-Mini \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser inkling \
  --enable-auto-tool-choice --tool-call-parser inkling \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

What this directory changes, and why:

- **`--gpu-memory-utilization 0.8`** where the recipe leaves vLLM's default
  0.9 (~108 GiB of a ~120 GiB Spark, ~12 left for the OS). `../vllm-b12x/`
  runs 0.8 on a Spark and this project has seen machines wedge at high
  fractions on unified memory. `GPU_UTIL=0.9` is the recipe exactly.
- **Tool calling on** (the recipe's opt-in `tool_calling` feature), as in
  the SGLang twin. `TOOL_CALL_PARSER=` turns it off.
- **`--served-model-name inkling-2-mini`, port 8888**, as every server here.
- **No `--privileged`**: the seccomp / AppArmor options `../vllm-b12x/` runs
  vLLM with instead.

## Knobs

`config.sh`, each overridable from the shell:

| Variable | Default | |
|---|---|---|
| `SPEC_TOKENS` | 7 | DSpark proposals per step; 0 = plain decode |
| `DRAFT_SAMPLE_METHOD` | probabilistic | the draft's sampling; empty = vLLM's default |
| `MAX_MODEL_LEN` | 262144 | the recipe's Spark cap (1048576 natively) |
| `GPU_UTIL` | 0.8 | `--gpu-memory-utilization` (the recipe: vLLM's 0.9) |
| `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS` | vLLM's | concurrent requests, prefill chunk |
| `REASONING_PARSER`, `TOOL_CALL_PARSER` | inkling, inkling | empty = off |
| `PORT`, `HOST`, `API_KEY` | 8888, 0.0.0.0, none | as for `../serve.sh` |
| `MIN_FREE_GIB` | GPU_UTIL of the pool + 2 | refuse to start below this MemAvailable |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `vllm serve` flags; more `NAME=value` for the container |

## The two engines' recipes side by side

| | SGLang (`../sglang-inkling/`) | vLLM (here) |
|---|---|---|
| Image | `lmsysorg/sglang:inkling-2-mini`, 2026-11-20 20:16 | `vllm/vllm-openai:inkling-2-mini`, 2026-11-20 21:36 |
| Draft | DSpark, 8 draft tokens verified per step | DSpark, 7 proposals, probabilistic draft sampling |
| Memory | `--mem-fraction-static 0.45` (verified) | 0.8 here (recipe: vLLM default 0.9) |
| Context | model default | 262144 (recipe's Spark cap) |
| Parsers | `inkling` / `inkling2` | `inkling` / `inkling` |
| Published on Spark | gsm8k 93.56% / 93.93% | "verified on DGX Spark", no numbers |

SGLang's 8 is the verify width (7 proposals plus the target's own token),
so the two drafts likely do the same work per step; that is a reading of the
flags, not something either recipe states. The recipes' active-parameter
counts differ (SGLang 5.4B, vLLM 3.5B) because they count different things:
the model card says 26.4B total, 3.4B active, and each extra loop pass adds
about 1.0B effective, so 5.4B effective per token at three passes.

For a fair A/B, run both at the same memory share and context, and compare
`../bench/perf.py` (decode, aggregate, TTFT) and HumanEval at temperature
0, where draft sampling matters least.

## Measuring: reasoning effort, not enable_thinking

Inkling's chat template ignores `enable_thinking` (what `../bench/` sends
for "thinking off") and reads only `reasoning_effort`: `none`, `minimal`,
`low`, `medium`, `high` (the default, 0.9), `xhigh`, or a number in
[0, 0.99]. Its card: the minimum "does not fully disable reasoning". On a
palindrome prompt at temperature 0 the default thought 398 tokens and hit a
400-token cap with no answer; `none` thought 133 and answered. So measure
with the template key set, for both engines:

```bash
GB10_TEMPLATE_KWARGS='{"reasoning_effort": "none"}' python3 ../bench/perf.py
```

Requests that send no temperature get 1.0 (the checkpoint's
`generation_config.json`, which vLLM applies); `../bench/` sends 0.

## First boot here

2026-11-21, image `c2dc759f5984...`, vLLM `0.1.dev25855+g7cb5dae22`, model
revision `ed40e7da`, defaults (DSpark 7, GPU_UTIL 0.8): 5 min 13 s to
ready, nothing cached yet.

- Weights: 18.19 GiB in 102.5 s (~180 MiB/s, one shard read cold; vLLM
  turns its prefetch off on ext4, `--safetensors-load-strategy=prefetch`
  forces it). Weights and draft take 19.74 GiB.
- Warmup: JIT 7 s, FlashInfer autotune 7 s, CUDA graphs 100 s (piecewise
  up to 512 tokens, full up to 64 requests, plus the draft's), 2.83 GiB.
- KV: 68.68 GiB, 1,342,548 tokens, 5.12 requests of 262,144. vLLM's
  extensible KV cache says utilisation below 1.0 is only needed for memory
  it does not see, "the CPU on a unified-memory device" among them: that
  is this machine, so 0.8 stays. With the server up and a desktop session
  beside it, `free -g` showed 11 GiB available.
- Warnings worth knowing: the scheduler caps a step at 2048 tokens with 7
  draft tokens (`MAX_BATCHED_TOKENS` raises it, untested); KV padding
  layers "may waste at most" 12.5% / 12.5% / 350% in three cache groups;
  the draft gets no image embeddings (text-only draft inputs on image
  prompts, so expect lower acceptance there); the target's attention runs
  FlashAttention 2 (SGLang's recipe: FA4).
- First requests, temperature 0: DSpark accepted 3.1-3.2 tokens a step
  (~32% of drafted tokens) on reasoning text; 44 tok/s end to end for 512
  tokens at `reasoning_effort` none, one stream.

## First measurements

2026-11-21, same build and defaults, the server restarted after a cache
drop and compaction (111.7 GiB free at start). The 27B column is the best
native NVFP4 + DFlash2 profile in `../results/RESULTS.md`.

`../bench/perf.py` ran **without** `GB10_TEMPLATE_KWARGS`, so its "thinking
off" rows are reasoning text at the template's default effort (0.9), and
the long-context rows flag "the answer looks wrong": with 8 tokens allowed
the model only begins its reasoning ("The user has provided a long..."),
which is not a broken answer. Decode on a code answer at effort `none` is
still to measure.

| | Inkling-2-Mini, vLLM | Qwen3.8-27B NVFP4, SGLang |
|---|---:|---:|
| Single-stream decode, 700 tokens | 79.6 tok/s (reasoning text; 4.74 tokens a step, 53% accepted) | 94.2 (code) |
| TTFT, short prompt | 184 ms | ~190 ms |
| Aggregate at 32 streams, 300 tokens each | 404.2 tok/s | 597.8 |
| Prefill 2K / 8K / 32K / 101K | 2,245 / 2,847 / 2,503 / 2,300 tok/s | 2,768 / 2,147 / 1,841 / 1,223 |
| Decode during a 32K prefill, 1 / 4 streams | 4.6 / 19.9 tok/s, longest pause 0.79 / 0.91 s | 4.2-8.0 / 14.5-18.7, 1.28 / 1.31 s |

HumanEval (`../scripts/run-humaneval.sh nothink` with
`GB10_TEMPLATE_KWARGS='{"reasoning_effort": "none"}' GB10_MAX_TOKENS=4096`,
4 streams, 923 s, 120,369 tokens): 96.3% (158/164), five of the six misses
truncated at 4096. Those five re-run at 16384: three answered and passed,
HumanEval/116 and /145 hit 16384 again (never closed their reasoning).
**98.2% (161/164)**; the one wrong answer is HumanEval/118. Effort `none`
still thinks: 734 tokens a problem on average, against ~200 for the 27B
with thinking off (93.9% NVFP4, 97.6% FP8), so per answer it costs about
3.7x the tokens.

`perf.py` again at effort `none`
(`GB10_TEMPLATE_KWARGS='{"reasoning_effort": "none"}'`, 14:45 UTC, same
server): the code answer drafts worse than the reasoning text did.

| | effort 0.9 (above) | effort `none` |
|---|---:|---:|
| Single-stream decode, 700 tokens | 79.6 tok/s, 4.74 tokens a step, 53% accepted | **57.1** (55.3-57.4), 3.48 a step, 35% |
| TTFT, short prompt | 184 ms | 202 ms |
| Aggregate at 8 / 16 / 32 streams | 173.2 / 253.7 / 404.2 | 167.4 / 246.1 / 374.8 |
| Prefill 2K / 8K / 32K / 101K | 2,245 / 2,847 / 2,503 / 2,300 | 2,568 / 1,969 / 2,445 / 2,254 |
| Decode during a 32K prefill, 1 / 4 streams | 4.6 / 19.9, pause 0.79 / 0.91 s | 3.4 / 14.4, pause 0.84 / 1.60 s |

The long-context rows still flag "the answer looks wrong": at `none` the
model still opens with reasoning, and 8 tokens hold only its start. Host
memory during the run: 2.9 GiB free (page cache aside) with the server and
a desktop session up.

Reading: slower than the 27B at decode (57 against 94 tok/s on the same
code prompt) and aggregate (375-404 against 598), faster at prefill from
32K up (2.0x at 101K, 1.3x at 32K) and gentler on streams during a long
prefill; more accurate on HumanEval, at several times the tokens. Next:
`DRAFT_SAMPLE_METHOD=` (greedy drafting; the probabilistic method may cost
acceptance at temperature 0), the SGLang twin (FA4), `MAX_BATCHED_TOKENS`
above 2048.

Notes: the container runs as root, so
`~/.cache/gb10-vllm-inkling` (compile caches, kept between runs) ends up
root-owned; the weights are mounted read-only (`MODEL_RO=0` if the loader
wants to write beside them).
