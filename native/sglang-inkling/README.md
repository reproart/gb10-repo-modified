# Inkling-2-Mini on SGLang (LMSYS container)

`thinkingmachines/Inkling-2-Mini` is Thinking Machines' looped
Mixture-of-Experts model: 26.4B parameters, 3.4B active (5.4B effective
per token with the loop's three passes, as its card counts), text and image
input, Apache-2.0. 24 layers of 128 routed experts (top-8, no shared
expert); layers 6-17 run three passes per token, so every token makes 48
layer visits. Routed experts are NVFP4 in layers 0-19 and MXFP8 in 20-23. A
3-layer DSpark draft head is bundled in the checkpoint. ~18.2 GiB on disk.

SGLang support is not in a release yet: it is
[sgl-project/sglang#51926](https://github.com/sgl-project/sglang/pull/51926)
(open on 2026-11-21, the day this directory was written). The cookbook names
one runtime for DGX Spark, LMSYS's `lmsysorg/sglang:inkling-2-mini` image, so
this directory runs that image, pinned by digest in `image.lock`, with the
cookbook's DGX Spark recipe as knobs, the way `../vllm-b12x/` runs its
model's container. When a release carries Inkling-2-Mini, it moves to
`../models/inkling-2-mini.sh` with a lock file like every other profile, and
this directory goes.

```bash
../scripts/mem-prep.sh idle && hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 \
  --local-dir /models/thinkingmachines/Inkling-2-Mini
./pull.sh       # the container (~15 GB compressed), the digest in image.lock
./serve.sh      # foreground, Ctrl-C stops; http://<host>:8888/v1, "inkling-2-mini"
```

Stop the SGLang server first (one pool of memory); `serve.sh` refuses
otherwise, and the other servers here refuse while this one runs.

## The recipe

The cookbook's two `dgx-spark` cells
(`docs/src/snippets/configs/thinkingmachines/inkling-2-mini.jsx` in the
SGLang repository), both marked verified, flag for flag:

| | Balanced | DSpark (default here) |
|---|---|---|
| Environment | `MAX_JOBS=4` | the same, plus `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1`, `SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1`, `SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1`, `SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1` |
| Parallelism | `--tp 1 --ep 1` | the same |
| Kernels | `--attention-backend fa4 --dtype bfloat16 --page-size 128` | the same |
| Speculative | none | `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8` |
| Memory | `--mem-fraction-static 0.45` | the same |
| The rest | `--grammar-backend xgrammar --enable-multimodal --reasoning-parser inkling --tool-call-parser inkling2` | the same |
| gsm8k (cookbook, 1319 questions, temperature 0) | 93.56% | 93.93% |

Added here: `--served-model-name inkling-2-mini` and `--enable-metrics`
(`../bench/perf.py` reads the accept length from it).

## Knobs

`config.sh`, each overridable from the shell:

| Variable | Default | |
|---|---|---|
| `STRATEGY` | dspark | `dspark` or `balanced` (the cookbook's two recipes) |
| `DRAFT_TOKENS` | 8 | `--speculative-num-draft-tokens` with DSpark |
| `MEM_FRACTION` | 0.45 | `--mem-fraction-static`; the cookbook's value, see below |
| `MAX_RUNNING`, `CONTEXT_LENGTH` | SGLang's | `--max-running-requests`, `--context-length` |
| `MULTIMODAL` | 1 | `--enable-multimodal`; 0 = text only |
| `REASONING_PARSER`, `TOOL_CALL_PARSER` | inkling, inkling2 | empty = off |
| `PORT`, `HOST`, `API_KEY` | 8888, 0.0.0.0, none | as for `../serve.sh` |
| `MIN_FREE_GIB` | MEM_FRACTION of the pool + 4 | refuse to start below this MemAvailable |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `sglang serve` flags; more `NAME=value` for the container, replacing a recipe value of the same name |

## What to watch

- **`MEM_FRACTION=0.45`.** The cookbook's Spark value, and the only one it
  verified there. Measured: 410,240 tokens of KV with ~50 GiB still available. This project runs the 27B at 0.85 and has seen machines wedge above
  that, so 0.6-0.85 is worth measuring rather than assuming.
- **Loop options do not apply to this checkpoint.** The cookbook's
  `--inkling2-loop-num-passes` and `--inkling2-loop-share-cache` need a
  variable-loop checkpoint (`loop_config.is_varloop: true`); revision
  `ed40e7da` has `is_varloop: false` (3 passes fixed), and SGLang refuses
  either flag at startup on it. DSpark refuses a non-default pass count in
  any case.
- **Decode cost.** 5.4B active, but 48 layer visits per token: the cookbook
  says batch-1 decode costs about the same per token as Inkling-2-Small.
  Expect it to be faster than the dense 27B on a Spark (less weight read per
  token), but that is an estimate until `../bench/perf.py` says otherwise.
- **SM12x defaults in the PR.** On SM120-class GPUs the PR forces the FA4
  attention backend, a P2P check and page size 128 when unset ("page size 1
  left 0.5 GB after the KV pool; requests ran out of memory").
- **The image tracks an open PR.** The tag was pushed 2026-11-20 20:16 UTC;
  the PR got commits after that. `image.lock` keeps this build until it is
  moved on purpose (`rm image.lock && EXPECT_PREFIX= ./pull.sh`).
- **Root-owned caches.** The container runs as root, so
  `~/.cache/gb10-sglang-inkling` (JIT and compile caches, kept between runs)
  ends up root-owned. The weights are mounted read-only (`MODEL_RO=0` if the
  loader wants to write beside them).

The vLLM twin, [`../vllm-inkling/`](../vllm-inkling/README.md), serves the
same weights on the same port and model id, for an engine A/B; its README
puts the two recipes side by side.

Measured 2026-11-24 with the recipe as it is: `../vllm-inkling/README.md`,
"Against the SGLang twin" (54.2 tok/s, 420.6 at 48 streams, prefill to 2,887
tok/s, other streams freeze during a long prefill). Boot 6.4 min; KV 410,240
tokens, 48 requests, context 1,048,576. HumanEval not run. Measure like the SGLang profiles, but with the template's own switch: Inkling's
chat template ignores `enable_thinking` (what `../bench/` sends for
"thinking off") and reads only `reasoning_effort` (default 0.9, "high";
`none` still thinks briefly), so

```bash
GB10_TEMPLATE_KWARGS='{"reasoning_effort": "none"}' python3 ../bench/perf.py
```

plus HumanEval and `DRAFT_TOKENS` / `MEM_FRACTION` sweeps. Details and the
vLLM twin's first boot: `../vllm-inkling/README.md`.
