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

Not measured here yet. Notes: the container runs as root, so
`~/.cache/gb10-vllm-inkling` (compile caches, kept between runs) ends up
root-owned; the weights are mounted read-only (`MODEL_RO=0` if the loader
wants to write beside them).
