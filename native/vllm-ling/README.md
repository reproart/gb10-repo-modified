# Ling-3.2-air (NVFP4 W4A16) on vLLM

[`primitive-ai/Ling-3.2-air-NVFP4A16`](https://huggingface.co/primitive-ai/Ling-3.2-air-NVFP4A16)
is inclusionAI's Ling-3.2-air (124B MoE, 5.5B active, 35 KDA + 7 MLA layers,
images and video, one MTP layer, MIT) with its routed experts in NVFP4 and
BF16 activations; everything else is BF16. 72.6 GiB on disk. The FP8 release
(129.3 GB) does not fit one Spark.

Measured 2026-11-24, see the end; the rest is from the card, the model's
files and vLLM's source.

```bash
../scripts/mem-prep.sh idle && hf download primitive-ai/Ling-3.2-air-NVFP4A16 \
  --revision d4107e488e0cbbc06c4ebd6552da25e0cbb114b9 \
  --local-dir /models/primitive-ai/Ling-3.2-air-NVFP4A16
./pull.sh       # the container (12.2 GB compressed), the digest in image.lock
./serve.sh      # foreground, Ctrl-C stops; http://<host>:8888/v1, "ling-3.2-air"
```

## Why vLLM, and which build

The SGLang cookbook's recipes (open PR
[#52593](https://github.com/sgl-project/sglang/pull/52593)) are TP2-TP8 on the
FP8 release, with no Spark cell. This checkpoint's card names vLLM at commit
`a2bcb5657a88` (2026-11-05). Its `nightly-` tag is gone from Docker Hub;
`image.lock` pins `cu134-nightly-` of the same commit, which has an arm64
build. Its CUDA 13.4 runs on driver 580 (13.0): checked 2026-11-24 with a BF16
matmul and then the model (vLLM 0.33.0rc1.dev806).

## Knobs

`config.sh`, each overridable from the shell:

| Variable | Default | |
|---|---|---|
| `MODEL_DIR` | `/models/primitive-ai/Ling-3.2-air-NVFP4A16` | the weights |
| `MAX_MODEL_LEN` | 32768 | the card's; the checkpoint's maximum is 262144 |
| `GPU_UTIL` | 0.7 | `--gpu-memory-utilization`; 0.8 is too much, see below |
| `SPEC_TOKENS` | 1 | proposals from the MTP layer; 0 = off |
| `MOE_BACKEND` | vLLM's | `--moe-backend` |
| `MULTIMODAL` | 1 | 0 = `--language-model-only` |
| `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS` | vLLM's | |
| `REASONING_PARSER`, `TOOL_CALL_PARSER` | ling3 | empty = off |
| `PORT`, `HOST`, `API_KEY` | 8888, 0.0.0.0, none | |
| `MIN_FREE_GIB` | GPU_UTIL of the pool + 2 | refuse to start below this MemAvailable |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `vllm serve` flags; more `NAME=value` |

## What to watch

- **Answers before speed.** Ling-3.0-flash gave NaN logits ("!!!!") on SGLang
  here and its speed numbers were worthless, so `../bench/consistency.py`
  comes first.
- **The SwiGLU clamp** (4, experts of layers 35-41). At this commit vLLM
  filters out NVFP4 backends that drop it. Its Marlin path clamps before
  SiLU, `SiLU(min(g, 4))`; the reference is `min(SiLU(g), 4)`. They differ
  only on saturated gates, by at most 1.8% (3.93 against 4).
- **Context.** The SGLang page says 1M through YaRN in `config.json`; neither
  this checkpoint's nor the FP8 release's current one has it (262144).
- **MTP** was slower than plain decode on the SGLang page (169 against 279
  tok/s, 4x GB300); here it was faster (Measured), so 1 is the default.
- **`--trust-remote-code`**, as the card: Python from the checkpoint directory
  runs in the container.
- Thinking is on by default; the template reads `enable_thinking`, which
  `../bench/` already sends. The default sampling is 1.0 / 0.95 / 20.

The card's numbers (RTX PRO 6000, 8K in / 512 out): 104.9 tok/s alone, 555 at
32 streams; drift from FP8 +10.5e-3 nats/token; knowledge 90.2 (FP8 90.3).

## Measured (2026-11-24, GX10, no MTP, thinking off)

`MAX_MODEL_LEN=131072 EXTRA_ARGS=--enable-prompt-tokens-details`, GPU_UTIL 0.7,
every NVFP4 layer on Marlin (vLLM's own choice). Weights loaded from NVMe in
57-78 s; the first boot took 5.2 min to ready.

Three boots, same settings but for MTP. The first plain one was ~18% slower
than the second, in prefill too (cause not found), so compare MTP with the
second.

| | plain, boot 1 | plain, boot 2 | MTP (`SPEC_TOKENS=1`) |
|---|---|---|---|
| decode, 1 stream, code | 18.6 | 22.7 | 42.3 tok/s (1.94 tok/step, 94-96% accepted) |
| decode, 1 stream, Russian prose | 18.7 | 22.6 | 39.2 tok/s (1.8 tok/step) |
| aggregate, 2 streams | 40.0 | 47.5 | 69.6 |
| 4 | 71.4 | 79.1 | 112.6 |
| 8 | 105.4 | 128.5 | 169.2 |
| 16 | 160.8 | 184.0 | 244.8 |
| 24 | 187.6 | 236.1 | 296.2 |
| 32 | 228.7 | 267.5 | 252.0 (queued) |
| prefill 2K / 8K / 32K / 101K | 2259 / 2510 / 2364 / 1754 | 2581 / 2958 / 2785 / 2126 | 2526 / 2934 / 2719 / 2007 |
| TTFT, short prompt | 154 ms | 129 ms | 132 ms |
| KV tokens | 992,165 | - | 424,423 |

| HumanEval | |
|---|---|
| thinking off, no MTP (boot 1) | 154/164 (93.9%), none truncated |
| thinking off, MTP (default settings, 32K) | 152/164 (92.7%), none truncated |
| thinking on, no MTP (boot 1), 16384-token budget | 161/164 (98.2%); 2 of the 3 failures (32, 116) hit the budget |

- **MTP** is 1.86x on code and 1.73x on Russian prose against boot 2, the
  reverse of the SGLang page's result on GB300, and level at 32 streams, where
  a smaller KV pool (or a cap) queues some requests. The MTP HumanEval run fails
  95 and 108 besides the plain run's ten; 136 of the 164 answers are the same
  text. One run each, 2-3 problems flip between runs here, so it shows no loss and excludes only
  a large one. 99.3% of drafts were accepted in it.
- **Default settings** (MTP, 32K): KV 236,326 tokens, 22 GiB available after
  the boot and 21 after HumanEval.
- **Memory.** At 0.8 (97.3 GiB asked) vLLM held 104.5 GiB and the system had
  3.4 GiB available, against earlyoom's 1 GiB threshold. At 0.7 it held 88.1
  GiB (992,165 KV tokens) with 18 GiB available; the 101K-token prefill took
  another 11 GiB for a moment (3.7 GiB available at the lowest; 2.4 with MTP, and after
  the whole perf.py 2.2-2.7). The default is 0.7; for 100K prompts lower it
  or close other programs.
- **Answers.** No NaN; `consistency.py`'s six tool-call checks pass. Its four
  FAILs are determinism: the same greedy request, cold twice, already differs
  in its first-token logprobs (top-5 gap 0.22-0.26). The card says vLLM's
  forward pass for this model is not bit-reproducible. The cold-vs-hit gap
  is as large, so the cache restore cannot be told apart from that noise:
  a wrong restore is not shown, nor excluded.
- **HumanEval** failed 32, 83, 91, 120, 127, 130, 132, 134, 145, 163 with
  thinking off; 32, 91, 116 with thinking on. One run each.
- The 262144 window, images and video were not tried.
