# Measured results

One ASUS DGX Spark, August 2026, box otherwise idle for every figure.

> **Build.** Every figure here was measured on the Docker build: the
> MiaAI-Lab toolkit's patched SGLang image, see
> [BUILD-MANIFEST.md](BUILD-MANIFEST.md). The recipe now runs stock SGLang
> 0.5.20 natively (`serve.sh`), carrying over the serving flags the
> toolkit passed except `--mem-fraction-static`: 0.80 instead of 0.85. Expect
> the same ordering; re-measure before quoting absolute numbers for the
> native build.

**Compute** GB10, 20 cores, 128 GB unified, 916 GB NVMe. Ubuntu 24.04,
kernel 6.17.0-1031-nvidia, aarch64. Driver 580.173.02, CUDA 13.0.
Docker 29.2.1, nvidia-container-toolkit 1.20 via CDI.

**Serving.** SGLang + `RadixArk/Qwen3.8-27B-NVFP4` + `z-lab/Qwen3.8-27B-DFlash2`,
`--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, 262144 context,
`--mem-fraction-static 0.85`, CPU pinned to `5-9,15-19` (the Cortex-X5 cores).

**Draft tokens differ by section.** Everything below was measured at
`--speculative-num-draft-tokens 8` — TTFT, thinking-on, prose, concurrency,
prefill, long context and HumanEval — **except** the draft=16 rows, which are
labelled as such. The two settings are not interchangeable: draft=16 is +28% on
single-stream and −10% on aggregate, so mixing them in one figure is misleading.

Throughput is code generation (LRUCache prompt) at temperature 0, counting
`completion_tokens` over wall time.

## Latency and single-stream

| Metric | Result |
|---|---:|
| Time to first token | 190 ms (184–193, n=5) |
| Decode, thinking off, draft=16 | **78.6 tok/s** (n=5; 77.0–78.9) |
| Decode, thinking off, draft=8 | 61.3 tok/s |
| Decode, before concurrency tuning | 63.1 tok/s |
| Decode, thinking on | 46.7 tok/s |
| Prose decode | 26.0 tok/s |
| Gateway overhead (LiteLLM vs direct) | 62.3 vs 62.5 — none |

## Concurrency

Aggregate tok/s at each `max_running_requests` cap.

| Streams | cap 4 | cap 12 | cap 16 | TTFT p50 (16) | TTFT max (16) |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.3 | 53.4 | 52.9 | 0.20 s | 0.20 s |
| 2 | 93.1 | 92.6 | 87.8 | 0.75 s | 1.20 s |
| 4 | 187.1 | 191.5 | 193.1 | 0.41 s | 0.41 s |
| 8 | 190.1 | 296.0 | 315.9 | 0.37 s | 0.37 s |
| 16 | — | 307.8 | **480.7** | 0.68 s | 0.68 s |
| 24 | — | 319.6 | 416.5 | 0.47 s | 9.98 s |
| 32 | — | 368.9 | 469.8 | 5.23 s | 11.09 s |

At the shipped cap of 4, going 4 → 8 gains 2% — that flatline is config, not
hardware.

**The peak tracks the cap, it is not a hardware limit.** At cap 12 the peak sits
at 32 streams; at cap 16 it moves to 16. TTFT p50 and max are identical at the
cap because nothing queues there, and the 24/32 rows are queueing against it —
not the GPU running out. Raising `max-mamba-cache-size` moves the peak again
(pool 160 → 474.0 at 32 streams). See [REPRODUCTION.md](REPRODUCTION.md).

## Prefill / long context

| Prompt | TTFT | Prefill rate |
|---:|---:|---:|
| 2,448 tok | 1.54 s | 1,591 tok/s |
| 9,698 tok | 4.46 s | **2,173 tok/s** |
| 38,698 tok | 20.42 s | 1,895 tok/s |
| 120,865 tok | **94.89 s** | 1,274 tok/s |

Prefill peaks near 10K tokens then degrades. The 262K context is real, but a
~120K prompt costs ~95 s before the first token.

## Draft tokens

`--speculative-num-draft-tokens`, all values on the same build. `accept_len` is
mean accepted tokens per step; `accept_rate` the fraction of drafted tokens kept.

| draft | single | agg @16 | accept_len | accept_rate |
|---:|---:|---:|---:|---:|
| 6 | 49.5 | 404.2 | 5.25 | 0.85 |
| 7 | 55.7 | 386.6 | 5.89 | 0.82 |
| 8 (default) | 61.3 | 429.2 | 6.63 | 0.80 |
| 10 | 65.2 | **435.1** | 7.18 | 0.69 |
| 12 | 75.1 | 402.8 | 8.27 | 0.66 |
| 14 | 78.8 | 386.5 | 8.87 | 0.61 |
| 16 | **78.6** | 384.8 | 9.52 | 0.57 |
| 20 | 72.0 | 330.7 | 8.48 | 0.40 |
| 24 | 69.0 | 284.5 | 8.40 | 0.32 |

**+28% single-stream** from the default. The optima diverge: 16 for interactive,
**10 for concurrent serving**.

`accept_len` rises to 9.52 at draft=16 then falls at 20 and 24 — past 16 the
drafter cannot sustain longer correct runs, so the extra draft compute buys
rejected tokens. Under 16-stream saturation that wasted compute competes with
real work, which is why aggregate peaks much earlier.

draft=16 has now been measured three times: **85.7** in-sweep here, **78.6** on a
fresh boot here, **84.4** on a [second machine](REPRODUCTION.md). The table
publishes 78.6 because that was the fresh-boot re-measurement, but two of the
three cluster at 84–86, so **78.6 is probably the low end of the spread rather
than the centre** and the true figure is likely nearer +40% than +28%.

That spread is also why 12 / 14 / 16 are not distinguishable from each other —
only the broad shape is trustworthy.

Quality is unaffected: the second machine measured HumanEval at both settings and
found ±2 problems, inside the documented nondeterminism band. Expected, since
every draft token is verified against the target model — a wider window changes
throughput, not output.

Sweep values contradict the "block-7 peak" reported elsewhere for this stack;
that figure was measured on DSpark, not DFlash2.

## Long-context concurrency

Unique content per stream (`cached_tokens == 0`), so this is real KV demand.

| Streams @ 83K | Unique KV | Wall | TTFT max |
|---:|---:|---:|---:|
| 4 | 332,710 | 277.4 s | 277.1 s |
| 8 | 665,932 | 555.8 s | 555.4 s |
| 12 | 998,767 | 832.8 s | 832.2 s |

**69.4 s per stream, dead linear.** Prefill fully serialises — `#new-seq: 1` per
batch at ~1,000 tok/s — so concurrency does not help long-context work; it just
makes everyone wait proportionally longer. Decode throughput is the wrong
predictor for agent workloads over large inputs.

A 16-stream run did not fail on memory: the model was too busy to answer a 5 s
health probe, so the supervisor restarted it (see Traps in the README).

**Test design matters here.** Build every prompt from shared filler and the
radix cache deduplicates them — the first version of this test measured cache
hits, not capacity. Assert `cached_tokens == 0`.

## mem-fraction

| Config | Single-stream | 16 streams | 32 streams | Free GPU |
|---|---:|---:|---:|---:|
| 0.82 + 1M cap | 60.0 | **480.7** | 469.8 | 20.6 GB |
| 0.90 + 1M cap | 62.3 | 468.7 | 468.7 | 25.3 GB |
| 0.90, pool uncapped | 52.3 | 468.7 | 384.7 | 8.1 GB |
| **0.85 + 1M cap** | 61.8 | 472.0 | **477.0** | **26.1 GB** |

**Not a performance lever.** A 10-point spread moves throughput by less than
run-to-run noise. Uncapping the pool grows it 1,048,576 → 1,496,047 tokens, but
nothing uses the space (peak observed usage 67%) and the lost headroom cost 18%
at 32 streams. 0.85 with the cap kept is the settled default: baseline
throughput, most free memory.

## Quality — HumanEval

164 problems, temperature 0, each executed against its real unit tests in a
`--network none` container. Not self-judged.

| Mode | pass@1 | Tokens/problem | Wall |
|---|---:|---:|---:|
| Thinking off | 93.9% (154/164) | ~200 | 3 min |
| Thinking on | **97.0%** (159/164) | ~945 | 19 min |

Thinking fixed 7 genuine failures (77, 91, 93, 103, 116, 127, 160).

Of the 5 remaining thinking-mode failures, **only one is a wrong answer**
(HumanEval/115). The other four burned a full 16,384-token budget without
emitting code. Excluding runaways: 99.4% (159/160).

**Four measurement traps**, all hit here:

- **Truncation reads as a quality regression.** A first thinking run at a 4,096
  cap scored 90.9%; 14 of its 15 failures had `finish_reason == "length"`.
  Always check it before quoting a thinking-mode score.
- **Boot-to-boot variance is ~8% on single-stream**, against <2% run-to-run
  within one instance. Several small deltas in this document sit inside that
  band; treat only large moves as signal.
- **Greedy is not bitwise deterministic.** Temperature 0 still flips 2–3
  problems between runs, because dynamic batching plus speculative decoding
  changes reduction order. Treat pass@1 as ±2 problems.
- **Harness bugs masquerade as model errors.** A naive run scored 92.7% where 5
  of 12 "failures" were mine: the fence regex required a *closing* fence, and
  HumanEval/38 and /50 define a helper above the target that the tests need.

## Thermals

At 96% GPU utilisation, sustained:

| | |
|---|---:|
| GPU temperature, decode | 59 °C (idle 36 °C) |
| GPU temperature, sustained prefill | 74 °C |
| SM clock | 2405 MHz |
| Throttle reasons active | `0x0` — none |
| ACPI thermal zones | 59–70 °C |

Prefill, not decode, is the thermally interesting workload: 74 °C leaves 6 °C of
margin to the 80 °C suspend threshold, against 21 °C on decode-heavy work.

Reported ~32 W board power under load is implausibly low for GB10 — that rail
appears to cover only part of the SoC. Don't use it for power budgeting.

## Memory

| `max-mamba-cache-size` | GDN pool | Free GPU after load | Peak aggregate |
|---:|---:|---:|---:|
| 20 | 3.8 GB | 38.1 GB | 190 tok/s |
| 64 | 12.2 GB | 24.1 GB | 369 tok/s |
| 80 | 15.6 GB | 20.6 GB | **480.7 tok/s** |

At 80 slots: `ssm_state` 5.70 GB + `intermediate_ssm_state_cache` 9.56 GB +
conv caches 0.38 GB — these hold only with `--mamba-ssm-dtype bfloat16`. Left
unset the SSM state resolves to `float32`, which doubles this to 30.9 GB and
leaves ~12 GB free. On this hybrid architecture concurrency is bought with
mamba state, not KV cache — worth remembering when sizing a second model
alongside it.

## FP8 target (2026-08-27)

`Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` on the same DFlash2 image and drafter,
measured against NVFP4 on fresh boots the same day, same box. Both at the
max-aggregate config: **pool 160 / cap 32 / draft 10**, `--kv-cache-dtype
fp8_e4m3`, `--mamba-ssm-dtype bfloat16`, 0.85, CUDA graphs to bs 32.

Provenance note: the NVFP4 leg ran its then-current default revision
`319f741c` — upstream moved main off the pinned `554ebba9` (the revision
behind every earlier table here) on 2026-08-22, and the standalone launcher
does not pin the target. Caught by listing the cached snapshots, not by the
boot log. The same-day pairing below stands as measured; comparisons against
the older tables carry this caveat.

| | NVFP4 | FP8 | FP8 ÷ NVFP4 |
|---|---:|---:|---:|
| Single-stream decode | 70.1–70.2 | 49.5–51.3 | 0.70–0.73 |
| Peak aggregate | **571.1–572.7** @ 32 | 494.0–513.8 @ 32 | 0.86–0.90 |
| TTFT | 197–198 ms | 273–293 ms | ×1.4 |
| Prefill, 9.7K prompt | 2,658 tok/s | 1,065 tok/s | 0.40 |
| TTFT, 121K prompt | 91.9 s | 140.2 s | ×1.5 |
| KV pool after load | 913,336 tok | 701,546 tok | |
| Free GPU after load | 12.1 GB | 13.3 GB | |

Pool 160 moves the NVFP4 peak 480.7 → 572 (two runs 0.3% apart) — the peak
tracks the pool, again — and 32-stream TTFT stays sub-second, so nothing
queues. The FP8 aggregate gap *narrows* with concurrency: 0.76 at 16 streams,
0.86–0.90 at 32, while single-stream sits at 0.70–0.73 in every config
measured. (Two boots of the FP8 config differ 4% on peak and 3.5% on
single-stream — inside the ~8% band.) Moving
from the baseline config (draft 8, pool 80) to this one lifted single-stream
by the same ~15% on both targets (61.3 → 70.1, 44.6 → 51.3); a controlled
draft sweep on FP8 would be needed to separate the draft effect from the pool
effect.

**Quality runs the other way.** HumanEval, thinking off (the FP8 leg was
measured at pool 80 / draft 8; batch size does not change the checkpoint):

| | NVFP4 | FP8 |
|---|---:|---:|
| pass@1, thinking off | 93.9% (10 fails) | **97.6%** (4 fails) |
| Failure kinds | wrong answers, truncations frequent | 4 wrong answers, zero truncations |

FP8 with thinking off already matches NVFP4 with thinking on (97.0%) at a
fifth of the tokens. With thinking on, FP8 completed **157/157 — zero wrong
answers** — but 7 of 164 ran past the 16,384-token budget (~1,569 tokens per
problem against ~945 on NVFP4): the milder quant thinks longer and runs away
more often. Retry of those seven at a 65,536 cap is pending; until it lands,
quote the thinking-on figure as 100% excl. truncation with that caveat.

KV dtype measured as a null on NVFP4, thinking off: 8- and 16-bit KV give the
same 9–10 failures and the same throughput.

The memory arithmetic closes across both legs: at pool 160 they differ by
211,790 KV tokens, which is the FP8 weight delta (~8.5 GB) at the ~40
KB/KV-token rate implied by the mem-fraction table. With the 0.196 GB/slot GDN
cost this sizes any future config: pool 160 fits both targets at 0.85 with
≥12 GB free; pool 240 (~47 GB of GDN) fits neither. A third checkpoint under
the same config validates the rate a second time: the finetune above, whose
BF16 `lm_head` and unquantized tails sit 87,690 KV tokens (≈3.6 GB) below
NVFP4's pool, exactly where the head delta predicts.

`spec_accept_length` 7.30 / `spec_accept_rate` 0.700 at draft=10 on NVFP4
(server `/metrics`) — within 2% of the 7.18 / 0.69 of the draft sweep, so the
wider window verifies as expected. The same drafter against the FP8 target:
7.11 / 0.678 — acceptance ~3% lower, a slight distribution mismatch that costs
~2.6% of tokens per verify step, a minor share of the FP8 speed gap. A third
target — an uncensored finetune of the same family
(`orcarouter/Qwen3.8-27B-Uncensored-NVFP4` @ `69d21348`, same config) —
accepts **7.32 / 0.702**: the mild finetune did not move the target
distribution at all. Useful evidence that the DFlash2 stack ports to finetunes
of this family unchanged.

That finetune is also a side datum (not part of the A/B) that quant recipe
matters as much as width: it keeps the NVFP4 MLPs but carries a BF16 `lm_head`
in a compressed-tensors layout, and at identical acceptance it runs 15% slower
single-stream and 25–40% slower prefill than RadixArk, while its 32-stream
aggregate matches (569 vs 572) — the head GEMM dominates verify and prefill
logits, and batching hides it.

Boot: FP8's first boot took 77 min (flashinfer autotune compiling the FP8 GEMM
path); every later boot 5.5 min, the triton cache being mounted. NVFP4 boots
in 3.4 min. The autotune cost is once per config.

## Native build, first run (2026-09-24)

Stock SGLang 0.5.20 from pip, no Docker (`serve.sh`, profile
`qwen3.8-27b`). Target
`orcarouter/Qwen3.8-27B-Uncensored-NVFP4` @ `69d21348`, the finetune in the
FP8 section above, at the same max-aggregate config: draft 10, pool 160 /
cap 32, `--mamba-ssm-dtype bfloat16`, fp8 KV, 8192-token chunks. One
difference: `--mem-fraction-static 0.80` instead of 0.85 (KV pool 670,609
tokens). One boot, one run.

| | Docker build (above) | Native |
|---|---:|---:|
| Single-stream decode | ~60 (15% below RadixArk's 70.1) | **56.5** tok/s |
| Peak aggregate @ 32 | 569 | **558.0** tok/s |
| TTFT, short prompt | — | 239 ms |
| Prefill, 32K prompt | — | 1,105 tok/s |
| TTFT, 101K prompt | ~115–129 s at 121K (RadixArk's 91.9 s, 25–40% slower) | 117.6 s |
| `spec_accept_length` | 7.32 | 7.7–8.6 |

**Within the noise of one boot.** Single-stream is 5% and aggregate 2% below
the Docker figures, against the ~8% boot-to-boot spread measured on this
stack; long prefill lands inside the band the finetune's 25–40% prefill
penalty predicts. A native-vs-Docker verdict needs several boots of each.
The accept lengths come from different SGLang builds and are not compared.

Two things to re-check:

- **The 8K prefill point, 1,262 tok/s**, is lower than both the 2K (2,073)
  and the trend. The warmup's longest prompt is ~3K tokens, so 8K was
  probably the first extend at that size and paid a one-off kernel compile
  inside its TTFT. `perf.py --only prefill` on the warm server settles it.
- **83 °C at 101K prefill**, with no throttle reported and no suspend,
  against 74 °C sustained prefill in the Docker-era measurements (at 121K).
  Room temperature, the box, or the 80 °C suspend figure under "Thermals":
  one run cannot tell which, but the 80 °C figure was not a hard limit here.

## Reference comparison

| Configuration | Reported | Source |
|---|---:|---|
| FP8 + vLLM | ~32 tok/s | widely-shared Reddit recipe |
| NVFP4 + vLLM | +29–34% over FP8 | NVIDIA developer forums |
| NVFP4 + SGLang + DFlash2 | 50–51 tok/s | MiaAI-Lab, hasso5703 |
| **This build**, draft=8 | 61.3 single / **480.7** aggregate | measured here |
| **This build**, draft=16 | **78.6** single / 384.8 aggregate | measured here |
| **This build**, draft=10, pool 160 / cap 32 | 70.1 single / **572.7** aggregate | measured here |
| FP8 target, same stack and config | 49.5–51.3 single / 494.0–513.8 aggregate | measured here |

Prompt shapes differ across sources, so absolute comparison is approximate. The
FP8-vs-NVFP4 and vLLM-vs-SGLang ordering is the durable part.
