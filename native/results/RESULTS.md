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
  and the trend. It came back at 1,333 on a second boot (below), so it is
  not a one-off kernel compile. It sits at the low edge of what the
  finetune's 25–40% prefill penalty predicts from RadixArk's 9.7K figures
  (1,300–2,000 tok/s).
- **83 °C at 101K prefill**, with no throttle reported and no suspend,
  against 74 °C sustained prefill in the Docker-era measurements (at 121K).
  Room temperature, the box, or the 80 °C suspend figure under "Thermals":
  one run cannot tell which, but the 80 °C figure was not a hard limit here.

### Draft 16 / cap 16 on the native build

Same finetune and build, next boot, only `DRAFT_TOKENS` and `MAX_RUNNING`
changed (the `qwen3.8-27b-single` profile; GDN pool 80).

| | 10 / 32 | 16 / 16 | |
|---|---:|---:|---:|
| Single-stream decode | 56.4 | **70.9** tok/s | **+26%** |
| `spec_accept_length` | 7.7–8.6 | 8.7–9.8 | |
| TTFT, short prompt | 239 ms | 246 ms | |
| Aggregate @ 16 streams | 439.6 | 375.4 tok/s | −15% |
| Peak aggregate | **558.0 @ 32** | 375.4 @ 16 | −33% |
| Prefill, 8K prompt | 1,262 | 1,333 tok/s | |

The direction and size match the Docker-era sweep (draft 10 → 16: +20%
single-stream, −11.5% aggregate at 16 streams), and +26% is well outside the
~8% boot-to-boot spread. The 10 / 32 single-stream figure repeated at 56.4 on
a later run of the same boot. At 24 and 32 streams the 16 / 16 config queues
against its cap of 16, as expected.

### RadixArk NVFP4 on the native build: same checkpoint as the Docker leg

`RadixArk/Qwen3.8-27B-NVFP4` @ `319f741c`, the revision of the FP8 section's
NVFP4 leg, at its config (draft 10, pool 160 / cap 32), on a second GB10
host. The only flag that differs is `--mem-fraction-static` 0.80 vs 0.85.
This is the like-for-like native-vs-Docker comparison.

| | Docker (FP8 section) | Native |
|---|---:|---:|
| Single-stream decode | 70.1–70.2 | **70.5** tok/s |
| Peak aggregate @ 32 | 571.1–572.7 | **597.8** tok/s |
| TTFT, short prompt | 197–198 ms | **191 ms** |

**Decode and concurrency: the native build is at least as fast.** Against
the Uncensored finetune on the native build (above), the FP4 head is +25%
single-stream, +7% aggregate, −20% TTFT and +35–50% prefill, which confirms
that finetune's BF16 `lm_head` as the cost (two hosts, so host-to-host
spread is in there too).

**Prefill: the Docker-era figures came from a different benchmark.** The
same native server, measured by both:

| Prompt | Older benchmark (behind every prefill figure above) | Current `bench/perf.py` |
|---|---:|---:|
| ~2K | 2,448 tok: 2,679 tok/s | 2,063 tok: 2,501 tok/s |
| ~10K | 9,698 tok: 2,406 tok/s | 8,092 tok: 1,594 tok/s |
| ~35K | 38,698 tok: 2,162 tok/s | 32,382 tok: 1,714 tok/s |
| 100K+ | 120,865 tok: 89.37 s (1,352 tok/s) | 101,120 tok: 83.70 s (1,208 tok/s) |

Through the older benchmark, the native build lands where Docker did (9.7K:
2,406 vs 2,658; 121K: 89.4 s vs 91.9 s), so there is no native prefill
regression. But the older benchmark reports a 121K prompt as cheaper per
token than a 101K one, which quadratic attention does not allow, and its
prompt sizes have not moved by a token in a month: fixed text, very likely
sharing prefixes that the radix cache serves. That is the trap described
under "Long-context concurrency". Until the server log's `#cached-token`
confirms or clears it, read every prefill figure above this section as
optimistic, and the current benchmark's curve as the honest one.

### HumanEval on the native build

Against the RadixArk server above (`319f741c`, draft 10 / cap 32), with the
Docker-sandboxed harness from the parent project (`scripts/run-humaneval.sh`
there; it is not part of this Docker-free project). 164 problems,
temperature 0.

| Mode | Docker build (`554ebba9`, draft 8) | Native build |
|---|---:|---:|
| Thinking off | 93.9% (154/164) | **95.1% (156/164)**, 8 wrong answers |
| Thinking on, `reasoning_effort` xhigh | 97.0% (159/164), ~945 tok/problem, 4 truncated | — |
| Thinking on, `reasoning_effort` medium | — | **98.8% (162/164)**, ~737 tok/problem, 1 truncated, 1 wrong |

**No quality regression.** Thinking off is +2 problems, inside the ±2 band
greedy decoding gives on this stack, and the set of failures shuffled the way
that band predicts (77, 103, 116 now pass; 101, 108, 145 now fail). The
checkpoint revision also differs from the Docker leg's, so a small true
difference could hide in there; the draft settings cannot, since every draft
token is verified.

**`medium` effort looks like the better thinking mode for code**: one
problem short of perfect at ~22% fewer tokens than xhigh, and one runaway
instead of four. xhigh was not re-run natively, so this is a cross-build,
one-run comparison. HumanEval/145 failed in every mode.

### Request cap 12 against 32 (2026-09-25)

`nvidia/Qwen3.8-27B-NVFP4` (revision not recorded), draft 10, 0.80, same
server and bench, only `--max-running-requests` / `--max-mamba-cache-size`
changed: 32 / 160 against 12 / 72, the new default. Aggregate tok/s,
300 tokens per stream, current `perf.py`; the cap-12 column is two runs.

| Streams | Cap 32 | Cap 12 |
|---:|---:|---:|
| 1 | 59.0 | 58.1 / 63.4 |
| 2 | 119.4 | 117.3 / 107.4 |
| 4 | 187.2 | 187.5 / 183.5 |
| 8 | 264.2 | 273.1 / 245.3 |
| 12 | — | **404.4** (36.9 per stream, TTFT 0.35 s) |
| 16 | 454.4 | queued (12 run, 4 wait: 306.0) |
| 18 | — | queued (TTFT max 9.1 s: 348.5) |
| 32 | 577.7 | queued (372.4) |

Single-stream decode 69.7 / 69.9 tok/s at cap 12, TTFT 200 ms.

**Up to the cap nothing is lost**: 1-8 streams agree within the
run-to-run spread (the 8-stream row varies by ~10% between runs of the
same config). What cap 12 gives up is only the throughput above 12
concurrent requests; a burst above the cap waits for a free slot (the 18
row: p50 TTFT unchanged, the last 6 requests wait ~9 s). The freed GDN
state goes to the KV pool.

The 8-stream row has the highest TTFT of the sweep in all three runs
(0.51 s, against 0.35 s at 12 and 0.42 s at 16 with cap 32). Not
investigated; it costs ~0.15 s once per request.

### Draft 10 / 11 / 12 at cap 12 (2026-09-25/26)

Same server as above (`nvidia/Qwen3.8-27B-NVFP4`, cap 12, 0.80), only
`DRAFT_TOKENS` changed. Aggregate tok/s, 300 tokens per stream; draft 10 two
runs (one at 12 streams), draft 11 and 12 three each.

| Streams | Draft 10 | Draft 11 | Draft 12 |
|---:|---:|---:|---:|
| 2 | 117.3 / 107.4 | **124.0 / 123.3 / 124.0** | 129.3 / 120.5 / 126.1 |
| 4 | **187.5 / 183.5** | 169.3 / 167.7 / 170.2 | 160.8 / 166.4 / 161.1 |
| 8 | 273.1 / 245.3 | **320.7 / 312.0 / 324.5** | 316.7 / 312.7 / 304.5 |
| 12 | **404.4** | 402.6 / 392.3 / 408.0 | 387.8 / 399.4 / 388.1 |
| mean of the four levels | 240 | **253** | 248 |

Against draft 10 (means): draft 11 is +10% at 2, -9% at 4, +23% at 8 and
-1% at 12; draft 12 is +11%, -12%, +20% and -3%. Draft 11 matches draft 12
at 2 streams and beats it at 4, 8 and 12. Single-stream decode (the separate
section, 700 tokens): 69.7 / 69.9 at draft 10, **73.6** at draft 11, 74.1 at
draft 12. accept_len single-stream: 7.4-8.6 at draft 10, 8.0-8.5 at 11,
8.5-9.4 at 12. Draft 11 became the base profile's default.

The 4-stream dip reproduces at draft 11 and 12 in every run (4 streams take
as long as 8), and so does the 8-stream gain. It is not CUDA-graph padding:
with speculative decoding SGLang 0.5.20 captures every batch size from 1 to
8. The verify batch is streams x draft tokens (40 rows at 4 x 10, 44 at
4 x 11, 48 at 4 x 12, 80 / 88 / 96 at 8), so a kernel-shape effect is the
likely cause; not profiled.

Prefill at draft 12 (unique prompts): 2,751 / 1,795 / 1,593 / 1,132 tok/s at
2K / 8K / 32K / 101K, 84 °C at the end of the 101K prompt, no suspend.

### RadixArk NVFP4 at cap 12: draft 11 vs 15 (2026-09-26)

`RadixArk/Qwen3.8-27B-NVFP4` (FP4 `lm_head`; revision not recorded) in place
of `nvidia/Qwen3.8-27B-NVFP4`; every other value is the base profile's (cap
12, pool 72, 0.80, bf16 GDN state, fp8 KV). Aggregate tok/s.

| | nvidia, draft 11 (3 runs, mean) | RadixArk, draft 11 | RadixArk, draft 15 (2 runs) |
|---|---:|---:|---:|
| Single-stream decode (700-token code answer) | 73.6 | 74.1 | **90.6 / 89.6 / 89.5** |
| accept_len there | 8.0-8.5 | 8.1-9.1 | 8.7-11.1 |
| Sweep, 1 stream (300 tokens) | — | **71.2** | 67.0 / 65.7 |
| 2 streams | 123.8 | 120.5 | 121.2 / 118.7 |
| 4 | 169.1 | 197.3 | 188.5 / 201.7 |
| 8 | **319.1** | 294.6 | 302.5 / 308.8 |
| 12 | **401.0** | 394.3 | 381.0 / 321.3 |
| Prefill 2K / 8K | — | — | 2,721 / 2,054 tok/s |

**The target makes no difference at draft 11:** 74.1 against 73.6
single-stream, within a few % at 2, 8 and 12 streams. (So the FP4 head is
not what separates these two NVFP4 exports; nvidia's is either FP4 too or
does not cost decode here.) The one exception: RadixArk has **no dip at 4
streams** (197 vs 169), so the dip seen at draft 11 and 12 is specific to the
nvidia checkpoint's kernel shapes.

**Draft 15's gain depends on the text.** On the 700-token code answer it is
+21% (89.5-90.6 vs 74.1). On the sweep's 300-token answers to varied prompts
it is gone: 66-67 vs 71 at one stream, even at 2-8 streams, and the 12-stream
row is lower in both runs (the 321 one had a slow wave: 11.2 s wall). A
longer draft pays only while acceptance stays high, which long, predictable
code gives and short or varied answers do not. The base profile keeps 11;
draft 15 suits long code generation for one user (`DRAFT_TOKENS=15`).

### Decode profile of the 27B (2026-09-30)

RadixArk/Qwen3.8-27B-NVFP4 + DFlash2, 11 draft tokens, cap 12,
`bench/profile_decode.py` (40 steps, ~111 ms each under the profiler). The
script's first cut filed cuBLASLt's `nvjet_sm121_qqtst_*` kernels under BF16;
`qq` are e4m3 operands (next to them `_static_quant_fp8`, 128 calls a step),
so they are the checkpoint's FP8 W8A8 layers. Regrouped by hand (the script
now has "dense GEMM, FP8 (cuBLASLt)"):

| Kernels | Calls a step | Avg | wall ms/step | Share |
|---|---:|---:|---:|---:|
| NVFP4 CUTLASS, the 64 MLPs (gate_up + down) | 126 | 339 us | ~43 | 38% |
| NVFP4 CUTLASS, lm_head (verify + DFlash2 candidates) | 2 | 3.1 ms | ~6 | 6% |
| FP8 W8A8 cuBLASLt `192x48`: GDN in_proj_qkvz / attention qkv_proj | 62 | 372 us | ~22 | 20% |
| FP8 W8A8 cuBLASLt `64x96`: GDN out_proj / attention o_proj | 62 | 191 us | ~12 | 11% |
| BF16 sm80 WMMA `128x1`: the DFlash2 draft (~3 GB of BF16) | 20 | 735 us | ~15 | 13% |
| BF16 small GEMMs (`128x2` + split-K reduce): GDN in_proj_ba, 96 x 5120 | ~48 | ~66 us | ~2 | 2% |
| GDN, norms, attention | | | ~6 | 6% |

So the target is quantized almost throughout: `FP8_SIDE=1` converts 75
layers, 0.24 GiB -> 0.13 GiB: GDN's 48 in_proj_ba (96 x 5120, "48 with N
padded"; the BF16 small GEMMs above) and 27 vision-tower qkv; the rest of
the vision tower stays BF16 and never runs in text decode. At most ~2%.
What is left to try:

- the MLPs' NVFP4 GEMM runs at ~40% of the weight-read speed (gate_up is
  ~50 MB of FP4 plus scales, ~180 us at 273 GB/s, against 339 us for the
  average call): `FP4_GEMM_BACKEND=marlin` (W4A16) or another
  `--fp4-gemm-backend`;
- the draft: `FP8_DRAFT=1`, its MLPs and o_proj to FP8 weight-only
  (halves ~13% of the step at the cost of some accept_len, if any);
- the FP8 projections are already near the weight-read speed for
  in_proj_qkvz (~84 MB, ~310 us ideal, 372 us); out_proj reaches ~60%.

To measure.

## Qwen3.8-Flash-Next, one Spark (2026-09-27)

`RadixArk/Qwen3.8-Flash-Next-NVFP4`, `models/qwen3.8-flash-next.sh` as
shipped: SGLang 0.5.20, TP 1, NEXTN MTP 3/1/4, cap 8, GDN pool 40 (fp32
state), 0.85, bf16 KV, the n-gram table read in place from the checkpoint
(patches/gb10_ple_mmap.py: "128 shard tensors left in place", nothing
written). Weight load ~7 minutes.

| | Here | Cookbook, same cell |
|---|---:|---:|
| Single-stream decode (code, thinking off, 700 tokens) | **39.5 tok/s** (37.6-39.8) | 27.5 tok/s |
| MTP accept_len | 3.55-3.73 of 4 | 2.9-3.5 |
| TTFT, same requests | 0.23 s | — |

The cookbook's figure is TPOT on random prompts, so the protocols differ;
the vLLM int4 AutoRound recipe on the same box does ~66 tok/s.
`--moe-runner-backend marlin` did not boot on stock 0.5.20: the load-time
NVFP4 -> Marlin repack ran out of memory at 114.65 GiB allocated, about half
way through the MoE layers. With patches/gb10_marlin_lean.py (one copy less
per layer, gc between layers) it boots:

| MoE kernels | Single-stream decode | accept_len |
|---|---:|---:|
| cutlass (default, W4A4) | 39.5 tok/s | 3.55-3.73 |
| Marlin (W4A16) | 40.2 tok/s (37.9-40.4) | 3.55-3.70 |

**Marlin changes nothing (+2%)**, so the expert GEMMs are not what holds a
step at ~90 ms (40 tok/s at ~3.6 tokens per step), against ~53 ms for the
int4 vLLM recipe on this box. `FP4_GEMM_BACKEND=marlin` on top: 39.6 tok/s,
the same again.

**Where a decode step goes** (bench/profile_decode.py, Marlin MoE + Marlin
FP4 GEMM, 40 scheduler steps during one code answer; GPU busy 98% of the
span, ~94 ms per step; times per step, under the profiler):

| Kernels | ms/step | Share |
|---|---:|---:|
| BF16 GEMM, cuBLAS `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_*` | ~38 | 40% |
| BF16 GEMV `gemvx` (~9 calls of 2.3 ms; the size of the 248K-row lm_head) | ~20 | 22% |
| MoE experts (Marlin) | ~19 | 20% |
| `_hc_mix` (gated residual) + grouped RMSNorm | ~12 | 13% |
| PLE gather (the patched kernel, 1 call per step) | 5.8 | 6% |
| GDN, QSA, routing, sampling | < 4 | < 4% |

So the step is BF16-bound, not MoE-bound: the layers the checkpoint keeps in
BF16 run on sm80 WMMA 16x16 kernels (SGLang's faster BF16 backends are
SM90/SM100 only), and the BF16 lm_head is read on every draft step. The vLLM
recipe keeps exactly these small (FP8 side layers, int8 lm_head, a 65K draft
vocabulary).

**The two cheap levers, one at a time** (cutlass MoE, single-stream decode):

| | tok/s | accept_len |
|---|---:|---:|
| Profile as above | 39.5 | 3.55-3.73 |
| `DRAFT_VOCAB` (65,536-token draft head, `--speculative-token-map`) | **46.9** (44.8-48.2) | 3.58-3.80 |
| `BLAS=cublaslt` (`TORCH_BLAS_PREFER_CUBLASLT=1`) | 40.1 | 3.58-3.73 |

The draft vocabulary is **+19% with acceptance unchanged** on this English
code prompt, and is now the profile's default. cuBLASLt picks no better BF16
kernels than cuBLAS here. What remains between this and the int4 vLLM recipe
(~66 tok/s) is mostly the BF16 dense layers and the target's full lm_head,
which that recipe carries in FP8 and int8.

**Profile with the draft vocabulary** (cutlass MoE, 40 steps, ~81 ms per step
under the profiler):

| Kernels | ms/step | Share |
|---|---:|---:|
| BF16 GEMM, cuBLAS sm80 WMMA 16x16 (~260 calls per step: GDN in/out, attention qkv/o, shared expert, PLE key/value) | ~39 | 48% |
| MoE experts (flashinfer CUTLASS grouped GEMM) + the BF16 MTP experts | ~21 | 26% |
| PLE gather | 7.4 | 9% |
| `_hc_mix` + grouped RMSNorm | ~14 | — |
| `gemvx` (draft lm_head, now 65K rows) | 6.2 | 8% (was ~20 ms) |
| one `CatArrayBatchedCopy` per step | 3.0 | 4% |

The draft vocabulary took the head from ~20 to ~6 ms per step. The BF16
dense layers are now half the step; `FP8_SIDE=1`
(patches/gb10_fp8_side.py) converts them to FP8 weight-only at load, on
SGLang's FP8 Marlin GEMM.

**Concurrency, with the draft vocabulary** (cap 8, 300-token answers):

| Streams | 1 | 2 | 4 | 8 | 12 (4 queued) |
|---|---:|---:|---:|---:|---:|
| Aggregate tok/s | 46.3 | 76.5 | 121.1 | **188.2** | 159.5 |
| Per stream | 48.0 | 40.1 | 32.0 | 24.8 | 24.7 |

The cookbook's cell reported 71.7 tok/s output at 8; the vLLM recipe on this
box 259 tok/s at 8.

**`FP8_SIDE=1`** (the BF16 side layers to FP8 weight-only on Marlin at load,
draft vocabulary on):

| | Draft vocab only | + FP8_SIDE |
|---|---:|---:|
| Single-stream decode (700-token code answer) | 46.9 | **56.7 tok/s** (53.9-57.6) |
| accept_len | 3.58-3.80 | 3.55-3.65 |
| Aggregate at 1 / 2 / 4 / 8 streams | 46.3 / 76.5 / 121.1 / 188.2 | 48.3 / 85.5 / 129.7 / **206.5** |

+21% single-stream, +4-12% across the sweep; from the cookbook's 39.5 to
56.7 so far (+44%). Tool calling with thinking on passed the owner's own
harder tests on this build.

**HumanEval, thinking off** (164 problems, temperature 0, the parent
project's sandboxed harness, 4 streams), against the int4 AutoRound recipe on
vLLM on a second GB10:

| | This build (NVFP4 + FP8_SIDE, SGLang) | Intel AutoRound int4 (vLLM) |
|---|---:|---:|
| pass@1 | **96.3%** (158/164) | **96.3%** (158/164) |
| Excluding truncated | 96.9% | 96.9% |
| Wrong answers | 32, 89, 132, 140, 145 | 32, 101, 140, 145, 163 |
| Truncated at 2048 tokens | 113 | 132 |
| Harness throughput at 4 streams | 102.7 tok/s | 132.1 tok/s |

Same score; 32, 140 and 145 fail on both, the rest shuffles within the
+-2-problem band greedy decoding shows on this set. FP8_SIDE is now the
profile default.

Thinking on, `reasoning_effort` medium: **98.8% (162/164)**, one runtime
error (67) and one wrong answer (145, after 11K tokens); ~630 tokens per
problem, 97.3 tok/s aggregate at 4 streams. The same as the 27B builds with
medium thinking: at most one real miss, 145 failing everywhere.

**Profile with FP8_SIDE** (40 steps, ~68 ms per step under the profiler, from
~81):

| Kernels | ms/step |
|---|---:|
| MoE experts, flashinfer CUTLASS NVFP4 grouped GEMM | ~18.6 |
| FP8 Marlin GEMMs (the converted side layers) | ~15.8 (was ~39 in BF16) |
| BF16 GEMMs still on sm80 WMMA (~100 calls per step: layers left in BF16) | ~11.8 |
| `_hc_mix` + grouped RMSNorm | ~12.8 |
| PLE gather | 5.7 |
| `gemvx` (draft head, 65K rows) | 5.7 |
| MTP BF16 experts (`Fused_Moe_Kernel_sm80`, `MoeFCGemm`) | 2.2 |

The next item is the ~12 ms of BF16 GEMMs the patch left alone. The boot log
named them: 221 layers converted (5.68 -> 2.84 GiB), 36 left in BF16, all
GDN `in_proj_ba` [96 x 2560], whose width is off Marlin's 64-column tile.
Tiny (0.5 MB each), but alone on cuBLAS they take ~245 us a call, 36 calls
per step: ~8.8 ms. (Before the conversion they rode in one fused GEMM with
in_proj_qkvz.) The patch now zero-pads such a layer to the tile (96 -> 128)
and slices the output back. The MTP draft: 4 layers converted, 0.10 -> 0.05
GiB.

**With `in_proj_ba` padded** (2026-09-28): the boot log has all 257 layers
converted, 36 of them padded, 0 left in BF16; decode **56.3 tok/s**, the same
as 56.7: the ~8.8 ms the cuBLAS calls took became FP8 Marlin time (~19.3 ms
for 236 calls a step, was ~15.8 for 200), not a shorter step. Where a step
goes now (~66 ms under the profiler):

| Kernels | ms/step |
|---|---:|
| FP8 Marlin GEMMs (all side layers) | ~19.3 |
| MoE experts, NVFP4 grouped GEMM | ~18.2 |
| `_hc_mix` + grouped RMSNorm | ~13 |
| lm_heads in BF16: target (verify, once) ~5.4 + draft `gemvx` (~1.2 x ~5) | ~11 |
| PLE gather | 4.4 |
| GDN | 3.3 |
| MTP BF16 experts | 2.2 |

The heads are the last BF16 GEMMs of size, hence `FP8_DRAFT_HEAD` (on by
default: the draft only proposes tokens) and `FP8_HEAD` (the target head,
off until HumanEval passes); see the next section.

**KV pool.** With FP8_SIDE, MTP and the lean load the boot log reads
`max_total_num_tokens=374272 ... available_gpu_mem=16.09 GB` at
mem-fraction 0.85: **374,272 tokens**, four times the ~93K in the
cookbook's cell. The PLE table read in place (no 16 GB host copy) and the
halved side layers are where it comes from.

### FP8 heads (2026-09-28)

`patches/gb10_fp8_side.py` converts the output heads with the same per-channel
FP8 Marlin path, right after the EAGLE worker's `init_lm_head` (the draft's
head is a 65,536-row slice of the target's BF16 head, cut there; the KV pools
are already sized and no CUDA graph is captured yet). Boot log lines:
`FP8 head (draft): lm_head [65536 x 2560] BF16 -> FP8 weight-only (Marlin); 320 MiB -> 160 MiB`
and, with `FP8_HEAD=1`, the same for `(target)`.

| | heads BF16 | draft head FP8 (default) | + target head FP8 (`FP8_HEAD=1`) |
|---|---:|---:|---:|
| decode, tok/s (median of 5) | 56.3 | **60.8 / 61.0** | **63.4** |
| accept_len | 3.6-3.8 | 3.60-3.85 | 3.48-3.73 |
| HumanEval, thinking off | 96.3% (158/164) | (the draft cannot change answers) | **97.0%** (159/164) |
| HumanEval, thinking medium | 98.8% | | **100%** (164/164) |

Draft head: **+8%**, acceptance unchanged (the draft's top token rarely
moves under per-channel FP8). The profile (40 steps, ~61 ms each under the
profiler) has no `gemvx` 65K-row calls left. What is still BF16:

| Kernel | calls/step | avg | ms/step |
|---|---:|---:|---:|
| `wmma ... 128x1` = the target head at verify (M = 4, 1.27 GB) | 1 | 5.4 ms | 5.4 |
| `wmma ... 128x2`, small layers (the boot log now lists which) | ~64 | 45 us | 2.9 |
| `gemvx` | ~2 | 107 us | 0.2 |

The rest of the step: FP8 Marlin side layers ~20.8 ms (239 calls, ~87 us;
~3.2 GB a step, ~150 GB/s), NVFP4 MoE ~18.4, `_hc_mix` ~6.9, GDN ~3.2.

Target head on FP8 (`FP8_HEAD=1`): 63.4 tok/s, +4% over the draft head
alone, 56.3 -> 63.4 (+13%) for both heads. In the profile the 5.4 ms BF16
call is gone and Marlin gained one call a step at ~3.4 ms (the 1.27 GB head
as 0.64 GB of FP8, ~190 GB/s). Still BF16: the ~64 small `wmma 128x2`
calls (~2.9 ms a step).

HumanEval with both heads on FP8: 97.0% without thinking (misses 32, 113,
132, 145, 163; 104.9 tok/s aggregate at 4 streams) and 100% with medium
thinking (~600 tokens a problem, 105.4 tok/s aggregate): not worse than BF16
heads, within the +-1-2 problems greedy runs move. `FP8_HEAD=1` is now the
default.

**What is still BF16** (the new boot-log line): in the target, the MoE
routers `mlp.gate` [512 x 2560] x48 and `shared_expert_gate` [1 x 2560] x48,
the QSA indexer `index_qk_proj` [640 x 2560] x12, the hyper-connection
weights (`input_mix_weight_down/up` [320 x 10240] / [10240 x 320],
`block_inject_weight` [4 x 10240], x48 each for attention and MLP, used by
the `_hc_mix` kernel itself, not a GEMM) and the vision tower; in the MTP
draft the same per layer plus `fc_embedding` / `fc_hidden` [2560 x 2560]
(plain nn.Linear, the ~2 `gemvx` calls a step). The ~64 `wmma 128x2` calls
a step are then the 48 + 12 routers and indexer projections of the target
(M = 4 at verify) plus the draft's. Both select something (experts, attended
tokens), so they stay BF16; `SKINNY_BF16=1` (patches/gb10_skinny.py) runs
them on a Triton GEMM built for a few rows instead: same weights, FP32
accumulation.

**`SKINNY_BF16=1`: no gain.** Decode 61.8 / 62.0 / 62.0 tok/s (three runs)
against 63.4; in the profile the Triton kernel replaced the 2574 WMMA calls
and took 45.0 us a call, the same as cuBLAS's 45.8. Two unrelated kernels
landing on the same time pointed away from the kernel: SGLang runs both
layers on a second stream under CUDA graphs (qwen4_exp.py: the QSA indexer
overlaps the qkv projection; qwen2_moe.py forward_normal_dual_stream: the
router overlaps the shared expert), so they share the GPU with a Marlin GEMM
and their duration is contention, mostly off the critical path. Kernel time
(2614 ms) above busy time (2249 ms) in the same trace is that overlap, ~9 ms
a step. `SKINNY_BF16` stays off. `bench/profile_decode.py` now prints a
"wall" column that splits overlapped time among the kernels in flight, so
such a group shows its real share.

**Wall time per step** (the same two traces re-read with the wall column;
~57 ms a step under the profiler):

| Group | ms/step (wall) | share | note |
|---|---:|---:|---|
| MoE experts, NVFP4 grouped GEMM (+ MTP's BF16 experts ~2.2) | ~21.6 | 38% | |
| FP8 Marlin (side layers + both heads) | ~19.8 | 35% | ~4.4 GB a step: ~220 GB/s, near GB10's bandwidth |
| `_hc_mix` (hyper-connection mix) | ~7.4 | 13% | not overlapped; 13 MB of BF16 weights a call |
| GDN | ~2.2 | 4% | half overlapped |
| routers + indexer (skinny or WMMA) | ~1.5-1.9 | 3% | half overlapped |
| norms / elementwise | ~1.9 | 3% | |

The skinny GEMM saved ~0.4 ms a step of wall time: noise, as measured. The
FP8 Marlin layers read at close to the memory bandwidth, so only fewer bytes
would make them faster. The hyper-connection mix is the one sizable item
that reads BF16 weights on the critical path: `FP8_HC=1`
(patches/gb10_fp8_hc.py) keeps input_mix_weight_down/up in FP8 with a
per-row scale and runs a copy of SGLang's persistent kernel that reads FP8.

**`FP8_HC=1`, first run** (2026-09-28): boot log "97 hyper-connection mixes
to FP8, 1.18 GiB -> 0.60 GiB" (target) and 3 in the MTP draft. Decode
**65.0 / 65.6 tok/s** against 63.4 (+3-3.5%), accept_len 3.42-3.75. The
kernel: 42.9 us a call against 67.0, wall ~4.9 ms a step against ~7.4.
Two things the single-stream number hid, fixed after this run:

- TTFT went 0.17-0.18 -> 0.20-0.21 s: prefill (over 16 rows) took a torch
  path that dequantized each weight to FP32 per call (~200 elementwise and
  copy kernels, 36-76 us each, in the trace). Now each weight is converted
  to BF16 once per call and the row scales multiply the outputs.
- SGLang's kernel stops at 16 rows, and so did the copy: a verify at 8
  requests (32 rows) would have gone to that torch path on every step. The
  copy now takes 16-, 32- and 64-row tiles (the wider ones with a shorter K
  tile and two stages for shared memory; if they fail to build, inputs over
  16 rows fall back to the torch path with a warning).

**After the fixes, full `perf.py`** (two runs; no "kernel failed" warning, so
the 32/64-row tiles built on GB10):

| | run 1 | run 2 |
|---|---:|---:|
| TTFT, short prompt | 135 ms | 134 ms |
| Single-stream decode | **65.9** (ttft 0.17 s) | **65.7** |
| Aggregate at 1 / 2 / 4 / 8 streams | 60.8 / 99.0 / 148.4 / **223.9** | 65.0 / 102.0 / 152.6 / 212.3 |
| Prefill 2K / 8K / 32K / 101K, tok/s | 1458 / 1648 / 1661 / 1499 | 1444 / 1511 / 1447 / 1229 |

TTFT is back to 0.17 s, and the sweep is above the FP8_SIDE-only build at
every level (48.3 / 85.5 / 129.7 / 206.5): +15-26% at 1-4 streams, +3-8% at
8. Prefill varies between the runs (the second ran warmer, 74-76 C at
101K).

### Where the day ended (2026-09-28)

| Step | Single-stream decode |
|---|---:|
| Cookbook cell (stock SGLang 0.5.20, NVFP4, MTP) | 39.5 |
| + 65K draft vocabulary | 46.9 |
| + BF16 side layers on FP8 Marlin (`FP8_SIDE`) | 56.3-56.7 |
| + draft head FP8 (`FP8_DRAFT_HEAD`) | 60.9 |
| + target head FP8 (`FP8_HEAD`; HumanEval 97.0% / 100%) | 63.4 |
| + hyper-connection mix FP8 (`FP8_HC`; HumanEval 97.6% / 99.4%) | **65.7-65.9** |

**HumanEval with `FP8_HC=1`** (on top of FP8 side layers and both heads):
97.6% without thinking (160/164; misses 32, 140, 145, 163; 110.3 tok/s
aggregate at 4 streams) and 99.4% with medium thinking (163/164; only 145,
after 7.4K tokens, the problem every build fails). Not worse than any
earlier step (96.3 -> 97.0 -> 97.6 without thinking: within the +-1-2
problems greedy runs move). `FP8_HC=1` is now the default, and with it
every row of this table.

+67% over the cookbook's cell. What is left is bound by memory bandwidth:
the NVFP4 MoE (~38% of a step) and the FP8 Marlin layers (~35%, ~220 GB/s)
read close to what GB10 delivers, so further gains need fewer bytes (a
different checkpoint, or the MTP's BF16 experts in 4 bits for the draft),
not faster kernels.

### Looked at, postponed: local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (QAD)

A quantization-aware-distilled checkpoint (124 GB on disk, 36 files).
ModelOpt `MIXED_PRECISION` with a `quantized_layers` map, the format SGLang
0.5.20's ModelOptMixedPrecisionConfig reads: routed experts NVFP4 (static
activation scales), attention / GDN / QSA indexer / shared experts MXFP8
(E4M3 + E8M0 scale per 32, dynamic MXFP8 activations), MTP routed experts
W4A16 NVFP4, routers / hyper-connection / PLE projections / lm_head BF16.
The PLE table is NVFP4 (`ple_embedding_dtype: nvfp4`): per shard `weight`
U8 [2.5M, 80] + `weight_scale` F8 [2.5M, 10] and one `weight_scale_2`,
~29 GB against ~51 GB in FP8.

What it would take here: SGLang 0.5.20 loads PLE shards only as FP8 or BF16
(qwen4_exp.py load_qwen4_exp_ple_shard), so gb10_ple_mmap would need an
NVFP4 gather (unpack + block scale + global scale); and MXFP8 W8A8 on sm121
is an open question against the FP8 Marlin path at ~220 GB/s. Expected
speed about today's 65.8 tok/s (plus: 4-bit MTP experts, a smaller table;
unknown: the MXFP8 kernel); the gain would be quality (QAD, shorter
thinking), which HumanEval, near its ceiling here, barely shows. Postponed
until a runtime reads the NVFP4 table, or answers on the RadixArk build
give a reason.

## Ornith-1.5-35B-A3B, one Spark (2026-09-29)

r0b0tlab/Ornith-1.5-35B-A3B-NVFP4-W4A16 (W4A16 on every Linear, BF16 heads
and MTP), SGLang 0.5.20, `models/ornith-1.5-35b.sh` as first written: MoE
marlin, flashinfer attention, FP8 KV, MTP 1 step / 2 draft tokens, cap 16.

| | |
|---|---:|
| TTFT, short prompt | 78 ms |
| Single-stream decode | 46.3 tok/s, **accept_len 1.00** |
| Aggregate at 1 / 2 / 4 / 8 / 16 streams | 45.9 / 87.8 / 150.8 / 234.7 / **326.2** |
| Prefill 2K / 8K / 32K / 101K, tok/s | 7339 / 4721 / 4667 / 3305 (79 C at 101K) |

accept_len 1.00 on every request: no MTP proposal was ever accepted, so this
is plain decoding plus the draft's cost. The card's run of the same
checkpoint (SGLang 0.5.6.post3, triton attention) accepted 1.74. Checked in
0.5.20 and ruled out: the MTP layer is built in BF16 (`_mtp_quant_config`
returns None for a serialized modelopt_fp4 checkpoint), and the draft's BF16
experts run on the Triton MoE runner despite `--moe-runner-backend marlin`
(UnquantizedFusedMoEMethod picks it). The profile's one departure from the
card was flashinfer attention, so `SPEC=mtp` now runs triton, and Ornith
AI's DFlash draft was added (`SPEC=dflash`).

**All four modes** (single-stream decode, greedy, 700 tokens):

| Mode | tok/s | accept_len |
|---|---:|---:|
| no draft (`SPEC=off`) | **72.0** | - |
| MTP 1/1/2, flashinfer attention | 46.3 | 1.00 |
| MTP 1/1/2, triton attention (the card's) | 44.5 | 1.00 |
| DFlash, 8 draft tokens (the card's) | 38.1 | 1.00 |
| DFlash, 12 | 35.1 | 1.00 |

Two unrelated drafts under two attention backends, and not one token
accepted: the fault sits in what they share on the target's side, the
hidden states both drafts are fed from, or the verify pass itself, not in a
draft. Without one the model decodes at 72 tok/s (TTFT 63 ms), the top of
the card's 63-77. `SPEC=off` is the default until that is found.

It is the verify pass: with `SPEC=mtp`, "What is 19*23? Answer with just the
number." (greedy, thinking off) came back as one token repeated,
`òòòò...` for all 32 tokens. Output stuck on one token is how the "GB10
NEXTN collapse" looked on Flash-Next (NaN routing; fixed there in 0.5.20's
moe_fused_gate and route_radix), but this is the W4A16 / Marlin MoE path.

And with no draft the same question returns the same `òòòò...`: the model
itself is broken on this build, and the 72 tok/s above is speed on garbage.
The suspect is in 0.5.20's W4A16 Marlin path. SGLang fuses q/k/v, GDN
in_proj_qkv + in_proj_z and every gate + up into one GEMM, while ModelOpt
quantized each Hugging Face Linear on its own, with its own FP32 global
scale; the Marlin path keeps one per fused layer: the max over the shards
for dense layers (with a warning, "weight_scale_2 differs across fused
parallel layers"), the gate's for experts ("w1_weight_scale_2 must match
w3_weight_scale_2"). The card's SGLang 0.5.6 ran these layers on FlashInfer
CUTLASS instead. `patches/gb10_nvfp4_scales.py` (`NVFP4_SCALES`, default on
in this profile) corrects both exactly: dense output column blocks times
g_i / g_max, and each expert's down-projection global scale times
g_up / g_gate (silu(gate) * up is linear in up).

Result: the same `òòòò...` with the patch, so the scales were not the fault
(the patch stays, off). Not chased further: the profile now defaults to the
original BF16 weights (ornith-ai/Ornith-1.5-35B-A3B, 67 GB) quantized to
FP8 at load (`--quantization fp8`, online FP8 for dense layers and MoE in
0.5.20; the FP8 path is the one Qwen3.8-27B-FP8 runs on this GB10), which
is also the target the DFlash draft was trained against. `WEIGHTS=w4a16`
keeps the r0b0tlab checkpoint for a newer SGLang.

**Ornith AI's FP8 checkpoint** (ornith-ai/Ornith-1.5-35B-A3B-FP8,
quantization read from the checkpoint, MoE backend SGLang's choice): "437",
right. With Ornith AI's DFlash draft:

| | decode tok/s | accept_len | aggregate 1 / 2 / 4 / 8 / 16 streams | TTFT |
|---|---:|---:|---|---:|
| no draft | 39.8 | - | 39.1 / 74.9 / 121.8 / 182.6 / 274.1 | 80 ms |
| **DFlash, 8 draft tokens** | **83.8** | 4.5-5.5 | 71.4 / 114.8 / 158.8 / 245.8 / **345.4** | 102 ms |
| DFlash, 12 | 76.7 | 4.3-5.4 | 62.6 / 95.4 / 142.9 / 209.1 / 305.8 | 108 ms |
| DFlash, 16 | 75.7 | 5.7-6.6 | 66.7 / 94.0 / 129.6 / 184.8 / 266.0 | 112 ms |

DFlash at 8 (the card's block size) wins at every level: x2.1 single-stream,
+26% at 16 streams; 12 and 16 accept more per step and lose more to the
longer verify. It is the profile's default now. Prefill 3.4-6.5K tok/s in
every run, and the GPU reached 83-84 C at the 101K prompt.

Decode profile (DFlash 16, 40 steps, ~65 ms each under the profiler):

| Kernel group | wall ms/step | share |
|---|---:|---:|
| MoE, `fused_moe_kernel` (Triton, FP8), 80 calls, ~390 us each | ~30.8 | 47% |
| BF16 dense GEMMs on sm80 WMMA: 43 calls at ~411 us + 110 at ~62 us | ~24 | 37% |
| FP8 dense GEMMs (CUTLASS) + per-token activation quant | ~2.9 | 4% |
| GDN, norms | ~5.6 | 9% |

The BF16 dense GEMMs are layers the FP8 checkpoint keeps in BF16 (its
ignore list names them); per-channel FP8 on Marlin took the same kind of
layers from ~39 to ~16 ms a step on Flash-Next (`FP8_SIDE`).

The ignore list (compressed-tensors FP8: per-channel weights, dynamic
per-token activations): `lm_head`, routers, `shared_expert_gate`, every
`linear_attn.*` (the GDN projections), MTP, vision. **`FP8_SIDE=1`**
(`GB10_FP8_SIDE_TARGET` pointed at Qwen3_5MoeForConditionalGeneration): 117
GDN projections to FP8 weight-only on Marlin, 2.08 -> 1.04 GiB; "437" right.

| | decode tok/s | accept_len | aggregate 1 / 2 / 4 / 8 / 16 streams |
|---|---:|---:|---|
| FP8 + DFlash 8 | 83.8 | 4.5-5.5 | 71.4 / 114.8 / 158.8 / 245.8 / 345.4 |
| **+ FP8_SIDE** | **97.6** (93.2-102.0) | 4.6-5.5 | **98.0** / 130.2 / 168.4 / 249.0 / **353.0** |

+16% single-stream, +37% at the sweep's first level, +2% at 16 streams
(where the MoE dominates). The boot log also names the MoE's missing
kernel configs: `Using default MoE kernel config ... E=256,N=512,
device_name=NVIDIA_GB10,dtype=fp8_w8a8,per_channel_quant=True.json` (and
`_down`); SGLang 0.5.20 ships a GB10 config only for E=128, N=768.

**HumanEval** (164 problems, greedy): the same with `FP8_SIDE` on and off,
147/164 (89.6%) with thinking off and 161/164 (98.2%) with medium thinking.
Ornith is a reasoning model, and thinking off is not its mode. `FP8_SIDE=1`
is now the profile's default: FP8 checkpoint + DFlash 8 + FP8 GDN
projections, 97.6 tok/s.

**Tuned Triton MoE config** (2026-09-30). SGLang's tuner
(`scripts/patch-moe-tuner.py` first: v0.5.20's tuner fails on FP8
per-channel checkpoints) over batch sizes 1-8192 took ~24 h on the Spark,
25 min to 1 h 50 per size. The result is committed:
`moe-configs/configs/triton_3_7_1/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,per_channel_quant=True.json`;
the boot log shows "Using MoE kernel config from .../moe-configs/..." (and
reuses it for the down projection, which this tuner does not write).

| | decode tok/s | aggregate 1 / 2 / 4 / 8 / 16 streams |
|---|---:|---|
| default MoE config | 97.6 | 98.0 / 130.2 / 168.4 / 249.0 / 353.0 |
| tuned, first run after boot | 93.0 | 91.1 / 116.3 / 159.8 / 236.7 / 344.2 |
| tuned, second run | 96.4 | 92.2 / 120.2 / 167.6 / 246.4 / **361.7** |

No measurable gain: single-stream moves 86-102 tok/s run to run with the
DFlash accept length (4.1-6.3), and 16 streams gained 2.5% at most. The
defaults SGLang picks for FP8 on this GPU (`_use_low_smem_fp8_default`)
were already close; the MoE here reads its weights at close to what the
memory delivers. Two more runs with it: 97.9 / 94.9 tok/s single-stream,
345.4 / 343.1 at 16 streams. Over four runs 93-98 and 343-362 against 97.6
and 353 without it: no gain, so the profile does not load it by default
(`MOE_TUNED=1` does); the file stays in `moe-configs/`.

The 2K prompt's TTFT in section 4 went 0.34-0.36 s in the five untuned runs
and 0.78 / 0.38 / 0.76 / 0.37 s in the four tuned ones. The slow ones: the
first run after boot (2129 tokens, everything compiling) and a 2080-token
prompt, the only one of the nine divisible by 16. Triton compiles a separate
kernel variant when an integer argument is divisible by 16, so the first
such length on a running server pays a one-time JIT (~0.4 s), whatever the
MoE config. A reading, not yet checked: a steady-state server would not show
it twice for the same length.

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
