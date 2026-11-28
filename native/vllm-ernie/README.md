# ERNIE 5.5 Flash Preview on two Sparks (vLLM + vLLM-ERNIE)

Baidu's ERNIE-5.5-Flash-Preview: a 280B MoE (14B active; 37 compressed HCA
layers + 6 MLA-DSA layers, 512 routed experts, mHC), MXFP4 experts with FP8
elsewhere, a bundled 3-layer DSpark draft, 1M context. 154.5 GiB of weights,
so TP=2 + EP=2 across two Sparks over the CX7 link (~80 GiB a node).

**Not served here yet** (2026-11-27: written with one Spark on hand). Everything
below is from the sources named, except this: the image builds on gx10-1cc3
(`2232c1939f7b`, 24.5 GB; torch 2.13.0+cu130, sm 12.1), `vllm_ernie`,
tilelang 0.1.12, humming 0.1.17, sparknet 0.6.0 and flashinfer 0.7.0.post1
import, and with `SPARK_ERNIE_RUNTIME=1` all 13 patch groups of
`spark_ernie_runtime.patch.install()` apply on vLLM 0.32.0 without errors.

Two ways to serve it, on one image:

- `PROFILE=spark` (default): [jjang-ai/ernie-5.5-dgx-spark](https://github.com/jjang-ai/ernie-5.5-dgx-spark)
  at `5b9f1c2` (2026-11-27), run with its own launcher inside the container:
  18 runtime patches loaded through `sitecustomize` (fused router, SM12.1 mHC,
  Humming MXFP4 MoE, packed KV groups, vocab-parallel probabilistic DSpark,
  adaptive verification, packed BF16 LM head, ...). `MODE=tp2-dspark` is its
  `2x-1m-fast` (1M context, DSpark 7, KV pinned at 9 GiB a rank, 8 requests);
  `MODE=tp2-ar` its pure-AR profile.
- `PROFILE=stock`: [baidu/vLLM-ERNIE](https://github.com/baidu/vLLM-ERNIE)
  as released, its README's FP4 command on this layout (FP8 KV, DSpark 7,
  probabilistic drafts, 128K context), FP8 indexer (the MXFP4 one is for
  data-centre Blackwell). The baseline the runtime's numbers are against.

```bash
cp config.local.sh.example config.local.sh   # the pair's addresses, if not ../tensorfold-dsv41's
./build.sh                                   # the image here, then copied to the worker
./serve.sh                                   # PROFILE=spark MODE=tp2-dspark
PROFILE=stock ./serve.sh
./status.sh; ./stop.sh
```

## Weights

The same path on both nodes (`MODEL_DIR`, or `WEIGHTS=` for the defaults):

| `WEIGHTS` | Repository | Licence |
|---|---|---|
| `dealign` (default) | `dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4`, revision `cf6a1332` | DealignAI Community v0.3 |
| `baidu` | `baidu/ERNIE-5.5-Flash-Preview` | Apache-2.0 |

```bash
../scripts/mem-prep.sh idle && hf download dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4 \
  --local-dir /models/dealignai/ERNIE-5.5-Flash-Preview-UNCENSORED-MXFP4
../scripts/mem-prep.sh idle && hf download baidu/ERNIE-5.5-Flash-Preview \
  --local-dir /models/baidu/ERNIE-5.5-Flash-Preview
```

Both are 154.5 GiB with MXFP4 experts. `dealign` is the runtime's measured
checkpoint: a direct weight edit (abliteration) of Baidu's, whose card says
the experts and the DSpark head are byte-identical to the base, with MMLU-500
(thinking off) 85.8% against 87.4% and MMLU-200 (high) level at 92.5%. Its
licence allows personal and internal business use and redistribution, but
**no hosted inference for anyone else**, paid or free (section 3). `baidu`
should run the same runtime (same architecture and expert bytes); not checked.

## The image (`docker/Dockerfile`, `./build.sh`)

`vllm/vllm-openai:v0.32.0` (by digest; vLLM-ERNIE needs exactly 0.32.0) plus,
all with `--no-deps` so the base's vllm and torch stay:
vLLM-ERNIE `ba322a56` (2026-11-24), `tilelang==0.1.12`,
`humming-kernels==0.1.17` (the versions the runtime's audit lists), sparknet
`v0.6.0` (christopherowen/dgx-spark-networking), and the runtime checkout at
`/opt/spark-ernie`, inactive unless `SPARK_ERNIE_RUNTIME=1` (its launcher sets
it). The audit's torch is 2.13.0; the base image's is whatever vLLM 0.32.0
shipped with.

## What to watch

- **Two edits the runtime does not ship.** Its own audit
  (`docs/ARCHITECTURE_AND_FLAGS_AUDIT.md`, item 2) found two files changed in
  the author's venv: `vllm_ernie/.../nvidia/attention.py` (+24 lines: pick
  the SM120 FlashInfer backends on capability 12) and vLLM's
  `flashinfer_mla_sparse_sm120.py` (+2 lines: pass `sinks`). Neither is in the
  repository; vLLM-ERNIE's current code may already pick the backend. The
  first boot shows it.
- **sparknet is off** (`SPARKNET=0`): its one-shot collectives need the pair's
  peer-HCA and subnet maps, which the runtime's `spark-ernie discover` writes;
  here they go in `config.local.sh` (`config.local.sh.example`). Plain NCCL
  over RoCE meanwhile.
- **Memory.** The runtime's 2x numbers: 79.7 GiB of weights a node plus 9 GiB
  KV (1,143,108 tokens); contexts past ~400-500K ran out of unified memory on
  rank 0. Its tp2-dspark asks for `gpu_memory_utilization` 0.86, above this
  project's usual 0.7-0.8; watch `free` on both nodes (`./status.sh`).
- **Their numbers** (2x, T=1.0, their benchmarks; nobody else has reproduced
  them yet): code 84.4 tok/s, prose 42.6, 32K prefill ~2,000 tok/s, 306.5
  tok/s at 16 streams; stock 2x 18.4-20.4.
- Thinking is on by default; requests turn it off with
  `chat_template_kwargs: {"enable_thinking": false}`, which `../bench/` sends.
