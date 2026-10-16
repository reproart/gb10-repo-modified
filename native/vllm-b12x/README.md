# Qwen3.8-Flash-Next-FP4-Local on vLLM (b12x container)

`nvidia/Qwen3.8-Flash-Next-FP4-Local` is NVIDIA's quantization-aware-distilled
(QAD) NVFP4 build of Qwen3.8-Flash-Next: routed experts NVFP4, attention and
shared experts MXFP8, MTP experts NVFP4, the n-gram (PLE) table NVFP4; ~96 GiB.
Against BF16 on its card: GPQA Diamond 91.5% vs 92.2%, ToolEvalBench 88.7% vs
89.4%. The model profile `qwen3.8-flash-next` serves RadixArk's PTQ build on
SGLang with our FP8 patches; this one is the candidate on answer quality.

SGLang cannot load it (0.5.21 checked): its NVFP4 scale factors use a new
lossless compression, and `qwen4_exp.py` reads PLE shards only as FP8 or BF16.
Upstream vLLM cannot either. The card's runtime is a community container,
`eugr/spark-vllm-b12x` (vLLM with the b12x kernels and loader), so this
directory runs exactly that, in Docker, with the card's DGX Spark command.

```bash
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir /models/nvidia/Qwen3.8-Flash-Next-FP4-Local
./pull.sh       # the container (~9 GB), pinned by digest in image.lock: commit that file
./serve.sh      # foreground, Ctrl-C stops; http://<host>:8888/v1, "qwen3.8-flash-next-fp4-local"
```

Stop the SGLang server first (one pool of memory); `serve.sh` refuses
otherwise. Knobs (config.sh, each overridable from the shell):

| Variable | Default | |
|---|---|---|
| `MAX_NUM_SEQS` | 8 | concurrent requests |
| `MTP_TOKENS` | 3 (0 with `PLE_OFFLOAD=none`) | MTP draft tokens, 0 = off; the card's 4 ties on one stream and is 4-9% slower at 4-8 requests |
| `MAX_MODEL_LEN` | 262144 | context (1048576 with `YARN=1`, 65536 with `PLE_OFFLOAD=none`) |
| `YARN`, `YARN_FACTOR` | 0, 4 | YaRN past the native 262144: the checkpoint's own RoPE settings with `rope_type` yarn, via `--hf-overrides` |
| `MAX_BATCHED_TOKENS` | 4096 | prefill chunk |
| `GPU_UTIL` | 0.8 / 0.6 / 0.85 | `--gpu-memory-utilization` for `PLE_OFFLOAD` disk / cpu / none |
| `KV_CACHE_DTYPE` | fp8 | |
| `PLE_OFFLOAD` | disk | n-gram table: `disk` (io_uring, the card's Spark setting), `cpu` (pinned host RAM, the card's RTX PRO setup; on a Spark it eats into the same memory), `none` (in the GPU budget) |
| `PORT`, `HOST`, `API_KEY` | 8888, 0.0.0.0, none | as for `../serve.sh` |
| `PROFILE`, `PROFILE_DIR` | 0, /tmp/gb10-profile | vLLM's profiler endpoints and trace directory, for `../bench/profile_decode.py` (no Python stacks, no summary table: those blocked the engine for minutes) |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `vllm serve` flags; more `NAME=value` for the container, replacing a card value of the same name (`EXTRA_ENV=VLLM_B12X_MOE_FP4_FORCE_A16=0`) |

How the container controls the table (`./show-env.sh`): `VLLM_PLE_CPU_OFFLOAD`
(default 1, offloaded) and `VLLM_PLE_TABLE_MEMORY` ram | disk | shared. So
`none` sets `VLLM_PLE_CPU_OFFLOAD=0` and loads the table with the weights
inside the budget: 100.5 GB for the target (the table ~27 GiB) and 4.1 GB
for the MTP head, ~97.5 GiB of a ~119.7 GiB Spark. At GPU_UTIL 0.85 with MTP
4 and 262144 the kernel OOM-killed vLLM ~10 minutes after the weights
loaded, so `none` defaults to MTP off and a 65536 context: a mode to measure
what the disk reads cost, not one to work in. `cpu` pins it in host RAM, which on a Spark is the
same memory, so the budget drops to 0.6: there for comparison. The boot line
`Resolved Engram configuration: ... table_memory='device'` is DeepSeek V4.1's
Engram table, not this one. With `disk`, `free -g` at 0.8 showed 102 GiB
used, 19 available and 2 in page cache: the table is neither in RAM nor in
the page cache (io_uring reads around it).

YaRN is static (every request is scaled, short ones too), so `YARN=1` only
for sessions that need more than 262144 tokens; `YARN=1 MAX_MODEL_LEN=524288`
for a shorter cap. One 1M-token request takes most of the fp8 KV pool.

Pinning: `pull.sh` without `image.lock` pulls `:latest`, compares its digest
with the `03a956544e9f...` Docker Hub showed for linux/arm64 on 2026-10-16
(`EXPECT_PREFIX`) and writes `image.lock`; with it, pulls that digest only.
`rm image.lock && ./pull.sh` moves to a newer build on purpose. If the note
about a different digest appears, the image may be multi-platform (Hub lists
the arm64 manifest, docker the index): `docker buildx imagetools inspect
eugr/spark-vllm-b12x:latest` shows both.

Measuring, as for the SGLang profile: `python3 ../bench/perf.py` (vLLM's
metrics give the MTP acceptance), HumanEval, and `bench/answers.py collect`
on the same questions for both builds, then `answers.py compare`. Worth
watching besides tok/s: how many tokens the thinking takes (QAD trained the
end-of-thinking decision).

First boot here (2026-10-16, image sha256:03a956544e9f...3595, model
revision b9ce18a6, defaults): 15.5 minutes to ready — weights 2 min (71.7 GB
read for the target, 4.1 GB for MTP), then b12x kernel autotuning in two
phases of ~4.7 min each (6,429 and 1,504 candidates, nothing cached yet) and
45 s of JIT warmup. Weights take 70.3 GiB in the pool; at GPU_UTIL 0.8 that
leaves 18.6 GiB of fp8 KV, 987,930 tokens (3.77 requests of 262,144). vLLM
logs `Resolved Engram configuration: ... table_memory='device'`, which is
DeepSeek V4.1's table, not this model's (see above). Second boot: 4 minutes
(autotuning cached).

Notes: the container runs as root, so `~/.cache/gb10-vllm-b12x` (kernel and
compile caches, kept between runs) ends up root-owned; the weights are
mounted read-only (`MODEL_RO=0` if the b12x loader wants to write beside
them). First measurements: results/RESULTS.md, "Qwen3.8-Flash-Next-FP4-Local".
