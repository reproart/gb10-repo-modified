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
| `MTP_TOKENS` | 4 | MTP draft tokens, 0 = off |
| `MAX_MODEL_LEN` | 262144 | context (1048576 with `YARN=1`) |
| `YARN`, `YARN_FACTOR` | 0, 4 | YaRN past the native 262144: the checkpoint's own RoPE settings with `rope_type` yarn, via `--hf-overrides` |
| `MAX_BATCHED_TOKENS` | 4096 | prefill chunk |
| `GPU_UTIL` | 0.8 / 0.9 | `--gpu-memory-utilization`; 0.9 with `PLE_OFFLOAD=none` |
| `KV_CACHE_DTYPE` | fp8 | |
| `PLE_OFFLOAD` | disk | n-gram table: `disk` (io_uring, the card's Spark setting), `cpu` (host memory, the card's RTX PRO setting), `none` (in GPU memory) |
| `PORT`, `HOST`, `API_KEY` | 8888, 0.0.0.0, none | as for `../serve.sh` |
| `PROFILE`, `PROFILE_DIR` | 0, /tmp/gb10-profile | vLLM's profiler endpoints and trace directory, for `../bench/profile_decode.py` |
| `EXTRA_ARGS`, `EXTRA_ENV` | | more `vllm serve` flags; more `NAME=value` for the container |

`PLE_OFFLOAD=none` puts the table (NVFP4, ~29 GB) inside the GPU budget
with the weights, so less KV is left; GPU_UTIL then defaults to 0.9
(~12 GiB left to the OS, close to earlyoom's margin), and `serve.sh` wants
that much free memory before it starts. `none` sets neither PLE variable and
leaves the container's default; `./show-env.sh PLE` prints which PLE
variables the container's vLLM defines and their defaults, to confirm.

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
logs the table setting as `Resolved Engram configuration: ... cpu_offload=True,
table_memory='device'` with `PLE_OFFLOAD=disk`: still to be read against
`./show-env.sh` before trusting what each mode does.

Notes: the container runs as root, so `~/.cache/gb10-vllm-b12x` (kernel and
compile caches, kept between runs) ends up root-owned; the weights are
mounted read-only (`MODEL_RO=0` if the b12x loader wants to write beside
them). First measurements: results/RESULTS.md, "Qwen3.8-Flash-Next-FP4-Local".
