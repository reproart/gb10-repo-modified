# GLM-5.3-Flash GSQ-RCO GGUFs on llama.cpp (one or two Sparks)

[pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF](https://huggingface.co/pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF):
GLM-5.3-Flash (320B, multimodal) in non-uniform 3.0 / 3.5-bit GGUFs. It
needs llama.cpp PR 27773 (the `glm5-next` architecture), so this is a
separate stack from the SGLang one in the rest of `native/`, with its own
build. Stop the SGLang server before starting this one: a Spark has one
pool of memory.

| File | Size | One Spark (~115 GiB free) | Two Sparks |
|---|---:|---|---|
| `GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf` | 109.4 GiB | text only, short context, at the edge | yes |
| `GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf` | 127.7 GiB | no: more than the machine has | yes (the default) |
| `GLM-5.3-Flash-mmproj-BF16.gguf` (vision) | 1.08 GiB | not with the 3.0-bit | yes |

## Files

| | |
|---|---|
| `build.sh` | clones llama.cpp at the card's commit (`de25343`, PR 27773), applies `patches/native-f32-mmf.patch`, builds `llama-server`, `ggml-rpc-server` and `llama-cli` for sm_121 with the RPC backend |
| `serve-single.sh` | one Spark, 3.0-bit, text, 8K context, 1 slot: does the build and the file work at all |
| `rpc-worker.sh` | the second Spark: offers its GPU over the direct link |
| `serve-dual.sh` | the first Spark: llama-server over both GPUs, 3.5-bit + vision, 32K shared by 4 slots |
| `config.sh` | paths, port (8889), API key, strict-F32 switch |

The patch only matters with `GGML_CUDA_MMF_F32_DISABLE=1`: it keeps F32
matrix products off TF32, as in the card's evaluation. The scripts set it
(and `NVIDIA_TF32_OVERRIDE=0`) by default; `STRICT_F32=0` turns both off.

## 1. One Spark: does it work

```bash
cd native/llamacpp
./build.sh                                    # ~10-20 min
hf download pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf \
  GLM-5.3-Flash-mmproj-BF16.gguf --local-dir /models/GLM-5.3-Flash-GSQ-RCO-GGUF
./serve-single.sh
```

It prints the memory budget first and refuses when less than
`HEADROOM_GIB` (6) would be left after the weights. Ways to get there: no
other GPU work, the desktop off (`sudo systemctl isolate multi-user.target`;
`graphical.target` brings it back), `CTX=4096`. `FORCE=1` tries anyway. If
earlyoom kills the server (`journalctl -u earlyoom`), the 3.0-bit file does
not fit one Spark; don't disable earlyoom to get past that, the kernel's own
OOM killer is worse.

Check, then measure:

```bash
curl -s localhost:8889/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 19*23? Answer with the number."}],"max_tokens":64}'
GB10_BASE_URL=http://127.0.0.1:8889/v1 python3 ../bench/perf.py --only decode
```

## 2. Two Sparks

### The link

One QSFP cable between the two Sparks' ConnectX-7 ports, a static address
on each end. Find the port with the cable (`ip -br link`: state UP), then on
the first Spark, e.g. `/etc/netplan/60-spark-link.yaml`:

```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:          # the cabled port, from ip -br link
      addresses: [10.10.10.1/24]
      mtu: 9000
```

and the same with `10.10.10.2/24` on the second; `sudo netplan apply` on
both, then `ping 10.10.10.2`. MTU 9000 on both ends or on neither.
Optional: `iperf3 -s` on one, `iperf3 -c 10.10.10.2 -P 4` on the other.

**A second cable, or bonding the two: no gain here.** llama.cpp RPC keeps
one TCP connection per remote GPU, and a bond spreads connections, not the
packets of one connection, across its links. And what crosses the link per
token is a hidden-state vector per request at the split point, kilobytes:
decode waits on the round trip (tens of microseconds), not on bandwidth.
Bandwidth matters once, when the worker's half of the weights (~65 GB) goes
over at the first start; the worker's RPC cache makes later starts send only
hashes. One cable is enough; the second port can stay free (or carry
something else).

### Start

Build on both (`./build.sh`: the same commit, the RPC protocol is not
stable across versions). The GGUFs are needed on the first Spark only.

```bash
# second Spark
BIND=10.10.10.2 ./rpc-worker.sh

# first Spark
hf download pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF GLM-5.3-Flash-GSQ-RCO-3.5bit.gguf \
  GLM-5.3-Flash-mmproj-BF16.gguf --local-dir /models/GLM-5.3-Flash-GSQ-RCO-GGUF
WORKER=10.10.10.2 ./serve-dual.sh
```

`rpc-worker.sh` listens on the link's address only and refuses `0.0.0.0`:
the RPC server has no authentication. Its cache is
`~/.cache/llama.cpp/rpc` (~65 GB); delete it to free the disk.

Knobs: `QUANT=3.0bit`, `CTX`, `NP` (slots sharing the context),
`MMPROJ=0`, `CACHE_RAM` (host-side prompt cache, MiB), `TENSOR_SPLIT=55,45`
(head first; the default splits by free memory), `SPLIT_MODE=tensor`,
`EXTRA_ARGS`.

### What to expect

Two Sparks buy memory, not speed. With `--split-mode layer` (the default)
each machine holds half of the layers and a token runs through them one
after the other, so single-stream decode is about what one Spark would give
if the model fit, minus two hops per token. Prefill and several requests at
once do better: llama.cpp pipelines micro-batches through both GPUs (the
RPC backend supports the async compute and events that needs).
`SPLIT_MODE=tensor` splits every layer across both GPUs so they read their
halves at the same time, the only mode that can beat one Spark on decode,
paying two exchanges per layer over the link; it is marked experimental in
llama.cpp and untested over RPC here. Compare the two with `perf.py`.

The card's quality figures: MMLU-Pro 60.55% (3.5-bit) and 60.00% (3.0-bit)
against 61.95% for Q8_0, perplexity +4.5% / +9.1%. Its IFEval and GSM8K rows
are 16 and 8 items.
