#!/usr/bin/env bash
# Launcher for scripts/serve-sglang.sh: every knob of this recipe, with why it
# is set the way it is. Edit the values for your machine (or keep your real
# settings as a local-only commit on top of this). Anything commented out or
# left unset falls back to the script's defaults.
#
#   ./serve.sh            start the server in the foreground (Ctrl-C stops it)
#   ./serve.sh install    install SGLang into the venv below (not the weights)
#   ./serve.sh manifest   print the resolved install (scripts/build-manifest.sh)
#
# The systemd unit from scripts/install-service.sh runs this file, so after
# an edit: sudo systemctl restart gb10-sglang
cd "$(dirname "$0")" || exit 1

# Where the venv lives: $GB10_WORKDIR/venv. `./serve.sh install` creates it;
# it also carries the hf CLI and what HumanEval needs.
export GB10_WORKDIR="$HOME/spark"

# Required: the target checkpoint and the DFlash2 draft, downloaded once with
# a pinned --revision (README, "Weights"). Targets measured here:
#   Qwen/Qwen3.8-27B-FP8 @ 017b9c7a (default): Qwen's own checkpoint and the
#     more accurate one (HumanEval 97.6% with thinking off, vs 93.9%).
#   RadixArk/Qwen3.8-27B-NVFP4 @ 554ebba9: ~40% faster single-stream, 2.5x
#     faster prefill (results/RESULTS.md, "FP8 target").
# Draft: z-lab/Qwen3.8-27B-DFlash2 @ 50307d4c (incoai/Qwen3.8-27B-DFlash2, the
# SGLang cookbook's name, mirrors the same weights).
# Already in an HF cache? Point at the snapshot directory instead of copying,
# e.g. ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7a...
# Either way the server prints the revision it found at startup.
export MODEL_DIR=/models/Qwen3.8-27B-FP8
export DRAFT_DIR=/models/Qwen3.8-27B-DFlash2

# Hub access while serving: 1 = offline (HF_HUB_OFFLINE=1). The weights are
# local, so a lookup should never happen; offline makes one fail loudly
# instead of quietly downloading. 0 if the boot fails asking for the network.
export HF_OFFLINE=1

# Draft tokens per step, the largest single-stream lever. The optima diverge:
# 10 wins aggregate throughput (435 vs 385 tok/s at 16 streams on NVFP4),
# 16 wins a single stream (78.6 vs 65.2 tok/s, +28% over the default 8).
# Past 16 accept_len falls and both get worse. Any value other than the
# draft's block size (8) logs "DFLASH block size mismatch" at boot; harmless.
export DRAFT_TOKENS=10

# Concurrent requests. Concurrency on this hybrid model is bought with GDN
# state, not KV: the script sets --max-mamba-cache-size to 5 x this (4 slots
# + 1 for the DFlash2 verify; SGLang clamps the cap to pool / 5 otherwise) and
# captures decode CUDA graphs up to this batch size. Each slot costs 0.196 GB
# (bf16 state), so 32 -> 160 slots -> ~31 GB. 32 took NVFP4 to 572 tok/s
# aggregate; 48 (~47 GB of state) failed to boot on 128 GB. For one
# interactive user, 4 gives that memory back to the KV pool.
export MAX_RUNNING=32
# Override the derived values only to experiment:
#export MAMBA_CACHE=160
#export CUDA_GRAPH_BS=32

# Memory fraction of the unified 128 GB that SGLang may take for weights, GDN
# state and KV. Not a speed lever: 0.82 / 0.85 / 0.90 measure the same. It is
# a stability one: at 0.85 the host keeps ~8 GB, exactly DGX OS earlyoom's
# threshold, and earlyoom SIGTERMs the scheduler during boot or a long
# prefill (exit code -15, no traceback; journalctl -u earlyoom). The SGLang
# cookbook lost 15 of 48 GB10 configs at 0.85, none at 0.80. 0.85 ran fine
# here under Docker. Above 0.85 add --max-total-tokens 1048576 to EXTRA_ARGS:
# a bigger KV pool goes unused and the lost headroom cost 18% at 32 streams.
export MEM_FRACTION=0.80

# Prefill chunk. 8192 is what everything in results/ was measured with and
# favours prefill throughput; the cookbook's 2048 keeps decode smoother
# while long prompts are being prefilled alongside it.
export CHUNKED_PREFILL=8192
# Prefill CUDA graphs: 0 = off, as measured here; 1 = on, as in the cookbook's
# GB10 cell. Not benchmarked here.
export PREFILL_CUDA_GRAPH=0

# Context: 262144 is the model's native length.
export CONTEXT_LENGTH=262144

# API. Clients see the model as SERVED_MODEL_NAME; bench/ reads it from
# /v1/models. HOST=127.0.0.1 keeps the port on this machine only (e.g.
# behind a reverse proxy); 0.0.0.0 publishes it on every interface.
export PORT=8888
export HOST=0.0.0.0
export SERVED_MODEL_NAME=qwen3.8-27b-sglang

# API key (Bearer token) for the OpenAI- and Anthropic-compatible endpoints:
#   non-empty -> clients must send "Authorization: Bearer <key>" (--api-key)
#   empty     -> no auth (closed network)
# Generate: openssl rand -hex 32. To keep the secret out of git, read it from
# a file outside the repo instead:
#   export API_KEY="$(cat "$HOME/.qwen-api-key")"
# bench/ then needs GB10_API_KEY set to the same value.
export API_KEY=''

# CPU pinning: GB10's ten Cortex-X5 cores (the A725 efficiency cores are
# 0-4 and 10-14), so the scheduler and tokenizer stay off the slow cores.
# Empty = no pinning.
export CPUSET=5-9,15-19

# Any other SGLang flags, appended last, so they override the script's own.
export EXTRA_ARGS=''

case "${1:-}" in
  '')       exec scripts/serve-sglang.sh ;;
  install)  exec scripts/01-install.sh ;;
  manifest) exec scripts/build-manifest.sh ;;
  *) echo "usage: $0 [install | manifest]" >&2; exit 2 ;;
esac
