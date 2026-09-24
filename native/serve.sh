#!/usr/bin/env bash
# Launcher: the machine's settings, and which model profile to serve. What
# belongs to a model (weights, SGLang version, its flags) lives in
# models/<profile>.sh; edit those for the model, this file for the machine
# (or keep your real settings as a local-only commit on top of both).
#
#   ./serve.sh                        serve the default PROFILE below
#   ./serve.sh gemma4-31b             serve another profile (models/gemma4-31b.sh)
#   ./serve.sh [profile] install      install the SGLang version that profile uses
#   ./serve.sh [profile] manifest     print the resolved install (scripts/build-manifest.sh)
#
# One model at a time: a second server on the same port refuses to start.
# The systemd unit from scripts/install-service.sh runs this file with one
# profile, so after an edit: sudo systemctl restart gb10-sglang
cd "$(dirname "$0")" || exit 1

# The profile ./serve.sh serves when none is named. Profiles: ls models/
export PROFILE=qwen3.8-27b

# Where the venvs live. `./serve.sh install` creates one per SGLang version,
# $GB10_WORKDIR/venv-sglang-<version>, and points $GB10_WORKDIR/venv (with the
# hf CLI in it) at the one installed last. Profiles on the same version share
# one venv.
export GB10_WORKDIR="$HOME/spark"

# Hub access while serving: 1 = offline (HF_HUB_OFFLINE=1). The weights are
# local, so a lookup should never happen; offline makes one fail loudly
# instead of quietly downloading. 0 if the boot fails asking for the network.
export HF_OFFLINE=1

# Parallel compiler jobs for kernels built on first use (FlashInfer JIT, e.g.
# the CUTLASS FP4 GEMMs, while CUDA graphs are captured). They compile after
# the server has taken its memory fraction, and each nvcc on those templates
# takes several GB: left unset, ninja runs one per core, ~22 at once, and the
# kernel kills them (exit 137, "Capture cuda graph failed: Ninja build
# failed"). 2 fits in what is left at 0.80; the first boot is slower, later
# ones load the cached result (~/.cache/sglang/.cache/flashinfer).
export JIT_JOBS=2

# API. Clients see the model under the profile's SERVED_MODEL_NAME; bench/
# reads it from /v1/models. HOST=127.0.0.1 keeps the port on this machine
# only (e.g. behind a reverse proxy); 0.0.0.0 publishes it on every interface.
export PORT=8888
export HOST=0.0.0.0

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

cmd=serve
for arg in "$@"; do
  case "$arg" in
    install|manifest) cmd="$arg" ;;
    -*|'') echo "usage: $0 [profile] [install | manifest]" >&2; exit 2 ;;
    *) PROFILE="$arg" ;;
  esac
done
if [ ! -f "models/$PROFILE.sh" ]; then
  echo "no profile models/$PROFILE.sh; profiles: $(cd models && ls -- *.sh | sed 's/\.sh$//' | paste -sd' ')" >&2
  exit 2
fi

case "$cmd" in
  serve)    exec scripts/serve-sglang.sh ;;
  install)  exec scripts/01-install.sh ;;
  manifest) exec scripts/build-manifest.sh ;;
esac
