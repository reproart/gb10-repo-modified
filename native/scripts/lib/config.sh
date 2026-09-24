# shellcheck shell=bash disable=SC2034  # variables are used by the scripts that source this
# Shared defaults for the scripts in scripts/. Source it, don't run it.
#
# Precedence: variables already in the environment > the model profile
# (models/$PROFILE.sh, whose values are all "${VAR:-default}") > the defaults
# here. ./serve.sh in the repo root exports the machine settings and PROFILE.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The model profile: its weights, SGLang version and flags (model_args).
PROFILE="${PROFILE:-qwen3.8-27b}"
if [ ! -f "$ROOT/models/$PROFILE.sh" ]; then
  echo "no profile models/$PROFILE.sh; profiles: $(cd "$ROOT/models" && ls -- *.sh | sed 's/\.sh$//' | paste -sd' ')" >&2
  exit 2
fi
# shellcheck source=/dev/null
. "$ROOT/models/$PROFILE.sh"

GB10_WORKDIR="${GB10_WORKDIR:-$HOME/spark}"
# One venv per SGLang version, so installing a new one never touches the one
# that works. $GB10_WORKDIR/venv is a symlink to the last one installed.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.20}"
SGLANG_INDEX="${SGLANG_INDEX:-}"
VENV="$GB10_WORKDIR/venv-sglang-$SGLANG_VERSION"

# Local checkpoint directories (README, "Weights"): an `hf download
# --local-dir` target, or a snapshot directory inside an HF cache. The
# profile sets them; DRAFT_DIR is empty for a model served without a draft.
MODEL_DIR="${MODEL_DIR:?the profile must set MODEL_DIR}"
DRAFT_DIR="${DRAFT_DIR:-}"

# The Hub commit a checkpoint directory holds, or "unknown". A cache snapshot
# is named after it; `hf download --local-dir` records it per file in
# .cache/huggingface/download/<file>.metadata (first line).
revision_of() {
  local dir="${1%/}" meta
  if [[ "$dir" =~ /snapshots/([0-9a-f]{40})$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  meta="$dir/.cache/huggingface/download/config.json.metadata"
  if [ -r "$meta" ]; then
    head -1 "$meta"
  else
    echo unknown
  fi
}

# Triton compiles a small C launcher at first use, so the server needs the
# interpreter's headers (Python.h) and a C compiler. The official Docker image
# ships both; DGX OS lacks the headers, and the failure then shows up mid-boot
# as "fatal error: Python.h" and "Triton is not supported on current
# platform". Prints one line per missing piece; returns 1 if any.
check_build_deps() {  # $1 = python interpreter
  local py="$1" inc ver ok=0
  inc="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null)"
  ver="$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null)"
  if [ ! -f "$inc/Python.h" ]; then
    echo "  Python.h (for Triton): sudo apt install python${ver}-dev"; ok=1
  fi
  if ! command -v "${CC:-cc}" >/dev/null && ! command -v gcc >/dev/null; then
    echo "  a C compiler (for Triton): sudo apt install build-essential"; ok=1
  fi
  return $ok
}
