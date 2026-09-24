# shellcheck shell=bash disable=SC2034  # variables are used by the scripts that source this
# Shared defaults for the scripts in scripts/. Source it, don't run it.
#
# Variables already in the environment win: ./serve.sh in the repo root
# exports the settings, and anything it leaves unset gets the default here.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GB10_WORKDIR="${GB10_WORKDIR:-$HOME/spark}"
VENV="$GB10_WORKDIR/venv"

# Target checkpoint. fp8 is Qwen's own checkpoint; nvfp4 is the one behind
# every table in results/RESULTS.md before the FP8 section. Any other repo:
# TARGET=custom TARGET_PATH=<repo> TARGET_REV=<commit sha>.
TARGET="${TARGET:-fp8}"
case "$TARGET" in
  fp8)    _path=Qwen/Qwen3.8-27B-FP8;        _rev=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a ;;
  nvfp4)  _path=RadixArk/Qwen3.8-27B-NVFP4;  _rev=554ebba9b5f1b79dc11246341960360e6ef05ef4 ;;
  custom) _path=""; _rev="" ;;
  *) echo "TARGET must be fp8, nvfp4 or custom, got '$TARGET'" >&2; exit 2 ;;
esac
TARGET_PATH="${TARGET_PATH:-$_path}"
TARGET_REV="${TARGET_REV:-$_rev}"
[ -n "$TARGET_PATH" ] || { echo "TARGET=custom needs TARGET_PATH" >&2; exit 2; }
[ -n "$TARGET_REV" ] || echo "note: no TARGET_REV - the repo's mutable default branch will load" >&2

# DFlash2 draft. incoai/Qwen3.8-27B-DFlash2 (the SGLang cookbook's name) is
# a mirror of the same weights.
DRAFT_PATH="${DRAFT_PATH:-z-lab/Qwen3.8-27B-DFlash2}"
DRAFT_REV="${DRAFT_REV:-50307d4c4cde6860d4eee73e2547cd786fe8e8a4}"
unset _path _rev
