#!/usr/bin/env bash
# End-to-end HumanEval: generate -> execute (sandboxed) -> report.
#
#   ./scripts/run-humaneval.sh            # thinking off
#   ./scripts/run-humaneval.sh think      # thinking on, xhigh (slow: ~20 min)
#   ./scripts/run-humaneval.sh think-medium  # thinking at reasoning_effort=medium
#
# Python and the hf CLI come from the venv `./serve.sh install` creates
# ($GB10_WORKDIR/venv, default ~/spark/venv, which has huggingface_hub, pandas
# and pyarrow). Override with GB10_PYTHON / GB10_HF.
#
# This is the one place the recipe still uses Docker, and it needs no GPU:
# execution happens inside a `--network none` container because this runs
# model-generated code. The repo is mounted read-only, and the container is
# capped on memory, processes and CPU: GB10 memory is unified and the serving
# engine already holds most of it, so a runaway candidate must hit its own
# limit rather than push the host (and the model server) into OOM.
set -euo pipefail

MODE="${1:-nothink}"
case "$MODE" in
  nothink|think|think-medium) ;;
  *) echo "unknown mode '$MODE' (nothink | think | think-medium)" >&2; exit 2 ;;
esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${GB10_WORKDIR:-$HOME/spark}/venv"
if [ -x "$VENV/bin/python3" ]; then DEFAULT_PY="$VENV/bin/python3"; else DEFAULT_PY=python3; fi
if [ -x "$VENV/bin/hf" ]; then DEFAULT_HF="$VENV/bin/hf"; else DEFAULT_HF=hf; fi
PY="${GB10_PYTHON:-$DEFAULT_PY}"
HF="${GB10_HF:-$DEFAULT_HF}"
SANDBOX_MEM="${GB10_SANDBOX_MEMORY:-2g}"

"$PY" -c "import pandas, pyarrow" 2>/dev/null || {
  echo "$PY lacks pandas/pyarrow. Run ./serve.sh install (creates $VENV)" >&2
  echo "or point GB10_PYTHON at an interpreter that has them." >&2
  exit 1; }

command -v docker >/dev/null || {
  echo "docker not found: it is needed to sandbox the generated code (no GPU required)" >&2
  exit 1; }

mkdir -p "$ROOT/results" "$ROOT/data"

if ! find "$ROOT/data/humaneval" -name '*.parquet' 2>/dev/null | grep -q .; then
  echo "== fetching HumanEval =="
  command -v "$HF" >/dev/null || {
    echo "hf CLI not found ($HF). Run ./serve.sh install or set GB10_HF." >&2
    exit 1; }
  "$HF" download openai/openai_humaneval --repo-type dataset \
    --local-dir "$ROOT/data/humaneval"
fi

echo "== generating ($MODE) =="
( cd "$ROOT/bench/humaneval" && "$PY" generate.py "$MODE" )

echo
echo "== executing in sandbox (memory $SANDBOX_MEM) =="
docker run --rm --network none \
  --memory "$SANDBOX_MEM" --memory-swap "$SANDBOX_MEM" --pids-limit 256 --cpus 4 \
  --security-opt no-new-privileges \
  -v "$ROOT:/w:ro" -w /tmp python:3.12-slim \
  python /w/bench/humaneval/execute.py "/w/results/gen_${MODE}.json" \
  > "$ROOT/results/exec_${MODE}.json"

echo
echo "== report =="
"$PY" "$ROOT/bench/humaneval/report.py" \
  "$ROOT/results/exec_${MODE}.json" "$ROOT/results/gen_${MODE}.json"
