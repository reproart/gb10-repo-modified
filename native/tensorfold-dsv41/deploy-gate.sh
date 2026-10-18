#!/usr/bin/env bash
# Deploy a new engine image with an automatic rollback: restart both ranks on NEW (IMAGE for this run only), then
# ./verify.sh, the 32-test suite and the --parallel grammar check. Any failure restarts them on OLD.
#   ./deploy-gate.sh NEW_TAG OLD_TAG        (tags of tensorfold-dsv41, both already on both nodes: ./build.sh)
#   SKIP_BENCH=1 ./deploy-gate.sh ...       (skip the concurrency benchmark at the end)
# On success the ranks run NEW, but config.sh still names OLD: set TF_COMMIT (or IMAGE) in config.local.sh, as the
# last line says, or the next ./serve.sh starts OLD again.
set -u
[ $# -eq 2 ] || { echo "usage: ./deploy-gate.sh NEW_TAG OLD_TAG" >&2; exit 2; }
NEW=$1 OLD=$2
# shellcheck source=config.sh
. "$(dirname "$0")/config.sh"
OUT="$ROOT/runs/$(date -u +%Y-%m-%d)"
mkdir -p "$OUT"
export V41_BASE_URL="$API" V41_MODEL="$SERVED_MODEL_NAME"
# The suite exits 1 on any failed test. Without a router key the 2 router tests cannot pass, so skip that group then.
SUITE_GROUPS=""
[ -n "${V41_ROUTER_API_KEY:-${DGX_ROUTER_API_KEY:-}}" ] || SUITE_GROUPS="speed tools schema reasoning format long_context"

swap() {  # $1 = tag: restart both ranks on it
  (cd "$ROOT" && IMAGE="tensorfold-dsv41:$1" ./stop.sh && IMAGE="tensorfold-dsv41:$1" ./serve.sh) \
    > "$OUT/gate-$1-start.log" 2>&1
}
rollback() { log "ROLLBACK to $OLD: $*"; swap "$OLD"; log "ROLLED_BACK"; exit 1; }

swap "$NEW" || rollback "start failed"
log "serving $NEW"
(cd "$ROOT" && ./verify.sh) > "$OUT/verify-$NEW.log" 2>&1 || rollback "verify failed (see $OUT/verify-$NEW.log)"
log "VERIFY_OK"
cd "$OUT" || exit 1
# shellcheck disable=SC2086
python3 -I -u "$ROOT/tests/v41_suite.py" $SUITE_GROUPS > "suite-$NEW.log" 2>&1; echo "SUITE_EXIT=$?" >> "suite-$NEW.log"
log "$(grep -ac '^\[PASS' "suite-$NEW.log") pass, $(grep -ac '^\[FAIL' "suite-$NEW.log") fail"
grep -q "SUITE_EXIT=0" "suite-$NEW.log" || rollback "suite failed"
python3 -I -u "$ROOT/tests/v41_grammar_check.py" 2 > "grammar-check-$NEW.log" 2>&1
grep -q "passed$" "grammar-check-$NEW.log" && tail -1 "grammar-check-$NEW.log" | grep -q "^\([0-9]*\)/\1 passed$" \
  || rollback "grammar check failed"
log "$(tail -1 "grammar-check-$NEW.log")"
keep="keep it: IMAGE=\"\${IMAGE:-tensorfold-dsv41:$NEW}\" in config.local.sh (or TF_COMMIT, if NEW is a commit)"
if [ "${SKIP_BENCH:-0}" = 1 ]; then log "GATE_DONE (bench skipped); $keep"; exit 0; fi
python3 -I -u "$ROOT/tests/v41_bench.py" > "bench-$NEW.log" 2>&1
log "GATE_DONE ($OUT); $keep"
