#!/usr/bin/env bash
# shellcheck disable=SC2016  # the checks are single-quoted on purpose: eval'd later
# watchdog.sh against stubs: docker (the container's state), curl (vLLM's
# /metrics), /proc/meminfo, and stop/serve commands that record their calls.
# No Docker, GPU or server needed.   scripts/test-watchdog.sh
set -uo pipefail
VFA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/state"
cat > "$T/bin/docker" <<'EOF'
#!/bin/sh
[ "$1" = inspect ] && { echo "${FAKE_RUNNING:-true}"; exit 0; }
exit 0
EOF
cat > "$T/bin/curl" <<'EOF'
#!/bin/sh
[ -f "$FAKE_METRICS" ] && { cat "$FAKE_METRICS"; exit 0; }
exit 7
EOF
printf '#!/bin/sh\necho stop >> "$CALLS"\n' > "$T/bin/fake-stop"
printf '#!/bin/sh\necho serve >> "$CALLS"\nexit "${FAKE_SERVE_RC:-0}"\n' > "$T/bin/fake-serve"
chmod +x "$T/bin/"*

export PATH="$T/bin:$PATH" GB10_STATE_DIR="$T/state" GB10_MEMINFO="$T/meminfo" FAKE_METRICS="$T/metrics"
export CALLS="$T/calls" WATCHDOG_STOP_CMD="$T/bin/fake-stop" WATCHDOG_SERVE_CMD="$T/bin/fake-serve"
export WATCHDOG_SLEEP=true WATCHDOG_INTERVAL=60 WATCHDOG_IDLE_MIN=5 WATCHDOG_MIN_AVAIL_MB=2048 WATCHDOG_COOLDOWN_MIN=60
export API_KEY=""   # (config.local.sh may set one)

pass=0 fail=0
check() { if eval "$2"; then pass=$((pass + 1)); echo "  ok   $1"; else fail=$((fail + 1)); echo "  FAIL $1"; fi; }
avail() { printf 'MemTotal: 127533056 kB\nMemAvailable: %d kB\n' $(($1 * 1024)) > "$T/meminfo"; }
requests() {  # running waiting, as vLLM exposes them
  printf '# HELP vllm:num_requests_running x\nvllm:num_requests_running{engine="0",model_name="qwen3.8-flash-next"} %s\nvllm:num_requests_waiting{engine="0",model_name="qwen3.8-flash-next"} %s\n' "$1.0" "$2.0" > "$T/metrics"
}
reset() { rm -f "$CALLS" "$T/state/"*; : > "$CALLS"; unset FAKE_RUNNING FAKE_SERVE_RC; }
run() { WATCHDOG_CHECKS=$1 "$VFA/watchdog.sh" > "$T/out" 2>&1; echo $? > "$T/rc"; }

reset; avail 1500; requests 0 0; run 5
check "low memory, idle 4 min (5 checks): no restart" '[ ! -s "$CALLS" ]'
reset; avail 1500; requests 0 0; run 6
check "low memory, idle 5 min: stop then serve" '[ "$(tr "\n" " " < "$CALLS")" = "stop serve " ]'
check "  logged with the numbers" 'grep -q "restarting .*available 1500 MiB < 2048 MiB, no request for 5 min" "$T/state/vllm-flash-ar-watchdog.log"'
check "  last restart recorded" '[ -s "$T/state/vllm-flash-ar-watchdog.last-restart" ]'
reset; avail 1500; requests 1 0; run 12
check "low memory, a request running: no restart" '[ ! -s "$CALLS" ]'
reset; avail 1500; requests 0 2; run 12
check "low memory, requests waiting: no restart" '[ ! -s "$CALLS" ]'
reset; avail 9000; requests 0 0; run 12
check "idle, enough memory: no restart" '[ ! -s "$CALLS" ]'
reset; avail 1500; rm -f "$T/metrics"; run 12
check "low memory, API not answering (loading): no restart" '[ ! -s "$CALLS" ]'
reset; avail 1500; requests 0 0; export FAKE_RUNNING=false; run 12
check "server not running: never started by the watchdog" '[ ! -s "$CALLS" ]'
check "  said once" '[ "$(grep -c "is not running" "$T/out")" = 1 ]'
reset; avail 1500; requests 0 0; date +%s > "$T/state/vllm-flash-ar-watchdog.last-restart"; run 12
check "restarted under an hour ago: no restart (cooldown)" '[ ! -s "$CALLS" ]'
check "  said once" '[ "$(grep -c "cooldown 60 min" "$T/out")" = 1 ]'
reset; avail 1500; requests 0 0; echo $(( $(date +%s) - 3700 )) > "$T/state/vllm-flash-ar-watchdog.last-restart"; run 6
check "cooldown over: restart" '[ "$(tr "\n" " " < "$CALLS")" = "stop serve " ]'
reset; avail 1500; requests 0 0; export FAKE_SERVE_RC=1; run 20
check "serve fails: the watchdog stops (exit 1), no second try" '[ "$(cat "$T/rc")" = 1 ] && [ "$(tr "\n" " " < "$CALLS")" = "stop serve " ]'
check "  logged" 'grep -q "serve.sh failed" "$T/state/vllm-flash-ar-watchdog.log"'
reset; avail 9000; requests 0 0
WATCHDOG_SLEEP=sleep WATCHDOG_CHECKS=0 "$VFA/watchdog.sh" > "$T/out1" 2>&1 &
first=$!
for _ in $(seq 40); do [ -s "$T/state/vllm-flash-ar-watchdog.pid" ] && break; sleep 0.1; done
WATCHDOG_CHECKS=1 "$VFA/watchdog.sh" > "$T/out2" 2>&1
check "a second watchdog exits: one at a time" 'grep -q "already runs" "$T/out2"'
kill "$first"; export t0; t0=$(date +%s%N)
for _ in $(seq 50); do [ -d "/proc/$first" ] || break; sleep 0.1; done
check "SIGTERM ends it at once, mid-sleep (lock freed for the next serve)" '[ ! -d "/proc/$first" ] && [ $(( ($(date +%s%N) - t0) / 1000000 )) -lt 3000 ]'
check "  pid file removed" '[ ! -f "$T/state/vllm-flash-ar-watchdog.pid" ]'
WATCHDOG_INTERVAL=0 WATCHDOG_CHECKS=1 "$VFA/watchdog.sh" > "$T/out" 2>&1
check "bad setting refused" 'grep -q "WATCHDOG_INTERVAL must be at least 1" "$T/out"'

echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
