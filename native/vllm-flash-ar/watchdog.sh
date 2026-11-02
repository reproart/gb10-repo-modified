#!/usr/bin/env bash
# Restart the server before earlyoom stops it: when available memory is low AND
# the server has had no request for a while, ./stop.sh then ./serve.sh, logged.
#
#   ./watchdog.sh                in the foreground (prints what it logs)
#   WATCHDOG=1 ./serve.sh        ./serve.sh starts it in the background after the
#                                server is up, ./stop.sh stops it (WATCHDOG in
#                                config.local.sh to keep it on); off by default
#   tail -f ~/.cache/gb10/vllm-flash-ar-watchdog.log
#
# Why: idle, this server loses memory outside torch's allocator (~7 MiB a
# minute on a GX10, 2026-11-02; README, "earlyoom stops the engine"), and
# earlyoom, once the host has less than its SIGTERM line (1 GiB after
# ../scripts/host-tuning.sh), stops the engine at whatever moment that is, and
# with RESTART=no it stays down. A restart costs the prefix cache (the next turn
# of a long conversation is prefilled again) and ~5 minutes without an API, so
# it waits for both conditions:
#   available memory (MemAvailable) under WATCHDOG_MIN_AVAIL_MB (2048), and
#   no running or waiting request (vLLM's /metrics) for WATCHDOG_IDLE_MIN (5)
#   minutes, read again just before the stop.
# At most one restart per WATCHDOG_COOLDOWN_MIN (60); if ./serve.sh fails it
# logs that and exits instead of trying again. It never starts a server that is
# not running (stopped by hand, or by earlyoom). Set WATCHDOG_MIN_AVAIL_MB above
# earlyoom's SIGTERM line: with DGX OS's own earlyoom (~8 GB) that is ~10240.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

STATE=${GB10_STATE_DIR:-$HOME/.cache/gb10}
WLOG="$STATE/vllm-flash-ar-watchdog.log"
PIDFILE="$STATE/vllm-flash-ar-watchdog.pid"
LAST="$STATE/vllm-flash-ar-watchdog.last-restart"
MEMINFO=${GB10_MEMINFO:-/proc/meminfo}             # (overridable for scripts/test-watchdog.sh)
SLEEP=${WATCHDOG_SLEEP:-sleep}
STOP_CMD=${WATCHDOG_STOP_CMD:-$VFA_ROOT/stop.sh}
SERVE_CMD=${WATCHDOG_SERVE_CMD:-$VFA_ROOT/serve.sh}
CHECKS=${WATCHDOG_CHECKS:-0}                        # 0 = forever
for v in WATCHDOG_MIN_AVAIL_MB WATCHDOG_IDLE_MIN WATCHDOG_INTERVAL WATCHDOG_COOLDOWN_MIN CHECKS; do
  case "${!v}" in ''|*[!0-9]*) echo "$v must be a whole number, not '${!v}'" >&2; exit 2 ;; esac
done
[ "$WATCHDOG_INTERVAL" -ge 1 ] || { echo "WATCHDOG_INTERVAL must be at least 1" >&2; exit 2; }
mkdir -p "$STATE"

# One watchdog at a time.
exec 9>"$STATE/vllm-flash-ar-watchdog.lock"
flock -n 9 || { echo "a watchdog already runs (pid $(cat "$PIDFILE" 2>/dev/null))"; exit 0; }
echo $$ > "$PIDFILE"
sleeper=""
trap 'rm -f "$PIDFILE"' EXIT
# ./stop.sh's kill must end it now, not after the current sleep: the lock goes with it.
trap '[ -n "$sleeper" ] && kill "$sleeper" 2>/dev/null; exit 0' TERM INT

wlog() { local l; l="[watchdog $(date -u +%FT%TZ)] $*"; echo "$l"; echo "$l" >> "$WLOG"; }

avail_mb() { awk '/^MemAvailable:/ { printf "%d", $2 / 1024 }' "$MEMINFO"; }

# Running + waiting requests from vLLM's /metrics; nothing on failure.
in_flight() {
  local auth=()
  [ -n "$API_KEY" ] && auth=(-H "Authorization: Bearer $API_KEY")
  curl -fsS -m 5 ${auth[@]+"${auth[@]}"} "http://127.0.0.1:$PORT/metrics" 2>/dev/null |
    awk '/^vllm:num_requests_(running|waiting)[{ ]/ { n += $NF; seen = 1 } END { if (seen) printf "%d", n }'
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ]; }

# idle = consecutive checks that saw no request; need + 1 of them span need intervals
need=$(( (WATCHDOG_IDLE_MIN * 60 + WATCHDOG_INTERVAL - 1) / WATCHDOG_INTERVAL ))
wlog "watching $NAME: restart when available < ${WATCHDOG_MIN_AVAIL_MB} MiB and no request for" \
  "${WATCHDOG_IDLE_MIN} min (checks every ${WATCHDOG_INTERVAL}s), at most once per ${WATCHDOG_COOLDOWN_MIN} min"
idle=0 note=""
n=0
while :; do
  n=$((n + 1))
  if ! running; then
    [ "$note" = down ] || wlog "$NAME is not running: waiting (the watchdog never starts it)"
    note=down idle=0
  else
    [ "$note" = down ] && wlog "$NAME runs again"
    [ "$note" = down ] && note=""
    q=$(in_flight)
    if [ -z "$q" ]; then
      idle=0   # API not answering (loading, or no metrics): not idle
    elif [ "$q" -gt 0 ]; then
      idle=0
    else
      idle=$((idle + 1))
    fi
    a=$(avail_mb)
    if [ "$a" -lt "$WATCHDOG_MIN_AVAIL_MB" ] && [ "$idle" -gt "$need" ]; then
      now=$(date +%s) last=$(cat "$LAST" 2>/dev/null || echo 0)
      if [ $((now - last)) -lt $((WATCHDOG_COOLDOWN_MIN * 60)) ]; then
        [ "$note" = cooldown ] || wlog "available ${a} MiB, idle: restart due, but the last one was" \
          "$(( (now - last) / 60 )) min ago (cooldown ${WATCHDOG_COOLDOWN_MIN} min)"
        note=cooldown
      elif [ "$(in_flight)" != 0 ]; then
        idle=0   # a request came in, or the API stopped answering, since the last check
      else
        wlog "restarting $NAME: available ${a} MiB < ${WATCHDOG_MIN_AVAIL_MB} MiB, no request for" \
          "$(( (idle - 1) * WATCHDOG_INTERVAL / 60 )) min"
        echo "$now" > "$LAST"
        WATCHDOG_CHILD=1 "$STOP_CMD" >> "$WLOG" 2>&1
        if WATCHDOG_CHILD=1 "$SERVE_CMD" >> "$WLOG" 2>&1; then
          wlog "restarted: available now $(avail_mb) MiB"
          idle=0 note=""
        else
          wlog "./serve.sh failed (output above in this log; docker logs $NAME): the watchdog stops here"
          exit 1
        fi
      fi
    elif [ "$note" = cooldown ]; then
      note=""
    fi
  fi
  [ "$CHECKS" -gt 0 ] && [ "$n" -ge "$CHECKS" ] && break
  "$SLEEP" "$WATCHDOG_INTERVAL" &
  sleeper=$!
  wait "$sleeper"
  sleeper=""
done
