#!/usr/bin/env bash
# One line about the host's memory every INTERVAL seconds, to the terminal and
# to a log, to find what fills a GB10 while a model server sits there: earlyoom
# says how little was left when it acted, not who took it.
#
#   scripts/mem-watch.sh [INTERVAL] [COUNT]     default 60 s, forever (COUNT 0)
#   nohup scripts/mem-watch.sh >/dev/null 2>&1 &    in the background (or in tmux)
#   GB10_MEM_WATCH_LOG=path                     default ~/.cache/gb10/mem-watch.log
#
# Fields, MiB, from /proc/meminfo:
#   avail  MemAvailable, what earlyoom compares with its limits (1 GiB SIGTERM
#          after scripts/host-tuning.sh, ~8 GB by DGX OS default)
#   free   MemFree;  cache  Cached (page cache incl. shmem, e.g. a mmapped table);
#   dirty  Dirty + Writeback (written, not yet on disk: a download or copy);
#   anon   AnonPages (process heaps: Python, browsers, compilers)
#   swap   swap in use
#   other  MemTotal minus all of the above, slab, page tables and kernel stacks:
#          on GB10 mostly what the GPU driver holds (CUDA allocations of a model
#          server: they appear in no process's RSS)
# then the five largest processes by RSS (name:MiB). A line whose avail fell
# with anon or other rising names the kind of memory that grew; the process
# list, who.
set -uo pipefail

INTERVAL=${1:-60}
COUNT=${2:-0}
LOG=${GB10_MEM_WATCH_LOG:-$HOME/.cache/gb10/mem-watch.log}
MEMINFO=${GB10_MEMINFO:-/proc/meminfo}   # (overridable for a test)
case "$INTERVAL$COUNT" in *[!0-9]*) echo "usage: $0 [INTERVAL_SECONDS] [COUNT]" >&2; exit 2 ;; esac
[ "$INTERVAL" -ge 1 ] || { echo "INTERVAL must be at least 1 second" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"

line() {
  local m top
  m=$(awk '
    { v[$1] = $2 }
    END {
      mib = 1024
      other = v["MemTotal:"] - v["MemFree:"] - v["Buffers:"] - v["Cached:"] - v["SwapCached:"] \
              - v["AnonPages:"] - v["Slab:"] - v["PageTables:"] - v["KernelStack:"]
      if (other < 0) other = 0
      printf "avail=%d free=%d cache=%d dirty=%d anon=%d swap=%d other=%d",
        v["MemAvailable:"] / mib, v["MemFree:"] / mib, v["Cached:"] / mib,
        (v["Dirty:"] + v["Writeback:"]) / mib, v["AnonPages:"] / mib,
        (v["SwapTotal:"] - v["SwapFree:"]) / mib, other / mib
    }' "$MEMINFO")
  top=$(ps -eo rss=,comm= --sort=-rss 2>/dev/null | awk 'NR <= 5 { r = $1; $1 = ""; sub(/^ +/, ""); gsub(/ /, "_"); printf "%s%s:%d", (NR > 1 ? " " : ""), $0, r / 1024 }')
  printf '%s %s top: %s\n' "$(date -u +%FT%TZ)" "$m" "$top"
}

n=0
while :; do
  line | tee -a "$LOG"
  n=$((n + 1))
  [ "$COUNT" -gt 0 ] && [ "$n" -ge "$COUNT" ] && break
  sleep "$INTERVAL"
done
