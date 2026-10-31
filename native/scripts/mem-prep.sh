#!/usr/bin/env bash
# GB10 memory around a model server, for every server in this project. Each
# command runs here and, over ssh, on every ssh target named after it (the
# worker of a pair; the script goes over stdin, the worker needs no checkout).
#
#   scripts/mem-prep.sh status [target...]   free memory, the part of it in blocks of 2 MiB or more, uptime
#   scripts/mem-prep.sh idle [target...]     exit 1, saying what, while anything runs on a GPU there
#   scripts/mem-prep.sh prep [target...]     with the GPU idle: drop the page cache, compact free memory
#   scripts/mem-prep.sh sudoers [--undo] [target...]
#                                            let `prep` make its two writes without a password (asks for it once)
#
# Why (sudoingX/dgx-spark-deepseek-v41, FINDINGS 1, 6, 22, 24; their numbers):
# - CUDA allocations on GB10 do not reclaim the page cache: a 67 MB one failed
#   with 45 GB "available" and 8 GB free. Engines size their pools to free memory.
# - After days of uptime free memory splits into small blocks; a GPU job then
#   ran ~2x slower than on its twin, and over 16 boots of one server the decode
#   rounds were faster the more free 2 MiB+ blocks there were at boot. Dropping
#   the cache and compacting takes under a second with the GPU idle (a box up for
#   12 days stayed fragmented until a reboot). This is a one-off before a start,
#   not background compaction (vm.compaction_proactiveness=0 stays).
# - A Spark stopped answering (no ssh, no ping) while a 25 GB copy ran beside a
#   GPU job: `idle` is the check before downloads, image copies and rsyncs.
# Never during a serve: the page cache holds hot data (DeepSeek's Engram rows).
#
# `prep` writes 1 to /proc/sys/vm/drop_caches and /proc/sys/vm/compact_memory
# with `sudo -n tee` (or directly as root, e.g. in a unit's ExecStartPre=+).
# Without passwordless sudo for those two it says so and changes nothing;
# `sudoers` adds exactly that rule (/etc/sudoers.d/gb10-mem-prep), `--undo`
# removes it. Each start appends a line to ~/.cache/gb10/mem-prep.log, which
# bench/perf.py prints in its header.
set -uo pipefail

SELF="${BASH_SOURCE[0]:-}"   # empty on a worker, where it arrives over stdin
RULE_FILE=/etc/sudoers.d/gb10-mem-prep
BUDDYINFO=${GB10_BUDDYINFO:-/proc/buddyinfo}   # (both overridable for scripts/test-mem-prep.sh)
VM_DIR=${GB10_VM_DIR:-/proc/sys/vm}
here=$(hostname -s 2>/dev/null || hostname)

blocks() {  # "<free GiB> <GiB of it in free blocks of 2 MiB or more>", from /proc/buddyinfo
  awk -v ps="$(getconf PAGESIZE)" 'BEGIN { k = 0; while (ps * 2 ^ k < 2097152) k++ }
    / zone / { for (i = 5; i <= NF; i++) { b = $i * ps * 2 ^ (i - 5); all += b; if (i - 5 >= k) big += b } }
    END { printf "%.1f %.1f\n", all / 2 ^ 30, big / 2 ^ 30 }' "$BUDDYINFO"
}
uptime_days() { awk '{ printf "%.1f", $1 / 86400 }' /proc/uptime; }

busy() {  # what runs on the GPU here, one line each; nothing when idle
  local c
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null |
      awk -F', *' 'NF { print "GPU process " $1 " " $2 }'
  fi
  if command -v docker >/dev/null; then
    for c in $(docker ps -q 2>/dev/null); do
      docker inspect --format '{{.Name}} {{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' "$c" 2>/dev/null |
        awk 'tolower($2 " " $3) ~ /gpu|nvidia/ { sub(/^\//, "", $1); print "GPU container " $1 }'
    done
  fi
  # Model servers outside containers, by process name (the kernel's comm, 15
  # characters): a word in some other program's arguments is no server (earlyoom's
  # --prefer list names them all).
  ps -eo pid=,comm=,args= 2>/dev/null | awk '
    { name = $2 }
    (name ~ /^python/ && $0 ~ /sglang[.]launch_server|vllm[.]entrypoints/) ||
    name ~ /^(sglang|VLLM)::/ || name == "llama-server" || name == "ggml-rpc-server" ||
    ((name == "vllm" || name == "tensorfold") && $0 ~ / serve( |$)/) {
      pid = $1; sub(/^ *[0-9]+ +[^ ]+ +/, ""); print "server process " pid " " substr($0, 1, 100)
    }'
}

wr() {  # write 1 to /proc/sys/vm/$1 as root, without asking for a password
  if [ "$(id -u)" = 0 ]; then echo 1 > "$VM_DIR/$1"; else echo 1 | sudo -n tee "$VM_DIR/$1" >/dev/null 2>&1; fi
}

local_cmd() {  # one command on this machine
  local b0 b1 what
  case "$1" in
    status)
      read -r f g <<< "$(blocks)"
      echo "[$here] memory: free $f GiB, $g GiB of it in 2 MiB+ blocks, up $(uptime_days) days" ;;
    idle)
      what=$(busy | sort -u)
      [ -z "$what" ] && return 0
      while IFS= read -r l; do echo "[$here] $l"; done <<< "$what"
      return 1 ;;
    prep)
      what=$(busy | sort -u)
      if [ -n "$what" ]; then
        echo "[$here] memory left as it is: the GPU is in use ($(echo "$what" | head -1))"
        return 0
      fi
      b0=$(blocks)
      sync
      if wr drop_caches && wr compact_memory; then
        b1=$(blocks)
        read -r f g <<< "$b1"
        echo "[$here] memory: free $f GiB, $g GiB of it in 2 MiB+ blocks after the cache drop and compaction" \
          "(before: ${b0% *} / ${b0#* }), up $(uptime_days) days"
        what=dropped
      else
        read -r f g <<< "$b0"
        echo "[$here] memory: free $f GiB, $g GiB of it in 2 MiB+ blocks, up $(uptime_days) days;" \
          "not dropped or compacted (no passwordless sudo for it: scripts/mem-prep.sh sudoers)"
        what=as-is
      fi
      mkdir -p "$HOME/.cache/gb10" 2>/dev/null &&
        printf '%s %s %s free=%s big=%s before_free=%s before_big=%s up_days=%s\n' "$(date -u +%FT%TZ)" "$here" "$what" \
          "$f" "$g" "${b0% *}" "${b0#* }" "$(uptime_days)" >> "$HOME/.cache/gb10/mem-prep.log"
      return 0 ;;
    sudoers|sudoers-undo)
      local user tmp rc
      user=$(id -un)
      if [ "$1" = sudoers-undo ]; then
        sudo rm -f "$RULE_FILE" && echo "[$here] removed $RULE_FILE"
        return
      fi
      tmp=$(mktemp)
      printf '%s\n' "# gb10 scripts/mem-prep.sh: the page-cache drop and memory compaction before a model server starts" \
        "$user ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches, /usr/bin/tee /proc/sys/vm/compact_memory" > "$tmp"
      sudo visudo -cqf "$tmp" && sudo install -m 0440 -o root -g root "$tmp" "$RULE_FILE"
      rc=$?
      rm -f "$tmp"
      [ "$rc" = 0 ] && echo "[$here] $RULE_FILE: $user may run those two writes without a password"
      return "$rc" ;;
  esac
}

cmd=${1:-status}; shift || true
undo=0
if [ "$cmd" = sudoers ] && [ "${1:-}" = --undo ]; then undo=1; shift; fi
case "$cmd" in status|idle|prep|sudoers) ;; *)
  echo "usage: $0 status|idle|prep [ssh-target...] | sudoers [--undo] [ssh-target...]" >&2; exit 2 ;;
esac
[ "$undo" = 1 ] && cmd=sudoers-undo

rc=0
local_cmd "$cmd" || rc=1
for t in "$@"; do
  if [ "${cmd%-undo}" = sudoers ]; then
    # sudo may ask for the password there: a terminal, so the function goes as an argument, not over stdin
    ssh -t -o ConnectTimeout=10 "$t" "bash -c $(printf '%q' "RULE_FILE=$RULE_FILE; here=\$(hostname -s); $(declare -f local_cmd); local_cmd $cmd")" || rc=1
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$t" bash -s -- "$cmd" < "$SELF"
    case $? in 0) ;; 255) echo "[$t] cannot be reached over ssh"; rc=1 ;; *) rc=1 ;; esac
  fi
done
# The verdict once, where it was asked (a worker's copy runs from stdin, with no SELF).
if [ "$cmd" = idle ] && [ "$rc" != 0 ] && [ -n "$SELF" ]; then
  if [ "${ALLOW_BUSY_GPU:-0}" = 1 ]; then
    echo "ALLOW_BUSY_GPU=1: going on beside it" >&2; rc=0
  else
    echo "Large copies beside GPU work have hung a Spark (no ssh, no ping): stop it first, or ALLOW_BUSY_GPU=1." >&2
  fi
fi
exit "$rc"
