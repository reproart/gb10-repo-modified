#!/usr/bin/env bash
# Checks before a start, on both nodes. It refuses (exit 1) on the faults that otherwise show as a hang or a crash
# minutes into the start: missing files, different images, a bad RoCE GID, no memory, no disk, a busy port.
#   ./preflight.sh          (./serve.sh runs it first; PREFLIGHT=0 ./serve.sh skips it)
set -uo pipefail
# shellcheck source=config.sh
. "$(dirname "$0")/config.sh"

bad=0
wrun() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }   # (stdin open: wssh closes it)
ok() { echo "[ ok ] $*"; }
no() { echo "[FAIL] $*"; bad=1; }

wssh true 2>/dev/null && ok "ssh to the worker ($WORKER_SSH)" || { no "ssh to the worker ($WORKER_SSH) failed"; exit 1; }
ping -c 2 -W 2 "$WORKER_IP" >/dev/null 2>&1 && ok "the worker answers on the CX7 link ($WORKER_IP)" \
  || no "no answer from $WORKER_IP: check the cable and docs/NETWORK.md"

cpuset_fast() {  # $1 = a CPU list (5-9,15-19): OK when every core in it is one of the fastest (cpufreq)
  local max f c part want="" fast="" bad=""
  max=$(cat /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq 2>/dev/null | sort -n | tail -1)
  [ -n "$max" ] || { echo "OK CPUSET $1 (no cpufreq here: cores not checked)"; return; }
  for f in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq; do
    c=${f#/sys/devices/system/cpu/cpu}; c=${c%%/*}
    [ "$(cat "$f")" = "$max" ] && fast="$fast $c"
  done
  # shellcheck disable=SC2086  # a word list on purpose
  fast=" $(printf '%s\n' $fast | sort -n | paste -sd' ')"
  for part in ${1//,/ }; do
    if [[ $part == *-* ]]; then want="$want $(seq -s ' ' "${part%-*}" "${part#*-}")"; else want="$want $part"; fi
  done
  for c in $want; do case " $fast " in *" $c "*) ;; *) bad="$bad $c" ;; esac; done
  if [ -z "$bad" ]; then echo "OK CPUSET $1: the fast cores ($((max / 1000)) MHz)"
  else echo "FAIL CPUSET $1 has slower cores:$bad (the fastest:$fast; set CPUSET, or empty for none)"; fi
}

check_node() {  # $1 = label, $2 = command runner ("" for here, wssh for the worker)
  local who=$1 run=$2 out
  out=$($run bash -s <<EOF
set -u
$(declare -f cpuset_fast)
[ -z "$CPUSET" ] || cpuset_fast "$CPUSET"
for d in "$MODEL_VIEW/model" "$MODEL_VIEW/engram"; do [ -d "\$d" ] || echo "FAIL missing \$d"; done
[ -f "$MODEL_VIEW/model/chat_template.jinja" ] || echo "FAIL missing $MODEL_VIEW/model/chat_template.jinja"
n=\$(ls "$MODEL_VIEW"/model/*.safetensors 2>/dev/null | wc -l); [ "\$n" -ge 43 ] || echo "FAIL \$n of 43 weight shards"
docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null | sed 's/^/IMAGE /' || echo "FAIL no image $IMAGE"
for dev in \$(echo "$RDMA_DEVICES" | tr ',' ' '); do
  if [ "$GID_INDEX" = "-1" ]; then echo "OK gid auto (\$dev)"; continue; fi
  g=\$(cat /sys/class/infiniband/\$dev/ports/1/gids/$GID_INDEX 2>/dev/null)
  t=\$(cat /sys/class/infiniband/\$dev/ports/1/gid_attrs/types/$GID_INDEX 2>/dev/null)
  case "\$g" in ""|0000:0000:0000:0000:0000:0000:0000:0000) echo "FAIL \$dev GID $GID_INDEX is empty: see docs/NETWORK.md";;
    *) [ "\$t" = "RoCE v2" ] && echo "OK \$dev GID $GID_INDEX \$g (\$t)" || echo "FAIL \$dev GID $GID_INDEX is \$t, not RoCE v2";; esac
done
avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo); [ "\$avail" -ge 100 ] \
  && echo "OK \$avail GiB memory available" || echo "FAIL only \$avail GiB memory available (need ~100): stop other GPU programs"
mkdir -p "$CACHE_DIR"; free=\$(df -BG --output=avail "$CACHE_DIR" | tail -1 | tr -dc 0-9)
[ -d "$CACHE_DIR/tf-ds-rank" ] && need=5 || need=110
[ "\$free" -ge "\$need" ] && echo "OK \$free GiB free for the cache" || echo "FAIL \$free GiB free in $CACHE_DIR (need \$need)"
[ -z "${CGROUP_PARENT-tensorfold.slice}" ] || [ "\$(systemctl show -p MemorySwapMax --value ${CGROUP_PARENT-tensorfold.slice} 2>/dev/null)" = 0 ] \
  && echo "OK no swap for the engine" || echo "FAIL ${CGROUP_PARENT-tensorfold.slice} has swap or is missing: run ./host-tuning.sh"
diff=""
for kv in vm.swappiness=10 vm.compaction_proactiveness=0 vm.extfrag_threshold=500 vm.min_free_kbytes=1048576 \
    vm.watermark_scale_factor=50; do
  k=\${kv%%=*}; [ "\$(cat /proc/sys/\$(echo "\$k" | tr . /) 2>/dev/null)" = "\${kv#*=}" ] || diff="\$diff \$k"
done
[ -z "\$diff" ] && echo "OK memory settings of ./host-tuning.sh" \
  || echo "FAIL memory settings differ (\${diff# }): run ./host-tuning.sh"
systemctl is-active -q earlyoom && grep -qE -- "--prefer [^ ]*tensorfold" /etc/default/earlyoom 2>/dev/null \
  && echo "OK earlyoom stops the engine before the node runs out of memory" \
  || echo "FAIL earlyoom is not running with the settings of ./host-tuning.sh: run it again"
docker ps --format '{{.Names}}' | grep -q '^tfv41-r' && echo "FAIL a tfv41 rank already runs: ./stop.sh first" || true
EOF
)
  while IFS= read -r line; do
    case "$line" in
      FAIL*) no "$who: ${line#FAIL }" ;;
      OK*) ok "$who: ${line#OK }" ;;
      IMAGE*) echo "${line#IMAGE }" > "/tmp/tfv41-image-$who" ;;
    esac
  done <<< "$out"
}
check_node head ""
check_node worker wrun
if [ -s /tmp/tfv41-image-head ] && cmp -s /tmp/tfv41-image-head /tmp/tfv41-image-worker; then
  ok "both nodes hold $IMAGE ($(cut -c8-19 /tmp/tfv41-image-head))"
else
  no "the nodes hold different images (or none) for $IMAGE: run ./build.sh"
fi
rm -f /tmp/tfv41-image-head /tmp/tfv41-image-worker
ss -ltn | grep -q ":$PORT " && no "port $PORT is in use on the head" || ok "port $PORT is free"
[ "$bad" = 0 ] && echo "preflight: all checks passed" || { echo "preflight: fix the FAIL lines, then start again"; exit 1; }
