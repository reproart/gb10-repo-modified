#!/usr/bin/env bash
# Host memory settings for one Spark serving a model on SGLang (or llama.cpp,
# or vLLM). Needs sudo; changes nothing it cannot undo.
#
#   ./scripts/host-tuning.sh           apply (again: the same result)
#   ./scripts/host-tuning.sh --check   show the live values against these, change nothing
#   ./scripts/host-tuning.sh --undo    put back the values from before the first apply
#
# Why. A model here holds 90-105 of the 121.6 GiB. Under load the kernel
# reclaims and compacts memory on the allocating thread (direct compaction),
# swaps other programs, and a long prefill stalls for 1-4 s per chunk. The
# two-Spark DeepSeek recipe measured it (tensorfold-dsv41/runs/2026-10-18):
# with background compaction on, 46 of 248 prompt chunks were slow (over
# 1.3 s); with these values, 2. SGLang's prefill does not gain: it takes its
# pool at start and the kernel never stalled without these either (A/B in
# results/RESULTS.md, "Host tuning"). What it gets is earlyoom at 1 GiB (so
# the 27B profiles default to MEM_FRACTION 0.85) and a server that cannot swap.
#
# Changes (kernel defaults in brackets):
#   1. /etc/sysctl.d/99-gb10.conf:
#        vm.swappiness = 10               [60]   swap other programs less
#        vm.compaction_proactiveness = 0  [20]   no background compaction while a prompt runs
#        vm.extfrag_threshold = 500       [500]  compact before reclaiming
#        vm.min_free_kbytes = 1048576     [boot] keep 1 GiB free for the kernel
#        vm.watermark_scale_factor = 50   [10]   free memory in the background before programs wait
#   2. earlyoom (required; DGX OS has it): SIGTERM below 1 GiB available, SIGKILL
#      below 0.5 GiB, preferring the model server (sglang::scheduler, python,
#      tensorfold, llama-server, vLLM) and never sshd, systemd, dockerd or the
#      display stack, so a full machine loses the server, not ssh. DGX OS's own
#      threshold (~8 GB, the 0.85 trap in models/qwen3.8-27b.sh) goes with it.
#   3. With 1 applied, ./serve.sh starts SGLang in a user scope with no swap
#      (NOSWAP in scripts/serve-sglang.sh); the systemd unit of
#      scripts/install-service.sh has MemorySwapMax=0 already.
# The values are the ones tensorfold-dsv41/host-tuning.sh sets on the pair, so
# either script leaves a machine fit for both; that one adds its no-swap slice.
set -euo pipefail

node_script() {
  cat <<'EOF'
set -euo pipefail
eo=/etc/default/earlyoom
conf=/etc/sysctl.d/99-gb10.conf
state=/var/lib/gb10-host-tuning
before=$state/sysctl.before
eostate=$state/earlyoom.before
keys="vm.swappiness vm.compaction_proactiveness vm.extfrag_threshold vm.min_free_kbytes vm.watermark_scale_factor"
want="vm.swappiness=10 vm.compaction_proactiveness=0 vm.extfrag_threshold=500 vm.min_free_kbytes=1048576 vm.watermark_scale_factor=50"
args="-r 3600 -p -M 1048576,524288 -s 100,100 --prefer ^(sglang|python|tensorfold|llama-server|VLLM) --avoid ^(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager|gdm|Xorg|nvidia)"

undo() {  # the machine as it was before the first apply
  rm -f "$conf"
  if [ -f "$before" ]; then
    sysctl -q -p "$before"
    note="sysctl values from before the first apply are back"
  else
    sysctl -q -w vm.swappiness=60 vm.compaction_proactiveness=20 vm.extfrag_threshold=500 vm.watermark_scale_factor=10
    sysctl -q --system || true
    note="no saved values: kernel defaults set, vm.min_free_kbytes goes back at the next boot"
  fi
  if [ -f "$eo.pre-gb10" ]; then
    mv "$eo.pre-gb10" "$eo"
    was_enabled=enabled; was_active=active
    if [ -f "$eostate" ]; then read -r was_enabled was_active < "$eostate"; fi
    if [ "$was_enabled" = disabled ]; then systemctl disable -q earlyoom || true; fi
    if [ "$was_active" = active ]; then systemctl restart earlyoom || true; else systemctl stop earlyoom || true; fi
  fi
  rm -rf "$state"
}

check() {  # print the live values against ours; exit 0 when all match
  local k v w ok=0
  for kv in $want; do
    k=${kv%%=*} w=${kv#*=} v=$(sysctl -n "$k")
    if [ "$v" = "$w" ]; then echo "  ok   $k = $v"; else echo "  diff $k = $v (here $w)"; ok=1; fi
  done
  if systemctl is-active -q earlyoom && grep -qF -- "--prefer ^(sglang|" "$eo" 2>/dev/null; then
    echo "  ok   earlyoom with these settings"
  else
    echo "  diff earlyoom: $(systemctl is-active earlyoom 2>/dev/null || true), $(grep '^EARLYOOM_ARGS=' "$eo" 2>/dev/null || echo 'no EARLYOOM_ARGS')"
    ok=1
  fi
  return "$ok"
}

case "${1:-}" in
  --undo) undo; echo "undone ($note)"; exit 0 ;;
  --check) check && echo "host tuning: applied" || echo "host tuning: not (fully) applied"; exit 0 ;;
  "") ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac
if [ ! -f "$eo" ] || ! command -v earlyoom >/dev/null; then
  echo "earlyoom is not installed: sudo apt install earlyoom, then run this again" >&2
  exit 1
fi
for k in $keys; do sysctl -n "$k" >/dev/null; done

# The live values and earlyoom's state before the first apply, for --undo.
mkdir -p "$state"
if [ ! -f "$conf" ] && [ ! -f "$before" ]; then
  for k in $keys; do echo "$k = $(sysctl -n "$k")"; done > "$before.new"
  mv "$before.new" "$before"
fi
if [ ! -f "$eo.pre-gb10" ] && [ ! -f "$eostate" ]; then
  echo "$(systemctl is-enabled earlyoom 2>/dev/null || true) $(systemctl is-active earlyoom 2>/dev/null || true)" > "$eostate"
fi
# From here a failure puts the machine back as it was.
rollback() {
  local rc=$?
  [ "$rc" = 0 ] && return
  set +e; trap - EXIT
  undo >/dev/null 2>&1
  echo "setup failed; the machine is back to its state before the first apply" >&2
  exit "$rc"
}
trap rollback EXIT
for kv in $want; do echo "${kv%%=*} = ${kv#*=}"; done > "$conf"
sysctl -q -p "$conf"
# No quotes around the arguments: the init script passes these words to
# earlyoom as they are (systemd removes quotes).
[ -f "$eo.pre-gb10" ] || cp "$eo" "$eo.pre-gb10"
{ grep -v '^EARLYOOM_ARGS=' "$eo.pre-gb10" || true; printf 'EARLYOOM_ARGS="%s"\n' "$args"; } > "$eo.new"
mv "$eo.new" "$eo"
systemctl enable -q earlyoom
systemctl restart earlyoom
sleep 1
if ! systemctl is-active -q earlyoom; then
  echo "earlyoom did not start with the new settings (journalctl -u earlyoom shows why)" >&2
  exit 1
fi
trap - EXIT
check
EOF
}

case "${1:-}" in ""|--undo|--check) ;; *) echo "usage: $0 [--check|--undo]" >&2; exit 2 ;; esac
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
node_script > "$tmp"
if [ "${1:-}" = --check ]; then bash "$tmp" --check; else sudo bash "$tmp" "${1:-}"; fi
if [ -z "${1:-}" ]; then
  echo "applied. Restart the server so it starts without swap (sudo systemctl restart gb10-sglang," \
    "or Ctrl-C and ./serve.sh); ./scripts/host-tuning.sh --undo takes it all back."
fi
