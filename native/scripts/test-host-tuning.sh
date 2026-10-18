#!/usr/bin/env bash
# scripts/test-host-tuning.sh: runs the root part of scripts/host-tuning.sh
# against stub sysctl / systemctl and temporary paths; changes nothing on the
# machine. Checks: the first apply saves the live values and a second keeps
# them; --undo restores them, the earlyoom file and its boot / run state;
# --undo with nothing saved falls back to kernel defaults; a missing or dead
# earlyoom stops the apply and rolls the machine back; --check changes nothing.
set -euo pipefail
src=${1:-$(dirname "$0")/host-tuning.sh}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/etc/sysctl.d" "$T/etc/default" "$T/var/lib"
printf 'EARLYOOM_ARGS="-r 3600"\n' > "$T/etc/default/earlyoom"
printf '%s\n' "vm.swappiness=33" "vm.compaction_proactiveness=7" "vm.extfrag_threshold=444" \
  "vm.min_free_kbytes=67584" "vm.watermark_scale_factor=12" > "$T/state"
cat > "$T/bin/sysctl" <<STUB
#!/usr/bin/env bash
st="$T/state"; log="$T/log"
set_kv() { k=\${1%%=*}; v=\${1#*=}; k=\${k// /}; v=\${v// /}; grep -v "^\$k=" "\$st" > "\$st.n" || true; echo "\$k=\$v" >> "\$st.n"; mv "\$st.n" "\$st"; }
case "\$*" in
  "-n "*) grep "^\$2=" "\$st" | cut -d= -f2 ;;
  "-q -w "*) shift 2; for kv in "\$@"; do set_kv "\$kv"; done ;;
  "-q -p "*) while IFS= read -r l; do [ -n "\$l" ] && set_kv "\$(echo "\$l" | tr -d ' ')"; done < "\$3" ;;
  "-q --system") echo "sysctl --system" >> "\$log" ;;
  *) echo "unexpected sysctl \$*" >&2; exit 9 ;;
esac
STUB
cat > "$T/bin/systemctl" <<STUB
#!/usr/bin/env bash
case "\$*" in
  "is-enabled earlyoom") if [ -f "$T/eo-enabled" ]; then echo enabled; else echo disabled; exit 1; fi ;;
  "is-active"*) [ -f "$T/inactive" ] && exit 3; if [ -f "$T/eo-active" ]; then echo active; else echo inactive; exit 3; fi ;;
  "enable -q earlyoom") touch "$T/eo-enabled"; echo "systemctl \$*" >> "$T/log" ;;
  "disable -q earlyoom") rm -f "$T/eo-enabled"; echo "systemctl \$*" >> "$T/log" ;;
  "restart earlyoom") [ -f "$T/restartfail" ] && exit 1; touch "$T/eo-active"; echo "systemctl \$*" >> "$T/log" ;;
  "stop earlyoom") rm -f "$T/eo-active"; echo "systemctl \$*" >> "$T/log" ;;
  *) echo "systemctl \$*" >> "$T/log" ;;
esac
STUB
touch "$T/eo-enabled" "$T/eo-active"
printf '#!/bin/sh\n' > "$T/bin/earlyoom"
printf '#!/bin/sh\n' > "$T/bin/sleep"
chmod +x "$T/bin/"*
awk '/^  cat <<'"'"'EOF'"'"'$/ {on=1; next} on && /^EOF$/ {exit} on' "$src" \
  | sed -e "s#/etc/sysctl.d#$T/etc/sysctl.d#g" -e "s#/var/lib/#$T/var/lib/#g" \
        -e "s#/etc/default/earlyoom#$T/etc/default/earlyoom#g" > "$T/node.sh"
run() { PATH="$T/bin:$PATH" bash "$T/node.sh" "$@" >/dev/null; }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; cat "$T/state"; exit 1; fi; }
b="$T/var/lib/gb10-host-tuning/sysctl.before"
conf="$T/etc/sysctl.d/99-gb10.conf"
eo="$T/etc/default/earlyoom"

run
check "first apply saves the live values" "grep -q 'vm.swappiness = 33' '$b' && grep -q 'vm.min_free_kbytes = 67584' '$b'"
check "first apply sets the tuned values" "grep -q '^vm.swappiness=10$' '$T/state' && grep -q '^vm.compaction_proactiveness=0$' '$T/state' && grep -q '^vm.min_free_kbytes=1048576$' '$T/state' && grep -q '^vm.watermark_scale_factor=50$' '$T/state'"
args_line=$(grep '^EARLYOOM_ARGS=' "$eo")
case "$args_line" in *"'"*) noq=no ;; *"-M 1048576,524288 "*"--prefer ^(sglang|python|tensorfold|llama-server|VLLM) --avoid ^("*) noq=yes ;; *) noq=no ;; esac
check "first apply writes the earlyoom args, no quotes, the server preferred" "[ $noq = yes ]"
out=$(PATH="$T/bin:$PATH" bash "$T/node.sh" --check)
check "--check reports applied" "echo '$out' | grep -q 'host tuning: applied'"
run
check "second apply keeps the saved values" "grep -q 'vm.swappiness = 33' '$b'"
run --undo
check "undo restores all five values" "grep -q '^vm.swappiness=33$' '$T/state' && grep -q '^vm.compaction_proactiveness=7$' '$T/state' && grep -q '^vm.extfrag_threshold=444$' '$T/state' && grep -q '^vm.min_free_kbytes=67584$' '$T/state' && grep -q '^vm.watermark_scale_factor=12$' '$T/state'"
check "undo removes the saved values and the conf" "[ ! -f '$b' ] && [ ! -f '$conf' ]"
check "undo restores the earlyoom file" "grep -q '^EARLYOOM_ARGS=\"-r 3600\"$' '$eo' && [ ! -f '$eo.pre-gb10' ]"
out=$(PATH="$T/bin:$PATH" bash "$T/node.sh" --check)
check "--check reports not applied, and changes nothing" "echo '$out' | grep -q 'not (fully) applied' && [ ! -f '$conf' ] && [ ! -f '$b' ]"
run
rm -f "$b"
run --undo
check "undo with nothing saved sets kernel defaults" "grep -q '^vm.swappiness=60$' '$T/state' && grep -q '^vm.watermark_scale_factor=10$' '$T/state'"
check "undo with nothing saved reloads the other sysctl files" "grep -q 'sysctl --system' '$T/log'"

mv "$eo" "$T/eo.keep"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply without earlyoom must fail"; exit 1; fi
check "apply without earlyoom stops with a message and changes nothing" "grep -q 'earlyoom is not installed' '$T/err' && [ ! -f '$conf' ] && [ ! -f '$b' ]"
mv "$T/eo.keep" "$eo"
rolled_back="[ ! -f '$conf' ] && [ ! -f '$b' ] && grep -q '^vm.swappiness=60$' '$T/state' \
  && grep -q '^EARLYOOM_ARGS=\"-r 3600\"$' '$eo' && [ ! -f '$eo.pre-gb10' ]"
touch "$T/inactive"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply with a dead earlyoom must fail"; exit 1; fi
check "a dead earlyoom fails the apply and rolls the machine back" "grep -q 'earlyoom did not start' '$T/err' && $rolled_back"
rm -f "$T/inactive"
touch "$T/restartfail"
if PATH="$T/bin:$PATH" bash "$T/node.sh" >/dev/null 2>"$T/err"; then echo "FAIL apply with a failed restart must fail"; exit 1; fi
check "a failed earlyoom restart rolls the machine back" "$rolled_back"
rm -f "$T/restartfail"
e="$T/var/lib/gb10-host-tuning/earlyoom.before"
rm -f "$T/eo-enabled" "$T/eo-active"
run
check "apply enables and starts a disabled, stopped earlyoom" "[ -f '$T/eo-enabled' ] && [ -f '$T/eo-active' ] && grep -q '^disabled inactive$' '$e'"
run --undo
check "undo disables and stops it again" "[ ! -f '$T/eo-enabled' ] && [ ! -f '$T/eo-active' ] && [ ! -f '$e' ]"
echo "all checks passed"
