#!/usr/bin/env bash
# scripts/test-mem-prep.sh: scripts/mem-prep.sh against stub nvidia-smi, docker,
# pgrep, sudo, ssh and a made-up /proc/buddyinfo; changes nothing on the
# machine. Checks: the 2 MiB+ share of free memory (4 KiB and 64 KiB pages);
# idle / busy from a GPU process, a GPU container, a server process, on the
# worker too; ALLOW_BUSY_GPU; an unreachable worker; prep drops and compacts
# only with the GPU idle and passwordless sudo, and logs each start.
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)/mem-prep.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/bin" "$T/vm" "$T/home"
# 4 KiB pages: order 9 = 2 MiB. 100 free blocks at order 0 (400 KiB), 3 at order 9 (6 MiB), 2 at order 10 (8 MiB).
printf 'Node 0, zone   Normal %s\n' "100 0 0 0 0 0 0 0 0 3 2" > "$T/buddyinfo"
cat > "$T/bin/getconf" <<STUB
#!/bin/sh
echo \${FAKE_PAGESIZE:-4096}
STUB
cat > "$T/bin/nvidia-smi" <<'STUB'
#!/bin/sh
[ -n "${FAKE_GPU_APP:-}" ] && echo "$FAKE_GPU_APP"
exit 0
STUB
cat > "$T/bin/docker" <<'STUB'
#!/bin/sh
case "$1" in
  ps) [ -n "${FAKE_CTR:-}" ] && echo abc123 ;;
  inspect) echo "/${FAKE_CTR} ${FAKE_CTR_REQ:-null} []" ;;
esac
exit 0
STUB
cat > "$T/bin/pgrep" <<'STUB'
#!/bin/sh
[ -n "${FAKE_SERVER:-}" ] && { echo "4242 $FAKE_SERVER"; exit 0; }
exit 1
STUB
cat > "$T/bin/id" <<'STUB'
#!/bin/sh
case "$1" in -u) echo 1000 ;; -un) echo tester ;; esac
STUB
cat > "$T/bin/sudo" <<'STUB'
#!/bin/sh
[ "$1" = -n ] && shift
[ -n "${FAKE_NO_SUDO:-}" ] && exit 1
exec "$@"
STUB
# ssh <opts> target bash -s -- cmd: run it here, as the worker (FAKE_WORKER_* become the worker's state)
cat > "$T/bin/ssh" <<'STUB'
#!/bin/bash
while [[ $1 == -* ]]; do [[ $1 == -o ]] && shift; shift; done
t=$1; shift
[ "$t" = down ] && exit 255
FAKE_GPU_APP=${FAKE_WORKER_GPU_APP:-} FAKE_CTR= FAKE_SERVER= "$@"
STUB
printf '#!/bin/sh\necho %s\n' testhost > "$T/bin/hostname"
chmod +x "$T/bin/"*
run() { env PATH="$T/bin:$PATH" HOME="$T/home" GB10_BUDDYINFO="$T/buddyinfo" GB10_VM_DIR="$T/vm" bash "$src" "$@"; }
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; echo "$out"; exit 1; fi; }

out=$(run status)
check "status: free and the 2 MiB+ part of it (4 KiB pages)" "echo \"\$out\" | grep -q 'free 0.0 GiB, 0.0 GiB'"
printf 'Node 0, zone   Normal %s\n' "0 0 0 0 0 0 0 0 0 0 65536" > "$T/buddyinfo"     # 65536 x 4 MiB = 256 GiB
printf 'Node 0, zone   Normal %s\n' "262144 0 0 0 0 0 0 0 0 0 0" >> "$T/buddyinfo"      # + 1 GiB in 4 KiB pages
out=$(run status)
check "status: 257 GiB free, 256 in 2 MiB+ blocks" "echo \"\$out\" | grep -q 'free 257.0 GiB, 256.0 GiB of it in 2 MiB+'"
out=$(FAKE_PAGESIZE=65536 run status)
check "status: with 64 KiB pages order 5 is 2 MiB" "echo \"\$out\" | grep -q 'free 4112.0 GiB, 4096.0 GiB'"

out=$(run idle 2>&1); rc=$?
check "idle: nothing on the GPU, exit 0" "[ $rc = 0 ] && [ -z \"\$out\" ]"
set +e
out=$(FAKE_GPU_APP="77, python3" run idle 2>&1); rc=$?
check "idle: a GPU process, exit 1 and its pid" "[ $rc = 1 ] && echo \"\$out\" | grep -q 'GPU process 77 python3' && echo \"\$out\" | grep -q 'ALLOW_BUSY_GPU'"
out=$(FAKE_CTR=glm53-flash-tf FAKE_CTR_REQ='[{\"Capabilities\":[[\"gpu\"]]}]' run idle 2>&1); rc=$?
check "idle: a container with a GPU" "[ $rc = 1 ] && echo \"\$out\" | grep -q 'GPU container glm53-flash-tf'"
out=$(FAKE_CTR=gpu-named-but-cpu-only run idle 2>&1); rc=$?
check "idle: a container without a GPU does not count" "[ $rc = 0 ]"
out=$(FAKE_SERVER="python -m sglang.launch_server --model x" run idle 2>&1); rc=$?
check "idle: a server process" "[ $rc = 1 ] && echo \"\$out\" | grep -q 'server process 4242'"
out=$(FAKE_WORKER_GPU_APP="9, tensorfold" run idle worker 2>&1); rc=$?
check "idle: busy on the worker fails it too" "[ $rc = 1 ] && echo \"\$out\" | grep -q 'GPU process 9 tensorfold'"
out=$(ALLOW_BUSY_GPU=1 FAKE_GPU_APP="77, python3" run idle 2>&1); rc=$?
check "idle: ALLOW_BUSY_GPU=1 goes on" "[ $rc = 0 ] && echo \"\$out\" | grep -q 'going on beside it'"
out=$(run idle down 2>&1); rc=$?
check "idle: an unreachable worker fails, and says so" "[ $rc = 1 ] && echo \"\$out\" | grep -q 'cannot be reached'"
set -e

out=$(FAKE_GPU_APP="77, python3" run prep)
check "prep: GPU in use, nothing written" "echo \"\$out\" | grep -q 'left as it is' && [ ! -e '$T/vm/drop_caches' ]"
out=$(FAKE_NO_SUDO=1 run prep)
check "prep: no passwordless sudo, nothing written, says how" "echo \"\$out\" | grep -q 'mem-prep.sh sudoers' && [ ! -e '$T/vm/drop_caches' ]"
out=$(run prep worker)
check "prep: drops and compacts here and on the worker" "[ \"\$(cat '$T/vm/drop_caches')\" = 1 ] && [ \"\$(cat '$T/vm/compact_memory')\" = 1 ] && [ \"\$(echo \"\$out\" | grep -c 'after the cache drop and compaction')\" = 2 ]"
check "prep: each start logged" "[ \"\$(grep -c ' dropped free=257.0 big=256.0 ' '$T/home/.cache/gb10/mem-prep.log')\" = 2 ] && grep -q ' as-is ' '$T/home/.cache/gb10/mem-prep.log'"
echo "all checks passed"
