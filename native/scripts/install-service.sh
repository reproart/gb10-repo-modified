#!/usr/bin/env bash
# Install ./serve.sh as a systemd service, gb10-sglang.service. Needs sudo.
#
#   ./scripts/install-service.sh                         # default profile, start now
#   ./scripts/install-service.sh gemma4-31b              # another profile
#   ./scripts/install-service.sh gemma4-31b --no-start   # install and enable only
#
# The service runs `./serve.sh <profile>` as you, from this checkout, so its
# settings are serve.sh's and models/<profile>.sh's: edit them, then
# `sudo systemctl restart gb10-sglang`. One unit, one model: running this again
# with another profile switches it. Logs go to the journal, which rotates
# them: `journalctl -u gb10-sglang -f`.
#
# MemoryMax is the cgroup limit docker --memory used to set (both write
# memory.max), so a runaway process is killed before the host starves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT=gb10-sglang
RUN_USER="${SUDO_USER:-$USER}"
MEMORY_MAX="${SERVICE_MEMORY_MAX:-110G}"
profile="" start=1
for arg in "$@"; do
  case "$arg" in
    --no-start) start=0 ;;
    -*) echo "usage: $0 [profile] [--no-start]" >&2; exit 2 ;;
    *) profile="$arg" ;;
  esac
done
if [ -n "$profile" ] && [ ! -f "$ROOT/models/$profile.sh" ]; then
  echo "no profile models/$profile.sh" >&2; exit 2
fi

[ "$RUN_USER" != root ] || { echo "run this as your own user (it calls sudo itself)" >&2; exit 1; }

sudo tee "/etc/systemd/system/$UNIT.service" >/dev/null <<EOF
[Unit]
Description=SGLang on GB10 (profile: ${profile:-serve.sh default})
Wants=network-online.target
After=network-online.target
# A config that cannot boot should stop, not loop: 3 tries per 20 minutes.
StartLimitIntervalSec=1200
StartLimitBurst=3

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$ROOT
ExecStart=$ROOT/serve.sh${profile:+ $profile}
Restart=on-failure
RestartSec=30
TimeoutStopSec=90
MemoryMax=$MEMORY_MAX
MemorySwapMax=0
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$UNIT"
if [ "$start" = 1 ]; then
  sudo systemctl restart "$UNIT"
  echo "started. Follow the boot with: journalctl -u $UNIT -f"
  echo "ready when: curl -s http://127.0.0.1:${PORT:-8888}/v1/models"
fi
