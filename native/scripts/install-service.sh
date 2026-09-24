#!/usr/bin/env bash
# Install ./serve.sh as a systemd service, gb10-sglang.service. Needs sudo.
#
#   ./scripts/install-service.sh              # install, enable at boot, start
#   ./scripts/install-service.sh --no-start   # install and enable only
#
# The service runs ./serve.sh as you, from this checkout, so its settings are
# that file's: edit it, then `sudo systemctl restart gb10-sglang`. Logs go to
# the journal, which rotates them: `journalctl -u gb10-sglang -f`.
#
# MemoryMax is the cgroup limit docker --memory used to set (both write
# memory.max), so a runaway process is killed before the host starves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT=gb10-sglang
RUN_USER="${SUDO_USER:-$USER}"
MEMORY_MAX="${SERVICE_MEMORY_MAX:-110G}"

[ "$RUN_USER" != root ] || { echo "run this as your own user (it calls sudo itself)" >&2; exit 1; }

sudo tee "/etc/systemd/system/$UNIT.service" >/dev/null <<EOF
[Unit]
Description=SGLang: Qwen3.8-27B + DFlash2 (gb10 recipe)
Wants=network-online.target
After=network-online.target
# A config that cannot boot should stop, not loop: 3 tries per 20 minutes.
StartLimitIntervalSec=1200
StartLimitBurst=3

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$ROOT
ExecStart=$ROOT/serve.sh
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
if [ "${1:-}" != --no-start ]; then
  sudo systemctl restart "$UNIT"
  echo "started. Follow the boot with: journalctl -u $UNIT -f"
  echo "ready when: curl -s http://127.0.0.1:${PORT:-8888}/v1/models"
fi
