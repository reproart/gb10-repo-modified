#!/usr/bin/env bash
# Stop both ranks (Mia's stop.sh: their logs are kept under ~/.cache/tensorfold-glm53/logs).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
need_kit
kit_env
exec "$KIT_DIR/stop.sh" "$@"
