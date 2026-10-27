#!/usr/bin/env bash
# Once (and after KIT_COMMIT changes): check out Mia's recipe at KIT_COMMIT,
# then run its scripts/prepare.sh, which on both Sparks pulls the pinned image
# (or builds it), downloads the checkpoint and DFlash2 into HF_CACHE on the head
# and copies them to the worker over the CX7 link, checked file by file.
# Resumable: every step skips what is already done. ~205 GB a node.
#   ./setup.sh
#   ./setup.sh --rebuild       build the image locally instead of pulling it
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"

if [ ! -d "$KIT_DIR/.git" ]; then
  log "cloning $KIT_REPO into $KIT_DIR"
  git clone -q "$KIT_REPO" "$KIT_DIR"
fi
if [ "$(git -C "$KIT_DIR" rev-parse HEAD)" != "$KIT_COMMIT" ]; then
  git -C "$KIT_DIR" diff --quiet HEAD || { echo "$KIT_DIR has local changes: commit, stash or drop them first" >&2; exit 1; }
  git -C "$KIT_DIR" cat-file -e "$KIT_COMMIT^{commit}" 2>/dev/null || git -C "$KIT_DIR" fetch -q origin
  git -C "$KIT_DIR" checkout -q --detach "$KIT_COMMIT"
fi
log "recipe at $(git -C "$KIT_DIR" log -1 --format='%h %ad %s' --date=short)"

mkdir -p "$HF_CACHE" 2>/dev/null || { echo "cannot create $HF_CACHE: sudo mkdir -p $HF_CACHE && sudo chown $(id -un) $HF_CACHE" >&2; exit 1; }
wssh "mkdir -p '$HF_CACHE'" || { echo "cannot create $HF_CACHE on the worker (same fix there)" >&2; exit 1; }

kit_env
exec "$KIT_DIR/scripts/prepare.sh" "$@"
