#!/usr/bin/env bash
# Install the Demolition Pod Quadlet manifest into
# ~/.config/containers/systemd/demolition/ and reload systemd --user.
#
# Idempotent — copying over an existing manifest is fine. The unit is
# `demolition-pod.service` (Quadlet generates this from `demolition.pod`).

set -euo pipefail

POD_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$HOME/.config/containers/systemd/demolition"

log() { printf '[install-quadlet] %s\n' "$*"; }

mkdir -p "$DEST"
cp -f "$POD_ROOT/quadlet/demolition.pod" "$DEST/demolition.pod"
log "installed: $DEST/demolition.pod"

if ! command -v systemctl >/dev/null 2>&1; then
    log "systemctl not found; skip daemon-reload"
    exit 0
fi

if ! systemctl --user list-units >/dev/null 2>&1; then
    log "systemd --user not reachable; skip daemon-reload"
    exit 0
fi

systemctl --user daemon-reload
log "systemd --user daemon-reload OK"

if systemctl --user list-unit-files | grep -q '^demolition-pod\.service'; then
    log "unit available: demolition-pod.service"
    log "start with:    systemctl --user start demolition-pod.service"
    log "status with:   systemctl --user status demolition-pod.service"
else
    log "demolition-pod.service not yet generated (Quadlet may need a podman version refresh)"
fi
