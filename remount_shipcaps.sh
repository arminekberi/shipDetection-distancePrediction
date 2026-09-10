#!/usr/bin/env bash
# Re-mount the rig's /root/shipcaps as shipcaps_remote/ over rclone's NFS loopback.
# macOS sometimes silently drops this mount (e.g. after sleep/network changes) with
# no error in the log - if `ls shipcaps_remote` comes up empty, just rerun this.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

pkill -f "rclone nfsmount shipcam" 2>/dev/null || true
sleep 1

mkdir -p shipcaps_remote
nohup rclone nfsmount shipcam:/root/shipcaps shipcaps_remote \
  --vfs-cache-mode=full \
  --dir-cache-time=1m \
  --log-file=.rclone-mount.log \
  --log-level=INFO \
  > /dev/null 2>&1 &
disown

sleep 4
if mount | grep -q "on $(pwd)/shipcaps_remote "; then
  echo "mounted: shipcaps_remote/ -> shipcam:/root/shipcaps"
  ls shipcaps_remote | head -5
else
  echo "mount did not come up - check .rclone-mount.log" >&2
  exit 1
fi
