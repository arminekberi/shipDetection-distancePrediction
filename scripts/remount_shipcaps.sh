#!/usr/bin/env bash
# Re-mount a rig's /root/shipcaps as a local dir over rclone's NFS loopback.
#
#   scripts/remount_shipcaps.sh            -> shipcam:  (192.168.1.104, ShipCam0) -> shipcaps_remote/
#   scripts/remount_shipcaps.sh shipcam1   -> shipcam1: (192.168.1.107, ShipCam1) -> shipcaps1_remote/
#
# macOS sometimes silently drops these mounts (e.g. after sleep/network changes) with
# no error in the log - the mount point then reverts to a plain EMPTY LOCAL DIR, so
# reads of source video fail and writes land on the Mac disk instead of the rig.
# If `ls <mountpoint>` comes up empty, just rerun this.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

REMOTE="${1:-shipcam}"
case "$REMOTE" in
  shipcam)  MOUNT="shipcaps_remote";  LOG=".rclone-mount.log" ;;
  shipcam1) MOUNT="shipcaps1_remote"; LOG=".rclone-mount1.log" ;;
  *) echo "unknown remote: $REMOTE (expected 'shipcam' or 'shipcam1')" >&2; exit 2 ;;
esac

# The pattern has to pin the remote with its colon: a bare "shipcam" also matches
# the shipcam1 process, so remounting one rig would tear down the other.
pkill -f "rclone nfsmount $REMOTE:" 2>/dev/null || true
umount "$MOUNT" 2>/dev/null || true
sleep 1

mkdir -p "$MOUNT"
nohup rclone nfsmount "$REMOTE:/root/shipcaps" "$MOUNT" \
  --vfs-cache-mode=full \
  --dir-cache-time=1m \
  --log-file="$LOG" \
  --log-level=INFO \
  > /dev/null 2>&1 &
disown

sleep 4
if mount | grep -q "on $(pwd)/$MOUNT "; then
  echo "mounted: $MOUNT/ -> $REMOTE:/root/shipcaps"
  ls "$MOUNT" | head -5
else
  echo "mount did not come up - check $LOG" >&2
  exit 1
fi
