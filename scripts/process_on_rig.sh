#!/usr/bin/env bash
# Process rig recordings locally, upload results to the rig over ssh, delete local copies.
#   scripts/process_on_rig.sh 20260909-120335 20260904-113444 ...
# Source video is read through the rclone mount; uploads never go through it
# (a dropped mount silently turns into a local dir and "succeeds").
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source venv/bin/activate || exit 1

HOST="${RIG_HOST:-root@192.168.1.104}"
KEY="${RIG_KEY:-$HOME/.ssh/id_ed25519_shipcam}"
REMOTE_ROOT="/root/shipcaps/_processed"
WEIGHTS="${WEIGHTS:-weights/boat_v4_s_best.pt}"
STAGING="$(mktemp -d)"
trap 'rmdir "$STAGING" 2>/dev/null' EXIT
rig() { ssh -i "$KEY" -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "$@"; }

for id in "$@"; do
  folder="shipcam0_${id//-/_}"
  dir="$STAGING/$folder"
  mkdir -p "$dir"
  rig true || { echo "rig unreachable, stopping before $id"; break; }

  echo "=== $(date +%H:%M:%S) $id -> $folder"
  if ! python run.py --camera "shipcaps_remote/$id/ShipCam0.mp4" --no-display \
      --yolo-weights "$WEIGHTS" --tracker csrt \
      --output "$dir/$folder.mp4" --log "$dir/$folder.csv" > "$dir/process_log.txt" 2>&1 \
      || [ ! -s "$dir/$folder.mp4" ]; then
    echo "  generation failed, log kept at $dir/process_log.txt"
    continue
  fi

  rig "mkdir -p $REMOTE_ROOT/$folder" && \
    scp -i "$KEY" -q "$dir"/* "$HOST:$REMOTE_ROOT/$folder/" || { echo "  upload failed, kept $dir"; continue; }
  local_size=$(stat -f%z "$dir/$folder.mp4")
  remote_size=$(rig "stat -c%s $REMOTE_ROOT/$folder/$folder.mp4" 2>/dev/null)
  if [ "$local_size" = "$remote_size" ]; then
    echo "  uploaded ($local_size bytes)"
    rm -rf "$dir"
  else
    echo "  size mismatch (local=$local_size remote=$remote_size), kept $dir"
  fi
done
