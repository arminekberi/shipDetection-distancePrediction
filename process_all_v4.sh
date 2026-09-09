#!/bin/bash
cd /Users/armin/Desktop/depth-anything
source venv/bin/activate

WEIGHTS="runs/detect/runs_boat_yolo/boat_v4_s/weights/best.pt"

while IFS='|' read -r video folder; do
  [ -z "$video" ] && continue
  mkdir -p "results/by_recording/$folder"

  for tracker in csrt kalman; do
    out_video="results/by_recording/$folder/yolo_${tracker}_v4s.mp4"
    if [ -f "$out_video" ]; then
      echo "=== $folder : $tracker already done, skipping ==="
      continue
    fi
    echo "=== $folder : $tracker ==="
    python run.py \
      --camera "$video" \
      --width 640 --height 360 \
      --sync --no-display \
      --yolo-weights "$WEIGHTS" \
      --tracker "$tracker" \
      --yolo-conf 0.45 --yolo-low-conf 0.15 --yolo-imgsz 960 \
      --box-smoothing 0.35 --max-jump-frac 0.40 --grace-frames 4 --reacquire-frames 8 \
      --output "results/by_recording/$folder/yolo_${tracker}_v4s.mp4" \
      --log "results/by_recording/$folder/yolo_${tracker}_v4s.csv" \
      > "results/by_recording/$folder/process_${tracker}_v4s_log.txt" 2>&1
    echo "  done $folder/$tracker exit=$?"
  done
done <<'EOF'
testGerçekRenkli.mp4|testGercekRenkli
testGerçekSiyahBeyaz.mp4|testGercekSiyahBeyaz
test1.mp4|test1
test2.mp4|test2
teknedenTekneyeGörüntü/renkliTekneTekne.mp4|renkliTekneTekne
teknedenTekneyeGörüntü/renksizTekneTekne.mp4|renksizTekneTekne
results/siyahBeyazTespit.mp4|siyahBeyazTespit
EOF

echo "ALL 7 VIDEOS x 2 TRACKERS DONE"
