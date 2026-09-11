#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source venv/bin/activate || exit 1

WEIGHTS="runs/detect/runs_boat_yolo/boat_v4_s/weights/last.pt"
LOG="results/yolo_train_v4s_log.txt"

mkdir -p results
MAX_RETRIES="${MAX_RETRIES:-3}"
for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    echo "=== resume_loop: launching training $(date) ===" >> "$LOG"
    python -c "
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
results = model.train(resume=True)
" >> "$LOG" 2>&1
    STATUS=$?
    echo "=== resume_loop: python exited with status $STATUS $(date) ===" >> "$LOG"
    if [ $STATUS -eq 0 ]; then
        echo "=== resume_loop: training finished normally, stopping loop ===" >> "$LOG"
        break
    fi
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        echo "Training failed after $MAX_RETRIES attempts; inspect $LOG" >&2
        exit "$STATUS"
    fi
    sleep 5
done
echo "YOLO V4 (yolov8s) TRAIN DONE (resume_loop complete)"
