#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source venv/bin/activate || exit 1

WEIGHTS="runs/detect/runs_boat_yolo/boat_v5_s/weights/last.pt"
LOG="results/yolo_train_v5_log.txt"

mkdir -p results
MAX_RETRIES="${MAX_RETRIES:-3}"
for ((attempt=1; attempt<=MAX_RETRIES; attempt++)); do
    if [ -f "$WEIGHTS" ]; then
        echo "=== resume_loop_v5b: resuming from checkpoint $(date) ===" >> "$LOG"
        python -c "
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
results = model.train(resume=True)
" >> "$LOG" 2>&1
    else
        echo "=== resume_loop_v5b: launching fresh training $(date) ===" >> "$LOG"
        python train_v5.py >> "$LOG" 2>&1
    fi
    STATUS=$?
    echo "=== resume_loop_v5b: python exited with status $STATUS $(date) ===" >> "$LOG"
    if [ $STATUS -eq 0 ]; then
        echo "=== resume_loop_v5b: training finished normally, stopping loop ===" >> "$LOG"
        break
    fi
    if [ "$attempt" -eq "$MAX_RETRIES" ]; then
        echo "Training failed after $MAX_RETRIES attempts; inspect $LOG" >&2
        exit "$STATUS"
    fi
    sleep 5
done
echo "YOLO V5 TRAIN DONE (resume_loop_v5b complete)"
