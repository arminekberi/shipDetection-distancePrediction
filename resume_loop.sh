#!/bin/bash
cd /Users/armin/Desktop/depth-anything
source venv/bin/activate

WEIGHTS="runs/detect/runs_boat_yolo/boat_v4_s/weights/last.pt"
LOG="results/yolo_train_v4s_log.txt"

while true; do
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
    sleep 5
done
echo "YOLO V4 (yolov8s) TRAIN DONE (resume_loop complete)"
