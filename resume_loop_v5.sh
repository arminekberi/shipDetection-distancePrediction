#!/bin/bash
cd /Users/armin/Desktop/depth-anything
source venv/bin/activate

WEIGHTS="runs/detect/runs_boat_yolo/boat_v4s_hsvaug/weights/last.pt"
LOG="results/yolo_train_v5_hsvaug_log.txt"

while true; do
    if [ -f "$WEIGHTS" ]; then
        echo "=== resume_loop_v5: resuming from checkpoint $(date) ===" >> "$LOG"
        python -c "
from ultralytics import YOLO
model = YOLO('$WEIGHTS')
results = model.train(resume=True)
" >> "$LOG" 2>&1
    else
        echo "=== resume_loop_v5: launching fresh training $(date) ===" >> "$LOG"
        python train_v5_aug.py >> "$LOG" 2>&1
    fi
    STATUS=$?
    echo "=== resume_loop_v5: python exited with status $STATUS $(date) ===" >> "$LOG"
    if [ $STATUS -eq 0 ]; then
        echo "=== resume_loop_v5: training finished normally, stopping loop ===" >> "$LOG"
        break
    fi
    sleep 5
done
echo "YOLO V5 (hsv-aug) TRAIN DONE (resume_loop_v5 complete)"
