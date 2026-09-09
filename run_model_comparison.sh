#!/bin/bash
cd /Users/armin/Desktop/depth-anything
source venv/bin/activate
python model_comparison.py >> results/model_comparison/full_run_log.txt 2>&1
echo "MODEL COMPARISON FULL RUN DONE (exit $?)"
