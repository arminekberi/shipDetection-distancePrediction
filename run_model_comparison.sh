#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source venv/bin/activate || exit 1
python model_comparison.py >> results/model_comparison/full_run_log.txt 2>&1
echo "MODEL COMPARISON FULL RUN DONE (exit $?)"
