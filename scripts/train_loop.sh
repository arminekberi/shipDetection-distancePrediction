#!/bin/bash
# Retry train.py with auto-resume after crashes; stops after MAX_RETRIES failures.
#   scripts/train_loop.sh v5_s [train.py args...]
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source venv/bin/activate || exit 1

LOG="results/train_$1.log"
MAX_RETRIES="${MAX_RETRIES:-3}"
mkdir -p results
for ((attempt = 1; attempt <= MAX_RETRIES; attempt++)); do
  echo "=== attempt $attempt $(date) ===" >> "$LOG"
  python train.py "$@" --resume >> "$LOG" 2>&1 && { echo "training finished: $LOG"; exit 0; }
  sleep 5
done
echo "training failed after $MAX_RETRIES attempts; inspect $LOG" >&2
exit 1
