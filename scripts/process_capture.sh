#!/usr/bin/env bash
# Everything that happens AFTER scripts/capture_session.sh: frames and telemetry in,
# a fitted camera calibration and a bearings-only TMA score out.
#
# Split from the capture script on purpose. Capture is the part that cannot be
# repeated - the boats have moved on - so it does the least it can and stops. This
# part is pure post-processing over files already on disk and the rig, and can be
# re-run with a different seed box, depth model or split as many times as needed.
#
#   scripts/process_capture.sh results/capture_20260918-143012_vessel-run-1 \
#       20260918-143012 506,287,605,344
#
# The seed box is x1,y1,x2,y2 of the target in the FIRST exported frame, in the
# working resolution (--width below), because no detector on this rig finds it.
set -euo pipefail

OUT="${1:?usage: process_capture.sh CAPTURE_DIR SESSION_ID SEED_X1,Y1,X2,Y2}"
SESSION="${2:?session id, e.g. 20260918-143012}"
SEED="${3:?seed box x1,y1,x2,y2 in the working resolution}"

RIG="${RIG:-192.168.1.104}"
CAM="${CAM:-cam0}"
WIDTH="${WIDTH:-2304}"             # half native. The waterline angle is what carries
HEIGHT="${HEIGHT:-1296}"           # range, so pixel rows are range metres - do not
                                   # drop to 1280x720 unless the target is very close.
OBSERVER_SHIP="${OBSERVER_SHIP:-1}"; OBSERVER_TAG="${OBSERVER_TAG:-T01}"
TARGET_SHIP="${TARGET_SHIP:-2}";    TARGET_TAG="${TARGET_TAG:-T03}"
HEADING="${HEADING:-0}"            # free for a station-keeping observer: fit absorbs it
DEPTH_MODEL="${DEPTH_MODEL:-waterline}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.6}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -f "$OUT/telemetry.jsonl" ] || { echo "no $OUT/telemetry.jsonl - was this capture_session.sh?" >&2; exit 1; }
PY=venv/bin/python

echo "== 1/6 observer + target state from UWB =="
# --static declares zero velocity: correct for a rig holding station, wrong for one
# under way. The target is the one that moves, so it gets measured velocities.
$PY rig_state_logger.py export "$OUT/telemetry.jsonl" --ship "$OBSERVER_SHIP" --tag "$OBSERVER_TAG" \
  --fixed-heading-deg "$HEADING" --static --out "$OUT/observer.csv" | tail -5
$PY rig_state_logger.py export "$OUT/telemetry.jsonl" --ship "$TARGET_SHIP" --tag "$TARGET_TAG" \
  --fixed-heading-deg 0 --out "$OUT/target.csv" | tail -5

echo "== 2/6 track the target through the recording =="
$PY track_target.py --session "$SESSION" --seed "$SEED" --rig "$RIG" --camera "$CAM" \
  --width "$WIDTH" --height "$HEIGHT" --out "$OUT/track"

echo "== 3/6 bearings -> observations =="
$PY bearings_to_observations.py "$OUT/track/bearings.csv" \
  --depth-model "$DEPTH_MODEL" --camera "$CAM" --out "$OUT/observations.csv"

echo "== 4/6 split and manifest =="
$PY scripts/build_manifest.py "$OUT" --capture-id "$SESSION" \
  --train-fraction "$TRAIN_FRACTION" --observer-heading-deg "$HEADING"

echo "== 5/6 prepare + fit (train only) =="
$PY rig_calibration.py prepare "$OUT/manifest.json" --out "$OUT/dataset" | tail -20
$PY rig_calibration.py fit "$OUT/dataset" --out "$OUT/calibration.json"

echo "== 6/6 evaluate on val: bearing, corrected range, bearings-only TMA =="
$PY rig_calibration.py evaluate "$OUT/dataset" --calibration "$OUT/calibration.json" \
  --split val --out "$OUT/evaluation.json"

echo
echo "calibration : $OUT/calibration.json"
echo "evaluation  : $OUT/evaluation.json"
echo "check the track by eye first: $OUT/track/track_montage.jpg"
