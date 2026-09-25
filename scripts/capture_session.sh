#!/usr/bin/env bash
# Record a rig session together with the UWB ground truth that makes it calibratable.
#
# The two have to start together: a recording without concurrent telemetry has no
# truth to fit against, which is what made every session before 2026-09-17 unusable
# for calibration. Frame filenames carry a wall-clock ms stamp and UWB `host_time`
# is wall clock too (both within ~0.1 s of this Mac), so the two align on wall time.
#
#   scripts/capture_session.sh 120                  # 2 minutes, cam0, colour, native
#   scripts/capture_session.sh 120 vessel-run-3     # with a label
#
# Stops cleanly on Ctrl-C. Recordings stay on the rig; only the telemetry lands here.
set -euo pipefail

RIG="${RIG:-192.168.1.104}"
CAM="${CAM:-cam0}"                 # cam0 is the FORWARD camera on ship 1 - the one that sees vessels
PIXFMT="${PIXFMT:-bgr}"            # colour: a grey recording can never be made colour afterwards
RES="${RES:-native}"
MODE="${MODE:-record}"             # record only: mode=detect builds the tag detector
DETECTOR="${DETECTOR:-apriltag}"   # MUST be the CPU detector. The saved pipeline resolves to
                                   # hybrid/gpu, which imports torch, and this rig's torch is
                                   # built for CUDA 13 while the box has 12.6 - it dies on a
                                   # missing libcudart.so.13 before the first frame.
BROKER="${BROKER:-192.168.1.110}"
GB_PER_MIN=10                      # raw, per camera

SECONDS_TO_RECORD="${1:?usage: capture_session.sh SECONDS [LABEL]}"
LABEL="${2:-session}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="results/capture_${STAMP}_${LABEL}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$OUT"

need_gb=$(python3 -c "print(round($SECONDS_TO_RECORD/60*$GB_PER_MIN,1))")
free_gb=$(curl -fsS -m 10 "http://$RIG:8080/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["disk_free_gb"])')
echo "rig $RIG: ${free_gb} GB free, this run needs about ${need_gb} GB"
python3 -c "import sys; sys.exit(0 if $free_gb - $need_gb > 5 else 1)" || {
  echo "REFUSING: less than 5 GB would be left. Free space on the rig first." >&2; exit 1; }

busy=$(curl -fsS -m 10 "http://$RIG:8080/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["running"])')
[ "$busy" = "False" ] || { echo "REFUSING: the rig is already recording." >&2; exit 1; }

echo "starting UWB/IMU logger -> $OUT/telemetry.jsonl"
venv/bin/python rig_state_logger.py record --broker "$BROKER" \
  --seconds "$((SECONDS_TO_RECORD + 10))" --out "$OUT/telemetry.jsonl" >"$OUT/logger.log" 2>&1 &
LOGGER=$!
sleep 3                            # let the logger connect and settle before frame one

stop_rig() {
  echo "stopping rig recording"
  curl -fsS -m 60 -X POST "http://$RIG:8080/stop?overlay=0" || true   # overlay would re-run the broken GPU detector
  echo
}
trap 'stop_rig; wait "$LOGGER" 2>/dev/null || true' INT TERM

echo "starting rig recording: mode=$MODE cams=$CAM res=$RES pixfmt=$PIXFMT detector=$DETECTOR"
# /start answers 200 even when it refuses, so the ok field is what must be checked.
# A live preview holds the camera (Argus is exclusive) and is the usual reason.
curl -fsS -m 30 -X POST "http://$RIG:8080/preview/stop" >/dev/null 2>&1 || true
reply=$(curl -fsS -m 30 -X POST \
  "http://$RIG:8080/start?mode=$MODE&fmt=raw&cams=$CAM&res=$RES&pixfmt=$PIXFMT&detector=$DETECTOR")
echo "$reply"
python3 -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get('ok') else 1)" "$reply" || {
  echo "REFUSING: the rig did not start recording." >&2; kill "$LOGGER" 2>/dev/null || true; exit 1; }
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 2
  live=$(curl -fsS -m 10 "http://$RIG:8080/status" | python3 -c 'import json,sys; print(json.load(sys.stdin)["running"])')
  [ "$live" = "True" ] && break
done
[ "$live" = "True" ] || { echo "REFUSING: worker started but died." >&2; kill "$LOGGER" 2>/dev/null || true; exit 1; }
echo "confirmed: rig worker is recording"
date -u +"recording started %Y-%m-%dT%H:%M:%SZ" | tee "$OUT/started_utc.txt"

sleep "$SECONDS_TO_RECORD"
trap - INT TERM
stop_rig

echo "waiting for the telemetry logger to finish"
wait "$LOGGER" 2>/dev/null || true

session=$(curl -fsS -m 15 "http://$RIG:8080/recordings" \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)["sessions"]; print(s[0]["id"] if s else "")')
frames=$(ssh -i "$HOME/.ssh/id_ed25519_shipcam" -o BatchMode=yes -o ConnectTimeout=10 "root@$RIG" \
  "python3 -c \"
import glob,re,os,json
fs=glob.glob('/root/shipcaps/$session/*/*.bgr')
t=sorted(int(re.search(r'_(\\d+)\\.bgr',os.path.basename(p)).group(1))/1000 for p in fs)
print(json.dumps({'n':len(t),'first':t[0] if t else 0,'last':t[-1] if t else 0}))\"")

python3 - "$OUT" "$session" "$RIG" "$CAM" "$SECONDS_TO_RECORD" "$frames" <<'PY'
import json, sys, pathlib
out, session, rig, cam, secs, frames = sys.argv[1:7]
f = json.loads(frames)
rows = [json.loads(l) for l in open(pathlib.Path(out, "telemetry.jsonl"))]
ts = [r["received_at"] for r in rows]
topics = {}
for r in rows:
    topics[r["topic"]] = topics.get(r["topic"], 0) + 1
# The only question that decides whether this session is calibratable.
lo, hi = max(f["first"], min(ts)), min(f["last"], max(ts))
overlap = max(0.0, hi - lo)
span = f["last"] - f["first"]
report = {"rig": rig, "session_id": session, "camera": cam, "requested_s": int(secs),
          "frames": f["n"], "frame_span_s": round(span, 1),
          "telemetry_span_s": round(max(ts) - min(ts), 1),
          "overlap_s": round(overlap, 1),
          "frames_with_truth": int(overlap / span * f["n"]) if span > 0 else 0,
          "uwb_ship1": sum(v for k, v in topics.items() if "/1/uwb/position" in k),
          "uwb_ship2": sum(v for k, v in topics.items() if "/2/uwb/position" in k),
          "topics": topics,
          "note": "frames stay on the rig under /root/shipcaps/<session_id>"}
pathlib.Path(out, "session.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"\nrig session      : {session}  ({f['n']} frames, {span:.1f}s)")
print(f"telemetry        : {out}/telemetry.jsonl")
print(f"OVERLAP          : {overlap:.1f}s  -> about {report['frames_with_truth']} frames carry ground truth")
print(f"UWB fixes        : ship1 {report['uwb_ship1']}, ship2 {report['uwb_ship2']}")
for name, n in (("ship 1", report["uwb_ship1"]), ("ship 2", report["uwb_ship2"])):
    if n == 0:
        print(f"  WARNING: no UWB from {name} - there is no true range without it")
if overlap < span * 0.9:
    print("  WARNING: telemetry does not cover the whole recording")
PY
echo "done. Frames remain on the rig; nothing large was copied here."
