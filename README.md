# Boat detection and distance estimation

Python tools for reviewing server recordings, detecting boats with YOLO,
tracking them with CSRT or Kalman, and estimating depth with Depth Anything V2.
The deployed HydroRL AprilTag pipeline is a separate project; its `yildiz`
weights detect markers, not boats.

## Setup

Use Python 3.11 and install the pinned dependencies in a virtual environment:

```sh
python3.11 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

GPU availability depends on your environment. Camera access and interactive
annotation require a desktop session. Model downloads may be needed on first use.

## Process a recording

```sh
venv/bin/python run.py \
  --camera shipcaps_remote/20260910-140833/ShipCam0.mp4 \
  --yolo-weights weights/boat_v4_s_best.pt \
  --tracker kalman --no-display \
  --output results/review.mp4 --log results/review.csv
```

File replay uses synchronous inference. YOLO receives the native decoded frame;
`--width/--height` control tracking/depth display coordinates. Output uses the
source FPS and includes the first frame. H.264 encoding must be supported by
your OpenCV build; an unavailable writer is an error.

The CSV keeps its original first four columns and adds source time and the
uncalibrated model distance. `distance_raw_m` is the **unsmoothed, calibrated**
estimate; `distance_model_m` is the value before calibration. Calibration defaults
to identity. Fit a correction using independent measurements for the actual
model/camera; a historical pool fit is not a universal distance calibration.
A `.meta.json` sidecar records model, arguments, FPS, frame count and run status.
Live asynchronous camera mode remains a preview path and is not frame-exact.

## Review recordings

```sh
venv/bin/python panel_server.py
```

Open <http://127.0.0.1:8001/control_panel.html>. The default viewer reads server
results from `shipcaps_remote/_processed/`. Use `?results=local` for results in
`results/by_recording/`. The old `/kontrol_paneli.html` URL redirects to the
English viewer. The server binds to loopback and serves only viewer assets and
recording directories.

`POST /api/process` accepts JSON with a relative `video` path and optional
`target` (`x1,y1,x2,y2` in 640x360 coordinates). Without a target it uses the
bundled boat v4 detector. Each job gets a new result directory and manifest
entry. The viewer reports measurement coverage, not precision or recall.

## Dataset review and training

```sh
venv/bin/python audit_dataset.py yolo_dataset_v4 --hash-images --out results/dataset_audit.json
venv/bin/python label_video.py --video path/to/recording.mp4 --split train --sample-every 5
```

Audit is read-only and exits 2 when it finds issues, 1 on input/I/O errors.
Check `AUDIT.md` before training the current server dataset. Preserve entire
recordings and capture sessions within a single split. A wrong detection box
is not evidence that the frame contains no boat. Only use `--negative` for
recordings reviewed and confirmed to be empty of boats.

Repeated `ShipCam0.mp4` names now include their parent recording directory in
new label prefixes. Existing legacy names are retained and need provenance
review before importing the same recordings again. Label writes validate and
clip boxes and commit labels only after successful image writes. This is not a
transaction across two files; an interruption may leave an image without a label,
which the dataset auditor detects.

`train_*.py` and the English Colab notebooks are experiment entry points; importing
a training script does not start training. The v6 Colab notebook uses YOLOv8s;
`train_v6.py` uses YOLO11m. They are distinct experiments. Resume shell wrappers
use the current checkout and stop after three failed attempts (`MAX_RETRIES`).
Colab bundles and runtime hardware remain external inputs.

Existing recording paths containing Turkish characters are data identifiers;
they are preserved so the existing dataset and server recordings remain usable.
User-facing text, code comments and project documentation are in English.

## Verification

```sh
venv/bin/python -m unittest discover -s tests -v
node tests/test_panel_data.js
venv/bin/python -m pip check
```

See [AUDIT.md](AUDIT.md) for findings, limits and the retraining sequence, and
[GLARE_REVIEW.md](GLARE_REVIEW.md) for the server glare/color experiment.
