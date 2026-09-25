# Ship detection and distance estimation

## About

Offline Python tools for two instrumented model boats ("rigs", ShipCam0 and
ShipCam1) in a covered test basin. The project:

- detects boats in rig recordings with a fine-tuned YOLO model and tracks them
  (CSRT, Kalman or ByteTrack with persistent ids);
- estimates range with Depth Anything V2 or from the waterline angle;
- estimates the other vessel's position, course and speed from bearings
  (target motion analysis, TMA);
- records UWB/IMU telemetry from the rigs over MQTT and fits camera calibration
  against it;
- simulates the basin (digital twin) so the pipeline can be scored against a
  known truth;
- ships a local web viewer for processed recordings and a dataset review UI for
  labeling.

The software running *on* the rigs lives in the separate `hydrolink` repository.

## Stack

- **Frontend:** static HTML/CSS/vanilla JS in `viewer/`, no build step
- **Backend:** Python 3.12, `http.server` from the standard library (`viewer/server.py`)
- **ML:** PyTorch 2.8, Ultralytics YOLO 8.4, Hugging Face Transformers (Depth Anything V2), OpenCV 5
- **Database:** SQLite file for dataset review decisions, created on first use
- **Infrastructure:** none to deploy. Optional: the rigs on the lab LAN (HTTP API, SSH, rclone mount, MQTT broker), Google Colab for GPU training
- **External APIs:** Hugging Face Hub and Ultralytics asset downloads (public, no key)

## Project structure

| Path | Purpose |
| --- | --- |
| `boatdet/` | core library: video I/O, detection, tracking, depth, fusion, TMA, UWB, digital twin, dataset I/O |
| `run.py` | main entry point: process a video or camera into an annotated video + CSV |
| `viewer/` | local web server, recording viewer (`control_panel.html`), dataset review UI (`dataset_review.html`) |
| `weights/` | trained boat detectors (`*.pt`) and training provenance of the latest one |
| `calibration/` | example manifest and state CSV for the two-rig calibration contract |
| `scripts/` | rig capture/processing, rclone mount, training retry loop, reports |
| `tests/` | `unittest` suite and two Node.js checks for the viewer JS |
| `docs/` | audits, evaluation results, design notes; start with the files linked below |
| `label_video.py`, `hard_mining.py`, `audit_dataset.py`, `repair_dataset.py`, `prepare_reviewed_training.py`, `review_background.py` | dataset labeling, mining, audit and repair |
| `train.py`, `train_reviewed_yolo.py`, `colab_train.ipynb`, `train_rig_state.py` | training |
| `pool_sim.py`, `tma_sim.py`, `bearing_noise.py` | simulation and error budget |
| `rig_state_logger.py`, `rig_state_producer.py`, `track_target.py`, `bearings_to_observations.py`, `rig_calibration.py` | rig telemetry, tracking and calibration pipeline |
| `evaluate_*.py`, `compare_trials.py`, `depth_benchmark.py`, `analyze_glare.py`, `calibrate.py` | measurement tools |

Generated and local-only folders (git-ignored, not in the handoff archive):
`venv/`, `results/`, `runs/`, `yolo_dataset_v4*`, `shipcaps_remote/`, `shipcaps1_remote/`.

## Requirements

- Python **3.12** (the pinned `numpy==2.5.3` has no Python 3.11 build)
- Node.js 18+ only for the two JS tests
- ~1.2 GB disk for the virtualenv, ~400 MB for the depth model cache
- GPU optional: CUDA, Apple MPS or CPU is picked automatically (`boatdet/device.py`)
- A desktop session for live camera input and interactive labeling
- For rig work only: LAN access to the rigs, the rig SSH key, `rclone` with remotes `shipcam` and `shipcam1`

## Installation

```sh
python3.12 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.lock.txt
```

`requirements.txt` lists the direct dependencies; `requirements.lock.txt` pins
every package to the versions the project was tested with. The Depth Anything
model (`depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf`) downloads on the
first `run.py`; `train.py` presets download `yolov8s.pt` / `yolo11m.pt`.

Datasets and review data are too large for the code archive and come separately
as `ship-detection-data.zip`. Both archives contain a top-level `ship-detection/`
folder, so unzip the data archive in the same directory as the code archive and it
merges into place (see [Database](#database)).

## Environment variables

```sh
cp .env.example .env
set -a; source .env; set +a      # nothing loads .env automatically
```

Offline work needs none of them. All have defaults for the lab network:

| Variable | Used by | Purpose |
| --- | --- | --- |
| `RIG` | `scripts/capture_session.sh`, `scripts/process_capture.sh`, `track_target.py` | rig IP (HTTP API on :8080, SSH) |
| `RIG_HOST` | `scripts/process_on_rig.sh` | SSH target, `root@<ip>` |
| `RIG_KEY` | `scripts/process_on_rig.sh`, `track_target.py` | SSH private key for the rigs |
| `BROKER` | `rig_state_logger.py`, `rig_state_producer.py`, `scripts/capture_session.sh` | MQTT broker with UWB/IMU telemetry (port 1883, no auth) |
| `HYDROLINK_DIR` | `track_target.py` | `hydrolink` checkout with the camera intrinsics |
| `WEIGHTS` | `scripts/process_on_rig.sh` | detector checkpoint |

The capture scripts also take per-run knobs (`CAM`, `PIXFMT`, `WIDTH`, …);
they are documented at the top of each script.

## Development

Process a recording (the main workflow):

```sh
venv/bin/python run.py --camera path/to/ShipCam0.mp4 \
  --yolo-weights weights/boat_v4_s_best.pt --tracker kalman --no-display \
  --output results/review.mp4 --log results/review.csv
```

`--tracker csrt|kalman|bytetrack|none`; `bytetrack` adds persistent ids, TTC,
optional segmentation, calibration, IMU attitude and TMA
(see [docs/TRACKING_SEGMENTATION.md](docs/TRACKING_SEGMENTATION.md), [docs/TMA.md](docs/TMA.md)).
`--camera 0` uses a live camera. `--max-frames N` for a quick check.

Viewer and dataset review UI:

```sh
venv/bin/python viewer/server.py
```

Other entry points:

```sh
venv/bin/python pool_sim.py run --scenario weave --out results/sim/weave --check   # basin digital twin
venv/bin/python tma_sim.py --scenario all --csv results/tma_sim.csv                # TMA on simulated truth
venv/bin/python train.py --help                                                    # training presets
venv/bin/python label_video.py --plan                                              # dataset plan
venv/bin/python audit_dataset.py yolo_dataset_v4_color --out results/audit.json    # read-only audit
```

Tests:

```sh
venv/bin/python -m unittest discover -s tests
node tests/test_panel_data.js && node tests/test_dataset_review.js
venv/bin/python -m pip check
```

There is no linter or type checker configured.

## Production build

There is no build step and nothing to deploy: the project is a set of Python
command-line tools plus a local, loopback-only viewer. "Production" means running
`run.py` / the scripts from an installed virtualenv. The viewer server is a
development tool and is not meant to be exposed on a network.

## Docker

Not used.

## Database

The only database is the SQLite file of the dataset review UI:
`results/manual_training_review_20260914/review/decisions.sqlite3`. Tables are
created automatically; there are no migrations or seeds. It holds 5,016 manual
review decisions and ships in `ship-detection-data.zip` together with:

- `yolo_dataset_v4_color/` — the default training dataset of `train.py`
- `results/manual_training_review_20260914/` — source images and evidence the review UI reads
- `results/reviewed_yolo_20260917/` — the reviewed train/val/test split built from those decisions
- `results/capture_*`, `results/rig_calib_20260917/` — rig captures with UWB ground truth (not repeatable)

Without it, everything except the dataset review UI and training still runs.
Not included: `yolo_dataset_v4_mono/` (evidence only), `manual_source.tar` (the
original archive of `source_dataset/`), processed videos and audit outputs under
`results/`, which the tools regenerate.

## Main URLs

- Recording viewer: <http://127.0.0.1:8001/control_panel.html> (`?results=local` reads `results/by_recording/`)
- Dataset review: <http://127.0.0.1:8001/dataset_review.html>
- Viewer API: `GET /api/status`, `POST /api/process` with `{"video": "<relative path>", "target": "x1,y1,x2,y2"}`
- Rig control API (on the rig, not this project): `http://<RIG>:8080/status`

## Important notes

- **Color input only.** Every checkpoint collapsed on grayscale input; train and
  run on color recordings. `yolo_dataset_v4_mono` is evidence, not a training target.
- **Default detector** is `weights/boat_v4_s_best.pt` (viewer and scripts).
  `boat_reviewed_20260917_best.pt` is the latest fine-tune from reviewed labels;
  its `args.yaml`/`results.csv` sit next to it. Evaluate before switching.
- **Distance calibration defaults to identity.** Fit one with `calibrate.py` for
  the actual rig; a historical pool fit is not universal.
- **Bearings-only range is not observable in the basin** (not enough parallax);
  there the waterline gives range and TMA gives course/speed plus an ambiguity flag.
- **Keep whole recordings and capture sessions inside one dataset split.**
  Read [docs/AUDIT.md](docs/AUDIT.md) and [docs/DATASET_REPAIR.md](docs/DATASET_REPAIR.md) before training.
- **rclone mounts drop silently** on macOS: an empty `shipcaps_remote/` means
  rerun `scripts/remount_shipcaps.sh` (or `... shipcam1`).
- The dataset review UI, `review_background.py` output and
  [docs/RIG_CALIBRATION.md](docs/RIG_CALIBRATION.md) are in Russian; everything else is in English.
- Recording paths with Turkish characters are data identifiers; keep them as they are.

Further reading: [HANDOFF.md](HANDOFF.md), [latest audit](docs/AUDIT_20260914.md),
[evaluation](docs/EVALUATION.md), [simulation](docs/SIMULATION.md),
[two-rig calibration](docs/RIG_CALIBRATION.md), [dataset review UI](docs/DATASET_REVIEW_UI.md).
