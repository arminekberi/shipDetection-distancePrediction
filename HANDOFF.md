# Handoff

## What this is

A research codebase for detecting and ranging a second vessel from the cameras of
two model boats in a covered test basin. It is a set of offline Python tools and a
local viewer, not a service. Setup and commands are in [README.md](README.md).

## Architecture

```
rig recordings (.mp4 / raw .bgr)  ─┐
live camera                        ├─> run.py ─> boatdet.pipeline
                                   │     detection (YOLO) -> tracking -> depth / waterline range
                                   │     -> fusion -> optional TMA -> overlay video + CSV + .meta.json
rig MQTT telemetry (UWB, IMU) ─────┴─> rig_state_logger.py / rig_state_producer.py -> state CSVs
                                          │
track_target.py -> bearings_to_observations.py -> rig_calibration.py (prepare / fit / evaluate)

boatdet.twin + pool_sim.py : simulated basin writing the same file formats, scored against truth
viewer/server.py           : loopback HTTP server, runs run.py jobs, serves the two web UIs
```

## Main components

| Component | Where | Notes |
| --- | --- | --- |
| Processing pipeline | `boatdet/pipeline.py`, `run.py` | per-frame orchestration and CLI |
| Detection | `boatdet/detection.py`, `boatdet/proposal.py`, `weights/` | Ultralytics YOLO, optional tiling/ROI and two-stage proposals |
| Tracking | `boatdet/tracking.py`, `boatdet/mot.py` | CSRT/Kalman single target, ByteTrack multi-target with occlusion retention |
| Range | `boatdet/depth.py`, `boatdet/geometry.py` | Depth Anything V2; range from the water-contact pixel and the water plane |
| Segmentation / fusion | `boatdet/segmentation.py`, `boatdet/fusion.py` | optional; no segmentation model ships with the repo |
| Target motion analysis | `boatdet/tma.py`, `boatdet/egomotion.py` | bearings-only EKF and range-hypothesis bank |
| Rig telemetry | `boatdet/uwb.py`, `rig_state_logger.py`, `rig_state_producer.py` | UWB multilateration, per-ship state |
| Calibration / supervision | `rig_calibration.py`, `boatdet/supervision.py`, `train_rig_state.py` | train-only fits, evaluated on a held-out split |
| Digital twin | `boatdet/twin.py`, `pool_sim.py` | synthetic basin; all output marked `synthetic: true` |
| Dataset tooling | `boatdet/dataset.py`, `label_video.py`, `hard_mining.py`, `audit_dataset.py`, `repair_dataset.py`, `viewer/dataset_review.*` | review decisions in SQLite |
| Tunables | `boatdet/config.py` | all thresholds and defaults in one place |

The core logic is `boatdet/`; top-level scripts are thin CLIs over it.

## External services

- **Hugging Face Hub**: Depth Anything V2 weights, public, downloaded on first use.
- **Ultralytics**: base YOLO weights for training presets, downloaded on first use.
- **Rigs** (lab LAN only): ShipCam0 `192.168.1.104`, ShipCam1 `192.168.1.107`, each
  with an HTTP control API on `:8080`, root SSH with a key, and `/root/shipcaps`
  mounted locally through rclone. MQTT broker `192.168.1.110:1883` (no auth).
- **Google Colab / Drive**: optional GPU training via `colab_train.ipynb`; the
  notebook clones `github.com/arminekberi/shipDetection-distancePrediction`.
- **hydrolink** repository: needed by `track_target.py` for camera intrinsics.

## What must be set up to run

- Offline (video files, simulation, tests, viewer): Python 3.12 venv from
  `requirements.lock.txt`. Nothing else.
- Training / dataset review: unzip `ship-detection-data.zip` into the project root.
- Rig work: LAN access, the rig SSH private key (`~/.ssh/id_ed25519_shipcam` by
  default), rclone remotes named `shipcam` and `shipcam1`, a `hydrolink` checkout,
  and `.env` if the addresses differ. None of these are in the archive.

## Known limitations

From measurements recorded in `docs/`:

- The trained detector does not find the target vessel in current basin footage;
  `track_target.py` needs a human seed box, and CSRT carries it.
- Bearings-only range is not observable at basin scale; TMA reports `ambiguous`
  with an interval there ([docs/TMA.md](docs/TMA.md)).
- No own-ship navigation source exists, so TMA is validated only in simulation.
- Ship 2's cameras are dead; cam0/cam1 extrinsics on the rig contradict the
  physical camera orientation, so bearings are kept camera-relative.
- Simulation found open issues: TMA false-converges after a long bearing gap
  (`TMA_MAX_DT_S` clamp), the waterline range fit is biased by its own noise, and
  3-anchor UWB fixes weakly pass the RMSE gate ([docs/SIMULATION.md](docs/SIMULATION.md)).
- Distance calibration is identity by default; no maritime segmentation model is shipped.
- Rig addresses have lab-network defaults in code (overridable, see `.env.example`).
- The dataset review export is not a ready training split; recording identity
  still needs review ([docs/DATASET_REVIEW_UI.md](docs/DATASET_REVIEW_UI.md)).

## Before using results for anything that matters

- Run the test suite on the target machine (`unittest` + the two Node checks).
- Evaluate any checkpoint on independently reviewed recordings before making it
  the default in `viewer/server.py` and `scripts/process_on_rig.sh`.
- Fit a distance calibration for the actual rig with `calibrate.py`.
- Do not expose `viewer/server.py` beyond loopback; it runs jobs on request and
  has no authentication.
- Rig scripts start recordings on real hardware; `scripts/capture_session.sh`
  checks disk space and refuses when the rig is busy, but read it before running.
