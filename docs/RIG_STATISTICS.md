# Reproducible rig statistics

Prepared and smoke-tested on 2026-09-14. Run from the repository root. Existing
rig settings and recordings are read; these commands do not change camera
settings or install/switch models on the rigs.

## Inventory and current state

```sh
mkdir -p results/rig_review
ssh -i "$HOME/.ssh/id_ed25519_shipcam" -o BatchMode=yes root@192.168.1.104 python3 - \
  < scripts/rig_inventory.py > results/rig_review/rig0.json
ssh -i "$HOME/.ssh/id_ed25519_shipcam" -o BatchMode=yes root@192.168.1.107 python3 - \
  < scripts/rig_inventory.py > results/rig_review/rig1.json
```

The standalone standard-library script reports capture manifests, actual
timestamp-based FPS, raw file-size mismatches, filename-order time reversals,
disk space, package versions, model hashes and live API state. It reads both
`.bgr` and legacy `.gray` frames without loading entire recordings into memory.
Per-frame exposure and gain are not inferred from configured limits.

## Detector and background statistics on recordings

After choosing representative recordings, use a native raw camera directory
when possible; a compressed preview loses small-target detail. The example
camera names must match the rig. A read-through mount must be available first.

```sh
venv/bin/python compare_trials.py \
  rig0=shipcaps_remote/20260911-144611/ShipCam0 \
  rig1=shipcaps1_remote/20260911-144607/ShipCam1 \
  --weights weights/boat_v4_s_best.pt --threads 4 \
  --sample-every 1 --max-frames 100 --save-overlays \
  --out results/rig_review/recordings
```

These historical example scenes are not a paired exposure experiment. For an
exposure comparison use A/B/A-repeat recordings with comparable scene, lighting
and target motion, and change one camera parameter at a time.

The output includes source FPS/resolution, processed frame counts, detection
coverage, highlight fractions in the configured water band and tracked boxes,
target sharpness, track counts, p50/p95 detector and pipeline latency, sampling
throughput and sampled overlays. Coverage does not measure recall: false
detections can increase it. Sampling throughput includes decoding, warmup and
statistics; detector latency excludes those. The pipeline here has no depth.

Executing this on the Mac measures the Mac. Actual Jetson latency needs this
same measured pipeline (or the actual production backend) on the Jetson with
compatible dependencies; the inventory alone does not benchmark GPU inference.
Record power mode, temperature, resolution, backend, checkpoint hash and camera
settings beside that run. Do not compare timing from different hardware as an
architecture improvement.

## Labeled comparison and report generation

```sh
venv/bin/python evaluate_models.py \
  --data yolo_dataset_v4_color --splits val test \
  --weights weights/boat_v4_s_best.pt weights/boat_v4s_frozen_best.pt \
    weights/boat_v4s_frozen_color_best.pt weights/boat_v4s_frozen_v3_best.pt \
    weights/boat_v4s_frozen_v3_unfrozen_best.pt weights/boat_v5_s_best.pt \
  --out results/rig_review/models
venv/bin/python scripts/report_statistics.py results/rig_review
```

`report_statistics.py` explicitly separates the current 704 newer and 111
historical test images and checks identical frame/label pairing across models.
Review and update that provenance grouping when the test set changes. Results
are Markdown, JSON and a standalone PNG chart. Pin thresholds before looking at
test results; model selection belongs on reviewed validation sessions.

For foreground/background separation, collect masks and reviewed empty scenes
as described in [FOREGROUND_BACKGROUND_PLAN.md](FOREGROUND_BACKGROUND_PLAN.md).
Range MAE/RMSE requires a separate table of independently measured target
distances and frame timestamps. Tracking metrics require target identities
throughout the labeled sequence. Leave those metrics unreported until that
ground truth is available.
