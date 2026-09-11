# Paired detector evaluation — 2026-09-10

This is a measured input-preprocessing comparison using existing checkpoints,
not a retrained model or an independent test-set result. It isolates the change
from a 640x360 intermediate resize to the decoded recording resolution. It does
not measure the additional tracking fixes, depth accuracy or the HydroRL tag
proposer patch.

## Protocol

- Source: server recording `20260909-123824/ShipCam0.mp4`, 1280x720,
  246 frames. This is an archived preview, not the original sensor-resolution capture.
- Labels: matching `20260909_123824_*` validation annotations. All 246 saved
  annotation images were compared against the correspondingly indexed decoded
  frame resized to 640x360. Mean absolute pixel differences were below 2.28/255
  (consistent with JPEG encoding), supporting the frame correspondence.
- The invalid label for frame 152 was excluded from **every** arm, without
  editing the source dataset. The paired set contains 245 frames: 47 positive
  frames/boxes and 198 labeled negatives.
- Three bundled checkpoints, each tested with resized color, native color and
  native grayscale repeated into three channels. No augmentation or retraining.
- CPU FP32, four PyTorch threads, Ultralytics 8.4.138, `imgsz=960`, NMS IoU 0.7.
  Predictions collected at confidence 0.15; primary results use the existing
  strong-detection threshold 0.45. Matching uses box IoU >= 0.5 and descending
  confidence, with each ground-truth box matched at most once.
- Latencies include prediction/preprocessing after a warmup call, excluding
  video decoding. They are local CPU measurements, not Jetson throughput.

## Results at confidence 0.45

| Checkpoint | Input | TP | FP | FN | Precision | Recall | F1 | Median prediction ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| boat_v4_s_best | resized_color | 22 | 28 | 25 | 0.440 | 0.468 | 0.454 | 81.4 |
| boat_v4_s_best | native_color | 23 | 28 | 24 | 0.451 | 0.489 | 0.469 | 84.5 |
| boat_v4_s_best | native_gray | 0 | 3 | 47 | 0.000 | 0.000 | 0.000 | 78.2 |
| boat_v4s_frozen_best | resized_color | 19 | 22 | 28 | 0.463 | 0.404 | 0.432 | 77.1 |
| boat_v4s_frozen_best | native_color | 22 | 18 | 25 | 0.550 | 0.468 | 0.506 | 77.9 |
| boat_v4s_frozen_best | native_gray | 4 | 14 | 43 | 0.222 | 0.085 | 0.123 | 77.7 |
| boat_v5_s_best | resized_color | 6 | 15 | 41 | 0.286 | 0.128 | 0.176 | 73.3 |
| boat_v5_s_best | native_color | 6 | 15 | 41 | 0.286 | 0.128 | 0.176 | 80.5 |
| boat_v5_s_best | native_gray | 0 | 3 | 47 | 0.000 | 0.000 | 0.000 | 75.8 |

For the default v4 checkpoint, native color improved recall from 46.8% to
48.9% with the same 28 false positives; F1 rose from 0.454 to 0.469. Two frames
gained a matched detection (132, 133) and one lost it (110). Visual inspection
of those paired overlays confirms a mixed, small change rather than a broad
quality breakthrough. At confidence 0.15, TP increased from 23 to 25 while FP
increased from 86 to 87.

The frozen v4 checkpoint improved from 19 to 22 TP and reduced FP from 22 to
18. V5 had no TP/FP improvement at 0.45 and performed worse than either v4
checkpoint on this sequence. These results do not justify automatically
promoting another checkpoint based on this one validation recording.

All three color-trained checkpoints degraded substantially when their native
input was converted to grayscale. Keep color for these checkpoints. The
resolution fix has measurable but modest benefits here; dataset quality and
coverage remain the next bottlenecks. No model weights or live settings were
changed.

Local evidence: `results/full_audit/paired_evaluation/summary.json`, per-frame
JSON files in the same directory, and `comparison.jpg` showing paired frames.

### Artifact identities

- Source video SHA-256: `fcc560fe8fb547d79fdf439de4821366456163ba3cf546c68c2a9f1de0095d44`
- `boat_v4_s_best.pt` SHA-256: `1d32ae3b115f333b78af39135dd53d41940d5e7285165b01060decf87225efaa`
- `boat_v4s_frozen_best.pt` SHA-256: `9abb87634b0f860e21113b82166b291df144e9347c71ea89a472cf1de65e09ac`
- `boat_v5_s_best.pt` SHA-256: `d4b18169e0c3c8a1b404961a5558c7e26b4b9aa97c8c9669f98082b2e879b25f`

## Reproduce

Keep downloaded media and labels outside Git. With the audit artifacts in the
local `results/full_audit` directory:

```sh
venv/bin/python evaluate_inputs.py \
  --video results/full_audit/validation_source.mp4 \
  --labels results/full_audit/validation/labels/val \
  --images results/full_audit/validation/images/val \
  --prefix 20260909_123824 \
  --weights weights/boat_v4_s_best.pt weights/boat_v4s_frozen_best.pt weights/boat_v5_s_best.pt \
  --out results/full_audit/paired_evaluation
```

The script writes a summary with video/checkpoint hashes and settings, plus
per-frame predictions and labels for each arm. It decodes frames incrementally,
checks alignment and rejects invalid annotation rows rather than treating them
as negatives. The JSON includes results at both 0.15 and 0.45 confidence.

## Interpretation limits

These are existing validation annotations from one short, correlated sequence.
Label completeness and historical training provenance are not fully verified.
An unmatched prediction counts as a false positive against the supplied labels,
which may include localization mismatches or unlabeled objects. The numerical
results must therefore not be generalized to independent field accuracy.

The grayscale arm tests inference-time conversion of color-trained checkpoints.
It does not establish the performance of a model trained with grayscale data
or grayscale augmentation. Model selection and retraining need a reviewed,
session-separated test set, plus repair/review of the 108 invalid dataset rows
identified in `AUDIT.md`. No training job or live deployment was started.

The incremental-decoding implementation was re-run for native-color v4: all
245 per-frame predictions exactly matched the initial comparison run. The
evaluation scorer has three regression tests; the complete Python suite passes
28 tests, and the viewer JavaScript regression checks pass.
