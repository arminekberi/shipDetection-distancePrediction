# Dataset audit and repair — 2026-09-11

Covers `yolo_dataset_v4_color` (3,575 images) and `yolo_dataset_v4_mono` (7,125),
the color/mono split datasets. `repair_dataset.py` applied the fixes; it moves
files and never deletes, so every change is reversible. Both datasets now pass
`audit_dataset.py --hash-images` with no issues.

## Findings

| Dataset | Finding | Evidence | Repair |
| --- | --- | --- | --- |
| color | One recording present in both splits | `test1` (train) and `20260902-151121_ShipCam0` (val) are the same footage: 159 frames byte-identical at the same frame indices, pixel MAE 0.000. 15.6% of val, 18.6% of its positives. | All 167 val frames of that recording quarantined; the neighbouring 8 frames are the same footage. |
| color | Reserved test recording in training | `renkliTekneTekne` (111 frames), reserved in `TEST_VIDEOS`, was labeled into train. | Moved to the test split. |
| color | One capture session across splits | `20260902-150054_ShipCam0` (train, 34) and `_ShipCam1` (val, 19) are simultaneous views of one session. | The 19 val frames moved to train. |
| mono | Reserved test recording in both splits | `renksizTekneTekne`: 33 frames train, 224 val — reserved, and one recording spanning splits. | All 257 frames moved to the test split. |
| both | Boxes reaching outside the frame | 43 color, 144 mono. | Clipped to the image; a box with nothing left inside quarantines the frame instead of becoming a background label. |

## Counts after repair

| Dataset | train | val | test |
| --- | ---: | ---: | ---: |
| color | 2,466 (1,686 positive) | 831 (690) | 111 (90) |
| mono | 6,012 (2,910) | 856 (452) | 257 (229) |

Quarantined frames are under `<dataset>/_quarantine/<split>/`, outside the
directories Ultralytics reads. `dataset.yaml` in both datasets was rewritten for
this checkout; it previously pointed at another machine.

## Annotation convention is not consistent

The duplicated color recording was annotated twice, independently, on identical
pixels — a direct measurement of labeling noise:

- median IoU between the two passes **0.481**; 97 of 159 frames below 0.5,
  the threshold used to call a detection correct;
- the difference is systematic, not random: one pass drew boxes 1.93x larger in
  area (84x44 px median against 63x31 px) with centers within 2 px.

So two box conventions coexist in this data, and localization differences of
this size cannot be resolved by any IoU 0.5 metric computed on it. The paired
evaluation in [EVALUATION.md](EVALUATION.md) reports F1 changes (0.454 -> 0.469)
well inside that noise. Repair cannot fix this; the convention has to be defined
and the affected recordings reviewed.

## Consequences for existing checkpoints

`weights/boat_v4s_frozen_color_best.pt` was trained on this color dataset before
repair (`train_args.data=/content/work/yolo_dataset_v4_color/dataset.yaml`), so
its validation numbers include the duplicated recording and the split capture
session. It also saw `renkliTekneTekne`, which is now the test split, so that
split is only an honest test for models retrained on the repaired data.

`docs/EVALUATION.md` evaluates on recording `20260909-123824`, which is in the
color training split. Those numbers do not transfer to color-trained checkpoints.

## Training input decision

Training uses color recordings only; `train.py` defaults to
`yolo_dataset_v4_color/dataset.yaml`. Boats are not reliably distinguishable in
the monochrome recordings, and the paired evaluation already showed that every
color-trained checkpoint collapses on grayscale input. `yolo_dataset_v4_mono` is
retained, repaired, as evidence; it is not a training target.

## Remaining work

1. Define one box convention and re-review the recordings labeled with the other.
2. Build an independent test set from reviewed, session-separated recordings;
   the current test splits are recordings pulled out of training data.
3. Retrain on the repaired color dataset before comparing checkpoints, and
   report per-condition precision/recall rather than validation fitness.
