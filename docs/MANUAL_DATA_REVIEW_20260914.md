# Existing manual annotations and training provenance — 2026-09-14

The existing manual boat boxes and reviewed background frames should be reused
for detector training. Separate shoreline labels or segmentation masks are not
required to train the existing boat detector. Before retraining, the concrete
problem is reconciling different dataset/annotation versions and identifying the
images each training run actually used.

## What was compared

Read `/root/shipcaps/_dataset` on rig 0 and compared its labels and image hashes
with both local datasets. All 5,016 remote images were decoded on the rig;
only metadata, hashes and annotation text were transferred. The comparison also
used decoded-pixel hashes, so differences in file headers do not explain the
unmatched images. No labels, image files or model weights were changed.

The source contains 4,524 training images (1,269 positive / 3,255 background)
and 492 validation images (66 positive / 426 background). Every nonempty label
uses class 0 and five-field YOLO box coordinates. There are no separate shore,
water or sky classes in these files. The original manual-labeling tool stores
a drawn boat box or an empty background label.

| Source image correspondence | Count |
| --- | ---: |
| Exact image in local color train | 512 |
| Exact image in local color test | 157 |
| Exact image in local mono train | 264 |
| Exact image in local mono validation | 81 |
| No identical image or decoded pixels in either local dataset | 4,002 |

At the existing color-fraction heuristic (at least 5% of pixels with channel
spread >12), 2,437 source images are color candidates and 2,579 are monochrome
or low-color candidates. Of the 2,437, **1,768 have no exact match in the local
color dataset**, 512 are in color train and 157 in color test. This does not
mean all 1,768 add new scenes: many are adjacent, previously subsampled frames.
Low color fraction is a review signal, not proof that a color camera was absent.

## Annotation versions differ on the same pixels

For the 512 source images matching local color training images:

| Label relationship | Count |
| --- | ---: |
| Both versions background | 277 |
| Both positive, valid, different boxes | 214 |
| One version boat, the other background | 11 |
| Invalid source box | 10 |

Among the 214 valid positive pairs, median IoU is 0.552 and 74 pairs have IoU
below 0.5. These are not just formatting differences. Across all source images,
108 label rows fail in-frame box validation. For the 157 matches in local color
test, 4 images have boat/background disagreement, 74 have different valid boxes,
19 have invalid source boxes and 60 are background in both versions.

[Visual comparison of six matched frames](../results/manual_training_review_20260914/annotation_versions.png):
red boxes are the rig's source labels; cyan boxes are the current local training
labels. The pixels displayed are the identical local copies, not new downloads.

On frames `20260909_123131_00068`, `_00070` and `_00072`, the local boxes cover
a visible yellow boat while the rig source labels are empty. Conversely, on
`20260909_123304_00112`, the source red box covers the yellow target while the
local cyan box is displaced to its left. This visual sample supports reconciling
individual conflicts rather than blindly treating either whole folder as the
correct version. It does not establish who created either annotation version.

## What the checkpoints establish

| Checkpoint | Embedded completion timestamp | Dataset path in checkpoint |
| --- | --- | --- |
| boat_v4_s_best (current default) | 2026-09-05 03:35 +03:00 | yolo_dataset_v4/dataset.yaml |
| boat_v4s_frozen_best | 2026-09-08 14:16 +03:00 | yolo_dataset_v4/dataset.yaml |
| boat_v5_s_best | 2026-09-08 13:07 UTC | /content/work/yolo_dataset_v4/dataset.yaml |
| boat_v4s_frozen_v3_best | 2026-09-10 10:16 UTC | /content/work/yolo_dataset_v4/dataset.yaml |
| boat_v4s_frozen_v3_unfrozen_best | 2026-09-10 11:17 UTC | /content/work/yolo_dataset_v4/dataset.yaml |
| boat_v4s_frozen_color_best | 2026-09-10 13:49 UTC | /content/work/yolo_dataset_v4_color/dataset.yaml |

The default checkpoint predates the September 9–10 recordings. It cannot be
presented as a model retrained on that later footage. The newer checkpoints do
reference these dataset families, but no per-file training manifest is embedded:
the saved path alone cannot prove which snapshot or annotation revision was used.
File modification times on the rig are not reliable creation/training provenance.

The 157 matched test images are also in the rig source's current training split.
Historical exports are not available to prove which checkpoint saw them. The
704-frame diagnostic subset must therefore retain its provenance caveat; current
local split separation alone is not evidence of independent historical testing.

## Prepared import plan

The copied frames can now be reviewed in the local
[annotation review interface](http://127.0.0.1:8001/dataset_review.html).
See the [review guide](DATASET_REVIEW_UI.md) for queue definitions, editing,
decision history and export. The UI includes all local color/mono matches,
so its 455 version-conflict frames cover more than the color/train comparison
above. The name-collision queue contains 492 frames (246 train/validation pairs).

`results/manual_training_review_20260914/training_import_plan.json` lists every
source frame, its pixel/label hashes, inferred session and reasons for holding it.
There are **821 preliminary training-import candidates** after separating low-color
inputs, ambiguous generic recording names, current held-out sessions, invalid
boxes and conflicting/already-present annotations. This is a plan, not a trained
dataset. Candidate label completeness and session grouping still need review.

The next training dataset should preserve the original manual labels as a
versioned source, record any reviewed conflict resolution, preserve held-out
sessions, and save an exact image/label manifest with the training run. Reuse the
already annotated frames first; collect new footage or masks only to address
measured coverage gaps or to run the optional segmentation experiment.

The user explicitly authorized copying the full source dataset into the project
on September 14. After the rig returned online, the transfer completed at
09:31 UTC. The separate snapshot is
`results/manual_training_review_20260914/source_dataset/`: **5,016 images and
5,016 label files**, with every file matching its saved source SHA-256 and no
missing or extra files. The extracted dataset contains 355,978,189 bytes.
The original archive is retained as `manual_source.tar`; its SHA-256 and the
verification totals are in `transfer_verification.json` and `transfer_status.json`.
Remote originals, local training datasets and annotation contents were preserved.
This is a faithful source snapshot, not a cleaned training dataset: the review
queues above still apply. No retraining has run.

Local verification decoded all 5,016 images successfully and found no missing
image/label pairs or identical image/pixel hashes across splits. The source audit
confirmed 108 invalid boxes and 246 shared frame stems across train/validation,
plus one shared recording/session name. Shared names do not establish identical
pixels; recording identity still needs reconciliation before training.

## Local evidence

- `remote_images.json`, `remote_labels.json`: source inventory and label contents.
- `remote_frame_metadata.json`: image shape, decoded hash and color fraction,
  computed on the rig.
- `frame_crosswalk.json`, `pixel_crosswalk.json`: matches against local datasets.
- `box_comparison.json`, `annotation_examples.json`, `annotation_versions.png`:
  annotation discrepancies on identical pixels.
- `invalid_manual_labels.json`, `training_import_plan.json`: explicit review queues.
- `source_dataset/`, `manual_source.tar`: verified source snapshot and archive.
- `transfer_verification.json`, `source_dataset_audit.json`: transfer integrity
  and local image/annotation checks.

All files above are under `results/manual_training_review_20260914/`, outside Git.
