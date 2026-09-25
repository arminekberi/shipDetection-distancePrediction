# Code, data and deployment audit — 2026-09-10

## Scope and evidence

Reviewed the local boat detection/depth repository: inference and tracking,
annotation/import tools, training scripts and four Colab notebooks, calibration
and comparison utilities, shell entry points, dependencies, and the recording
viewer/API. Separately read the deployed HydroRL capture/settings/detector code
on the server, its recording metadata, and the boat dataset. The deployed tree
has no `.git` directory and is not this repository's working tree.

Evidence is retained locally under `results/full_audit/`: read-only dataset
reports, a server source snapshot, and test logs. Media, deployed source dumps,
private machine state and model binaries from this audit are not added to Git.
`GLARE_REVIEW.md` describes the separate 25-frame color/glare experiment.

The available dataset has no independent test split. A subsequent paired
validation experiment is documented in `EVALUATION.md`; it measures the input
resolution change using existing weights. No checkpoint was retrained, and the
results do not establish independent field accuracy.

## Main findings

| Priority | Finding | Evidence and effect | Disposition |
| --- | --- | --- | --- |
| P1 | Different tasks were being conflated | Local `boat_v4/...` weights detect boats; deployed `yildiz` is an AprilTag pose proposer. The inspected recording selects `detector=apriltag`, bypassing the proposer entirely. | Explicit in documentation; no live detector switch. |
| P1 | Invalid training boxes | 108 annotation rows violate the single-class boat box constraints, including frame boundaries. | Auditor added; new writes validate/clip boxes. Existing labels retained for review. |
| P1 | Recording identifiers collide | 246 frame basenames occur in both train and val under the generic `ShipCam0` recording key. | New ShipCam prefixes include the recording directory. SHA-256 checking found no identical image bytes across splits; basename collisions alone do not prove leakage. Historical provenance remains unresolved. |
| P1 | Weak evaluation coverage | No test split; validation has only 66 positive images and 426 negatives. Every positive label has one box, so omitted secondary boats require human inspection. | Independent reviewed test data is required before selecting/retraining a model. |
| P1 | Wrong boxes could become false negatives | Hard-mining importer wrote empty labels for all indices marked as an incorrect detection, even if a boat might exist elsewhere. | Incorrect boxes now skip by default; negative labels require an explicit confirmed-negative index list. |
| P1 | Small-target detail was discarded | YOLO received a 640x360 resize before its requested 960-pixel inference preprocessing. Upscaling cannot restore discarded detail. | Native decoded frames now feed YOLO; boxes map back to working coordinates, including Kalman and the YOLO-only runner. |
| P1 | False detections could suppress correct candidates | Confidence was ranked before motion gating. CSRT could propagate indefinitely without a YOLO anchor; Kalman reacquisition retained old motion. | Candidate gating before ranking, bounded CSRT propagation, and Kalman reinitialization, with regression tests. |
| P1 | Calibration silently crossed domains | One pool/camera fit was applied by default, including when selecting another depth model. “Raw” readouts were already corrected. | Identity calibration by default; CSV includes the pre-calibration model value. Rig-specific corrections must be explicit. |
| P2 | Replay time and frame correspondence were unreliable | Manual seeding consumed frame zero; annotated output always used 15 FPS; offline async mode paired different frames. | Replay includes frame zero, uses source FPS, and is synchronous for file sources. CSV source times and result metadata are written. |
| P2 | Successful-looking empty output | Writers were unchecked; invalid depth could produce NaN; worker `_stop` shadowed a Thread method. | Checked H.264 writers, finite positive depth sampling, resource cleanup, and joinable worker. |
| P2 | Viewer overstated results | Stale training scores and a fixed depth-model identity were embedded in the UI. Tracking/depth availability was called detection rate. New manifest entries were hidden by an old allowlist. | Show supplied result metadata, label measurement coverage accurately, remove historical scores and folder filter. |
| P2 | Local API could report misleading work | Without a target it tracked the frame center; source paths excluded server recording directories; repeated jobs overwrote files. | Default boat detector, validated relative source paths/targets, unique job directories and manifest publication after completion. |
| P2 | Unnecessary HTTP exposure and fragile input handling | Wildcard bind served project metadata; malformed JSON/length/target input was not handled consistently. | Loopback bind, restricted asset roots, bounded JSON input, same-origin checks and synchronized job state. This remains a local development server. |
| P2 | Annotation could silently lose progress | Failed image writes were ignored; back navigation immediately skipped the previous label; negative mode unnecessarily opened a GUI. | Validate writes, commit image before label, allow review of earlier session frames, headless negative mode, and explicit failure exit. |
| P2 | Test-only recordings entered mining defaults | Two recordings marked test-only were in the mining list, including their unlabeled tails. | Removed and guarded in scanner/importer. Past contamination is not automatically repaired. |
| P3 | Tooling was difficult to reproduce | Absolute paths referred to another Mac, retry loops never stopped, notebook YAML rewrites depended on one old path, and prose was mixed-language. | Checkout-relative shell paths, bounded retries, YAML-based path updates, pinned notebook Ultralytics, English UI/messages/notebooks/docs, import-safe training scripts. |

## Dataset snapshot

The read-only audit of `/root/shipcaps/_dataset` produced:

| Split | Images | Labels | Positive images | Negative images |
| --- | ---: | ---: | ---: | ---: |
| train | 4,524 | 4,524 | 1,269 | 3,255 |
| val | 492 | 492 | 66 | 426 |
| test | 0 | 0 | 0 | 0 |

No missing image/label pairs or zero-byte images were reported. The hash pass
found no byte-identical images shared across splits. Hash equality cannot detect
near-duplicates, re-encoding, adjacent frames, renamed sessions or simultaneous
camera views. Empty labels are only valid negatives after reviewing the scene.
The auditor reads detection labels with class 0; it is not an AprilTag pose-label
validator.

## Deployed HydroRL findings

The GPU detector declares `wants_colour=False`. The worker therefore converts BGR
to grayscale before calling it, despite its proposer being trained on color.
The detector already constructs a separate grayscale image for refinement/readout,
so retaining BGR for proposal generation is a concrete correction. Also,
`refine_iters=0` is replaced by the default because the worker uses `value or default`.

A small, reviewable patch is provided at `docs/hydrorl-proposer-input.patch`.
It was checked against the downloaded deployed source, but is **not applied to
the running server** and has not been validated on Jetson TensorRT or an annotated
marker test set. Publish it through the HydroRL source repository, then compare
identical full-resolution frames before deploying. The user has not supplied
that repository location; there is no Git checkout in `/root/HydroRL`. From that
repository, check applicability with `git apply --check docs/hydrorl-proposer-input.patch`.

The deployed `compare_detectors.py` treats full-frame AprilTag output as ground
truth. That measures agreement with another imperfect detector, not true recall:
it excludes tags AprilTag misses and can conceal false IDs. Prefer manual tag
corners/IDs with an explicit “unreadable/occluded” category. The 1280-wide archived
MP4 is suitable for preview, not a substitute for native capture when assessing
small markers. The server also reports only two decoded-tag frames in 666
processed frames for the selected archive; these counts do not establish recall
without knowing how many visible tags were present.

## Remaining limitations and next training run

1. Recover recording provenance for generic `ShipCam0_*` labels. Review the 108
   flagged rows and frames with multiple boats. Correct or remove samples only
   after inspection; do not turn uncertain positives into negatives.
2. Split by complete capture sessions, including related views from multiple
   cameras. Create an independent test set spanning reflections, motion blur,
   target sizes, lighting, occlusions and true empty scenes.
3. Establish a fixed baseline for the exact checkpoint. Report per-condition
   precision/recall and localization quality, plus latency and measurement
   availability separately. Include all visible boats in detection ground truth.
4. Train color-input models with fixed split/seed/settings. Compare on the same reviewed test frames. Select by
   held-out task accuracy; confidence or tracking continuity alone is insufficient.
5. Validate distance on independently measured ranges for the actual camera and
   depth model. Existing calibration-fit residuals from four points are training
   residuals, not an independent estimate of distance accuracy.

Live asynchronous preview still uses the latest available depth and is not
frame-exact. Long annotation sessions still decode sampled frames into RAM; use
sampling and bounded clips until a disk-backed review index is implemented.
Coasted/tracked boxes do not prove a current YOLO detection; `--tracker none`
reports detections alone. The depth-only and CSRT-only baseline scripts were
removed after this audit (Git history retains them); they were experiments, not
validated production detectors. Legacy MPEG-4 Part 2 recordings may not play in
browsers; new outputs require H.264 support. Existing recorded metadata is
not retrospectively corrected, so old calibration declarations may be stale.

## Verification

- 25 local Python regression tests passed, including first-frame/FPS replay,
  calibration identity, native-coordinate mapping, tracking gates, dataset
  writes, source validation, clipping diagnostics and thread shutdown.
- Node checks cover named CSV columns, invalid numeric values, measurement
  coverage, timestamps and HTML escaping. Inline JavaScript syntax was checked.
- Python source, all notebook JSON/code cells and shell syntax were checked.
- `pip check` reports no broken requirements in the existing virtual environment.
- 19 deployed pipeline-settings tests passed against the local source snapshot.
- HTTP smoke checks passed for viewer assets, status, denied `.git` access and
  malformed/cross-origin requests. No processing job was launched by these checks.
- English UI and the annotated/source toggle were inspected in the browser with
  real local manifest data. Legacy source paths/media may still be unavailable.

No model training, live-service restart, dataset relabeling or camera-setting
change was performed. These require the reviewed data and deployment validation
above; the audit does not claim that software changes recover clipped pixels.
