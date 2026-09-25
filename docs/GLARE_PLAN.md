# Color-camera glare mitigation

## What the existing evidence supports

Use color input throughout detection and depth inference. A polarizer and camera
exposure affect captured information; processing a decoded image cannot recover
clipped detail. Treat contrast enhancement as a detector experiment, not as glare
removal.

The last read camera settings allow exposure 500–65,487 microseconds, analog gain
1–16 and digital gain 1–8, with exposure compensation 0 EV, saturation 1 and AE
unlocked. These are limits, not measured per-frame exposure/gain values.

The 25 sampled frames of `20260910-140833/ShipCam0.mp4` have a maximum near-white
fraction of 1.20% over the full frame. In the lower 55% of the image, which includes
water and parts of the boat, the maximum is only 0.202% at the same threshold
(all BGR channels >= 250). This does not exclude glare below that threshold or
prove sensor clipping. It does show why darkening the entire image solely from
the full-frame white-pixel count is an unreliable automatic control rule.

## Camera comparison to run next

Use the rig's existing camera-settings controls, one camera at a time. Save the
current settings, preserve color (`saturation=1`), and record the actual exposure
and gain metadata when available. Keep the view, target path and lighting as
similar as possible. Record the baseline again at the end to detect drift in
conditions. Camera settings are not changed by this document or the evaluator.

| Trial | exposure_compensation | Other changes | Purpose |
| --- | ---: | --- | --- |
| A | 0.0 EV | None | Current baseline |
| B | -0.5 EV | None | Mild highlight protection |
| C | -1.0 EV | None | Stronger protection if B remains clipped |
| A repeat | 0.0 EV | None | Check whether scene/lighting changed |

Start with short 20–30 second recordings that include a visible moving target.
Choose by target recall/localization and blur, as well as clipped area within
the target/water region. A darker background and fewer white pixels alone do
not establish an improvement. Reject a setting if target recall drops.

Compare the trial recordings with `compare_trials.py`; it reads recordings only:

```sh
venv/bin/python compare_trials.py A=REC_A/ShipCam1.mp4 B=REC_B/ShipCam1.mp4 \
  C=REC_C/ShipCam1.mp4 A2=REC_A2/ShipCam1.mp4 \
  --weights weights/boat_v4_s_best.pt --out results/trials_YYYYMMDD
```

Per trial it reports detection coverage (unlabeled recordings, so a recall proxy
that a false positive also raises), the highlight share in the water band and in
tracked boxes, target sharpness (Laplacian variance, for blur) and the number of
glare-suspect tracks. Review the saved overlays before choosing. The rig's own
`ship_camera_processing/tools/exposure_sweep.py` scores shutter caps by AprilTag
decoding, not by boat detection, so it does not answer this comparison.

Baseline on three existing recordings (not trials: scenes differ, compare only
within one trial series), `results/trials_baseline_20260911/trials.json`:

| Recording | Coverage | Water highlight (median) | Target sharpness | Tracks | Glare-suspect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260911-135950 ShipCam0 | 0.256 | 0.047 | 183 | 2 | 0 |
| 20260911-140009 ShipCam1 | 0.228 | 0.029 | 91 | 3 | 0 |
| 20260911-140315 ShipCam1 | 0.883 | 0.073 | 90 | 4 | 2 |

Do not change exposure compensation, exposure cap, both gain caps and AE region
all at once. If target edges show motion blur, a separate well-lit trial can
compare `exposure_max_us=10000` (10 ms) with the current cap. AE may raise gain
when shutter time is limited, increasing noise. Lowering gain alone can make AE
use longer shutter times and increase blur. Very short 2–4 ms limits have already
been reported as harmful to tag decoding in the deployed source comments.

Keep `aelock=false` during initial comparisons. Do not blindly move metering to
dark water: excluding bright sky can make AE increase exposure and worsen the
reflection clipping. Only test a region after confirming the relevant sensor
coordinate system and the target trajectory.

## Optics

A rotatable polarizing filter in front of the lens is the first physical trial
for water reflections. Rotate it while observing both the reflection and target,
and check several camera headings. Its effect depends on viewing angle and
orientation; it also reduces transmitted light. Keep the shutter/blur tradeoff
in view. A lens hood helps with direct off-axis light entering the lens; it does
not remove reflections from the water. See the primary optics explanation in
[Edmund Optics: Introduction to Polarization](https://www.edmundoptics.com/knowledge-center/application-notes/optics/introduction-to-polarization).

Argus exposes exposure compensation and exposure/gain controls; support and
ranges still depend on the deployed sensor/driver. See
[NVIDIA Argus camera controls](https://docs.nvidia.com/jetson/archives/r36.2/ApiReference/group__V4L2Argus.html).
Do not assume an HDR mode exists for this capture setup without verifying it.

## Processing and evaluation

The optional `native_color_clahe` evaluator arm uses the same color-preserving
LAB luminance CLAHE as `boatdet.detection.apply_clahe` (clip limit 2.5, 8x8 tiles).
It does not convert model input to grayscale. Default evaluation modes remain
`resized_color` and `native_color`; production CLAHE remains opt-in.

```sh
venv/bin/python evaluate_inputs.py \
  --video results/full_audit/validation_source.mp4 \
  --labels results/full_audit/validation/labels/val \
  --images results/full_audit/validation/images/val \
  --prefix 20260909_123824 \
  --weights weights/boat_v4_s_best.pt weights/boat_v4s_frozen_best.pt \
  --modes native_color native_color_clahe \
  --out results/full_audit/clahe_evaluation
```

The detector comparison uses existing validation annotations from one recording,
not an independent test set. Full per-frame results and hashes are retained
locally in the output directory. Prediction timings exclude the separately
applied CLAHE operation and cannot be used to claim its runtime cost is zero.

Do not black out highlights, inpaint a boat behind a reflection or reject a box
merely because its pixels are bright. Those operations can erase real boats or
invent visual evidence. Record highlight overlap as a review signal. Retain
color glare examples in a reviewed, session-separated dataset, including partial
occlusion and true negative reflections, before retraining.

### Highlight review signal — implemented 2026-09-11

`boatdet/glare.py` measures, for every detection of `run.py --tracker bytetrack`,
the share of box pixels on the native frame with **any** BGR channel >= 250.
Tracks smooth it and set `glare_suspect` at 0.30 (`GLARE_SUSPECT_FRACTION`); both
appear as the last columns of `--track-log`, and in the overlay as a `glare
suspect` line on the track's label with a triangle at the box corner.
Nothing is filtered or down-weighted by it.

The any-channel test is deliberate. On the 2026-09-11 pool recordings a ceiling
light reflection was tracked as a boat for 234 frames of `20260911-140315`
ShipCam1 (confidence 0.27–0.50, 81 frames >= 0.45). Mean per-box values over
all tracks of three recordings, every third frame:

| Box | Any channel >= 250 | All channels >= 250 | Luma p95 |
| --- | ---: | ---: | ---: |
| Eight boat tracks | 0.000–0.033 | 0.000–0.013 | 67–180 |
| The reflection | 0.876 | 0.033 | 252 |

The reflection is yellowish, so its blue channel stays below the threshold and
the all-channels fraction used elsewhere in this document barely separates it.
This is one reflection in one hall; a white hull or a sunlit boat can be bright
too, which is why the signal flags for review instead of deciding.

### Glare candidates for the dataset

`hard_mining.py scan --min-highlight 0.3` keeps every box above that highlight
share rather than the top box per frame. On the three recordings it returned
58 candidates, all on the reflection, in 56 frames of `20260911-140315`
(`results/glare_candidates_20260911/`). Those frames also show the real boat, so
they are **not** empty negatives: importing them with `--confirmed-negatives`
would label a visible boat as background. They need an annotation with the boat
box and no reflection box, which `hard_mining.py import` cannot write (one box
per frame); it now refuses a frame whose boxes carry conflicting verdicts. Label
them with `label_video.py` after review, keeping the session in one split.

### Measured CLAHE comparison — 2026-09-11

245 matched validation frames (47 positive boxes and 198 labeled negatives),
confidence 0.45 and IoU 0.5. Frame 152 was excluded from both arms because its
annotation is invalid. The weights, frames and inference settings are identical.

| Checkpoint | Input | TP | FP | FN | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| boat_v4_s_best | native_color | 23 | 28 | 24 | 0.469 |
| boat_v4_s_best | native_color_clahe | 11 | 26 | 36 | 0.262 |
| boat_v4s_frozen_best | native_color | 22 | 18 | 25 | 0.506 |
| boat_v4s_frozen_best | native_color_clahe | 16 | 24 | 31 | 0.368 |

Keep CLAHE disabled by default: it reduced recall and F1 for both checkpoints. The
exposure trials above are prepared recommendations, not measured improvements;
they require fresh captures. No live camera setting was changed.
