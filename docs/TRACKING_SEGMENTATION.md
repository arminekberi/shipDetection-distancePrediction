# Multi-object tracking, segmentation and two-stage detection

`run.py --tracker bytetrack` adds a multi-object perception path next to the
existing single-target trackers (`csrt`, `kalman`, `none`), which are unchanged.

```
frame ─┬─ [proposer] ─ detector ─────┐
       │   band | horizon | seg      ├─ fusion ─ tracker ─ ego-motion ─ state filter ─ tracks
       └─ segmentation (every N) ────┘      │                                     │
                                  unknown obstacles                  distance, bearing,
                                  water-contact point                velocity, TTC
```

## Modules

| Module | Role |
| --- | --- |
| `boatdet/config.py` | every default (`TRACK_*`, `SEGMENTATION_*`, `FUSION_*`, `TTC_*`, `SHOW_*`, `ROI_*`) and the frozen config objects |
| `boatdet/mot.py` | `MultiObjectTracker`, `TrackState` |
| `boatdet/motion.py` | timestamp-driven constant-velocity Kalman filter, closing speed, TTC |
| `boatdet/egomotion.py` | `EgoMotionProvider` (null by default, recorded samples via `--imu-log`), rotation-induced pixel shift |
| `boatdet/geometry.py` | `CameraCalibration`, ray / water-plane intersection |
| `boatdet/segmentation.py` | `SegmentationModel.predict(frame) -> SegmentationResult`, every-N runner, obstacle extraction |
| `boatdet/fusion.py` | detector + segmentation fusion, water-contact point |
| `boatdet/proposal.py` | fixed band, horizon (Sobel + RANSAC) and segmentation proposers, `TwoStageDetector` |
| `boatdet/apriltag.py` | tag observation interface; inert unless `--apriltag-log` is given |
| `boatdet/pipeline.py` | wires the stages, times each one |
| `boatdet/overlay.py`, `boatdet/telemetry.py` | debug view; stage timings and per-track CSV |
| `evaluate_roi.py` | full frame vs. region proposals on labeled images |

## Tracking

ByteTrack-style association, chosen over BoT-SORT because no ReID weights exist
for these boats and the deployment target is a Jetson Orin Nano. Per frame:

1. every track is predicted to the frame timestamp (real capture stamps for raw
   frames; `dt` is never assumed), minus the pixel shift of camera rotation when
   an IMU sample and intrinsics are available;
2. confident detections (`--yolo-conf`) match by IoU plus an HSV-histogram cost;
3. weak detections (`--yolo-low-conf`) may only continue an existing track, at a
   stricter IoU — a boat briefly weakened by glare or spray keeps its id;
4. a motion gate matches what IoU cannot: a target that moved further than its
   own width or changed area up to 4x in one frame (fast relative motion, rapid
   closing);
5. unmatched tracks coast on the prediction for `--track-max-missed-frames`
   frames and at most `TRACK_MAX_COAST_S` seconds; the second bound matters at
   the 4–6 fps the rig records.

### Keeping a vessel through an occlusion

Use `--tracker bytetrack --track-occlusion-seconds 8` to keep a confirmed vessel's
ID for up to eight seconds since its last detection, independent of frame rate.
The automatic processing mode in `viewer/server.py` enables this combination;
manual box tracking still uses CSRT. Restart the viewer server after updating it
and process a recording again to generate new tracking results.

```sh
venv/bin/python run.py --camera path/to/video.mp4 \
  --yolo-weights weights/boat_v4_s_best.pt --tracker bytetrack \
  --track-occlusion-seconds 8 --no-display --sync \
  --output results/occlusion.mp4 --track-log results/occlusion_tracks.csv
```

During a missing detection, the image Kalman filter advances the last known
motion, retaining the appearance descriptor and ID. A dashed `PREDICTED` box
shows that this is an estimate; its label gives the time since the last detection.
Neither the hidden box nor its old water contact is sampled as a fresh range or
bearing measurement. The per-track CSV and JSON record `tracking_status` and
`seconds_since_detection`; metric fields on predicted rows remain filter estimates.
The legacy per-frame range log selects only currently detected targets.

Reappearance requires compatible class, bounded position/scale changes, and a
matching HSV descriptor when available. The center search can expand with filter
uncertainty, up to twice the normal motion gate. Without appearance evidence,
only a strong detection can recover the ID. An unknown segmented obstacle cannot
attach to a confirmed vessel in this mode. Expired identities are removed before
matching a new detection, so an arbitrarily late return cannot revive an old ID.
Unconfirmed tracks and unknown obstacles keep the normal frame/second limits.

This does not see through a pier: it predicts through missing detections, which
may also be caused by detector failure. No segmentation model is required.
Abrupt hidden maneuvers, camera motion without compensation, or similar-looking
boats can still prevent reliable recovery. Set the duration for the expected
occlusion; `0` disables this mode. The regression suite covers a four-second
absence at 5 and 30 fps, appearance mismatches, obstacle substitution, expiry,
and exclusion of the hidden background from range measurements. Validation on
a real occlusion clip is still required for scene-specific performance.

Tracks are reported after `--track-min-hits` matches. Each carries its own
Kalman filter in the image and, once a range exists, a second one in meters.
Velocity is read from the filter state, never differenced from two raw
measurements. TTC = D / closing speed only while the target closes faster than
`TTC_MIN_CLOSING_SPEED` **and** faster than `TTC_MIN_SIGMA` standard deviations
of the filter's closing-speed uncertainty; otherwise it is `None`, never
infinite or NaN.

Range source per track, first available wins: tag pose → water plane
(`--calibration` + water-contact pixel) → depth model sampled above the
water-contact row (so a reflection cannot pull the median). Each source has its
own measurement noise; `measurement_source` in the logs names it.

## Segmentation

No maritime segmentation weights ship with this repository, so the branch is
off by default and never produces a mask it did not infer. `--segmentation-weights`
accepts an Ultralytics segmentation checkpoint or its exported TensorRT
`.engine` (FP16 with `--fp16`). Model class names map to water / obstacle / sky
through `SEGMENTATION_CLASS_MAP`; shore, pier, boat and buoy are kept as extra
masks and folded into the obstacle mask. `--segmentation-every N` reuses the
last mask between inferences and marks it `reused`.

Obstacle masks are opened, split into connected components, filtered by
`--min-obstacle-area`, and turned into boxes, centroids and water-contact
points.

## Fusion

* detection + obstacle mask → confirmed vessel, water contact from the mask;
* obstacle region without a detection → `unknown_obstacle`, tracked, never
  relabelled as a boat;
* detection mostly in sky with no obstacle pixels → confidence scaled by
  `FUSION_SKY_CONF_SCALE`, kept;
* segmentation never removes a detection.

Water-contact source priority: keypoint model (none exists yet) → segmentation
→ bottom of the box. A tag is deliberately not a water-contact source: it is
mounted above the waterline.

## Two-stage detection (`--roi`)

`band` searches a fixed band of the frame height (`--roi-band`, default
0.35–0.78, from the label statistics: 99% of boat centers lie in 0.40–0.73).
`horizon` fits a line to the strongest vertical gradients and follows pitch and
roll; a weak or over-tilted fit falls back to the band. `segmentation` starts
the band just above the segmented water. The detector, trackers, CSV and viewer
see the unchanged `candidates()` interface; `--tile-grid` tiles inside the band.

Measured with `evaluate_roi.py` on the 831 validation images (existing labels,
640x360 annotation copies, `boat_v4s_frozen_v3_best.pt`, IoU >= 0.5, conf >= 0.45,
MPS timing), `results/roi_eval_val/summary.json`:

| Arm | Band ceiling | TP | FP | FN | Precision | Recall | F1 | ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full frame | 1.000 | 316 | 134 | 374 | 0.702 | 0.458 | 0.554 | 115 |
| band | 0.984 | 339 | 125 | 351 | 0.731 | 0.491 | 0.588 | 67 |
| horizon | 0.988 | 339 | 124 | 351 | 0.732 | 0.491 | 0.588 | 79 |
| band + 1x2 tiles | 0.984 | 353 | 218 | 337 | 0.618 | 0.512 | 0.560 | 147 |
| horizon + 1x2 tiles | 0.988 | 349 | 216 | 341 | 0.618 | 0.506 | 0.556 | 159 |
| full frame + 2x2 tiles | 1.000 | 365 | 515 | 325 | 0.415 | 0.529 | 0.465 | 357 |

The band gains 3 recall points, fewer false positives and ~40% of detector time.
Tiles add recall but multiply false positives; on 640x360 copies they only
upsample, so tiles cut from native 4608x2592 frames remain untested. In the
pool the horizon fit succeeds on 28% of frames (walls, lanes and roof trusses,
no true horizon) and otherwise falls back to the band; open water is untested.
A 20–40 frame sample had suggested a much larger gain; the full split does not
confirm its size.

## Configuration files

Calibration (`--calibration`), keys are case-insensitive:

```json
{"CAMERA_MATRIX": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], "DIST_COEFFS": [k1, k2, p1, p2, k3],
 "CAMERA_HEIGHT_M": 1.2, "IMAGE_SIZE": [4608, 2592], "MOUNT_PITCH_DEG": -3.0,
 "CAMERA_TO_IMU_TRANSFORM": [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}
```

Body frame is FRD (x forward, y starboard, z down); pitch nose-up positive. The
intrinsics are rescaled from `IMAGE_SIZE` to the working frame. Without a
calibration, water-plane range and pixel bearing are `None`.

Orientation samples (`--imu-log`), seconds on the `frame_timer` time base:

```json
[{"timestamp": 0.00, "roll_deg": 1.2, "pitch_deg": -0.4, "yaw_deg": 10.0}, ...]
```

## Running

```sh
# tracks, velocity, TTC; depth-model range
venv/bin/python run.py --camera REC/ShipCam0.mp4 --yolo-weights weights/boat_v4_s_best.pt \
  --tracker bytetrack --output out.mp4 --log out.csv --track-log tracks.csv

# plus horizon ROI with tiles, water-plane range, and segmentation every 2nd frame
  ... --roi horizon --tile-grid 1x2 --calibration cam0.json \
      --segmentation-weights seg.engine --segmentation-every 2 --fp16

# production: no window, no video, no overlays drawn
  ... --no-display --track-log tracks.csv
```

The per-track CSV ends with `highlight_fraction` and `glare_suspect`, a review
signal for specular reflections described in [GLARE_PLAN.md](GLARE_PLAN.md); it
never removes a detection.

The output video pairs the camera view with the depth map. Both panels carry the
same boxes, so a target the detector claims can be checked against the range the
depth map puts it at; the depth panel adds a colorbar labelled in calibrated
meters (`--depth-range NEAR,FAR` fixes that range across runs), and the footer
legend names every mark. Track color is the reserved status scale — tracked,
unknown obstacle, collision risk — and every state is also written as a word and
drawn in its own line style, because green and red collapse into each other
under the commonest color blindness.

`--debug` prints stage timings per frame; otherwise the console stays quiet and
`.meta.json` receives the mean timings (`detector_ms`, `segmentation_ms`,
`tracking_ms`, `total_ms`, `fps`). Overlays can be switched off one by one
(`--no-track-overlay`, `--no-segmentation-overlay`, `--no-water-contact-overlay`,
`--no-distance-overlay`, `--no-ttc-overlay`); a headless run without `--output`
draws none.

## Keeping the old behavior

Omit `--tracker bytetrack`, `--roi` and `--segmentation-weights`. The default
tracker is still `csrt`, the per-frame CSV header is unchanged, and the viewer
API still calls `run.py` exactly as before. Multi-object options are rejected
unless `--tracker bytetrack` is selected.
