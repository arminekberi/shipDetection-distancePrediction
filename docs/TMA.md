# Bearings-only target motion analysis

For calibration and supervised training against both rigs' recorded states,
see [the two-rig data pipeline](RIG_CALIBRATION.md). It keeps target telemetry
out of inference inputs and evaluates held-out recordings separately.

Estimating the other vessel's state `[x, y, speed, course]` from our own camera,
given our own navigation. The camera measures direction, never range, so range is
inferred from how the bearing changes while we move — and where our motion carries
no such information, the estimator says so instead of inventing a number.

```
track bearing ─ own-ship nav ─┬─ BearingObservation ─┬─ EKF r1 ─┐
                              │  (absolute bearing,  ├─ EKF r2  ├─ weighted mixture ─ TargetEstimate
   camera calibration ────────┘   sensor position)   ├─  ...    │   + parallax test    [x, y, v, course]
                                                     └─ EKF rN ─┘                      + status
```

## Modules

| Module | Role |
| --- | --- |
| `boatdet/tma.py` | `OwnShipState`, `OwnShipTrack`, `BearingObservation`, `BearingOnlyEKF`, `RangeHypothesisTma`, `MultiTargetTma`, `TargetEstimate` |
| `boatdet/config.py` | `TMA_*` defaults and `TmaConfig` |
| `tma_sim.py` | scenario simulator: known truth, bearing noise only |
| `bearing_noise.py` | measures the bearing jitter that sets `--tma-pixel-sigma` |
| `boatdet/pipeline.py` | the `tma` stage: bearing per track, own-ship state per frame |
| `boatdet/geometry.py` | `bearing_from_ray`: bow-relative bearing of a pixel, roll and pitch included |
| `tests/test_tma.py` | filter, bank, observability, own-ship track, pipeline and overlay tests |

Frame: local tangent plane, `x` east, `y` north, meters; angles radians
counter-clockwise from `+x`. `compass_from_course` / `course_from_compass`
convert to and from compass degrees, which is what `TargetEstimate.course_deg`
and `.bearing_deg` report.

## Scale: this rig records one scene

The camera records a covered test basin and nothing else — tens of meters of range,
model speeds, roughly 4 fps. Every default in `boatdet/config.py` is scaled for it:
a 1–120 m range window, a 3 m/s speed bound, a 40 s parallax window. At sea each of
those is an order of magnitude out, so `tma_sim.py` keeps open-water scenarios that
carry their own overrides (`OPEN_WATER`) and the method stays exercised at both
scales. The basin extent below is an assumption; replace it with the tank's
measured dimensions.

The scale change is not cosmetic. Range observability depends on how far the
observer can move **across** the line of sight, and a tank is narrow. The section
on what the basin allows works that number out.

## The state vector, and which angle is which

`TargetEstimate.state_vector` is `[x_m, y_m, speed_mps, course_rad]`. Three
angles are easy to confuse and are kept apart:

| Quantity | Meaning | Where |
| --- | --- | --- |
| bearing | direction from us to the target | `TargetEstimate.bearing_rad`, measured |
| course | direction the target travels | `TargetEstimate.course_rad`, estimated |
| heading | direction the target's bow points | not observable from a bearing; set and drift differ from course |

Inside the filter the state is Cartesian, `[x, y, vx, vy]`: a constant-velocity
model is linear there, and speed and course are derived on output. At a standstill
the course is undefined, and the estimate reports `course_sigma_rad = pi` rather
than a confident zero.

## What the method can and cannot recover

One bearing is a ray. Two bearings from two positions intersect — but only if the
target held still between them. A moving target needs the own ship to supply
geometry the target's own motion cannot imitate.

If the own ship holds a straight course at constant speed and the target does the
same, the range is **not observable**: a whole family of target trajectories, at
different ranges and speeds, produces exactly the same sequence of bearings. No
filter removes that ambiguity, and one that reports a tight range in that geometry
is reporting its initialization, not a measurement. The fix is geometric, not
algorithmic: a course or speed change by the own ship, with a component across the
line of sight.

This is why the estimator carries a **bank of range hypotheses** rather than one
guess. `TMA_HYPOTHESES` filters are seeded on the first bearing, tiling
`[TMA_MIN_RANGE_M, TMA_MAX_RANGE_M]` in log-spaced slices — the prior is the whole
range window, not a point inside it. Each is a bearings-only EKF; the bank weights
them by measurement likelihood, and the published estimate is the weighted mixture.
Several hypotheses surviving with comparable weight is the correct answer to an
ambiguous geometry.

Two details keep that honest:

- **Weight flattening** (`TMA_WEIGHT_FORGET_S`). Accumulating hundreds of
  negligible likelihood differences collapses any bank onto a single range, even
  where the geometry determines none. The weights decay toward uniform with this
  time constant, so collapse requires sustained evidence.
- **The parallax test** (`TMA_PARALLAX_SIGMAS`). The own-ship track over the recent
  window is fitted with a straight line; the residuals, projected across the line
  of sight, are the part of our motion a constant-velocity target cannot mimic.
  Divided by the range they are an angle, and that angle has to beat the bearing
  noise by `TMA_PARALLAX_SIGMAS` before the estimate may be called `converged`.
  The peak latches: a manoeuvre flown earlier still determined the range, and
  information that has since gone stale reappears as a growing range sigma.

`status` is part of the answer:

| status | meaning |
| --- | --- |
| `initializing` | fewer than `TMA_MIN_UPDATES` bearings |
| `ambiguous` | the geometry has not determined the range; use `range_low_m..range_high_m`, not `range_m` |
| `converged` | parallax beat the bearing noise and the range sigma is under `TMA_CONVERGED_RANGE_FRAC` of the range |

`range_low_m` / `range_high_m` are a central 90 % (`TMA_CREDIBLE_LEVEL`) interval
read off the mixture along the line of sight, not mean ± sigma: while the range is
undetermined the posterior is multimodal, and a single sigma would describe a
distribution that is not there.

## Measured behaviour

`venv/bin/python tma_sim.py --scenario all` — bearings at 1 Hz with 0.5° noise,
240 s, seed 0, defaults otherwise. The simulator injects bearing noise only: no
detector or calibration error, so these numbers are the estimator's own limit and
an upper bound on rig performance, never a prediction of it.

| scenario | own ship | final status | converged at | final range error | speed error | course error | truth inside interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `straight` | one course | ambiguous | never | 43 % | 3.3 m/s | -14.5° | 100 % of updates |
| `maneuver` | one turn at 90 s | converged | 118 s | 5.7 % | 0.51 m/s | -2.7° | 100 % |
| `zigzag` | legs every 60 s | converged | 123 s | 7.9 % | 0.31 m/s | -8.4° | 100 % |
| `stationary-target` | one course, target stopped | ambiguous | never | 88 % | 4.4 m/s | 84.4° | 100 % |
| `closing` | dogleg at 80 s | converged | 85 s | 2.5 % | 0.42 m/s | -1.5° | 100 % |

The two ambiguous rows are the result, not a failure: the geometry carries no
range information, the point estimate is duly wrong, and the reported interval
still covers the truth at every update. `stationary-target` stays ambiguous even
though a stopped target would be solvable by triangulation — under a
constant-velocity model the estimator is not told the target is stopped, and it
does not assume it.

A coverage sweep over 8 seeds (scratch, not committed) is what set
`TMA_WEIGHT_FORGET_S = 60`: at 120 s and above, coverage in the ambiguous
scenarios fell to 0.3–0.8 while the observable scenarios gained nothing.

Cost: ~0.5 ms per bearing per track with 12 hypotheses, independent of frame rate
(the parallax history is subsampled to `PARALLAX_SAMPLES` per window).

## Running it on a recording

```sh
venv/bin/python run.py \
  --camera capture.mp4 --yolo-weights weights/boat_v4_s_best.pt --tracker bytetrack \
  --calibration rig.json --own-ship-log nav.json \
  --tma-min-range 50 --tma-max-range 2000 --tma-pixel-sigma 2.5 --tma-heading-sigma-deg 1.0 \
  --track-log results/tracks.csv --no-display
```

`--own-ship-log` is what turns the stage on, and it requires `--calibration`:
without intrinsics a pixel is not a bearing, and the run is refused rather than
filled with a default lens. The remaining `--tma-*` flags are the range window and
the bearing error budget; `--tma-lever FORWARD,STARBOARD` places the camera
relative to the navigation reference point.

### The own-ship log

JSON (a list, or `{"samples": [...]}`) or CSV, one record per navigation fix:

```json
[{"timestamp": 0.00, "latitude": 40.9712, "longitude": 29.0361, "heading_deg": 87.4, "speed_mps": 5.1},
 {"timestamp": 0.50, "latitude": 40.9712, "longitude": 29.0362, "heading_deg": 87.9, "speed_mps": 5.1}]
```

`timestamp` is on the video clock. Position is `latitude`/`longitude` (projected
onto the tangent plane through the first sample) or `x_m`/`y_m` directly. Heading
is `heading_deg` (compass) or `heading_rad` (math convention) and is **required**:
it is what places a bow-relative bearing in the world. Speed is optional and is
carried for reporting only. Frames further than `--own-ship-max-age-s` from a
sample produce no bearing at all — an interpolated own position would be read
downstream as target motion.

### What comes out

`--track-log` gains nine columns: `tma_x_m`, `tma_y_m`, `tma_speed_mps`,
`tma_course_deg`, `tma_range_m`, `tma_range_low_m`, `tma_range_high_m`,
`tma_parallax`, `tma_status`. They are empty for any track no navigation could
place. `tma_range_m` is only meaningful where `tma_status` is `converged`;
elsewhere the interval is the answer.

The overlay draws one line per track, and it does not draw a range the geometry
has not earned: `TMA 620 m 4.2 m/s 225deg` when converged,
`TMA 410-1280 m ambiguous` when not. `--no-motion-overlay` hides it.

The bearing itself comes from `bearing_from_ray` on the box center, which rotates
the full pixel ray through the recorded attitude. The older `bearing_from_pixel`
reads a column against the principal point and is exact only for a level camera:
at 20° of roll a target 120 px above center is 3.2° off, which is six times the
detector's own contribution.

## Measuring the bearing error budget

`bearing_noise.py` measures how much a tracked reference point wobbles around its
own smooth motion: a short window is fitted with a quadratic and the residual of
its center sample, scaled by its leverage, is the jitter. The fit removes genuine
target motion; what is left is noise.

```sh
venv/bin/python bearing_noise.py --video capture.mp4 --weights weights/boat_v4_s_best.pt \
  --conf 0.15 --min-samples 20 --focal-px 600
venv/bin/python bearing_noise.py results/multitrack        # from existing track logs
```

### What has been measured

Raw detector jitter on eight rig source recordings, copied locally to
`results/rig_copies/` (see the README there), measured in the 640x360 working
frame, 3534 residuals from 41 chains:

| reference point | RMS | robust (MAD) | p90 | at f = 600 px |
| --- | --- | --- | --- | --- |
| raw box center | 0.69 px | 0.42 px | 0.98 px | 0.066° |
| raw box bottom (v) | 0.67 px | 0.35 px | 0.96 px | 0.064° |

These are **source** files: no overlay burnt into the frame. That distinction cost
a measurement — an earlier run of this tool used files under `results/` that turned
out to be rendered outputs of previous runs, complete with drawn boxes and a depth
panel, and produced a number (1.8 px) that described the overlay. Anything fed to
`--video` has to be checked for that first.

The same tool over the existing track CSVs reads 0.46 px, but `run.py` publishes a
Kalman-filtered center, so that figure is a floor with the filter's smoothing
already applied, not the detector's own jitter.

The method itself is checked against known noise in `tests/test_bearing_noise.py`:
fed a series with a 3.0 px sigma it recovers 3.0 px, and without the leverage
correction it would report 2.4 px.

Two limits apply to any figure this tool produces:

* **It cannot see slow bias.** A quadratic fit over a 9-sample window removes
  anything slower than the window, and the box center drifting as a vessel changes
  aspect is exactly that.
* **It is the detector term only.** Heading and mounting uncertainty are not in a
  recording at all and add in quadrature.

`TMA_PIXEL_SIGMA_PX` is therefore 1.5 px — above the 0.69 px measured, leaving room
for the drift the method cannot see. It is not the leading term regardless: at the
default 1° of heading uncertainty the error budget is

    bearing sigma = sqrt(0.14 deg^2 + 1.0 deg^2 + 0.3 deg^2) = 1.05 deg

of which the detector contributes 0.14°. **Heading is the budget.** Lowering the
pixel term to zero would change the total by under 1 %.

## Using the module directly

```python
from boatdet.config import TmaConfig
from boatdet.tma import BearingObservation, MultiTargetTma, OwnShipState, bearing_sigma

bank = MultiTargetTma(TmaConfig(min_range_m=50.0, max_range_m=2000.0))

own = OwnShipState(timestamp=t, x_m=x, y_m=y, heading_rad=heading, speed_mps=speed)
sigma = bearing_sigma(pixel_sigma_px=4.0, focal_px=calibration.camera_matrix[0, 0],
                      heading_sigma_rad=math.radians(1.0), mount_sigma_rad=math.radians(0.3))
observation = BearingObservation.from_relative(own, track.bearing_deg, sigma,
                                               lever_forward_m=2.0, lever_starboard_m=0.0)
estimate = bank.observe(track.track_id, observation)
if estimate.status == 'converged':
    x2, y2, speed2, course2 = estimate.state_vector
```

`track.bearing_deg` is the bow-relative bearing `boatdet/geometry.py` already
produces from a pixel column and the intrinsics; `from_relative` turns it into an
absolute bearing and attaches the camera position, lever arm included.
`OwnShipTrack(samples).at(timestamp)` interpolates recorded navigation to a frame
timestamp and returns `None` rather than extrapolating past `max_age_s`.

### What you have to supply

- **Bearing sigma.** Compose it with `bearing_sigma` from detector jitter, focal
  length and attitude uncertainty, and measure the first term with
  `bearing_noise.py` (below). Detector confidence is not an angular variance.
- **Own-ship position and heading per frame**, on the same clock as the video.
  A heading error enters every bearing directly: 1° of heading error is 1° of
  bearing error, which at 1 km is 17 m across the line of sight.
- **A range window** that covers the scenario. Nothing outside
  `[min_range_m, max_range_m]` can ever be reported.
- **Roll and pitch**, if the camera is not gyro-stabilized — they belong in the
  ray that produces the relative bearing (`water_intersection` already takes an
  `Orientation`), not in this module.

## What the wiring has been run against

The pipeline path was exercised end to end on a local recording
(`results/by_recording/shipcam0_20260904_113444`, 400 frames, boat_v4_s,
bytetrack) with a nominal calibration and a **synthetic** own-ship log, since no
navigation was recorded with that footage — and, as the section above explains,
that file is a rendered output of an earlier run in an indoor tank, so the
detections in it are not vessels either. 2 tracks, 106 logged rows, every one
`initializing` or `ambiguous`. Nothing in that run is a measurement; it shows the
stage runs, fills its columns, draws its label and refuses to converge without
geometry.

Cost on that run: 0.09 ms per frame for the `tma` stage against 46.05 ms for the
detector.

The first version of that run used a synthetic log that changed course by 55° in
one sample. The bank collapsed onto a confident 10 m — the only range at which an
instantaneous bearing jump is arithmetically possible. The input was impossible,
but the failure mode is not: a navigation glitch or a track id swap produces the
same thing, which is what the gate below was added for.

## What the basin allows, and what it does not

Range comes from cross-LOS baseline, and in a tank the baseline is the tank's width.
`tma_sim.py --scenario basin-weave` flies the largest weave the width allows — 8°
alternations every 15 s at 0.8 m/s, about 1.7 m of lateral excursion — against a
target 30–55 m away. Measured over 8 seeds:

| own-ship heading sigma | bearing sigma | parallax needed | parallax achieved | converged | range error |
| --- | --- | --- | --- | --- | --- |
| 1.0° (default) | 1.05° | 0.055 | 0.027 | 0/8 | 170 % |
| 0.5° | 0.60° | 0.031 | 0.026 | 0/8 | 195 % |
| 0.3° | 0.45° | 0.023 | 0.025 | 0/8 | 203 % |
| 0.1° | 0.35° | 0.018 | 0.024 | 0/8 | 209 % |
| 0.03° | 0.33° | 0.018 | 0.024 | 0/8 | 210 % |

**Better heading does not rescue it.** Past about 0.3° the bearing noise is already
dominated by the pixel and mounting terms, so the required parallax stops falling
while the achievable parallax is fixed by the tank. Even where the parallax test is
passed, `geometric_range_sigma` bounds the range to roughly ±25 % of itself, which
is a bracket rather than a fix — and the point estimate is measurably worse than
that bound, because a Cartesian EKF biases in near-degenerate geometry. Every basin
scenario is therefore reported `ambiguous`, which is the correct answer.

The conclusion for the basin is not about tuning: **range should come from the
waterline, not from bearings.** `boatdet/geometry.py:water_intersection` intersects
the ray with the water plane, and a basin is where that works best — the camera
height is fixed and knowable, the water is flat, the waterline is visible. What
bearings-only adds here is the course and speed of the other model, a cross-check
on the waterline range, and an honest flag when the geometry supports neither.

Open water is the opposite case: there the own ship can manoeuvre kilometres across
the line of sight, and the same code converges to 2.5–8 % (see the table above).

## The bound the geometry sets

Triangulation over a cross-LOS baseline `b` resolves range to about
`r · sigma_beta / (b/r)`, and `b/r` is exactly the parallax ratio this module
measures. `geometric_range_sigma` computes it, and the published `range_sigma_m` is
floored by it: where the filters claim better, the interval is widened to what the
parallax supports, and with no parallax at all to the declared range window. A
Cartesian EKF on bearings is known to linearize its way into confidence the
geometry never provided, and this is the check that catches it.

The effect is measurable. Before the floor, the interval covered the truth in 49 %
of updates on `basin-straight` and 27 % on `basin-crossing`; after it, 100 % in
both, while the open-water scenarios are unchanged — after a real manoeuvre the
bound is far looser than what the filters already report.

## Rejecting bearings nothing can explain

A bearing whose normalized innovation exceeds `TMA_GATE_SIGMAS` under **every**
hypothesis is more likely a bad detection or a navigation glitch than evidence, and
is dropped: the filters still advance in time, but none absorbs it. A gate alone
would be a lockout, though — after a real discontinuity every subsequent bearing
disagrees forever — so `TMA_GATE_MAX_CONSECUTIVE` rejections in a row reseed the
bank on the current bearing, discarding the accumulated evidence, the parallax
history and the observability earned with them. `TargetEstimate` reports both
counts as `rejected` and `reseeds`.

A `range_log_prior` mirrors the speed prior: the range window is a declared prior,
so a hypothesis that drifts more than `TMA_RANGE_PRIOR_FACTOR` outside it is
penalised. The reported interval may still extend past the window — it is a tail
of the mixture, not a hypothesis mean.

Adding the gate left every simulator scenario unchanged.

## Not done yet

- **No own-ship navigation source exists in this repository**, the same gap as
  `boatdet/egomotion.py`. Every number above the simulator's is therefore untested
  against real navigation: recorded position and heading on the video clock are
  the one input that cannot be substituted.
- Not validated against an independent truth (AIS or GNSS of the second vessel).
  Simulation bounds the estimator; only that comparison bounds the system.
- Bearing from the box center inherits the aspect-change bias of the box: the
  center moves as the vessel turns. A waterline or keypoint reference would be
  steadier, and the bias is currently carried in the sigma rather than measured.
- Roll and pitch reach the bearing only through `--imu-log`. Without it
  `bearing_from_ray` uses the mounting angles alone, and a rolling platform adds
  bearing error the estimator is not told about.
- The viewer does not show the motion estimate; only the CSV and the rendered
  overlay do.
