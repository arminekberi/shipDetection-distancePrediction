# Basin digital twin

`boatdet/twin.py` simulates the YTU basin: both hulls, their UWB tags, IMUs,
motor readback and ship 1's forward camera. `pool_sim.py` runs it and scores
the real pipeline against its truth.

The twin exists because the basin cannot score the pipeline. Ship 2's cameras
are dead, nothing positions a boat, and no reference measures heading. So the
twin writes output in the formats the real tools already read, and those tools
run on it unchanged:

| Twin output | Format of | Read by |
| --- | --- | --- |
| `telemetry.jsonl` | `rig_state_logger.py record` | `rig_state_producer.py replay` / `calibrate-heading`, `rig_state_logger.py export` |
| `bearings.csv` | `track_target.py` | `bearings_to_observations.py` |
| `observations.csv`, `state_ship{1,2}.csv`, `manifest.json` (from `dataset`) | two-rig supervision contract | `rig_calibration.py prepare / fit / evaluate`, `train_rig_state.py` |
| `truth_ship{1,2}.csv`, `camera_truth.csv`, `session.json` | twin only | `pool_sim.py check` |

Everything is marked `synthetic: true`. A number from the twin tells you what the
code does with data of this shape. It does not tell you the rigs produce data of
this shape.

## Commands

```sh
venv/bin/python pool_sim.py list
venv/bin/python pool_sim.py run --scenario weave --out results/sim/weave --check
venv/bin/python pool_sim.py check results/sim/weave          # check.json, map.png, errors.png
venv/bin/python pool_sim.py dataset --out results/sim/dataset --runs 6
venv/bin/python rig_calibration.py prepare results/sim/dataset/manifest.json --out results/sim/dataset/prepared
```

`check` drives the live objects from the capture: `rig_state_producer.ShipState`
ticked at 10 Hz, `heading_offset_samples` (the `calibrate-heading` core),
`RangeHypothesisTma`, and waterline range. `--nlos-deg 60` switches on the
steep-path NLOS hypothesis (see below). Ship 1 is always the observer, because
it has the only working camera.

| Scenario | What it exercises |
| --- | --- |
| `static` | both hulls where they sat on 2026-09-18, on the anchor sets measured there; the noise floor |
| `range-sweep` | ship 1 holds and films, ship 2 zigzags 3–22 m out; the calibration capture, and the base of `dataset` |
| `weave` | ship 1 weaves toward a crossing target; bearings-only observability |
| `heading-cal` | ship 2 on long straight legs; what `calibrate-heading` needs |
| `pivot` | ship 1 pivots and sweeps its camera |
| `corner` | ship 2 drives into the corner where it parked on 2026-09-18 |
| `pursuit` | ship 2 chases ship 1 on true positions. This is a baseline, **not** the deployed DQN |

## What is measured and what is assumed

Taken from measurement: anchor survey, tag baselines and plane heights
(`boatdet.uwb`), the anchor sets measured per rig, IMU drift and noise per ship,
ship 2's broken `timestamp_capture`, clock offsets, cam0 intrinsics and
distortion, 4.31 fps, and 0.69 px of track jitter (5 px at native resolution).

Assumed, and each assumption can be swapped out in `boatdet/twin.py`:

- **Hull mass, drag and thrust** are not identified. They give 0.35 m/s at
  PWM 20, ~1 m/s at full thrust and ~29°/s in a PWM ±40 pivot, with a 1.2 s motor
  lag. Replace them once a step response has been logged.
- **Tag mounting angles** (32° for ship 1, −14° for ship 2), **camera mount
  yaw** (1.7°), **camera height** (0.30 m) and **water extent** (the wall-anchor
  rectangle) are unsurveyed placeholders.
- **Anchor selection**: the four nearest anchors by slant range, with jitter and
  hysteresis. The real firmware rule is unknown.
- **Detection** is a probability on box height, not a model of the trained
  detector. On real footage that detector misses this target entirely.
- **The motor readback** uses the producer's own assumed 48 ± 50 scale, so this
  conversion is circular and the twin does not test it.

## Findings (2026-09-24, synthetic)

1. **`RangeHypothesisTma` false-converges after a bearing gap.** In `weave` the
   target leaves the frame for about 20 s. When it comes back, the bank reports
   `converged` at 29–37 m with a tight interval, while the true range is 12.6 m.
   This happens even with perfect bearings: 114 converged frames with the interval
   missing the truth. The same run fed continuously gives 0. The mechanism is in
   `BearingOnlyEKF`: `dt` is clamped to `TMA_MAX_DT_S = 2.0`, so 20 s of silence
   is predicted as 2 s of motion and 2 s of process noise.
   `rig_calibration.py evaluate` avoids this, because it builds a new estimator
   after gaps longer than 2 s. Whether the `run.py` stage does the same when it
   reacquires a track has not been checked.
2. **The steep-path NLOS hypothesis contradicts the rigs.** On 2026-09-18 ship 2
   at (19.4, 15.3) solved to 0.035 m RMSE on a set that included A03 at ~73°
   elevation. With the hypothesis on, that same position rejects 811 fixes in
   60 s. It is therefore off by default. With it off, the twin has no mechanism
   for the degenerate anchor set, because that mechanism is still unknown.
3. **Geometry alone does not explain the firmware's `uwb/position` error.** A
   free 3D least-squares solve of unbiased ranges on ship 1's measured set
   (A02, A04, A08, A09) is unbiased in x, y and z. The rigs show z = 2.9 m and a
   1.3–2.3 m error in y. That needs biased ranges or a different solve inside the
   firmware.
4. **The RMSE gate is sound on 4-anchor bursts, weak on 3-anchor ones.** Under
   the NLOS hypothesis, no 4-anchor fix with more than 1 m of error passed the
   0.25 m gate. Among 3-anchor fixes (one range dropped), 6 of 252 did. Three
   ranges for two unknowns leave the RMSE almost no redundancy.
5. **The waterline range fit is biased by its own noise.** Depression noise sits
   in a denominator, so raw range has a heavy tail. `rig_calibration.py fit`
   recovers the mount yaw to 0.006° (1.706° against 1.700°). For range it fits a
   scale of 0.257 plus 1.9 m of offset, against a true camera height of 0.30 m.
   Single-frame waterline range error is 7–15 % at the assumed 0.30 m height.
6. **`calibrate-heading` works from a hull under way, not from one holding
   station.** Ship 2 on straight legs recovers its mounting angle to 1–3° with a
   13–17° spread. A holding hull still produces "samples" from velocity noise, but
   their spread (90–145°) trips the tool's own 20° warning, as it should.
7. **In the basin, bearings-only TMA stays `ambiguous`.** This agrees with
   [TMA.md](TMA.md): the intervals cover the truth, and range has to come from
   elsewhere.
