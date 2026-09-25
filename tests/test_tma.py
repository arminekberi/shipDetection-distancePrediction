import json
import math
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

import numpy as np

from boatdet.config import MotConfig, OverlayConfig, TmaConfig
from boatdet.tma import (BearingObservation, BearingOnlyEKF, MultiTargetTma, OwnShipState,
                         OwnShipTrack, RangeHypothesisTma, bearing_sigma, compass_from_course,
                         course_from_compass, geometric_range_sigma, wrap_angle)
from boatdet.detection import Detection
from boatdet.geometry import CameraCalibration
from boatdet.overlay import motion_label, track_labels
from boatdet.pipeline import PerceptionPipeline
from boatdet.telemetry import TRACK_CSV_HEADER, TrackLogger
from tma_sim import SCENARIOS, own_ship_at, target_at

WORKING = (640, 360)
CAMERA_MATRIX = [[600.0, 0.0, 320.0], [0.0, 600.0, 180.0], [0.0, 0.0, 1.0]]

NOISE = math.radians(0.5)


def bearings(scenario, seed=0, duration_s=240, rate_hz=1.0, noise_rad=NOISE):
    """Noisy bearing observations of a simulated scenario, with the truth alongside."""
    rng = np.random.default_rng(seed)
    for step in range(int(duration_s * rate_hz) + 1):
        timestamp = step / rate_hz
        own, target = own_ship_at(scenario, timestamp), target_at(scenario, timestamp)
        truth = math.atan2(target[1] - own.y_m, target[0] - own.x_m)
        yield BearingObservation(timestamp, wrap_angle(truth + rng.normal(0.0, noise_rad)),
                                 noise_rad, own.x_m, own.y_m), \
            math.hypot(target[0] - own.x_m, target[1] - own.y_m)


def run_scenario(name, **kwargs):
    # Defaults are scaled for the covered basin; an open-water scenario carries the
    # configuration its own scale needs, exactly as tma_sim.py runs it.
    estimator = RangeHypothesisTma(SCENARIOS[name].configured(TmaConfig()))
    estimate = true_range = None
    covered = 0
    total = 0
    for observation, true_range in bearings(SCENARIOS[name], **kwargs):
        estimate = estimator.update(observation)
        total += 1
        covered += estimate.range_low_m <= true_range <= estimate.range_high_m
    return estimate, true_range, covered / total


class AngleTests(unittest.TestCase):
    def test_wrap_keeps_differences_short(self):
        self.assertAlmostEqual(wrap_angle(math.radians(350) - math.radians(10)), math.radians(-20))
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)

    def test_compass_conversion_round_trips(self):
        for compass in (0.0, 45.0, 90.0, 180.0, 275.0):
            self.assertAlmostEqual(compass_from_course(course_from_compass(compass)), compass, places=6)
        self.assertAlmostEqual(compass_from_course(0.0), 90.0)   # east

    def test_bearing_sigma_adds_independent_sources(self):
        self.assertAlmostEqual(bearing_sigma(3.0, 600.0), 0.005)
        combined = bearing_sigma(3.0, 600.0, heading_sigma_rad=0.012, mount_sigma_rad=0.004)
        self.assertAlmostEqual(combined, math.sqrt(0.005 ** 2 + 0.012 ** 2 + 0.004 ** 2))
        with self.assertRaises(ValueError):
            bearing_sigma(3.0, 0.0)


class GeometricBoundTests(unittest.TestCase):
    def test_the_bound_is_the_range_times_the_angle_the_baseline_subtends(self):
        # 10 m of cross-LOS baseline at 1000 m is a 0.01 rad parallax; with 0.005 rad
        # of bearing noise the range cannot be known better than half of itself.
        self.assertAlmostEqual(geometric_range_sigma(1000.0, 0.01, 0.005), 500.0)
        self.assertAlmostEqual(geometric_range_sigma(1000.0, 0.10, 0.005), 50.0)

    def test_no_parallax_means_no_range_information(self):
        self.assertEqual(geometric_range_sigma(500.0, 0.0, 0.005), math.inf)
        self.assertEqual(geometric_range_sigma(0.0, 0.1, 0.005), math.inf)

    def test_the_reported_interval_never_beats_the_bound(self):
        estimator = RangeHypothesisTma(SCENARIOS['maneuver'].configured(TmaConfig()))
        for observation, _ in bearings(SCENARIOS['maneuver']):
            estimate = estimator.update(observation)
        bound = geometric_range_sigma(estimate.range_m, estimate.peak_parallax_ratio,
                                      observation.sigma_rad)
        self.assertGreaterEqual(estimate.range_sigma_m, bound - 1e-6)

    def test_a_geometry_with_no_parallax_reports_the_prior_window(self):
        config = SCENARIOS['straight'].configured(TmaConfig())
        estimator = RangeHypothesisTma(config)
        for observation, _ in bearings(SCENARIOS['straight']):
            estimate = estimator.update(observation)
        self.assertLessEqual(estimate.range_low_m, config.min_range_m)
        self.assertGreaterEqual(estimate.range_high_m, config.max_range_m)


class FilterTests(unittest.TestCase):
    def observation(self, bearing_rad=0.0, timestamp=0.0, sensor=(0.0, 0.0)):
        return BearingObservation(timestamp, bearing_rad, NOISE, sensor[0], sensor[1])

    def test_seed_places_the_state_on_the_ray_at_the_requested_range(self):
        filt = BearingOnlyEKF.seeded(self.observation(math.radians(30), sensor=(10.0, -5.0)), 400.0, 50.0)
        self.assertAlmostEqual(filt.state[0], 10.0 + 400.0 * math.cos(math.radians(30)), places=6)
        self.assertAlmostEqual(filt.state[1], -5.0 + 400.0 * math.sin(math.radians(30)), places=6)
        # Uncertainty is 50 m along the ray and the bearing noise across it, not a circle.
        along = np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
        across = np.array([-along[1], along[0]])
        self.assertAlmostEqual(math.sqrt(along @ filt.covariance[:2, :2] @ along), 50.0, places=6)
        self.assertAlmostEqual(math.sqrt(across @ filt.covariance[:2, :2] @ across), 400.0 * NOISE, places=6)

    def test_measurement_jacobian_matches_a_numerical_derivative(self):
        filt = BearingOnlyEKF.seeded(self.observation(), 500.0, 40.0)
        filt.state = np.array([420.0, 310.0, 1.0, -2.0])
        sensor = np.array([30.0, -20.0])

        def bearing(state):
            return math.atan2(state[1] - sensor[1], state[0] - sensor[0])

        offset = filt.state[:2] - sensor
        range_sq = float(offset @ offset)
        analytic = np.array([-offset[1] / range_sq, offset[0] / range_sq])
        for axis in (0, 1):
            step = np.zeros(4)
            step[axis] = 1e-4
            numeric = (bearing(filt.state + step) - bearing(filt.state - step)) / 2e-4
            self.assertAlmostEqual(analytic[axis], numeric, places=9)

    def test_bearings_shrink_the_across_range_uncertainty_a_coasting_filter_keeps(self):
        corrected = BearingOnlyEKF.seeded(self.observation(), 500.0, 40.0)
        coasted = BearingOnlyEKF.seeded(self.observation(), 500.0, 40.0)
        for step in range(1, 20):
            corrected.correct(self.observation(timestamp=float(step)))
            coasted.predict(float(step))
        across = np.array([0.0, 1.0])   # bearing zero: the line of sight runs along +x
        self.assertLess(across @ corrected.covariance[:2, :2] @ across,
                        across @ coasted.covariance[:2, :2] @ across)
        self.assertTrue(np.all(np.linalg.eigvalsh(corrected.covariance) > 0))  # Joseph form stays PSD

    def test_prediction_without_measurements_only_grows_uncertainty(self):
        filt = BearingOnlyEKF.seeded(self.observation(), 500.0, 40.0)
        trace = np.trace(filt.covariance)
        filt.predict(30.0)
        self.assertGreater(np.trace(filt.covariance), trace)

    def test_implausible_speed_is_penalised_and_never_vetoed(self):
        filt = BearingOnlyEKF.seeded(self.observation(), 500.0, 40.0, TmaConfig(max_target_speed_mps=10.0))
        filt.state[2:] = [4.0, 3.0]
        self.assertEqual(filt.speed_log_prior(), 0.0)
        filt.state[2:] = [40.0, 30.0]
        self.assertLess(filt.speed_log_prior(), -1.0)
        self.assertTrue(math.isfinite(filt.speed_log_prior()))


class BankTests(unittest.TestCase):
    def test_hypotheses_tile_the_configured_range_window(self):
        config = TmaConfig(min_range_m=50.0, max_range_m=2000.0, hypotheses=8)
        estimator = RangeHypothesisTma(config)
        estimator.update(BearingObservation(0.0, 0.0, NOISE, 0.0, 0.0))
        ranges = sorted(float(np.hypot(*f.position)) for f in estimator.filters)
        self.assertEqual(len(ranges), 8)
        self.assertGreater(ranges[0], 50.0)
        self.assertLess(ranges[-1], 2000.0)
        self.assertAlmostEqual(sum(estimator.weights()), 1.0, places=9)

    def test_own_ship_manoeuvre_makes_the_range_observable(self):
        estimate, true_range, covered = run_scenario('maneuver')
        self.assertEqual(estimate.status, 'converged')
        self.assertTrue(estimate.observable)
        self.assertLess(abs(estimate.range_m - true_range) / true_range, 0.20)
        self.assertLess(abs(estimate.speed_mps - SCENARIOS['maneuver'].target_speed_mps), 2.0)
        self.assertLess(abs(math.degrees(wrap_angle(
            estimate.course_rad - course_from_compass(SCENARIOS['maneuver'].target_course_deg)))), 15.0)
        self.assertGreater(covered, 0.9)

    def test_a_straight_own_ship_never_claims_to_know_the_range(self):
        estimate, true_range, covered = run_scenario('straight')
        self.assertNotEqual(estimate.status, 'converged')
        self.assertFalse(estimate.observable)
        self.assertLess(estimate.parallax_ratio, 1e-9)
        # The interval has to stay wide enough to hold the truth it cannot resolve.
        self.assertGreater(covered, 0.9)
        self.assertLessEqual(estimate.range_low_m, true_range)
        self.assertGreaterEqual(estimate.range_high_m, true_range)

    def test_a_stationary_target_is_still_ambiguous_under_a_constant_velocity_model(self):
        estimate, _, covered = run_scenario('stationary-target')
        self.assertEqual(estimate.status, 'ambiguous')
        self.assertGreater(covered, 0.9)

    def test_parallax_latches_so_a_past_manoeuvre_keeps_counting(self):
        estimator = RangeHypothesisTma(SCENARIOS['maneuver'].configured(TmaConfig()))
        peak = 0.0
        for observation, _ in bearings(SCENARIOS['maneuver']):
            estimate = estimator.update(observation)
            peak = max(peak, estimate.parallax_ratio)
        self.assertLess(estimate.parallax_ratio, 1e-9)      # the turn left the window
        self.assertAlmostEqual(estimate.peak_parallax_ratio, peak)
        self.assertTrue(estimate.observable)

    def test_one_impossible_bearing_is_rejected_and_changes_nothing(self):
        estimator = RangeHypothesisTma(SCENARIOS['maneuver'].configured(TmaConfig()))
        for observation, _ in bearings(SCENARIOS['maneuver'], duration_s=60):
            before = estimator.update(observation)
        # A bearing 40 degrees off: no hypothesis explains it, so none may absorb it.
        outlier = BearingObservation(61.0, wrap_angle(observation.bearing_rad + math.radians(40)),
                                     NOISE, observation.sensor_x_m, observation.sensor_y_m)
        after = estimator.update(outlier)
        self.assertEqual(after.rejected, 1)
        self.assertEqual(after.reseeds, 0)
        self.assertAlmostEqual(after.range_m, before.range_m, delta=0.05 * before.range_m)

    def test_a_bank_that_keeps_disagreeing_restarts_instead_of_locking_out(self):
        config = SCENARIOS['maneuver'].configured(TmaConfig(gate_max_consecutive=5))
        estimator = RangeHypothesisTma(config)
        for observation, _ in bearings(SCENARIOS['maneuver'], duration_s=60):
            estimator.update(observation)
        offset = math.radians(40)   # a step no target motion can produce: a nav or id failure
        for step in range(1, 12):
            estimate = estimator.update(BearingObservation(
                60.0 + step, wrap_angle(observation.bearing_rad + offset), NOISE,
                observation.sensor_x_m, observation.sensor_y_m))
        self.assertEqual(estimate.reseeds, 1)
        self.assertEqual(estimate.rejected, 5)
        self.assertLess(estimate.peak_parallax_ratio, 1e-9)   # the old evidence went with the bank
        self.assertLess(estimate.parallax_ratio, 1e-9)        # and so did the manoeuvre history
        self.assertNotEqual(estimate.status, 'converged')

    def test_the_gate_can_be_turned_off(self):
        estimator = RangeHypothesisTma(SCENARIOS['maneuver'].configured(TmaConfig(gate_sigmas=0.0)))
        for observation, _ in bearings(SCENARIOS['maneuver'], duration_s=60):
            estimator.update(observation)
        estimate = estimator.update(BearingObservation(
            61.0, wrap_angle(observation.bearing_rad + math.radians(40)), NOISE,
            observation.sensor_x_m, observation.sensor_y_m))
        self.assertEqual(estimate.rejected, 0)

    def test_a_hypothesis_is_penalised_for_leaving_the_stated_range_window(self):
        config = TmaConfig(min_range_m=100.0, max_range_m=1000.0, range_prior_factor=2.0)
        observation = BearingObservation(0.0, 0.0, NOISE, 0.0, 0.0)
        inside = BearingOnlyEKF.seeded(observation, 500.0, 50.0, config)
        outside = BearingOnlyEKF.seeded(observation, 20.0, 5.0, config)
        self.assertEqual(inside.range_log_prior(observation), 0.0)
        self.assertLess(outside.range_log_prior(observation), -2.0)

    def test_bearings_must_arrive_in_time_order(self):
        estimator = RangeHypothesisTma(TmaConfig())
        estimator.update(BearingObservation(10.0, 0.0, NOISE, 0.0, 0.0))
        with self.assertRaises(ValueError):
            estimator.update(BearingObservation(9.0, 0.0, NOISE, 0.0, 0.0))

    def test_a_gap_without_bearings_widens_the_estimate(self):
        estimator = RangeHypothesisTma(SCENARIOS['maneuver'].configured(TmaConfig()))
        for observation, _ in bearings(SCENARIOS['maneuver'], duration_s=150):
            estimate = estimator.update(observation)
        coasted = estimator.estimate(estimate.timestamp + 60.0)
        self.assertGreater(coasted.position_sigma_m, estimate.position_sigma_m)
        self.assertEqual(coasted.updates, estimate.updates)

    def test_a_standstill_reports_an_unknown_course_instead_of_zero(self):
        estimator = RangeHypothesisTma(TmaConfig())
        estimate = estimator.update(BearingObservation(0.0, 0.3, NOISE, 0.0, 0.0))
        self.assertEqual(estimate.speed_mps, 0.0)
        self.assertEqual(estimate.course_sigma_rad, math.pi)
        self.assertEqual(estimate.status, 'initializing')

    def test_the_bank_is_never_pruned_into_a_single_mode(self):
        config = SCENARIOS['maneuver'].configured(TmaConfig(min_hypotheses=3, min_weight=0.4))
        estimator = RangeHypothesisTma(config)
        for observation, _ in bearings(SCENARIOS['maneuver'], duration_s=120):
            estimator.update(observation)
        self.assertGreaterEqual(len(estimator.filters), 3)

    def test_the_state_vector_is_the_documented_four_components(self):
        estimate, _, _ = run_scenario('closing', duration_s=120)
        vector = estimate.state_vector
        self.assertEqual(vector.shape, (4,))
        np.testing.assert_allclose(vector[:2], [estimate.x_m, estimate.y_m])
        self.assertAlmostEqual(vector[2], estimate.speed_mps)
        self.assertIn('course_deg', estimate.as_dict())


class MultiTargetTests(unittest.TestCase):
    def test_tracks_are_estimated_independently(self):
        bank = MultiTargetTma(TmaConfig())
        for step in range(30):
            timestamp = float(step)
            bank.observe(1, BearingObservation(timestamp, 0.2, NOISE, timestamp * 4.0, 0.0))
            bank.observe(7, BearingObservation(timestamp, -0.9, NOISE, timestamp * 4.0, 0.0))
        estimates = bank.estimates()
        self.assertEqual(set(estimates), {1, 7})
        self.assertNotAlmostEqual(estimates[1].bearing_rad, estimates[7].bearing_rad)
        bank.drop(7)
        self.assertEqual(set(bank.estimates()), {1})


class OwnShipTrackTests(unittest.TestCase):
    def samples(self):
        return [OwnShipState(0.0, 0.0, 0.0, math.radians(170), 5.0),
                OwnShipState(10.0, 50.0, 0.0, math.radians(-170), 5.0)]

    def test_interpolation_takes_the_short_way_round_the_heading_wrap(self):
        track = OwnShipTrack(self.samples(), max_age_s=10.0)
        middle = track.at(5.0)
        self.assertAlmostEqual(middle.x_m, 25.0)
        self.assertAlmostEqual(abs(math.degrees(middle.heading_rad)), 180.0, places=6)

    def test_a_stale_request_returns_nothing_rather_than_an_extrapolation(self):
        track = OwnShipTrack(self.samples(), max_age_s=1.0)
        self.assertIsNone(track.at(30.0))
        self.assertIsNone(track.at(5.0))        # the gap between samples is too wide to bridge
        held = track.at(10.5)                   # just past the end: the last sample is held
        self.assertEqual((held.x_m, held.y_m), (50.0, 0.0))

    def test_the_camera_lever_arm_moves_with_the_heading(self):
        state = OwnShipState(0.0, 0.0, 0.0, heading_rad=math.pi / 2)
        x, y = state.sensor_position(lever_forward_m=3.0, lever_starboard_m=1.0)
        self.assertAlmostEqual(x, 1.0, places=6)    # bow is +y, so starboard is +x
        self.assertAlmostEqual(y, 3.0, places=6)

    def test_a_bow_relative_bearing_becomes_an_absolute_one(self):
        state = OwnShipState(0.0, 100.0, 200.0, heading_rad=math.radians(90))
        observation = BearingObservation.from_relative(state, relative_bearing_deg=30.0, sigma_rad=NOISE)
        self.assertAlmostEqual(math.degrees(observation.bearing_rad), 60.0, places=6)
        self.assertEqual((observation.sensor_x_m, observation.sensor_y_m), (100.0, 200.0))


def own_ship_path(timestamp, turn_at_s=None, turn_to_deg=70.0, turn_rate_deg_s=3.0, speed_mps=5.0):
    """Own ship east at `speed_mps`, turning towards `turn_to_deg` at `turn_at_s` if asked.

    One definition shared by the navigation file and the detector stub, so the
    bearings the stub produces are the ones the navigation implies. An image that
    moves independently of the recorded own motion is a scenario no target can
    produce, and the estimator is right to reject it.
    """
    x = y = 0.0
    step = 0.05
    heading = 90.0
    for index in range(int(timestamp / step)):
        now = index * step
        if turn_at_s is not None and now >= turn_at_s:
            heading = max(turn_to_deg, 90.0 - turn_rate_deg_s * (now - turn_at_s))
        course = math.radians(90.0 - heading)
        x += speed_mps * step * math.cos(course)
        y += speed_mps * step * math.sin(course)
    return OwnShipState(timestamp=timestamp, x_m=x, y_m=y,
                        heading_rad=course_from_compass(heading), speed_mps=speed_mps)


class ScenarioDetector:
    """Detector stub whose box column is the geometry the navigation describes."""

    def __init__(self, turn_at_s=None, target=(886.0, 156.0), target_speed_mps=3.0,
                 target_course_deg=70.0, rate_hz=1.0):
        self.turn_at_s = turn_at_s
        self.target = target
        self.velocity = (target_speed_mps * math.cos(course_from_compass(target_course_deg)),
                         target_speed_mps * math.sin(course_from_compass(target_course_deg)))
        self.rate_hz = rate_hz
        self.frame = 0

    def candidates(self, frame, working_size):
        timestamp = self.frame / self.rate_hz
        self.frame += 1
        own = own_ship_path(timestamp, self.turn_at_s)
        target = (self.target[0] + self.velocity[0] * timestamp,
                  self.target[1] + self.velocity[1] * timestamp)
        absolute = math.atan2(target[1] - own.y_m, target[0] - own.x_m)
        relative = wrap_angle(own.heading_rad - absolute)   # positive to starboard
        u = CAMERA_MATRIX[0][2] + CAMERA_MATRIX[0][0] * math.tan(relative)
        return [((u - 15.0, 150.0, u + 15.0, 175.0), 0.9)]


def calibration():
    return CameraCalibration(camera_matrix=np.array(CAMERA_MATRIX), camera_height_m=3.0,
                             image_size=WORKING)


def nav_file(directory, turn_at_s=None, samples=121, step_s=0.5):
    """The same own-ship path, written out as a navigation log."""
    records = []
    for index in range(samples):
        state = own_ship_path(index * step_s, turn_at_s)
        records.append({'timestamp': state.timestamp, 'x_m': state.x_m, 'y_m': state.y_m,
                        'heading_deg': compass_from_course(state.heading_rad),
                        'speed_mps': state.speed_mps})
    path = Path(directory) / 'nav.json'
    path.write_text(json.dumps(records))
    return path


def pipeline(directory, turn_at_s=None, **kwargs):
    return PerceptionPipeline(ScenarioDetector(turn_at_s), WORKING, MotConfig(min_hits=1),
                              calibration=calibration(),
                              own_ship=OwnShipTrack.load(nav_file(directory, turn_at_s), max_age_s=1.0),
                              tma_config=TmaConfig(**kwargs))


class PipelineWiringTests(unittest.TestCase):
    def frames(self, pipe, count=40, rate_hz=1.0):
        frame = np.zeros((WORKING[1], WORKING[0], 3), dtype=np.uint8)
        result = None
        for index in range(count):
            result = pipe.process(frame, timestamp=index / rate_hz, frame_index=index)
        return result

    def test_navigation_and_intrinsics_produce_a_world_frame_state(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = pipeline(directory, turn_at_s=20.0)
            result = self.frames(pipe)
        track = result.tracks[0]
        self.assertIsNotNone(track.motion)
        self.assertEqual(track.motion.updates, 40)
        self.assertEqual((track.motion.rejected, track.motion.reseeds), (0, 0))
        self.assertIsNotNone(result.own_ship)
        self.assertIn('tma', result.timer.stages)
        self.assertIn(track.motion.status, ('ambiguous', 'converged'))

    def test_navigation_without_intrinsics_is_refused_instead_of_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            own_ship = OwnShipTrack.load(nav_file(directory))
            with self.assertRaisesRegex(ValueError, 'intrinsics'):
                PerceptionPipeline(ScenarioDetector(), WORKING, own_ship=own_ship)

    def test_a_frame_the_navigation_does_not_cover_is_left_unplaced(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = pipeline(directory)
            self.frames(pipe, count=5)
            result = pipe.process(np.zeros((WORKING[1], WORKING[0], 3), dtype=np.uint8),
                                  timestamp=500.0, frame_index=99)
        self.assertIsNone(result.own_ship)
        self.assertEqual(result.tracks[0].motion.updates, 5)   # no bearing was added

    def test_estimators_are_dropped_with_their_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = pipeline(directory)
            self.frames(pipe, count=5)
            self.assertEqual(set(pipe.tma.estimators), {1})
            pipe.detector.candidates = lambda frame, size: []
            for index in range(5, 40):
                pipe.process(np.zeros((WORKING[1], WORKING[0], 3), dtype=np.uint8),
                             timestamp=float(index), frame_index=index)
            self.assertEqual(pipe.tma.estimators, {})

    def test_the_track_csv_carries_the_motion_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            pipe = pipeline(directory, turn_at_s=20.0)
            result = self.frames(pipe)
            path = Path(directory) / 'tracks.csv'
            with TrackLogger(path) as log:
                log.write_tracks(0, 0.0, result.tracks)
            header, row = path.read_text().splitlines()
        self.assertEqual(header.split(','), TRACK_CSV_HEADER)
        columns = dict(zip(header.split(','), row.split(',')))
        self.assertEqual(columns['tma_status'], result.tracks[0].motion.status)
        self.assertTrue(float(columns['tma_range_low_m']) <= float(columns['tma_range_m'])
                        <= float(columns['tma_range_high_m']))

    def test_a_track_without_navigation_leaves_the_motion_columns_empty(self):
        from boatdet.mot import MultiObjectTracker
        tracker = MultiObjectTracker(WORKING, MotConfig(min_hits=1))
        tracks = tracker.update([Detection((10, 20, 50, 60), .8)], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tracks.csv'
            with TrackLogger(path) as log:
                log.write_tracks(0, 0.0, tracks)
            row = path.read_text().splitlines()[1]
        self.assertEqual(row.split(',')[-9:], [''] * 8 + [''])


class OverlayTests(unittest.TestCase):
    def estimate(self, status, **kwargs):
        from boatdet.tma import TargetEstimate
        fields = dict(timestamp=0.0, x_m=100.0, y_m=200.0, speed_mps=4.2,
                      course_rad=course_from_compass(225.0), range_m=620.0,
                      bearing_rad=0.4, range_sigma_m=40.0, range_low_m=410.0, range_high_m=1280.0,
                      position_sigma_m=50.0, speed_sigma_mps=0.4, course_sigma_rad=0.05,
                      effective_hypotheses=1.2, rejected=0, reseeds=0, parallax_ratio=0.05,
                      peak_parallax_ratio=0.05, observable=True, status=status, updates=50)
        return TargetEstimate(**dict(fields, **kwargs))

    def test_an_ambiguous_state_is_drawn_as_an_interval_not_a_range(self):
        label = motion_label(self.estimate('ambiguous'))
        self.assertIn('410-1280 m', label)
        self.assertIn('ambiguous', label)
        self.assertNotIn('620', label)

    def test_a_converged_state_is_drawn_as_range_speed_and_course(self):
        label = motion_label(self.estimate('converged'))
        self.assertIn('620 m', label)
        self.assertIn('4.2 m/s', label)
        self.assertIn('225deg', label)

    def test_the_motion_line_can_be_turned_off(self):
        track = SimpleNamespace(track_id=1, class_name='boat', confidence=0.9, apriltag_id=None,
                                distance_m=None, ttc_s=None, glare_suspect=False,
                                motion=self.estimate('converged'))
        self.assertTrue(any('TMA' in line for line in track_labels(track, OverlayConfig())))
        self.assertFalse(any('TMA' in line for line in
                             track_labels(track, OverlayConfig(show_motion=False))))


if __name__ == '__main__':
    unittest.main()
