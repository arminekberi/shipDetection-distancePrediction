"""The basin twin: its physics is sane, and its wire output is what the real tools read.

The point of the twin is that `rig_state_producer`, `calibrate-heading`,
`bearings_to_observations` and `rig_calibration` accept its output UNCHANGED, so
most of these push twin output through those tools rather than testing the twin
against itself.
"""
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

import pool_sim
import rig_state_producer as producer
from boatdet.twin import (CameraParams, Camera, Hold, Hull, HullParams, HullState, Pool,
                          ShipSetup, Twin, UwbParams, Waypoints, default_clock, default_imu,
                          default_tags)
from boatdet.uwb import TagSolver, ytu_setup


def settle(left, right, seconds=30.0):
    hull = Hull(HullParams(), HullState(15.0, 15.0, 0.0))
    rng = np.random.default_rng(0)
    for _ in range(int(seconds / 0.01)):
        hull.step(0.01, left, right, rng, Pool())
    return hull.s


def records(recording):
    return [{'topic': topic, 'received_at': received, 'payload': payload}
            for received, topic, payload in recording.messages]


class HullTests(unittest.TestCase):
    def test_base_speed_is_a_slow_model_speed(self):
        state = settle(20, 20)
        self.assertGreater(state.u, 0.25)
        self.assertLess(state.u, 0.5)

    def test_opposed_screws_pivot_without_making_way(self):
        state = settle(40, -40)
        self.assertLess(state.r, -0.3)          # port screw ahead turns the bow to starboard
        self.assertLess(math.hypot(state.x - 15.0, state.y - 15.0), 1.0)

    def test_walls_hold_the_hull_in_the_pool(self):
        state = settle(100, 100, seconds=60.0)
        pool = Pool()
        self.assertTrue(pool.contains(state.x, state.y, margin=pool.margin_m - 1e-9))
        self.assertGreater(state.wall_contacts, 0)


class WireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setups = [ShipSetup(1, (19.4, 22.0, -90.0), Hold(), default_tags(1), imu=default_imu(1),
                            clock=default_clock(1), camera=CameraParams()),
                  ShipSetup(2, (16.0, 12.0, 0.0), Waypoints([(26.0, 12.0)], pwm=25.0), default_tags(2),
                            imu=default_imu(2), clock=default_clock(2))]
        cls.recording = Twin(setups, seed=3).run(40.0)

    def test_the_real_tag_solver_recovers_each_twin_tag(self):
        truth = {row['t_wall']: row for row in self.recording.truth[1]}
        stamps = sorted(truth)
        solver = TagSolver(ytu_setup('T01'))
        fixes = []
        for received, topic, payload in self.recording.messages:
            if topic.startswith('ship/1/uwb/ranging/T01/'):
                fixes += solver.accept(topic, payload, arrival=received)
        self.assertGreater(len(fixes), 300)
        errors = []
        for fix in fixes:
            nearest = truth[min(stamps, key=lambda t: abs(t - fix.t))]
            errors.append(math.hypot(fix.x - nearest['T01_x_m'], fix.y - nearest['T01_y_m']))
        self.assertLess(np.median(errors), 0.1)

    def test_the_producer_publishes_a_state_close_to_the_truth(self):
        states = producer.build_ships([1, 2], {1: default_tags(1).baseline_to_bow_deg,
                                               2: default_tags(2).baseline_to_bow_deg})
        for record in records(self.recording):
            states[record['payload']['ship_id']].accept(record['topic'], record['payload'],
                                                         arrival=record['received_at'])
        for ship, st in states.items():
            payload, why = st.payload(st.last_hull.t)
            self.assertIsNotNone(payload, why)
            last = self.recording.truth[ship][-1]
            self.assertLess(math.hypot(payload['px'] - last['mid_x_m'], payload['py'] - last['mid_y_m']), 0.2)
            error = (payload['hd'] - last['heading_rad'] + math.pi) % (2 * math.pi) - math.pi
            self.assertLess(abs(math.degrees(error)), 25.0)

    def test_the_producer_refuses_to_publish_an_unsurveyed_heading(self):
        states = producer.build_ships([2], {})
        for record in records(self.recording):
            if record['payload']['ship_id'] == 2:
                states[2].accept(record['topic'], record['payload'], arrival=record['received_at'])
        payload, why = states[2].payload(states[2].last_hull.t)
        self.assertIsNone(payload)
        self.assertIn('unsurveyed', why)

    def test_calibrate_heading_finds_the_mounting_angle_of_a_hull_under_way(self):
        offsets, _ = producer.heading_offset_samples(records(self.recording), 2)
        mean, sd = producer.circular_stats(offsets)
        truth = math.radians(default_tags(2).baseline_to_bow_deg)
        self.assertLess(abs(math.degrees((mean - truth + math.pi) % (2 * math.pi) - math.pi)), 6.0)

    def test_imu_yaw_is_clockwise_positive(self):
        setups = [ShipSetup(1, (15.0, 15.0, 0.0), lambda t, me, w: (-30.0, 30.0), default_tags(1),
                            imu=replace(default_imu(1), drift_deg_per_min=0.0, yaw_sd_deg=0.0),
                            clock=default_clock(1))]
        recording = Twin(setups, seed=0, camera_ship=None).run(3.0)
        yaws = [p['yaw'] for _, topic, p in recording.messages if topic.endswith('imu/data')]
        self.assertGreater(recording.truth[1][-1]['heading_rad'], 0.1)   # turned CCW
        self.assertLess(yaws[-1] - yaws[0], -5.0)                        # PLC yaw fell


class CameraTests(unittest.TestCase):
    def camera(self, **overrides):
        return Camera(replace(CameraParams(), pixel_sigma_px=0.0, detect_max=1.0, detect_h50_px=0.0,
                              **overrides), np.random.default_rng(0))

    def test_a_target_dead_ahead_reads_minus_the_mount_yaw(self):
        own, target = HullState(10.0, 10.0, 0.0), HullState(20.2, 10.0, 0.0)
        row, truth = self.camera().observe(own, target, HullParams())
        self.assertEqual(row['status'], 'ok')
        self.assertAlmostEqual(truth['true_rel_bearing_deg'], 0.0, places=6)
        self.assertAlmostEqual(row['az_deg'], -CameraParams().mount_yaw_deg, delta=0.05)

    def test_a_target_to_starboard_reads_a_positive_azimuth(self):
        own, target = HullState(10.0, 10.0, 0.0), HullState(20.0, 6.0, 0.0)
        row, truth = self.camera(mount_yaw_deg=0.0).observe(own, target, HullParams())
        self.assertGreater(row['az_deg'], 15.0)
        self.assertAlmostEqual(row['az_deg'], truth['true_az_deg'], delta=0.5)

    def test_the_waterline_gives_range_without_noise(self):
        own, target = HullState(10.0, 10.0, 0.0), HullState(18.0, 10.0, math.pi / 2)
        params = CameraParams()
        row, truth = self.camera().observe(own, target, HullParams())
        waterline = params.height_m / math.tan(math.radians(-row['el_water_deg']))
        # The nearest waterline edge is half a beam closer than the hull centre.
        self.assertAlmostEqual(waterline, truth['true_range_m'] - HullParams().beam_m / 2, delta=0.15)

    def test_a_target_behind_the_camera_is_not_seen(self):
        own, target = HullState(10.0, 10.0, 0.0), HullState(4.0, 10.0, 0.0)
        row, _ = self.camera().observe(own, target, HullParams())
        self.assertEqual(row['status'], 'lost')


class PipelineTests(unittest.TestCase):
    def test_check_scores_a_run_and_writes_every_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            recording, description = pool_sim.simulate('static', seed=1, duration_s=20.0)
            pool_sim.write_run(recording, folder, 'static', description, 1)
            report = pool_sim.check(folder, plots=False)
            for name in ('telemetry.jsonl', 'bearings.csv', 'observations.csv', 'state_ship1.csv',
                         'state_ship2.csv', 'session.json', 'check.json'):
                self.assertTrue((Path(folder) / name).exists(), name)
            self.assertTrue(report['synthetic'])
            for ship in (1, 2):
                self.assertLess(report['producer'][ship]['position_error_m']['median'], 0.1)
            # A hull that never makes way says nothing about its mounting angle.
            self.assertFalse(report['heading_calibration'][2].get('trusted', False))
            self.assertLess(report['camera']['azimuth_error_deg']['median'], 0.5)

    def test_rig_calibration_recovers_the_camera_mount_bias(self):
        from boatdet.supervision import prepare
        import rig_calibration
        with tempfile.TemporaryDirectory() as folder:
            manifest = pool_sim.dataset(folder, runs=3, seed=7, duration_s=40.0)
            rows, provenance = prepare(manifest, Path(folder) / 'prepared')
            calibration = rig_calibration.fit(rows, provenance)
            model = calibration['models']['twin-ship1-cam0']
            self.assertAlmostEqual(model['bearing_bias_deg'], CameraParams().mount_yaw_deg, delta=0.2)
            self.assertTrue(calibration['synthetic'])


if __name__ == '__main__':
    unittest.main()
