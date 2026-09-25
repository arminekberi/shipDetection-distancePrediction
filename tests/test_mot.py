import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from boatdet.apriltag import ReplayAprilTagSource, TagObservation
from boatdet.config import MotConfig
from boatdet.detection import Detection
from boatdet.egomotion import Orientation, rotation_pixel_shift
from boatdet.geometry import CameraCalibration, water_intersection
from boatdet.motion import ConstantVelocityFilter, time_to_collision
from boatdet.mot import MultiObjectTracker

WORKING = (640, 360)
K = np.array([[600.0, 0, 320], [0, 600.0, 180], [0, 0, 1]])


def box(x, y=100, w=40, h=40):
    return (x, y, x + w, y + h)


def tracker(**config):
    return MultiObjectTracker(WORKING, MotConfig(**{'min_hits': 2, 'appearance_weight': 0.0, **config}))


class TrackIdentityTests(unittest.TestCase):
    def test_id_survives_short_missed_detection(self):
        mot = tracker(max_missed_frames=5)
        for i in range(4):
            mot.update([Detection(box(100 + 4 * i), .8)], i * .1)
        for i in range(4, 7):
            tracks = mot.update([], i * .1)
            self.assertEqual([t.track_id for t in tracks], [1])
        tracks = mot.update([Detection(box(128), .8)], .7)
        self.assertEqual([(t.track_id, t.missed_frames) for t in tracks], [(1, 0)])

    def test_track_dropped_after_max_missed_frames(self):
        mot = tracker(max_missed_frames=2)
        for i in range(3):
            mot.update([Detection(box(100), .8)], i * .1)
        for i in range(3, 6):
            mot.update([], i * .1)
        self.assertEqual(mot.tracks, [])

    def test_fast_target_is_not_a_new_track(self):
        mot = tracker()
        for i in range(4):
            mot.update([Detection(box(100 + 5 * i), .8)], i * .1)
        # 85 px jump: the boxes no longer overlap, only the motion gate can associate.
        tracks = mot.update([Detection(box(200), .8)], .4)
        self.assertEqual([t.track_id for t in tracks], [1])
        self.assertEqual(len(mot.tracks), 1)

    def test_rapid_scale_change_keeps_identity(self):
        mot = tracker()
        for i in range(3):
            mot.update([Detection(box(200), .8)], i * .1)
        mot.update([Detection((190, 90, 260, 160), .8)], .3)
        self.assertEqual(len(mot.tracks), 1)

    def test_crossing_tracks_stay_separate(self):
        mot = tracker()
        for i in range(12):
            left, right = 100 + 30 * i, 430 - 30 * i
            mot.update([Detection(box(left, 100), .8), Detection(box(right, 160), .8)], i * .1)
            tracks = {t.track_id: t for t in mot.confirmed()}
        self.assertEqual(sorted(tracks), [1, 2])
        # Track 1 started on the left and must end on the right.
        self.assertGreater(tracks[1].center_px[0], tracks[2].center_px[0])

    def test_weak_detection_keeps_track_alive_but_cannot_start_one(self):
        mot = tracker()
        for i in range(3):
            mot.update([Detection(box(100), .8)], i * .1)
        tracks = mot.update([Detection(box(101), .2)], .3)
        self.assertEqual([(t.track_id, t.missed_frames) for t in tracks], [(1, 0)])
        mot.update([Detection(box(500, 250), .2)], .4)
        self.assertEqual(len(mot.tracks), 1)

    def test_apriltag_binds_and_reconnects_after_crossing(self):
        mot = tracker()
        tag = lambda x, y: TagObservation(17, (x + 20, y + 20))
        for i in range(3):
            mot.update([Detection(box(100, 100), .8), Detection(box(400, 100), .8)], i * .1,
                       tags=[tag(100, 100)])
        owner = next(t for t in mot.states() if t.apriltag_id == 17)
        # Tag lost for two frames, then the tagged boat reappears where the other one was predicted.
        mot.update([Detection(box(100, 100), .8), Detection(box(400, 100), .8)], .3)
        tracks = mot.update([Detection(box(400, 100), .8), Detection(box(100, 100), .8)], .4,
                            tags=[tag(100, 100)])
        rebound = next(t for t in tracks if t.apriltag_id == 17)
        self.assertEqual(rebound.track_id, owner.track_id)

    def test_timestamps_drive_prediction(self):
        slow, fast = tracker(), tracker()
        for mot, dt in ((slow, .2), (fast, .1)):
            for i in range(10):
                mot.update([Detection(box(100 + 10 * i), .8)], i * dt)
        # Same pixel travel per frame, so the fast track moves twice the px/s.
        ratio = fast.tracks[0].image_filter.velocity[0] / slow.tracks[0].image_filter.velocity[0]
        self.assertAlmostEqual(float(ratio), 2.0, delta=.3)


class TtcTests(unittest.TestCase):
    def filtered_ttc(self, distances, dt=.1):
        mot = tracker(min_hits=1)
        for i, d in enumerate(distances):
            mot.update([Detection(box(200), .8)], i * dt)
            state = mot.observe(1, distance_m=d, timestamp=i * dt)
        return state

    def test_invalid_distance_gives_no_ttc(self):
        for distance in (None, float('nan'), float('inf'), 0.0, -3.0):
            self.assertIsNone(time_to_collision(distance, 5.0, .2, 120))
        state = self.filtered_ttc([float('nan')] * 3)
        self.assertIsNone(state.ttc_s)
        self.assertIsNone(state.distance_m)

    def test_approaching_target_has_positive_finite_ttc(self):
        state = self.filtered_ttc([60, 57, 54, 51, 48, 45, 42, 39])
        self.assertIsNotNone(state.ttc_s)
        self.assertTrue(math.isfinite(state.ttc_s) and state.ttc_s > 0)
        self.assertAlmostEqual(state.ttc_s, state.distance_m / 30, delta=.5)

    def test_receding_target_has_no_ttc(self):
        self.assertIsNone(self.filtered_ttc([40, 43, 46, 49, 52, 55]).ttc_s)
        self.assertIsNone(time_to_collision(30.0, -2.0, .2, 120))

    def test_range_noise_on_static_target_gives_no_ttc(self):
        mot = tracker(min_hits=1)
        rng = np.random.default_rng(1)
        ttcs = []
        for i in range(40):
            mot.update([Detection(box(200), .8)], i * .22)
            state = mot.observe(1, distance_m=11 + rng.normal(0, 1.0), timestamp=i * .22, measurement_noise=4.0)
            ttcs.append(state.ttc_s)
        self.assertLessEqual(sum(t is not None for t in ttcs), 2)

    def test_coasting_is_bounded_in_seconds(self):
        mot = tracker(max_missed_frames=100, max_coast_s=1.0)
        for i in range(3):
            mot.update([Detection(box(100), .8)], i * .2)
        mot.update([], 1.0)
        self.assertEqual(len(mot.tracks), 1)
        mot.update([], 1.7)
        self.assertEqual(mot.tracks, [])

    def test_slow_or_distant_target_has_no_ttc(self):
        self.assertIsNone(time_to_collision(30.0, .05, .2, 120))
        self.assertIsNone(time_to_collision(500.0, 1.0, .2, 120))

    def test_velocity_is_filtered_not_differenced(self):
        f = ConstantVelocityFilter(1, measurement_noise=.5, process_noise=2.0)
        rng = np.random.default_rng(0)
        for i in range(30):
            f.correct([50 - 3.0 * i * .1 + rng.normal(0, .3)], timestamp=i * .1)
        self.assertAlmostEqual(float(f.velocity[0]), -3.0, delta=1.0)

    def test_long_gap_resets_velocity(self):
        f = ConstantVelocityFilter(1, max_dt=1.0)
        for i in range(5):
            f.correct([10.0 * i], timestamp=float(i))
        f.predict(10.0)
        self.assertEqual(float(f.velocity[0]), 0.0)


class GeometryTests(unittest.TestCase):
    def test_water_plane_range_and_horizon(self):
        calibration = CameraCalibration(K, 2.0)
        estimate = water_intersection((320, 210), calibration)
        self.assertAlmostEqual(estimate.distance_m, 40.0, places=6)
        self.assertAlmostEqual(estimate.bearing_deg, 0.0)
        self.assertIsNone(water_intersection((320, 180), calibration))
        self.assertIsNone(water_intersection((320, 100), calibration))

    def test_missing_calibration_gives_no_range(self):
        self.assertIsNone(water_intersection((320, 300), None))

    def test_camera_pitch_changes_range(self):
        level = water_intersection((320, 210), CameraCalibration(K, 2.0))
        nose_down = water_intersection((320, 210), CameraCalibration(K, 2.0), Orientation(pitch_rad=math.radians(-1)))
        mount_up = water_intersection((320, 210), CameraCalibration(K, 2.0, mount_pitch_deg=1.0))
        # The same pixel looks more steeply down with the bow down, so it is nearer.
        self.assertLess(nose_down.distance_m, level.distance_m)
        self.assertGreater(mount_up.distance_m, level.distance_m)

    def test_ego_rotation_shift(self):
        shift = rotation_pixel_shift(Orientation(), Orientation(yaw_rad=math.radians(1)), K, (320, 180))
        self.assertAlmostEqual(shift[0], -600 * math.tan(math.radians(1)), places=4)
        self.assertAlmostEqual(shift[1], 0.0, places=6)
        # Nose up: the scene, horizon included, moves down in the image.
        shift = rotation_pixel_shift(Orientation(), Orientation(pitch_rad=math.radians(1)), K, (320, 180))
        self.assertAlmostEqual(shift[1], 600 * math.tan(math.radians(1)), places=4)
        self.assertIsNone(rotation_pixel_shift(Orientation(), Orientation(), None, (0, 0)))

    def test_ego_motion_compensation_keeps_static_target(self):
        """A static boat seen while the camera yaws must not gain velocity."""
        mot = MultiObjectTracker(WORKING, MotConfig(min_hits=1, appearance_weight=0.0), camera_matrix=K)
        for i in range(8):
            yaw = math.radians(1.5 * i)
            u = 320 - 600 * math.tan(yaw)
            mot.update([Detection((u - 20, 100, u + 20, 140), .8)], i * .1, orientation=Orientation(yaw_rad=yaw))
        compensated = abs(float(mot.tracks[0].image_filter.velocity[0]))
        plain = MultiObjectTracker(WORKING, MotConfig(min_hits=1, appearance_weight=0.0))
        for i in range(8):
            u = 320 - 600 * math.tan(math.radians(1.5 * i))
            plain.update([Detection((u - 20, 100, u + 20, 140), .8)], i * .1)
        self.assertLess(compensated, abs(float(plain.tracks[0].image_filter.velocity[0])) / 4)


class RigTagLogTests(unittest.TestCase):
    def test_rig_detections_scale_to_working_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'detections.jsonl'
            (Path(directory) / 'manifest.json').write_text(json.dumps({'width': 4608, 'height': 2592}))
            log.write_text(json.dumps({'file': 'ShipCam0_0000003_1.bgr', 'seq': 3, 'markers': [
                {'tag_id': 11, 'corners': [[720, 360], [792, 360], [792, 432], [720, 432]],
                 'bearing': {'az_cam_deg': -30.5}}]}) + '\n')
            source = ReplayAprilTagSource.from_rig_log(log, WORKING)
            tags = source.detect(None, frame_index=2)
            self.assertEqual(len(tags), 1)
            self.assertEqual(tags[0].tag_id, 11)
            self.assertAlmostEqual(tags[0].center_px[0], 756 * 640 / 4608)
            self.assertEqual(tags[0].bearing_deg, -30.5)
            self.assertFalse(tags[0].has_pose)


if __name__ == '__main__':
    unittest.main()
