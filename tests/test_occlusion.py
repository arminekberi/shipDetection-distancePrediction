"""Occlusion lifecycle, identity gates and honest prediction output."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from boatdet.config import MotConfig, UNKNOWN_OBSTACLE_CLASS
from boatdet.detection import Detection
from boatdet.mot import MultiObjectTracker
from boatdet.overlay import track_labels, track_state
from boatdet.pipeline import PerceptionPipeline
from boatdet.telemetry import TrackLogger


def detection(x=100, confidence=.9, **kwargs):
    return Detection((x, 100, x + 40, 130), confidence, **kwargs)


def colored_frame(x=100, color=(0, 0, 255)):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[100:130, int(x):int(x)+40] = color
    return frame


def tracker(**kwargs):
    return MultiObjectTracker((640, 360), MotConfig(**{
        'occlusion_seconds': 8., 'min_hits': 3, **kwargs}))


class OcclusionTests(unittest.TestCase):
    def warm(self, mot, moving=False):
        for i in range(20):
            x = 100 + (i * 2 if moving else 0)
            mot.update([detection(x)], i / 10, colored_frame(x))

    def test_same_id_after_four_seconds_hidden_at_different_frame_rates(self):
        for fps in (5, 30):
            with self.subTest(fps=fps):
                mot = tracker()
                self.warm(mot, moving=True)
                descriptor = mot.tracks[0].descriptor.copy()
                for i in range(1, 4 * fps + 1):
                    tracks = mot.update([], 1.9 + i / fps, colored_frame(0))
                    self.assertEqual([t.track_id for t in tracks], [1])
                track = tracks[0]
                self.assertAlmostEqual(track.seconds_since_detection, 4.)
                self.assertEqual(track.as_record()['tracking_status'], 'predicted')
                self.assertAlmostEqual(track.center_px[0], 238., delta=8.)
                np.testing.assert_array_equal(mot.tracks[0].descriptor, descriptor)
                tracks = mot.update([detection(220)], 6., colored_frame(220))
                self.assertEqual([(t.track_id, t.missed_frames) for t in tracks], [(1, 0)])
                self.assertEqual(tracks[0].seconds_since_detection, 0.)

    def test_timeout_expires_before_late_matching_detection(self):
        mot = tracker()
        self.warm(mot)
        mot.update([], 2.)
        mot.update([detection()], 10., colored_frame())
        self.assertEqual([t.track_id for t in mot.states()], [2])

    def test_no_unlimited_ghost_or_extension_for_unconfirmed_tracks(self):
        mot = tracker()
        self.warm(mot)
        for i in range(20, 105):
            mot.update([], i / 10)
        self.assertEqual(mot.states(), [])
        mot = tracker(max_missed_frames=1)
        mot.update([detection()], 0.)
        mot.update([], .1)
        mot.update([], .2)
        self.assertEqual(mot.states(), [])

    def test_occluding_unknown_obstacle_cannot_take_boat_identity(self):
        mot = tracker()
        self.warm(mot)
        mot.update([detection(class_name=UNKNOWN_OBSTACLE_CLASS, source='segmentation')], 2.)
        boat = next(t for t in mot.states() if t.track_id == 1)
        self.assertEqual(boat.missed_frames, 1)
        self.assertEqual(boat.source, 'yolo')
        self.assertEqual(len(mot.states()), 2)

    def test_different_appearance_cannot_reclaim_even_with_exact_overlap(self):
        mot = tracker()
        self.warm(mot)
        mot.update([], 2.)
        mot.update([detection()], 2.1, colored_frame(color=(255, 0, 0)))
        self.assertEqual(next(t for t in mot.states() if t.track_id == 1).missed_frames, 2)
        self.assertEqual(len(mot.states()), 2)

    def test_weak_detection_requires_appearance_to_reclaim(self):
        mot = tracker(appearance_weight=0)
        self.warm(mot)
        mot.update([], 2.)
        mot.update([detection(confidence=.2)], 2.1)
        self.assertEqual(mot.confirmed()[0].missed_frames, 2)
        mot = tracker()
        self.warm(mot)
        mot.update([], 2.)
        self.assertEqual(mot.update([detection(confidence=.2)], 2.1, colored_frame())[0].missed_frames, 0)

    def test_prediction_overlay_and_csv_are_not_new_measurements(self):
        mot = tracker()
        self.warm(mot)
        mot.observe(1, distance_m=12, timestamp=1.9, source='depth')
        track = mot.update([], 2.)[0]
        track.ttc_s = 3.  # Even a retained metric prediction must not read as a fresh risk fix.
        self.assertEqual(track_state(track), 'PREDICTED')
        self.assertIn('PREDICTED', track_labels(track)[0])
        self.assertFalse(any('TTC' in line or '12.0 m' in line for line in track_labels(track)))
        self.assertIsNone(track.water_contact_px)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tracks.csv'
            logger = TrackLogger(path)
            logger.write_tracks(20, 2., [track])
            logger.close()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['tracking_status'], 'predicted')
            self.assertEqual(float(row['seconds_since_detection']), .1)
            self.assertNotIn(None, row)

    def test_pipeline_does_not_sample_occluding_background_as_boat_range(self):
        class Detector:
            visible = True
            def candidates(self, *args):
                return [(detection().bbox_xyxy, .9)] if self.visible else []
        detector = Detector()
        pipeline = PerceptionPipeline(detector, (640, 360), mot_config=MotConfig(occlusion_seconds=8))
        for i in range(5):
            result = pipeline.process(colored_frame(), timestamp=i / 10,
                                      depth_m=np.full((360, 640), 10.))
        detector.visible = False
        for i in range(5, 40):
            result = pipeline.process(colored_frame(), timestamp=i / 10,
                                      depth_m=np.full((360, 640), 1.))
        self.assertEqual([t.track_id for t in result.tracks], [1])
        self.assertEqual(result.tracks[0].raw_distance_m, 10.)
        self.assertGreater(result.tracks[0].measurement_age, 30)
        self.assertIsNone(result.tracks[0].visual_depth_m)
        self.assertIsNone(result.tracks[0].visual_bearing_deg)

    def test_invalid_occlusion_limit_is_rejected(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MotConfig(occlusion_seconds=value)


if __name__ == '__main__':
    unittest.main()
