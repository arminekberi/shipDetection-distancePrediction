import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import run
from boatdet.apriltag import TagObservation
from boatdet.config import (TAGGED_VESSEL_CLASS, UNKNOWN_OBSTACLE_CLASS, FusionConfig, MotConfig,
                            OverlayConfig, SegmentationConfig)
from boatdet.detection import Detection, YoloDetector
from boatdet.fusion import fuse
from boatdet.overlay import render
from boatdet.pipeline import PerceptionPipeline
from boatdet.segmentation import (OBSTACLE, SKY, WATER, SegmentationModel, SegmentationResult,
                                  SegmentationRunner, SegmentationUnavailable, UltralyticsSegmentation,
                                  extract_obstacles)

WORKING = (640, 360)


def scene():
    """Sky above row 150, water below, one boat blob and one unknown blob on the water."""
    mask = np.full((360, 640), WATER, np.uint8)
    mask[:150] = SKY
    mask[160:200, 100:180] = OBSTACLE   # boat, reflection below it stays water
    mask[220:250, 400:430] = OBSTACLE   # debris no detector class covers
    return SegmentationResult(mask, 7.5)


class FakeSegmentation(SegmentationModel):
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        return SegmentationResult(self.result.mask.copy(), self.result.inference_time_ms)


class FakeYolo:
    names = {0: 'boat'}
    def __init__(self):
        self.boxes = []

    def predict(self, *args, **kwargs):
        return [SimpleNamespace(boxes=self.boxes)]


def yolo_box(box, confidence):
    return SimpleNamespace(xyxy=np.array([box], dtype=float), conf=[confidence], cls=[0])


class ObstacleTests(unittest.TestCase):
    def test_small_components_are_noise(self):
        result = scene()
        result.mask[300:302, 50:52] = OBSTACLE
        obstacles = extract_obstacles(result, min_area=100)
        self.assertEqual(len(obstacles), 2)
        self.assertTrue(all(o.area_px >= 100 for o in obstacles))

    def test_water_contact_is_the_blob_bottom(self):
        boat = min(extract_obstacles(scene(), 100), key=lambda o: o.bbox_xyxy[0])
        self.assertEqual(boat.water_contact_px, (139.5, 199.0))
        self.assertEqual(boat.bbox_xyxy, (100.0, 160.0, 180.0, 200.0))


class FusionTests(unittest.TestCase):
    def test_segmentation_only_obstacle_is_unknown(self):
        result = scene()
        fused = fuse([], result, extract_obstacles(result, 100))
        self.assertEqual({d.class_name for d in fused}, {UNKNOWN_OBSTACLE_CLASS})
        self.assertTrue(all(d.source == 'segmentation' for d in fused))

    def test_known_vessel_on_obstacle_mask_stays_a_vessel(self):
        result = scene()
        # The detector box runs 20 px below the hull, onto its reflection.
        boat = Detection((100, 158, 180, 220), .8)
        fused = fuse([boat], result, extract_obstacles(result, 100))
        vessels = [d for d in fused if d.class_name == 'boat']
        self.assertEqual(len(vessels), 1)
        self.assertTrue(vessels[0].segmentation_confirmed)
        self.assertEqual(len([d for d in fused if d.class_name == UNKNOWN_OBSTACLE_CLASS]), 1)
        # Water contact comes from the mask, not from the reflection-stretched box bottom.
        self.assertEqual(vessels[0].water_contact_source, 'segmentation')
        self.assertLess(vessels[0].water_contact_px[1], 201)

    def test_sky_detection_is_weakened_not_dropped(self):
        result = scene()
        fused = fuse([Detection((300, 20, 340, 60), .8)], result, [])
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(fused[0].confidence, .8 * FusionConfig().sky_conf_scale)

    def test_fallback_contact_is_box_bottom(self):
        fused = fuse([Detection((10, 20, 50, 60), .8)])
        self.assertEqual((fused[0].water_contact_px, fused[0].water_contact_source), ((30, 60), 'bbox_bottom'))

    def test_tag_without_detection_creates_nothing_by_default(self):
        # Tags also hang on infrastructure; a bare tag is not a vessel.
        self.assertEqual(fuse([], tags=[TagObservation(11, (300, 200))]), [])

    def test_listed_vessel_tag_seeds_identity_only_track(self):
        corners = ((295, 195), (305, 195), (305, 205), (295, 205))
        config = FusionConfig(apriltag_seeds_tracks=True, apriltag_vessel_ids=(11,))
        fused = fuse([], tags=[TagObservation(11, (300, 200), corners), TagObservation(19, (50, 20))],
                     config=config)
        self.assertEqual([d.class_name for d in fused], [TAGGED_VESSEL_CLASS])
        self.assertIsNone(fused[0].water_contact_px)  # a marker is not the waterline

    def test_tag_inside_detection_does_not_duplicate_it(self):
        fused = fuse([Detection((250, 150, 350, 250), .8)], tags=[TagObservation(11, (300, 200))])
        self.assertEqual([d.class_name for d in fused], ['boat'])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.yolo = FakeYolo()
        self.frame = np.zeros((360, 640, 3), np.uint8)

    def pipeline(self, runner=None, **kwargs):
        return PerceptionPipeline(YoloDetector(self.yolo), WORKING, MotConfig(min_hits=1), runner,
                                  fusion_config=FusionConfig(enabled=runner is not None), **kwargs)

    def test_segmentation_every_n_frames_reuses_the_mask(self):
        model = FakeSegmentation(scene())
        runner = SegmentationRunner(model, SegmentationConfig(every_n_frames=3))
        results = [runner.process(self.frame, i) for i in range(6)]
        self.assertEqual(model.calls, 2)
        self.assertEqual([r.reused for r in results], [False, True, True, False, True, True])

    def test_unknown_obstacle_is_tracked_and_timed(self):
        pipeline = self.pipeline(SegmentationRunner(FakeSegmentation(scene())))
        self.yolo.boxes = [yolo_box((100, 158, 180, 220), .8)]
        result = pipeline.process(self.frame, timestamp=0.0, frame_index=0)
        self.assertEqual(sorted(t.class_name for t in result.tracks), ['boat', UNKNOWN_OBSTACLE_CLASS])
        for stage in ('detector', 'segmentation', 'tracking', 'total'):
            self.assertIn(stage, pipeline.timer.stages)
        render(self.frame, result)  # overlay draws every field without failing

    def test_overlays_off_leave_frame_unchanged(self):
        pipeline = self.pipeline(SegmentationRunner(FakeSegmentation(scene())))
        self.yolo.boxes = [yolo_box((100, 158, 180, 220), .8)]
        result = pipeline.process(self.frame, timestamp=0.0, frame_index=0)
        np.testing.assert_array_equal(render(self.frame, result, OverlayConfig.disabled()), self.frame)

    def test_detector_only_pipeline_works_without_segmentation(self):
        pipeline = self.pipeline()
        self.yolo.boxes = [yolo_box((100, 158, 180, 220), .8)]
        result = pipeline.process(self.frame, timestamp=0.0, frame_index=0)
        self.assertIsNone(result.segmentation)
        self.assertEqual([t.class_name for t in result.tracks], ['boat'])
        self.assertIsNone(result.tracks[0].distance_m)  # no depth, no calibration: no invented range

    def test_unknown_conf_below_track_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'never open a track'):
            PerceptionPipeline(YoloDetector(self.yolo), WORKING, MotConfig(conf_threshold=.6),
                               segmentation_runner=SegmentationRunner(FakeSegmentation(scene())),
                               fusion_config=FusionConfig(unknown_conf=.5))


class WeightsTests(unittest.TestCase):
    def test_missing_weights_raise_a_clear_error(self):
        with self.assertRaisesRegex(SegmentationUnavailable, 'not found'):
            UltralyticsSegmentation.load(SegmentationConfig(enabled=True, model_path='/nonexistent/seg.pt'), WORKING)
        with self.assertRaisesRegex(SegmentationUnavailable, 'not set'):
            UltralyticsSegmentation.load(SegmentationConfig(), WORKING)

    def test_enabled_without_path_is_a_configuration_error(self):
        with self.assertRaises(ValueError):
            SegmentationConfig(enabled=True)

    def test_disabled_segmentation_never_loads_weights(self):
        args = run.build_parser().parse_args(['--yolo-weights', 'w.pt', '--tracker', 'bytetrack'])
        with patch('boatdet.segmentation.UltralyticsSegmentation.load') as load:
            runner, config = run.build_segmentation(args, WORKING, 'cpu')
        load.assert_not_called()
        self.assertIsNone(runner)
        self.assertFalse(config.enabled)


class BackwardCompatibilityTests(unittest.TestCase):
    def parse(self, *argv):
        parser = run.build_parser()
        args = parser.parse_args(list(argv))
        run.validate(parser, args)
        return args

    def test_default_tracker_is_unchanged(self):
        args = self.parse('--camera', 'x.mp4', '--yolo-weights', 'w.pt', '--no-display')
        self.assertEqual(args.tracker, 'csrt')
        self.assertIsNone(args.segmentation_weights)

    def test_legacy_trackers_build_without_new_modules(self):
        args = self.parse('--camera', 'x.mp4', '--yolo-weights', 'w.pt', '--tracker', 'kalman')
        with patch('run.build_detector', return_value=YoloDetector(FakeYolo())):
            tracker = run.build_tracker(args, np.zeros((360, 640, 3), np.uint8))
        self.assertEqual(type(tracker).__name__, 'KalmanTracker')

    def test_multi_object_options_require_bytetrack(self):
        for argv in (['--segmentation-weights', 's.pt'], ['--track-log', 't.csv'], ['--calibration', 'c.json']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    patch('sys.stderr'):
                self.parse('--camera', 'x.mp4', '--yolo-weights', 'w.pt', *argv)

    def test_bytetrack_needs_a_detector(self):
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            self.parse('--camera', 'x.mp4', '--target', '1,1,20,20', '--tracker', 'bytetrack')

    def test_legacy_csv_header_is_unchanged(self):
        self.assertEqual(run.CSV_HEADER, ['frame', 'distance_raw_m', 'distance_smoothed_m',
                                          'inference_ms', 'time_s', 'distance_model_m'])


class TrackLogTests(unittest.TestCase):
    def test_track_log_writes_every_field(self):
        from boatdet.mot import MultiObjectTracker
        from boatdet.telemetry import TRACK_CSV_HEADER, TrackLogger
        mot = MultiObjectTracker(WORKING, MotConfig(min_hits=1))
        tracks = mot.update([Detection((10, 20, 50, 60), .8, water_contact_px=(30, 60),
                                       water_contact_source='bbox_bottom')], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tracks.csv'
            with TrackLogger(path) as log:
                log.write_tracks(0, 0.0, tracks)
            header, row = path.read_text().splitlines()
        self.assertEqual(header.split(','), TRACK_CSV_HEADER)
        self.assertEqual(row.split(',')[2:4], ['1', 'boat'])


if __name__ == '__main__':
    unittest.main()
