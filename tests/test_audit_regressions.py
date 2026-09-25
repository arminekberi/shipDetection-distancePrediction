import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from audit_dataset import audit
from boatdet.config import MotConfig, RoiConfig
from boatdet.dataset import save_annotation, session_key
from boatdet.detection import YoloDetector
from boatdet.geometry import CameraCalibration
from boatdet.pipeline import PerceptionPipeline
from boatdet.proposal import HorizonProposer
from evaluate_roi import band_coverage, run_arm
from repair_dataset import Frame, clip_boxes
from run import track_distances


class DetectorTests(unittest.TestCase):
    def test_only_boat_class_is_forwarded_from_a_multiclass_checkpoint(self):
        boxes = [SimpleNamespace(xyxy=[[10, 10, 30, 30]], conf=[.9], cls=[c]) for c in (0, 8)]
        model = SimpleNamespace(names={0: 'person', 8: 'boat'})
        model.predict = lambda *a, **kw: [SimpleNamespace(boxes=boxes)]
        detector = YoloDetector(model)
        self.assertEqual(detector.class_ids, [8])
        self.assertEqual(len(detector.candidates(np.zeros((60, 60, 3), np.uint8), (60, 60))), 1)

    def test_marker_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'no boat'):
            YoloDetector(SimpleNamespace(names={0: 'apriltag'}))

    def test_overlapping_tiles_do_not_create_two_tracks_for_one_boat(self):
        # Width 100, tiles [0:60] and [40:100]; one boat at x=45..55.
        calls = iter([[(45, 10, 55, 30, .9), (5, 10, 15, 30, .7)],
                      [(5, 10, 15, 30, .8)]])
        def predict(*a, **kw):
            return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=[r[:4]], conf=[r[4]], cls=[0])
                                          for r in next(calls)])]
        model = SimpleNamespace(names={0: 'boat'}, predict=predict)
        detector = YoloDetector(model, tile_grid=(1, 2), tile_overlap=1/3)
        found = detector.candidates(np.zeros((40, 100, 3), np.uint8), (100, 40))
        self.assertEqual(len(found), 2)
        self.assertEqual([c for _, c in found], [.9, .7])


class RangeTests(unittest.TestCase):
    def test_lost_target_does_not_measure_background_or_rewind_filter(self):
        found = [((20, 40, 50, 60), .8)]
        detector = SimpleNamespace(candidates=lambda *a: found)
        pipeline = PerceptionPipeline(detector, (100, 80), MotConfig(min_hits=1))
        frame = np.zeros((80, 100, 3), np.uint8)
        pipeline.process(frame, timestamp=0, depth_m=np.full((80, 100), 10.0))
        found.clear()
        track = pipeline.process(frame, timestamp=.2, depth_m=np.full((80, 100), 2.0)).tracks[0]
        self.assertEqual(track.measurement_age, 1)
        self.assertAlmostEqual(track.distance_m, 10)
        self.assertEqual(pipeline.tracker.tracks[0].metric_filter.timestamp, .2)
        self.assertEqual(track_distances(track, 1, 0), (None, None, 10))

    def test_water_plane_does_not_remeasure_stale_contact(self):
        found = [((20, 40, 50, 60), .8)]
        calibration = CameraCalibration(np.array([[100., 0, 50], [0, 100, 40], [0, 0, 1]]), 1)
        pipeline = PerceptionPipeline(SimpleNamespace(candidates=lambda *a: found), (100, 80),
                                      MotConfig(min_hits=1), calibration=calibration)
        frame = np.zeros((80, 100, 3), np.uint8)
        pipeline.process(frame, timestamp=0)
        found.clear()
        track = pipeline.process(frame, timestamp=.2).tracks[0]
        self.assertEqual(track.measurement_source, 'water_plane')
        self.assertEqual(track.measurement_age, 1)

    def test_detector_only_allows_high_confidence_without_segmentation(self):
        pipeline = PerceptionPipeline(SimpleNamespace(candidates=lambda *a: []), (100, 80),
                                      MotConfig(conf_threshold=.9))
        self.assertEqual(pipeline.process(np.zeros((80, 100, 3), np.uint8)).tracks, [])


class DatasetAuditTests(unittest.TestCase):
    def test_labeling_plan_does_not_reintroduce_repaired_session_leaks(self):
        from label_video import PLAN
        sessions = {}
        for video, split, _, _ in PLAN:
            self.assertNotIn('20260902-151121', video)
            key = session_key(Path(video).stem + '_00000')
            sessions.setdefault(key, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in sessions.values()))

    def test_empty_and_corrupt_datasets_do_not_pass(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'images/train').mkdir(parents=True)
            Path(root, 'labels/train').mkdir(parents=True)
            self.assertIn('empty_dataset', {i['kind'] for i in audit(root)['issues']})
            Path(root, 'images/train/a_00000.jpg').write_bytes(b'not an image')
            Path(root, 'labels/train/a_00000.txt').write_text('')
            self.assertIn('undecodable_image', {i['kind'] for i in audit(root, verify_images=True)['issues']})

    def test_session_identity_normalizes_camera_order_and_timestamp_separator(self):
        self.assertEqual(session_key('20260910-143827_ShipCam0_00001'),
                         session_key('ShipCam1_20260910_143827_00002'))
        with tempfile.TemporaryDirectory() as root:
            for split, tag, value in [('train', '20260910-143827_ShipCam0', 20),
                                       ('val', 'ShipCam1_20260910_143827', 40)]:
                save_annotation(root, split, tag, 0, np.full((20, 40, 3), value, np.uint8))
            self.assertIn('session_across_splits', {i['kind'] for i in audit(root)['issues']})

    def test_identical_pixels_in_different_formats_are_detected(self):
        with tempfile.TemporaryDirectory() as root:
            for split, tag in [('train', 'one'), ('val', 'two')]:
                save_annotation(root, split, tag, 0, np.full((20, 40, 3), 70, np.uint8))
            original = Path(root, 'images/val/two_00000.jpg')
            cv2.imwrite(str(original.with_suffix('.png')), cv2.imread(str(original)))
            original.unlink()
            kinds = {i['kind'] for i in audit(root, True, True)['issues']}
            self.assertIn('identical_pixels_across_splits', kinds)
            self.assertNotIn('identical_image_across_splits', kinds)

    def test_nan_and_non_numeric_labels_are_quarantined_not_invented(self):
        with tempfile.TemporaryDirectory() as root:
            for index, line in enumerate(['0 nan .5 .1 .1', '0 .5 .5 inf .1', 'invalid row']):
                save_annotation(root, 'train', 'a', index, np.zeros((40, 40, 3), np.uint8))
                label = Path(root, f'labels/train/a_{index:05d}.txt')
                label.write_text(line)
                f = Frame('train', Path(root, f'images/train/a_{index:05d}.jpg'), label)
                clipped, quarantined = clip_boxes([f], True)
                self.assertEqual((clipped, quarantined), (0, [f]))
                self.assertEqual(label.read_text(), line)

    def test_repair_cannot_overwrite_an_existing_pair(self):
        with tempfile.TemporaryDirectory() as root:
            for split, value in [('train', 20), ('test', 40)]:
                save_annotation(root, split, 'a', 0, np.full((20, 40, 3), value, np.uint8))
            label = Path(root, 'labels/train/a_00000.txt')
            f = Frame('train', Path(root, 'images/train/a_00000.jpg'), label)
            before = Path(root, 'images/test/a_00000.jpg').read_bytes()
            with self.assertRaises(FileExistsError):
                f.move(Path(root), 'test')
            self.assertTrue(f.image.exists())
            self.assertEqual(Path(root, 'images/test/a_00000.jpg').read_bytes(), before)


class RoiTests(unittest.TestCase):
    def test_roll_is_measured_in_pixels_not_normalized_aspect_ratio(self):
        # 0.5 normalized slope at 16:9 is 15.7 degrees, within the 20 degree limit.
        from test_proposal import horizon_frame
        proposer = HorizonProposer(RoiConfig(smoothing=1))
        self.assertIsNotNone(proposer.detect_horizon(horizon_frame(80, 260)))

    def test_narrow_band_cannot_match_a_box_just_because_center_is_inside(self):
        self.assertEqual(band_coverage([[0, 0, 1, 1]], (40, 60), 100), 0)
        self.assertEqual(band_coverage([[0, .4, 1, .6]], (40, 60), 100), 1)

    def test_horizon_state_resets_between_recordings(self):
        states = []
        def propose(proposer, frame):
            states.append(proposer.line)
            proposer.line = (1, 2)
            return None
        items = [('a_00001', Path('unused'), []), ('a_00002', Path('unused'), []),
                 ('b_00001', Path('unused'), [])]
        model = SimpleNamespace(names={0: 'boat'}, predict=lambda *a, **kw: [SimpleNamespace(boxes=[])])
        with patch.object(HorizonProposer, 'propose', propose), \
                patch('evaluate_roi.cv2.imread', return_value=np.zeros((40, 100, 3), np.uint8)):
            run_arm(model, items, 'horizon', None, 960, RoiConfig())
        self.assertEqual(states, [None, (1, 2), None])


class TrialStatisticsTests(unittest.TestCase):
    def test_water_band_excludes_bright_deck_and_statistics_are_saved(self):
        from compare_trials import evaluate
        with tempfile.TemporaryDirectory() as root:
            video = Path(root, 'trial.avi')
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (100, 80))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((80, 100, 3), np.uint8)
            frame[64:] = 255
            for _ in range(3):
                writer.write(frame)
            writer.release()
            model = SimpleNamespace(names={0: 'boat'},
                                    predict=lambda *a, **kw: [SimpleNamespace(boxes=[])])
            args = SimpleNamespace(low_conf=.15, conf=.45, imgsz=960, band=(.25, .5),
                                   max_frames=None, sample_every=1, out=root, save_overlays=True)
            row = evaluate('A', str(video), model, args)
            self.assertEqual(row['frames'], 3)
            self.assertEqual(row['coverage'], 0)
            self.assertEqual(row['source_size'], (100, 80))
            self.assertEqual(row['water_highlight_median'], 0)
            self.assertIsNotNone(row['detector_ms']['p95'])
            self.assertTrue(Path(root, 'A/000000.jpg').is_file())


class RawInputTests(unittest.TestCase):
    def test_restarted_counter_does_not_rewind_capture_time(self):
        from boatdet.video import RawCapture
        with tempfile.TemporaryDirectory() as root:
            for name, value in [('cam_000001_1789136801541.bgr', 2),
                                ('cam_000002_1789136762371.bgr', 1)]:
                Path(root, name).write_bytes(np.full((9, 16, 3), value, np.uint8).tobytes())
            cap = RawCapture(root)
            self.assertEqual(cap.read()[1][0, 0, 0], 1)
            self.assertEqual(cap.read()[1][0, 0, 0], 2)
            self.assertGreater(cap.frame_time(1), cap.frame_time(0))

    def test_legacy_gray_suffix_with_color_manifest_preserves_bgr(self):
        from boatdet.video import RawCapture
        with tempfile.TemporaryDirectory() as root:
            frame = np.full((4, 6, 3), (10, 40, 200), np.uint8)
            Path(root, 'ShipCam1_000001_1789137984255.gray').write_bytes(frame.tobytes())
            Path(root, 'manifest.json').write_text(json.dumps(dict(
                width=6, height=4, channels=3, pix_fmt='bgr24', bytes_per_frame=72)))
            cap = RawCapture(root)
            ok, decoded = cap.read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(decoded, frame)

    def test_real_luma_manifest_is_converted_to_three_equal_channels(self):
        from boatdet.video import RawCapture
        with tempfile.TemporaryDirectory() as root:
            frame = np.arange(24, dtype=np.uint8).reshape(4, 6)
            Path(root, 'ShipCam1_000001_1789137984255.gray').write_bytes(frame.tobytes())
            Path(root, 'manifest.json').write_text(json.dumps(dict(
                width=6, height=4, channels=1, pix_fmt='gray', bytes_per_frame=24)))
            ok, decoded = RawCapture(root).read()
            self.assertTrue(ok)
            np.testing.assert_array_equal(decoded, np.repeat(frame[:, :, None], 3, axis=2))

    def test_ambiguous_gray_suffix_without_manifest_is_rejected(self):
        from boatdet.video import RawCapture
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'frame.gray').write_bytes(bytes(72))
            with self.assertRaisesRegex(ValueError, 'needs manifest'):
                RawCapture(root)


class SegmentationGeometryTests(unittest.TestCase):
    def test_letterbox_padding_is_removed_before_mask_mapping(self):
        import torch
        from boatdet.segmentation import OBSTACLE, UltralyticsSegmentation
        # Model input would be 512x320 for a 640x360 frame (16px vertical padding).
        native = np.zeros((1, 360, 640), np.uint8)
        native[:, 280:320, 100:180] = 1
        padded = np.pad(cv2.resize(native[0], (512, 288)), ((16, 16), (0, 0)))[None]
        def predict(*a, **kw):
            masks = native if kw.get('retina_masks') else padded
            return [SimpleNamespace(masks=SimpleNamespace(data=torch.from_numpy(masks)),
                                    boxes=SimpleNamespace(cls=torch.tensor([0])))]
        model = SimpleNamespace(names={0: 'boat'}, predict=predict)
        result = UltralyticsSegmentation(model, (640, 360)).predict(np.zeros((360, 640, 3), np.uint8))
        np.testing.assert_array_equal(result.mask == OBSTACLE, native[0].astype(bool))


if __name__ == '__main__':
    unittest.main()
