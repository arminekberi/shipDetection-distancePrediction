import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import run
from audit_dataset import audit
from boatdet import overlay
from boatdet.config import OverlayConfig
from boatdet.dataset import recording_tag, require_mounted_destination, save_annotation
from boatdet.depth import DepthEstimator, DepthFrame, distance_in_box
from viewer import server as panel_server


class DatasetTests(unittest.TestCase):
    def test_server_recordings_have_distinct_tags(self):
        self.assertNotEqual(recording_tag('/captures/20260910-120000/ShipCam0.mp4'),
                            recording_tag('/captures/20260910-130000/ShipCam0.mp4'))

    def test_clip_box_and_audit_written_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            save_annotation(directory, 'train', 'recording', 0,
                            np.zeros((20, 40, 3), np.uint8), (-5, 2, 20, 12))
            report = audit(directory)
            self.assertEqual(report['issues'], [])
            self.assertEqual(report['counts']['train']['positive'], 1)

    def test_failed_image_encoding_cannot_commit_label(self):
        with tempfile.TemporaryDirectory() as directory, patch('boatdet.dataset.cv2.imencode', return_value=(False, None)):
            with self.assertRaises(OSError):
                save_annotation(directory, 'train', 'recording', 0, np.zeros((20, 40, 3), np.uint8))
            self.assertFalse((Path(directory) / 'labels').exists())

    def test_dropped_mount_rejects_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch('boatdet.dataset.os.path.ismount', return_value=False):
            with self.assertRaisesRegex(OSError, 'Remote mount'):
                require_mounted_destination(Path(directory) / 'shipcaps_remote' / '_dataset')

    def test_audit_distinguishes_duplicate_names_from_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            for split, value in [('train', 0), ('val', 128)]:
                save_annotation(directory, split, 'recording', 0, np.full((20, 40, 3), value, np.uint8))
            kinds = {i['kind'] for i in audit(directory, hash_images=True)['issues']}
            self.assertIn('frame_across_splits', kinds)
            self.assertNotIn('identical_image_across_splits', kinds)


class PipelineTests(unittest.TestCase):
    def test_distance_rejects_invalid_model_pixels(self):
        depth = np.array([[np.nan, 3], [np.inf, -2]], dtype=float)
        self.assertEqual(distance_in_box(depth, (0, 0, 2, 2)), 3)
        self.assertIsNone(distance_in_box(np.zeros((2, 2)), (0, 0, 2, 2)))

    def test_invalid_target_is_rejected(self):
        for box in ('1,1,1,2', '-1,0,5,5', '0,0,999,999'):
            with self.subTest(box=box), self.assertRaises(ValueError):
                run.parse_box(box, 640, 360)

    def test_replay_keeps_first_frame_fps_and_uncalibrated_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'result.mp4'
            log = out.with_suffix('.csv')
            cap = MagicMock()
            frame = np.zeros((32, 32, 3), np.uint8)
            cap.read.side_effect = [(True, frame)] * 3 + [(False, None)]
            cap.get.return_value = 7.0
            tracker = MagicMock()
            tracker.update.return_value = (True, (4, 4, 10, 10))
            depth = np.full((32, 32), 4.5)
            argv = ['run.py', '--camera', 'test.mp4', '--no-display', '--target', '4,4,14,14',
                    '--width', '32', '--height', '32', '--output', str(out), '--log', str(log)]
            with patch('sys.argv', argv), patch('run.open_video', return_value=cap), \
                 patch('boatdet.tracking.cv2.TrackerCSRT_create', return_value=tracker), \
                 patch.object(DepthEstimator, 'load', return_value=DepthEstimator(None, None, 'cpu', (32, 32))), \
                 patch.object(DepthEstimator, 'infer', return_value=DepthFrame(depth, frame.copy(), 2.0, (4.0, 5.0))), \
                 patch('boatdet.video.cv2.VideoWriter') as writer:
                run.main()
                self.assertEqual(writer.call_args.args[2], 7.0)
                self.assertEqual(tuple(writer.call_args.args[3]), overlay.composed_size((32, 32)))
                self.assertEqual(writer.return_value.write.call_count, 3)
            with log.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['frame'] for r in rows], ['0', '1', '2'])
            self.assertEqual(float(rows[0]['distance_raw_m']), 4.5)
            self.assertEqual(float(rows[0]['distance_model_m']), 4.5)
            self.assertAlmostEqual(float(rows[2]['time_s']), 2 / 7, places=6)
            meta = json.loads(out.with_suffix('.meta.json').read_text())
            self.assertTrue(meta['sync'])
            self.assertFalse(meta['calibrated'])
            self.assertEqual(meta['frames'], 3)
            cap.release.assert_called_once()


def fake_track(track_id=1, bbox=(40, 60, 120, 140), class_name='boat', ttc=None, glare=False):
    return SimpleNamespace(track_id=track_id, bbox_xyxy=bbox, class_name=class_name, confidence=0.8,
                           distance_m=12.5, relative_velocity_mps=-1.5, ttc_s=ttc,
                           measurement_source='depth', apriltag_id=None, glare_suspect=glare,
                           water_contact_px=None, missed_frames=0)


class DepthPanelTests(unittest.TestCase):
    """The depth map carries the same targets as the camera panel, and its own legend."""

    def setUp(self):
        self.depth_color = np.full((180, 320, 3), 40, np.uint8)
        self.span = (2.0, 20.0)

    def colors(self, panel):
        return {tuple(int(c) for c in color) for color in panel.reshape(-1, 3)}

    def test_tracked_vessel_is_drawn_on_the_depth_panel(self):
        panel = overlay.render_depth(self.depth_color, [fake_track()], self.span)
        self.assertIn(overlay.TRACK_COLOR, self.colors(panel))
        self.assertIn(overlay.WARNING_COLOR,
                      self.colors(overlay.render_depth(self.depth_color, [fake_track(ttc=4.0)], self.span)))

    def test_overlays_off_leave_the_depth_map_and_its_legend_alone(self):
        panel = overlay.render_depth(self.depth_color, [fake_track()], span=None,
                                     config=OverlayConfig.disabled())
        self.assertNotIn(overlay.TRACK_COLOR, self.colors(panel))
        np.testing.assert_array_equal(panel[40:, :200], self.depth_color[40:, :200])

    def test_colorbar_ticks_report_calibrated_meters(self):
        plain = overlay.render_depth(self.depth_color, [], self.span)
        calibrated = overlay.render_depth(self.depth_color, [], self.span, calibration=(2.0, 5.0))
        self.assertFalse(np.array_equal(plain, calibrated), 'tick labels ignored the depth calibration')

    def test_labels_keep_clear_of_the_colorbar_gutter(self):
        panel = overlay.render_depth(self.depth_color, [fake_track(bbox=(250, 40, 310, 90))], self.span)
        gutter = panel[:, panel.shape[1] - overlay.COLORBAR_GUTTER:]
        self.assertNotIn(overlay.TRACK_COLOR, self.colors(gutter))


class ComposedFrameTests(unittest.TestCase):
    def test_frame_size_and_legend_match_the_declared_output_size(self):
        panels = [np.zeros((180, 320, 3), np.uint8)] * 2
        frame = overlay.compose(panels, status='total 40 ms (25 fps)')
        self.assertEqual((frame.shape[1], frame.shape[0]), overlay.composed_size((320, 180)))
        legend = frame[frame.shape[0] - overlay.LEGEND_HEIGHT:].reshape(-1, 3).astype(int)
        for kind, color, label in overlay.LEGEND_ITEMS:
            if color is None:
                continue
            # Swatches are antialiased, so the nearest drawn color stands in for an exact match.
            distance = np.abs(legend - np.array(color)).sum(axis=1).min()
            self.assertLess(distance, 30, f'{label} is missing from the legend')


class PanelValidationTests(unittest.TestCase):
    def test_nested_server_source_and_invalid_targets(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(panel_server, 'ROOT', Path(directory)):
            video = Path(directory) / 'recording' / 'ShipCam0.mp4'
            video.parent.mkdir()
            video.touch()
            self.assertEqual(panel_server.validate_request({'video': 'recording/ShipCam0.mp4', 'target': '0,0,20,20'}),
                             ('recording/ShipCam0.mp4', '0,0,20,20'))
            for request in ([], {'video': '../bad.mp4'}, {'video': str(video)},
                            {'video': 'recording/ShipCam0.mp4', 'target': '20,20,0,0'}):
                with self.subTest(request=request), self.assertRaises(ValueError):
                    panel_server.validate_request(request)


if __name__ == '__main__':
    unittest.main()
