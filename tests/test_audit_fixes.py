import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import run
import panel_server
from audit_dataset import audit
from dataset_io import recording_tag, save_annotation, require_mounted_destination


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
        with tempfile.TemporaryDirectory() as directory, patch('dataset_io.cv2.imencode', return_value=(False, None)):
            with self.assertRaises(OSError):
                save_annotation(directory, 'train', 'recording', 0, np.zeros((20, 40, 3), np.uint8))
            self.assertFalse((Path(directory) / 'labels').exists())

    def test_dropped_mount_rejects_writes(self):
        with tempfile.TemporaryDirectory() as directory, patch('dataset_io.os.path.ismount', return_value=False):
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
        self.assertEqual(run.distance_in_box(depth, (0, 0, 2, 2)), 3)
        self.assertIsNone(run.distance_in_box(np.zeros((2, 2)), (0, 0, 2, 2)))

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
            with patch('sys.argv', argv), patch('run.cv2.VideoCapture', return_value=cap), \
                 patch('run.cv2.TrackerCSRT_create', return_value=tracker), \
                 patch('run.AutoImageProcessor.from_pretrained'), \
                 patch('run.AutoModelForDepthEstimation.from_pretrained'), \
                 patch('run.DepthWorker.infer', return_value=(depth, frame.copy(), 2.0)), \
                 patch('run.cv2.VideoWriter') as writer:
                run.main()
                self.assertEqual(writer.call_args.args[2], 7.0)
                self.assertEqual(writer.return_value.write.call_count, 3)
            with log.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['frame'] for r in rows], ['0', '1', '2'])
            self.assertEqual(float(rows[0]['distance_raw_m']), 4.5)
            self.assertAlmostEqual(float(rows[2]['time_s']), 2/7, places=6)
            meta = json.loads(out.with_suffix('.meta.json').read_text())
            self.assertTrue(meta['sync'])
            self.assertFalse(meta['calibrated'])
            self.assertEqual(meta['frames'], 3)
            cap.release.assert_called_once()


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
