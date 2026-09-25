import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from analyze_glare import analyze, highlight_metrics, normalized_roi
from boatdet.dataset import check_training_sources


class GlareTests(unittest.TestCase):
    def test_reserved_test_video_cannot_be_used_for_hard_mining(self):
        check_training_sources(['new_recording.mp4'])
        for video in ('renkliTekneTekne.mp4', '/recordings/renksizTekneTekne.mp4'):
            with self.subTest(video=video), self.assertRaisesRegex(ValueError, 'Reserved test'):
                check_training_sources([video])

    def test_local_highlight_and_single_channel_are_distinguished(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:10, :10] = 255
        frame[20:30, :10, 2] = 255
        result = highlight_metrics(frame)
        self.assertEqual(result['near_white_fraction'], .01)
        self.assertEqual(result['any_channel_high_fraction'], .02)
        self.assertEqual(result['largest_near_white_fraction'], .01)
        self.assertEqual(result['luma_p50'], 0)

    def test_sky_can_be_excluded_without_modifying_pixels(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:50] = 255
        original = frame.copy()
        self.assertEqual(highlight_metrics(frame)['near_white_fraction'], .5)
        self.assertEqual(highlight_metrics(frame, roi=(0, .5, 1, 1))['near_white_fraction'], 0)
        np.testing.assert_array_equal(frame, original)

    def test_full_white_and_near_white(self):
        frame = np.full((10, 10, 3), 252, dtype=np.uint8)
        result = highlight_metrics(frame)
        self.assertEqual(result['near_white_fraction'], 1)
        self.assertEqual(result['exact_white_fraction'], 0)
        self.assertEqual(highlight_metrics(frame, threshold=255)['near_white_fraction'], 0)

    def test_invalid_roi(self):
        for value in ('0,1,1,0', '0,0,2,1', '0,0,0,1', 'nan,0,1,1', '0,0,1'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                normalized_roi(value)

    def test_video_report_and_original_png(self):
        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / 'input.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                for brightness in (0, 80, 255):
                    writer.write(np.full((32, 32, 3), brightness, dtype=np.uint8))
            finally:
                writer.release()
            out = Path(root) / 'review'
            result = analyze(video, out, sample_every=1, top_k=1)
            self.assertEqual(result['decoded_frames'], 3)
            self.assertEqual(result['top_frames'][0]['frame'], 2)
            saved = cv2.imread(str(out / result['top_frames'][0]['image']))
            self.assertEqual(saved.shape, (32, 32, 3))
            with (out / 'frames.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            self.assertEqual(float(rows[-1]['time_s']), .2)
            with self.assertRaises(FileExistsError):
                analyze(video, out)

    def test_missing_video_fails(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'video not found'):
                analyze(Path(root) / 'missing.mp4', Path(root) / 'out')


if __name__ == '__main__':
    unittest.main()
