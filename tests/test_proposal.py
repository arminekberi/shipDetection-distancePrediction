import unittest
from types import SimpleNamespace

import numpy as np

from boatdet.config import RoiConfig
from boatdet.detection import YoloDetector
from boatdet.proposal import (FixedBandProposer, HorizonProposer, SegmentationProposer, TwoStageDetector,
                              build_proposer, fit_line_ransac)
from boatdet.segmentation import SKY, WATER, SegmentationResult

WORKING = (640, 360)


def horizon_frame(left=150, right=190, height=360, width=640):
    frame = np.zeros((height, width, 3), np.uint8)
    rows, columns = np.mgrid[:height, :width]
    below = rows >= left + (right - left) * columns / width
    frame[~below] = (200, 190, 180)
    frame[below] = (60, 50, 30)
    return frame


class RecordingYolo:
    names = {0: 'boat'}
    def __init__(self, boxes):
        self.boxes, self.seen = boxes, []

    def predict(self, source, **kwargs):
        self.seen.append(source.shape)
        return [SimpleNamespace(boxes=[SimpleNamespace(xyxy=np.array([b], float), conf=[c], cls=[0]) for b, c in self.boxes])]


class ProposerTests(unittest.TestCase):
    def test_fixed_band_rows(self):
        self.assertEqual(FixedBandProposer((.25, .75)).propose(np.zeros((400, 10, 3), np.uint8)), (100, 300))

    def test_ransac_ignores_outliers(self):
        xs = np.linspace(0, 1, 50)
        ys = .3 * xs + .4
        ys[::7] = .95
        slope, intercept, fraction = fit_line_ransac(xs, ys, .01)
        self.assertAlmostEqual(slope, .3, places=3)
        self.assertAlmostEqual(intercept, .4, places=3)
        self.assertGreater(fraction, .8)

    def test_horizon_follows_roll(self):
        proposer = HorizonProposer(RoiConfig(smoothing=1.0))
        top, bottom = proposer.propose(horizon_frame(150, 190))
        self.assertTrue(proposer.last_fit_ok)
        slope, intercept = proposer.line
        self.assertAlmostEqual(intercept, 150 / 360, delta=.01)
        self.assertAlmostEqual(slope, 40 / 360, delta=.01)
        self.assertLess(top, 150)
        self.assertGreater(bottom, 190)

    def test_horizon_moves_with_pitch(self):
        high = HorizonProposer(RoiConfig(smoothing=1.0)).propose(horizon_frame(120, 120))
        low = HorizonProposer(RoiConfig(smoothing=1.0)).propose(horizon_frame(220, 220))
        self.assertLess(high[0], low[0])

    def test_no_horizon_falls_back_to_fixed_band(self):
        proposer = HorizonProposer()
        band = proposer.propose(np.full((360, 640, 3), 90, np.uint8))
        self.assertFalse(proposer.last_fit_ok)
        self.assertEqual(band, FixedBandProposer().propose(np.zeros((360, 640, 3))))

    def test_segmentation_band_starts_above_water(self):
        mask = np.full((360, 640), SKY, np.uint8)
        mask[200:] = WATER
        runner = SimpleNamespace(last=SegmentationResult(mask, 1.0))
        top, bottom = SegmentationProposer(runner, RoiConfig(margin_above=.1)).propose(np.zeros((360, 640, 3)))
        self.assertEqual((top, bottom), (164, 360))

    def test_segmentation_mode_requires_a_runner(self):
        with self.assertRaises(ValueError):
            build_proposer(RoiConfig(mode='segmentation'))
        self.assertIsNone(build_proposer(RoiConfig()))


class TwoStageTests(unittest.TestCase):
    def test_band_boxes_map_back_to_working_frame(self):
        yolo = RecordingYolo([((100, 20, 300, 60), .8)])
        detector = TwoStageDetector(FixedBandProposer((.5, 1.0)), YoloDetector(yolo))
        native = np.zeros((720, 1280, 3), np.uint8)
        (box, confidence), = detector.candidates(native, WORKING)
        self.assertEqual(yolo.seen, [(360, 1280, 3)])  # only the band reaches the detector
        self.assertEqual(detector.last_band, (360, 720))
        np.testing.assert_allclose(box, (50, 180 + 10, 150, 180 + 30))
        self.assertEqual(confidence, .8)

    def test_tiling_runs_inside_the_band(self):
        yolo = RecordingYolo([])
        detector = TwoStageDetector(FixedBandProposer((.25, .75)), YoloDetector(yolo, tile_grid=(1, 2)))
        detector.candidates(np.zeros((400, 800, 3), np.uint8), WORKING)
        self.assertEqual(len(yolo.seen), 2)
        self.assertTrue(all(shape[0] == 200 for shape in yolo.seen))


if __name__ == '__main__':
    unittest.main()
