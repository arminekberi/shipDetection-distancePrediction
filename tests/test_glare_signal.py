import unittest

import numpy as np

from boatdet.config import MotConfig
from boatdet.detection import Detection
from boatdet.glare import highlight_fraction
from boatdet.mot import MultiObjectTracker

WORKING = (640, 360)


class HighlightTests(unittest.TestCase):
    def test_yellow_reflection_counts_although_blue_stays_low(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        frame[400:440, 1000:1100] = (120, 250, 255)  # BGR yellow-white streak
        self.assertAlmostEqual(highlight_fraction(frame, (500, 200, 550, 220), WORKING), 1.0)
        self.assertEqual(highlight_fraction(frame, (0, 0, 50, 20), WORKING), 0.0)

    def test_degenerate_box_gives_none(self):
        self.assertIsNone(highlight_fraction(np.zeros((360, 640, 3), np.uint8), (10, 10, 10, 30), WORKING))

    def test_suspect_track_is_flagged_not_dropped(self):
        mot = MultiObjectTracker(WORKING, MotConfig(min_hits=1))
        for i in range(3):
            tracks = mot.update([Detection((100, 100, 140, 120), .8, highlight_fraction=.9),
                                 Detection((300, 100, 340, 120), .8, highlight_fraction=.01)], i * .1)
        flags = {t.track_id: t.glare_suspect for t in tracks}
        self.assertEqual(flags, {1: True, 2: False})


if __name__ == '__main__':
    unittest.main()
