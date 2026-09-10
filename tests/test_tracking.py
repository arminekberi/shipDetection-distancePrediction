import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from run import DepthWorker, YoloCsrtTracker, YoloKalmanTracker, tiled_detect


def detection(box, confidence):
    return SimpleNamespace(xyxy=np.array([box], dtype=float), conf=[confidence])


class FakeYolo:
    def __init__(self):
        self.boxes = []

    def __call__(self, *args, **kwargs):
        return [SimpleNamespace(boxes=self.boxes)]

    predict = __call__


class FakeCsrt:
    def init(self, frame, box):
        self.box = box

    def update(self, frame):
        return True, self.box


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.yolo = FakeYolo()
        self.patch = patch('run.cv2.TrackerCSRT_create', FakeCsrt)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_distant_glare_does_not_hide_nearby_detection(self):
        for cls in (YoloCsrtTracker, YoloKalmanTracker):
            with self.subTest(tracker=cls.__name__):
                tracker = cls(self.yolo, max_jump_frac=.1, box_smoothing=1)
                self.yolo.boxes = [detection((40, 80, 80, 100), .8)]
                self.assertTrue(tracker.update(self.frame)[0])
                self.yolo.boxes = [detection((500, 200, 560, 240), .99),
                                   detection((45, 80, 85, 100), .7)]
                ok, box = tracker.update(self.frame)
                self.assertTrue(ok)
                self.assertGreaterEqual(box[0], 40)
                self.assertLess(box[0], 80)
                self.assertEqual(tracker.unconfirmed_count, 0)

    def test_tiled_csrt_considers_all_candidates(self):
        tracker = YoloCsrtTracker(self.yolo, max_jump_frac=.1, tile_grid=(1, 1))
        self.yolo.boxes = [detection((40, 80, 80, 100), .8)]
        tracker.update(self.frame, self.frame)
        self.yolo.boxes = [detection((500, 200, 560, 240), .99),
                           detection((45, 80, 85, 100), .7)]
        box, confidence = tracker._detect_best(self.frame, self.frame)
        self.assertEqual(tuple(box), (45, 80, 85, 100))
        self.assertEqual(confidence, .7)

    def test_native_detection_coordinates_are_scaled_to_working_frame(self):
        native = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.yolo.boxes = [detection((80, 160, 160, 200), .8)]
        for cls in (YoloCsrtTracker, YoloKalmanTracker):
            with self.subTest(tracker=cls.__name__):
                tracker = cls(self.yolo)
                self.assertEqual(tracker.update(self.frame, native), (True, (40, 80, 40, 20)))

    def test_csrt_success_cannot_propagate_forever_without_yolo(self):
        tracker = YoloCsrtTracker(self.yolo, reacquire_frames=2, grace_frames=1)
        self.yolo.boxes = [detection((40, 80, 80, 100), .8)]
        tracker.update(self.frame)
        self.yolo.boxes = []
        self.assertTrue(tracker.update(self.frame)[0])
        self.assertTrue(tracker.update(self.frame)[0])
        self.assertTrue(tracker.update(self.frame)[0])
        self.assertFalse(tracker.update(self.frame)[0])
        self.yolo.boxes = [detection((500, 200, 560, 240), .9)]
        self.assertEqual(tracker.update(self.frame), (True, (500, 200, 60, 40)))

    def test_kalman_reacquisition_resets_velocity_and_size(self):
        tracker = YoloKalmanTracker(self.yolo, max_jump_frac=.1, reacquire_frames=2)
        self.yolo.boxes = [detection((40, 80, 80, 100), .8)]
        tracker.update(self.frame)
        tracker.kf.statePost = np.array([[60], [90], [5], [2]], dtype=np.float32)
        self.yolo.boxes = [detection((500, 200, 560, 240), .9)]
        tracker.update(self.frame)
        tracker.update(self.frame)
        self.assertEqual(tracker.update(self.frame), (True, (500, 200, 60, 40)))
        np.testing.assert_array_equal(tracker.kf.statePost[:, 0], [530, 220, 0, 0])

    def test_weak_distant_detection_cannot_reacquire(self):
        for cls in (YoloCsrtTracker, YoloKalmanTracker):
            with self.subTest(tracker=cls.__name__):
                tracker = cls(self.yolo, max_jump_frac=.1, reacquire_frames=2, grace_frames=0)
                self.yolo.boxes = [detection((40, 80, 80, 100), .8)]
                tracker.update(self.frame)
                self.yolo.boxes = [detection((500, 200, 560, 240), .2)]
                for _ in range(4):
                    result = tracker.update(self.frame)
                self.assertFalse(result[0])

    def test_tiles_cover_bottom_and_right_edges(self):
        frame = np.zeros((101, 103, 3), dtype=np.uint8)
        frame[-1, -1] = 255
        seen = []

        def predict(tile, **kwargs):
            seen.append(tile.copy())
            return [SimpleNamespace(boxes=[])]

        self.yolo.predict = predict
        tiled_detect(self.yolo, frame, (3, 3), .2, .15, 960, False, (640, 360))
        self.assertEqual(len(seen), 9)
        np.testing.assert_array_equal(seen[-1][-1, -1], [255, 255, 255])

    def test_tiles_do_not_leave_gaps_at_zero_overlap(self):
        frame = np.zeros((101, 103, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(101)[:, None]
        frame[:, :, 1] = np.arange(103)[None, :]
        covered = np.zeros(frame.shape[:2], dtype=bool)

        def predict(tile, **kwargs):
            covered[tile[:, :, 0], tile[:, :, 1]] = True
            return [SimpleNamespace(boxes=[])]

        self.yolo.predict = predict
        tiled_detect(self.yolo, frame, (3, 3), 0, .15, 960, False, (640, 360))
        self.assertTrue(covered.all())

    def test_worker_can_be_joined_and_overlay_does_not_modify_snapshot(self):
        worker = DepthWorker(None, None, 'cpu', 4, 3, 4, False)
        _, color, _ = worker.snapshot()
        color[:] = 255
        self.assertFalse(worker.depth_color.any())
        worker.start()
        worker.stop()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
