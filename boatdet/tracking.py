"""Single-target trackers: update(frame, native_frame) -> (ok, (x, y, w, h)) in working coords."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

LOST = (False, (0, 0, 0, 0))


@dataclass(frozen=True)
class TrackerConfig:
    conf: float = 0.45           # trusted detection, re-anchors the track
    low_conf: float = 0.15       # weak detection, last resort
    box_smoothing: float = 0.35  # EMA factor, 1 = raw boxes
    max_jump_frac: float = 0.40  # max center jump per frame, fraction of short side
    grace_frames: int = 4        # frames coasted before reporting loss
    reacquire_frames: int = 8    # unanchored frames before a confident hit may relocate the track


def select_detection(candidates, center, max_jump_px):
    """Gate all candidates before ranking, so a distant false hit cannot hide a boat."""
    valid = []
    for box, confidence in candidates:
        x1, y1, x2, y2 = box
        if not np.isfinite((*box, confidence)).all() or x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if center is not None and np.hypot((x1 + x2) / 2 - center[0], (y1 + y2) / 2 - center[1]) > max_jump_px:
            continue
        valid.append((box, confidence))
    return max(valid, key=lambda item: item[1], default=(None, 0.0))


def ema(alpha, new, old):
    return tuple(alpha * n + (1 - alpha) * o for n, o in zip(new, old))


class Tracker(ABC):
    @abstractmethod
    def update(self, frame, native_frame=None):
        """(ok, (x, y, w, h)); native_frame, when given, feeds the detector."""


class ManualCsrtTracker(Tracker):
    """CSRT seeded once from a user box; no detector."""

    def __init__(self, frame, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(frame, (x1, y1, x2 - x1, y2 - y1))

    def update(self, frame, native_frame=None):
        ok, box = self.csrt.update(frame)
        return bool(ok), tuple(int(v) for v in box)


class DetectorTracker(Tracker):
    """Base for detector-driven trackers: candidate gating and loss counters."""

    def __init__(self, detector, working_size, config=TrackerConfig()):
        self.detector = detector
        self.config = config
        self.width, self.height = working_size
        self.max_jump_px = min(working_size) * config.max_jump_frac
        self.miss_count = 0         # frames without any box, bounds coasting
        self.unconfirmed_count = 0  # frames without a detector anchor, bounds reacquisition

    def _detect(self, frame, native_frame, center):
        """(box, conf, reacquiring): best gated candidate, else a confident one after sustained loss."""
        source = native_frame if native_frame is not None else frame
        found = [c for c in self.detector.candidates(source, (self.width, self.height))
                 if c[1] >= self.config.low_conf]
        box, conf = select_detection(found, center, self.max_jump_px)
        if box is None and center is not None and self.unconfirmed_count >= self.config.reacquire_frames:
            box, conf = select_detection([c for c in found if c[1] >= self.config.conf], None, self.max_jump_px)
            return box, conf, box is not None
        return box, conf, False

    def _anchored(self):
        self.miss_count = self.unconfirmed_count = 0

    def _missed(self):
        """Count a miss; True while coasting is still allowed."""
        self.miss_count += 1
        self.unconfirmed_count += 1
        return self.miss_count <= self.config.grace_frames


class DetectionTracker(DetectorTracker):
    """Best detection per frame, no temporal state; diagnostic baseline."""

    def update(self, frame, native_frame=None):
        box, _, _ = self._detect(frame, native_frame, None)
        if box is None:
            return LOST
        x1, y1, x2, y2 = (int(v) for v in box)
        return True, (x1, y1, x2 - x1, y2 - y1)


class CsrtTracker(DetectorTracker):
    """YOLO re-anchors CSRT on trusted detections; CSRT bridges misses.

    Order per frame: trusted detection, CSRT (at most reacquire_frames without
    an anchor), weak detection, coasting on the last box for grace_frames.
    Accepted boxes are jump-gated and EMA-smoothed.
    """

    def __init__(self, detector, working_size, config=TrackerConfig()):
        super().__init__(detector, working_size, config)
        self.csrt = None
        self.smoothed_box = None  # (cx, cy, w, h)

    def update(self, frame, native_frame=None):
        center = self.smoothed_box[:2] if self.smoothed_box is not None else None
        box, conf, reacquiring = self._detect(frame, native_frame, center)
        if reacquiring:
            self.smoothed_box = self.csrt = None

        if box is not None and conf >= self.config.conf and self._anchor(frame, box):
            self._anchored()
            return True, self._xywh()

        if self.csrt is not None and self.unconfirmed_count < self.config.reacquire_frames:
            ok, tracked = self.csrt.update(frame)
            if ok and self._accept(tracked):
                self.miss_count = 0
                self.unconfirmed_count += 1
                return True, self._xywh()

        if box is not None and self._anchor(frame, box):
            self._anchored()
            return True, self._xywh()

        if self._missed() and self.smoothed_box is not None:
            return True, self._xywh()
        return LOST

    def _accept(self, xywh):
        """Smooth a raw box in; reject implausible teleports (glare, false hits)."""
        x, y, w, h = xywh
        current = (x + w / 2, y + h / 2, w, h)
        if self.smoothed_box is None:
            self.smoothed_box = current
            return True
        if np.hypot(current[0] - self.smoothed_box[0], current[1] - self.smoothed_box[1]) > self.max_jump_px:
            return False
        self.smoothed_box = ema(self.config.box_smoothing, current, self.smoothed_box)
        return True

    def _anchor(self, frame, box_xyxy):
        x1, y1, x2, y2 = (int(v) for v in box_xyxy)
        xywh = (x1, y1, x2 - x1, y2 - y1)
        if xywh[2] < 2 or xywh[3] < 2 or not self._accept(xywh):
            return False
        self.csrt = cv2.TrackerCSRT_create()
        self.csrt.init(frame, xywh)
        return True

    def _xywh(self):
        cx, cy, w, h = self.smoothed_box
        return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


class KalmanTracker(DetectorTracker):
    """Constant-velocity Kalman gate: detections must land near the predicted position.

    Resists CSRT-style drift onto glare that resembles the target. Only after
    reacquire_frames without an accepted detection may a confident hit elsewhere
    reset position, velocity and size.
    """

    def __init__(self, detector, working_size, config=TrackerConfig()):
        super().__init__(detector, working_size, config)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
        self.initialized = False
        self.smoothed_wh = None

    def update(self, frame, native_frame=None):
        prediction = self.kf.predict()
        predicted = (float(prediction[0, 0]), float(prediction[1, 0]))
        box, _, reacquiring = self._detect(frame, native_frame, predicted if self.initialized else None)
        if box is None:
            if self._missed() and self.initialized:
                return True, self._xywh(*predicted)
            return LOST

        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if not self.initialized or reacquiring:
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
            self.smoothed_wh = None
            self.initialized = True
        else:
            self.kf.correct(np.array([[cx], [cy]], np.float32))
        size = (float(x2 - x1), float(y2 - y1))
        self.smoothed_wh = size if self.smoothed_wh is None else ema(self.config.box_smoothing, size, self.smoothed_wh)
        self._anchored()
        return True, self._xywh(float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

    def _xywh(self, cx, cy):
        w, h = self.smoothed_wh
        # Keep coasting boxes inside the frame, depth needs pixels.
        cx = min(max(cx, w / 2), self.width - w / 2)
        cy = min(max(cy, h / 2), self.height - h / 2)
        return int(cx - w / 2), int(cy - h / 2), int(w), int(h)
