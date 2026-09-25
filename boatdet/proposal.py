"""Stage one of two-stage detection: where in the frame a boat can be.

In the labeled recordings 99% of boat centers lie between 40% and 73% of the
frame height, and a median boat covers 0.4% of the frame. Most of each frame
is sky, roof or deck. A proposer returns the horizontal band worth searching;
TwoStageDetector runs the unchanged detector on that band only. Trackers see
the same candidates() interface as before.
"""
import math
from abc import ABC, abstractmethod

import cv2
import numpy as np

from boatdet.config import RoiConfig


class Proposer(ABC):
    """propose(frame) -> (top_row, bottom_row) in frame pixels, or None for the full frame."""

    @abstractmethod
    def propose(self, frame):
        """Rows of the band to search."""


def _rows(height, low, high):
    top, bottom = int(math.floor(low * height)), int(math.ceil(high * height))
    top, bottom = max(0, top), min(height, bottom)
    return (top, bottom) if bottom - top >= 2 else None


class FixedBandProposer(Proposer):
    """Constant band; right for a level camera, wrong as soon as the boat rolls."""

    def __init__(self, band=RoiConfig.band):
        self.band = band

    def propose(self, frame):
        return _rows(frame.shape[0], *self.band)


def fit_line_ransac(xs, ys, tolerance, iterations=100, seed=0):
    """(slope, intercept, inlier_fraction) of the dominant line y = a x + b, or None."""
    if len(xs) < 2:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        i, j = rng.choice(len(xs), 2, replace=False)
        if xs[i] == xs[j]:
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        intercept = ys[i] - slope * xs[i]
        inliers = np.abs(ys - (slope * xs + intercept)) <= tolerance
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 2:
        return None
    slope, intercept = np.polyfit(xs[best], ys[best], 1)
    return float(slope), float(intercept), float(best.mean())


class HorizonProposer(Proposer):
    """Band around the horizon found by vertical-gradient edges and a RANSAC line.

    No training and no labels, ~2 ms on a 320 px wide copy. The line follows
    pitch and roll, which a fixed band cannot. A fit that is weak or too tilted
    falls back to the fixed band rather than guessing.
    """

    def __init__(self, config=RoiConfig()):
        self.config = config
        self.fallback = FixedBandProposer(config.band)
        self.line = None           # (slope, intercept) in fractions of height per width fraction
        self.last_fit_ok = False

    def detect_horizon(self, frame):
        """(slope, intercept) of the horizon in normalized coords (x, y in [0, 1]), or None."""
        height, width = frame.shape[:2]
        scale = self.config.work_width / width
        small = cv2.resize(frame, (self.config.work_width, max(2, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
        gray = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        rows = gray.shape[0]
        top, bottom = int(self.config.search[0] * rows), int(self.config.search[1] * rows)
        region = gradient[top:bottom]
        if region.size == 0:
            return None
        columns = np.arange(0, region.shape[1], 2)
        peaks = region[:, columns].argmax(axis=0)
        strength = region[peaks, columns]
        # Weak columns (uniform water or wall) carry no horizon evidence.
        keep = strength >= max(0.5 * np.median(strength), self.config.min_edge)
        if keep.sum() < max(8, len(columns) // 5):
            return None
        xs = columns[keep].astype(float) / region.shape[1]
        ys = (peaks[keep] + top).astype(float) / rows
        fit = fit_line_ransac(xs, ys, tolerance=2.5 / rows)
        if fit is None:
            return None
        slope, intercept, inlier_fraction = fit
        tilt = math.degrees(math.atan(slope * height / width))
        if inlier_fraction < self.config.min_inlier_frac or abs(tilt) > self.config.max_tilt_deg:
            return None
        return slope, intercept

    def propose(self, frame):
        fit = self.detect_horizon(frame)
        self.last_fit_ok = fit is not None
        if fit is None:
            self.line = None
            return self.fallback.propose(frame)
        alpha = self.config.smoothing
        self.line = fit if self.line is None else tuple(alpha * n + (1 - alpha) * o
                                                        for n, o in zip(fit, self.line))
        slope, intercept = self.line
        ends = (intercept, slope + intercept)  # y at the left and right edge
        return _rows(frame.shape[0], max(0.0, min(ends) - self.config.margin_above),
                     min(1.0, max(ends) + self.config.margin_below))


class SegmentationProposer(Proposer):
    """Band from the latest water mask: from above the water's top edge to the frame bottom.

    Free once the segmentation branch runs anyway; falls back to the horizon
    until a first mask exists.
    """

    def __init__(self, runner, config=RoiConfig(), min_water_frac=0.2):
        self.runner = runner
        self.config = config
        self.min_water_frac = min_water_frac
        self.fallback = HorizonProposer(config)

    def propose(self, frame):
        result = getattr(self.runner, 'last', None)
        if result is None:
            return self.fallback.propose(frame)
        water_rows = np.flatnonzero(result.water_mask.mean(axis=1) >= self.min_water_frac)
        if not water_rows.size:
            return self.fallback.propose(frame)
        first = water_rows[0] / result.water_mask.shape[0]
        return _rows(frame.shape[0], max(0.0, first - self.config.margin_above), 1.0)


class TwoStageDetector:
    """Proposer + detector with the YoloDetector.candidates interface.

    The detector sees the band crop at full source resolution; boxes are mapped
    back to working-frame coordinates, so trackers, CSV and viewer are unchanged.
    Tiling configured on the inner detector applies inside the band.
    """

    def __init__(self, proposer, detector):
        self.proposer = proposer
        self.detector = detector
        self.last_band = None  # (top, bottom) rows in source pixels, for overlays and logs

    def candidates(self, frame, working_size):
        band = self.proposer.propose(frame)
        self.last_band = band
        if band is None:
            return self.detector.candidates(frame, working_size)
        top, bottom = band
        height = frame.shape[0]
        crop = frame[top:bottom]
        scale_y = working_size[1] / height
        crop_size = (working_size[0], (bottom - top) * scale_y)
        offset = top * scale_y
        return [((x1, y1 + offset, x2, y2 + offset), confidence)
                for (x1, y1, x2, y2), confidence in self.detector.candidates(crop, crop_size)]


def build_proposer(config, segmentation_runner=None):
    """Proposer for a RoiConfig, or None for full-frame detection."""
    if config.mode == 'none':
        return None
    if config.mode == 'band':
        return FixedBandProposer(config.band)
    if config.mode == 'horizon':
        return HorizonProposer(config)
    if segmentation_runner is None:
        raise ValueError('ROI mode "segmentation" needs the segmentation branch (--segmentation-weights)')
    return SegmentationProposer(segmentation_runner, config)
