"""Depth Anything V2 metric depth inference, colorization and distance sampling."""
import threading
import time
from collections import namedtuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Best laser-calibration MAE/RMSE among the benchmarked variants (depth_benchmark.py).
DEFAULT_MODEL = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf'
DEFAULT_INFER_SIZE = 392

# One-hue sequential ramp (data-viz palette, blue 100 -> 700), near = light, far = dark.
# A single hue keeps the map readable as magnitude and leaves the warm end of the
# spectrum free for the track status colors drawn on top of it.
DEPTH_RAMP_HEX = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
                  '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
SPAN_PERCENTILES = (2, 98)
SPAN_SMOOTHING = 0.15   # EMA on the color range, so the map does not flicker between frames
DepthFrame = namedtuple('DepthFrame', 'depth_m color infer_ms span')


def _ramp_lut(hex_colors):
    """256x1x3 BGR lookup table interpolated through the ramp stops."""
    stops = np.array([[int(h[i:i + 2], 16) for i in (5, 3, 1)] for h in hex_colors], np.float32)
    positions = np.linspace(0, 255, len(stops))
    lut = np.stack([np.interp(np.arange(256), positions, stops[:, c]) for c in range(3)], axis=1)
    return lut.round().astype(np.uint8).reshape(256, 1, 3)


# Reversed: the lookup index is proximity, so 255 (nearest) takes the lightest stop.
DEPTH_LUT = _ramp_lut(DEPTH_RAMP_HEX[::-1])


def depth_span(depth_m, previous=None, smoothing=SPAN_SMOOTHING):
    """(near_m, far_m) of the color range, eased toward the previous frame's span.

    Percentiles clip the few extreme pixels a monocular model always produces;
    easing keeps the same object the same color from one frame to the next, which
    a per-frame rescale does not.
    """
    finite = depth_m[np.isfinite(depth_m)]
    if not finite.size:
        return previous or (0.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, SPAN_PERCENTILES))
    hi = max(hi, lo + 1e-6)
    if previous is None:
        return lo, hi
    return (previous[0] + smoothing * (lo - previous[0]),
            previous[1] + smoothing * (hi - previous[1]))


def colorize_depth(depth_m, span=None):
    """Sequential heatmap over `span` meters; near = light, far = dark."""
    lo, hi = span or depth_span(depth_m)
    hi = max(hi, lo + 1e-6)
    normalized = 255.0 - np.clip((np.nan_to_num(depth_m, nan=hi) - lo) / (hi - lo), 0, 1) * 255.0
    return cv2.applyColorMap(normalized.astype(np.uint8), DEPTH_LUT)


def distance_in_box(depth_m, box):
    """Median finite positive depth inside an xyxy box, or None."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    values = depth_m[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


def fit_calibration(raw, true):
    """Least-squares linear correction: true = scale * raw + offset."""
    scale, offset = np.polyfit(raw, true, 1)
    return float(scale), float(offset)


class DepthEstimator:
    """Synchronous per-frame depth at a fixed output size."""

    def __init__(self, processor, model, device, out_size, infer_size=DEFAULT_INFER_SIZE, fp16=False,
                 span=None):
        self.processor = processor
        self.model = model
        self.device = device
        self.out_size = out_size  # (width, height)
        self.infer_size = infer_size
        self.fp16 = fp16
        # Fixed (near_m, far_m) color range, or None to track the scene with an EMA.
        self.fixed_span = span
        self.span = span

    @classmethod
    def load(cls, model_id, device, out_size, infer_size=DEFAULT_INFER_SIZE, fp16=False, span=None):
        fp16 = fp16 and device != 'cpu'
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        return cls(processor, model.half() if fp16 else model, device, out_size, infer_size, fp16, span)

    def infer(self, frame):
        """DepthFrame(depth_m, color, infer_ms, span) for one BGR frame."""
        start = time.time()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=image, return_tensors='pt',
                                size={'height': self.infer_size, 'width': self.infer_size}).to(self.device)
        if self.fp16:
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        width, height = self.out_size
        depth_m = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(height, width)])[0]['predicted_depth'].cpu().numpy()
        self.span = self.fixed_span or depth_span(depth_m, self.span)
        return DepthFrame(depth_m, colorize_depth(depth_m, self.span),
                          (time.time() - start) * 1000, self.span)


class DepthWorker(threading.Thread):
    """Background depth on the latest submitted frame; live preview never blocks."""

    def __init__(self, estimator):
        super().__init__(daemon=True)
        self.estimator = estimator
        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop_event = threading.Event()
        width, height = estimator.out_size
        self._result = DepthFrame(None, np.zeros((height, width, 3), dtype=np.uint8), 0.0, None)

    def submit_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def snapshot(self):
        """Latest DepthFrame; color copy so overlays cannot mutate worker state."""
        with self._lock:
            return self._result._replace(color=self._result.color.copy())

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame, self._latest_frame = self._latest_frame, None
            if frame is None:
                time.sleep(0.005)
                continue
            result = self.estimator.infer(frame)
            with self._lock:
                self._result = result
