import argparse
import csv
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def colorize_depth(depth_m):
    # scale to whatever range is actually present in this frame (2nd-98th percentile, robust to outliers)
    # instead of a fixed max-depth, so the map stays visible/informative regardless of absolute scale
    lo, hi = np.percentile(depth_m, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    depth = np.clip((depth_m - lo) / (hi - lo), 0, 1) * 255.0
    # invert so nearby objects (small depth) render hot/bright, matching intuitive "closer = brighter"
    depth = 255.0 - depth
    return cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_INFERNO)


def distance_in_box(depth_m, box):
    h, w = depth_m.shape
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    values = depth_m[y1:y2, x1:x2]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.median(values)) if values.size else None


def apply_clahe(frame_bgr, clip_limit=2.5, tile_grid=8):
    """Experimental local contrast enhancement on the L channel.

    This cannot recover clipped highlights and may amplify glare or change the
    detector's input distribution. Keep disabled unless validated on held-out video.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def tiled_detect(yolo_model, native_frame, grid, overlap_frac, conf, imgsz, augment, scale_to_wh,
                 return_candidates=False):
    """Split the full-resolution frame into a grid of overlapping tiles, run YOLO on each
    tile independently (so a small/far boat occupies far more of the tile's pixels than it
    would in the full downscaled frame), then map the best hit back to the pipeline's
    working (WIDTH, HEIGHT) coordinate space. Returns (box_xyxy_in_working_coords, conf) or
    (None, None) if nothing was found in any tile.

    Measured on renkliTekneTekne.mp4 (a clip the model generalizes poorly on): baseline
    median YOLO confidence 0.294, strong(>=0.45)-detection rate 21%. Tiling alone: 0.243
    median, 17% strong (more coverage, lower confidence per hit). TTA alone: 0.271 median,
    23% strong. Combined (this function, with augment=True): 0.354 median, 32% strong -
    the two compound rather than cancel out, likely because tiling gives the model a
    larger, clearer view of the boat and TTA then extracts more signal from that view.
    """
    nh, nw = native_frame.shape[:2]
    rows, cols = grid
    if not (1 <= rows <= nh and 1 <= cols <= nw and 0 <= overlap_frac < 1):
        raise ValueError('tile grid must fit the image and overlap must be in [0, 1)')
    tile_h = int(np.ceil(nh / (rows - (rows - 1) * overlap_frac)))
    tile_w = int(np.ceil(nw / (cols - (cols - 1) * overlap_frac)))

    best_conf = -1.0
    best_box = None
    candidates = []
    out_w, out_h = scale_to_wh
    for r in range(rows):
        for c in range(cols):
            y0 = round(r * (nh - tile_h) / (rows - 1)) if rows > 1 else 0
            x0 = round(c * (nw - tile_w) / (cols - 1)) if cols > 1 else 0
            tile = native_frame[y0:y0 + tile_h, x0:x0 + tile_w]
            res = yolo_model.predict(tile, conf=conf, imgsz=imgsz, verbose=False, augment=augment)[0]
            if len(res.boxes) == 0:
                continue
            for best in res.boxes:
                c_val = float(best.conf[0])
                tx1, ty1, tx2, ty2 = best.xyxy[0].tolist()
                # tile-local -> native -> working (WIDTH,HEIGHT) coordinates
                nx1, ny1, nx2, ny2 = tx1 + x0, ty1 + y0, tx2 + x0, ty2 + y0
                sx, sy = out_w / nw, out_h / nh
                mapped_box = (nx1 * sx, ny1 * sy, nx2 * sx, ny2 * sy)
                candidates.append((mapped_box, c_val))
                if c_val > best_conf:
                    best_box, best_conf = mapped_box, c_val
    if return_candidates:
        return candidates
    return best_box, (best_conf if best_box is not None else None)


def select_detection(candidates, center, max_jump_px):
    """Gate all candidates before ranking, so a distant false hit cannot hide a boat."""
    valid = []
    for box, confidence in candidates:
        x1, y1, x2, y2 = box
        if not np.isfinite((*box, confidence)).all() or x2 - x1 < 2 or y2 - y1 < 2:
            continue
        if center is not None:
            jump = np.hypot((x1 + x2) / 2 - center[0], (y1 + y2) / 2 - center[1])
            if jump > max_jump_px:
                continue
        valid.append((box, confidence))
    return max(valid, key=lambda item: item[1], default=(None, 0.0))


class DepthWorker(threading.Thread):
    """Runs the (slow) depth model on whatever frame is most recent, in the background,
    so the camera preview loop never blocks on inference."""

    def __init__(self, processor, model, device, width, height, infer_size, use_fp16):
        super().__init__(daemon=True)
        self.processor = processor
        self.model = model
        self.device = device
        self.width = width
        self.height = height
        self.infer_size = infer_size
        self.use_fp16 = use_fp16

        self._lock = threading.Lock()
        self._latest_frame = None
        self._stop_event = threading.Event()

        self.depth_m = None
        self.depth_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.infer_ms = 0.0

    def submit_frame(self, frame_bgr):
        with self._lock:
            self._latest_frame = frame_bgr

    def stop(self):
        self._stop_event.set()

    def snapshot(self):
        # Drawing overlays must not mutate the image retained by the worker.
        with self._lock:
            return self.depth_m, self.depth_color.copy(), self.infer_ms

    def infer(self, frame):
        """Run the model on one frame and return (depth_m, depth_color, infer_ms). Usable directly
        (synchronous, exhaustive offline processing) or via the background thread loop below."""
        t0 = time.time()
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image_pil, return_tensors='pt',
            size={'height': self.infer_size, 'width': self.infer_size},
        ).to(self.device)
        if self.use_fp16:
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        post_processed = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(self.height, self.width)]
        )
        depth_m = post_processed[0]['predicted_depth'].cpu().numpy()
        depth_color = colorize_depth(depth_m)
        infer_ms = (time.time() - t0) * 1000
        return depth_m, depth_color, infer_ms

    def run(self):
        while not self._stop_event.is_set():
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            if frame is None:
                time.sleep(0.005)
                continue

            result = self.infer(frame)
            with self._lock:
                self.depth_m, self.depth_color, self.infer_ms = result


class YoloCsrtTracker:
    """Detection+tracking fusion: YOLO re-detects the target every frame (fast, ~25ms);
    whenever it finds one above conf_threshold, CSRT is re-initialized on that box (so we
    never drift). On frames where YOLO misses that bar, CSRT's own tracking carries us through.

    Raw boxes (from either source) are noisy frame-to-frame on wavy water, so on top of that:
    an implausible jump (center moving further than max_jump_frac of the frame's short side in
    one frame) is rejected outright rather than trusted, and accepted boxes are EMA-smoothed
    before being handed back - this is what actually gets used for depth sampling / drawing.

    Two more fallbacks kick in only when both of the above fail this frame, to raise detection
    coverage: a weak YOLO detection (conf_threshold > score >= low_conf_threshold) is accepted
    as a last resort, and failing even that, the last known smoothed box is coasted for up to
    grace_frames frames (assuming the target hasn't actually vanished, just was briefly missed).
    """

    def __init__(self, yolo_model, conf_threshold=0.45, low_conf_threshold=0.15, width=640, height=360,
                 box_smoothing=0.35, max_jump_frac=0.40, yolo_imgsz=960, grace_frames=4,
                 tile_grid=None, tile_overlap=0.2, augment=False, clahe=False, reacquire_frames=8):
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.grace_frames = grace_frames
        self.csrt = None
        self.width = width
        self.height = height
        self.max_jump_px = min(width, height) * max_jump_frac
        self.box_smoothing = box_smoothing
        self.smoothed_box = None  # (cx, cy, w, h) floats
        self.miss_count = 0
        self.unconfirmed_count = 0
        self.reacquire_frames = reacquire_frames
        self._reacquiring = False
        self.tile_grid = tile_grid  # e.g. (2, 2); measured to compound well with augment (see tiled_detect docstring)
        self.tile_overlap = tile_overlap
        self.augment = augment
        self.clahe = clahe

    def _detect_best(self, frame, native_frame):
        """Returns (box_xyxy, conf) for the single best detection this frame, using tiled
        detection on the native-resolution frame if configured, otherwise a plain pass over
        the working-resolution frame - either way, optionally through CLAHE and/or TTA."""
        if self.tile_grid is not None and native_frame is not None:
            source = apply_clahe(native_frame) if self.clahe else native_frame
            candidates = tiled_detect(self.yolo_model, source, self.tile_grid, self.tile_overlap,
                                      self.low_conf_threshold, self.yolo_imgsz, self.augment,
                                      (self.width, self.height), return_candidates=True)
        else:
            source = native_frame if native_frame is not None else frame
            source = apply_clahe(source) if self.clahe else source
            results = self.yolo_model(source, verbose=False, conf=self.low_conf_threshold,
                                   imgsz=self.yolo_imgsz, augment=self.augment)[0]
            scale = np.array([self.width / source.shape[1], self.height / source.shape[0]] * 2)
            candidates = [((b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0]) * scale,
                           float(b.conf[0])) for b in results.boxes]
        center = self.smoothed_box[:2] if self.smoothed_box is not None else None
        box, conf = select_detection(candidates, center, self.max_jump_px)
        self._reacquiring = False
        if box is None and self.unconfirmed_count >= self.reacquire_frames:
            box, conf = select_detection(
                [(b, c) for b, c in candidates if c >= self.conf_threshold], None, self.max_jump_px)
            self._reacquiring = box is not None
        return box, conf

    def _accept(self, raw_box):
        x, y, w, h = raw_box
        cx, cy = x + w / 2, y + h / 2

        if self.smoothed_box is None:
            self.smoothed_box = (cx, cy, w, h)
            return True

        pcx, pcy, pw, ph = self.smoothed_box
        jump = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        if jump > self.max_jump_px:
            return False  # implausible teleport - most likely a false-positive/glare, reject

        a = self.box_smoothing
        self.smoothed_box = (
            a * cx + (1 - a) * pcx,
            a * cy + (1 - a) * pcy,
            a * w + (1 - a) * pw,
            a * h + (1 - a) * ph,
        )
        return True

    def _smoothed_xywh(self):
        cx, cy, w, h = self.smoothed_box
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    def _reinit_from_box(self, frame, box_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        w, h = x2 - x1, y2 - y1
        if w >= 2 and h >= 2 and self._accept((x1, y1, w, h)):
            self.csrt = cv2.TrackerCSRT_create()
            self.csrt.init(frame, (x1, y1, w, h))
            return True
        return False

    def update(self, frame, native_frame=None):
        best_box, best_conf = self._detect_best(frame, native_frame)
        if self._reacquiring:
            self.smoothed_box = None
            self.csrt = None

        # 1) a strong, trusted detection - always wins, re-anchors CSRT
        if best_box is not None and best_conf >= self.conf_threshold and self._reinit_from_box(frame, best_box):
            self.miss_count = 0
            self.unconfirmed_count = 0
            return True, self._smoothed_xywh()

        # 2) no strong detection this frame - let CSRT carry on from its last anchor
        if self.csrt is not None and self.unconfirmed_count < self.reacquire_frames:
            ok, box = self.csrt.update(frame)
            if ok and self._accept(box):
                self.miss_count = 0
                self.unconfirmed_count += 1
                return True, self._smoothed_xywh()

        # 3) CSRT also failed/unavailable - fall back to a weak YOLO detection as last resort
        if best_box is not None and best_conf >= self.low_conf_threshold and self._reinit_from_box(frame, best_box):
            self.miss_count = 0
            self.unconfirmed_count = 0
            return True, self._smoothed_xywh()

        # 4) nothing found at all - coast on the last known box for a few frames rather than
        # instantly reporting "no detection" (the target likely didn't just vanish)
        self.miss_count += 1
        self.unconfirmed_count += 1
        if self.smoothed_box is not None and self.miss_count <= self.grace_frames:
            return True, self._smoothed_xywh()

        return False, (0, 0, 0, 0)


class YoloKalmanTracker:
    """Detection+tracking fusion using a constant-velocity Kalman filter instead of CSRT.

    CSRT tracks by appearance similarity, which lets it drift onto a patch of glare that
    happens to look like the target. A Kalman filter instead predicts where the target
    physically should be next (from its recent velocity) and only accepts a YOLO detection
    as real if it falls close to that prediction - a detection that jumps to a plausible-
    looking but physically distant spot (the classic glare-jump failure mode) is rejected
    outright. This gate applies to every detection, strong or weak, before ranking
    candidates by confidence. A detection is only allowed to ignore the gate and
    reinitialize the track at a new location after `reacquire_frames` consecutive frames
    with no accepted detection - genuine re-acquisition after the target was actually lost,
    not fresh drift.
    """

    def __init__(self, yolo_model, conf_threshold=0.45, low_conf_threshold=0.15, width=640, height=360,
                 box_smoothing=0.35, max_jump_frac=0.40, yolo_imgsz=960, grace_frames=4,
                 reacquire_frames=8):
        self.yolo_model = yolo_model
        self.conf_threshold = conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.yolo_imgsz = yolo_imgsz
        self.grace_frames = grace_frames
        self.reacquire_frames = reacquire_frames
        self.width = width
        self.height = height
        self.max_jump_px = min(width, height) * max_jump_frac
        self.box_smoothing = box_smoothing

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 9.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        self.initialized = False
        self.smoothed_wh = None  # (w, h), EMA-smoothed box size
        self.miss_count = 0  # consecutive frames with no accepted detection (grace-coasting budget)
        self.unconfirmed_count = 0  # consecutive frames with no accepted detection (reacquire-gate override)

    def _accept_size(self, w, h):
        if self.smoothed_wh is None:
            self.smoothed_wh = (float(w), float(h))
            return
        a = self.box_smoothing
        pw, ph = self.smoothed_wh
        self.smoothed_wh = (a * w + (1 - a) * pw, a * h + (1 - a) * ph)

    def _xywh(self, cx, cy):
        w, h = self.smoothed_wh
        # unbounded constant-velocity coasting can otherwise walk the box off-frame entirely
        # during a long detection gap, making it useless (no pixels left to sample depth from)
        cx = min(max(cx, w / 2), self.width - w / 2)
        cy = min(max(cy, h / 2), self.height - h / 2)
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    def update(self, frame, native_frame=None):
        pred = self.kf.predict()
        pred_cx, pred_cy = float(pred[0, 0]), float(pred[1, 0])

        source = native_frame if native_frame is not None else frame
        results = self.yolo_model(source, verbose=False, conf=self.low_conf_threshold, imgsz=self.yolo_imgsz)[0]
        scale = np.array([self.width / source.shape[1], self.height / source.shape[0]] * 2)
        candidates = [((b.xyxy[0].cpu().numpy() if hasattr(b.xyxy[0], 'cpu') else b.xyxy[0]) * scale,
                       float(b.conf[0])) for b in results.boxes]
        center = (pred_cx, pred_cy) if self.initialized else None
        box, best_conf = select_detection(candidates, center, self.max_jump_px)
        reacquiring = False
        if box is None and self.initialized and self.unconfirmed_count >= self.reacquire_frames:
            box, best_conf = select_detection(
                [(b, c) for b, c in candidates if c >= self.conf_threshold], None, self.max_jump_px)
            reacquiring = box is not None

        if box is not None and best_conf >= self.low_conf_threshold:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1

            accepted = False
            if not self.initialized:
                accepted = True  # nothing to gate against yet
            else:
                jump = ((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2) ** 0.5
                if jump <= self.max_jump_px:
                    accepted = True
                elif best_conf >= self.conf_threshold and self.unconfirmed_count >= self.reacquire_frames:
                    accepted = True  # sustained loss + a confident hit elsewhere - real reacquisition

            if accepted:
                if not self.initialized or reacquiring:
                    self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0
                    self.smoothed_wh = None
                    self.initialized = True
                else:
                    self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
                self._accept_size(w, h)
                self.miss_count = 0
                self.unconfirmed_count = 0
                out_cx, out_cy = float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0])
                return True, self._xywh(out_cx, out_cy)

        # no accepted detection this frame - coast on the Kalman prediction alone
        self.miss_count += 1
        self.unconfirmed_count += 1
        if self.initialized and self.miss_count <= self.grace_frames:
            return True, self._xywh(pred_cx, pred_cy)

        return False, (0, 0, 0, 0)


def select_target(frame, width, height):
    print('Draw a box around the object to track, then press ENTER/SPACE. Press ESC to track the center of the frame instead.')
    frame = cv2.resize(frame, (width, height))
    x, y, w, h = cv2.selectROI('select object to track', frame, showCrosshair=True)
    cv2.destroyWindow('select object to track')
    if w < 2 or h < 2:
        w, h = width // 5, height // 5
        x, y = (width - w) // 2, (height - h) // 2
    print(f'--target "{x},{y},{x+w},{y+h}"')
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (x, y, w, h))
    return tracker


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value  # file path


def parse_box(value, width, height):
    x1, y1, x2, y2 = (int(v) for v in value.split(','))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2-x1 >= 2 and y2-y1 >= 2):
        raise ValueError('target must be a nonempty box inside the working frame')
    return (x1, y1, x2, y2)


def default_box(width, height):
    w, h = width // 5, height // 5
    x, y = (width - w) // 2, (height - h) // 2
    return (x, y, x + w, y + h)


def main():
    parser = argparse.ArgumentParser(description='Metric distance to a tracked object using Depth Anything V2 (metric). Source can be a camera index or a video file path.')
    parser.add_argument('--camera', type=parse_source, default=0, help='camera index, or a path to a video file')
    parser.add_argument('--model', type=str, default='depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf', help='HF model id - Indoor-Base is the default per model_comparison.py\'s laser-calibration results (best MAE/RMSE among Indoor-Small/Base/Large and Outdoor-Small/Large); pass Indoor-Small for faster live inference or another variant to compare')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--infer-size', type=int, default=392, help='resolution fed to the model (independent of camera/display resolution); lower = faster but coarser depth')
    parser.add_argument('--fp16', action='store_true', help='use half-precision inference for extra speed at some accuracy cost (off by default; full quality)')
    parser.add_argument('--smoothing', type=float, default=0.3, help='exponential-smoothing factor for the displayed distance, 0-1 (lower = smoother/slower to react, 1 = no smoothing)')
    parser.add_argument('--calib-scale', type=float, default=1.0, help='linear calibration: corrected = calib_scale * raw + calib_offset (see calibrate.py). Default is identity. A historical Indoor-Base pool fit used scale=0.9796 and offset=-2.2695; do not apply it to another rig without independent validation.')
    parser.add_argument('--calib-offset', type=float, default=0.0, help='linear calibration offset, in meters (see --calib-scale)')
    parser.add_argument('--no-display', action='store_true', help='headless mode: no preview window, no interactive target selection (use --target or the frame center)')
    parser.add_argument('--target', type=str, default=None, help='x1,y1,x2,y2 box (in --width/--height pixel coords) to track instead of clicking one interactively; required in --no-display unless you want the frame center')
    parser.add_argument('--output', type=str, default=None, help='path to save the annotated raw+depth video (mp4)')
    parser.add_argument('--log', type=str, default=None, help='path to save a CSV log of (frame, distance_raw, distance_smoothed, inference_ms)')
    parser.add_argument('--max-frames', type=int, default=None, help='stop after this many frames (useful for quick tests on long videos)')
    parser.add_argument('--sync', action='store_true', help='process every single frame in order (blocking inference, no dropped frames) instead of the background-thread/best-effort mode; much slower but exhaustive - intended for --no-display offline analysis')
    parser.add_argument('--yolo-weights', type=str, default=None, help='path to a fine-tuned YOLO .pt file; when given, the target is auto-detected every frame (YOLO re-detect + CSRT fallback) instead of manual --target/click selection')
    parser.add_argument('--yolo-conf', type=float, default=0.45, help='confidence required to trust a YOLO detection outright and re-anchor CSRT on it')
    parser.add_argument('--yolo-low-conf', type=float, default=0.15, help='lower confidence accepted only as a last resort, when CSRT has also failed this frame')
    parser.add_argument('--yolo-imgsz', type=int, default=960, help='resolution YOLO runs inference at (independent of --width/--height); higher helps catch small/far targets, at some speed cost')
    parser.add_argument('--box-smoothing', type=float, default=0.35, help='EMA factor for the tracked box position/size (YOLO mode only), 0-1, lower = smoother/slower to react')
    parser.add_argument('--max-jump-frac', type=float, default=0.40, help='YOLO mode only: reject a box whose center jumps more than this fraction of the shorter frame side in one frame (filters glare/false-positive teleports)')
    parser.add_argument('--grace-frames', type=int, default=4, help='YOLO mode only: keep coasting on the last known box for this many consecutive frames before declaring the target lost')
    parser.add_argument('--tracker', type=str, default='csrt', choices=['csrt', 'kalman'], help='YOLO mode only: "csrt" (appearance-based, default) or "kalman" (constant-velocity motion filter that gates every detection - strong or weak - by physical plausibility, only allowing a hijack to a new location after --reacquire-frames of sustained loss)')
    parser.add_argument('--reacquire-frames', type=int, default=8, help='consecutive frames without a YOLO anchor before a confident detection can reinitialize the track elsewhere; also bounds CSRT-only propagation')
    parser.add_argument('--tile-grid', type=str, default=None, help='csrt tracker only: e.g. "2x2" - detect on overlapping tiles of the native-resolution frame instead of one shrunk frame (helps small/far objects; grid_size x more YOLO calls per frame). Measured on a poorly-generalizing clip: alone this raised detection coverage but lowered average confidence per hit - combine with --augment to recover confidence too (see tiled_detect docstring for the numbers)')
    parser.add_argument('--tile-overlap', type=float, default=0.2, help='fractional overlap between adjacent tiles, only with --tile-grid')
    parser.add_argument('--augment', action='store_true', help='csrt tracker only: YOLO test-time augmentation (multi-scale/flip ensembling) - no retraining, slower per detection call, raised both coverage and confidence in testing, especially combined with --tile-grid')
    parser.add_argument('--clahe', action='store_true', help='csrt tracker only: local contrast enhancement (CLAHE) before detection - measured to REDUCE confidence on its own in testing (likely over-amplifies glare); off by default, kept as an option for further experimentation only')
    args = parser.parse_args()

    if args.no_display and not (args.target or args.yolo_weights):
        parser.error('headless processing requires --target or --yolo-weights; a center box is not a detection')
    if args.target and args.yolo_weights:
        parser.error('choose either manual --target or automatic --yolo-weights')
    # File replay must pair depth and tracking from the same frame.
    args.sync = args.sync or isinstance(args.camera, str)
    if min(args.width, args.height, args.infer_size, args.yolo_imgsz) < 2:
        parser.error('image dimensions must be at least 2')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if not (0 < args.smoothing <= 1 and 0 < args.box_smoothing <= 1 and 0 < args.max_jump_frac <= 1):
        parser.error('smoothing and jump fractions must be in (0, 1]')
    if not (math.isfinite(args.calib_scale) and args.calib_scale > 0 and math.isfinite(args.calib_offset)):
        parser.error('calibration requires a positive finite scale and finite offset')
    if args.target:
        try:
            parse_box(args.target, args.width, args.height)
        except ValueError as exc:
            parser.error(str(exc))
    if args.tile_grid:
        try:
            rows, cols = map(int, args.tile_grid.lower().split('x'))
            if min(rows, cols) < 1 or not 0 <= args.tile_overlap < 1:
                raise ValueError
        except ValueError:
            parser.error('--tile-grid must be positive ROWSxCOLS and overlap must be in [0,1)')
    if args.grace_frames < 0 or args.reacquire_frames < 1:
        parser.error('--grace-frames must be nonnegative and --reacquire-frames must be positive')
    if not 0 <= args.yolo_low_conf <= args.yolo_conf <= 1:
        parser.error('require 0 <= --yolo-low-conf <= --yolo-conf <= 1')
    if args.tracker == 'kalman' and (args.tile_grid or args.augment or args.clahe):
        parser.error('--tile-grid, --augment and --clahe are supported only by --tracker csrt')

    device = get_device()
    use_fp16 = device != 'cpu' and args.fp16
    print(f'device: {device}, fp16: {use_fp16}')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Could not open source {args.camera!r}')
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # drop stale queued frames instead of falling behind (no-op for video files)

    worker = writer = log_file = None
    frame_idx = 0
    try:
        ret, first_native = cap.read()
        if not ret:
            raise RuntimeError('Source opened but decoded no frames')
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        source_fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 15.0
        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
        if use_fp16:
            model = model.half()
        font = cv2.FONT_HERSHEY_SIMPLEX

        worker = DepthWorker(processor, model, device, args.width, args.height, args.infer_size, use_fp16)

        if args.yolo_weights:
            from ultralytics import YOLO
            yolo_model = YOLO(args.yolo_weights)
            if args.tracker == 'kalman':
                tracker = YoloKalmanTracker(yolo_model, conf_threshold=args.yolo_conf, low_conf_threshold=args.yolo_low_conf,
                                             width=args.width, height=args.height, box_smoothing=args.box_smoothing,
                                             max_jump_frac=args.max_jump_frac, yolo_imgsz=args.yolo_imgsz,
                                             grace_frames=args.grace_frames, reacquire_frames=args.reacquire_frames)
            else:
                tile_grid = None
                if args.tile_grid:
                    rows, cols = (int(v) for v in args.tile_grid.lower().split('x'))
                    tile_grid = (rows, cols)
                tracker = YoloCsrtTracker(yolo_model, conf_threshold=args.yolo_conf, low_conf_threshold=args.yolo_low_conf,
                                           width=args.width, height=args.height, box_smoothing=args.box_smoothing,
                                           max_jump_frac=args.max_jump_frac, yolo_imgsz=args.yolo_imgsz,
                                           grace_frames=args.grace_frames, tile_grid=tile_grid,
                                           tile_overlap=args.tile_overlap, augment=args.augment, clahe=args.clahe,
                                           reacquire_frames=args.reacquire_frames)
        elif args.target:
            box = parse_box(args.target, args.width, args.height)
            first_frame = cv2.resize(first_native, (args.width, args.height))
            x1, y1, x2, y2 = box
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))
        elif args.no_display:
            box = default_box(args.width, args.height)
            first_frame = cv2.resize(first_native, (args.width, args.height))
            x1, y1, x2, y2 = box
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first_frame, (x1, y1, x2 - x1, y2 - y1))
        else:
            tracker = select_target(first_native, args.width, args.height)
        smoothed_distance = None

        writer = None
        if args.output:
            # avc1 (H.264), not mp4v (MPEG-4 Part 2): mp4v plays in VLC/ffmpeg but no
            # browser will decode it, which made every --output video unplayable in
            # control_panel.html even though the file itself was fine.
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(args.output, fourcc, source_fps, (args.width * 2, args.height))
            if not writer.isOpened():
                raise RuntimeError(f'Could not open H.264 output writer: {args.output}')

        if args.log:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, 'w', newline='') if args.log else None
        log_writer = csv.writer(log_file) if log_file else None
        if log_writer:
            log_writer.writerow(['frame', 'distance_raw_m', 'distance_smoothed_m', 'inference_ms', 'time_s', 'distance_model_m'])

        if not args.sync:
            worker.start()
        metadata = dict(vars(args), fps=source_fps, calibrated=(args.calib_scale != 1 or args.calib_offset != 0),
                        target_selection=('Automatic (YOLO + ' + args.tracker + ')' if args.yolo_weights else 'Manual CSRT'),
                        source_video=str(args.camera), frames=0, status='running')
        metadata_path = Path(args.output or args.log).with_suffix('.meta.json') if args.output or args.log else None
        if metadata_path:
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        while True:
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            if frame_idx == 0:
                native_frame = first_native
            else:
                ret, native_frame = cap.read()
                if not ret:
                    break
            raw_image = cv2.resize(native_frame, (args.width, args.height))

            if args.sync:
                depth_m, depth_color, infer_ms = worker.infer(raw_image)
            else:
                worker.submit_frame(raw_image.copy())
                depth_m, depth_color, infer_ms = worker.snapshot()

            if isinstance(tracker, (YoloCsrtTracker, YoloKalmanTracker)):
                tracked, (tx, ty, tw, th) = tracker.update(raw_image, native_frame=native_frame)
            else:
                tracked, (tx, ty, tw, th) = tracker.update(raw_image)

            distance_raw = distance_model = None
            if tracked:
                box = (int(tx), int(ty), int(tx + tw), int(ty + th))
                distance_raw = distance_in_box(depth_m, box) if depth_m is not None else None
                if distance_raw is not None:
                    distance_model = distance_raw
                    distance_raw = args.calib_scale * distance_model + args.calib_offset
                    if distance_raw <= 0 or not math.isfinite(distance_raw):
                        distance_raw = None
                if distance_raw is not None:
                    smoothed_distance = distance_raw if smoothed_distance is None else \
                        args.smoothing * distance_raw + (1 - args.smoothing) * smoothed_distance

                else:
                    smoothed_distance = None

                x1, y1, x2, y2 = box
                cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if smoothed_distance is not None:
                    label = (f'{smoothed_distance:.2f} m (raw {distance_raw:.2f})' if distance_raw is not None
                              else f'{smoothed_distance:.2f} m (raw: no depth here)')
                    cv2.putText(raw_image, label, (x1, max(20, y1 - 10)), font, 0.7, (0, 255, 0), 2)
                    print(f'frame {frame_idx}: {label}  (inference: {infer_ms:.0f} ms)', end='\r')
            else:
                smoothed_distance = None
                cv2.putText(raw_image, 'tracking lost - press r to reselect', (10, 30), font, 0.7, (0, 0, 255), 2)

            cv2.putText(raw_image, 'r: reselect target  q: quit', (10, args.height - 40), font, 0.5, (200, 200, 200), 1)
            cv2.putText(raw_image, f'inference: {infer_ms:.0f} ms', (10, args.height - 10), font, 0.6, (0, 255, 255), 2)

            combined = cv2.hconcat([raw_image, depth_color])

            if writer:
                writer.write(combined)
            if log_writer:
                sm = '' if smoothed_distance is None else f'{smoothed_distance:.4f}'
                rw = '' if distance_raw is None else f'{distance_raw:.4f}'
                log_writer.writerow([frame_idx, rw, sm, f'{infer_ms:.1f}', f'{frame_idx/source_fps:.6f}',
                                     '' if distance_model is None else f'{distance_model:.4f}'])

            frame_idx += 1
            if not args.no_display:
                cv2.imshow('Depth Anything V2 - distance to tracked object', combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('r'):
                    tracker = select_target(raw_image, args.width, args.height)
                    smoothed_distance = None

        if metadata_path:
            metadata.update(frames=frame_idx, status='complete')
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    finally:
        if worker is not None:
            worker.stop()
            if worker.is_alive():
                worker.join(timeout=2.0)
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
            for _ in range(4):  # macOS sometimes needs a few extra event-loop pumps to actually close the window
                cv2.waitKey(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\ninterrupted, shutting down')
