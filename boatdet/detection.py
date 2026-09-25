"""YOLO boat candidates mapped to working-frame coordinates."""
from dataclasses import dataclass

import cv2
import numpy as np

BOAT_CLASS_ID = 0
BOAT_CLASS_NAME = 'boat'


@dataclass(eq=False)
class Detection:
    """One frame observation handed to the tracker, from YOLO or from segmentation.

    The fusion fields stay None while the segmentation branch is disabled, so a
    detector-only run carries exactly the information it always did.
    """
    bbox_xyxy: tuple
    confidence: float
    class_id: int = BOAT_CLASS_ID
    class_name: str = BOAT_CLASS_NAME
    source: str = 'yolo'
    water_contact_px: tuple = None
    water_contact_source: str = None
    sky_fraction: float = None
    obstacle_fraction: float = None
    segmentation_confirmed: bool = False
    highlight_fraction: float = None

    @property
    def center_px(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def bottom_center_px(self):
        x1, _, x2, y2 = self.bbox_xyxy
        return (x1 + x2) / 2, y2


def as_detections(candidates, class_id=BOAT_CLASS_ID, class_name=BOAT_CLASS_NAME, source='yolo'):
    """Wrap [(xyxy, confidence)] from YoloDetector.candidates as Detection records."""
    return [Detection(tuple(float(v) for v in box), float(confidence), class_id, class_name, source)
            for box, confidence in candidates
            if np.isfinite((*box, confidence)).all() and box[2] - box[0] >= 2 and box[3] - box[1] >= 2]


def apply_clahe(frame, clip_limit=2.5, tile_grid=8):
    """Experimental L-channel CLAHE; cannot recover clipped highlights, may amplify glare."""
    l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def tiles(frame, grid, overlap):
    """Yield (x0, y0, tile) over an overlapping rows x cols grid covering every pixel."""
    h, w = frame.shape[:2]
    rows, cols = grid
    if not (1 <= rows <= h and 1 <= cols <= w and 0 <= overlap < 1):
        raise ValueError('tile grid must fit the image and overlap must be in [0, 1)')
    tile_h = int(np.ceil(h / (rows - (rows - 1) * overlap)))
    tile_w = int(np.ceil(w / (cols - (cols - 1) * overlap)))
    for r in range(rows):
        for c in range(cols):
            y0 = round(r * (h - tile_h) / (rows - 1)) if rows > 1 else 0
            x0 = round(c * (w - tile_w) / (cols - 1)) if cols > 1 else 0
            yield x0, y0, frame[y0:y0 + tile_h, x0:x0 + tile_w]


class YoloDetector:
    """All YOLO candidates of one frame as [(xyxy_working, confidence)].

    Tiling + TTA measured on a poorly generalizing clip: median conf 0.294 -> 0.354,
    strong (>=0.45) rate 21% -> 32%. Either alone gained less. CLAHE lowered confidence.
    """

    def __init__(self, model, conf=0.15, imgsz=960, augment=False, clahe=False,
                 tile_grid=None, tile_overlap=0.2, nms_iou=0.7):
        self.model = model
        names = model.names
        names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
        self.class_ids = [int(key) for key, name in names.items()
                          if str(name).lower() in ('boat', 'ship', 'vessel')]
        if not self.class_ids:
            raise ValueError('checkpoint has no boat/ship/vessel class; check the detector weights')
        if not 0 <= conf <= 1 or not 0 <= nms_iou <= 1:
            raise ValueError('confidence and NMS IoU must be in [0, 1]')
        self.conf = conf
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.augment = augment
        self.clahe = clahe
        self.tile_grid = tile_grid
        self.tile_overlap = tile_overlap

    def candidates(self, frame, working_size):
        """Detect on the given (native) frame, rescale boxes to working_size."""
        source = apply_clahe(frame) if self.clahe else frame
        parts = tiles(source, self.tile_grid, self.tile_overlap) if self.tile_grid else [(0, 0, source)]
        sx, sy = working_size[0] / source.shape[1], working_size[1] / source.shape[0]
        found = []
        for x0, y0, tile in parts:
            result = self.model.predict(tile, conf=self.conf, imgsz=self.imgsz,
                                        classes=self.class_ids, iou=self.nms_iou,
                                        augment=self.augment, verbose=False)[0]
            for box in result.boxes:
                if int(box.cls[0]) not in self.class_ids:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                mapped = ((x1 + x0) * sx, (y1 + y0) * sy, (x2 + x0) * sx, (y2 + y0) * sy)
                found.append((mapped, float(box.conf[0])))
        if not self.tile_grid or len(found) < 2:
            return found
        # Per-tile NMS cannot remove duplicates produced by adjacent crops.
        # Merge in the common frame before any tracker can create extra IDs.
        from boatdet.mot import iou
        kept = []
        for candidate in sorted(found, key=lambda item: -item[1]):
            if all(iou(candidate[0], previous[0]) <= self.nms_iou for previous in kept):
                kept.append(candidate)
        return kept
