"""Maritime semantic segmentation: water, obstacle and sky.

The detector only knows the classes it was trained on. This branch runs beside
it so that debris, logs, pontoons, pier structures, the shoreline and unusual
vessels become obstacles even when YOLO reports nothing.

No segmentation weights ship with this repository. Without a configured model
the branch stays off and the pipeline behaves exactly as before; it never
invents a mask.
"""
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from boatdet.config import SEGMENTATION_CLASS_MAP, SegmentationConfig

WATER, OBSTACLE, SKY, UNLABELED = 0, 1, 2, 255
CATEGORY_IDS = {'water': WATER, 'obstacle': OBSTACLE, 'sky': SKY}


class SegmentationUnavailable(RuntimeError):
    """Raised when segmentation is requested but its weights cannot be used."""


@dataclass
class SegmentationResult:
    """Semantic labels of one frame in working-frame coordinates.

    `mask` holds WATER/OBSTACLE/SKY/UNLABELED. `extra_masks` keeps any further
    category the model provides (shore, pier, boat, buoy) without widening the
    minimum API the safety logic depends on.
    """
    mask: np.ndarray
    inference_time_ms: float
    frame_index: int = -1
    extra_masks: dict = field(default_factory=dict)
    reused: bool = False

    @property
    def size(self):
        return self.mask.shape[1], self.mask.shape[0]

    @property
    def water_mask(self):
        return self.mask == WATER

    @property
    def obstacle_mask(self):
        return self.mask == OBSTACLE

    @property
    def sky_mask(self):
        return self.mask == SKY

    def category_fraction(self, box, category_mask):
        """Share of a working-frame xyxy box covered by a boolean mask."""
        height, width = self.mask.shape
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(width, int(x1))), max(0, min(width, int(x2)))))
        y1, y2 = sorted((max(0, min(height, int(y1))), max(0, min(height, int(y2)))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return 0.0
        return float(category_mask[y1:y2, x1:x2].mean())


@dataclass
class SegmentedObstacle:
    """One connected obstacle region; identity is per frame, tracking assigns the real id."""
    temp_id: int
    bbox_xyxy: tuple
    centroid_px: tuple
    area_px: int
    water_contact_px: tuple = None
    confidence: float = None


class SegmentationModel(ABC):
    """predict(frame) -> SegmentationResult, in working-frame coordinates."""

    @abstractmethod
    def predict(self, frame):
        """Semantic labels for one BGR frame."""

    def close(self):
        """Release backend resources; backends that hold none need not override."""


class UltralyticsSegmentation(SegmentationModel):
    """Ultralytics segmentation checkpoint (.pt) or exported TensorRT engine (.engine).

    Ultralytics loads an exported engine through the same API, so a Jetson
    deployment only swaps the configured path; FP16 follows the configuration.
    """

    def __init__(self, model, out_size, config=SegmentationConfig(), class_map=None, device=None):
        self.model = model
        self.out_size = out_size  # (width, height)
        self.config = config
        self.class_map = dict(SEGMENTATION_CLASS_MAP if class_map is None else class_map)
        self.device = device
        self.names = getattr(model, 'names', {}) or {}

    @classmethod
    def load(cls, config, out_size, device=None, class_map=None):
        if not config.model_path:
            raise SegmentationUnavailable(
                'SEGMENTATION_MODEL_PATH is not set; segmentation stays disabled')
        path = Path(config.model_path)
        if not path.exists():
            raise SegmentationUnavailable(
                f'segmentation weights not found: {path}. No maritime segmentation model ships '
                f'with this repository; set SEGMENTATION_MODEL_PATH (--segmentation-weights) to a '
                f'water/obstacle/sky model, or leave segmentation disabled.')
        from ultralytics import YOLO
        model = YOLO(str(path))
        if not str(getattr(model, 'task', '')).startswith('segment'):
            raise SegmentationUnavailable(f'{path} is not a segmentation checkpoint (task={model.task!r})')
        return cls(model, out_size, config, class_map, device)

    def category_of(self, class_index):
        """Semantic category of a model class, by name; unmapped names keep their own."""
        name = str(self.names.get(class_index, class_index)).strip().lower()
        return self.class_map.get(name, name)

    def predict(self, frame):
        start = time.perf_counter()
        result = self.model.predict(frame, imgsz=self.config.input_size, conf=self.config.conf,
                                    half=self.config.fp16 and self.device not in (None, 'cpu'),
                                    device=self.device, retina_masks=True, verbose=False)[0]
        # Native masks have had letterbox padding removed by Ultralytics. Resizing
        # padded network masks directly would shift hull edges and the waterline.
        width, height = self.out_size
        mask = np.full((height, width), UNLABELED, dtype=np.uint8)
        extra = {}
        masks = getattr(result, 'masks', None)
        if masks is not None and masks.data is not None and len(masks.data):
            data = masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else \
                np.zeros(len(data), dtype=int)
            # Paint sky first, then water, so an obstacle on either always wins.
            order = {'sky': 0, 'water': 1}
            for layer, class_index in sorted(zip(data, classes), key=lambda item: order.get(
                    self.category_of(item[1]), 2)):
                category = self.category_of(class_index)
                resized = cv2.resize(layer.astype(np.uint8), (width, height),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
                if category in CATEGORY_IDS:
                    mask[resized] = CATEGORY_IDS[category]
                else:
                    extra[category] = np.logical_or(extra.get(category, False), resized)
                    mask[resized] = OBSTACLE if category in self.config.obstacle_categories else mask[resized]
        return SegmentationResult(mask, (time.perf_counter() - start) * 1000, extra_masks=extra)


class SegmentationRunner:
    """Runs a model every N frames and reuses the previous labels in between.

    Reuse is what keeps the branch affordable next to the detector; a reused
    result is flagged so downstream consumers and the log can tell it apart.
    """

    def __init__(self, model, config=SegmentationConfig()):
        self.model = model
        self.config = config
        self.last = None
        self.frames = 0
        self.inferences = 0

    def process(self, frame, frame_index=None):
        """SegmentationResult for this frame, or None while the branch is off."""
        if self.model is None:
            return None
        index = self.frames if frame_index is None else frame_index
        self.frames += 1
        if self.last is not None and index % self.config.every_n_frames:
            return SegmentationResult(self.last.mask, self.last.inference_time_ms, self.last.frame_index,
                                      self.last.extra_masks, reused=True)
        result = self.model.predict(frame)
        result.frame_index = index
        self.inferences += 1
        self.last = result
        return result


def water_contact_point(component, water_mask=None, search_rows=3):
    """Lowest stable pixel of a region, preferring a row that touches water.

    The bottom of a detector box drifts below the hull on reflections and wake;
    the boundary between the object and the water does not.
    """
    rows = np.flatnonzero(component.any(axis=1))
    if not rows.size:
        return None
    bottom = int(rows[-1])
    if water_mask is not None:
        height = component.shape[0]
        for row in range(bottom, max(bottom - 2 * search_rows, int(rows[0])) - 1, -1):
            below = slice(row + 1, min(height, row + 1 + search_rows))
            columns = np.flatnonzero(component[row])
            if columns.size and below.start < height and \
                    water_mask[below, columns.min():columns.max() + 1].any():
                bottom = row
                break
    columns = np.flatnonzero(component[bottom])
    return float(np.median(columns)), float(bottom)


def extract_obstacles(result, min_area, water_mask=None):
    """Connected obstacle regions above `min_area`, as SegmentedObstacle records."""
    if result is None:
        return []
    obstacle = result.obstacle_mask.astype(np.uint8)
    if not obstacle.any():
        return []
    # Opening removes speckle that would otherwise become one-pixel obstacles.
    obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(obstacle, connectivity=8)
    water = result.water_mask if water_mask is None else water_mask
    obstacles = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        component = labels == index
        contact = water_contact_point(component, water)
        obstacles.append(SegmentedObstacle(
            temp_id=len(obstacles) + 1,
            bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
            centroid_px=(float(centroids[index][0]), float(centroids[index][1])),
            area_px=int(area),
            water_contact_px=contact))
    return obstacles
