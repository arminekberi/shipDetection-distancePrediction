"""AprilTag identity and pose injected from the deployed tag pipeline.

The HydroRL proposer/decoder is a separate project (see docs/GLARE_REVIEW.md);
this module is the seam through which its detections reach tracking. Nothing
here decodes tags, and no source fabricates one: without a real tag stream the
null source reports nothing and the YOLO path is unaffected.
"""
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TagObservation:
    """One decoded tag in working-frame coordinates, with pose when the source has it."""
    tag_id: int
    center_px: tuple
    corners_px: tuple = None
    position_m: tuple = None   # (x_forward, y_right) in the vessel body frame
    distance_m: float = None
    bearing_deg: float = None
    confidence: float = 1.0

    @property
    def has_pose(self):
        """True only for a metric fix; a bearing-only tag is identity plus direction."""
        return self.position_m is not None or self.distance_m is not None

    @property
    def bbox_xyxy(self):
        """Axis-aligned box around the tag itself, not around the vessel carrying it."""
        if not self.corners_px:
            u, v = self.center_px
            return u - 1.0, v - 1.0, u + 1.0, v + 1.0
        xs = [c[0] for c in self.corners_px]
        ys = [c[1] for c in self.corners_px]
        return min(xs), min(ys), max(xs), max(ys)


class AprilTagSource(ABC):
    """Tag observations for one frame."""

    @abstractmethod
    def detect(self, frame, timestamp=None, frame_index=None):
        """List of TagObservation in working-frame coordinates."""

    def available(self):
        return True


class NullAprilTagSource(AprilTagSource):
    """No tag pipeline attached."""

    def detect(self, frame, timestamp=None, frame_index=None):
        return []

    def available(self):
        return False


def _scaled_center(corners, scale):
    """Tag center in working coordinates from its four native-resolution corners."""
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return sum(xs) / len(xs) * scale[0], sum(ys) / len(ys) * scale[1]


class ReplayAprilTagSource(AprilTagSource):
    """Per-frame tag detections exported by the tag pipeline as JSON.

    Records are [{"frame": int, "tag_id": int, "center_px": [u, v], ...}]; pixel
    coordinates must already be in working-frame coordinates.
    """

    def __init__(self, by_frame):
        self.by_frame = by_frame

    @classmethod
    def load(cls, path):
        records = json.loads(Path(path).read_text())
        records = records.get('detections', records) if isinstance(records, dict) else records
        by_frame = defaultdict(list)
        for record in records:
            center = record['center_px']
            by_frame[int(record['frame'])].append(TagObservation(
                tag_id=int(record['tag_id']),
                center_px=(float(center[0]), float(center[1])),
                corners_px=tuple(tuple(map(float, c)) for c in record['corners_px']) if record.get('corners_px') else None,
                position_m=tuple(map(float, record['position_m'])) if record.get('position_m') else None,
                distance_m=float(record['distance_m']) if record.get('distance_m') is not None else None,
                bearing_deg=float(record['bearing_deg']) if record.get('bearing_deg') is not None else None,
                confidence=float(record.get('confidence', 1.0))))
        return cls(dict(by_frame))

    @classmethod
    def from_rig_log(cls, path, working_size, frame_names=None, native_size=None):
        """Read the rig detections.jsonl, keyed by the raw frame file it names.

        Without `frame_names` (replaying the preview mp4) the 1-based `seq` of the
        raw frame is the frame index. `native_size` defaults to the manifest.json
        beside the log. The deployed pipeline reports tag identity and
        camera-frame bearing but no range, so distance stays None.
        """
        path = Path(path)
        if native_size is None:
            manifest = json.loads((path.parent / 'manifest.json').read_text())
            native_size = (int(manifest['width']), int(manifest['height']))
        index_of = {Path(name).name: index for index, name in enumerate(frame_names or ())}
        scale = (working_size[0] / native_size[0], working_size[1] / native_size[1])
        by_frame = defaultdict(list)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if frame_names is not None:
                index = index_of.get(Path(record.get('file', '')).name)
            else:
                index = int(record['seq']) - 1 if 'seq' in record else None
            if index is None:
                continue
            for marker in record.get('markers', ()):
                corners = marker.get('corners')
                if not corners:
                    continue
                bearing = marker.get('bearing') or {}
                by_frame[index].append(TagObservation(
                    tag_id=int(marker['tag_id']),
                    center_px=_scaled_center(corners, scale),
                    corners_px=tuple((c[0] * scale[0], c[1] * scale[1]) for c in corners),
                    bearing_deg=float(bearing['az_cam_deg']) if 'az_cam_deg' in bearing else None))
        return cls(dict(by_frame))

    def detect(self, frame, timestamp=None, frame_index=None):
        return list(self.by_frame.get(frame_index, ()))
