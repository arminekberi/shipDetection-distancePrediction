"""Camera ego-motion interface.

No IMU stream exists in this repository, so the default provider returns
nothing and tracking runs unchanged. Feed a real orientation source (rig
telemetry, an autopilot bridge) through this interface; never synthesize one,
an invented attitude would be indistinguishable from target motion.

Angles are radians in the right-handed vessel body frame FRD (x forward,
y starboard, z down): roll starboard-down positive, pitch nose-up positive,
yaw to starboard positive. The camera frame is x right, y down, z forward.
"""
import json
import math
from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Camera axes (x right, y down, z forward) expressed in body axes (FRD); a proper
# rotation, so rotation signs keep their meaning in both frames.
CAMERA_TO_BODY = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Orientation:
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    timestamp: float = 0.0

    @property
    def valid(self):
        return all(math.isfinite(v) for v in (self.roll_rad, self.pitch_rad, self.yaw_rad))


def rotation_matrix(orientation):
    """World-from-body rotation, Z-Y-X (yaw, pitch, roll) aerospace convention."""
    cr, sr = math.cos(orientation.roll_rad), math.sin(orientation.roll_rad)
    cp, sp = math.cos(orientation.pitch_rad), math.sin(orientation.pitch_rad)
    cy, sy = math.cos(orientation.yaw_rad), math.sin(orientation.yaw_rad)
    yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return yaw @ pitch @ roll


class EgoMotionProvider(ABC):
    """Orientation of the camera platform at a capture timestamp."""

    @abstractmethod
    def get_orientation(self, timestamp):
        """Orientation at (or nearest to) `timestamp`, or None when unavailable."""

    def available(self):
        return True


class NullEgoMotion(EgoMotionProvider):
    """No attitude source; apparent image motion is used as-is."""

    def get_orientation(self, timestamp):
        return None

    def available(self):
        return False


class SampledEgoMotion(EgoMotionProvider):
    """Recorded attitude samples, nearest sample within `max_age_s`.

    Samples are [{"timestamp": s, "roll_rad": .., "pitch_rad": .., "yaw_rad": ..}],
    degrees accepted as roll_deg/pitch_deg/yaw_deg. Seconds share the time base
    of boatdet.video.frame_timer.
    """

    def __init__(self, samples, max_age_s=0.5):
        self.samples = sorted(samples, key=lambda o: o.timestamp)
        self.stamps = [o.timestamp for o in self.samples]
        self.max_age_s = float(max_age_s)

    @classmethod
    def load(cls, path, max_age_s=0.5):
        records = json.loads(Path(path).read_text())
        records = records.get('samples', records) if isinstance(records, dict) else records
        samples = []
        for record in records:
            angles = {key: float(record.get(key, record.get(key.replace('_rad', '_deg'), 0.0)))
                      for key in ('roll_rad', 'pitch_rad', 'yaw_rad')}
            for key in list(angles):
                if key not in record:
                    angles[key] = math.radians(angles[key])
            samples.append(Orientation(**angles, timestamp=float(record['timestamp'])))
        if not samples:
            raise ValueError(f'no orientation samples in {path}')
        return cls(samples, max_age_s)

    def get_orientation(self, timestamp):
        if not self.samples:
            return None
        index = bisect_left(self.stamps, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.samples)]
        nearest = min(candidates, key=lambda i: abs(self.stamps[i] - timestamp))
        sample = self.samples[nearest]
        return sample if abs(sample.timestamp - timestamp) <= self.max_age_s else None


def rotation_pixel_shift(previous, current, camera_matrix, point, camera_to_body=None):
    """Pixel displacement of a static world point caused by camera rotation.

    Subtract it from the apparent target motion so platform yaw/pitch/roll is not
    read as target velocity. Needs intrinsics; returns None without them.
    """
    if previous is None or current is None or camera_matrix is None:
        return None
    if not (previous.valid and current.valid):
        return None
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    mount = CAMERA_TO_BODY if camera_to_body is None else np.asarray(camera_to_body, dtype=float).reshape(3, 3)
    # Camera-from-world at both instants, via the fixed camera-to-body mounting.
    before = mount.T @ rotation_matrix(previous).T
    after = mount.T @ rotation_matrix(current).T
    ray = np.linalg.inv(camera_matrix) @ np.array([point[0], point[1], 1.0])
    rotated = camera_matrix @ (after @ before.T) @ ray
    if not np.isfinite(rotated).all() or abs(rotated[2]) < 1e-9:
        return None
    return float(rotated[0] / rotated[2] - point[0]), float(rotated[1] / rotated[2] - point[1])
