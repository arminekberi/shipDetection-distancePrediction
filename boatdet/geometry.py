"""Monocular range from a water-contact pixel and the water plane.

Nothing here is hardcoded: without calibration every estimate is None, which
keeps an uncalibrated rig honest instead of publishing invented meters.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boatdet.egomotion import CAMERA_TO_BODY, Orientation, rotation_matrix


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics, lens distortion and the camera pose above the water plane."""
    camera_matrix: np.ndarray
    camera_height_m: float
    dist_coeffs: np.ndarray = None
    camera_to_imu: np.ndarray = None      # 3x3 rotation, camera axes -> body axes (FRD)
    mount_pitch_deg: float = 0.0          # positive tilts the camera up
    mount_roll_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    image_size: tuple = None              # (width, height) the intrinsics were measured at

    @classmethod
    def load(cls, path):
        """Read a JSON calibration; keys follow the configuration names."""
        data = json.loads(Path(path).read_text())
        data = {key.lower(): value for key, value in data.items()}
        matrix = np.asarray(data['camera_matrix'], dtype=float).reshape(3, 3)
        height = float(data['camera_height_m'])
        if not np.isfinite(matrix).all() or matrix[2, 2] == 0 or height <= 0:
            raise ValueError(f'{path}: needs a finite CAMERA_MATRIX and a positive CAMERA_HEIGHT_M')
        transform = data.get('camera_to_imu_transform')
        return cls(camera_matrix=matrix / matrix[2, 2],
                   camera_height_m=height,
                   dist_coeffs=np.asarray(data['dist_coeffs'], dtype=float).ravel() if data.get('dist_coeffs') else None,
                   camera_to_imu=np.asarray(transform, dtype=float).reshape(3, 3) if transform else None,
                   mount_pitch_deg=float(data.get('mount_pitch_deg', 0.0)),
                   mount_roll_deg=float(data.get('mount_roll_deg', 0.0)),
                   mount_yaw_deg=float(data.get('mount_yaw_deg', 0.0)),
                   image_size=tuple(data['image_size']) if data.get('image_size') else None)

    def scaled(self, from_size, to_size):
        """Same camera, intrinsics rescaled to another image size."""
        sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        return CameraCalibration(matrix, self.camera_height_m, self.dist_coeffs, self.camera_to_imu,
                                 self.mount_pitch_deg, self.mount_roll_deg, self.mount_yaw_deg,
                                 tuple(to_size))

    def body_from_camera(self):
        """Camera axes to body axes, mounting rotation included."""
        mount = CAMERA_TO_BODY if self.camera_to_imu is None else self.camera_to_imu
        return rotation_matrix(Orientation(math.radians(self.mount_roll_deg),
                                           math.radians(self.mount_pitch_deg),
                                           math.radians(self.mount_yaw_deg))) @ mount


@dataclass(frozen=True)
class RangeEstimate:
    """Relative position on the water plane; x forward, y to starboard."""
    x_relative_m: float
    y_relative_m: float
    distance_m: float
    bearing_deg: float


def undistort_point(point, calibration):
    """Ideal pixel coordinates; unchanged when no distortion coefficients are configured."""
    if calibration.dist_coeffs is None or not np.any(calibration.dist_coeffs):
        return float(point[0]), float(point[1])
    import cv2  # only needed on the distorted path
    source = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float64)
    ideal = cv2.undistortPoints(source, calibration.camera_matrix, calibration.dist_coeffs,
                                P=calibration.camera_matrix)
    return float(ideal[0, 0, 0]), float(ideal[0, 0, 1])


def water_intersection(point_px, calibration, orientation=None):
    """Range to a water-contact pixel, or None above the horizon / without calibration.

    The ray through the pixel is rotated into the body frame (vessel roll and
    pitch included when an IMU sample is supplied) and intersected with the
    plane `camera_height_m` below the camera. Bearing stays relative to the bow,
    so platform yaw is deliberately not applied.
    """
    if calibration is None or point_px is None:
        return None
    u, v = undistort_point(point_px, calibration)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    ray = np.linalg.inv(calibration.camera_matrix) @ np.array([u, v, 1.0])
    rotation = calibration.body_from_camera()
    if orientation is not None and orientation.valid:
        rotation = rotation_matrix(Orientation(orientation.roll_rad, orientation.pitch_rad, 0.0)) @ rotation
    forward, starboard, down = rotation @ ray
    if not np.isfinite([forward, starboard, down]).all() or down <= 1e-6:
        return None  # at or above the horizon: the ray never meets the water
    scale = calibration.camera_height_m / down
    x, y = float(scale * forward), float(scale * starboard)
    distance = math.hypot(x, y)
    if not math.isfinite(distance) or distance <= 0:
        return None
    return RangeEstimate(x, y, distance, math.degrees(math.atan2(y, x)))


def bearing_from_ray(point_px, calibration, orientation=None):
    """Bow-relative bearing of any pixel, roll and pitch included.

    `bearing_from_pixel` reads a column against the principal point, which is
    exact only for a level camera. Here the whole ray is rotated into the body
    frame, so a rolling platform no longer turns vertical image position into
    bearing error — and unlike `water_intersection` this needs no camera height
    and works above the horizon, which is where distant vessels sit.
    """
    if calibration is None or point_px is None:
        return None
    u, v = undistort_point(point_px, calibration)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    ray = np.linalg.inv(calibration.camera_matrix) @ np.array([u, v, 1.0])
    rotation = calibration.body_from_camera()
    if orientation is not None and orientation.valid:
        rotation = rotation_matrix(Orientation(orientation.roll_rad, orientation.pitch_rad, 0.0)) @ rotation
    forward, starboard, _ = rotation @ ray
    if not np.isfinite([forward, starboard]).all() or forward <= 1e-9:
        return None  # abeam or behind the camera: not a bearing this lens can report
    return math.degrees(math.atan2(starboard, forward))


def bearing_from_pixel(u, calibration):
    """Horizontal bearing of an image column, or None without intrinsics."""
    if calibration is None:
        return None
    matrix = calibration.camera_matrix
    return math.degrees(math.atan2(float(u) - matrix[0, 2], matrix[0, 0]))


def position_from_range(distance_m, bearing_deg):
    """(x_forward, y_starboard) from a range and bearing, or None if either is missing."""
    if distance_m is None or bearing_deg is None or not math.isfinite(distance_m) or distance_m <= 0:
        return None
    angle = math.radians(bearing_deg)
    return distance_m * math.cos(angle), distance_m * math.sin(angle)
