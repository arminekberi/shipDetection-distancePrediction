"""Constant-velocity filtering and time-to-collision.

Velocities come from the filter state, never from differencing two raw
measurements: a single noisy pixel or depth sample would otherwise dominate
the closing speed and the derived TTC.
"""
import math

import numpy as np


class ConstantVelocityFilter:
    """[position, velocity] Kalman filter over `dim` position components.

    Timestamps drive the transition matrix, so uneven frame intervals (raw rig
    captures) and dropped frames are handled without assuming a frame rate.
    """

    def __init__(self, dim, measurement_noise=9.0, process_noise=25.0, max_dt=1.0):
        if dim < 1:
            raise ValueError('dim must be positive')
        self.dim = dim
        self.measurement_noise = float(measurement_noise)
        self.process_noise = float(process_noise)
        self.max_dt = float(max_dt)
        self.state = np.zeros(2 * dim)
        self.covariance = np.eye(2 * dim)
        self.timestamp = None
        self.initialized = False

    @property
    def position(self):
        return self.state[:self.dim].copy()

    @property
    def velocity(self):
        return self.state[self.dim:].copy()

    def initialize(self, measurement, timestamp):
        """Seed position from a measurement and reset velocity to unknown."""
        self.state = np.concatenate([np.asarray(measurement, dtype=float).ravel(), np.zeros(self.dim)])
        self.covariance = np.diag([self.measurement_noise] * self.dim + [self.process_noise] * self.dim) * 10.0
        self.timestamp = timestamp
        self.initialized = True

    def predict(self, timestamp):
        """Advance the state to `timestamp` and return the predicted position."""
        if not self.initialized:
            return None
        dt = 0.0 if self.timestamp is None else float(timestamp - self.timestamp)
        if dt < 0 or dt > self.max_dt:
            # A long or backwards gap makes the old velocity meaningless.
            self.state[self.dim:] = 0.0
            dt = 0.0
        transition = np.eye(2 * self.dim)
        transition[:self.dim, self.dim:] = np.eye(self.dim) * dt
        gain = np.concatenate([np.eye(self.dim) * (dt ** 2 / 2), np.eye(self.dim) * dt])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + \
            gain @ gain.T * self.process_noise + np.eye(2 * self.dim) * 1e-6
        self.timestamp = timestamp
        return self.position

    def correct(self, measurement, timestamp=None, measurement_noise=None):
        """Fuse a position measurement; initializes the filter on first use."""
        measurement = np.asarray(measurement, dtype=float).ravel()
        if measurement.size != self.dim or not np.isfinite(measurement).all():
            raise ValueError(f'measurement must be {self.dim} finite components')
        if not self.initialized:
            self.initialize(measurement, timestamp if timestamp is not None else 0.0)
            return self.position
        if timestamp is not None:
            self.predict(timestamp)
        noise = self.measurement_noise if measurement_noise is None else float(measurement_noise)
        observation = np.zeros((self.dim, 2 * self.dim))
        observation[:, :self.dim] = np.eye(self.dim)
        innovation_cov = observation @ self.covariance @ observation.T + np.eye(self.dim) * noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation_cov)
        self.state = self.state + gain @ (measurement - observation @ self.state)
        self.covariance = (np.eye(2 * self.dim) - gain @ observation) @ self.covariance
        return self.position

    def shift(self, offset):
        """Translate the filtered position, e.g. to remove camera ego-motion."""
        offset = np.asarray(offset, dtype=float).ravel()
        if offset.size != self.dim or not np.isfinite(offset).all():
            raise ValueError(f'offset must be {self.dim} finite components')
        self.state[:self.dim] += offset


def closing_speed_sigma(position, covariance, dim):
    """Standard deviation of the radial closing speed from the filter covariance."""
    position = np.asarray(position, dtype=float)
    distance = float(np.linalg.norm(position))
    if distance <= 1e-6:
        return None
    direction = position / distance
    velocity_cov = np.asarray(covariance)[dim:, dim:]
    variance = float(direction @ velocity_cov @ direction)
    return math.sqrt(variance) if variance >= 0 and math.isfinite(variance) else None


def closing_speed(position, velocity):
    """Radial speed toward the camera in m/s; positive means approaching, None if degenerate."""
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    if position.size != velocity.size or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        return None
    distance = float(np.linalg.norm(position))
    if distance <= 1e-6:
        return None
    return float(-np.dot(position, velocity) / distance)


def time_to_collision(distance_m, closing_mps, min_closing_speed, max_value,
                      closing_sigma=None, min_sigma=0.0):
    """D / |v_radial| while the target closes, else None.

    None rather than infinity for receding, drifting or unmeasured targets: the
    UI and the logs must never carry inf, NaN or an implausible horizon. With a
    filter uncertainty, the closing speed must also stand out from it, so range
    noise on a stationary target does not read as an approach.
    """
    if distance_m is None or closing_mps is None:
        return None
    if not (math.isfinite(distance_m) and math.isfinite(closing_mps)) or distance_m <= 0:
        return None
    if closing_mps < max(min_closing_speed, 1e-6):
        return None
    if closing_sigma is not None and closing_mps < min_sigma * closing_sigma:
        return None
    ttc = distance_m / closing_mps
    return ttc if math.isfinite(ttc) and 0 < ttc <= max_value else None
