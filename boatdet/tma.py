"""Bearings-only target motion analysis: own-ship navigation plus bearings, out
comes the other vessel's state [x, y, speed, course] in a local world frame.

A single bearing fixes a direction and nothing else, so range is never measured
here: it is inferred from how the bearing evolves while the own ship moves. When
both vessels run straight at constant speed that inference is not unique, and no
filter can repair it. This module therefore carries a bank of range hypotheses
instead of one confident guess, reports the mixture rather than its strongest
mode, and publishes an observability flag derived from the own ship's manoeuvre
against the bearing noise. An estimate whose `status` is not 'converged' is a
direction with a range interval, not a position.

Frame: local tangent plane, x east, y north, meters, angles radians
counter-clockwise from +x (`compass_from_course` converts for the bridge).
Bearings are absolute in that frame; `BearingObservation.from_relative` turns a
bow-relative bearing (what `boatdet.geometry` produces) into one.
"""
import csv
import json
import math
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from boatdet.config import TmaConfig

EARTH_RADIUS_M = 6378137.0
KNOT_MPS = 0.514444

TWO_PI = 2.0 * math.pi
# Abramowitz and Stegun 7.1.26: erf to ~1.5e-7, vectorized. The mixture CDF is
# evaluated on a grid once per update per track, so math.erf per point is too slow.
_ERF_P = 0.3275911
_ERF_COEFFS = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


PARALLAX_SAMPLES = 120   # sensor positions kept per window; a 25 fps feed needs no more


def _normal_quantile(level):
    """Two-sided normal quantile for a credible level; bisection on the CDF.

    Small and exact enough at this precision, and it keeps scipy out of the
    dependency list for one number.
    """
    low, high = 0.0, 10.0
    target = 0.5 * (1.0 + level)
    for _ in range(60):
        middle = 0.5 * (low + high)
        if 0.5 * (1.0 + _erf(middle / math.sqrt(2.0))) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def geometric_range_sigma(range_m, parallax_ratio, bearing_sigma_rad):
    """Smallest range uncertainty the observed parallax can support, at any range.

    Triangulation over a cross-LOS baseline `b` resolves range to roughly
    `r * sigma_beta / (b / r)`, and `b / r` is exactly the parallax ratio. No
    amount of filtering beats it: a filter reporting less has linearized its way
    into a confidence the geometry never provided, which is the classic failure of
    a Cartesian EKF on bearings. Zero parallax means no range information at all.
    """
    if parallax_ratio <= 0 or range_m <= 0:
        return math.inf
    return float(range_m) * float(bearing_sigma_rad) / float(parallax_ratio)


def _erf(x):
    """Element-wise error function of a numpy array."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + _ERF_P * np.abs(x))
    series = np.zeros_like(t)
    for coefficient in reversed(_ERF_COEFFS):
        series = (series + coefficient) * t
    return np.sign(x) * (1.0 - series * np.exp(-x * x))


def wrap_angle(angle):
    """Angle folded into [-pi, pi); the only safe way to difference bearings."""
    return (float(angle) + math.pi) % TWO_PI - math.pi


def compass_from_course(course_rad):
    """Math-convention course (CCW from east) as compass degrees (CW from north)."""
    return math.degrees(wrap_angle(math.pi / 2 - course_rad)) % 360.0


def course_from_compass(compass_deg):
    """Compass degrees (CW from north) as a math-convention course in radians."""
    return wrap_angle(math.pi / 2 - math.radians(compass_deg))


def local_from_geodetic(latitude_deg, longitude_deg, origin_deg):
    """Degrees to local east/north meters on the tangent plane through `origin_deg`.

    An equirectangular projection: sub-meter over the few kilometers a camera can
    see, and it keeps the frame the one every angle here is measured in. It is not
    a survey projection and must not be reused as one.
    """
    latitude_0, longitude_0 = origin_deg
    x = math.radians(longitude_deg - longitude_0) * EARTH_RADIUS_M * math.cos(math.radians(latitude_0))
    y = math.radians(latitude_deg - latitude_0) * EARTH_RADIUS_M
    return float(x), float(y)


def bearing_sigma(pixel_sigma_px, focal_px, heading_sigma_rad=0.0, mount_sigma_rad=0.0):
    """Standard deviation of one bearing, composed from its independent sources.

    Detector jitter enters through the lens (`pixel_sigma_px / focal_px`), the
    platform through heading and mounting uncertainty. Detection confidence is
    not an angular variance and must never be substituted for one.
    """
    if focal_px is None or focal_px <= 0:
        raise ValueError('focal_px must be positive')
    return math.sqrt((float(pixel_sigma_px) / float(focal_px)) ** 2
                     + float(heading_sigma_rad) ** 2 + float(mount_sigma_rad) ** 2)


@dataclass(frozen=True)
class OwnShipState:
    """Known state of the observing vessel at one instant, in the world frame."""
    timestamp: float
    x_m: float
    y_m: float
    heading_rad: float = 0.0        # bow direction, CCW from +x
    speed_mps: float = 0.0
    course_rad: float = None        # direction of travel; defaults to the heading

    def sensor_position(self, lever_forward_m=0.0, lever_starboard_m=0.0):
        """Camera position in the world frame, mounting offset rotated by heading."""
        cos_h, sin_h = math.cos(self.heading_rad), math.sin(self.heading_rad)
        # Starboard is the bow direction turned clockwise.
        x = self.x_m + lever_forward_m * cos_h + lever_starboard_m * sin_h
        y = self.y_m + lever_forward_m * sin_h - lever_starboard_m * cos_h
        return float(x), float(y)


@dataclass(frozen=True)
class BearingObservation:
    """One absolute bearing with the sensor position it was taken from."""
    timestamp: float
    bearing_rad: float
    sigma_rad: float
    sensor_x_m: float
    sensor_y_m: float

    @classmethod
    def from_relative(cls, own_ship, relative_bearing_deg, sigma_rad, timestamp=None,
                      lever_forward_m=0.0, lever_starboard_m=0.0):
        """Bow-relative bearing (positive to starboard) as an absolute observation.

        Starboard is clockwise in this frame, so the relative angle is subtracted
        from the heading. Attitude belongs in the ray that produced the relative
        bearing, not here.
        """
        sensor_x, sensor_y = own_ship.sensor_position(lever_forward_m, lever_starboard_m)
        bearing = wrap_angle(own_ship.heading_rad - math.radians(float(relative_bearing_deg)))
        return cls(float(own_ship.timestamp if timestamp is None else timestamp),
                   bearing, float(sigma_rad), sensor_x, sensor_y)

    def __post_init__(self):
        if not (math.isfinite(self.bearing_rad) and self.sigma_rad > 0):
            raise ValueError('a bearing needs a finite angle and a positive sigma')


@dataclass(frozen=True)
class TargetEstimate:
    """Published state of the observed vessel, with the uncertainty that qualifies it.

    `status` is part of the answer: 'converged' means the geometry supported the
    range, 'ambiguous' means several ranges still explain every bearing equally
    well, and the interval, not the mean, is the result in that case.
    """
    timestamp: float
    x_m: float
    y_m: float
    speed_mps: float
    course_rad: float
    range_m: float
    bearing_rad: float
    range_sigma_m: float
    range_low_m: float
    range_high_m: float
    position_sigma_m: float
    speed_sigma_mps: float
    course_sigma_rad: float
    effective_hypotheses: float
    rejected: int
    reseeds: int
    parallax_ratio: float
    peak_parallax_ratio: float
    observable: bool
    status: str
    updates: int

    @property
    def state_vector(self):
        """[x, y, speed, course] in meters, m/s and radians."""
        return np.array([self.x_m, self.y_m, self.speed_mps, self.course_rad])

    @property
    def course_deg(self):
        return compass_from_course(self.course_rad)

    @property
    def bearing_deg(self):
        return compass_from_course(self.bearing_rad)

    def as_dict(self):
        return {'timestamp': round(self.timestamp, 3), 'x_m': round(self.x_m, 2),
                'y_m': round(self.y_m, 2), 'speed_mps': round(self.speed_mps, 3),
                'course_deg': round(self.course_deg, 2), 'range_m': round(self.range_m, 2),
                'bearing_deg': round(self.bearing_deg, 2),
                'range_sigma_m': round(self.range_sigma_m, 2),
                'range_low_m': round(self.range_low_m, 2), 'range_high_m': round(self.range_high_m, 2),
                'position_sigma_m': round(self.position_sigma_m, 2),
                'speed_sigma_mps': round(self.speed_sigma_mps, 3),
                'course_sigma_deg': round(math.degrees(self.course_sigma_rad), 2),
                'effective_hypotheses': round(self.effective_hypotheses, 2),
                'rejected': self.rejected, 'reseeds': self.reseeds,
                'parallax_ratio': round(self.parallax_ratio, 5),
                'peak_parallax_ratio': round(self.peak_parallax_ratio, 5),
                'observable': self.observable, 'status': self.status, 'updates': self.updates}


class BearingOnlyEKF:
    """One range hypothesis: extended Kalman filter over [x, y, vx, vy].

    Cartesian state with an angular measurement, so the filter is consistent only
    while the range uncertainty stays a modest fraction of the range; the bank
    above is what keeps that true by splitting the range prior into slices.
    """

    def __init__(self, config=TmaConfig()):
        self.config = config
        self.state = np.zeros(4)
        self.covariance = np.eye(4)
        self.timestamp = None
        self.log_weight = 0.0
        self.initialized = False

    @classmethod
    def seeded(cls, observation, range_m, range_sigma_m, config=TmaConfig()):
        """Filter placed on the bearing ray at `range_m`, uncertain along it and across it."""
        filt = cls(config)
        cos_b, sin_b = math.cos(observation.bearing_rad), math.sin(observation.bearing_rad)
        filt.state = np.array([observation.sensor_x_m + range_m * cos_b,
                               observation.sensor_y_m + range_m * sin_b, 0.0, 0.0])
        # Along the line of sight the slice width rules; across it the bearing noise does.
        rotation = np.array([[cos_b, -sin_b], [sin_b, cos_b]])
        position = rotation @ np.diag([range_sigma_m ** 2,
                                       (range_m * observation.sigma_rad) ** 2]) @ rotation.T
        # Velocity prior: zero mean, wide enough that the plausible speed band sits inside 2 sigma.
        velocity_var = (config.max_target_speed_mps / 2.0) ** 2
        filt.covariance = np.block([[position, np.zeros((2, 2))],
                                    [np.zeros((2, 2)), np.eye(2) * velocity_var]])
        filt.timestamp = observation.timestamp
        filt.initialized = True
        return filt

    @property
    def position(self):
        return self.state[:2].copy()

    @property
    def velocity(self):
        return self.state[2:].copy()

    def predict(self, timestamp):
        """Constant-velocity prediction with white-acceleration process noise."""
        if not self.initialized:
            return None
        dt = float(timestamp) - float(self.timestamp)
        if dt <= 0:
            return self.state.copy()
        dt = min(dt, self.config.max_dt_s)
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        psd = self.config.process_noise_psd
        block = np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]]) * psd
        noise = np.zeros((4, 4))
        for axis in (0, 1):
            index = np.array([axis, axis + 2])
            noise[np.ix_(index, index)] = block
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + noise
        self.timestamp = float(timestamp)
        return self.state.copy()

    def innovation(self, observation):
        """(residual, variance, jacobian) of one bearing against the predicted state.

        Separate from `correct` so the bank can weigh a bearing before any filter
        absorbs it: a sample every hypothesis rejects must change none of them.
        """
        self.predict(observation.timestamp)
        dx = self.state[0] - observation.sensor_x_m
        dy = self.state[1] - observation.sensor_y_m
        range_sq = dx * dx + dy * dy
        if range_sq < 1e-6:
            return None  # the sensor is on top of the estimate: no bearing information
        jacobian = np.array([[-dy / range_sq, dx / range_sq, 0.0, 0.0]])
        residual = wrap_angle(observation.bearing_rad - math.atan2(dy, dx))
        variance = (jacobian @ self.covariance @ jacobian.T).item() + observation.sigma_rad ** 2
        if not math.isfinite(variance) or variance <= 0:
            return None
        return residual, variance, jacobian

    def correct(self, observation, prepared=None):
        """Fuse one bearing; returns the log-likelihood this hypothesis assigns to it."""
        prepared = self.innovation(observation) if prepared is None else prepared
        if prepared is None:
            return -math.inf
        innovation, innovation_var, jacobian = prepared
        gain = (self.covariance @ jacobian.T) / innovation_var
        self.state = self.state + (gain * innovation).ravel()
        # Joseph form: the covariance stays symmetric positive definite over long runs.
        factor = np.eye(4) - gain @ jacobian
        self.covariance = factor @ self.covariance @ factor.T + \
            gain @ gain.T * observation.sigma_rad ** 2
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        return -0.5 * (innovation ** 2 / innovation_var + math.log(TWO_PI * innovation_var))

    def speed_log_prior(self):
        """Penalty for a hypothesis implying an implausible speed; a prior, not a veto."""
        speed = float(np.hypot(self.state[2], self.state[3]))
        excess = speed - self.config.max_target_speed_mps
        if excess <= 0:
            return 0.0
        return -0.5 * (excess / max(self.config.speed_prior_slack_mps, 1e-6)) ** 2

    def range_log_prior(self, observation):
        """Penalty for a hypothesis that has wandered outside the stated range window.

        The window is a declared prior — nothing outside it is reportable — so a
        filter that drifts out of it has stopped describing the scenario. Measured
        as a ratio, because the window spans two orders of magnitude.
        """
        factor = self.config.range_prior_factor
        if factor <= 1:
            return 0.0
        range_m = math.hypot(self.state[0] - observation.sensor_x_m,
                             self.state[1] - observation.sensor_y_m)
        if range_m <= 0:
            return -math.inf
        excess = max(math.log(self.config.min_range_m / range_m),
                     math.log(range_m / self.config.max_range_m), 0.0)
        return -0.5 * (excess / math.log(factor)) ** 2


class RangeHypothesisTma:
    """Bank of bearings-only EKFs over log-spaced range slices, weighted by evidence.

    The slices tile the configured range window, so the initial prior is the
    window itself rather than a guess inside it. Weights follow the measurement
    likelihoods; several surviving with comparable weight is the honest output of
    a geometry that does not determine range, not a failure to converge.
    """

    def __init__(self, config=TmaConfig(), track_id=None):
        self.config = config
        self.track_id = track_id
        self.filters = []
        self.updates = 0
        self.rejected = 0
        self.reseeds = 0
        self._consecutive_rejections = 0
        self.last_observation = None
        self.peak_parallax_ratio = 0.0   # best parallax the own ship has ever offered this track
        self._history = deque()

    @property
    def initialized(self):
        return bool(self.filters)

    def _reseed(self, observation):
        """Throw the bank away and rebuild it on this bearing.

        The parallax history goes too. It is own-ship geometry, but crediting the
        new bank with motion it collected no bearings during would let a restart
        inherit an observability it has not earned.
        """
        self.filters = []
        self.peak_parallax_ratio = 0.0
        self._consecutive_rejections = 0
        self._history.clear()
        self.reseeds += 1
        self._seed(observation)

    def _seed(self, observation):
        count = self.config.hypotheses
        ratio = (self.config.max_range_m / self.config.min_range_m) ** (1.0 / count)
        uniform = math.log(1.0 / count)
        for index in range(count):
            low = self.config.min_range_m * ratio ** index
            high = low * ratio
            # Mean and standard deviation of a uniform prior over the slice.
            filt = BearingOnlyEKF.seeded(observation, 0.5 * (low + high),
                                         (high - low) / math.sqrt(12.0), self.config)
            filt.log_weight = uniform
            self.filters.append(filt)

    def update(self, observation):
        """Fuse one bearing and return the estimate it produces.

        The first bearing seeds the bank rather than correcting it, so the estimate
        it returns is the range prior read through that one direction.
        """
        if self.last_observation is not None and \
                observation.timestamp < self.last_observation.timestamp:
            raise ValueError('bearings must arrive in non-decreasing time order')
        if not self.filters:
            self._seed(observation)
        else:
            # Exponential flattening of the accumulated evidence. Without it a bank run over
            # thousands of bearings collapses onto one range even where the geometry
            # determines none, because negligible per-bearing likelihood differences multiply.
            elapsed = observation.timestamp - self.last_observation.timestamp
            decay = 1.0 if self.config.weight_forget_s <= 0 else \
                math.exp(-max(elapsed, 0.0) / self.config.weight_forget_s)
            prepared = [filt.innovation(observation) for filt in self.filters]
            if self._gated(prepared):
                self.rejected += 1
                self._consecutive_rejections += 1
                if self._consecutive_rejections < self.config.gate_max_consecutive:
                    return self.estimate(observation.timestamp)
                # Bearing after bearing that nothing explains: the bank, not the sensor,
                # is wrong. Start again from the range prior and drop the evidence that
                # led here, including the parallax already earned — it belonged to a
                # state this bank no longer holds.
                self._reseed(observation)
                self.updates += 1
                self.last_observation = observation
                return self.estimate(observation.timestamp)
            self._consecutive_rejections = 0
            log_weights = []
            for filt, ready in zip(self.filters, prepared):
                log_weights.append(decay * filt.log_weight + filt.correct(observation, ready)
                                   + filt.speed_log_prior() + filt.range_log_prior(observation))
            self._normalize(log_weights)
            self._prune()
        self.updates += 1
        self.last_observation = observation
        # Subsampled: the manoeuvre shape is what matters, and a video-rate history
        # would make the straight-line fit below the most expensive step of the update.
        spacing = self.config.maneuver_window_s / PARALLAX_SAMPLES
        if not self._history or observation.timestamp - self._history[-1].timestamp >= spacing:
            self._history.append(observation)
        window = observation.timestamp - self.config.maneuver_window_s
        while len(self._history) > 2 and self._history[0].timestamp < window:
            self._history.popleft()
        estimate = self.estimate(observation.timestamp)
        if estimate is not None:
            self.peak_parallax_ratio = estimate.peak_parallax_ratio
        return estimate

    def _gated(self, prepared):
        """True when no hypothesis can explain this bearing, so none should absorb it.

        The filters have already been predicted to the observation time by
        `innovation`, so a rejected bearing still advances the bank in time; it
        simply contributes no evidence.
        """
        if self.config.gate_sigmas <= 0:
            return False
        distances = [residual ** 2 / variance for residual, variance, _ in
                     (ready for ready in prepared if ready is not None)]
        return bool(distances) and min(distances) > self.config.gate_sigmas ** 2

    def _normalize(self, log_weights):
        """Softmax over log weights; a bank that underflows entirely is reset to uniform."""
        finite = [w for w in log_weights if math.isfinite(w)]
        if not finite:
            for filt in self.filters:
                filt.log_weight = math.log(1.0 / len(self.filters))
            return
        peak = max(finite)
        total = math.log(sum(math.exp(w - peak) for w in log_weights if math.isfinite(w))) + peak
        for filt, log_weight in zip(self.filters, log_weights):
            filt.log_weight = log_weight - total

    def _prune(self):
        """Drop hypotheses the evidence has emptied, keeping the bank non-degenerate."""
        if len(self.filters) <= self.config.min_hypotheses:
            return
        threshold = math.log(self.config.min_weight)
        keep = [f for f in self.filters if f.log_weight > threshold]
        if len(keep) < self.config.min_hypotheses:
            keep = sorted(self.filters, key=lambda f: f.log_weight,
                          reverse=True)[:self.config.min_hypotheses]
        if len(keep) == len(self.filters):
            return
        self.filters = keep
        self._normalize([f.log_weight for f in self.filters])

    def weights(self):
        return np.array([math.exp(f.log_weight) for f in self.filters])

    def _mixture(self, timestamp):
        """Mean and covariance of the hypothesis mixture at `timestamp`."""
        for filt in self.filters:
            filt.predict(timestamp)
        weights = self.weights()
        weights = weights / weights.sum()
        states = np.array([f.state for f in self.filters])
        mean = weights @ states
        covariance = np.zeros((4, 4))
        for weight, filt in zip(weights, self.filters):
            spread = (filt.state - mean).reshape(4, 1)
            covariance += weight * (filt.covariance + spread @ spread.T)
        return mean, covariance, weights

    def _range_interval(self, sensor, weights, level=None):
        """Central credible interval of the range, read off the mixture along the ray.

        The mixture is multimodal while the range is undetermined, so a mean plus
        one sigma would describe a distribution that is not there.
        """
        level = self.config.credible_level if level is None else level
        tail = 0.5 * (1.0 - level)
        means, sigmas = [], []
        for filt in self.filters:
            offset = filt.position - np.asarray(sensor)
            distance = float(np.linalg.norm(offset))
            direction = offset / distance if distance > 1e-9 else np.array([1.0, 0.0])
            means.append(distance)
            sigmas.append(math.sqrt(max(float(direction @ filt.covariance[:2, :2] @ direction), 1e-9)))
        means, sigmas = np.array(means), np.array(sigmas)
        grid = np.linspace(max(0.0, means.min() - 4 * sigmas.max()), means.max() + 4 * sigmas.max(), 512)
        cdf = 0.5 * (1.0 + _erf((grid[:, None] - means) / (sigmas * math.sqrt(2.0)))) @ weights
        low = float(np.interp(tail, cdf, grid))
        high = float(np.interp(1.0 - tail, cdf, grid))
        return low, high

    def parallax(self, range_m):
        """Own-ship cross-LOS manoeuvre over the window, as a fraction of the range.

        Residuals against a straight-line fit of the sensor track measure exactly
        the part of the own motion a constant-velocity target cannot mimic. A
        straight own ship leaves none, which is why its range stays undetermined.
        """
        if len(self._history) < 3 or range_m <= 0:
            return 0.0
        times = np.array([o.timestamp for o in self._history])
        times = times - times.mean()
        positions = np.array([[o.sensor_x_m, o.sensor_y_m] for o in self._history])
        design = np.column_stack([np.ones_like(times), times])
        fit, *_ = np.linalg.lstsq(design, positions, rcond=None)
        residuals = positions - design @ fit
        mean_bearing = math.atan2(sum(math.sin(o.bearing_rad) for o in self._history),
                                  sum(math.cos(o.bearing_rad) for o in self._history))
        across = np.array([-math.sin(mean_bearing), math.cos(mean_bearing)])
        projected = residuals @ across
        return float(projected.max() - projected.min()) / float(range_m)

    def estimate(self, timestamp=None):
        """Mixture state, its uncertainty and the status that qualifies it."""
        if not self.filters:
            return None
        timestamp = self.last_observation.timestamp if timestamp is None else float(timestamp)
        mean, covariance, weights = self._mixture(timestamp)
        sensor = (self.last_observation.sensor_x_m, self.last_observation.sensor_y_m)
        offset = mean[:2] - np.asarray(sensor)
        range_m = float(np.linalg.norm(offset))
        direction = offset / range_m if range_m > 1e-9 else np.array([1.0, 0.0])
        range_sigma = math.sqrt(max(float(direction @ covariance[:2, :2] @ direction), 0.0))
        ratio = self.parallax(range_m)
        peak = max(ratio, self.peak_parallax_ratio)
        low, high = self._range_interval(sensor, weights)
        # The geometry sets a floor under the range uncertainty; where the filters
        # claim better, the interval is widened to what the parallax actually supports
        # and, with no parallax at all, to the declared range window.
        floor = geometric_range_sigma(range_m, peak, self.last_observation.sigma_rad)
        window = self.config.max_range_m - self.config.min_range_m
        range_sigma = max(range_sigma, min(floor, window))
        if floor >= window:
            # The parallax supports nothing finer than the prior itself, so that is
            # the honest interval; a numerically tiny parallax is no parallax.
            low, high = min(low, self.config.min_range_m), max(high, self.config.max_range_m)
        else:
            half = _normal_quantile(self.config.credible_level) * floor
            low, high = min(low, range_m - half), max(high, range_m + half)
        low = max(low, 0.0)
        speed = float(np.hypot(mean[2], mean[3]))
        velocity_cov = covariance[2:, 2:]
        if speed > 1e-6:
            unit = mean[2:] / speed
            across = np.array([-unit[1], unit[0]])
            speed_sigma = math.sqrt(max(float(unit @ velocity_cov @ unit), 0.0))
            course_sigma = math.sqrt(max(float(across @ velocity_cov @ across), 0.0)) / speed
        else:
            # Direction of travel is undefined at a standstill; report it as unknown, not as zero.
            speed_sigma = math.sqrt(max(float(np.trace(velocity_cov)) / 2.0, 0.0))
            course_sigma = math.pi
        effective = float(1.0 / np.square(weights).sum())
        # Parallax is evidence, not a live signal: a manoeuvre flown an hour ago still
        # determined the range, so the peak latches. Information that has since gone
        # stale shows up as a growing range sigma, which the status test below reads.
        observable = peak > self.config.parallax_sigmas * self.last_observation.sigma_rad
        return TargetEstimate(
            timestamp=timestamp, x_m=float(mean[0]), y_m=float(mean[1]), speed_mps=speed,
            course_rad=float(math.atan2(mean[3], mean[2])), range_m=range_m,
            bearing_rad=float(math.atan2(offset[1], offset[0])), range_sigma_m=range_sigma,
            range_low_m=low, range_high_m=high,
            position_sigma_m=math.sqrt(max(float(np.linalg.eigvalsh(covariance[:2, :2]).max()), 0.0)),
            speed_sigma_mps=speed_sigma, course_sigma_rad=min(course_sigma, math.pi),
            effective_hypotheses=effective, rejected=self.rejected, reseeds=self.reseeds,
            parallax_ratio=ratio, peak_parallax_ratio=peak,
            observable=observable,
            status=self._status(range_m, range_sigma, observable),
            updates=self.updates)

    def _status(self, range_m, range_sigma, observable):
        if self.updates < self.config.min_updates:
            return 'initializing'
        if not observable:
            return 'ambiguous'
        if range_m <= 0 or range_sigma / range_m > self.config.converged_range_frac:
            return 'ambiguous'
        return 'converged'


@dataclass
class MultiTargetTma:
    """One bearings-only estimator per track id, fed by the tracker's bearings."""
    config: TmaConfig = field(default_factory=TmaConfig)
    estimators: dict = field(default_factory=dict)

    def observe(self, track_id, observation):
        estimator = self.estimators.get(track_id)
        if estimator is None:
            estimator = self.estimators[track_id] = RangeHypothesisTma(self.config, track_id)
        return estimator.update(observation)

    def estimates(self, timestamp=None):
        """Current estimate per track id, skipping estimators that never saw a bearing."""
        results = {}
        for track_id, estimator in self.estimators.items():
            estimate = estimator.estimate(timestamp)
            if estimate is not None:
                results[track_id] = estimate
        return results

    def drop(self, track_id):
        self.estimators.pop(track_id, None)


class OwnShipTrack:
    """Recorded own-ship navigation, interpolated to a frame timestamp.

    Position is interpolated linearly and heading along the shorter arc. When the
    nearest sample is further away than `max_age_s` nothing is returned, and
    outside the recorded interval the end sample is held rather than extrapolated:
    an invented own position is indistinguishable from target motion downstream.
    """

    @classmethod
    def load(cls, path, max_age_s=1.0):
        """Read recorded navigation from JSON or CSV.

        Each record needs `timestamp` on the video clock, a position as either
        `x_m`/`y_m` or `latitude`/`longitude`, and a heading (`heading_deg`,
        `heading_rad`, or `heading`/`yaw_deg` in degrees). Speed is optional
        (`speed_mps` or `speed_kn`) and is carried through for reporting only:
        the estimator reads position and heading. Geodetic records are projected
        onto the tangent plane through the first sample.
        """
        path = Path(path)
        if path.suffix.lower() == '.csv':
            with path.open(newline='') as handle:
                records = [dict(row) for row in csv.DictReader(handle)]
        else:
            records = json.loads(path.read_text())
            records = records.get('samples', records) if isinstance(records, dict) else records
        if not records:
            raise ValueError(f'no own-ship samples in {path}')
        origin = None
        samples = []
        for record in records:
            record = {key.lower(): value for key, value in record.items() if value not in (None, '')}
            if 'x_m' in record and 'y_m' in record:
                x, y = float(record['x_m']), float(record['y_m'])
            elif 'latitude' in record and 'longitude' in record:
                latitude, longitude = float(record['latitude']), float(record['longitude'])
                origin = origin or (latitude, longitude)
                x, y = local_from_geodetic(latitude, longitude, origin)
            else:
                raise ValueError(f'{path}: a sample needs x_m/y_m or latitude/longitude')
            if 'heading_rad' in record:
                heading = float(record['heading_rad'])
            elif any(key in record for key in ('heading_deg', 'heading', 'yaw_deg')):
                compass = float(next(record[key] for key in ('heading_deg', 'heading', 'yaw_deg')
                                     if key in record))
                heading = course_from_compass(compass)
            else:
                raise ValueError(f'{path}: a sample needs a heading; bearings cannot be placed without one')
            speed = float(record.get('speed_mps', 0.0)) if 'speed_mps' in record else \
                float(record.get('speed_kn', 0.0)) * KNOT_MPS
            samples.append(OwnShipState(timestamp=float(record['timestamp']), x_m=x, y_m=y,
                                        heading_rad=wrap_angle(heading), speed_mps=speed))
        return cls(samples, max_age_s)

    def __init__(self, samples, max_age_s=1.0):
        self.samples = sorted(samples, key=lambda s: s.timestamp)
        self.stamps = [s.timestamp for s in self.samples]
        self.max_age_s = float(max_age_s)
        if not self.samples:
            raise ValueError('own-ship track needs at least one sample')

    def at(self, timestamp):
        index = bisect_left(self.stamps, timestamp)
        nearest = min((i for i in (index - 1, index) if 0 <= i < len(self.samples)),
                      key=lambda i: abs(self.stamps[i] - timestamp))
        if abs(self.stamps[nearest] - timestamp) > self.max_age_s:
            return None
        if index == 0 or index >= len(self.samples):
            return self.samples[nearest]
        before, after = self.samples[index - 1], self.samples[index]
        span = after.timestamp - before.timestamp
        if span <= 0:
            return before
        ratio = (timestamp - before.timestamp) / span
        heading = before.heading_rad + ratio * wrap_angle(after.heading_rad - before.heading_rad)
        return OwnShipState(timestamp=float(timestamp),
                            x_m=before.x_m + ratio * (after.x_m - before.x_m),
                            y_m=before.y_m + ratio * (after.y_m - before.y_m),
                            heading_rad=wrap_angle(heading),
                            speed_mps=before.speed_mps + ratio * (after.speed_mps - before.speed_mps))
