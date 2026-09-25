"""Digital twin of the YTU basin: two hulls, their sensors, and the wire they talk on.

Everything downstream of the rigs -- `rig_state_producer.py`, `rig_state_logger.py`,
`bearings_to_observations.py`, `rig_calibration.py`, `boatdet.tma` -- reads either
the raw MQTT capture or a tracker's bearings.csv. This module produces both, from a
world whose truth is known exactly, so each of those tools can be scored where the
real basin cannot score it: ship 2 has no working camera, nothing positions a boat,
and no reference measures heading.

What is modelled, and on what it rests:

  hull      3-DOF surge/sway/yaw with differential twin screws (`%MD1` left,
            `%MD2` right, "+ = forward", +/-100) and a first-order motor lag of
            1.2 s, the lag `run_pursuer.py` documents. Mass, drag and thrust are
            NOT identified from the rig: they are picked to give ~1 m/s flat out
            and ~0.35 m/s at StraightSail's BASE_SPEED 20. Replace them once a
            step response has been logged.
  pool      the rectangle spanned by the low wall anchors (A06-A09). The water
            edge itself has never been surveyed.
  UWB       the nine surveyed YTU anchors, four per burst at 10 Hz per tag, the
            set re-picked as the hull moves (measured behaviour, rule assumed:
            nearest by slant range with jitter). Gaussian range noise sized to the
            0.035 m RMSE of a healthy set. An optional NLOS bias on steep paths
            can stand in for the ceiling-only failure, but it is the open
            HYPOTHESIS for it and one the rigs contradict (see UwbParams), so it
            is off unless asked for. With it off, nothing here reproduces that
            failure: its mechanism is unknown.
            The firmware's own `uwb/position` is a free 3D solve of the same
            ranges, which lets z float exactly the way the rigs' does.
  IMU       yaw CW-positive in an arbitrary PLC reference, drifting at the rates
            measured per ship; ship 2's `timestamp_capture` optionally carries the
            12.35 h PLC-counter fault. `t_plc_raw` is common to both rigs.
  camera    cam0 intrinsics and distortion as calibrated on ship 1 (4608x2592,
            RMS 0.81 px), projected through OpenCV and read back through the same
            undistortion `track_target.py` uses, at the 4.31 fps a real capture
            measured. Detection is a probability on apparent box height, not a
            model of the detector: on real footage the trained weights miss this
            target entirely.

Frame: the anchor frame, x/y metres, z up; heading CCW from +x. Body axes are
forward and PORT (left) so the planar frame stays right-handed; wire formats that
want starboard get it converted at the edge.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from boatdet.uwb import RIG_TAGS, VENUES

ANCHORS = {key: tuple(value) for key, value in VENUES['ytu']['anchors'].items()}
WIRE_LABEL = {survey: wire for wire, survey in VENUES['ytu']['anchor_map'].items()}

# cam0 on ship 1, hydrolink ship_camera_processing/calib/cam0_intrinsics.yaml
# (21 views, RMS 0.81 px). Copied so the twin runs without a hydrolink checkout.
CAM0_MATRIX = ((1963.1865085404934, 0.0, 2303.5330298996842),
               (0.0, 1965.9828748101243, 1305.0763573981881),
               (0.0, 0.0, 1.0))
CAM0_DIST = (-0.075814306266531864, 0.14662091134944155, -0.00082295140041373541,
             0.00053356501313038735, -0.086341939146773405)
CAM0_SIZE = (4608, 2592)


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pool:
    """Water the hulls may occupy; a hull centre stops `margin_m` short of a wall."""
    x_min: float = 4.41
    x_max: float = 28.99
    y_min: float = 0.0
    y_max: float = 28.85
    margin_m: float = 0.5

    def contains(self, x, y, margin=None):
        m = self.margin_m if margin is None else margin
        return (self.x_min + m <= x <= self.x_max - m) and (self.y_min + m <= y <= self.y_max - m)


@dataclass(frozen=True)
class HullParams:
    """Unidentified: chosen for plausible speeds, see the module note."""
    mass_kg: float = 10.0
    yaw_inertia_kgm2: float = 0.8
    thrust_n: float = 3.0            # one screw at +100
    reverse_ratio: float = 0.6       # astern thrust is weaker than ahead
    motor_tau_s: float = 1.2
    screw_spacing_m: float = 0.30
    surge_linear: float = 2.0
    surge_quadratic: float = 4.0
    sway_linear: float = 15.0
    yaw_linear: float = 0.4
    yaw_quadratic: float = 0.3
    yaw_disturbance_nm: float = 0.02  # white torque noise: waves, cables, draughts
    length_m: float = 1.0
    beam_m: float = 0.4
    freeboard_m: float = 0.25


@dataclass
class HullState:
    x: float
    y: float
    psi: float                        # heading, CCW from +x
    u: float = 0.0                    # surge, m/s
    v: float = 0.0                    # sway to port, m/s
    r: float = 0.0                    # yaw rate CCW, rad/s
    level_l: float = 0.0              # applied thrust level, [-1, 1]
    level_r: float = 0.0
    cmd_l: float = 0.0                # commanded PWM, [-100, 100]
    cmd_r: float = 0.0
    wall_contacts: int = 0

    @property
    def velocity(self):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return self.u * c - self.v * s, self.u * s + self.v * c

    def to_world(self, forward, port):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return self.x + forward * c - port * s, self.y + forward * s + port * c


class Hull:
    def __init__(self, params, state, current=(0.0, 0.0)):
        self.p = params
        self.s = state
        self.current = tuple(current)

    def _thrust(self, level):
        return self.p.thrust_n * level * (1.0 if level >= 0 else self.p.reverse_ratio)

    def step(self, dt, cmd_l, cmd_r, rng, pool):
        p, s = self.p, self.s
        s.cmd_l = float(np.clip(cmd_l, -100, 100))
        s.cmd_r = float(np.clip(cmd_r, -100, 100))
        blend = 1.0 - math.exp(-dt / p.motor_tau_s)
        s.level_l += (s.cmd_l / 100.0 - s.level_l) * blend
        s.level_r += (s.cmd_r / 100.0 - s.level_r) * blend
        left, right = self._thrust(s.level_l), self._thrust(s.level_r)
        surge = left + right
        # Port screw pushing ahead turns the bow to starboard: negative CCW moment.
        moment = (right - left) * p.screw_spacing_m / 2.0
        moment += rng.normal(0.0, p.yaw_disturbance_nm)
        du = (surge - p.surge_linear * s.u - p.surge_quadratic * s.u * abs(s.u)) / p.mass_kg + s.v * s.r
        dv = -p.sway_linear * s.v / p.mass_kg - s.u * s.r
        dr = (moment - p.yaw_linear * s.r - p.yaw_quadratic * s.r * abs(s.r)) / p.yaw_inertia_kgm2
        s.u += du * dt
        s.v += dv * dt
        s.r += dr * dt
        s.psi = wrap(s.psi + s.r * dt)
        vx, vy = s.velocity
        s.x += (vx + self.current[0]) * dt
        s.y += (vy + self.current[1]) * dt
        self._walls(pool)

    def _walls(self, pool):
        s, m = self.s, pool.margin_m
        vx, vy = s.velocity
        hit = False
        if s.x < pool.x_min + m:
            s.x, vx, hit = pool.x_min + m, max(vx, 0.0), True
        elif s.x > pool.x_max - m:
            s.x, vx, hit = pool.x_max - m, min(vx, 0.0), True
        if s.y < pool.y_min + m:
            s.y, vy, hit = pool.y_min + m, max(vy, 0.0), True
        elif s.y > pool.y_max - m:
            s.y, vy, hit = pool.y_max - m, min(vy, 0.0), True
        if hit:
            c, sn = math.cos(s.psi), math.sin(s.psi)
            s.u, s.v = vx * c + vy * sn, -vx * sn + vy * c
            s.wall_contacts += 1


# --- controllers: (t, own HullState, {ship: HullState}) -> (cmd_l, cmd_r) PWM ---

class Hold:
    def __call__(self, t, me, world):
        return 0.0, 0.0


class Script:
    """Open-loop PWM segments `(until_s, left, right)`; zero after the last."""

    def __init__(self, segments):
        self.segments = tuple(segments)

    def __call__(self, t, me, world):
        for until, left, right in self.segments:
            if t < until:
                return float(left), float(right)
        return 0.0, 0.0


class Waypoints:
    """Steer at each point in turn; slow down while badly misaligned, pivot beyond 90 deg."""

    def __init__(self, points, pwm=25.0, radius_m=0.8, loop=False, kp=45.0, kd=25.0,
                 max_turn=40.0, start_s=0.0):
        self.points = [tuple(p) for p in points]
        self.pwm, self.radius = float(pwm), float(radius_m)
        self.loop, self.kp, self.kd, self.max_turn = loop, kp, kd, max_turn
        self.start_s = float(start_s)
        self.index = 0

    def steer(self, me, tx, ty, pwm):
        error = wrap(math.atan2(ty - me.y, tx - me.x) - me.psi)
        turn = float(np.clip(self.kp * error - self.kd * me.r, -self.max_turn, self.max_turn))
        base = pwm * max(0.0, math.cos(error))
        return base - turn, base + turn

    def __call__(self, t, me, world):
        if t < self.start_s:
            return 0.0, 0.0
        while self.index < len(self.points):
            tx, ty = self.points[self.index]
            if math.hypot(tx - me.x, ty - me.y) > self.radius:
                return self.steer(me, tx, ty, self.pwm)
            self.index += 1
            if self.index == len(self.points) and self.loop:
                self.index = 0
        return 0.0, 0.0


class Pursue(Waypoints):
    """Head for the other hull's TRUE position. A baseline, not the deployed DQN."""

    def __init__(self, target_ship, pwm=30.0, stand_off_m=2.5, **kwargs):
        super().__init__([], pwm=pwm, **kwargs)
        self.target_ship, self.stand_off = int(target_ship), float(stand_off_m)

    def __call__(self, t, me, world):
        if t < self.start_s:
            return 0.0, 0.0
        other = world[self.target_ship]
        if math.hypot(other.x - me.x, other.y - me.y) < self.stand_off:
            return 0.0, 0.0
        return self.steer(me, other.x, other.y, self.pwm)


# --- sensors ---

@dataclass(frozen=True)
class TagMount:
    """Two UWB tags on one hull. `baseline_to_bow_deg` is the unsurveyed mounting
    constant `rig_state_producer.py` needs: bow = baseline bearing + this."""
    bow: str
    stern: str
    baseline_m: float
    baseline_to_bow_deg: float
    mid_forward_m: float = 0.0
    mid_port_m: float = 0.0
    z_error_m: float = 0.0            # true tag plane minus the plane the solver assumes

    def positions(self, hull, heights):
        mx, my = hull.to_world(self.mid_forward_m, self.mid_port_m)
        angle = hull.psi - math.radians(self.baseline_to_bow_deg)
        hx, hy = 0.5 * self.baseline_m * math.cos(angle), 0.5 * self.baseline_m * math.sin(angle)
        return {self.bow: (mx + hx, my + hy, heights[self.bow] + self.z_error_m),
                self.stern: (mx - hx, my - hy, heights[self.stern] + self.z_error_m)}


@dataclass(frozen=True)
class UwbParams:
    rate_hz: float = 10.0
    sigma_m: float = 0.035             # a healthy set solves to ~0.035 m RMSE on the rigs
    anchor_bias_sd_m: float = 0.02    # fixed per anchor-tag pair, drawn once
    dropout: float = 0.02
    anchors_per_burst: int = 4
    repick_s: float = 4.0
    repick_jitter: float = 0.10       # log-normal jitter on the slant-range score
    repick_hysteresis: float = 0.8    # score multiplier for anchors already in the set
    # The steep-path NLOS bias is OFF by default because the rigs contradict it:
    # on 2026-09-18 ship 2 at (19.4, 15.3) solved to 0.035 m RMSE with A03 at ~73
    # deg elevation. Set a threshold (e.g. 60) to test the hypothesis anyway.
    nlos_elevation_deg: float = None
    nlos_bias_max_m: float = 0.8
    nlos_sigma_m: float = 0.15
    firmware_position: bool = True
    pinned_sets: tuple = ()           # ((tag, (survey ids...)), ...) held for the whole run


@dataclass(frozen=True)
class ImuParams:
    rate_hz: float = 50.0
    yaw_offset_deg: float = 0.0       # PLC yaw reading when the bow points along +x
    drift_deg_per_min: float = 0.0
    yaw_sd_deg: float = 0.1
    attitude_sd_deg: float = 0.3
    cal: tuple = (0, 3, 0, 0)
    broken_capture_clock: bool = False  # ship 2: capture stamp leaks the PLC counter


@dataclass(frozen=True)
class ClockParams:
    wall_offset_s: float = 0.0        # rig wall clock minus the Mac's
    plc_offset_ms: float = 0.0        # this rig's t_plc_raw minus the common base
    latency_s: float = 0.015
    latency_sd_s: float = 0.004


@dataclass(frozen=True)
class CameraParams:
    matrix: tuple = CAM0_MATRIX
    dist: tuple = CAM0_DIST
    image_size: tuple = CAM0_SIZE
    fps: float = 4.31
    frame_jitter_s: float = 0.02
    lever_forward_m: float = 0.20
    lever_port_m: float = 0.0
    height_m: float = 0.30            # optical centre above the water
    mount_yaw_deg: float = 1.7        # optical axis to STARBOARD of the bow
    mount_pitch_deg: float = 0.0      # positive up
    # 0.69 px of track jitter measured at 640x360 working size (bearing_noise.py)
    # is 5 px at native 4608 width.
    pixel_sigma_px: float = 5.0
    detect_h50_px: float = 30.0       # native box height at 50 % detection probability
    detect_width_px: float = 8.0
    detect_max: float = 0.97
    max_elevation_deg: float = 8.0    # track_target's off-water gate


@dataclass
class ShipSetup:
    ship: int
    start: tuple                      # (x, y, heading_deg)
    controller: object
    tags: TagMount
    hull: HullParams = field(default_factory=HullParams)
    uwb: UwbParams = field(default_factory=UwbParams)
    imu: ImuParams = field(default_factory=ImuParams)
    clock: ClockParams = field(default_factory=ClockParams)
    camera: CameraParams = None
    current: tuple = (0.0, 0.0)


def default_tags(ship):
    """The two tags each rig carries, at their measured baselines; mounting angles invented."""
    if ship == 1:
        return TagMount('T01', 'T02', baseline_m=0.713, baseline_to_bow_deg=32.0)
    return TagMount('T03', 'T04', baseline_m=0.488, baseline_to_bow_deg=-14.0)


def default_imu(ship):
    """Drift and noise as measured on 2026-09-17; offsets arbitrary, as on the rigs."""
    if ship == 1:
        return ImuParams(yaw_offset_deg=-94.6, drift_deg_per_min=2.7, yaw_sd_deg=0.41,
                         cal=(0, 3, 0, 0))
    return ImuParams(yaw_offset_deg=-109.8, drift_deg_per_min=-0.13, yaw_sd_deg=0.042,
                     cal=(0, 3, 3, 0), broken_capture_clock=True)


def default_clock(ship):
    if ship == 1:
        return ClockParams(wall_offset_s=0.055, plc_offset_ms=0.0)
    return ClockParams(wall_offset_s=0.030, plc_offset_ms=42.0)


class UwbTag:
    """One tag: which anchors it hears, what it measures, what its firmware solves."""

    def __init__(self, ship, label, params, rng):
        self.ship, self.label, self.p, self.rng = ship, label, params, rng
        self.anchor_set = None
        self.next_repick = 0.0
        self.bias = {a: rng.normal(0.0, params.anchor_bias_sd_m) for a in ANCHORS}
        self.burst = int(rng.integers(10**8, 10**9))
        self.seq = int(rng.integers(1000, 60000))
        self.estimate = None
        self.pans = None
        self.history = []             # (sim t, anchor ids) whenever the set changes

    def _slant(self, position, anchor):
        ax, ay, az = ANCHORS[anchor]
        return math.dist(position, (ax, ay, az))

    def _choose(self, t, position):
        pinned = dict(self.p.pinned_sets).get(self.label)
        if pinned:
            if self.anchor_set is None:
                self.anchor_set = tuple(sorted(pinned))
                self.history.append((t, self.anchor_set))
            self.next_repick = math.inf
            return
        # Anchors already in the set are preferred, so a set changes when the
        # geometry has moved, not whenever the jitter says so.
        keep = set(self.anchor_set or ())
        score = {a: self._slant(position, a) * math.exp(self.rng.normal(0.0, self.p.repick_jitter))
                 * (self.p.repick_hysteresis if a in keep else 1.0)
                 for a in ANCHORS}
        chosen = tuple(sorted(sorted(score, key=score.get)[:self.p.anchors_per_burst]))
        if chosen != self.anchor_set:
            self.history.append((t, chosen))
        self.anchor_set = chosen
        self.next_repick = t + self.p.repick_s

    def _nlos(self, position, anchor):
        threshold = self.p.nlos_elevation_deg
        if threshold is None:
            return 0.0
        ax, ay, az = ANCHORS[anchor]
        horizontal = math.hypot(ax - position[0], ay - position[1])
        elevation = math.degrees(math.atan2(az - position[2], horizontal))
        if elevation <= threshold:
            return 0.0
        weight = (math.sin(math.radians(elevation)) - math.sin(math.radians(threshold))) \
            / (1.0 - math.sin(math.radians(threshold)))
        return max(0.0, self.p.nlos_bias_max_m * weight + self.rng.normal(0.0, self.p.nlos_sigma_m))

    def measure(self, t, position):
        """One burst: [(offset_s, anchor, range_m)] in publish order."""
        if self.anchor_set is None or t >= self.next_repick:
            self._choose(t, position)
        self.burst += int(1e6 / self.p.rate_hz)
        self.seq += 1
        out = []
        for i, anchor in enumerate(self.anchor_set):
            if self.rng.random() < self.p.dropout:
                continue
            value = (self._slant(position, anchor) + self.bias[anchor]
                     + self.rng.normal(0.0, self.p.sigma_m) + self._nlos(position, anchor))
            out.append((0.002 * i, anchor, value))
        return out

    def firmware_solve(self, ranges):
        """Free 3D Gauss-Newton, the way the DWM firmware lets z float."""
        if len(ranges) < 4:
            return None
        anchors = np.array([ANCHORS[a] for _, a, _ in ranges])
        measured = np.array([r for _, _, r in ranges])
        guess = (np.array(self.estimate) if self.estimate is not None
                 else np.array([*anchors[:, :2].mean(axis=0), 1.5]))
        for _ in range(12):
            delta = guess - anchors
            dist = np.linalg.norm(delta, axis=1)
            if np.any(dist < 1e-6):
                return None
            jac = delta / dist[:, None]
            step = np.linalg.lstsq(jac, measured - dist, rcond=None)[0]
            guess = guess + step
            if np.linalg.norm(step) < 1e-5:
                break
        rms = float(np.sqrt(np.mean((np.linalg.norm(guess - anchors, axis=1) - measured) ** 2)))
        if not np.all(np.isfinite(guess)):
            return None
        self.estimate = guess
        self.pans = guess if self.pans is None else 0.5 * self.pans + 0.5 * guess
        return guess, rms


def _quaternion(yaw_deg, pitch_deg, roll_deg):
    y, p, r = (math.radians(a) / 2.0 for a in (yaw_deg, pitch_deg, roll_deg))
    return [round(v, 6) for v in (
        math.cos(r) * math.cos(p) * math.cos(y) + math.sin(r) * math.sin(p) * math.sin(y),
        math.sin(r) * math.cos(p) * math.cos(y) - math.cos(r) * math.sin(p) * math.sin(y),
        math.cos(r) * math.sin(p) * math.cos(y) + math.sin(r) * math.cos(p) * math.sin(y),
        math.cos(r) * math.cos(p) * math.sin(y) - math.sin(r) * math.sin(p) * math.cos(y))]


class Camera:
    """Projection of the other hull through cam0's lens, and the tracker's reading of it."""

    def __init__(self, params, rng):
        import cv2
        self.cv2 = cv2
        self.p, self.rng = params, rng
        self.K = np.array(params.matrix, dtype=float)
        self.dist = np.array(params.dist, dtype=float)
        self.frame = 0

    def axes(self, hull):
        """Camera centre and OpenCV axes (x right, y down, z optical) in the world frame."""
        cx, cy = hull.to_world(self.p.lever_forward_m, self.p.lever_port_m)
        centre = np.array([cx, cy, self.p.height_m])
        yaw = hull.psi - math.radians(self.p.mount_yaw_deg)
        pitch = math.radians(self.p.mount_pitch_deg)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        z_axis = math.cos(pitch) * forward + math.sin(pitch) * up
        y_axis = math.sin(pitch) * forward - math.cos(pitch) * up
        x_axis = np.cross(y_axis, z_axis)
        return centre, np.vstack((x_axis, y_axis, z_axis))

    def angles(self, pixel):
        """Undistorted azimuth (starboard +) and elevation (up +), as track_target.bearings."""
        point = np.array([[list(pixel)]], dtype=float)
        x, y = self.cv2.undistortPoints(point, self.K, self.dist)[0, 0]
        return (math.degrees(math.atan2(x, 1.0)),
                math.degrees(math.atan2(-y, math.hypot(x, 1.0))))

    def _project(self, points_cam):
        pixels, _ = self.cv2.projectPoints(points_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                           self.K, self.dist)
        return pixels.reshape(-1, 2)

    def observe(self, own, target, target_params):
        """A bearings.csv row (native pixels) plus the truth it was drawn from."""
        self.frame += 1
        centre, rotation = self.axes(own)
        half_l, half_b, top = target_params.length_m / 2, target_params.beam_m / 2, target_params.freeboard_m
        corners = []
        for f in (-half_l, half_l):
            for p in (-half_b, half_b):
                x, y = target.to_world(f, p)
                corners += [(x, y, 0.0), (x, y, top)]
        cam = (np.array(corners) - centre) @ rotation.T
        world_ref = np.array([target.x, target.y, top / 2])
        ref = rotation @ (world_ref - centre)
        truth = {'true_range_m': float(math.hypot(*(world_ref - centre)[:2])),
                 'true_az_deg': math.degrees(math.atan2(ref[0], ref[2])),
                 'true_rel_bearing_deg': math.degrees(wrap(
                     own.psi - math.atan2(world_ref[1] - centre[1], world_ref[0] - centre[0]))),
                 'visible': False, 'box_h_px': None}
        width, height = self.p.image_size
        # Past ~55 deg off axis the 5-term distortion model folds back on itself.
        if np.any(cam[:, 2] <= 0.05) or np.any(np.abs(cam[:, 0] / cam[:, 2]) > math.tan(math.radians(55))):
            return {'status': 'lost'}, truth
        pixels = self._project(cam)
        u1, v1 = pixels.min(axis=0)
        u2, v2 = pixels.max(axis=0)
        if u2 < 0 or v2 < 0 or u1 >= width or v1 >= height:
            return {'status': 'lost'}, truth
        u1, u2 = max(u1, 0.0), min(u2, width - 1.0)
        v1, v2 = max(v1, 0.0), min(v2, height - 1.0)
        truth.update(visible=True, box_h_px=float(v2 - v1))
        chance = self.p.detect_max / (1.0 + math.exp(-(v2 - v1 - self.p.detect_h50_px) / self.p.detect_width_px))
        if self.rng.random() > chance:
            return {'status': 'lost'}, truth
        noise = self.rng.normal(0.0, self.p.pixel_sigma_px, 2)
        centre_px = ((u1 + u2) / 2 + noise[0], (v1 + v2) / 2 + noise[1])
        water_px = (centre_px[0], v2 + self.rng.normal(0.0, self.p.pixel_sigma_px))
        az, el = self.angles(centre_px)
        _, el_water = self.angles(water_px)
        status = 'ok' if abs(el) <= self.p.max_elevation_deg else 'off_water'
        return {'status': status, 'u_px': centre_px[0], 'v_px': centre_px[1], 'az_deg': az,
                'el_deg': el, 'el_water_deg': el_water, 'box_w_px': float(u2 - u1),
                'box_h_px': float(v2 - v1)}, truth


# --- the world ---

@dataclass
class Recording:
    messages: list                    # (received_at, topic, payload)
    truth: dict                       # ship -> list of dicts
    frames: list                      # camera rows, bearings.csv shape
    frame_truth: list
    anchor_sets: dict                 # tag -> [(t, anchors)]
    setups: dict
    epoch: float
    camera_ship: int
    target_ship: int
    pool: Pool
    duration_s: float


class Twin:
    TRUTH_DT = 0.05

    def __init__(self, setups, pool=Pool(), dt=0.01, seed=0, epoch=1789729700.0,
                 camera_ship=1, target_ship=2):
        self.setups = {s.ship: s for s in setups}
        self.pool, self.dt, self.epoch = pool, float(dt), float(epoch)
        self.rng = np.random.default_rng(seed)
        self.camera_ship, self.target_ship = camera_ship, target_ship
        self.hulls = {}
        self.tags, self.heights, self.imu_next, self.pwm_next, self.uwb_next = {}, {}, {}, {}, {}
        for ship, setup in self.setups.items():
            x, y, heading = setup.start
            state = HullState(x, y, math.radians(heading))
            self.hulls[ship] = Hull(setup.hull, state, setup.current)
            self.heights[ship] = dict(RIG_TAGS[ship]['tags'])
            self.tags[ship] = {label: UwbTag(ship, label, setup.uwb, self.rng)
                               for label in (setup.tags.bow, setup.tags.stern)}
            self.imu_next[ship] = self.rng.uniform(0, 1 / setup.imu.rate_hz)
            self.pwm_next[ship] = self.rng.uniform(0, 0.05)
            self.uwb_next[ship] = {label: self.rng.uniform(0, 1 / setup.uwb.rate_hz)
                                   for label in self.tags[ship]}
        camera = self.setups[camera_ship].camera if camera_ship in self.setups else None
        self.camera = Camera(camera, self.rng) if camera else None
        self.frame_next = 0.0

    # clocks
    def wall(self, ship, t):
        return self.epoch + t + self.setups[ship].clock.wall_offset_s

    def arrival(self, ship, t):
        clock = self.setups[ship].clock
        return self.epoch + t + max(0.002, self.rng.normal(clock.latency_s, clock.latency_sd_s))

    def plc(self, ship, t):
        return int(999.9 * (t + 40000.0) + self.setups[ship].clock.plc_offset_ms)

    def run(self, duration_s):
        messages, frames, frame_truth = [], [], []
        truth = {ship: [] for ship in self.setups}
        steps = int(round(duration_s / self.dt))
        next_truth = 0.0
        for step in range(steps + 1):
            t = step * self.dt
            world = {ship: hull.s for ship, hull in self.hulls.items()}
            if t >= next_truth - 1e-9:
                for ship, hull in self.hulls.items():
                    truth[ship].append(self._truth_row(ship, t))
                next_truth += self.TRUTH_DT
            for ship in self.setups:
                self._sensors(ship, t, messages)
            if self.camera is not None and t >= self.frame_next:
                row, true = self.camera.observe(self.hulls[self.camera_ship].s,
                                                self.hulls[self.target_ship].s,
                                                self.setups[self.target_ship].hull)
                stamp = self.wall(self.camera_ship, t)
                frames.append({'frame': len(frames), 'ts_ms': int(round(stamp * 1000)), **row})
                frame_truth.append({'frame': len(frame_truth), 't_sim': round(t, 3), **true})
                self.frame_next = t + 1.0 / self.camera.p.fps + self.rng.normal(0, self.camera.p.frame_jitter_s)
            if step == steps:
                break
            for ship, setup in self.setups.items():
                cmd = setup.controller(t, self.hulls[ship].s, world)
                self.hulls[ship].step(self.dt, *cmd, self.rng, self.pool)
        messages.sort(key=lambda m: m[0])
        sets = {tag.label: tag.history for tags in self.tags.values() for tag in tags.values()}
        return Recording(messages, truth, frames, frame_truth, sets, self.setups, self.epoch,
                         self.camera_ship, self.target_ship, self.pool, duration_s)

    def _truth_row(self, ship, t):
        s = self.hulls[ship].s
        vx, vy = s.velocity
        tags = self.setups[ship].tags.positions(s, self.heights[ship])
        row = {'t_sim': round(t, 3), 't_wall': round(self.wall(ship, t), 4), 'x_m': s.x, 'y_m': s.y,
               'heading_rad': s.psi, 'vx_mps': vx, 'vy_mps': vy, 'u_mps': s.u, 'v_mps': s.v,
               'r_radps': s.r, 'level_l': s.level_l, 'level_r': s.level_r,
               'cmd_l': s.cmd_l, 'cmd_r': s.cmd_r, 'wall_contacts': s.wall_contacts}
        mx, my = s.to_world(self.setups[ship].tags.mid_forward_m, self.setups[ship].tags.mid_port_m)
        row.update(mid_x_m=mx, mid_y_m=my)
        for label, (x, y, z) in tags.items():
            row.update({f'{label}_x_m': x, f'{label}_y_m': y, f'{label}_z_m': z})
        return row

    def _sensors(self, ship, t, messages):
        setup, hull = self.setups[ship], self.hulls[ship].s
        positions = None
        for label, tag in self.tags[ship].items():
            if t < self.uwb_next[ship][label]:
                continue
            self.uwb_next[ship][label] += 1.0 / setup.uwb.rate_hz
            positions = positions or setup.tags.positions(hull, self.heights[ship])
            ranges = tag.measure(t, positions[label])
            for offset, anchor, value in ranges:
                wire = WIRE_LABEL[anchor]
                messages.append((self.arrival(ship, t + offset), f'ship/{ship}/uwb/ranging/{label}/{wire}',
                                 {'ship_id': ship, 'tag_label': label, 'anchor_label': wire,
                                  'range_m': round(value, 4), 'timestamp_sent': str(tag.burst),
                                  'timestamp_received': round(self.wall(ship, t + offset), 4)}))
            solved = tag.firmware_solve(ranges) if setup.uwb.firmware_position else None
            if solved is not None:
                (x, y, z), rms = solved
                px, py, pz = tag.pans
                messages.append((self.arrival(ship, t + 0.01), f'ship/{ship}/uwb/position/{label}',
                                 {'ship_id': ship, 'tag_label': label, 'n_anchors': len(ranges),
                                  'host_time': round(self.wall(ship, t + 0.01), 3),
                                  'device_us': tag.burst, 'fw': 'twin-dwm', 'seq': tag.seq,
                                  'pans': {'x': round(px, 3), 'y': round(py, 3), 'z': round(pz, 3),
                                           'qf': int(np.clip(100 - 150 * rms, 0, 100))},
                                  'lsq': {'x': round(x, 3), 'y': round(y, 3), 'z': round(z, 3),
                                          'rms_m': round(rms, 3), 'mode': '3D'}}))
        if t >= self.imu_next[ship]:
            self.imu_next[ship] += 1.0 / setup.imu.rate_hz
            messages.append((self.arrival(ship, t), f'ship/{ship}/imu/data', self._imu(ship, t)))
        if t >= self.pwm_next[ship]:
            self.pwm_next[ship] += 0.05
            messages.append((self.arrival(ship, t), f'ship/{ship}/motor/pwm',
                             {'ship_id': ship,
                              # Readback levels on the producer's assumed 48 +/- 50 scale.
                              'pwm_left': round(48.0 + 50.0 * hull.level_l, 2),
                              'pwm_right': round(48.0 + 50.0 * hull.level_r, 2),
                              'cmd_left': round(hull.cmd_l, 1), 'cmd_right': round(hull.cmd_r, 1),
                              'stale': False, 't_plc_raw': self.plc(ship, t)}))

    def _imu(self, ship, t):
        imu, hull = self.setups[ship].imu, self.hulls[ship].s
        yaw = -math.degrees(hull.psi) + imu.yaw_offset_deg + imu.drift_deg_per_min * t / 60.0
        yaw = (yaw + self.rng.normal(0.0, imu.yaw_sd_deg) + 180.0) % 360.0 - 180.0
        pitch, roll = self.rng.normal(0.0, imu.attitude_sd_deg, 2)
        wall = self.wall(ship, t) + 0.05
        capture = (wall + 12.35 * 3600 + self.rng.uniform(-150, 150)) if imu.broken_capture_clock else wall
        plc = self.plc(ship, t)
        return {'ship_id': ship, 'yaw': round(yaw, 4), 'pitch': round(pitch, 4), 'roll': round(roll, 4),
                'accel': [round(v, 2) for v in self.rng.normal((0.0, 0.0, 9.4), 0.05)],
                'gyro': [round(self.rng.normal(0, 0.3), 3), round(self.rng.normal(0, 0.3), 3),
                         round(-math.degrees(hull.r) + self.rng.normal(0, 0.3), 3)],
                'mag': [0.0, 0.0, 0.0], 'quat': _quaternion(yaw, pitch, roll), 'cal': list(imu.cal),
                'timestamp_capture': round(capture, 4), 't_plc_raw': plc, 't_sys_raw': plc + 1}
