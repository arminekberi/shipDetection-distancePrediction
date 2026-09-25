"""Publish `ship/<id>/state`, the topic the deployed RL pursuer reads and that
nothing on the rig produces.

`/root/HydroRL/rl_model/run_pursuer.py` subscribes to `ship/<own>/state` and
`ship/<target>/state` at 10 Hz and reads `px, py, vx, vy, hd, r` plus applied
thrust; from the target it reads only `px, py`. With no producer it fails safe
to idle forever, so the pursuer has a consumer and no producer. This is the
producer.

Where each field comes from:

  px, py   the midpoint of the hull's TWO UWB tags, each solved from its raw
           `uwb/ranging` burst with the tag height held FIXED. Not from
           `uwb/position`: the firmware's free 3D solve lets z float, which on
           ship 1 puts its tags at z = 2.9 and 4.4 m and moves y by 1.3-2.3 m.
           See `boatdet/uwb.py`.
  vx, vy   a constant-velocity Kalman over those midpoints, with R taken from
           the solver's own geometric covariance rather than a constant, so a
           burst solved from poor geometry is trusted less.
  hd       the bearing of the tag baseline plus a MOUNTING OFFSET THAT NOBODY
           HAS SURVEYED. See the safety note below -- this is the field that
           stops this tool from being run.
  r        differentiated IMU yaw. Yaw itself is gyro-only and drifts (ship 1
           +2.7 deg/min), but the DRIFT IS A CONSTANT OFFSET IN THE RATE and
           2.7 deg/min is 0.00079 rad/s against a MAX_ANGULAR_VELOCITY of 2.0,
           so the rate survives what the angle does not.
  motor_l/r  applied thrust from `motor/pwm` readback, not `cmd_*`: obs 5/6
           must be applied, because the hull has ~1.2 s of motor lag.

SAFETY -- why this does not publish by default.

  1. Publishing `ship/<id>/state` ARMS NOTHING BY ITSELF, but it removes the
     only thing currently stopping the pursuer: with no target the policy still
     returns action 10, half thrust ahead. It has no idle action, so for that
     runner "no data" already means MOTION, and giving it data means motion
     sooner. Never run this unattended; a person must be at the pool able to
     cut power.
  2. `hd` is not calibrated. The rotation between the tag baseline and the bow
     has never been measured, and neither has which tag is forward. A constant
     heading error rotates the pursuer's whole body frame, so it would chase a
     bearing that is wrong by that constant. This tool therefore REFUSES to
     publish a ship whose `heading_offset_deg` is None in `boatdet.uwb.RIG_TAGS`;
     `calibrate-heading` estimates it from a capture that contains motion.

  `run` without `--publish` computes everything and publishes nothing. That is
  the safe first test, and it is the default.

    venv/bin/python rig_state_producer.py run --seconds 30
    venv/bin/python rig_state_producer.py replay raw.jsonl
    venv/bin/python rig_state_producer.py calibrate-heading raw.jsonl --ship 2
"""
import argparse
from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from boatdet.uwb import (RIG_TAGS, RMSE_GATE_M, VENUES, HullTracker,
                         TagSolver, ytu_setup)

DT = 0.1                      # 10 Hz, the tick run_pursuer and the policy assume
STALE_AFTER_S = 0.5           # stop publishing rather than repeat a stale fix
PWM_NEUTRAL = 48.0            # observed readback at rest; see thrust_levels()
PWM_SPAN = 50.0
TOPICS = ('ship/+/uwb/ranging/#', 'ship/+/imu/data', 'ship/+/motor/pwm')


def thrust_levels(payload, neutral=PWM_NEUTRAL, span=PWM_SPAN):
    """Applied thrust in [-1, 1] from a `motor/pwm` readback.

    We publish `motor_l`/`motor_r` rather than leaving the runner to convert,
    and that is deliberate: `run_pursuer.py` uses `motor_l` when present and
    otherwise divides `pwm_left` by its OWN `--pwm-neutral 1500 --pwm-span 400`
    defaults. The rig publishes `pwm_left: 48.75`, so those defaults yield
    -3.6 instead of ~0.0 and feed the policy a thrust it was never trained on.
    Publishing the level directly keeps that conversion out of the loop.

    `neutral`/`span` here are read off the rest readback, NOT surveyed: the PLC
    clamps its own %MD1/%MD2 to +/-100 while the readback sits near 48 with a
    trim applied. Calibrate before trusting the magnitude.
    """
    if payload.get('stale', False):
        return None
    try:
        left = (float(payload['pwm_left']) - neutral) / span
        right = (float(payload['pwm_right']) - neutral) / span
    except (KeyError, TypeError, ValueError):
        return None
    return (max(-1.0, min(1.0, left)), max(-1.0, min(1.0, right)))


class VelocityFilter:
    """Constant-velocity Kalman over hull midpoints. State [x, y, vx, vy].

    The measurement noise is NOT a constant: each fix carries the covariance its
    anchor geometry earned, so a burst solved from a poor configuration widens R
    instead of dragging the velocity. That is the whole reason the solver
    computes a covariance at all.
    """

    def __init__(self, dt=DT, process_noise=0.5):
        self.dt = float(dt)
        q = float(process_noise)
        self.Q = np.diag([q * 0.05, q * 0.05, q, q]).astype(float)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.x = np.zeros(4)
        self.P = np.eye(4) * 10.0
        self.t = None
        self.initialized = False

    def update(self, t, z, R):
        dt = self.dt if self.t is None else max(1e-3, float(t) - self.t)
        self.t = float(t)
        if not self.initialized:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.eye(4) * 10.0
            self.initialized = True
            return self.outputs()
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self.Q * dt
        y = np.asarray(z, dtype=float) - x_pred[:2]
        S = P_pred[:2, :2] + np.asarray(R, dtype=float)
        K = P_pred[:, :2] @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        KH = np.zeros((4, 4))
        KH[:, :2] = K
        self.P = (np.eye(4) - KH) @ P_pred
        return self.outputs()

    def outputs(self):
        return self.x[:2].copy(), self.x[2:].copy()


class YawRate:
    """Yaw rate by differentiating IMU yaw, unwrapped, over a short window.

    The gyro triad would be more direct, but which axis is yaw depends on a
    mounting nobody has recorded, while `yaw` is unambiguous. Its drift is a
    constant that differentiation removes.
    """

    def __init__(self, window_s=0.5):
        self.window_s = float(window_s)
        self._samples = deque(maxlen=64)
        self._unwrapped = None
        self._last = None

    def add(self, t, yaw_deg):
        yaw = math.radians(float(yaw_deg))
        if self._unwrapped is None:
            self._unwrapped = yaw
        else:
            step = (yaw - self._last + math.pi) % (2.0 * math.pi) - math.pi
            self._unwrapped += step
        self._last = yaw
        self._samples.append((float(t), self._unwrapped))
        return self.rate()

    def rate(self):
        if len(self._samples) < 2:
            return 0.0
        t_end = self._samples[-1][0]
        window = [s for s in self._samples if t_end - s[0] <= self.window_s]
        if len(window) < 2:
            window = list(self._samples)[-2:]
        dt = window[-1][0] - window[0][0]
        if dt <= 0.0:
            return 0.0
        return (window[-1][1] - window[0][1]) / dt

    def yaw(self):
        return self._last


class ShipState:
    """Everything one hull needs, assembled from three independent feeds."""

    def __init__(self, ship, spec, *, heading_offset_deg=None,
                 max_rmse_m=RMSE_GATE_M):
        self.ship = int(ship)
        self.spec = spec
        self.max_rmse_m = float(max_rmse_m)
        self.n_rejected = 0
        self.last_rejected = None
        self.solvers = {tag: TagSolver(ytu_setup(tag, tag_z=z))
                        for tag, z in spec['tags'].items()}
        self.hull = HullTracker(ship, spec['bow'], spec['stern'])
        self.velocity = VelocityFilter()
        self.yaw_rate = YawRate()
        offset = (spec['heading_offset_deg'] if heading_offset_deg is None
                  else heading_offset_deg)
        self.heading_offset = (None if offset is None
                               else math.radians(float(offset)))
        self.motor = None
        self.last_hull = None
        self.n_hull = 0
        self.bearings = []

    @property
    def heading_calibrated(self):
        return self.heading_offset is not None

    def accept(self, topic, payload, *, retained=False, arrival=None):
        """Route one message. Returns a HullFix when one completes."""
        parts = topic.split('/')
        if 'ranging' in parts:
            tag = parts[parts.index('ranging') + 1]
            solver = self.solvers.get(tag)
            if solver is None:
                return None
            newest = None
            for fix in solver.accept(topic, payload, retained=retained,
                                     arrival=arrival):
                hull = self.hull.add(fix)
                if hull is None:
                    continue
                worst = max(hull.bow.rmse, hull.stern.rmse)
                if worst > self.max_rmse_m:
                    # The anchor set has gone bad under us. Drop the fix and let
                    # the state go stale rather than publish a position the
                    # ranges themselves do not agree on -- see RMSE_GATE_M.
                    self.n_rejected += 1
                    self.last_rejected = (worst, tuple(hull.bow.anchor_ids))
                    continue
                newest = hull
            if newest is not None:
                self._absorb(newest)
            return newest
        if parts[-1] == 'data' and 'imu' in parts:
            t = payload.get('timestamp_capture')
            yaw = payload.get('yaw')
            if t is not None and yaw is not None:
                # ship 2's timestamp_capture leaks the PLC counter and is 12 h
                # off, so the rate uses OUR receive time; a rate only needs a
                # consistent clock, not a correct epoch.
                self.yaw_rate.add(arrival if arrival is not None else time.time(),
                                  yaw)
            return None
        if parts[-1] == 'pwm':
            levels = thrust_levels(payload)
            if levels is not None:
                self.motor = levels
            return None
        return None

    def _absorb(self, hull):
        cov = self._pair_covariance(hull)
        self.velocity.update(hull.t, (hull.x, hull.y), cov)
        self.last_hull = hull
        self.n_hull += 1
        self.bearings.append((hull.t, hull.bearing_rad, hull.baseline_m))

    @staticmethod
    def _pair_covariance(hull):
        """Covariance of the MIDPOINT of two independent fixes: (Ca + Cb) / 4."""
        a, b = hull.bow.cov_xy, hull.stern.cov_xy
        return np.array([[(a[0] + b[0]) / 4.0, (a[1] + b[1]) / 4.0],
                         [(a[1] + b[1]) / 4.0, (a[2] + b[2]) / 4.0]])

    def payload(self, now):
        """The `ship/<id>/state` message, or None when it must not be sent."""
        hull = self.last_hull
        if hull is None or not self.velocity.initialized:
            if self.n_rejected:
                return None, (f'no usable hull fix: {self.n_rejected} rejected '
                              f'over {self.max_rmse_m} m RMSE')
            return None, 'no hull fix yet'
        age = now - hull.t
        if age > STALE_AFTER_S:
            # Stop publishing rather than repeat a stale fix: the runner keys
            # its own failsafe off the age of what it last received, and
            # republishing hides the dropout from it.
            return None, f'hull fix stale {age:.2f}s'
        if not self.heading_calibrated:
            return None, ('heading offset unsurveyed: the angle between the tag '
                          'baseline and the bow has never been measured')
        pos, vel = self.velocity.outputs()
        heading = (hull.bearing_rad + self.heading_offset) % (2.0 * math.pi)
        return {
            'ship_id': self.ship,
            'px': round(float(pos[0]), 4),
            'py': round(float(pos[1]), 4),
            'vx': round(float(vel[0]), 4),
            'vy': round(float(vel[1]), 4),
            'hd': round(float(heading), 5),
            'r': round(float(self.yaw_rate.rate()), 5),
            'motor_l': round(self.motor[0], 3) if self.motor else 0.0,
            'motor_r': round(self.motor[1], 3) if self.motor else 0.0,
            't_fix': round(hull.t, 3),
            'baseline_m': round(hull.baseline_m, 3),
            'src': 'rig_state_producer',
        }, None


def build_ships(ships, heading_offsets):
    return {s: ShipState(s, RIG_TAGS[s],
                         heading_offset_deg=heading_offsets.get(s))
            for s in ships}


def circular_stats(angles):
    """Mean direction and circular sd, in radians."""
    if not angles:
        return float('nan'), float('nan')
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    n = len(angles)
    R = math.hypot(s, c) / n
    sd = math.sqrt(-2.0 * math.log(R)) if 0.0 < R <= 1.0 else float('nan')
    return math.atan2(s, c), sd


def summarize(states, elapsed, published):
    print(f"\n{'=' * 74}")
    for ship, st in sorted(states.items()):
        print(f"ship {ship}: {st.n_hull} hull fixes in {elapsed:.1f}s "
              f"({st.n_hull / elapsed if elapsed else 0:.1f}/s), "
              f"paired {st.hull.n_paired} unpaired {st.hull.n_unpaired}")
        for tag, solver in sorted(st.solvers.items()):
            s = solver.stats()
            print(f"    {tag}: bursts={s['n_bursts']} fixes={s['n_fixes']} "
                  f"rmse={s['solve_rmse_m']} cond={s['solve_cond']} "
                  f"fail={s['fail_reasons']}")
        if st.n_rejected:
            worst, anchors = st.last_rejected
            heights = [VENUES['ytu']['anchors'][a][2] for a in anchors]
            print(f"    REJECTED {st.n_rejected} hull fixes over "
                  f"{st.max_rmse_m} m RMSE (worst {worst:.3f} m). Last bad set: "
                  f"{list(anchors)} at z={heights}. `cond` will look fine; see "
                  f"boatdet.uwb.RMSE_GATE_M.")
        if st.last_hull is not None:
            pos, vel = st.velocity.outputs()
            speed = float(np.hypot(vel[0], vel[1]))
            mean_b, sd_b = circular_stats([b for _, b, _ in st.bearings])
            lens = [L for _, _, L in st.bearings]
            print(f"    position ({pos[0]:.3f}, {pos[1]:.3f}) m   "
                  f"speed {speed:.3f} m/s   yaw rate {st.yaw_rate.rate():+.4f} rad/s")
            print(f"    baseline {sum(lens) / len(lens):.3f} m   bearing "
                  f"{math.degrees(mean_b) % 360:.1f} deg  sd {math.degrees(sd_b):.1f} deg"
                  + ("" if st.heading_calibrated
                     else "   [hd NOT PUBLISHABLE: mounting offset unsurveyed]"))
            print(f"    motor {st.motor}")
        payload, why = st.payload(time.time())
        if payload is None:
            print(f"    would not publish: {why}")
        else:
            print(f"    would publish: {json.dumps(payload)}")
    print(f"{'=' * 74}\npublished {published} messages")


def run(args):
    import paho.mqtt.client as mqtt

    offsets = dict(args.heading_offset or [])
    states = build_ships(args.ships, offsets)
    blocked = [s for s, st in states.items() if not st.heading_calibrated]
    if args.publish and blocked:
        raise SystemExit(
            f"refusing to publish ship {blocked}: the angle between the tag "
            f"baseline and the bow has never been surveyed, so `hd` would be "
            f"wrong by an unknown constant and the pursuer would chase a "
            f"rotated bearing. Measure it with `calibrate-heading` and pass "
            f"--heading-offset SHIP:DEG, or drop --publish.")
    if args.publish:
        print("!" * 74)
        print("PUBLISHING ship/<id>/state LIVE. The deployed pursuer has no idle")
        print("action: with no target it still returns half thrust ahead. A person")
        print("must be at the pool, able to cut power, for the whole of this run.")
        print("!" * 74, flush=True)

    counts = defaultdict(int)
    published = [0]
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, *_a, **_k):
        for topic in TOPICS:
            c.subscribe(topic, qos=1)
        print(f"subscribed {' '.join(TOPICS)}", flush=True)

    def on_message(_c, _u, message):
        try:
            payload = json.loads(message.payload.decode())
        except (ValueError, UnicodeDecodeError):
            counts['undecodable'] += 1
            return
        counts[message.topic] += 1
        ship = payload.get('ship_id')
        st = states.get(int(ship)) if ship is not None else None
        if st is not None:
            st.accept(message.topic, payload,
                      retained=bool(message.retain), arrival=time.time())

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=30)
    client.loop_start()

    start = time.time()
    next_tick = start + DT
    try:
        while time.time() - start < args.seconds:
            time.sleep(max(0.0, next_tick - time.time()))
            next_tick += DT
            now = time.time()
            for ship, st in sorted(states.items()):
                payload, why = st.payload(now)
                if payload is None:
                    continue
                if args.publish:
                    client.publish(f'ship/{ship}/state', json.dumps(payload),
                                   qos=0)
                    published[0] += 1
                if args.verbose:
                    print(f"ship {ship} {json.dumps(payload)}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()
    summarize(states, time.time() - start, published[0])


def replay(args):
    """Drive the same objects from a JSONL capture. No broker, no publishing."""
    states = build_ships(args.ships, dict(args.heading_offset or []))
    first = last = None
    rows = 0
    with Path(args.raw).open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            topic = record.get('topic', '')
            payload = record.get('payload')
            if not isinstance(payload, dict):
                continue
            arrival = record.get('received_at')
            ship = payload.get('ship_id')
            st = states.get(int(ship)) if ship is not None else None
            if st is None:
                continue
            rows += 1
            if first is None:
                first = arrival
            last = arrival
            st.accept(topic, payload, arrival=arrival)
    for st in states.values():
        for solver in st.solvers.values():
            for fix in solver.flush(force=True):
                hull = st.hull.add(fix)
                if hull is not None:
                    st._absorb(hull)
    elapsed = (last - first) if (first and last) else 0.0
    print(f"replayed {rows} rows spanning {elapsed:.1f}s from {args.raw}")
    summarize(states, elapsed, 0)


def heading_offset_samples(records, ship, min_speed=0.15):
    """(course over ground - tag baseline bearing) for each hull fix made under way.

    `records` are capture rows, `{'topic', 'received_at', 'payload'}`. Returns the
    samples in radians and the ShipState that produced them.
    """
    st = build_ships([ship], {})[ship]
    offsets = []
    for record in records:
        payload = record.get('payload')
        if not isinstance(payload, dict) or payload.get('ship_id') != ship:
            continue
        hull = st.accept(record.get('topic', ''), payload,
                         arrival=record.get('received_at'))
        if hull is None:
            continue
        _, vel = st.velocity.outputs()
        if float(np.hypot(vel[0], vel[1])) < min_speed:
            continue
        course = math.atan2(vel[1], vel[0])
        offsets.append((course - hull.bearing_rad + math.pi)
                       % (2.0 * math.pi) - math.pi)
    return offsets, st


def calibrate_heading(args):
    """Estimate the tag-baseline-to-bow offset from a capture WITH MOTION.

    While a hull makes way, its course over ground and its bow agree to within
    leeway, so the offset is the circular mean of (course - baseline bearing).
    A hull that is not moving says nothing about it, which is why this refuses
    a capture whose speed never rises above `--min-speed`.
    """
    def records():
        with Path(args.raw).open() as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue

    offsets, st = heading_offset_samples(records(), args.ship, args.min_speed)
    if not offsets:
        raise SystemExit(
            f"ship {args.ship} never exceeded {args.min_speed} m/s in this "
            f"capture, so it carries no information about the mounting offset. "
            f"A stationary hull cannot tell you which way its bow points -- "
            f"this needs a capture taken while the boat makes way.")
    mean, sd = circular_stats(offsets)
    print(f"ship {args.ship}: {len(offsets)} samples above {args.min_speed} m/s")
    print(f"  bow is {math.degrees(mean) % 360:.1f} deg from the "
          f"{st.spec['stern']}->{st.spec['bow']} baseline, circular sd "
          f"{math.degrees(sd):.1f} deg")
    print(f"  pass it back as:  --heading-offset {args.ship}:"
          f"{math.degrees(mean) % 360:.1f}")
    if math.degrees(sd) > 20.0:
        print("  WARNING: that spread is too wide to trust. Leeway, a weaving "
              "track or a short run all widen it; a straight steady leg is what "
              "this needs.")


def ship_offset(text):
    ship, _, degrees = text.partition(':')
    return int(ship), float(degrees)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--ships', type=int, nargs='+', default=[1, 2])
    common.add_argument('--heading-offset', type=ship_offset, nargs='*',
                        metavar='SHIP:DEG',
                        help='surveyed angle from the tag baseline to the bow')

    p = sub.add_parser('run', parents=[common],
                       help='subscribe live; computes everything, publishes '
                            'only with --publish')
    p.add_argument('--broker', default=os.environ.get('BROKER') or '192.168.1.110')
    p.add_argument('--port', type=int, default=1883)
    p.add_argument('--seconds', type=float, default=30.0)
    p.add_argument('--publish', action='store_true',
                   help='actually publish ship/<id>/state. Requires a surveyed '
                        'heading offset and a person at the pool.')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=run)

    p = sub.add_parser('replay', parents=[common],
                       help='drive the same code from a JSONL capture')
    p.add_argument('raw', type=Path)
    p.set_defaults(func=replay)

    p = sub.add_parser('calibrate-heading',
                       help='estimate the baseline-to-bow offset from a capture '
                            'that contains motion')
    p.add_argument('raw', type=Path)
    p.add_argument('--ship', type=int, required=True)
    p.add_argument('--min-speed', type=float, default=0.15)
    p.set_defaults(func=calibrate_heading)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
