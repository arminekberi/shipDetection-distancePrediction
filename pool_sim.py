"""Run the basin digital twin, and score the real pipeline against its truth.

`run` writes what the rigs would have produced -- a raw MQTT capture in
`rig_state_logger.py record` format and a `track_target.py` bearings.csv -- plus
the truth nothing on the rigs can measure. `check` feeds that capture through
the SAME code that runs against the live broker (`rig_state_producer.ShipState`,
`calibrate-heading`, `boatdet.tma`, the waterline range) and reports each error
against the truth. `dataset` builds a multi-run two-rig manifest that
`rig_calibration.py` consumes unchanged.

Every output carries `synthetic: true`. A number from here says what the code
does with data of this shape; it does not say the rigs produce data of this
shape. The twin's own assumptions are listed in `boatdet/twin.py`.

    venv/bin/python pool_sim.py list
    venv/bin/python pool_sim.py run --scenario range-sweep --out results/sim/range-sweep
    venv/bin/python pool_sim.py check results/sim/range-sweep
    venv/bin/python pool_sim.py dataset --out results/sim/dataset --runs 6
"""
import argparse
import bisect
import csv
from dataclasses import asdict, is_dataclass, replace
import json
import math
from pathlib import Path
import re

import numpy as np

from boatdet.uwb import RIG_TAGS, VENUES
from boatdet.twin import (ANCHORS, CameraParams, Hold, Pool, Pursue, Script, ShipSetup,
                          Twin, UwbParams, Waypoints, default_clock, default_imu,
                          default_tags, wrap)

BEARINGS_HEADER = ('frame', 'ts_ms', 't_s', 'status', 'u_px', 'v_px', 'az_deg', 'el_deg',
                   'el_water_deg', 'box_w_px', 'box_h_px')
STATE_HEADER = ('timestamp', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps',
                'heading_rad', 'roll_rad', 'pitch_rad',
                'position_sigma_m', 'heading_sigma_rad', 'valid')


def ship(number, start, controller, **kwargs):
    kwargs.setdefault('camera', CameraParams() if number == 1 else None)
    return ShipSetup(number, start, controller, default_tags(number), imu=default_imu(number),
                     clock=default_clock(number), **kwargs)


def pinned(number):
    """The anchor set each rig's tags were measured ranging to on 2026-09-18."""
    spec = RIG_TAGS[number]
    survey = tuple(VENUES['ytu']['anchor_map'][a] for a in spec['anchors'])
    return UwbParams(pinned_sets=tuple((tag, survey) for tag in spec['tags']))


# Ship 1 carries the only working camera (ship 2's imx708s fail their i2c probe),
# so ship 1 observes and ship 2 is the target in every scenario.
SCENARIOS = {
    'static': ('both hulls hold station where they sat on 2026-09-18, on the anchor sets measured there',
               60.0, lambda rng: [ship(1, (19.4, 22.0, -90.0), Hold(), uwb=pinned(1)),
                                  ship(2, (19.4, 15.3, 90.0), Hold(), uwb=pinned(2))]),
    'range-sweep': ('ship 1 holds and films; ship 2 zigzags 3-22 m out and back: the calibration capture',
                    200.0, lambda rng: [
                        ship(1, (16.5, 26.5, -90.0), Hold()),
                        ship(2, (16.0, 22.0, -90.0), Waypoints(
                            [(19.0, 16.0), (13.5, 10.0), (19.0, 5.0), (14.0, 4.0),
                             (18.5, 12.0), (15.0, 20.0), (17.0, 23.0)], pwm=25.0))]),
    'weave': ('ship 1 weaves toward a slow crossing target: the manoeuvre bearings-only range needs',
              90.0, lambda rng: [
                  ship(1, (16.0, 27.0, -90.0), Waypoints(
                      [(13.0, 23.0), (19.0, 19.0), (13.0, 15.0), (19.0, 11.0)], pwm=25.0)),
                  ship(2, (8.0, 6.0, 0.0), Waypoints([(27.0, 6.0)], pwm=12.0))]),
    'heading-cal': ('ship 2 runs long straight legs: what calibrate-heading needs',
                    150.0, lambda rng: [
                        ship(1, (16.5, 26.5, -90.0), Hold()),
                        ship(2, (7.0, 14.0, 0.0), Waypoints(
                            [(26.0, 14.0), (26.0, 11.0), (7.0, 11.0), (7.0, 14.0), (26.0, 14.0)],
                            pwm=25.0))]),
    'pivot': ('ship 1 pivots in place and sweeps its camera across a stationary ship 2',
              60.0, lambda rng: [
                  ship(1, (16.5, 22.0, -90.0), Script([(10, 0, 0), (22, 30, -30), (30, 0, 0),
                                                       (42, -30, 30), (60, 0, 0)])),
                  ship(2, (18.0, 12.0, 0.0), Hold())]),
    'corner': ('ship 2 drives into the corner where it parked on 2026-09-18: the degenerate anchor set',
               90.0, lambda rng: [
                   ship(1, (16.5, 22.0, 180.0), Hold()),
                   ship(2, (18.0, 15.0, 135.0), Waypoints([(4.9, 28.2)], pwm=25.0))]),
    'pursuit': ('ship 2 chases ship 1 on true positions: a baseline pursuer, NOT the deployed DQN',
                120.0, lambda rng: [
                    ship(1, (10.0, 22.0, 0.0), Waypoints(
                        [(24.0, 22.0), (24.0, 8.0), (10.0, 8.0), (10.0, 22.0)], pwm=20.0, loop=True)),
                    ship(2, (20.0, 14.0, 90.0), Pursue(1, pwm=30.0, start_s=5.0))]),
}


def jsonable(value):
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_csv(path, rows, header=None):
    header = header or list(rows[0])
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return '' if value is None else f'{value:.{digits}f}'


def simulate(name, seed=0, duration_s=None, nlos_deg=None, setups=None):
    description, default_duration, build = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    setups = setups or build(rng)
    if nlos_deg is not None:
        setups = [replace(s, uwb=replace(s.uwb, nlos_elevation_deg=nlos_deg)) for s in setups]
    twin = Twin(setups, seed=seed)
    return twin.run(duration_s or default_duration), description


def write_run(recording, out, scenario, description, seed):
    """Everything a rig would have recorded, the truth beside it, and the session record."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'telemetry.jsonl').open('w') as stream:
        for received, topic, payload in recording.messages:
            stream.write(json.dumps({'topic': topic, 'received_at': round(received, 6),
                                     'payload': payload}) + '\n')
    for number, rows in recording.truth.items():
        write_csv(out / f'truth_ship{number}.csv', rows)
        state = [{'timestamp': f"{r['t_wall']:.4f}", 'x_m': fmt(r['x_m']), 'y_m': fmt(r['y_m']),
                  'z_m': '0.0', 'vx_mps': fmt(r['vx_mps']), 'vy_mps': fmt(r['vy_mps']), 'vz_mps': '0.0',
                  'heading_rad': fmt(wrap(r['heading_rad']), 6), 'roll_rad': '0.0', 'pitch_rad': '0.0',
                  'position_sigma_m': '0.0', 'heading_sigma_rad': '0.0', 'valid': '1'}
                 for r in rows[::2]]              # 10 Hz, the rate the producer publishes
        write_csv(out / f'state_ship{number}.csv', state, STATE_HEADER)
    if recording.frames:
        first = recording.frames[0]['ts_ms']
        bearings = [{'frame': f['frame'], 'ts_ms': f['ts_ms'], 't_s': f"{(f['ts_ms'] - first) / 1000:.3f}",
                     'status': f['status'], **{k: fmt(f.get(k)) for k in BEARINGS_HEADER[4:]}}
                    for f in recording.frames]
        write_csv(out / 'bearings.csv', bearings, BEARINGS_HEADER)
        write_csv(out / 'camera_truth.csv', recording.frame_truth)
        write_observations(out)
    setups = recording.setups
    session = {
        'format': 'pool-twin-session-v1', 'synthetic': True, 'scenario': scenario,
        'description': description, 'seed': seed, 'duration_s': recording.duration_s,
        'epoch': recording.epoch, 'camera_ship': recording.camera_ship,
        'target_ship': recording.target_ship, 'pool': jsonable(recording.pool),
        'world_frame_id': 'ytu-uwb-anchor-frame (twin)',
        'ships': {str(n): {'start': s.start, 'controller': type(s.controller).__name__,
                           'tags': jsonable(s.tags), 'hull': jsonable(s.hull), 'uwb': jsonable(s.uwb),
                           'imu': jsonable(s.imu), 'clock': jsonable(s.clock),
                           'camera': jsonable(s.camera),
                           'wall_contacts': recording.truth[n][-1]['wall_contacts']}
                  for n, s in setups.items()},
        'anchor_sets': {tag: [[round(t, 2), [f'A0{a}' for a in anchors]] for t, anchors in sets]
                        for tag, sets in recording.anchor_sets.items()},
        'counts': {'messages': len(recording.messages), 'frames': len(recording.frames),
                   'frames_ok': sum(f['status'] == 'ok' for f in recording.frames)},
    }
    (out / 'session.json').write_text(json.dumps(session, indent=2) + '\n')
    return session


def write_observations(out):
    """bearings.csv -> observations.csv through the real adapter, waterline depth."""
    from bearings_to_observations import convert
    with (out / 'bearings.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    records, _ = convert(rows, depth_model='waterline', camera_height_m=1.0, horizon_el_deg=0.0,
                         min_depression_deg=0.05, target_length_m=1.0, focal_px=None, track_id='1')
    write_csv(out / 'observations.csv', records)


# --- check ---

class Truth:
    """Linear interpolation of one ship's truth by simulation time."""

    def __init__(self, path):
        with Path(path).open(newline='') as stream:
            self.rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        self.t = [r['t_sim'] for r in self.rows]

    def at(self, t, key, angle=False):
        i = bisect.bisect_left(self.t, t)
        if i <= 0 or i >= len(self.t):
            return None
        a, b = self.rows[i - 1], self.rows[i]
        f = (t - a['t_sim']) / (b['t_sim'] - a['t_sim'])
        delta = wrap(b[key] - a[key]) if angle else b[key] - a[key]
        value = a[key] + f * delta
        return wrap(value) if angle else value


def stats(values):
    values = np.abs(np.asarray([v for v in values if v is not None and math.isfinite(v)]))
    if not values.size:
        return None
    return {'n': int(values.size), 'median': round(float(np.median(values)), 4),
            'p95': round(float(np.quantile(values, 0.95)), 4), 'max': round(float(values.max()), 4)}


def signed(values):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)])
    if not values.size:
        return None
    return {'n': int(values.size), 'mean': round(float(values.mean()), 4),
            'sd': round(float(values.std()), 4)}


def load(folder):
    folder = Path(folder)
    session = json.loads((folder / 'session.json').read_text())
    records = [json.loads(line) for line in (folder / 'telemetry.jsonl').read_text().splitlines()]
    truths = {int(n): Truth(folder / f'truth_ship{n}.csv') for n in session['ships']}
    return session, records, truths


def check_producer(session, records, truths):
    """The live producer's objects, ticked at 10 Hz on the Mac clock, against truth."""
    import rig_state_producer as producer
    ships = sorted(truths)
    offsets = {n: session['ships'][str(n)]['tags']['baseline_to_bow_deg'] for n in ships}
    states = producer.build_ships(ships, offsets)
    epoch = session['epoch']
    result, traces = {}, {n: [] for n in ships}
    ticks = {n: {'total': 0, 'published': 0, 'reasons': {}} for n in ships}
    next_tick = records[0]['received_at'] + producer.DT if records else 0.0
    errors = {n: {'pos': [], 'vel': [], 'hd': [], 'r': []} for n in ships}

    def tick(now):
        for n, st in states.items():
            payload, why = st.payload(now)
            ticks[n]['total'] += 1
            if payload is None:
                key = re.split(r'[:\d]', why, maxsplit=1)[0].strip()
                ticks[n]['reasons'][key] = ticks[n]['reasons'].get(key, 0) + 1
                continue
            ticks[n]['published'] += 1
            wall = session['ships'][str(n)]['clock']['wall_offset_s']
            t_sim = payload['t_fix'] - epoch - wall
            truth = truths[n]
            x, y = truth.at(t_sim, 'mid_x_m'), truth.at(t_sim, 'mid_y_m')
            if x is None:
                continue
            e_pos = math.hypot(payload['px'] - x, payload['py'] - y)
            e_vel = math.hypot(payload['vx'] - truth.at(t_sim, 'vx_mps'),
                               payload['vy'] - truth.at(t_sim, 'vy_mps'))
            e_hd = math.degrees(wrap(payload['hd'] - truth.at(t_sim, 'heading_rad', angle=True)))
            e_r = payload['r'] - truth.at(now - epoch, 'r_radps') if truth.at(now - epoch, 'r_radps') is not None else None
            for key, value in (('pos', e_pos), ('vel', e_vel), ('hd', e_hd), ('r', e_r)):
                errors[n][key].append(value)
            traces[n].append((t_sim, e_pos, payload['px'], payload['py']))

    for record in records:
        while record['received_at'] >= next_tick:
            tick(next_tick)
            next_tick += producer.DT
        payload = record['payload']
        st = states.get(payload.get('ship_id'))
        if st is not None:
            st.accept(record['topic'], payload, arrival=record['received_at'])
    for n, st in states.items():
        result[n] = {
            'published_fraction': round(ticks[n]['published'] / max(1, ticks[n]['total']), 3),
            'not_published': ticks[n]['reasons'],
            'hull_fixes': st.n_hull, 'rejected_rmse': st.n_rejected,
            'position_error_m': stats(errors[n]['pos']),
            'velocity_error_mps': stats(errors[n]['vel']),
            'heading_error_deg': stats(errors[n]['hd']),
            'yaw_rate_error_radps': stats(errors[n]['r']),
            'tag_rmse_m': {tag: solver.stats()['solve_rmse_m'] for tag, solver in st.solvers.items()},
        }
    return result, traces


def check_firmware(session, records, truths):
    """How far the firmware's own free-z `uwb/position` sits from each tag, per axis."""
    out = {}
    epoch = session['epoch']
    for record in records:
        parts = record['topic'].split('/')
        if 'position' not in parts:
            continue
        payload = record['payload']
        n, tag = payload['ship_id'], payload['tag_label']
        t_sim = payload['host_time'] - epoch - session['ships'][str(n)]['clock']['wall_offset_s']
        truth = [truths[n].at(t_sim, f'{tag}_{axis}_m') for axis in 'xyz']
        if truth[0] is None:
            continue
        entry = out.setdefault(tag, {'dx': [], 'dy': [], 'dz': []})
        for axis, value in zip('xyz', truth):
            entry['d' + axis].append(payload['lsq'][axis] - value)
    return {tag: {k: signed(v) for k, v in e.items()} for tag, e in out.items()}


def check_heading(session, records):
    import rig_state_producer as producer
    out = {}
    for n in sorted(int(k) for k in session['ships']):
        offsets, _ = producer.heading_offset_samples(records, n)
        truth = session['ships'][str(n)]['tags']['baseline_to_bow_deg']
        if len(offsets) < 10:
            out[n] = {'samples': len(offsets), 'estimate_deg': None, 'truth_deg': truth,
                      'note': 'hull never made way; the offset is unobservable from this run'}
            continue
        mean, sd = producer.circular_stats(offsets)
        # calibrate-heading itself warns above 20 deg of spread; mirror that here.
        out[n] = {'samples': len(offsets), 'estimate_deg': round(math.degrees(mean), 2),
                  'truth_deg': truth, 'sd_deg': round(math.degrees(sd), 2),
                  'error_deg': round(math.degrees(wrap(mean - math.radians(truth))), 2),
                  'trusted': bool(math.degrees(sd) <= 20.0)}
    return out


def check_camera(session, folder, truths):
    """Bearings, waterline range and bearings-only TMA from the simulated cam0 track."""
    from boatdet.config import TmaConfig
    from boatdet.tma import BearingObservation, RangeHypothesisTma, bearing_sigma
    folder = Path(folder)
    if not (folder / 'bearings.csv').exists():
        return None, None
    n = session['camera_ship']
    camera = session['ships'][str(n)]['camera']
    wall = session['ships'][str(n)]['clock']['wall_offset_s']
    with (folder / 'bearings.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (folder / 'camera_truth.csv').open(newline='') as stream:
        truth_rows = list(csv.DictReader(stream))
    own = truths[n]
    sigma = bearing_sigma(camera['pixel_sigma_px'], camera['matrix'][0][0])
    tma = RangeHypothesisTma(TmaConfig())
    az_err, range_err, tma_rows = [], [], []
    visible = sum(r['visible'] == 'True' for r in truth_rows)
    for row, true in zip(rows, truth_rows):
        if row['status'] != 'ok':
            continue
        t_sim = int(row['ts_ms']) / 1000 - session['epoch'] - wall
        true_range = float(true['true_range_m'])
        az_err.append(float(row['az_deg']) - float(true['true_az_deg']))
        depression = -float(row['el_water_deg'])
        if depression > 0.05:
            waterline = camera['height_m'] / math.tan(math.radians(depression))
            range_err.append((waterline - true_range) / true_range)
        heading = own.at(t_sim, 'heading_rad', angle=True)
        if heading is None:
            continue
        x, y = own.at(t_sim, 'x_m'), own.at(t_sim, 'y_m')
        c, s = math.cos(heading), math.sin(heading)
        sensor = (x + camera['lever_forward_m'] * c - camera['lever_port_m'] * s,
                  y + camera['lever_forward_m'] * s + camera['lever_port_m'] * c)
        relative = float(row['az_deg']) + camera['mount_yaw_deg']
        estimate = tma.update(BearingObservation(t_sim, wrap(heading - math.radians(relative)),
                                                 sigma, *sensor))
        tma_rows.append({'t_sim': t_sim, 'true_range_m': true_range, 'range_m': estimate.range_m,
                         'low_m': estimate.range_low_m, 'high_m': estimate.range_high_m,
                         'status': estimate.status})
    statuses = {}
    for r in tma_rows:
        statuses[r['status']] = statuses.get(r['status'], 0) + 1
    converged = [abs(r['range_m'] - r['true_range_m']) / r['true_range_m']
                 for r in tma_rows if r['status'] == 'converged']
    covered = [r['low_m'] <= r['true_range_m'] <= r['high_m'] for r in tma_rows]
    report = {
        'frames': len(rows), 'target_in_view': visible,
        'tracked': sum(r['status'] == 'ok' for r in rows),
        'azimuth_error_deg': stats(az_err), 'azimuth_bias_deg': signed(az_err),
        'waterline_relative_range_error': stats(range_err),
        'tma_status_counts': statuses,
        'tma_converged_relative_error': stats(converged),
        'tma_interval_coverage': round(float(np.mean(covered)), 3) if covered else None,
        'tma_final': tma_rows[-1] if tma_rows else None,
    }
    return report, tma_rows


def broken(times, *series, gap_s=1.0):
    """Insert NaN where samples are further apart than `gap_s`, so a line never bridges a dropout."""
    out = [[] for _ in range(len(series) + 1)]
    for i, t in enumerate(times):
        if i and t - times[i - 1] > gap_s:
            for column in out:
                column.append(float('nan'))
        out[0].append(t)
        for column, values in zip(out[1:], series):
            column.append(values[i])
    return out


def plot(folder, session, truths, traces, tma_rows):
    """map.png and errors.png, one y-axis per panel, ship colours fixed by identity."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = {1: '#2a78d6', 2: '#eb6834'}
    plt.rcParams.update({'axes.edgecolor': muted, 'axes.labelcolor': ink, 'xtick.color': muted,
                         'ytick.color': muted, 'axes.facecolor': surface, 'figure.facecolor': surface,
                         'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False})
    pool = session['pool']
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.add_patch(plt.Rectangle((pool['x_min'], pool['y_min']), pool['x_max'] - pool['x_min'],
                               pool['y_max'] - pool['y_min'], fill=False, ec=muted, lw=1))
    for key, (x, y, z) in ANCHORS.items():
        ax.plot(x, y, marker='s' if z > 5 else '^', ms=8, color=muted, mec=surface, mew=1.5, ls='')
        ax.annotate(f'A0{key}', (x, y), xytext=(5, 4), textcoords='offset points', color=muted, fontsize=8)
    for n, truth in truths.items():
        xs, ys = [r['x_m'] for r in truth.rows], [r['y_m'] for r in truth.rows]
        ax.plot(xs, ys, color=colours.get(n, ink), lw=2, label=f'ship {n} truth')
        ax.plot(xs[0], ys[0], marker='o', ms=8, color=colours.get(n, ink), mec=surface, mew=2)
        if traces.get(n):
            ax.plot([t[2] for t in traces[n]], [t[3] for t in traces[n]], ls='', marker='.', ms=2,
                    color=colours.get(n, ink), alpha=0.35, label=f'ship {n} producer px, py')
    ax.set_aspect('equal')
    ax.set_xlabel('x, m (anchor frame)')
    ax.set_ylabel('y, m')
    ax.set_title(f"{session['scenario']}: trajectories  (square = ceiling anchor, triangle = wall)",
                 loc='left', color=ink)
    ax.grid(color=grid, lw=0.6)
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(folder) / 'map.png', dpi=140)
    plt.close(fig)

    panels = 2 if tma_rows else 1
    fig, axes = plt.subplots(panels, 1, figsize=(8, 3.2 * panels), squeeze=False)
    ax = axes[0][0]
    for n, trace in traces.items():
        if trace:
            t, error = broken([p[0] for p in trace], [p[1] for p in trace])
            ax.plot(t, error, color=colours.get(n, ink), lw=1.5, label=f'ship {n}')
    ax.set_ylabel('position error, m')
    ax.set_xlabel('time, s')
    ax.set_title('rig_state_producer px, py against truth', loc='left', color=ink)
    ax.grid(color=grid, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    if tma_rows:
        ax = axes[1][0]
        t, low, high, estimate, true = broken(
            [r['t_sim'] for r in tma_rows], [r['low_m'] for r in tma_rows],
            [r['high_m'] for r in tma_rows], [r['range_m'] for r in tma_rows],
            [r['true_range_m'] for r in tma_rows])
        ax.fill_between(t, low, high, color='#1baf7a', alpha=0.18, lw=0, label='TMA range interval')
        ax.plot(t, estimate, color='#1baf7a', lw=1.5, label='TMA range')
        ax.plot(t, true, color=ink, lw=2, label='true range')
        ax.set_ylim(0, max(r['true_range_m'] for r in tma_rows) * 2.0)
        ax.set_ylabel('range to ship 2, m')
        ax.set_xlabel('time, s')
        ax.set_title('bearings-only TMA from the simulated cam0 track (gaps: target out of view)',
                     loc='left', color=ink)
        ax.grid(color=grid, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(folder) / 'errors.png', dpi=140)
    plt.close(fig)


def check(folder, plots=True):
    session, records, truths = load(folder)
    producer, traces = check_producer(session, records, truths)
    camera, tma_rows = check_camera(session, folder, truths)
    report = {'format': 'pool-twin-check-v1', 'synthetic': True, 'scenario': session['scenario'],
              'producer': producer, 'firmware_position_error_m': check_firmware(session, records, truths),
              'heading_calibration': check_heading(session, records), 'camera': camera}
    (Path(folder) / 'check.json').write_text(json.dumps(report, indent=2) + '\n')
    if plots:
        plot(folder, session, truths, traces, tma_rows)
    return report


def summary(report):
    lines = [f"scenario {report['scenario']} (synthetic)"]
    for n, p in report['producer'].items():
        pos, vel, hd = p['position_error_m'], p['velocity_error_mps'], p['heading_error_deg']
        lines.append(f"  ship {n} producer: published {p['published_fraction']:.0%} of ticks, "
                     f"{p['rejected_rmse']} fixes rejected by the RMSE gate")
        if pos:
            lines.append(f"      position err median {pos['median']:.3f} m, p95 {pos['p95']:.3f} m;  "
                         f"velocity err median {vel['median']:.3f} m/s;  hd err median {hd['median']:.2f} deg")
        if p['not_published']:
            lines.append(f"      withheld: {p['not_published']}")
    for tag, e in sorted(report['firmware_position_error_m'].items()):
        lines.append(f"  firmware uwb/position {tag}: bias x {e['dx']['mean']:+.2f}  y {e['dy']['mean']:+.2f}  "
                     f"z {e['dz']['mean']:+.2f} m")
    for n, h in report['heading_calibration'].items():
        if h['estimate_deg'] is None:
            lines.append(f"  ship {n} calibrate-heading: {h['note']}")
        else:
            verdict = '' if h['trusted'] else '  [sd > 20: the tool would warn]'
            lines.append(f"  ship {n} calibrate-heading: {h['estimate_deg']:+.1f} deg vs truth "
                         f"{h['truth_deg']:+.1f} (error {h['error_deg']:+.1f}, sd {h['sd_deg']:.1f}, "
                         f"{h['samples']} samples){verdict}")
    c = report['camera']
    if c:
        lines.append(f"  camera: target in view {c['target_in_view']}/{c['frames']} frames, "
                     f"tracked {c['tracked']}")
        if c['azimuth_error_deg']:
            lines.append(f"      azimuth err median {c['azimuth_error_deg']['median']:.3f} deg; "
                         f"waterline range rel err median "
                         f"{(c['waterline_relative_range_error'] or {}).get('median', float('nan')):.1%}")
            final = c['tma_final']
            lines.append(f"      TMA statuses {c['tma_status_counts']}, interval coverage "
                         f"{c['tma_interval_coverage']:.0%}; final {final['status']} "
                         f"{final['range_m']:.1f} m vs true {final['true_range_m']:.1f} m")
    return '\n'.join(lines)


# --- dataset ---

def dataset(out, runs, seed, duration_s):
    """Several range-sweep runs, each its own recording group, split train/val/test by run."""
    if runs < 3:
        raise SystemExit('need at least three runs: one each for train, val and test')
    out = Path(out)
    rng = np.random.default_rng(seed)
    manifest = {'format': 'two-rig-supervision-v1', 'world_frame': 'shared_local_xyz_z_up',
                'world_frame_id': 'ytu-uwb-anchor-frame (twin)', 'synthetic': True,
                'max_state_gap_s': 0.5, 'sessions': []}
    for index in range(runs):
        split = 'test' if index == runs - 1 else 'val' if index == runs - 2 else 'train'
        station = (rng.uniform(13.0, 20.0), rng.uniform(25.0, 27.0), rng.uniform(-100.0, -80.0))
        points = [(rng.uniform(10.0, 24.0), rng.uniform(3.0, 22.0)) for _ in range(6)]
        setups = [ship(1, station, Hold()),
                  ship(2, (station[0], station[1] - 4.0, -90.0), Waypoints(points, pwm=25.0))]
        recording, description = simulate('range-sweep', seed=seed + index, duration_s=duration_s,
                                           setups=setups)
        folder = out / f'run{index:02d}'
        session = write_run(recording, folder, 'range-sweep', description, seed + index)
        camera = session['ships']['1']['camera']
        wall = {n: session['ships'][str(n)]['clock']['wall_offset_s'] for n in (1, 2)}
        target_hull = session['ships']['2']['hull']
        with (folder / 'observations.csv').open(newline='') as stream:
            stamps = [float(r['timestamp']) for r in csv.DictReader(stream)]
        manifest['sessions'].append({
            'id': f'twin-run{index:02d}', 'group': f'twin-run{index:02d}',
            'capture_id': f'twin-capture{index:02d}', 'split': split,
            'camera_id': 'twin-ship1-cam0', 'observations': f'run{index:02d}/observations.csv',
            'bearing_source': 'logged_visual_bearing',
            'observer': {'rig_id': 'ship1', 'path': f'run{index:02d}/state_ship1.csv',
                         'clock': {'offset_s': 0.0, 'scale': 1.0},
                         'reference_lever_frd_m': [camera['lever_forward_m'], -camera['lever_port_m'],
                                                   -camera['height_m']]},
            # observations run on ship 1's wall clock; ship 2's log on its own
            'target': {'rig_id': 'ship2', 'path': f'run{index:02d}/state_ship2.csv',
                       'clock': {'offset_s': round(wall[2] - wall[1], 6), 'scale': 1.0},
                       'reference_lever_frd_m': [0.0, 0.0, -target_hull['freeboard_m'] / 2]},
            'associations': [{'track_id': 1, 'start_s': stamps[0] - 1.0, 'end_s': stamps[-1] + 1.0}],
        })
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return out / 'manifest.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='describe the scenarios')
    p = sub.add_parser('run', help='simulate one scenario and write its capture')
    p.add_argument('--scenario', choices=sorted(SCENARIOS), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--duration-s', type=float)
    p.add_argument('--nlos-deg', type=float,
                   help='test the steep-path NLOS hypothesis: bias ranges to anchors above '
                        'this elevation (off by default; the rigs contradict it)')
    p.add_argument('--check', action='store_true', help='run check on the result straight away')
    p = sub.add_parser('check', help='score the real pipeline against a run\'s truth')
    p.add_argument('folder', type=Path)
    p.add_argument('--no-plots', action='store_true')
    p = sub.add_parser('dataset', help='multi-run two-rig manifest for rig_calibration.py')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--runs', type=int, default=6)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--duration-s', type=float, default=150.0)
    args = parser.parse_args()

    if args.command == 'list':
        for name, (description, duration, _) in SCENARIOS.items():
            print(f'{name:12s} {duration:5.0f} s  {description}')
    elif args.command == 'run':
        recording, description = simulate(args.scenario, args.seed, args.duration_s, nlos_deg=args.nlos_deg)
        session = write_run(recording, args.out, args.scenario, description, args.seed)
        print(f"{args.scenario}: {session['counts']['messages']} messages, "
              f"{session['counts']['frames']} frames ({session['counts']['frames_ok']} tracked) -> {args.out}")
        if args.check:
            print(summary(check(args.out)))
    elif args.command == 'check':
        print(summary(check(args.folder, plots=not args.no_plots)))
    else:
        manifest = dataset(args.out, args.runs, args.seed, args.duration_s)
        print(f'wrote {manifest}\nnext:\n'
              f'  venv/bin/python rig_calibration.py prepare {manifest} --out {args.out}/prepared\n'
              f'  venv/bin/python rig_calibration.py fit {args.out}/prepared --out {args.out}/calibration.json\n'
              f'  venv/bin/python rig_calibration.py evaluate {args.out}/prepared '
              f'--calibration {args.out}/calibration.json --split val --out {args.out}/eval_val.json')


if __name__ == '__main__':
    main()
