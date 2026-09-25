"""Record rig telemetry from MQTT and export the two-rig supervision state CSV.

Position comes from the UWB tags, in the anchor frame shared by every rig on the
same DWM network; that frame is the only thing both rigs measure in common, so it
is the `world_frame_id` the supervision manifest must declare.

Heading is not derivable from this telemetry. `ship/<id>/imu/data` publishes yaw
only while the PLC program runs, that yaw is CW-positive in the PLC's own
reference, and the rotation between that reference and the anchor axes has not
been surveyed. A rig that holds station therefore exports a declared constant
heading (`--fixed-heading-deg`) carrying an explicit `heading_sigma_rad`: for an
observer that never rotates, the unknown heading and the unknown camera mounting
angle are one constant, which `rig_calibration.py fit` absorbs into its per-camera
bearing bias. Do not use a declared heading for a rig that turns.

    venv/bin/python rig_state_logger.py record --seconds 120 --out raw.jsonl
    venv/bin/python rig_state_logger.py export raw.jsonl --ship 2 --tag T03 \
        --fixed-heading-deg 0 --static --out ship2_state.csv
"""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import time

import numpy as np

STATE_HEADER = ('timestamp', 'x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps',
                'heading_rad', 'roll_rad', 'pitch_rad',
                'position_sigma_m', 'heading_sigma_rad', 'valid')
TOPICS = ('ship/+/uwb/position/+', 'ship/+/uwb/ranging/#', 'ship/+/imu/data',
          'ship/+/motor/pwm')


def record(broker, port, seconds, out, topics=TOPICS):
    """Append every matching message to JSONL with the local receive time.

    Raw capture is kept separate from interpretation so a run is never lost to an
    export choice made afterwards.
    """
    import paho.mqtt.client as mqtt

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    with out.open('a') as stream:
        def on_connect(client, _userdata, _flags, code, *_):
            if code != 0:
                raise ConnectionError(f'{broker}:{port} refused the subscription: code {code}')
            for topic in topics:
                client.subscribe(topic, qos=0)

        def on_message(_client, _userdata, message):
            try:
                payload = json.loads(message.payload)
            except (ValueError, UnicodeDecodeError):
                counts['undecodable'] += 1
                return
            counts[message.topic] += 1
            stream.write(json.dumps({'topic': message.topic, 'received_at': time.time(),
                                     'payload': payload}, allow_nan=False) + '\n')

        version = getattr(mqtt, 'CallbackAPIVersion', None)
        client = mqtt.Client(version.VERSION1) if version else mqtt.Client()
        client.on_connect, client.on_message = on_connect, on_message
        client.connect(broker, port, 10)
        client.loop_start()
        try:
            time.sleep(seconds)
        finally:
            client.loop_stop()
            client.disconnect()
    return dict(counts)


def read_positions(path, ship, tag, source):
    """UWB fixes for one tag, as (rig clock seconds, x, y, z), earliest first.

    `host_time` is the publishing rig's own clock, which is what the manifest's
    per-log affine clock map is defined against; the local receive time is kept
    only as a fallback for messages that carry no rig stamp.
    """
    topic = f'ship/{ship}/uwb/position/{tag}'
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record_ = json.loads(line)
        if record_['topic'] != topic:
            continue
        fix = record_['payload'].get(source)
        if not fix:
            continue
        values = [fix.get(axis) for axis in ('x', 'y', 'z')]
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        stamp = record_['payload'].get('host_time', record_['received_at'])
        rows.append((float(stamp), *(float(v) for v in values)))
    rows.sort(key=lambda row: row[0])
    return rows


def deduplicate(rows, minimum_step_s=1e-6):
    """Drop repeated or reversed stamps; StateLog requires strictly increasing time."""
    kept, dropped = [], 0
    for row in rows:
        if kept and row[0] - kept[-1][0] < minimum_step_s:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def velocities(rows, window_s):
    """Least-squares slope over a centred time window, per axis.

    A two-sample difference of these fixes is dominated by fix noise: at 10 Hz,
    0.2 m of scatter is 2 m/s of apparent speed. The window trades lag for that
    noise and is reported in the sidecar so the smoothing is never implicit.
    """
    stamps = np.array([row[0] for row in rows])
    points = np.array([row[1:] for row in rows])
    result = np.zeros_like(points)
    for index, stamp in enumerate(stamps):
        inside = np.abs(stamps - stamp) <= window_s / 2
        if inside.sum() < 3:
            continue
        times = stamps[inside] - stamp
        if np.ptp(times) < window_s / 10:
            continue
        design = np.column_stack((times, np.ones(times.size)))
        result[index] = np.linalg.lstsq(design, points[inside], rcond=None)[0][0]
    return result


def scatter(rows, window_s):
    """Residual scatter around a local linear fit, as a single position sigma.

    For a rig holding station this is the fix noise itself. It is a scalar per
    log rather than a per-sample value because the export has no independent
    measure of an individual fix's quality.
    """
    stamps = np.array([row[0] for row in rows])
    points = np.array([row[1:3] for row in rows])
    residuals = []
    for index, stamp in enumerate(stamps):
        inside = np.abs(stamps - stamp) <= window_s / 2
        if inside.sum() < 3:
            continue
        times = stamps[inside] - stamp
        design = np.column_stack((times, np.ones(times.size)))
        fit = design @ np.linalg.lstsq(design, points[inside], rcond=None)[0]
        residuals.append(points[index] - fit[np.argmin(np.abs(times))])
    if not residuals:
        return None
    return float(np.sqrt(np.mean(np.square(np.array(residuals)))))


def export(path, ship, tag, out, *, source='pans', fixed_heading_deg=None,
           static=False, window_s=3.0, position_sigma_m=None, heading_sigma_deg=180.0):
    """Write the state CSV for one rig and a report of what the numbers rest on."""
    rows = read_positions(path, ship, tag, source)
    if not rows:
        raise ValueError(f'No {source} fixes for ship {ship} tag {tag} in {path}')
    rows, dropped = deduplicate(rows)
    if len(rows) < 3:
        raise ValueError(f'ship {ship} tag {tag}: need at least three distinct fixes')
    if fixed_heading_deg is None:
        raise ValueError('No heading source exists in this telemetry; declare '
                         '--fixed-heading-deg for a rig that holds station, and do '
                         'not export a rig that turns until yaw is surveyed')
    speed = np.zeros((len(rows), 3)) if static else velocities(rows, window_s)
    sigma = position_sigma_m if position_sigma_m is not None else scatter(rows, window_s)
    heading = math.radians(fixed_heading_deg)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(STATE_HEADER)
        for (stamp, x, y, z), (vx, vy, vz) in zip(rows, speed):
            writer.writerow([f'{stamp:.6f}', f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
                             f'{vx:.4f}', f'{vy:.4f}', f'{vz:.4f}',
                             f'{heading:.6f}', '0.0', '0.0',
                             '' if sigma is None else f'{sigma:.4f}',
                             f'{math.radians(heading_sigma_deg):.6f}', '1'])
    span = rows[-1][0] - rows[0][0]
    report = {'format': 'rig-state-export-v1', 'source_jsonl': str(Path(path).resolve()),
              'ship_id': ship, 'tag': tag, 'position_source': source,
              'samples': len(rows), 'dropped_nonincreasing': dropped,
              'duration_s': round(span, 3), 'rate_hz': round(len(rows) / span, 2) if span else None,
              'position_sigma_m': sigma, 'velocity': 'declared_zero' if static else f'lsq_window_{window_s}s',
              'heading': {'declared_deg': fixed_heading_deg, 'sigma_deg': heading_sigma_deg,
                          'measured': False,
                          'note': 'Constant declared heading. For a station-keeping observer this '
                                  'is absorbed with the camera mounting angle into the fitted '
                                  'per-camera bearing bias; it is not a measured attitude.'},
              'mean_position_m': [round(float(np.mean([r[i] for r in rows])), 4) for i in (1, 2, 3)]}
    Path(str(out) + '.meta.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('record', help='append raw MQTT telemetry to JSONL')
    p.add_argument('--broker', default=os.environ.get('BROKER') or '192.168.1.110')
    p.add_argument('--port', type=int, default=1883)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('export', help='write one rig state CSV from recorded JSONL')
    p.add_argument('raw', type=Path)
    p.add_argument('--ship', type=int, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--source', choices=('pans', 'lsq'), default='pans')
    p.add_argument('--fixed-heading-deg', type=float,
                   help='declared constant heading, CCW from +X of the anchor frame')
    p.add_argument('--heading-sigma-deg', type=float, default=180.0)
    p.add_argument('--static', action='store_true', help='declare zero velocity')
    p.add_argument('--window-s', type=float, default=3.0)
    p.add_argument('--position-sigma-m', type=float)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'record':
        counts = record(args.broker, args.port, args.seconds, args.out)
        print(json.dumps(counts, indent=2))
    else:
        print(json.dumps(export(args.raw, args.ship, args.tag, args.out, source=args.source,
                                fixed_heading_deg=args.fixed_heading_deg, static=args.static,
                                window_s=args.window_s, position_sigma_m=args.position_sigma_m,
                                heading_sigma_deg=args.heading_sigma_deg), indent=2))


if __name__ == '__main__':
    main()
