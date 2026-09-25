"""Synchronized two-rig supervision. Target telemetry never enters model features.

Canonical world frame: shared local right-handed XYZ, Z up, meters. Heading is
CCW from +X; roll/pitch follow the existing FRD aerospace convention (radians).
Each telemetry clock has an explicit affine map from the video clock.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from boatdet.egomotion import Orientation, rotation_matrix
from boatdet.geometry import CameraCalibration, bearing_from_ray
from boatdet.tma import wrap_angle

STATE_FIELDS = ('x_m', 'y_m', 'z_m', 'vx_mps', 'vy_mps', 'vz_mps',
                'heading_rad', 'roll_rad', 'pitch_rad')
ANGLE_FIELDS = ('heading_rad', 'roll_rad', 'pitch_rad')
SPLITS = ('train', 'val', 'test')
FEATURE_NAMES = ('dt_s', 'bearing_sin', 'bearing_cos', 'sensor_dx_m', 'sensor_dy_m',
                 'heading_sin', 'heading_cos', 'own_vx_mps', 'own_vy_mps')
TARGET_NAMES = ('target_origin_dx_m', 'target_origin_dy_m', 'target_vx_mps', 'target_vy_mps')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


class StateLog:
    """Strict interpolation: no extrapolation, no duplicate/reversed stamps, no gap bridging."""
    def __init__(self, path, clock, max_gap_s):
        self.path = Path(path)
        self.offset = finite(clock['offset_s'], 'clock.offset_s')
        self.scale = finite(clock['scale'], 'clock.scale')
        self.max_gap_s = finite(max_gap_s, 'max_gap_s')
        if self.scale <= 0 or self.max_gap_s <= 0:
            raise ValueError('clock.scale and max_gap_s must be positive')
        self.samples = []
        for row in read_csv(path):
            # Retain invalid rows as gaps: never interpolate straight across a failed fix.
            sample = {'timestamp': finite(row['timestamp'], 'timestamp'), 'valid': row.get('valid', '1') == '1'}
            if row.get('valid', '1') not in ('0', '1'):
                raise ValueError('valid must be 0 or 1')
            if sample['valid']:
                sample.update({k: finite(row[k], k) for k in STATE_FIELDS})
                for key in ('position_sigma_m', 'heading_sigma_rad'):
                    sample[key] = finite(row[key], key) if row.get(key) else None
                    if sample[key] is not None and sample[key] < 0:
                        raise ValueError(f'{key} must be nonnegative')
            self.samples.append(sample)
        self.stamps = [r['timestamp'] for r in self.samples]
        if not self.stamps or any(b <= a for a, b in zip(self.stamps, self.stamps[1:])):
            raise ValueError(f'{path}: need strictly increasing, unique timestamps')

    def at(self, video_timestamp):
        t = self.offset + self.scale * video_timestamp
        index = bisect_left(self.stamps, t)
        if index < len(self.stamps) and abs(self.stamps[index] - t) <= 1e-9:
            return dict(self.samples[index]) if self.samples[index]['valid'] else None
        if index == 0 or index == len(self.stamps):
            return None
        before, after = self.samples[index-1:index+1]
        span = after['timestamp'] - before['timestamp']
        if span / self.scale > self.max_gap_s or not before['valid'] or not after['valid']:
            return None
        fraction = (t - before['timestamp']) / span
        result = {'timestamp': t, 'valid': True}
        for key in STATE_FIELDS:
            delta = after[key] - before[key]
            if key in ANGLE_FIELDS:
                delta = wrap_angle(delta)
            result[key] = before[key] + fraction * delta
            if key in ANGLE_FIELDS:
                result[key] = wrap_angle(result[key])
        # Conservative endpoint uncertainty, not a falsely precise interpolated value.
        for key in ('position_sigma_m', 'heading_sigma_rad'):
            values = (before[key], after[key])
            result[key] = max(values) if all(v is not None for v in values) else None
        return result


def reference_position(state, lever):
    """Rig origin -> chosen reference, using a [forward, starboard, down] lever arm."""
    heading = state['heading_rad']
    c, s = math.cos(heading), math.sin(heading)
    level_to_world = np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.]])
    tilt = rotation_matrix(Orientation(state['roll_rad'], state['pitch_rad'], 0))
    return np.array([state[k] for k in ('x_m', 'y_m', 'z_m')]) + level_to_world @ tilt @ np.asarray(lever)


def truth(own, target, camera_lever, target_lever):
    sensor = reference_position(own, camera_lever)
    reference = reference_position(target, target_lever)
    delta = reference - sensor
    horizontal = float(np.linalg.norm(delta[:2]))
    if horizontal < 1e-6:
        raise ValueError('Coincident XY references have undefined horizontal bearing')
    absolute = math.atan2(delta[1], delta[0])
    return {'horizontal_range_m': horizontal, 'slant_range_m': float(np.linalg.norm(delta)),
            'relative_bearing_deg': math.degrees(wrap_angle(own['heading_rad'] - absolute)),
            'absolute_bearing_rad': absolute, 'reference_x_m': float(reference[0]),
            'reference_y_m': float(reference[1]), 'reference_z_m': float(reference[2]),
            'sensor_x_m': float(sensor[0]), 'sensor_y_m': float(sensor[1]), 'sensor_z_m': float(sensor[2]),
            **{f'target_{key}': target[key] for key in STATE_FIELDS},
            'observer_position_sigma_m': own['position_sigma_m'],
            'target_position_sigma_m': target['position_sigma_m'],
            'observer_heading_sigma_rad': own['heading_sigma_rad']}


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('format') != 'two-rig-supervision-v1' or data.get('world_frame') != 'shared_local_xyz_z_up':
        raise ValueError('Manifest must declare two-rig-supervision-v1 and shared_local_xyz_z_up')
    if not data.get('world_frame_id'):
        raise ValueError('Declare the common surveyed world_frame_id; independent local origins cannot be mixed')
    if not isinstance(data.get('sessions'), list) or not data['sessions']:
        raise ValueError('Manifest needs sessions')
    groups, captures, names = {}, {}, set()
    for session in data['sessions']:
        sid, split = session['id'], session['split']
        if not sid or sid in names or split not in SPLITS:
            raise ValueError('Session IDs must be unique and split must be train/val/test')
        names.add(sid)
        for field, assigned in (('group', groups), ('capture_id', captures)):
            key = session[field]
            if not key or assigned.setdefault(key, split) != split:
                raise ValueError(f'{field} cannot cross train/val/test: {key}')
        if not session.get('camera_id'):
            raise ValueError('Each session needs camera_id (including its lens/mount configuration)')
        for role in ('observer', 'target'):
            rig = session[role]
            rig['path'] = str((path.parent / rig['path']).resolve())
            lever = rig['reference_lever_frd_m']
            if len(lever) != 3:
                raise ValueError('reference_lever_frd_m needs forward, starboard, down')
            rig['reference_lever_frd_m'] = [finite(v, 'lever') for v in lever]
        if session['observer']['rig_id'] == session['target']['rig_id']:
            raise ValueError('Observer and target must be different rigs')
        session['observations'] = str((path.parent / session['observations']).resolve())
        if session.get('bearing_source') not in ('logged_visual_bearing', 'pixels_with_state_attitude'):
            raise ValueError('Declare bearing_source: logged_visual_bearing or pixels_with_state_attitude')
        if session['bearing_source'] == 'pixels_with_state_attitude':
            session['camera_calibration'] = str((path.parent / session['camera_calibration']).resolve())
            if len(session['working_size']) != 2 or any(type(v) is not int or v <= 0 for v in session['working_size']):
                raise ValueError('working_size must be positive integer width,height of logged pixels')
        if not session.get('associations'):
            raise ValueError('Explicit track-to-target associations are required')
        for association in session['associations']:
            association['start_s'] = finite(association['start_s'], 'start_s')
            association['end_s'] = finite(association['end_s'], 'end_s')
            if association['start_s'] >= association['end_s']:
                raise ValueError('Associations use nonempty [start_s, end_s) intervals')
        ordered = sorted(session['associations'], key=lambda a: a['start_s'])
        if any(b['start_s'] < a['end_s'] for a, b in zip(ordered, ordered[1:])):
            raise ValueError('Only one unambiguous target association per timestamp is allowed')
    return data


def prepare(manifest_path, output):
    manifest = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, rejected, provenance = [], Counter(), []
    seen_samples, hashes_by_split = set(), {}
    for session in manifest['sessions']:
        split, sid = session['split'], session['id']
        logs = {role: StateLog(session[role]['path'], session[role]['clock'], manifest['max_state_gap_s'])
                for role in ('observer', 'target')}
        camera = None
        if session['bearing_source'] == 'pixels_with_state_attitude':
            camera = CameraCalibration.load(session['camera_calibration'])
            if not camera.image_size:
                raise ValueError('Camera calibration must declare image_size for scaling logged pixels')
            camera = camera.scaled(camera.image_size, session['working_size'])
        file_hash = sha256(session['observations'])
        if hashes_by_split.setdefault(file_hash, split) != split:
            raise ValueError('Identical observation logs cannot cross train/val/test')
        provenance.append({'session': sid, 'split': split, 'group': session['group'],
                           'observations_sha256': file_hash,
                           'observer_sha256': sha256(session['observer']['path']),
                           'target_sha256': sha256(session['target']['path'])})
        if camera is not None:
            provenance[-1]['camera_calibration_sha256'] = sha256(session['camera_calibration'])
        previous = -math.inf
        for raw in read_csv(session['observations']):
            t = finite(raw['timestamp'], 'observation timestamp')
            if t < previous:
                raise ValueError(f'{sid}: observation rows must be time ordered')
            previous = t
            selected = [a for a in session['associations'] if str(a['track_id']) == raw['track_id'] and a['start_s'] <= t < a['end_s']]
            if not selected:
                rejected['unassociated'] += 1
                continue
            key = (session['capture_id'], session['camera_id'], raw['track_id'], t)
            if key in seen_samples:
                raise ValueError(f'Duplicate capture/camera/track/timestamp: {key}')
            seen_samples.add(key)
            if int(raw['missed_frames']) != 0 or raw.get('tracking_status', 'observed') != 'observed':
                rejected['predicted'] += 1
                continue
            bearing_fields = ('visual_center_u', 'visual_center_v') if camera is not None else ('visual_bearing_deg',)
            if any(raw.get(key) in (None, '') for key in bearing_fields):
                rejected['missing_visual_bearing'] += 1
                continue
            own, target = logs['observer'].at(t), logs['target'].at(t)
            if own is None or target is None:
                rejected['state_gap_or_invalid'] += 1
                continue
            values = truth(own, target, session['observer']['reference_lever_frd_m'],
                           session['target']['reference_lever_frd_m'])
            if camera is None:
                bearing = finite(raw['visual_bearing_deg'], 'visual_bearing_deg')
            else:
                pixel = tuple(finite(raw[key], key) for key in bearing_fields)
                if not (0 <= pixel[0] < session['working_size'][0] and 0 <= pixel[1] < session['working_size'][1]):
                    raise ValueError('Logged visual center lies outside working_size')
                bearing = bearing_from_ray(pixel, camera, Orientation(own['roll_rad'], own['pitch_rad']))
                if bearing is None:
                    rejected['invalid_camera_ray'] += 1
                    continue
            depth = finite(raw['visual_depth_m'], 'visual_depth_m') if raw.get('visual_depth_m') else None
            if depth is not None and depth <= 0:
                raise ValueError('visual_depth_m must be positive or empty')
            rows.append({'session': sid, 'group': session['group'], 'split': split,
                         'camera_id': session['camera_id'], 'track_id': raw['track_id'], 'timestamp': t,
                         'inputs': {'visual_bearing_deg': bearing, 'visual_depth_m': depth,
                                    'sensor_x_m': values['sensor_x_m'], 'sensor_y_m': values['sensor_y_m'],
                                    'own_heading_rad': own['heading_rad'], 'own_vx_mps': own['vx_mps'],
                                    'own_vy_mps': own['vy_mps']},
                         'truth': values})
    if not rows:
        raise ValueError(f'No usable synchronized observations: {dict(rejected)}')
    with (output / 'samples.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    report = {'format': 'two-rig-samples-v1', 'manifest_sha256': sha256(manifest_path),
              'samples_sha256': sha256(output / 'samples.jsonl'),
              'counts': dict(Counter(r['split'] for r in rows)), 'rejected': dict(rejected),
              'sessions': provenance, 'world_frame_id': manifest['world_frame_id'],
              'synthetic': bool(manifest.get('synthetic', False))}
    write_json(output / 'provenance.json', report)
    write_json(output / 'manifest.resolved.json', manifest)
    return rows, report


def read_samples(folder):
    folder = Path(folder)
    provenance = json.loads((folder / 'provenance.json').read_text())
    if sha256(folder / 'samples.jsonl') != provenance['samples_sha256']:
        raise ValueError('Prepared samples changed; rerun prepare')
    return [json.loads(line) for line in (folder / 'samples.jsonl').read_text().splitlines()]


def sequences(rows, output, window=32, max_gap_s=.5):
    if window < 2 or not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('window >= 2 and finite max_gap_s > 0 required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['session'], row['track_id'])].append(row)
    collected = {split: [] for split in SPLITS}
    for (split, session, track_id), series in groups.items():
        series.sort(key=lambda row: row['timestamp'])
        for end in range(window, len(series)+1):
            chunk = series[end-window:end]
            times = np.array([r['timestamp'] for r in chunk])
            if np.any(np.diff(times) <= 0) or np.any(np.diff(times) > max_gap_s):
                continue
            start = chunk[0]['inputs']
            features = []
            for i, row in enumerate(chunk):
                obs = row['inputs']
                angle = math.radians(obs['visual_bearing_deg'])
                features.append([0 if i == 0 else times[i]-times[i-1], math.sin(angle), math.cos(angle),
                                 obs['sensor_x_m']-start['sensor_x_m'], obs['sensor_y_m']-start['sensor_y_m'],
                                 math.sin(obs['own_heading_rad']), math.cos(obs['own_heading_rad']),
                                 obs['own_vx_mps'], obs['own_vy_mps']])
            last, obs = chunk[-1]['truth'], chunk[-1]['inputs']
            label = [last['target_x_m']-obs['sensor_x_m'], last['target_y_m']-obs['sensor_y_m'],
                     last['target_vx_mps'], last['target_vy_mps']]
            collected[split].append((features, label, session, track_id, times[-1]))
    counts = {}
    for split, items in collected.items():
        counts[split] = len(items)
        np.savez_compressed(output / f'{split}.npz',
                            X=np.asarray([r[0] for r in items], dtype=np.float32).reshape(-1, window, len(FEATURE_NAMES)),
                            y=np.asarray([r[1] for r in items], dtype=np.float32).reshape(-1, len(TARGET_NAMES)),
                            session=np.asarray([r[2] for r in items], dtype=str),
                            track_id=np.asarray([r[3] for r in items], dtype=str),
                            timestamp=np.asarray([r[4] for r in items]))
    write_json(output / 'schema.json', {'features': FEATURE_NAMES, 'targets': TARGET_NAMES,
               'window': window, 'max_gap_s': max_gap_s, 'counts': counts,
               'target_frame': 'position relative to camera at final frame, velocity in shared world axes',
               'bearing_only': True, 'uses_target_state_as_input': False,
               'files_sha256': {s: sha256(output / f'{s}.npz') for s in SPLITS}})
    return counts
