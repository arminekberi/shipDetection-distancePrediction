"""Prepare two-rig supervision, fit sensor corrections, and evaluate bearings-only TMA."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from boatdet.config import TmaConfig
from boatdet.supervision import prepare, read_samples, sequences, sha256, write_json
from boatdet.tma import BearingObservation, RangeHypothesisTma, wrap_angle


def errors(predicted, actual, angular=False):
    residual = np.asarray(predicted) - np.asarray(actual)
    if angular:
        residual = (residual + 180) % 360 - 180
    if not len(residual):
        return {'count': 0}
    return {'count': len(residual), 'mae': float(np.abs(residual).mean()),
            'rmse': float(np.sqrt(np.square(residual).mean())),
            'bias': float(residual.mean()), 'p95_abs': float(np.quantile(np.abs(residual), .95))}


def fit(rows, provenance):
    """Fit on train only, balancing recording groups rather than frame counts."""
    cameras = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            cameras[row['camera_id']].append(row)
    if not cameras:
        raise ValueError('No train samples; never fit calibration on val/test')
    models = {}
    for camera, group in cameras.items():
        counts = defaultdict(int)
        for r in group:
            counts[r['group']] += 1
        weights = np.array([1/counts[r['group']] for r in group])
        delta = np.radians([r['truth']['relative_bearing_deg'] - r['inputs']['visual_bearing_deg'] for r in group])
        vector = np.sum(weights * np.exp(1j * delta)) / weights.sum()
        if abs(vector) < .5:
            raise ValueError(f'{camera}: bearing residuals do not support a single mounting bias')
        bias = math.degrees(math.atan2(vector.imag, vector.real))
        residual = np.array([math.degrees(wrap_angle(v-math.radians(bias))) for v in delta])
        sigma = max(.01, float(np.sqrt(np.average(residual**2, weights=weights))))
        depth_rows = [r for r in group if r['inputs']['visual_depth_m'] is not None]
        depth = None
        if depth_rows:
            raw = np.array([r['inputs']['visual_depth_m'] for r in depth_rows])
            true = np.array([r['truth']['horizontal_range_m'] for r in depth_rows])
            counts = defaultdict(int)
            for r in depth_rows:
                counts[r['group']] += 1
            w = np.sqrt([1/counts[r['group']] for r in depth_rows])
            design = np.column_stack((raw, np.ones(len(raw))))
            if len(raw) < 3 or np.linalg.matrix_rank(design) < 2:
                raise ValueError(f'{camera}: need at least three depth samples spanning multiple distances')
            coefficients = np.linalg.lstsq(design*w[:,None], true*w, rcond=None)[0]
            scale, offset = map(float, coefficients)
            if scale <= 0:
                raise ValueError(f'{camera}: fitted depth scale is not positive; check synchronization/association')
            depth = {'scale': scale, 'offset_m': offset, 'raw_train_min_m': float(raw.min()),
                     'raw_train_max_m': float(raw.max()), 'samples': len(raw),
                     'target': 'horizontal_range_to_target_reference_m'}
        models[camera] = {'bearing_bias_deg': bias, 'bearing_sigma_deg': sigma,
                          'bearing_samples': len(group), 'depth': depth,
                          'train_groups': sorted({r['group'] for r in group})}
    return {'format': 'rig-sensor-calibration-v1', 'fitted_split': 'train', 'models': models,
            'samples_sha256': provenance['samples_sha256'], 'synthetic': provenance['synthetic'],
            'deployment_ready': False}


def evaluate(rows, calibration, split='val', tma_config=None, max_gap_s=2.):
    """Replay only visual bearings + observer positions; read target truth for scoring afterwards."""
    chosen = [r for r in rows if r['split'] == split]
    if not chosen:
        raise ValueError(f'No {split} rows')
    config = tma_config or TmaConfig()
    groups = defaultdict(list)
    for row in chosen:
        groups[(row['session'], row['track_id'])].append(row)
    reports = []
    for (session, track_id), series in groups.items():
        series.sort(key=lambda r:r['timestamp'])
        model = calibration['models'].get(series[0]['camera_id'])
        if model is None:
            raise ValueError(f'No train calibration for camera {series[0]["camera_id"]}')
        if series[0]['group'] in model['train_groups']:
            raise ValueError('Evaluation group overlaps calibration training groups')
        estimator = None
        previous = None
        predicted_bearings, true_bearings, depths, ranges = [], [], [], []
        raw_depths, outside_domain, tma_errors, covered, statuses = [], 0, [], [], defaultdict(int)
        converged_errors, position_errors, tma_range_errors = [], [], []
        for row in series:
            obs, gt, t = row['inputs'], row['truth'], row['timestamp']
            if previous is None or t - previous > max_gap_s:
                estimator = RangeHypothesisTma(config)
            previous = t
            relative = obs['visual_bearing_deg'] + model['bearing_bias_deg']
            absolute = wrap_angle(obs['own_heading_rad'] - math.radians(relative))
            prediction = estimator.update(BearingObservation(t, absolute, math.radians(model['bearing_sigma_deg']),
                                                              obs['sensor_x_m'], obs['sensor_y_m']))
            # No target field is passed to the estimator, even for initialization.
            predicted_bearings.append(relative)
            true_bearings.append(gt['relative_bearing_deg'])
            if obs['visual_depth_m'] is not None and model['depth']:
                d = model['depth']
                raw = obs['visual_depth_m']
                raw_depths.append(raw)
                depths.append(d['scale'] * raw + d['offset_m'])
                ranges.append(gt['horizontal_range_m'])
                outside_domain += not d['raw_train_min_m'] <= raw <= d['raw_train_max_m']
            statuses[prediction.status] += 1
            error = prediction.range_m - gt['horizontal_range_m']
            tma_errors.append(error)
            tma_range_errors.append(abs(error) / gt['horizontal_range_m'])
            covered.append(prediction.range_low_m <= gt['horizontal_range_m'] <= prediction.range_high_m)
            position_errors.append(math.hypot(prediction.x_m-gt['reference_x_m'], prediction.y_m-gt['reference_y_m']))
            # Position estimate tracks the visual reference; velocity truth is the rig origin.
            # Velocity metrics require zero target lever, so report only state-model labels here.
            if prediction.status == 'converged':
                converged_errors.append(error)
        reports.append({'session': session, 'track_id': track_id, 'samples': len(series),
                        'bearing_deg': errors(predicted_bearings, true_bearings, angular=True),
                        'raw_depth_vs_horizontal_m': errors(raw_depths, ranges),
                        'corrected_depth_m': errors(depths, ranges), 'depth_outside_train_domain': outside_domain,
                        'tma_range_m': errors(tma_errors, np.zeros(len(tma_errors))),
                        'tma_converged_range_m': errors(converged_errors, np.zeros(len(converged_errors))),
                        'tma_position_mae_m': float(np.mean(position_errors)),
                        'tma_relative_range_mae': float(np.mean(tma_range_errors)),
                        'tma_interval_coverage': float(np.mean(covered)), 'tma_status_counts': dict(statuses)})
    return {'split': split, 'sessions': reports, 'synthetic': calibration['synthetic'],
            'note': 'Calibration fit uses train only. TMA receives no target state or depth. '
                    'Raw monocular depth vs horizontal range is a diagnostic, not an assertion that their definitions match.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('prepare')
    p.add_argument('manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('fit')
    p.add_argument('dataset', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('evaluate')
    p.add_argument('dataset', type=Path)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--tma-config', type=Path, help='JSON of explicit TmaConfig overrides, chosen without test truth')
    p.add_argument('--max-gap-s', type=float, default=2.)
    p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('sequences')
    p.add_argument('dataset', type=Path)
    p.add_argument('--window', type=int, default=32)
    p.add_argument('--max-gap-s', type=float, default=.5)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        _, report = prepare(args.manifest, args.out)
    else:
        rows = read_samples(args.dataset)
        provenance = json.loads((args.dataset / 'provenance.json').read_text())
        if args.command == 'fit':
            report = fit(rows, provenance)
        elif args.command == 'evaluate':
            calibration = json.loads(args.calibration.read_text())
            if calibration['samples_sha256'] != provenance['samples_sha256']:
                raise ValueError('Calibration and evaluation must refer to the same prepared dataset')
            if not math.isfinite(args.max_gap_s) or args.max_gap_s <= 0:
                parser.error('--max-gap-s must be positive and finite')
            config = TmaConfig(**json.loads(args.tma_config.read_text())) if args.tma_config else TmaConfig()
            report = evaluate(rows, calibration, args.split, config, args.max_gap_s)
        else:
            report = sequences(rows, args.out, args.window, args.max_gap_s)
            write_json(args.out / 'provenance.json', provenance)
        if args.command != 'sequences':
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
