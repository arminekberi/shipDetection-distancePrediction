"""Turn `track_target.py` bearings into the observation CSV `rig_calibration.py` reads.

The two formats were never joined: the tracker writes per-frame camera angles, and
`boatdet.supervision.prepare` wants `timestamp, track_id, missed_frames,
tracking_status, visual_bearing_deg, visual_depth_m`. This is that adapter, and it
is also where the only monocular range estimate on this rig is produced.

**Bearing.** `visual_bearing_deg` is `az_deg` as the tracker measured it: an angle
from the CAMERA's optical axis, not a ship-relative or true bearing. That is
deliberate. For an observer that holds station, the unknown hull heading and the
unknown camera mounting angle are one constant, and `rig_calibration.py fit`
absorbs exactly that constant into its per-camera `bearing_bias_deg`. Feeding it a
hub-frame angle instead would inject this rig's broken cam0/cam1 extrinsics.

**Range.** A hull floats at z=0, so its waterline sits below the horizon by an
angle that depends on distance alone: `range = height / tan(depression)`. The
tracker's box centre cannot be used for this — it rides up and down with
superstructure and target aspect — which is why the waterline is tracked
separately. The camera height above water is NOT measured here: `--camera-height-m`
declares a unit (1.0 m by default) and the least-squares `scale` that `fit` solves
for recovers the true height. The same goes for `--horizon-el-deg`, a residual
camera tilt whose first-order effect lands in the fitted `offset_m`.

That is what makes this a *calibration* rather than a measurement: the raw column
only has to be linear in true range, and the two fitted coefficients supply the
physical constants nobody has surveyed on this hull.

`--depth-model width` is the fallback, `range = f * length / width_px`. It is worse:
apparent width collapses as the target turns bow-on, and that is target aspect, not
distance. Use it only to cross-check the waterline model, never as the primary.

    venv/bin/python bearings_to_observations.py results/track_143012/bearings.csv \
        --out results/track_143012/observations.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path


def convert(rows, *, depth_model, camera_height_m, horizon_el_deg,
            min_depression_deg, target_length_m, focal_px, track_id):
    out, counts = [], {'observed': 0, 'predicted': 0, 'no_depth': 0}
    for row in rows:
        observed = row['status'] == 'ok'
        counts['observed' if observed else 'predicted'] += 1
        record = {'timestamp': f"{int(row['ts_ms']) / 1000:.3f}", 'frame_id': row['frame'],
                  'track_id': track_id, 'missed_frames': 0 if observed else 1,
                  'tracking_status': 'observed' if observed else 'predicted',
                  'visual_bearing_deg': f"{float(row['az_deg']):.4f}" if observed else '',
                  'visual_depth_m': ''}
        if observed and depth_model != 'none':
            raw = None
            if depth_model == 'waterline' and row.get('el_water_deg'):
                depression = horizon_el_deg - float(row['el_water_deg'])
                if depression > min_depression_deg:
                    raw = camera_height_m / math.tan(math.radians(depression))
            elif depth_model == 'width' and row.get('box_w_px'):
                width = float(row['box_w_px'])
                if width > 0:
                    raw = focal_px * target_length_m / width
            if raw is not None and raw > 0 and math.isfinite(raw):
                record['visual_depth_m'] = f'{raw:.4f}'
            else:
                counts['no_depth'] += 1
        out.append(record)
    return out, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bearings', type=Path, help='bearings.csv from track_target.py')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--depth-model', choices=('waterline', 'width', 'none'), default='waterline')
    parser.add_argument('--camera-height-m', type=float, default=1.0,
                        help='declared unit, not a measurement; fitted scale recovers the truth')
    parser.add_argument('--horizon-el-deg', type=float, default=0.0,
                        help='elevation of the horizon in camera angles; residual tilt')
    parser.add_argument('--min-depression-deg', type=float, default=0.05,
                        help='below this the range is numerically unbounded and is dropped')
    parser.add_argument('--target-length-m', type=float, default=1.0, help='--depth-model width only')
    parser.add_argument('--camera', default='cam0', help='--depth-model width only, for the focal length')
    parser.add_argument('--track-id', default='1')
    args = parser.parse_args()

    focal_px = None
    if args.depth_model == 'width':
        from track_target import load_intrinsics
        focal_px = float(load_intrinsics(args.camera).K[0, 0])

    with args.bearings.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f'{args.bearings}: no rows')

    records, counts = convert(rows, depth_model=args.depth_model,
                              camera_height_m=args.camera_height_m,
                              horizon_el_deg=args.horizon_el_deg,
                              min_depression_deg=args.min_depression_deg,
                              target_length_m=args.target_length_m,
                              focal_px=focal_px, track_id=args.track_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    stamps = [float(r['timestamp']) for r in records]
    depths = [float(r['visual_depth_m']) for r in records if r['visual_depth_m']]
    report = {'format': 'observations-from-bearings-v1', 'source': str(args.bearings.resolve()),
              'rows': len(records), 'observed': counts['observed'], 'predicted': counts['predicted'],
              'depth_rows': len(depths), 'depth_dropped': counts['no_depth'],
              'raw_depth_span_m': [round(min(depths), 3), round(max(depths), 3)] if depths else None,
              'first_timestamp': stamps[0], 'last_timestamp': stamps[-1],
              'duration_s': round(stamps[-1] - stamps[0], 3),
              'bearing_frame': 'camera optical axis, starboard positive; NOT ship-relative or true',
              'depth_model': {'name': args.depth_model, 'camera_height_m': args.camera_height_m,
                              'horizon_el_deg': args.horizon_el_deg,
                              'target_length_m': args.target_length_m if args.depth_model == 'width' else None,
                              'focal_px': focal_px,
                              'measured': False,
                              'note': 'Raw proxy only. The affine scale/offset that rig_calibration.py '
                                      'fit solves for carry the unmeasured camera height and tilt.'}}
    Path(str(args.out) + '.meta.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
