"""Split one capture's observations into train/val and write its supervision manifest.

One recording is one session, but `rig_calibration.py` must fit on train and be read
on val, and `prepare` refuses to let the same observation file cross splits. So the
run is cut in time: the first `--train-fraction` of frames fit the camera constants,
the rest are scored with them. A time cut, not an interleave — interleaving would put
adjacent frames of the same continuous track on both sides and score the fit against
data it has effectively already seen.

Be clear about what a single-session split can and cannot show. The camera bias and
depth affine are constants, so carrying them 40 s forward is close to trivial; this
measures that the pipeline is sound and that the constants are stable, NOT that they
generalise to another day, another light or another target aspect. That needs a
second recording in another `--group`.

**Heading and lever must be declared together.** `--observer-heading-deg` is absorbed
into the fitted per-camera bearing bias for an observer that holds station, so its
value is free — but `--observer-lever-frd-m` is expressed in the observer's own
frame, which that declared heading orients. Declare heading 0 with lever 0, or the
surveyed heading with the surveyed lever. Never a real lever against a made-up heading.
"""
import argparse
import csv
import json
from pathlib import Path


def split_rows(rows, fraction):
    cut = max(1, min(len(rows) - 1, round(len(rows) * fraction)))
    return rows[:cut], rows[cut:]


def write_csv(path, rows, header):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, help='holds observations.csv, observer.csv, target.csv')
    parser.add_argument('--capture-id', required=True, help='rig session id, e.g. 20260918-143012')
    parser.add_argument('--camera-id', default='ship1-cam0-imx708-forward-v1')
    parser.add_argument('--group', help='defaults to the capture id; two recordings must differ')
    parser.add_argument('--world-frame-id', default='dwm-anchor-frame-A01-A09-yildiz-2026-09',
                        help='declared name of the shared UWB anchor frame both rigs resolve against')
    parser.add_argument('--train-fraction', type=float, default=0.6)
    parser.add_argument('--max-state-gap-s', type=float, default=0.5)
    parser.add_argument('--observer-heading-deg', type=float, default=0.0)
    parser.add_argument('--observer-lever-frd-m', default='0,0,0')
    parser.add_argument('--target-lever-frd-m', default='0,0,0')
    args = parser.parse_args()

    folder = args.folder
    with (folder / 'observations.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        header, rows = reader.fieldnames, list(reader)
    if len(rows) < 4:
        raise SystemExit(f'{folder}/observations.csv: {len(rows)} rows is too few to split')

    train, val = split_rows(rows, args.train_fraction)
    write_csv(folder / 'train_observations.csv', train, header)
    write_csv(folder / 'val_observations.csv', val, header)

    group = args.group or args.capture_id
    lever = lambda text: [float(v) for v in text.split(',')]
    sessions = []
    for split, part in (('train', train), ('val', val)):
        stamps = [float(r['timestamp']) for r in part]
        sessions.append({
            'id': f'{args.capture_id}-{split}', 'group': group,
            'capture_id': args.capture_id, 'split': split,
            'camera_id': args.camera_id,
            'observations': f'{split}_observations.csv',
            'bearing_source': 'logged_visual_bearing',
            'observer': {'rig_id': 'ship1', 'path': 'observer.csv',
                         'clock': {'offset_s': 0.0, 'scale': 1.0},
                         'reference_lever_frd_m': lever(args.observer_lever_frd_m)},
            'target': {'rig_id': 'ship2', 'path': 'target.csv',
                       'clock': {'offset_s': 0.0, 'scale': 1.0},
                       'reference_lever_frd_m': lever(args.target_lever_frd_m)},
            # Half-open [start, end): the epsilon keeps the final frame inside.
            'associations': [{'track_id': '1', 'start_s': stamps[0], 'end_s': stamps[-1] + 1e-3}]})

    manifest = {'format': 'two-rig-supervision-v1', 'world_frame': 'shared_local_xyz_z_up',
                'world_frame_id': args.world_frame_id, 'synthetic': False,
                'max_state_gap_s': args.max_state_gap_s, 'sessions': sessions}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'manifest': str(folder / 'manifest.json'),
                      'train_rows': len(train), 'val_rows': len(val),
                      'train_span_s': round(float(train[-1]['timestamp']) - float(train[0]['timestamp']), 1),
                      'val_span_s': round(float(val[-1]['timestamp']) - float(val[0]['timestamp']), 1),
                      'declared_heading_deg': args.observer_heading_deg,
                      'note': 'One capture split in time; a second recording in another --group '
                              'is what would test generalisation.'}, indent=2))


if __name__ == '__main__':
    main()
