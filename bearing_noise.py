"""Measure the bearing jitter of a tracked reference point on real recordings.

`TmaConfig.pixel_sigma_px` decides how much the bearings-only estimator trusts
every bearing, and a guess there is a guess about the whole result. This reads
the track CSVs `run.py --track-log` already writes and measures how much the
reference point wobbles around its own smooth motion: a short window is fitted
with a quadratic, and the residual of its center sample is the jitter. The fit
removes genuine target motion; what is left is detector and platform noise.

What the number is and is not:

* an **upper bound** on detector jitter — real high-frequency platform motion
  and true target yaw are in the residual too;
* a **lower bound** on the operational bearing error — heading, mounting and
  calibration errors are not in this recording at all, and they add in quadrature
  (`boatdet.tma.bearing_sigma`).

Coasted rows (`missed_frames > 0`) are predictions, not measurements, and are
excluded: they are smooth by construction and would flatter the result.

Two modes, and the difference between them is not cosmetic. Track CSVs publish a
Kalman-filtered center, so what they yield is the *residual* jitter the tracker
did not absorb — a floor. `--video` runs the detector frame by frame and chains
raw detections with no filter at all, which is the detector's own jitter.

    venv/bin/python bearing_noise.py results/multitrack --focal-px 600
    venv/bin/python bearing_noise.py --video capture.mp4 --weights weights/boat_v4_s_best.pt
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REFERENCES = {'box_center': lambda row: (row['x1'] + row['x2']) / 2.0,
              'water_contact': lambda row: row['water_contact_u']}


def read_tracks(path):
    """{track_id: [rows]} of measured (not coasted) rows, in time order."""
    tracks = defaultdict(list)
    with Path(path).open(newline='') as handle:
        for raw in csv.DictReader(handle):
            if int(raw.get('missed_frames') or 0) > 0:
                continue
            row = {'timestamp': float(raw['timestamp'])}
            for key in ('x1', 'x2', 'water_contact_u'):
                row[key] = float(raw[key]) if raw.get(key) not in (None, '') else None
            if row['x1'] is None or row['x2'] is None:
                continue
            tracks[raw['track_id']].append(row)
    return {track_id: sorted(rows, key=lambda r: r['timestamp'])
            for track_id, rows in tracks.items()}


def residuals(times, values, window, degree, max_gap_s):
    """Residual of each window's center sample against a local polynomial fit.

    The residual is scaled by its leverage, so fitting the sample that is being
    tested does not quietly shrink the measured noise.
    """
    out = []
    half = window // 2
    for center in range(half, len(values) - half):
        span = slice(center - half, center + half + 1)
        window_times, window_values = times[span], values[span]
        if np.diff(window_times).max() > max_gap_s:
            continue  # a gap in the track is not a smooth stretch to fit across
        design = np.vander(window_times - window_times[half], degree + 1)
        try:
            pseudo = np.linalg.pinv(design)
        except np.linalg.LinAlgError:
            continue
        leverage = float(design[half] @ pseudo[:, half])
        if leverage >= 1.0:
            continue
        fitted = design @ (pseudo @ window_values)
        out.append((window_values[half] - fitted[half]) / math.sqrt(1.0 - leverage))
    return out


def measure(paths, window=9, degree=2, min_samples=30, max_gap_s=0.5):
    """Per-file, per-reference jitter in pixels, plus the pooled residuals."""
    rows, pooled = [], defaultdict(list)
    for path in paths:
        tracks = read_tracks(path)
        for name, extract in REFERENCES.items():
            values_by_track = {}
            for track_id, track in tracks.items():
                series = [(row['timestamp'], extract(row)) for row in track if extract(row) is not None]
                if len(series) < max(min_samples, window):
                    continue
                times = np.array([t for t, _ in series])
                values = np.array([v for _, v in series])
                found = residuals(times, values, window, degree, max_gap_s)
                if found:
                    values_by_track[track_id] = found
            if not values_by_track:
                continue
            everything = [r for found in values_by_track.values() for r in found]
            pooled[name] += everything
            rows.append({'recording': Path(path).parent.name, 'reference': name,
                         'tracks': len(values_by_track), 'samples': len(everything),
                         'rms_px': float(np.sqrt(np.mean(np.square(everything)))),
                         'p90_px': float(np.percentile(np.abs(everything), 90))})
    return rows, pooled


def chain_detections(frames, max_jump_px, max_miss=3):
    """Group per-frame detections into chains by nearest center, no filtering.

    A deliberately plain association: the point is to measure raw detector jitter,
    and any filter here would remove the very quantity being measured. A chain
    survives `max_miss` frames without a detection — the gap is a hole in the
    series, never an invented sample, and `residuals` drops any window that spans
    a real time gap.
    """
    chains, open_chains = [], []
    for frame_index, (timestamp, boxes) in enumerate(frames):
        centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, box[3]) for box in boxes]
        pairs = sorted(((math.dist(chain['samples'][-1][1:3], center[:2]), index, chain_index)
                        for chain_index, chain in enumerate(open_chains)
                        for index, center in enumerate(centers)),
                       key=lambda item: item[0])
        used_detections, used_chains = set(), set()
        for distance, index, chain_index in pairs:
            if distance > max_jump_px or index in used_detections or chain_index in used_chains:
                continue
            used_detections.add(index)
            used_chains.add(chain_index)
            open_chains[chain_index]['samples'].append((timestamp, *centers[index]))
            open_chains[chain_index]['last'] = frame_index
        still_open = []
        for chain_index, chain in enumerate(open_chains):
            (still_open if frame_index - chain['last'] <= max_miss else chains).append(chain)
        open_chains = still_open
        for index, center in enumerate(centers):
            if index not in used_detections:
                open_chains.append({'samples': [(timestamp, *center)], 'last': frame_index})
    return [chain['samples'] for chain in chains + open_chains]


def measure_video(path, detector, working_size, max_frames, max_jump_px, window, degree,
                  min_samples, max_gap_s, max_miss=3):
    """Raw per-frame detector jitter on a recording, with no tracker in the way.

    Boxes are mapped to `working_size` before anything is measured, because that is
    the frame `--tma-pixel-sigma` is quoted in: a jitter measured on native pixels
    would overstate it by the working scale factor.
    """
    from boatdet.video import open_video, read_frames

    capture = open_video(path)
    fps = capture.get(5) or 1.0
    native = (int(capture.get(3)), int(capture.get(4)))
    capture.release()
    frames = [(index / fps, [box for box, _ in detector.candidates(frame, working_size)])
              for index, frame in read_frames(path, max_frames=max_frames)]
    rows, pooled = [], defaultdict(list)
    chains = [chain for chain in chain_detections(frames, max_jump_px, max_miss)
              if len(chain) >= max(min_samples, window)]
    for name, index in (('raw_box_center', 1), ('raw_box_bottom_v', 3)):
        found_all = []
        for chain in chains:
            times = np.array([sample[0] for sample in chain])
            values = np.array([sample[index] for sample in chain])
            found_all += residuals(times, values, window, degree, max_gap_s)
        if not found_all:
            continue
        pooled[name] += found_all
        rows.append({'recording': f'{Path(path).parent.name}/{Path(path).stem}',
                     'reference': name, 'tracks': len(chains),
                     'samples': len(found_all),
                     'rms_px': float(np.sqrt(np.mean(np.square(found_all)))),
                     'p90_px': float(np.percentile(np.abs(found_all), 90))})
    return rows, pooled, len(frames), native


def robust_sigma(values):
    """Median absolute deviation scaled to a Gaussian sigma; outliers do not set it."""
    values = np.asarray(values)
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='*', help='track CSVs, or directories searched for tracks.csv')
    parser.add_argument('--video', nargs='+', help='measure raw detector jitter on these recordings instead')
    parser.add_argument('--weights', help='detector checkpoint for --video')
    parser.add_argument('--imgsz', type=int, default=960, help='detector inference size for --video')
    parser.add_argument('--conf', type=float, default=0.25, help='detector confidence for --video')
    parser.add_argument('--working', default='640x360',
                        help='working frame the jitter is reported in, as run.py uses it')
    parser.add_argument('--max-frames', type=int, help='stop after N frames of --video')
    parser.add_argument('--max-jump-px', type=float, default=60.0,
                        help='furthest a detection may move between frames and still continue a chain')
    parser.add_argument('--max-miss', type=int, default=3,
                        help='frames a chain survives without a detection; the gap stays a gap')
    parser.add_argument('--focal-px', type=float,
                        help='focal length in working pixels; converts the jitter to degrees')
    parser.add_argument('--window', type=int, default=9, help='samples per local fit')
    parser.add_argument('--degree', type=int, default=2, help='polynomial degree of the local fit')
    parser.add_argument('--min-samples', type=int, default=30, help='shortest track measured')
    parser.add_argument('--max-gap-s', type=float, default=0.5,
                        help='a window containing a longer gap is skipped')
    parser.add_argument('--csv', help='write the per-recording table here')
    args = parser.parse_args()

    if args.video:
        if not args.weights:
            parser.error('--video needs --weights')
        try:
            working = tuple(int(v) for v in str(args.working).lower().split('x'))
            if len(working) != 2 or min(working) < 2:
                raise ValueError
        except ValueError:
            parser.error('--working takes WIDTHxHEIGHT')
        from ultralytics import YOLO
        from boatdet.detection import YoloDetector
        detector = YoloDetector(YOLO(args.weights), conf=args.conf, imgsz=args.imgsz)
        rows, pooled = [], defaultdict(list)
        for video in args.video:
            found, found_pooled, frames, native = measure_video(
                video, detector, working, args.max_frames, args.max_jump_px,
                args.window, args.degree, args.min_samples, args.max_gap_s, args.max_miss)
            rows += found
            for name, values in found_pooled.items():
                pooled[name] += values
            print(f'{video}: {frames} frames, {native[0]}x{native[1]} decoded, '
                  f'measured in {working[0]}x{working[1]}, raw detections, no tracker')
        print()
    else:
        paths = []
        for entry in (Path(p) for p in args.paths):
            paths += sorted(entry.rglob('tracks.csv')) if entry.is_dir() else [entry]
        if not paths:
            parser.error('no track CSVs found; pass paths or use --video')
        rows, pooled = measure(paths, args.window, args.degree, args.min_samples, args.max_gap_s)
        print('Track CSVs publish a Kalman-filtered center: these are floors, not the '
              'detector\'s own jitter.\nUse --video for that.\n')
    if not rows:
        parser.error('nothing was long enough to measure; lower --min-samples or check the inputs')

    columns = ('recording', 'reference', 'tracks', 'samples', 'rms_px', 'p90_px')
    print('  '.join(f'{name:>30}' if name == 'recording' else f'{name:>14}' for name in columns))
    for row in sorted(rows, key=lambda r: (r['recording'], r['reference'])):
        print('  '.join(f'{row[name]:>30}' if name == 'recording' else
                        (f'{row[name]:>14.2f}' if isinstance(row[name], float) else f'{row[name]:>14}')
                        for name in columns))

    print('\npooled over every recording')
    for name, found in pooled.items():
        rms = float(np.sqrt(np.mean(np.square(found))))
        # Both are reported because they disagree when the residuals have heavy tails,
        # and that disagreement is itself the finding: rare association or box failures
        # dominate the RMS while most frames are quiet.
        line = (f'  {name:>14}: {len(found):6d} samples  rms {rms:6.2f} px  '
                f'robust {robust_sigma(found):5.2f} px  p90 {np.percentile(np.abs(found), 90):6.2f} px')
        if args.focal_px:
            line += f'  =  {math.degrees(math.atan(rms / args.focal_px)):.3f} deg'
        print(line)
    if args.focal_px:
        print('\nThis is the detector term only. Add heading and mounting uncertainty in quadrature\n'
              '(boatdet.tma.bearing_sigma) before using it as --tma-pixel-sigma.')

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nwrote {len(rows)} rows to {path}')


if __name__ == '__main__':
    main()
