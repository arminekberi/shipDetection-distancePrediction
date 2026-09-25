"""Track one vessel through a rig recording and write its bearing per frame.

The seed box is given rather than detected, because `boat_v4_s_best` does not
find the target in this basin: on measured rig frames it returns nothing at the
real target and fires on the floating boom markers instead, whose yellow-and-dark
banding looks like the target's own yellow hull. Until a detector trained on this
footage exists, a human points at the target once and CSRT carries it.

Bearings come from hydrolink's `ship_camera_processing.bearing` intrinsics rather
than a local pinhole model: undistortion is worth up to 1.8 deg of azimuth at the
frame edge on this ~99 deg lens, which is two orders of magnitude past the pixel
grid's own limit. The angles are relative to the CAMERA's optical axis. They are
not ship-relative and not true: the hub->body extrinsics on this rig contradict
what the cameras physically do, and no time-aligned heading exists.

A hull floating on water sits within a few degrees of the horizon, so a frame
whose elevation leaves that band is CSRT having walked off onto the hall - the
`--max-elevation` gate drops those instead of reporting a confident wrong angle.

    venv/bin/python track_target.py --session 20260917-132159 \
        --seed 506,287,605,344 --out results/track_132159
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

HYDROLINK = Path(os.environ.get('HYDROLINK_DIR') or Path.home() / 'Documents' / 'hydrolink')
RIG_EXPORT = r'''
import glob, os, re, numpy as np, cv2, json
fs = glob.glob("/root/shipcaps/{session}/*/*.bgr")
key = lambda p: int(re.search(r"_(\d+)_", os.path.basename(p)).group(1))
fs = sorted(fs, key=key)
os.system("rm -rf {tmp}"); os.makedirs("{tmp}")
stamps = []
for i, p in enumerate(fs):
    a = np.fromfile(p, dtype=np.uint8).reshape({h}, {w}, 3)
    cv2.imwrite("{tmp}/%05d.jpg" % i,
                cv2.resize(a, ({ow}, {oh}), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    stamps.append(int(re.search(r"_(\d+)\.bgr", os.path.basename(p)).group(1)))
open("{tmp}/stamps.json", "w").write(json.dumps(stamps))
print(len(fs))
'''


def load_intrinsics(camera):
    """hydrolink's own loader, so the calibrated numbers have one reader."""
    sys.path.insert(0, str(HYDROLINK))
    from ship_camera_processing.bearing import Intrinsics
    return Intrinsics.load(HYDROLINK / 'ship_camera_processing' / 'calib' / f'{camera}_intrinsics.yaml')


def export_frames(rig, session, out, width, height, key, native=(4608, 2592)):
    """Downscale on the rig and copy only the small frames; raw stays there."""
    out.mkdir(parents=True, exist_ok=True)
    script = RIG_EXPORT.format(session=session, tmp='/tmp/track_export', h=native[1],
                               w=native[0], ow=width, oh=height)
    ssh = ['ssh', '-i', str(key), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', f'root@{rig}']
    count = subprocess.run(ssh + ['python3', '-'], input=script, text=True,
                           capture_output=True, check=True).stdout.strip().splitlines()[-1]
    subprocess.run(['scp', '-i', str(key), '-o', 'BatchMode=yes',
                    f'root@{rig}:/tmp/track_export/*', str(out)],
                   check=True, capture_output=True)
    return int(count)


def bearings(intrinsics, centre):
    """Undistorted azimuth (starboard positive) and elevation (up positive), degrees."""
    point = np.array([[list(centre)]], dtype=float)
    undistorted = cv2.undistortPoints(point, intrinsics.K, intrinsics.dist)
    x, y = float(undistorted[0, 0, 0]), float(undistorted[0, 0, 1])
    return (math.degrees(math.atan2(x, 1.0)),
            math.degrees(math.atan2(-y, math.hypot(x, 1.0))))


def track(frames, stamps, seed, intrinsics, max_elevation):
    """CSRT from the seed box; rows carry a reason when a frame yields nothing."""
    scale = intrinsics.width / cv2.imread(str(frames[0])).shape[1]
    tracker = cv2.TrackerCSRT.create()
    x1, y1, x2, y2 = seed
    tracker.init(cv2.imread(str(frames[0])), (x1, y1, x2 - x1, y2 - y1))
    rows = []
    for index, path in enumerate(frames):
        image = cv2.imread(str(path))
        if index == 0:
            ok, box = True, (x1, y1, x2 - x1, y2 - y1)
        else:
            ok, box = tracker.update(image)
        if not ok:
            rows.append({'frame': index, 'ts_ms': stamps[index], 'status': 'lost'})
            continue
        centre = ((box[0] + box[2] / 2) * scale, (box[1] + box[3] / 2) * scale)
        azimuth, elevation = bearings(intrinsics, centre)
        # The waterline, not the box centre, is what carries range: a hull floats at
        # z=0, so its depression below the horizon is a function of distance alone.
        # The box centre rides up and down with superstructure and is useless for it.
        water = ((box[0] + box[2] / 2) * scale, (box[1] + box[3]) * scale)
        _, water_elevation = bearings(intrinsics, water)
        status = 'ok' if abs(elevation) <= max_elevation else 'off_water'
        rows.append({'frame': index, 'ts_ms': stamps[index], 'status': status,
                     'u_px': centre[0], 'v_px': centre[1],
                     'az_deg': azimuth, 'el_deg': elevation,
                     'el_water_deg': water_elevation,
                     'box_w_px': box[2] * scale, 'box_h_px': box[3] * scale,
                     'box': [float(v) for v in box]})
    return rows


def montage(frames, rows, path, columns=3, tile=(426, 240)):
    """Six evenly spaced tracked frames, so the track is checked by eye, not assumed."""
    from PIL import Image, ImageDraw
    usable = [r for r in rows if r['status'] == 'ok']
    if not usable:
        return None
    picks = [usable[round(k * (len(usable) - 1) / 5)] for k in range(6)]
    width, height = tile
    sheet = Image.new('RGB', (width * columns, height * 2), 'black')
    draw = ImageDraw.Draw(sheet)
    start = rows[0]['ts_ms']
    for k, row in enumerate(picks):
        image = Image.open(frames[row['frame']])
        box = row['box']
        ImageDraw.Draw(image).rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                                        outline='lime', width=4)
        x, y = (k % columns) * width, (k // columns) * height
        sheet.paste(image.resize((width, height), Image.LANCZOS), (x, y))
        draw.text((x + 6, y + 6), f"t={(row['ts_ms'] - start) / 1000:4.1f}s "
                                  f"az={row['az_deg']:+.1f} el={row['el_deg']:+.1f}", fill='yellow')
    sheet.save(path, quality=93)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', help='rig session id; frames are exported from the rig')
    parser.add_argument('--frames', type=Path, help='local directory of already exported frames')
    parser.add_argument('--seed', required=True, help='x1,y1,x2,y2 of the target in the first frame')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rig', default=os.environ.get('RIG') or '192.168.1.104')
    parser.add_argument('--camera', default='cam0', help='cam0 is the forward camera on ship 1')
    parser.add_argument('--key', type=Path,
                        default=os.environ.get('RIG_KEY') or Path.home() / '.ssh' / 'id_ed25519_shipcam')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--max-elevation', type=float, default=8.0,
                        help='degrees; beyond this the track has left the water')
    args = parser.parse_args()
    if not (args.session or args.frames):
        parser.error('give --session or --frames')

    args.out.mkdir(parents=True, exist_ok=True)
    directory = args.frames or (args.out / 'frames')
    if args.session:
        count = export_frames(args.rig, args.session, directory, args.width, args.height, args.key)
        print(f'exported {count} frames from {args.session}')
    frames = sorted(directory.glob('*.jpg'))
    if not frames:
        raise SystemExit(f'no frames in {directory}')
    stamps = json.loads((directory / 'stamps.json').read_text())

    intrinsics = load_intrinsics(args.camera)
    seed = [int(v) for v in args.seed.split(',')]
    rows = track(frames, stamps, seed, intrinsics, args.max_elevation)
    good = [r for r in rows if r['status'] == 'ok']

    with (args.out / 'bearings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['frame', 'ts_ms', 't_s', 'status', 'u_px', 'v_px', 'az_deg', 'el_deg',
                         'el_water_deg', 'box_w_px', 'box_h_px'])
        for row in rows:
            number = lambda key, digits=3: f"{row[key]:.{digits}f}" if key in row else ''
            writer.writerow([row['frame'], row['ts_ms'], f"{(row['ts_ms'] - rows[0]['ts_ms']) / 1000:.3f}",
                             row['status'], number('u_px', 1), number('v_px', 1),
                             number('az_deg'), number('el_deg'), number('el_water_deg'),
                             number('box_w_px', 1), number('box_h_px', 1)])
    picture = montage(frames, rows, args.out / 'track_montage.jpg')
    summary = {'session': args.session, 'camera': args.camera, 'frames': len(rows),
               'tracked': len(good), 'lost': sum(1 for r in rows if r['status'] == 'lost'),
               'off_water': sum(1 for r in rows if r['status'] == 'off_water'),
               'azimuth_deg': [round(min(r['az_deg'] for r in good), 2),
                               round(max(r['az_deg'] for r in good), 2)] if good else None,
               'sweep_deg': round(max(r['az_deg'] for r in good) - min(r['az_deg'] for r in good), 2) if good else None,
               'bearing_source': 'camera optical axis, undistorted; NOT ship-relative or true'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if picture:
        print(f'check the track by eye: {picture}')


if __name__ == '__main__':
    main()
