"""Interactive boat labeling, one continuous resumable pass over sampled frames.

CSRT tracks forward from the last accepted box; without a valid box the window
waits for a decision. Needs a desktop session.

Keys: SPACE/y accept, r draw box, n no boat, d discard, b back, q/ESC quit.
"""
import argparse

import cv2

from boatdet.dataset import (SPLITS, label_path, recording_tag, require_mounted_destination,
                             save_annotation, write_dataset_yaml)
from boatdet.video import WORKING_SIZE, read_frames, resize_working

DATASET = 'yolo_dataset_v4'
DRIFT_CORNER_FRAC = 0.04  # CSRT failure mode: degenerate box pinned near (0, 0)
YELLOW = (0, 255, 255)

# Dataset plan: (video, split, sample_every, confirmed_negative).
# Whole recordings per split; laser calibration clips and TEST_VIDEOS excluded.
PLAN = [
    ('signal-2026-09-02-10-36-28-854.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150149_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150707_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150830_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-151003_ShipCam0.mp4', 'train', 1, False),
    ('test1.mp4', 'train', 1, False),
    ('testt.mp4', 'train', 2, False),
    ('testtt.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150054_ShipCam0.mp4', 'train', 1, True),
    ('teknedenTekneyeGörüntü/20260904-112014_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112149_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112600_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112749_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113010_ShipCam0.mp4', 'train', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113444_ShipCam0.mp4', 'train', 2, False),
    ('teknedenTekneyeGörüntü/20260904-113812_ShipCam0.mp4', 'train', 1, False),
    ('yeniTrain/20260902-150337_ShipCam0.mp4', 'val', 1, False),
    # 20260902-151121_ShipCam0 repeats test1.mp4; never reimport it into validation.
    ('test2.mp4', 'val', 1, False),
    ('testGerçekRenkli.mp4', 'val', 1, False),
    ('yeniTrain/20260902-150054_ShipCam1.mp4', 'train', 1, True),
    ('teknedenTekneyeGörüntü/20260904-112339_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-112926_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113234_ShipCam0.mp4', 'val', 1, False),
    ('teknedenTekneyeGörüntü/20260904-113321_ShipCam0.mp4', 'val', 1, False),
]


def print_plan(out):
    write_dataset_yaml(out)
    print(f'wrote {out}/dataset.yaml\n')
    for video, split, sample_every, negative in PLAN:
        flag = ' --negative' if negative else ''
        print(f'python label_video.py --video "{video}" --split {split} --out {out} --sample-every {sample_every}{flag}')


def tracked_box(tracker, frame):
    """CSRT box for this frame, or None when lost or drifted into the corner."""
    ok, box = tracker.update(frame)
    if not ok:
        return None
    x, y, w, h = box
    width, height = WORKING_SIZE
    drifted = (x + w / 2) / width < DRIFT_CORNER_FRAC and (y + h / 2) / height < DRIFT_CORNER_FRAC
    return None if drifted else box


def review(frames, pending, args, tag):
    window = f'{tag} [{args.split}]'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    counts = dict(accepted=0, drawn=0, negative=0, discarded=0)
    tracker = None
    i = 0
    while i < len(pending):
        idx = pending[i]
        frame = frames[idx]
        box = tracked_box(tracker, frame) if tracker else None
        if box is None:
            tracker = None

        shown = frame.copy()
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 1)
        status = 'SPACE/y=accept  r=redraw' if box is not None else 'no target - r=draw box'
        cv2.putText(shown, f'{tag} frame {idx}  ({i + 1}/{len(pending)})', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        cv2.putText(shown, status + '  n=no boat  d=discard  b=back  q=quit', (5, WORKING_SIZE[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        cv2.imshow(window, shown)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord('b'):
            tracker, i = None, max(0, i - 1)  # reseed from the revisited frame
            continue
        if key == ord('d'):
            counts['discarded'] += 1
            i += 1
            continue
        if key == ord('n'):
            save_annotation(args.out, args.split, tag, idx, frame)
            tracker = None
            counts['negative'] += 1
            i += 1
            continue
        if key == ord('r'):
            drawn = cv2.selectROI(window, frame, showCrosshair=True)
            if drawn[2] < 2 or drawn[3] < 2:
                continue  # cancelled
            box = drawn
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, tuple(int(v) for v in box))
            counts['drawn'] += 1
        elif key not in (ord(' '), ord('y')) or box is None:
            continue
        save_annotation(args.out, args.split, tag, idx, frame, box)
        counts['accepted'] += 1
        i += 1
    cv2.destroyAllWindows()
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video')
    parser.add_argument('--split', choices=SPLITS)
    parser.add_argument('--out', default=DATASET)
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--tag', help='label prefix override')
    parser.add_argument('--negative', action='store_true',
                        help='recording reviewed and confirmed empty: label every sampled frame as background')
    parser.add_argument('--plan', action='store_true', help='write dataset.yaml and print commands for PLAN')
    args = parser.parse_args()
    if args.plan:
        return print_plan(args.out)
    if not (args.video and args.split):
        parser.error('--video and --split are required')

    tag = recording_tag(args.video, args.tag)
    require_mounted_destination(args.out)
    # Kept in RAM: back navigation needs earlier frames.
    frames = {idx: resize_working(frame) for idx, frame in read_frames(args.video, args.sample_every)}
    if not frames:
        raise SystemExit(f'decoded 0 frames from {args.video}')
    pending = [idx for idx in frames if not label_path(args.out, args.split, tag, idx).exists()]
    print(f'{len(frames)} sampled frames, {len(pending)} unlabeled')
    if not pending:
        return

    if args.negative:
        for idx in pending:
            save_annotation(args.out, args.split, tag, idx, frames[idx])
        print(f'done: {len(pending)} reviewed negative frames saved')
        return

    counts = review(frames, pending, args, tag)
    print('this session: ' + ', '.join(f'{v} {k}' for k, v in counts.items()))
    print('rerun the same --video/--split to resume')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:  # dropped mount, unreadable video, bad tag
        raise SystemExit(f'{type(exc).__name__}: {exc}')
