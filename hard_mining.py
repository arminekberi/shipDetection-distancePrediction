"""Hard-example mining: scan recordings for detections, import the reviewed ones.

scan   -> candidates.json + numbered review images (and per-video contact sheets)
          --min-highlight F: every box whose highlight share reaches F instead of the
          top box per frame, to collect glare candidates for review
import -> positives for correct boxes; incorrect boxes skipped unless confirmed empty
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from boatdet.dataset import (check_training_sources, label_path, recording_tag,
                             require_mounted_destination, save_annotation)
from boatdet.detection import YoloDetector
from boatdet.glare import highlight_fraction
from boatdet.video import WORKING_SIZE, read_frames, resize_working

SHEET_COLUMNS = 6
CAPTION_H = 20


def parse_indices(text):
    """'1 4 7-9' -> {1, 4, 7, 8, 9}."""
    indices = set()
    for token in text.split():
        start, _, end = token.partition('-')
        indices.update(range(int(start), int(end or start) + 1))
    return indices


def captioned(image, text):
    cell = np.zeros((image.shape[0] + CAPTION_H, image.shape[1], 3), np.uint8)
    cell[CAPTION_H:] = image
    cv2.putText(cell, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return cell


def contact_sheet(cells):
    blank = np.zeros_like(cells[0])
    cells = cells + [blank] * (-len(cells) % SHEET_COLUMNS)
    rows = [np.hstack(cells[i:i + SHEET_COLUMNS]) for i in range(0, len(cells), SHEET_COLUMNS)]
    return np.vstack(rows)


def scan(args):
    from ultralytics import YOLO
    check_training_sources(args.videos)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = YoloDetector(YOLO(args.weights), conf=args.conf, imgsz=args.imgsz)
    candidates = []
    for video in args.videos:
        cells = []
        for idx, native in read_frames(video, args.sample_every):
            found = detector.candidates(native, WORKING_SIZE)
            if not found:
                continue
            if args.min_highlight is None:
                selected = [max(found, key=lambda item: item[1]) + (None,)]
            else:
                selected = [(box, conf, share) for box, conf in found
                            if (share := highlight_fraction(native, box, WORKING_SIZE)) is not None
                            and share >= args.min_highlight]
            for box, conf, share in selected:
                number = len(candidates)
                candidates.append(dict(video=video, frame=idx, conf=conf, box=box,
                                       **({} if share is None else {'highlight_fraction': round(share, 3)})))
                frame = resize_working(native)
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                caption = f'#{number} {video} f{idx} c{conf:.2f}' + ('' if share is None else f' hl{share:.2f}')
                cv2.imwrite(str(out / f'{number:04d}.jpg'), captioned(frame, caption))
                cells.append(captioned(frame, f'#{number}'))
        if cells:
            cv2.imwrite(str(out / f'sheet_{recording_tag(video)}.jpg'), contact_sheet(cells))
        print(f'{video}: {len(cells)} candidates >= conf {args.conf}')
    (out / 'candidates.json').write_text(json.dumps(candidates, indent=1) + '\n')
    print(f'{len(candidates)} candidates -> {out}/candidates.json; review images, list wrong indices')


def import_reviewed(args):
    candidates = json.loads(Path(args.candidates).read_text())
    wrong = parse_indices(Path(args.wrong).read_text())
    negative = parse_indices(Path(args.confirmed_negatives).read_text()) if args.confirmed_negatives else set()
    if not negative <= wrong:
        raise SystemExit('confirmed negatives must be a subset of the wrong-box indices')
    verdicts = {}
    for i, c in enumerate(candidates):
        verdict = 'negative' if i in negative else 'wrong' if i in wrong else 'positive'
        verdicts.setdefault((c['video'], c['frame']), set()).add(verdict)
    # One label per frame: a glare scan may list several boxes of one frame, which
    # must then share a verdict (a frame cannot be both empty and hold a boat).
    mixed = sorted(f'{video} f{frame}' for (video, frame), v in verdicts.items()
                   if len(v) > 1 or (v == {'positive'} and sum(
                       (c['video'], c['frame']) == (video, frame) for c in candidates) > 1))
    if mixed:
        raise SystemExit('frames with several boxes need one shared verdict (all wrong or all '
                         'confirmed negative): ' + ', '.join(mixed[:10]))
    check_training_sources(c['video'] for c in candidates)
    require_mounted_destination(args.out)

    by_video = {}
    for i, c in enumerate(candidates):
        if i not in wrong or i in negative:
            by_video.setdefault(c['video'], {})[c['frame']] = None if i in wrong else c['box']
    counts = dict(positive=0, negative=0, existing=0)
    for video, needed in by_video.items():
        tag = recording_tag(video)
        found = 0
        for idx, native in read_frames(video):
            if idx not in needed:
                continue
            found += 1
            box = needed[idx]
            if label_path(args.out, args.split, tag, idx).exists():
                counts['existing'] += 1
            else:
                xywh = None if box is None else (box[0], box[1], box[2] - box[0], box[3] - box[1])
                save_annotation(args.out, args.split, tag, idx, resize_working(native), xywh)
                counts['negative' if box is None else 'positive'] += 1
            if found == len(needed):
                break
        if found < len(needed):
            raise RuntimeError(f'decoded only {found}/{len(needed)} required frames from {video}')
    print(', '.join(f'{v} {k}' for k, v in counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    s = sub.add_parser('scan', help='detect candidates for review')
    s.add_argument('videos', nargs='+')
    s.add_argument('--weights', required=True)
    s.add_argument('--conf', type=float, default=0.15)
    s.add_argument('--imgsz', type=int, default=960)
    s.add_argument('--sample-every', type=int, default=2)
    s.add_argument('--min-highlight', type=float,
                   help='collect every box with at least this highlight share (glare review), '
                        'not just the top box per frame')
    s.add_argument('--out', default='review_hardmine')
    s.set_defaults(run=scan)

    i = sub.add_parser('import', help='write reviewed candidates into the dataset')
    i.add_argument('candidates', help='candidates.json from scan')
    i.add_argument('--wrong', required=True, help='text file of incorrect-box indices, e.g. "3 7 10-12"')
    i.add_argument('--confirmed-negatives', help='subset of --wrong reviewed as containing no boat')
    i.add_argument('--out', default='yolo_dataset_v4')
    i.add_argument('--split', default='train', choices=['train', 'val'])
    i.set_defaults(run=import_reviewed)

    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
