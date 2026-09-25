"""Two-stage detection ablation: full frame vs. region proposals, on labeled images.

Scores detector output against existing YOLO labels (IoU >= 0.5), plus the
proposer's own ceiling: the share of labeled boxes retaining at least half
their area inside the proposed band (the maximum achievable IoU is >= 0.5).

Labeled images are the 640x360 annotation copies, so this measures the effect
of the search region at training resolution, not the resolution gain a band
cut from a native 4608x2592 frame would add. Never modifies labels.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from boatdet.config import RoiConfig
from boatdet.dataset import parse_box_line
from boatdet.detection import YoloDetector
from boatdet.proposal import HorizonProposer, TwoStageDetector, build_proposer
from evaluate_inputs import summarize

ARMS = {  # name: (roi mode, tile grid)
    'full': ('none', None),
    'band': ('band', None),
    'horizon': ('horizon', None),
    'band_tiles': ('band', (1, 2)),
    'horizon_tiles': ('horizon', (1, 2)),
    'full_tiles': ('none', (2, 2)),
}


def recording(stem):
    return stem.rsplit('_', 1)[0]


def load_split(images, labels, limit=None):
    """[(stem, image path, normalized truth boxes)], skipping invalid labels."""
    items, excluded = [], []
    for image in sorted(Path(images).glob('*.jpg'))[:limit]:
        label = Path(labels) / f'{image.stem}.txt'
        if not label.exists():
            raise ValueError(f'missing label: {label}')
        try:
            truth = [parse_box_line(line) for line in label.read_text().splitlines() if line.strip()]
        except ValueError as exc:
            excluded.append(dict(image=image.name, reason=str(exc)))
            continue
        items.append((image.stem, image, truth))
    return items, excluded


def band_coverage(truth, band, height):
    """Boxes with enough retained area for a crop-contained prediction to reach IoU 0.5."""
    if band is None:
        return len(truth)
    top, bottom = band[0] / height, band[1] / height
    return sum(max(0, min(box[3], bottom) - max(box[1], top)) / (box[3] - box[1]) >= .5
               for box in truth)


def run_arm(model, items, mode, tile_grid, imgsz, roi_config):
    detector = YoloDetector(model, conf=.15, imgsz=imgsz, tile_grid=tile_grid)
    proposer = build_proposer(RoiConfig(**{**roi_config.__dict__, 'mode': mode}))
    detector = TwoStageDetector(proposer, detector) if proposer else detector
    rows, covered, fitted = [], 0, 0
    previous_recording = None
    for stem, path, truth in items:
        current_recording = recording(stem)
        if isinstance(proposer, HorizonProposer) and current_recording != previous_recording:
            proposer.line = None
        previous_recording = current_recording
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'cannot decode image: {path}')
        height, width = frame.shape[:2]
        start = time.perf_counter()
        candidates = detector.candidates(frame, (width, height))
        elapsed = (time.perf_counter() - start) * 1000
        band = getattr(detector, 'last_band', None)
        covered += band_coverage(truth, band, height)
        if isinstance(proposer, HorizonProposer):
            fitted += proposer.last_fit_ok
        predictions = [[x1 / width, y1 / height, x2 / width, y2 / height, confidence]
                       for (x1, y1, x2, y2), confidence in candidates]
        rows.append(dict(frame=stem, truth=truth, predictions=predictions, predict_ms=elapsed))
    return rows, covered, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('yolo_dataset_v4_color/images/val'))
    parser.add_argument('--labels', type=Path, default=Path('yolo_dataset_v4_color/labels/val'))
    parser.add_argument('--weights', required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--band', default=','.join(map(str, RoiConfig.band)), help='TOP,BOTTOM fractions')
    parser.add_argument('--limit', type=int, help='first N images, for a quick check')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    roi_config = RoiConfig(band=tuple(float(v) for v in args.band.split(',')))
    items, excluded = load_split(args.images, args.labels, args.limit)
    if not items:
        parser.error('no labeled images found')
    args.out.mkdir(parents=True, exist_ok=True)
    boxes = sum(len(t) for _, _, t in items)
    model = YOLO(args.weights)
    report = dict(images=len(items), positive_images=sum(bool(t) for _, _, t in items), boxes=boxes,
                  recordings=len({recording(s) for s, _, _ in items}), excluded=excluded,
                  weights=args.weights, imgsz=args.imgsz, band=roi_config.band, results=[],
                  limitations='Existing validation labels, not an independent reviewed test set. '
                              'Frames within a recording are correlated. 640x360 annotation copies, '
                              'so the band adds no resolution; no retraining.')
    for arm in args.arms:
        mode, tile_grid = ARMS[arm]
        rows, covered, fitted = run_arm(model, items, mode, tile_grid, args.imgsz, roi_config)
        per_recording = defaultdict(list)
        for row in rows:
            per_recording[recording(row['frame'])].append(row)
        entry = dict(arm=arm, roi=mode, tile_grid=tile_grid,
                     band_ceiling=covered / boxes if boxes else None,
                     horizon_fit_rate=fitted / len(items) if mode == 'horizon' else None,
                     metrics=[summarize(rows, threshold) for threshold in (.15, .45)],
                     recall_by_recording={name: round(summarize(r, .45)['recall'], 3)
                                          for name, r in sorted(per_recording.items())})
        report['results'].append(entry)
        (args.out / f'{arm}.json').write_text(json.dumps(rows) + '\n')
        (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
        strong = entry['metrics'][1]
        print(f"{arm:14s} ceiling={entry['band_ceiling']:.3f} tp={strong['tp']} fp={strong['fp']} "
              f"fn={strong['fn']} P={strong['precision']:.3f} R={strong['recall']:.3f} F1={strong['f1']:.3f} "
              f"{strong['median_predict_ms']:.0f} ms" +
              (f" horizon_fit={entry['horizon_fit_rate']:.2f}" if entry['horizon_fit_rate'] is not None else ''),
              flush=True)


if __name__ == '__main__':
    main()
