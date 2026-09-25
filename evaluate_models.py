"""Compare boat checkpoints on exactly the same saved, labeled images.

This is a diagnostic comparison, not evidence of an independent test unless
the recording provenance excludes all training/model-selection exposure.
Writes predictions, per-recording scores, and artifact hashes; never changes data.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

from boatdet.config import RoiConfig
from evaluate_inputs import summarize
from evaluate_roi import load_split, recording, run_arm


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('yolo_dataset_v4_color'))
    parser.add_argument('--splits', nargs='+', choices=('val', 'test'), default=['val', 'test'])
    parser.add_argument('--weights', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--imgsz', type=int, default=960)
    args = parser.parse_args()
    if args.threads < 1 or args.imgsz < 32:
        parser.error('threads must be positive and imgsz at least 32')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(settings=dict(device='cpu', fp16=False, imgsz=args.imgsz, threads=args.threads,
                                conf=.15, nms_iou=.7, torch=torch.__version__,
                                ultralytics=ultralytics.__version__, opencv=cv2.__version__),
                  data=str(args.data), datasets={}, results=[],
                  limitations='Saved annotation images, correlated frames, unreviewed label completeness. '
                  'Split membership alone does not establish independence from historical training. '
                  'No retraining or checkpoint selection is performed.')
    splits = {}
    for split in args.splits:
        labels = args.data / 'labels' / split
        items, excluded = load_split(args.data / 'images' / split, labels)
        if not items or excluded:
            raise ValueError(f'{split}: empty split or invalid labels ({len(excluded)})')
        splits[split] = items
        digest = hashlib.sha256()
        for stem, path, _ in items:
            digest.update(f'{stem}\0{sha256(path)}\0{sha256(labels / (stem + ".txt"))}\n'.encode())
        report['datasets'][split] = dict(frames=len(items), boxes=sum(len(t) for _, _, t in items),
                                         sha256=digest.hexdigest())
    for weights in args.weights:
        model = YOLO(str(weights)).to('cpu')
        for split, items in splits.items():
            # Warm up outside measured rows, without carrying temporal ROI state.
            model.predict(cv2.imread(str(items[0][1])), imgsz=args.imgsz, conf=.15,
                          iou=.7, device='cpu', verbose=False)
            rows, _, _ = run_arm(model, items, 'none', None, args.imgsz, RoiConfig())
            groups = defaultdict(list)
            for row in rows:
                groups[recording(row['frame'])].append(row)
            entry = dict(weights=str(weights), weights_sha256=sha256(weights), split=split,
                         metrics=[summarize(rows, t) for t in (.15, .45)],
                         by_recording={key: dict(frames=len(group), **summarize(group, .45))
                                       for key, group in sorted(groups.items())})
            report['results'].append(entry)
            (args.out / f'{weights.stem}_{split}.json').write_text(json.dumps(rows) + '\n')
            (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            m = entry['metrics'][1]
            print(f'{weights.stem} {split}: TP={m["tp"]} FP={m["fp"]} FN={m["fn"]} '
                  f'P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
