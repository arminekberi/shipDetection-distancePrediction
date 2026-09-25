"""Paired boat-detector input ablation on a recording with existing YOLO labels.

Measures detector outputs, not tracking, tag decoding or distance accuracy.
Never modifies labels. Invalid labels are excluded from every comparison arm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from boatdet.dataset import parse_box_line
from boatdet.detection import apply_clahe
from boatdet.video import read_frames, resize_working


def read_boxes(path):
    return [parse_box_line(line) for line in path.read_text().splitlines() if line.strip()]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def score(predictions, truth, threshold):
    used, overlaps = set(), []
    predictions = sorted((p for p in predictions if p[4] >= threshold), key=lambda p: -p[4])
    for p in predictions:
        candidates = [(iou(p, b), j) for j, b in enumerate(truth) if j not in used]
        overlap, j = max(candidates, default=(0, -1))
        if overlap >= .5:
            used.add(j)
            overlaps.append(overlap)
    return dict(tp=len(used), fp=len(predictions)-len(used), fn=len(truth)-len(used),
                matched_iou=overlaps)


def summarize(rows, threshold):
    scores = [score(r['predictions'], r['truth'], threshold) for r in rows]
    tp, fp, fn = (sum(s[k] for s in scores) for k in ('tp', 'fp', 'fn'))
    overlaps = [v for s in scores for v in s['matched_iou']]
    negative = [(r, s) for r, s in zip(rows, scores) if not r['truth']]
    return dict(confidence_threshold=threshold, iou_threshold=.5, tp=tp, fp=fp, fn=fn,
                precision=tp/(tp+fp) if tp+fp else 0, recall=tp/(tp+fn) if tp+fn else 0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                matched_mean_iou=float(np.mean(overlaps)) if overlaps else None,
                negative_frames=len(negative), negative_frames_with_fp=sum(s['fp'] > 0 for _, s in negative),
                median_predict_ms=float(np.median([r['predict_ms'] for r in rows])))


def paired_frames(video, selected):
    """Decode incrementally to keep long evaluations bounded in memory."""
    last = -1
    for index, frame in read_frames(video):
        last = index
        if index in selected:
            yield index, frame, selected[index]
    if selected and last < max(selected):
        raise ValueError('Video ended before all selected frames were decoded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Saved annotation images for frame alignment checks')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--weights', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['resized_color', 'native_color', 'native_color_clahe'],
                        default=['resized_color', 'native_color'])
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    frames, excluded, alignment = {}, [], []
    labels_digest = hashlib.sha256()
    for index, frame in read_frames(args.video):
        stem = f'{args.prefix}_{index:05d}'
        label = args.labels / f'{stem}.txt'
        if not label.exists():
            continue
        labels_digest.update(label.name.encode() + b'\0' + label.read_bytes() + b'\0')
        saved = cv2.imread(str(args.images / f'{stem}.jpg'))
        if saved is None:
            raise ValueError(f'Missing alignment image: {stem}')
        small = resize_working(frame)
        if saved.shape != small.shape:
            raise ValueError(f'Unexpected annotation image shape: {stem}')
        mae = float(np.abs(small.astype(np.float32) - saved.astype(np.float32)).mean())
        if mae > 5:
            raise ValueError(f'Frame alignment check failed: {stem}, MAE={mae}')
        alignment.append(mae)
        try:
            frames[index] = read_boxes(label)
        except ValueError as exc:
            excluded.append(dict(frame=index, label=label.name, reason=str(exc)))
    if not frames:
        raise ValueError('No valid paired frames')
    report = dict(video=args.video, video_sha256=hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
                  labels_sha256=labels_digest.hexdigest(), frames=len(frames), positive_frames=sum(bool(t) for t in frames.values()), excluded=excluded,
                  alignment_mae_median=float(np.median(alignment)), alignment_mae_max=max(alignment),
                  settings=dict(imgsz=args.imgsz, nms_iou=.7, device='cpu', fp16=False, threads=args.threads,
                                clahe=dict(clip_limit=2.5, tile_grid=8, channel='LAB L'),
                                ultralytics=ultralytics.__version__, torch=torch.__version__), results=[],
                  limitations='Existing validation annotations, not an independent reviewed test set. '
                  'One recording; neighboring frames are correlated. No retraining. '
                  'This isolates input preprocessing, not the complete tracker or depth pipeline. '
                  'The archive is 1280x720, below the original camera sensor resolution.')
    for weights in args.weights:
        model = YOLO(weights)
        digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        for mode in args.modes:
            rows = []
            for number, (index, frame, truth) in enumerate(paired_frames(args.video, frames)):
                source = resize_working(frame) if mode == 'resized_color' else frame
                if mode == 'native_color_clahe':
                    source = apply_clahe(source)
                if number == 0:
                    model.predict(source, imgsz=args.imgsz, conf=.15, iou=.7, device='cpu', verbose=False)
                start = time.perf_counter()
                result = model.predict(source, imgsz=args.imgsz, conf=.15, iou=.7, device='cpu', verbose=False)[0]
                elapsed = (time.perf_counter()-start)*1000
                preds = [[*b, float(c)] for b, c, cls in zip(result.boxes.xyxyn.cpu().tolist(),
                         result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()) if cls == 0]
                rows.append(dict(frame=index, truth=truth, predictions=preds, predict_ms=elapsed))
            name = f'{Path(weights).stem}_{mode}'
            (args.out / f'{name}.json').write_text(json.dumps(rows, indent=2)+'\n')
            entry = dict(weights=weights, weights_sha256=digest, mode=mode,
                         metrics=[summarize(rows, threshold) for threshold in (.15, .45)])
            report['results'].append(entry)
            (args.out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
