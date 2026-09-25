"""Benchmark Depth Anything V2 variants on laser-measured calibration clips.

Target per clip: CSRT seeded from its verified frame-0 label. Per model: median
depth per clip, linear fit to the laser distances, MAE/RMSE of the fit, latency.
Four points: the errors are training residuals, not an independent accuracy estimate.
"""
import argparse
import csv
import statistics as st
from pathlib import Path

import cv2
import numpy as np
import torch

from boatdet.depth import DepthEstimator, distance_in_box, fit_calibration
from boatdet.device import get_device
from boatdet.tracking import ManualCsrtTracker
from boatdet.video import WORKING_SIZE, h264_writer, read_frames, resize_working

OUT_DIR = Path('results/depth_benchmark')
SEED_LABELS = Path('yolo_dataset/labels')
MODELS = {name: f'depth-anything/Depth-Anything-V2-Metric-{name}-hf'
          for name in ('Indoor-Small', 'Indoor-Base', 'Indoor-Large', 'Outdoor-Small', 'Outdoor-Large')}
# (video, label tag, laser distance in meters)
CLIPS = [
    ('calibrationLazerMeasurements/3,7.mp4', '3_7', 3.7),
    ('calibrationLazerMeasurements/5.mp4', '5', 5.0),
    ('calibrationLazerMeasurements/6,6m.mp4', '6_6m', 6.6),
    ('calibrationLazerMeasurements/9.mp4', '9', 9.0),
]
SUMMARY_FIELDS = ['model', 'mae_m', 'rmse_m', 'avg_pct_error', 'ms_per_frame', 'calib_scale', 'calib_offset']


def seed_box(tag):
    """Frame-0 xyxy box in working coords from the verified YOLO label."""
    width, height = WORKING_SIZE
    for split in ('train', 'val'):
        path = SEED_LABELS / split / f'{tag}_00000.txt'
        if path.exists():
            cx, cy, w, h = (float(v) for v in path.read_text().split()[1:5])
            return (round((cx - w / 2) * width), round((cy - h / 2) * height),
                    round((cx + w / 2) * width), round((cy + h / 2) * height))
    raise FileNotFoundError(f'no frame-0 label for {tag!r} in {SEED_LABELS}/{{train,val}}')


def measure_clip(video, box, estimator, samples, writer, caption):
    """Depth samples of the tracked box, evenly spread over the clip.

    CSRT sees every frame to stay continuous; only depth is subsampled.
    """
    cap = cv2.VideoCapture(video)
    sample_every = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // samples)
    cap.release()
    distances, times, lost, tracker = [], [], 0, None
    for idx, native in read_frames(video):
        frame = resize_working(native)
        if tracker is None:
            tracker = ManualCsrtTracker(frame, box)
        ok, (x, y, w, h) = tracker.update(frame)
        if not ok:
            lost += 1
            continue
        if idx % sample_every:
            continue
        depth = estimator.infer(frame)
        d = distance_in_box(depth.depth_m, (x, y, x + w, y + h))
        if d is not None:
            distances.append(d)
            times.append(depth.infer_ms)
        depth_color = depth.color
        for image in (frame, depth_color):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame, f'{caption} frame {idx}  {d or 0:.2f}m', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        writer.write(np.hstack([frame, depth_color]))
    return distances, times, lost


def benchmark(name, model_id, device, args):
    out = OUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator.load(model_id, device, WORKING_SIZE, args.infer_size)
    writer = h264_writer(out / 'visual_review.mp4', 2.0, (WORKING_SIZE[0] * 2, WORKING_SIZE[1]))
    points, all_ms = [], []  # (true_m, median_raw_m)
    try:
        for video, tag, true_m in CLIPS:
            distances, times, lost = measure_clip(video, seed_box(tag), estimator, args.samples,
                                                  writer, f'laser {tag} (true={true_m}m)')
            all_ms += times
            print(f'  {tag}: n={len(distances)} lost={lost}' +
                  (f' median={st.median(distances):.2f}m' if distances else ' no samples, skipped'))
            if distances:
                points.append((true_m, st.median(distances)))
    finally:
        writer.release()
    if len(points) < 2:
        print(f'  not enough calibration points for {name}')
        return None
    true, raw = zip(*points)
    scale, offset = fit_calibration(raw, true)
    errors = [abs(scale * r + offset - t) for t, r in points]
    with (out / 'calibration_fit.csv').open('w', newline='') as f:
        rows = csv.writer(f)
        rows.writerow(['true_m', 'raw_median_m', 'corrected_m', 'abs_error_m'])
        rows.writerows((t, r, scale * r + offset, e) for (t, r), e in zip(points, errors))
    return dict(model=name, mae_m=round(st.mean(errors), 3), rmse_m=round(st.mean(e * e for e in errors) ** 0.5, 3),
                avg_pct_error=round(st.mean(100 * e / t for e, t in zip(errors, true)), 1),
                ms_per_frame=round(st.mean(all_ms), 1) if all_ms else None,
                calib_scale=round(scale, 4), calib_offset=round(offset, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer-size', type=int, default=392, help='same for every model')
    parser.add_argument('--samples', type=int, default=20, help='depth samples per clip')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    device = get_device()
    summary = []
    for name in args.models:
        print(f'=== {name}')
        row = benchmark(name, MODELS[name], device, args)
        if row:
            summary.append(row)
        if device == 'mps':
            torch.mps.empty_cache()
    with (OUT_DIR / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['model']:15s} MAE {row['mae_m']:.3f}m  RMSE {row['rmse_m']:.3f}m  "
              f"{row['avg_pct_error']:.1f}%  {row['ms_per_frame']} ms")


if __name__ == '__main__':
    main()
