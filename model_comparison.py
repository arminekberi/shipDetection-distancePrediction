"""
monoCameraLive - Depth Anything V2 variant comparison, DEPTH ESTIMATION ONLY.

Per the advisor's scope: boat detection (YOLO) is explicitly out of scope for this task.
This script does NOT import, run, or modify anything YOLO/dataset/training related - it
only reuses DepthWorker / distance_in_box / colorize_depth / get_device from run.py
(read-only import, run.py itself is untouched).

For the 4 laser-calibration clips, the target is located with a CSRT tracker seeded from
the exact box the user manually verified for each clip - recovered here from the frame-0
labels already sitting in yolo_dataset/labels/{train,val}/<tag>_00000.txt (an old, still-
intact artifact of the YOLO dataset-building work, read here only as a source of known-
good coordinates, not touched or regenerated).
"""
import argparse
import csv
import os
import statistics as st
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from run import DepthWorker, colorize_depth, distance_in_box, get_device

WIDTH, HEIGHT = 640, 360
OUT_DIR = 'results/model_comparison'

MODELS = [
    ('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf', 'Indoor-Small'),
    ('depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf', 'Indoor-Base'),
    ('depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf', 'Indoor-Large'),
    ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', 'Outdoor-Small'),
    ('depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf', 'Outdoor-Large'),
]

CALIB_CLIPS = [
    ('calibrationLazerMeasurements/3,7.mp4', '3_7', 3.7),
    ('calibrationLazerMeasurements/5.mp4', '5', 5.0),
    ('calibrationLazerMeasurements/6,6m.mp4', '6_6m', 6.6),
    ('calibrationLazerMeasurements/9.mp4', '9', 9.0),
]


def get_seed_box(tag):
    """Recover the frame-0 seed box (in 640x360 pixel xyxy) for a calibration clip from
    the pre-existing yolo_dataset labels (never regenerated/modified by this script)."""
    for split in ('train', 'val'):
        path = f'yolo_dataset/labels/{split}/{tag}_00000.txt'
        if os.path.exists(path):
            parts = open(path).read().split()
            cx, cy, w, h = (float(v) for v in parts[1:5])
            cx, cy, w, h = cx * WIDTH, cy * HEIGHT, w * WIDTH, h * HEIGHT
            return (round(cx - w / 2), round(cy - h / 2), round(cx + w / 2), round(cy + h / 2))
    raise FileNotFoundError(f'no frame-0 label found for tag={tag!r} in yolo_dataset/labels/{{train,val}}/')


def draw_review_frame(frame_bgr, depth_color, box, label, distance):
    """Side-by-side raw+box | depth-heatmap+box panel, with a text label identifying the
    source clip/variant and the measured distance - written into the per-model review
    video so results can be visually spot-checked, not just read off as numbers."""
    disp = frame_bgr.copy()
    dcol = depth_color.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.rectangle(dcol, (x1, y1), (x2, y2), (0, 255, 0), 1)
    dist_str = f'{distance:.2f}m' if distance is not None else '--'
    cv2.putText(disp, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(disp, dist_str, (5, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return np.hstack([disp, dcol])


def track_clip(video_path, seed_box, worker, max_frames=None, sample_every=1, writer=None, label=''):
    """CSRT-track the seed box across the whole clip (or a subsample of it), returning
    per-frame (raw_distance_m, inference_ms). Frames where CSRT loses the target are
    skipped (not treated as 0 or ignored silently - the caller sees fewer points than
    frames read, and this function reports how many)."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f'could not read {video_path}')
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    x1, y1, x2, y2 = seed_box
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

    distances = []
    infer_times = []
    idx = 0
    n_lost = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        # CSRT is updated on EVERY frame regardless of sampling, so its internal
        # appearance model stays continuous - only the (expensive) depth inference and
        # the resulting measurement are subsampled. Skipping frames in the CSRT update
        # itself (an earlier version of this script did that) starves it of the small
        # frame-to-frame motion it needs and risks losing the target on longer clips.
        ok, box = tracker.update(frame)
        if not ok:
            n_lost += 1
        elif idx % sample_every == 0:
            bx, by, bw, bh = [int(v) for v in box]
            depth_m, depth_color, infer_ms = worker.infer(frame)
            d = distance_in_box(depth_m, (bx, by, bx + bw, by + bh))
            if d is not None:
                distances.append(d)
                infer_times.append(infer_ms)
            if writer is not None:
                writer.write(draw_review_frame(frame, depth_color, (bx, by, bx + bw, by + bh),
                                                f'{label} frame {idx}', d))
        idx += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
    cap.release()
    return distances, infer_times, n_lost


def synthetic_grayscale(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def track_fixed_box(video_path, worker, box, max_frames=None, sample_every=1, to_gray=False, writer=None, label=''):
    """Same per-frame depth sampling as track_clip, but over a FIXED (non-tracked,
    non-YOLO) box - used only for the color-vs-monochrome consistency comparison, where
    we deliberately avoid any detection/tracking logic per the advisor's scope."""
    cap = cv2.VideoCapture(video_path)
    distances = []
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            if to_gray:
                frame = synthetic_grayscale(frame)
            depth_m, depth_color, _ms = worker.infer(frame)
            d = distance_in_box(depth_m, box)
            if d is not None:
                distances.append(d)
            if writer is not None:
                writer.write(draw_review_frame(frame, depth_color, box, f'{label} frame {idx}', d))
        idx += 1
    cap.release()
    return distances


def fixed_water_roi():
    """A fixed region over the pool's water surface (perspective texture, near-to-far
    depth gradient), the SAME pixel box in both testGerçekRenkli.mp4 and
    testGerçekSiyahBeyaz.mp4 (same camera framing). Chosen over a small center box after
    visually confirming the naive center box lands on the flat, saturated far wall
    (~19-20m, at/beyond the model's practical range) in both clips, which trivially gives
    near-zero variance regardless of color/monochrome input - not an informative test."""
    return (60, 190, 580, 330)


def fit_calibration(raw_vals, true_vals):
    scale, offset = np.polyfit(raw_vals, true_vals, 1)
    return float(scale), float(offset)


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def sample_every_for(video_path, target_samples):
    """Spacing that yields ~target_samples measurements evenly spread across the whole
    clip (not just its first target_samples frames), so short and long clips are all
    represented by the same NUMBER of samples rather than the same time window."""
    total = count_frames(video_path)
    return max(1, total // target_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--infer-size', type=int, default=392, help='matches run.py\'s default --infer-size, kept identical across all models for a fair comparison')
    parser.add_argument('--frames-per-video', type=int, default=20, help='evenly-spaced sample count per video (laser clips and color/mono clips alike) - NOT the whole video, to keep the 3-model comparison fast')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()
    print(f'device: {device}')

    summary_rows = []

    for model_id, short_name in MODELS:
        print(f'\n=== {short_name} ({model_id}) ===')
        model_dir = os.path.join(OUT_DIR, short_name)
        os.makedirs(model_dir, exist_ok=True)

        t_load0 = time.time()
        processor = AutoImageProcessor.from_pretrained(model_id)
        depth_model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        worker = DepthWorker(processor, depth_model, device, WIDTH, HEIGHT, args.infer_size, use_fp16=False)
        print(f'  loaded in {time.time()-t_load0:.1f}s')

        review_path = os.path.join(model_dir, 'visual_review.mp4')
        review_writer = cv2.VideoWriter(review_path, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (WIDTH * 2, HEIGHT))

        # --- 1) laser calibration clips: per-clip raw distance distribution ---
        clip_points = []  # (true_m, raw_median, raw_mean, raw_std, n_frames)
        all_infer_ms = []
        for video_path, tag, true_m in CALIB_CLIPS:
            seed_box = get_seed_box(tag)
            sample_every = sample_every_for(video_path, args.frames_per_video)
            distances, infer_times, n_lost = track_clip(video_path, seed_box, worker, sample_every=sample_every,
                                                          writer=review_writer, label=f'laser {tag} (true={true_m}m)')
            all_infer_ms.extend(infer_times)
            if not distances:
                print(f'  {tag}: NO valid distance samples (tracker lost immediately?) - skipping this point')
                continue
            raw_median = st.median(distances)
            raw_mean = st.mean(distances)
            raw_std = st.pstdev(distances) if len(distances) > 1 else 0.0
            clip_points.append((true_m, raw_median, raw_mean, raw_std, len(distances)))
            print(f'  {tag} (true={true_m}m): n={len(distances)} lost={n_lost} '
                  f'raw_median={raw_median:.2f}m raw_mean={raw_mean:.2f}m std={raw_std:.2f}m')
            with open(os.path.join(model_dir, f'laser_{tag}_raw.csv'), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame_sample_index', 'distance_raw_m', 'inference_ms'])
                for i, (d, ms) in enumerate(zip(distances, infer_times)):
                    w.writerow([i, d, ms])

        if len(clip_points) < 2:
            print(f'  ! not enough calibration points for {short_name}, skipping fit')
            continue

        true_vals = [p[0] for p in clip_points]
        raw_vals = [p[1] for p in clip_points]  # median-based, robust to per-frame outliers
        scale, offset = fit_calibration(raw_vals, true_vals)

        errors_abs = []
        errors_pct = []
        detail_rows = []
        for true_m, raw_median, raw_mean, raw_std, n in clip_points:
            corrected = scale * raw_median + offset
            abs_err = abs(corrected - true_m)
            pct_err = 100 * abs_err / true_m
            errors_abs.append(abs_err)
            errors_pct.append(pct_err)
            detail_rows.append((true_m, raw_median, corrected, abs_err, pct_err, raw_std, n))
            print(f'    true={true_m}m  raw_median={raw_median:.2f}  corrected={corrected:.2f}  '
                  f'abs_err={abs_err:.2f}m  pct_err={pct_err:.1f}%')

        mae = st.mean(errors_abs)
        rmse = (st.mean(e ** 2 for e in errors_abs)) ** 0.5
        # does error grow with distance? sign of correlation between true distance and abs error
        if len(true_vals) >= 3:
            err_trend_scale, _ = np.polyfit(true_vals, errors_abs, 1)
        else:
            err_trend_scale = float('nan')
        avg_infer_ms = st.mean(all_infer_ms) if all_infer_ms else float('nan')

        with open(os.path.join(model_dir, 'calibration_fit.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['true_m', 'raw_median_m', 'corrected_m', 'abs_error_m', 'pct_error', 'raw_std_m', 'n_frames'])
            for row in detail_rows:
                w.writerow(row)

        # --- 2) color vs monochrome consistency (fixed center box, no detection/tracking) ---
        box = fixed_water_roi()
        renkli_every = sample_every_for('testGerçekRenkli.mp4', args.frames_per_video)
        siyahbeyaz_every = sample_every_for('testGerçekSiyahBeyaz.mp4', args.frames_per_video)
        renkli_color = track_fixed_box('testGerçekRenkli.mp4', worker, box, sample_every=renkli_every,
                                        writer=review_writer, label='renkli (color)')
        renkli_gray = track_fixed_box('testGerçekRenkli.mp4', worker, box, sample_every=renkli_every, to_gray=True,
                                       writer=review_writer, label='renkli (synthetic gray)')
        siyahbeyaz_native = track_fixed_box('testGerçekSiyahBeyaz.mp4', worker, box, sample_every=siyahbeyaz_every,
                                             writer=review_writer, label='siyahBeyaz (native mono)')

        def consistency_stats(vals, name):
            if len(vals) < 2:
                print(f'    {name}: not enough samples ({len(vals)})')
                return None
            std = st.pstdev(vals)
            jitters = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
            mean_jitter = st.mean(jitters)
            print(f'    {name}: n={len(vals)} mean={st.mean(vals):.2f}m std={std:.3f}m mean_jitter={mean_jitter:.3f}m')
            return {'n': len(vals), 'mean': st.mean(vals), 'std': std, 'mean_jitter': mean_jitter}

        print('  color/mono consistency (fixed water-surface ROI, same pixel box in both clips):')
        cs_color = consistency_stats(renkli_color, 'renkli (native color)')
        cs_synth_gray = consistency_stats(renkli_gray, 'renkli (synthetic grayscale)')
        cs_native_gray = consistency_stats(siyahbeyaz_native, f'siyahBeyaz (native monochrome, every {siyahbeyaz_every} frames)')

        with open(os.path.join(model_dir, 'color_mono_consistency.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['variant', 'n', 'mean_m', 'std_m', 'mean_jitter_m'])
            for name, cs in [('renkli_color', cs_color), ('renkli_synthetic_gray', cs_synth_gray),
                              ('siyahbeyaz_native_mono', cs_native_gray)]:
                if cs:
                    w.writerow([name, cs['n'], cs['mean'], cs['std'], cs['mean_jitter']])

        mono_note = ''
        if cs_color and cs_synth_gray:
            diff = cs_synth_gray['std'] - cs_color['std']
            mono_note = f'synthetic-gray std {"+" if diff >= 0 else ""}{diff:.3f}m vs color'

        summary_rows.append({
            'model': short_name,
            'mae_m': round(mae, 3),
            'rmse_m': round(rmse, 3),
            'avg_pct_error': round(st.mean(errors_pct), 1),
            'ms_per_frame': round(avg_infer_ms, 1),
            'error_vs_distance_slope': round(err_trend_scale, 4) if err_trend_scale == err_trend_scale else None,
            'calib_scale': round(scale, 4),
            'calib_offset': round(offset, 4),
            'color_vs_mono_note': mono_note,
        })

        review_writer.release()
        print(f'  visual review video saved to {review_path}')

        # free memory before loading the next (possibly large) model
        del depth_model, processor, worker
        if device == 'mps':
            torch.mps.empty_cache()

    # --- final summary ---
    summary_csv = os.path.join(OUT_DIR, 'summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'mae_m', 'rmse_m', 'avg_pct_error', 'ms_per_frame',
                                           'error_vs_distance_slope', 'calib_scale', 'calib_offset', 'color_vs_mono_note'])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    header = f'{"Model":15s} {"MAE(m)":>8s} {"RMSE(m)":>8s} {"Ort.%hata":>10s} {"ms/kare":>9s}'
    print(header)
    for row in summary_rows:
        print(f'{row["model"]:15s} {row["mae_m"]:8.3f} {row["rmse_m"]:8.3f} '
              f'{row["avg_pct_error"]:9.1f}% {row["ms_per_frame"]:8.1f}ms')
    print(f'\nfull details saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
