"""Measure bright/clipped areas in decoded video, without changing detector inputs."""
import argparse
import csv
import heapq
import json
from pathlib import Path

import cv2
import numpy as np


def normalized_roi(value):
    try:
        roi = tuple(float(v) for v in value.split(','))
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError
        return roi
    except ValueError as exc:
        raise argparse.ArgumentTypeError('ROI must be x1,y1,x2,y2 in [0,1], with positive area') from exc


def highlight_metrics(frame, threshold=250, roi=(0, 0, 1, 1)):
    """Fractions of decoded BGR pixels, not a claim about sensor clipping or cause.

    White paint/sky also trigger these measurements. Lossy video may map clipped
    sensor values below 255, so expose the threshold and retain an exact-white count.
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError('expected a nonempty uint8 BGR image')
    if not 1 <= threshold <= 255:
        raise ValueError('threshold must be in [1, 255]')
    x1, y1, x2, y2 = normalized_roi(','.join(map(str, roi)))
    h, w = frame.shape[:2]
    left, top = int(x1 * w), int(y1 * h)
    right, bottom = int(np.ceil(x2 * w)), int(np.ceil(y2 * h))
    pixels = frame[top:bottom, left:right]
    near_white = np.all(pixels >= threshold, axis=2)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(near_white.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats) > 1 else 0
    return {
        'near_white_fraction': float(near_white.mean()),
        'exact_white_fraction': float(np.all(pixels == 255, axis=2).mean()),
        'any_channel_high_fraction': float(np.any(pixels >= threshold, axis=2).mean()),
        'largest_near_white_fraction': largest / near_white.size,
        'luma_p50': float(np.percentile(gray, 50)),
        'luma_p95': float(np.percentile(gray, 95)),
    }


def analyze(video, out, sample_every=30, threshold=250, roi=(0, 0, 1, 1), top_k=8,
            max_frames=None):
    if sample_every < 1 or top_k < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError('sample_every/max_frames must be positive; top_k must be nonnegative')
    if not 1 <= threshold <= 255:
        raise ValueError('threshold must be in [1, 255]')
    roi = normalized_roi(','.join(map(str, roi)))
    if not Path(video).is_file():
        raise ValueError(f'video not found: {video}')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'could not open video: {video}')
    out = Path(out)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if np.isfinite(fps) and fps > 0 else None
    reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    reported_frames = int(reported_frames) if np.isfinite(reported_frames) and reported_frames > 0 else None
    top = []  # bounded memory, full-resolution unmodified frames
    decoded = sampled = 0
    try:
        # Avoid silently mixing analyses or overwriting a previous review.
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'frames.csv').open('w', newline='') as f:
            writer = None
            while max_frames is None or decoded < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                idx = decoded
                decoded += 1
                if idx % sample_every:
                    continue
                metrics = highlight_metrics(frame, threshold, roi)
                row = {'frame': idx, 'time_s': idx / fps if fps else '', **metrics}
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                sampled += 1
                if top_k:
                    entry = (metrics['near_white_fraction'], idx, frame.copy())
                    if len(top) < top_k:
                        heapq.heappush(top, entry)
                    elif entry[:2] > top[0][:2]:
                        heapq.heapreplace(top, entry)
        if not decoded:
            raise ValueError(f'decoded 0 frames from {video}')
        saved = []
        for score, idx, frame in sorted(top, key=lambda item: item[:2], reverse=True):
            name = f'frame_{idx:08d}.png'
            if not cv2.imwrite(str(out / name), frame):
                raise OSError(f'could not save {out / name}')
            saved.append({'frame': idx, 'near_white_fraction': score, 'image': name})
        summary = {
            'video': str(Path(video).resolve()), 'fps': fps,
            'decoded_frames': decoded, 'sampled_frames': sampled,
            'reported_frames': reported_frames, 'sample_every': sample_every,
            'max_frames': max_frames, 'threshold': threshold, 'roi': roi,
            'possible_early_decode_end': bool(reported_frames and decoded < reported_frames
                                             and (max_frames is None or decoded < max_frames)),
            'note': 'Brightness is a review signal, not proof of glare or sensor clipping. '
                    'Times use nominal FPS; variable-frame-rate video may differ. '
                    'Saved PNGs preserve decoded pixels; no CLAHE, overlays or resizing.',
            'top_frames': saved,
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--sample-every', type=int, default=30)
    parser.add_argument('--threshold', type=int, default=250)
    parser.add_argument('--roi', type=normalized_roi, default=(0, 0, 1, 1),
                        help='normalized x1,y1,x2,y2, e.g. 0,0.45,1,1 for the lower frame')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    try:
        summary = analyze(args.video, args.out, args.sample_every, args.threshold,
                          args.roi, args.top_k, args.max_frames)
    except (ValueError, OSError, cv2.error) as exc:
        parser.exit(1, f'{exc}\n')
    print(f"Analyzed {summary['sampled_frames']} sampled frames; saved review to {args.out}")
    if summary['possible_early_decode_end']:
        print('Warning: decoding ended before the reported frame count; review may be incomplete.')


if __name__ == '__main__':
    main()
