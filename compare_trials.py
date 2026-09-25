"""Compare camera-setting trial recordings (docs/GLARE_PLAN.md) on boat-relevant signals.

    venv/bin/python compare_trials.py A=rec_a/ShipCam1.mp4 B=rec_b/ShipCam1.mp4 \\
        A2=rec_a2/ShipCam1.mp4 --weights weights/boat_v4_s_best.pt --out results/trials

Per trial: highlight share in the water band and inside tracked boxes, detection
coverage, sharpness of the target, and glare-suspect tracks. Trial recordings are
unlabeled, so coverage is a recall proxy that a false positive also raises;
review the glare-suspect count and the saved overlays before choosing a setting.
Reads recordings only; camera settings are never changed here.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from analyze_glare import highlight_metrics
from boatdet.config import ROI_BAND, MotConfig
from boatdet.detection import YoloDetector
from boatdet.pipeline import PerceptionPipeline
from boatdet.overlay import render
from boatdet.video import WORKING_SIZE, frame_timer, open_video, video_fps


def sharpness(frame, box, working_size):
    """Variance of the Laplacian inside a working-frame box on the source frame."""
    height, width = frame.shape[:2]
    sx, sy = width / working_size[0], height / working_size[1]
    x1, y1, x2, y2 = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if min(crop.shape[:2]) < 3:
        return None
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def evaluate(name, source, model, args):
    cap = open_video(source)
    fps = video_fps(cap)
    times = frame_timer(cap, fps)
    pipeline = PerceptionPipeline(YoloDetector(model, conf=args.low_conf, imgsz=args.imgsz), WORKING_SIZE,
                                  MotConfig(conf_threshold=args.conf, low_conf_threshold=args.low_conf))
    water_highlight, box_highlight, sharp, covered, frames = [], [], [], 0, 0
    track_ids, suspect_ids = set(), set()
    detector_ms, pipeline_ms = [], []
    source_size = None
    started = time.perf_counter()
    last_time = 0.0
    try:
        index = 0
        while args.max_frames is None or index < args.max_frames:
            ok, native = cap.read()
            if not ok:
                break
            if index % args.sample_every == 0:
                frames += 1
                source_size = (native.shape[1], native.shape[0])
                last_time = times(index)
                water_highlight.append(highlight_metrics(native, roi=(0, args.band[0], 1, args.band[1]))
                                       ['any_channel_high_fraction'])
                result = pipeline.process(cv2.resize(native, WORKING_SIZE), native, times(index), index)
                # First call includes model setup/warmup; keep it out of steady-state latency.
                if frames > 1:
                    detector_ms.append(pipeline.timer.stages['detector'])
                    pipeline_ms.append(pipeline.timer.stages['total'])
                if getattr(args, 'save_overlays', False) and (frames == 1 or frames % 50 == 0):
                    output = Path(args.out) / name
                    output.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(output / f'{index:06d}.jpg'),
                                       render(cv2.resize(native, WORKING_SIZE), result)):
                        raise OSError(f'could not save overlay for {name} frame {index}')
                covered += any(d.confidence >= args.conf for d in result.detections)
                for track in result.tracks:
                    track_ids.add(track.track_id)
                    if track.glare_suspect:
                        suspect_ids.add(track.track_id)
                    if track.missed_frames == 0:
                        if track.highlight_fraction is not None:
                            box_highlight.append(track.highlight_fraction)
                        value = sharpness(native, track.bbox_xyxy, WORKING_SIZE)
                        if value is not None:
                            sharp.append(value)
            index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'{name}: no frames decoded from {source}')
    elapsed = time.perf_counter() - started
    median = lambda values: round(float(np.median(values)), 4) if values else None
    percentiles = lambda values: dict(p50=median(values), p95=round(float(np.percentile(values, 95)), 4)
                                     if values else None)
    return dict(trial=name, source=str(source), frames=frames,
                source_fps=fps, source_size=source_size, decoded_frames=index,
                last_sample_time_s=last_time, elapsed_s=round(elapsed, 3),
                sampled_frames_per_second=round(frames / elapsed, 3),
                detector_ms=percentiles(detector_ms), pipeline_ms=percentiles(pipeline_ms),
                coverage=round(covered / frames, 4),
                water_highlight_median=median(water_highlight),
                water_highlight_p95=round(float(np.percentile(water_highlight, 95)), 4),
                box_highlight_median=median(box_highlight),
                target_sharpness_median=median(sharp),
                tracks=len(track_ids), glare_suspect_tracks=len(suspect_ids))


def parse_trial(value):
    name, _, path = value.partition('=')
    if not name or not path:
        raise argparse.ArgumentTypeError('trials are NAME=PATH')
    if name in ('.', '..') or '/' in name or '\\' in name:
        raise argparse.ArgumentTypeError('trial name must be a filename component')
    return name, path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trials', nargs='+', type=parse_trial, help='NAME=recording (mp4 or raw frame dir)')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--conf', type=float, default=MotConfig.conf_threshold)
    parser.add_argument('--low-conf', type=float, default=MotConfig.low_conf_threshold)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    parser.add_argument('--sample-every', type=int, default=1)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--band', type=float, nargs=2, default=ROI_BAND, metavar=('TOP', 'BOTTOM'),
                        help='water band for the highlight measurement, fractions of height')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--save-overlays', action='store_true', help='save first and every 50th sampled overlay')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.sample_every < 1:
        parser.error('--sample-every must be positive')
    if not 0 <= args.band[0] < args.band[1] <= 1:
        parser.error('--band must satisfy 0 <= TOP < BOTTOM <= 1')
    if len({name for name, _ in args.trials}) != len(args.trials):
        parser.error('trial names must be unique')
    from ultralytics import YOLO
    model = YOLO(args.weights)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, source in args.trials:
        rows.append(evaluate(name, source, model, args))
        from evaluate_models import sha256
        settings = dict(weights=args.weights, weights_sha256=sha256(args.weights),
                        imgsz=args.imgsz, conf=args.conf, low_conf=args.low_conf,
                        sample_every=args.sample_every, band=args.band,
                        threads=args.threads,
                        device=str(model.predictor.device), trials=rows,
                        limitations='Unlabeled recordings: coverage is not precision or recall. '
                        'Detector/pipeline latency excludes decoding and the first sampled frame. '
                        'Pipeline includes tracking but no depth model. End-to-end sampling throughput '
                        'includes decoding, warmup, image statistics and optional overlay writes.')
        (args.out / 'trials.json').write_text(json.dumps(settings, indent=2) + '\n')
        row = rows[-1]
        print(f"{row['trial']:6s} frames={row['frames']:4d} coverage={row['coverage']:.3f} "
              f"water_hl={row['water_highlight_median']} box_hl={row['box_highlight_median']} "
              f"sharp={row['target_sharpness_median']} tracks={row['tracks']} glare?={row['glare_suspect_tracks']}",
              flush=True)


if __name__ == '__main__':
    main()
