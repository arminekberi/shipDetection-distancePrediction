import argparse
import csv

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

from run import DepthWorker, apply_clahe, colorize_depth, distance_in_box, get_device, tiled_detect

WIDTH, HEIGHT = 640, 360


def main():
    parser = argparse.ArgumentParser(description='Pure per-frame YOLO detection (no CSRT tracking/coasting) + depth, for diagnosing whether CSRT smoothing is the source of mislocated boxes.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--yolo-weights', required=True)
    parser.add_argument('--yolo-conf', type=float, default=0.15, help='minimum confidence to accept a detection at all (no fallback tracking, so this is the only gate)')
    parser.add_argument('--yolo-imgsz', type=int, default=960)
    parser.add_argument('--model', type=str, default='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    parser.add_argument('--infer-size', type=int, default=392)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--augment', action='store_true', help='test-time augmentation (ultralytics multi-scale/flip ensembling at inference) - no retraining, just slower per-frame inference')
    parser.add_argument('--clahe', action='store_true', help='apply CLAHE local contrast enhancement to the frame before YOLO (no retraining)')
    parser.add_argument('--tile-grid', type=str, default=None, help='e.g. "2x2" - run YOLO on overlapping tiles of the native-resolution frame instead of one shrunk frame (no retraining; helps small/far objects at the cost of grid_size x more inference calls per frame)')
    parser.add_argument('--tile-overlap', type=float, default=0.2, help='fractional overlap between adjacent tiles, only with --tile-grid')
    args = parser.parse_args()

    tile_grid = None
    if args.tile_grid:
        rows, cols = (int(v) for v in args.tile_grid.lower().split('x'))
        tile_grid = (rows, cols)

    device = get_device()
    print(f'device: {device}')

    yolo = YOLO(args.yolo_weights)
    processor = AutoImageProcessor.from_pretrained(args.model)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device)
    depth_model.eval()
    worker = DepthWorker(processor, depth_model, device, WIDTH, HEIGHT, args.infer_size, use_fp16=False)

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (WIDTH * 2, HEIGHT))
    log_f = open(args.log, 'w', newline='')
    log_writer = csv.writer(log_f)
    log_writer.writerow(['frame', 'confidence', 'distance_m'])

    frame_idx = 0
    n_detected = 0
    while True:
        ret, native_frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(native_frame, (WIDTH, HEIGHT))

        box = None
        conf = None
        if tile_grid is not None:
            detect_source = apply_clahe(native_frame) if args.clahe else native_frame
            box, conf = tiled_detect(yolo, detect_source, tile_grid, args.tile_overlap,
                                      args.yolo_conf, args.yolo_imgsz, args.augment, (WIDTH, HEIGHT))
            if box is not None:
                box = tuple(int(v) for v in box)
        else:
            detect_frame = apply_clahe(frame) if args.clahe else frame
            res = yolo.predict(detect_frame, conf=args.yolo_conf, imgsz=args.yolo_imgsz, verbose=False, augment=args.augment)[0]
            if len(res.boxes) > 0:
                best = max(res.boxes, key=lambda b: float(b.conf[0]))
                conf = float(best.conf[0])
                box = tuple(int(v) for v in best.xyxy[0])

        depth_m, depth_color, infer_ms = worker.infer(frame)

        distance = None
        display = frame.copy()
        if box is not None:
            distance = distance_in_box(depth_m, box)
            n_detected += 1
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
            dist_str = f'{distance:.2f}m' if distance is not None else '--'
            label = f'conf {conf:.2f}  {dist_str}'
            label_y = y1 - 8 if y1 - 20 > 0 else y2 + 18
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 4), (0, 0, 0), -1)
            cv2.putText(display, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(display, f'frame {frame_idx}', (5, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        combined = np.hstack([display, depth_color])
        writer.write(combined)
        log_writer.writerow([frame_idx, conf if conf is not None else '', distance if distance is not None else ''])

        print(f'frame {frame_idx}: conf={conf} distance={distance}')
        frame_idx += 1

    cap.release()
    writer.release()
    log_f.close()
    print(f'\ndone: {n_detected}/{frame_idx} frames detected ({100*n_detected/frame_idx:.1f}%)')
    print(f'saved video to {args.output}, log to {args.log}')


if __name__ == '__main__':
    main()
