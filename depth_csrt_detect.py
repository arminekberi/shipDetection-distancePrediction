import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from run import colorize_depth, get_device
from depth_only_detect import find_boat_blob

WIDTH, HEIGHT = 640, 360


class DepthCsrtTracker:
    """Same detect+track fusion pattern as YoloCsrtTracker, but the detector is the
    depth-map blob finder (find_boat_blob) instead of a trained YOLO model - no training
    data, no fine-tuning, purely geometric (finds regions closer than their row's expected
    water-surface baseline). A found blob always re-anchors CSRT (there's no separate
    strong/weak confidence tier here, unlike YOLO's conf score); when no blob is found,
    CSRT carries on from its last anchor; if that also fails, coast on the last known box
    for a few frames before reporting no detection."""

    def __init__(self, width=640, height=360, grace_frames=4):
        self.csrt = None
        self.width = width
        self.height = height
        self.grace_frames = grace_frames
        self.last_box = None  # (x, y, w, h) ints
        self.miss_count = 0

    def update(self, frame, depth_m):
        box, _mask = find_boat_blob(depth_m)
        if box is not None:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w >= 2 and h >= 2:
                self.csrt = cv2.TrackerCSRT_create()
                self.csrt.init(frame, (x1, y1, w, h))
                self.last_box = (x1, y1, w, h)
                self.miss_count = 0
                return True, self.last_box

        if self.csrt is not None:
            ok, tbox = self.csrt.update(frame)
            if ok:
                self.last_box = tuple(int(v) for v in tbox)
                self.miss_count = 0
                return True, self.last_box

        self.miss_count += 1
        if self.last_box is not None and self.miss_count <= self.grace_frames:
            return True, self.last_box

        return False, (0, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description='Depth-Anything-only detection (geometric blob finder, no YOLO/training) + CSRT continuity, for evaluating whether tracking smooths out the raw per-frame blob-detection noise.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--model', type=str, default='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    parser.add_argument('--infer-size', type=int, default=392)
    parser.add_argument('--grace-frames', type=int, default=4)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()

    device = get_device()
    print(f'device: {device}')
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()
    tracker = DepthCsrtTracker(WIDTH, HEIGHT, grace_frames=args.grace_frames)

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (WIDTH * 2, HEIGHT))
    log_f = open(args.log, 'w', newline='')
    log_f.write('frame,distance_m\n')

    frame_idx = 0
    n_detected = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image_pil, return_tensors='pt', size={'height': args.infer_size, 'width': args.infer_size}).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        post = processor.post_process_depth_estimation(outputs, target_sizes=[(HEIGHT, WIDTH)])
        depth_m = post[0]['predicted_depth'].cpu().numpy()
        depth_color = colorize_depth(depth_m)

        tracked, (tx, ty, tw, th) = tracker.update(frame, depth_m)
        display = frame.copy()
        distance = None
        if tracked:
            x1, y1, x2, y2 = tx, ty, tx + tw, ty + th
            x1, x2 = sorted((max(0, x1), max(0, x2)))
            y1, y2 = sorted((max(0, y1), max(0, y2)))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                distance = float(np.median(depth_m[y1:y2, x1:x2]))
                n_detected += 1
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
                dist_str = f'{distance:.2f}m' if distance is not None else '--'
                cv2.putText(display, dist_str, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.putText(display, f'frame {frame_idx}', (5, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        combined = np.hstack([display, depth_color])
        writer.write(combined)
        dist_str = '' if distance is None else str(distance)
        log_f.write(f'{frame_idx},{dist_str}\n')

        frame_idx += 1

    cap.release()
    writer.release()
    log_f.close()
    print(f'done: {n_detected}/{frame_idx} frames detected ({100*n_detected/frame_idx:.1f}%)')
    print(f'saved to {args.output}')


if __name__ == '__main__':
    main()
