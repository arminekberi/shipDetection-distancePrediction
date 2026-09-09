import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from run import colorize_depth, get_device

WIDTH, HEIGHT = 640, 360


def expected_row_depth(depth_m):
    """Robust per-row baseline: the median depth across each row, smoothed vertically.
    Assumes the dominant content per row is flat water at a roughly consistent distance
    (perspective means distance is primarily a function of row, not column) - an object
    sitting on the water breaks this by being locally closer than its row's baseline."""
    row_median = np.median(depth_m, axis=1)
    k = 15
    kernel = np.ones(k) / k
    padded = np.pad(row_median, (k // 2, k // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')[:len(row_median)]
    return smoothed


def find_boat_blob(depth_m, min_area=30, max_area=8000):
    baseline = expected_row_depth(depth_m)
    residual = baseline[:, None] - depth_m  # positive = closer than expected (sticking up/out)
    mask = (residual > np.percentile(residual, 97)) & (residual > 0.15)
    mask = mask.astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    best_score = -1
    for i in range(1, n):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / max(h, 1)
        if aspect < 0.4 or aspect > 6.0:
            continue  # boats are wider-than-tall blobs, not thin vertical glare streaks
        # score favors larger, more centrally-located (horizontally) blobs
        cx = centroids[i][0]
        center_bonus = 1.0 - abs(cx - WIDTH / 2) / (WIDTH / 2)
        score = area * (0.5 + 0.5 * center_bonus)
        if score > best_score:
            best_score = score
            best = (x, y, x + w, y + h)
    return best, mask


def main():
    parser = argparse.ArgumentParser(description='Depth-Anything-ONLY boat detection: no YOLO, no training - finds the region closest relative to its row baseline (i.e. sticking above the water plane) via connected components on the depth map.')
    parser.add_argument('--video', required=True)
    parser.add_argument('--model', type=str, default='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    parser.add_argument('--infer-size', type=int, default=392)
    parser.add_argument('--output', required=True)
    parser.add_argument('--log', required=True)
    args = parser.parse_args()

    device = get_device()
    print(f'device: {device}')
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (WIDTH * 3, HEIGHT))
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

        box, mask = find_boat_blob(depth_m)
        display = frame.copy()
        distance = None
        if box is not None:
            x1, y1, x2, y2 = box
            distance = float(np.median(depth_m[y1:y2, x1:x2]))
            n_detected += 1
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(display, f'{distance:.2f}m', (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(display, f'frame {frame_idx}', (5, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        combined = np.hstack([display, depth_color, mask_bgr])
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
