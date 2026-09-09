import argparse
import os
import pickle

import cv2
import numpy as np
from ultralytics import YOLO

# (video_path, start_frame) - start_frame skips a prefix already labeled in yolo_dataset_v4
# (e.g. renkliTekneTekne/renksizTekneTekne were partially pulled into train/val already;
# only their genuinely-unlabeled tail is worth mining here).
VIDEOS = [
    ('8.mp4', 0),
    ('testt.mp4', 0),
    ('testtt.mp4', 0),
    ('signal-2026-09-02-10-36-16-525.mp4', 0),
    ('signal-2026-09-02-10-36-23-770.mp4', 0),
    ('signal-2026-09-02-10-36-32-768.mp4', 0),
    ('signal-2026-09-02-15-27-23-408.mp4', 0),
    ('teknedenTekneyeGörüntü/renkliTekneTekne.mp4', 69),
    ('teknedenTekneyeGörüntü/renksizTekneTekne.mp4', 224),
]

WIDTH, HEIGHT = 640, 360
SAMPLE_EVERY = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--out', default='review_v3')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)

    candidates = []
    thumbs = []  # (video, idx, conf, annotated_thumb)
    for video, start_frame in VIDEOS:
        cap = cv2.VideoCapture(video)
        idx = 0
        n_det = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx >= start_frame and idx % SAMPLE_EVERY == 0:
                frame_r = cv2.resize(frame, (WIDTH, HEIGHT))
                res = model.predict(frame_r, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                if len(res.boxes) > 0:
                    best = max(res.boxes, key=lambda b: float(b.conf[0]))
                    conf = float(best.conf[0])
                    x1, y1, x2, y2 = [float(v) for v in best.xyxy[0]]
                    candidates.append((video, idx, conf, (x1, y1, x2, y2)))

                    annotated = frame_r.copy()
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(annotated, f'{conf:.2f}', (int(x1), max(15, int(y1) - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    thumbs.append((video, idx, conf, annotated))
                    n_det += 1
            idx += 1
        cap.release()
        print(f'{video}: {n_det} detections >= conf {args.conf}')

    with open(os.path.join(args.out, 'candidates.pkl'), 'wb') as f:
        pickle.dump(candidates, f)
    print(f'\ntotal candidates: {len(candidates)}')
    print(f'saved to {args.out}/candidates.pkl')

    # numbered individual frames + grouped contact sheets, matching the earlier review workflow
    label_h = 20
    cols = 6
    for i, (video, idx, conf, annotated) in enumerate(thumbs):
        labeled = np.zeros((HEIGHT + label_h, WIDTH, 3), dtype=np.uint8)
        labeled[label_h:, :, :] = annotated
        cv2.putText(labeled, f'#{i} {video} f{idx} c{conf:.2f}', (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(args.out, f'{i:04d}.jpg'), labeled)

    by_video = {}
    for i, (video, idx, conf, annotated) in enumerate(thumbs):
        by_video.setdefault(video, []).append((i, annotated))

    for video, items in by_video.items():
        tag = os.path.splitext(os.path.basename(video))[0].replace(',', '_').replace(' ', '_')
        rows = (len(items) + cols - 1) // cols
        sheet = np.zeros((rows * (HEIGHT + label_h), cols * WIDTH, 3), dtype=np.uint8)
        for j, (i, annotated) in enumerate(items):
            r, c = divmod(j, cols)
            cell = np.zeros((HEIGHT + label_h, WIDTH, 3), dtype=np.uint8)
            cell[label_h:, :, :] = annotated
            cv2.putText(cell, f'#{i}', (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            sheet[r * (HEIGHT + label_h):(r + 1) * (HEIGHT + label_h), c * WIDTH:(c + 1) * WIDTH, :] = cell
        cv2.imwrite(os.path.join(args.out, f'sheet_{tag}.jpg'), sheet)
        print(f'sheet_{tag}.jpg: {len(items)} candidates, {rows} rows')


if __name__ == '__main__':
    main()
