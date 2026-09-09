import argparse
import cv2

from run import YoloCsrtTracker


def count_tracking_loss(video_path, weights_path):
    from ultralytics import YOLO
    yolo_model = YOLO(weights_path)
    tracker = YoloCsrtTracker(
        yolo_model,
        conf_threshold=0.45,
        low_conf_threshold=0.15,
        width=640,
        height=480,
        box_smoothing=0.35,
        max_jump_frac=0.40,
        yolo_imgsz=960,
        grace_frames=4,
        tile_grid=None,
        tile_overlap=0.2,
        augment=False,
        clahe=False,
    )

    cap = cv2.VideoCapture(video_path)
    total = 0
    lost = 0
    while True:
        ret, native_frame = cap.read()
        if not ret:
            break
        raw_image = cv2.resize(native_frame, (640, 480))
        tracked, _ = tracker.update(raw_image, native_frame=native_frame)
        total += 1
        if not tracked:
            lost += 1
    cap.release()
    return total, lost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--videos', nargs='+', required=True)
    args = parser.parse_args()

    for video in args.videos:
        total, lost = count_tracking_loss(video, args.weights)
        pct = 100.0 * lost / total if total else 0.0
        print(f'{video}: {lost}/{total} lost ({pct:.1f}%)')
