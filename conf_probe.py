import sys
import cv2
from ultralytics import YOLO

video = sys.argv[1]
weights = sys.argv[2] if len(sys.argv) > 2 else 'runs/detect/runs_boat_yolo/boat_v4_s/weights/best.pt'
W, H = 640, 360
model = YOLO(weights)

cap = cv2.VideoCapture(video)
idx = 0
print("frame,conf")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (W, H))
    res = model.predict(frame, conf=0.01, imgsz=960, verbose=False, device='mps')[0]
    conf = max((float(b.conf[0]) for b in res.boxes), default=0.0)
    print(f"{idx},{conf:.3f}")
    idx += 1
cap.release()
