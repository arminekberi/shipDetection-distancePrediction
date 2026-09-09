from ultralytics import YOLO

model = YOLO('yolov8s.pt')
results = model.train(
    data='yolo_dataset_v4/dataset.yaml',
    epochs=80,
    imgsz=640,
    device='mps',
    batch=16,
    patience=25,
    project='runs_boat_yolo',
    name='boat_v4s_hsvaug',
    hsv_h=0.03,
    hsv_s=0.9,
    hsv_v=0.6,
    verbose=True,
)
