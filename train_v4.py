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
    name='boat_v4_s',
    verbose=True,
)
