from ultralytics import YOLO


def main():
    model = YOLO('yolo11m.pt')
    results = model.train(
        data='yolo_dataset_v4/dataset.yaml',
        epochs=80,
        imgsz=960,
        device='mps',
        batch=16,
        patience=40,
        project='runs_boat_yolo',
        name='boat_v6_11m',
        verbose=True,
    )


if __name__ == '__main__':
    main()
