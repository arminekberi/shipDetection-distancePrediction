"""Train a YOLO boat detector from a named experiment preset.

    python train.py v5_s
    python train.py v5_s --name boat_mono --data yolo_dataset_v4_mono/dataset.yaml -o batch=32 -o patience=25
"""
import argparse
from pathlib import Path

import yaml

from boatdet.device import get_device

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'runs' / 'detect' / 'runs_boat_yolo'
BASE = dict(epochs=80, batch=16, imgsz=640, patience=25)

# Run name (boat_<key>) -> base weights and overrides.
EXPERIMENTS = {
    'v4_s': dict(model='yolov8s.pt'),
    'v4s_frozen': dict(model='yolov8s.pt', freeze=10),
    'v4s_hsvaug': dict(model='yolov8s.pt', hsv_h=0.03, hsv_s=0.9, hsv_v=0.6),
    'v5_s': dict(model='yolov8s.pt', imgsz=960, patience=40),
    'v6_11m': dict(model='yolo11m.pt', imgsz=960, patience=40),
}


def parse_override(text):
    key, sep, value = text.partition('=')
    if not sep or not key:
        raise argparse.ArgumentTypeError('expected KEY=VALUE')
    return key, yaml.safe_load(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    parser.add_argument('--name', help='run name, default boat_<experiment>')
    parser.add_argument('--data', default='yolo_dataset_v4_color/dataset.yaml',
                        help='training runs use the color dataset; see README')
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--device', help='default: cuda, mps or cpu')
    parser.add_argument('--resume', action='store_true', help='continue from last.pt when present')
    parser.add_argument('-o', '--override', type=parse_override, action='append', default=[],
                        help='extra Ultralytics train argument, repeatable')
    args = parser.parse_args()

    from ultralytics import YOLO
    name = args.name or f'boat_{args.experiment}'
    last = args.project / name / 'weights' / 'last.pt'
    if args.resume and last.is_file():
        # A finished checkpoint is stripped (epoch -1); resuming one silently starts a
        # fresh COCO run instead, so stop here rather than train the wrong thing.
        import torch
        if torch.load(last, map_location='cpu', weights_only=False).get('epoch', -1) < 0:
            return print(f'{name} already finished ({last}); pass --name for a new run')
        print(f'resuming {last}')
        YOLO(str(last)).train(resume=True)
        return
    settings = {**BASE, **EXPERIMENTS[args.experiment], **dict(args.override)}
    model = settings.pop('model')
    print(f'training {name}: {model} {settings}')
    YOLO(model).train(data=args.data, device=args.device or get_device(), project=str(args.project),
                      name=name, exist_ok=args.resume, verbose=True, **settings)


if __name__ == '__main__':
    main()
