"""Find YOLO detections in frames currently confirmed as background in the review UI.

Run from the project root: venv/bin/python review_background.py
Only the prediction report is written; decisions and source labels are untouched.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from viewer.dataset_review import ReviewStore, checked_boxes, confirmed_background, digest

ROOT = Path(__file__).resolve().parent


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def scan(store, detect, model_info, *, limit=None, out=None):
    with store.connect() as db:
        decisions = [json.loads(row[0]) for row in db.execute('SELECT payload FROM decisions ORDER BY id')]
    selected = [d for d in decisions if confirmed_background(d)]
    report = {'format': 'ship-background-review-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'model': model_info, 'total_background': len(selected), 'scanned': 0, 'matches': 0,
              'complete': False, 'frames': {}}
    out = Path(out) if out else store.db_path.parent / 'yolo_background.json'
    candidates = selected if limit is None else selected[:limit]
    for i, decision in enumerate(candidates, 1):
        fid = decision['id']
        store.verify_source(fid)
        frame = store.frame(fid)
        detections = detect(store.image_path(fid))
        checked_boxes([d['box'] for d in detections])
        if any(not 0 <= d['confidence'] <= 1 for d in detections):
            raise ValueError('Invalid detector confidence')
        report['frames'][fid] = {'decision_version': decision['version'],
                                'source_image_sha256': frame['sha256'],
                                'source_label_sha256': frame['label_sha256'], 'detections': detections}
        report['scanned'] = i
        report['matches'] += bool(detections)
        if detections:
            print(f'{fid}: {len(detections)} рамок, max conf={max(d["confidence"] for d in detections):.3f}', flush=True)
        if i % 25 == 0:
            write_report(out, report)
            print(f'Проверено {i}/{len(candidates)}, найдено случаев: {report["matches"]}', flush=True)
    report['complete'] = report['scanned'] == report['total_background']
    write_report(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights/boat_v4_s_best.pt')
    parser.add_argument('--conf', type=float, default=.15, help='Minimum boat confidence (default: .15)')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int, help='Scan only the first N background frames (partial report)')
    parser.add_argument('--out', type=Path, help='Alternative report path; omitted to update the review UI')
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f'Weights do not exist: {args.weights}')
    if not 0 < args.conf <= 1 or args.imgsz < 32 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Use 0 < conf <= 1, imgsz >= 32, threads >= 1, limit >= 1')
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from boatdet.detection import YoloDetector

    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights)).to('cpu')
    detector = YoloDetector(model, conf=args.conf, imgsz=args.imgsz)

    def detect(path):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode image: {path}')
        height, width = image.shape[:2]
        result = []
        for box, confidence in detector.candidates(image, (width, height)):
            normalized = [max(0., min(1., v / scale)) for v, scale in zip(box, (width, height, width, height))]
            if normalized[0] < normalized[2] and normalized[1] < normalized[3]:
                result.append({'box': normalized, 'confidence': confidence})
        return result

    report = scan(ReviewStore(ROOT), detect,
                  {'weights': str(args.weights.resolve()), 'sha256': digest(args.weights),
                   'conf': args.conf, 'imgsz': args.imgsz, 'iou': detector.nms_iou,
                   'device': 'cpu', 'ultralytics': ultralytics.__version__},
                  limit=args.limit, out=args.out)
    print(f'Проверено {report["scanned"]}/{report["total_background"]}; случаев: {report["matches"]}.')
    if not args.out:
        print('Откройте http://127.0.0.1:8001/dataset_review.html?queue=yolo_background')


if __name__ == '__main__':
    main()
