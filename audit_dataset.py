"""Read-only audit of a YOLO detection dataset; never repairs or relabels data."""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

from boatdet.dataset import SPLITS, TEST_VIDEOS, parse_box_line, session_key


def recording_key(stem):
    return re.sub(r'_\d{5,}$', '', stem)


def audit(root, hash_images=False, verify_images=False):
    root = Path(root)
    if not (root / 'labels').is_dir() or not (root / 'images').is_dir():
        raise ValueError(f'not a YOLO dataset (expected images/ and labels/): {root}')
    issues, counts = [], {}
    recordings, stems, hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    sessions, pixels = defaultdict(set), defaultdict(set)
    for split in SPLITS:
        label_dir, image_dir = root / 'labels' / split, root / 'images' / split
        labels = {p.stem: p for p in label_dir.glob('*.txt')}
        images = {}
        for p in image_dir.iterdir() if image_dir.is_dir() else ():
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                if p.stem in images:
                    issues.append({'kind': 'duplicate_image_stem', 'path': str(p)})
                images[p.stem] = p
        positive = negative = boxes = 0
        for stem, p in labels.items():
            stems[stem].add(split)
            recordings[recording_key(stem)].add(split)
            sessions[session_key(stem)].add(split)
            if recording_key(stem) in {Path(v).stem for v in TEST_VIDEOS} and split != 'test':
                issues.append({'kind': 'reserved_test_in_training', 'path': str(p)})
            if stem not in images:
                issues.append({'kind': 'label_without_image', 'path': str(p)})
            lines = [line for line in p.read_text().splitlines() if line.strip()]
            positive += bool(lines)
            negative += not lines
            for number, line in enumerate(lines, 1):
                try:
                    parse_box_line(line)
                except ValueError:
                    issues.append({'kind': 'invalid_boat_box', 'path': str(p), 'line': number})
                boxes += 1
        for stem, p in images.items():
            if stem not in labels:
                issues.append({'kind': 'image_without_label', 'path': str(p)})
            if p.stat().st_size == 0:
                issues.append({'kind': 'empty_image', 'path': str(p)})
            if hash_images:
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                hashes[digest].add(split)
            if verify_images:
                decoded = cv2.imread(str(p))
                if decoded is None:
                    issues.append({'kind': 'undecodable_image', 'path': str(p)})
                elif hash_images:
                    digest = hashlib.sha256(str(decoded.shape).encode() + decoded.tobytes()).hexdigest()
                    pixels[digest].add(split)
        counts[split] = dict(images=len(images), labels=len(labels), positive=positive,
                             negative=negative, boxes=boxes)
    if not any(c['images'] for c in counts.values()):
        issues.append({'kind': 'empty_dataset', 'path': str(root)})
    for kind, groups in (('recording_across_splits', recordings), ('session_across_splits', sessions),
                         ('frame_across_splits', stems), ('identical_image_across_splits', hashes),
                         ('identical_pixels_across_splits', pixels)):
        issues.extend({'kind': kind, 'key': key, 'splits': sorted(splits)}
                      for key, splits in groups.items() if len(splits) > 1)
    return {'root': str(root), 'counts': counts, 'issues': issues,
            'hash_images': hash_images,
            'verify_images': verify_images,
            'limitations': 'Boat class 0 / five-field detection labels only. Recording identity '
                           'is inferred by removing the final frame number; renamed recordings, '
                           'adjacent captures and simultaneous cameras require a provenance manifest. '
                           'Empty labels require human review to confirm no visible boats.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--hash-images', action='store_true')
    parser.add_argument('--verify-images', action='store_true',
                        help='decode every image; with --hash-images also compare decoded pixels')
    parser.add_argument('--out', help='JSON path; omit to print to stdout')
    args = parser.parse_args()
    try:
        result = audit(args.root, args.hash_images, args.verify_images)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'{exc}\n')
    output = json.dumps(result, indent=2) + '\n'
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output)
    else:
        print(output, end='')
    return 2 if result['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
