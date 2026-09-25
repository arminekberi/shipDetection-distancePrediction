"""Repair split hygiene of a YOLO boat dataset. Moves files; never deletes.

Fixes, in order:
  reserved    frames of reserved test recordings   -> test split
  duplicates  frames byte-identical to a train one -> _quarantine/ (training data cannot validate)
  sessions    a capture session spanning splits    -> the split holding most of its frames
  boxes       boxes reaching outside the image     -> clipped; degenerate ones quarantine the frame

Rerunning changes nothing. Audit the result with audit_dataset.py.
"""
import argparse
import collections
import hashlib
import math
import re
from pathlib import Path

import cv2

from boatdet.dataset import (SPLITS, TEST_VIDEOS, parse_box_line, require_mounted_destination,
                             session_key, write_dataset_yaml)

IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
RESERVED = {Path(v).stem for v in TEST_VIDEOS}
QUARANTINE = '_quarantine'


def recording_key(stem):
    return re.sub(r'_\d{5,}$', '', stem)


class Frame:
    def __init__(self, split, image, label):
        self.split, self.image, self.label = split, image, label
        self.stem = label.stem

    def digest(self):
        return hashlib.sha256(self.image.read_bytes()).hexdigest()

    def move(self, root, destination, apply=True):
        """Move image+label into another split, or into quarantine; apply=False only plans it."""
        if not apply:
            self.split = destination  # later steps must see the planned state
            return
        if destination == QUARANTINE:
            targets = (root / QUARANTINE / self.split / 'images' / self.image.name,
                       root / QUARANTINE / self.split / 'labels' / self.label.name)
        else:
            targets = (root / 'images' / destination / self.image.name,
                       root / 'labels' / destination / self.label.name)
        # Check both before moving either; replace() would erase existing annotations.
        if any(target.exists() for target in targets):
            raise FileExistsError(f'repair destination already exists: {targets}')
        for source, target in zip((self.image, self.label), targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
        self.split, self.image, self.label = destination, *targets


def load_frames(root):
    """Image/label pairs per split; unpaired files are left untouched for the auditor."""
    frames = []
    for split in SPLITS:
        images = {p.stem: p for p in (root / 'images' / split).glob('*')
                  if p.suffix.lower() in IMAGE_SUFFIXES}
        for label in sorted((root / 'labels' / split).glob('*.txt')):
            if label.stem in images:
                frames.append(Frame(split, images[label.stem], label))
    return frames


def plan_reserved(frames):
    """Reserved test recordings belong in the test split, wherever they were labeled."""
    return [(f, 'test') for f in frames
            if recording_key(f.stem) in RESERVED and f.split != 'test']


def plan_duplicates(frames):
    """Quarantine a recording that repeats training frames: it cannot validate or test.

    Whole recording, not only the byte-identical frames - the rest are neighbouring
    frames of the same footage, duplicated under another name.
    """
    by_digest = collections.defaultdict(list)
    for frame in frames:
        by_digest[frame.digest()].append(frame)
    duplicated = set()
    for group in by_digest.values():
        if 'train' in {f.split for f in group}:
            duplicated |= {(f.split, recording_key(f.stem)) for f in group if f.split != 'train'}
    return [(f, QUARANTINE) for f in frames if (f.split, recording_key(f.stem)) in duplicated]


def plan_sessions(frames):
    """Keep every capture session, including simultaneous cameras, inside one split."""
    active = [f for f in frames if f.split != QUARANTINE]
    sessions = collections.defaultdict(collections.Counter)
    for frame in active:
        sessions[session_key(frame.stem)][frame.split] += 1
    moves = []
    for key, counts in sessions.items():
        if len(counts) < 2:
            continue
        # A session already pinned to test stays there; otherwise majority wins.
        target = 'test' if 'test' in counts else counts.most_common(1)[0][0]
        moves += [(f, target) for f in active if session_key(f.stem) == key and f.split != target]
    return moves


def clip_boxes(frames, apply):
    """Clip boxes to the image; a box with nothing left inside quarantines its frame."""
    clipped, quarantined = 0, []
    for frame in frames:
        lines = [line for line in frame.label.read_text().splitlines() if line.strip()]
        repaired, changed = [], False
        for line in lines:
            try:
                x1, y1, x2, y2 = parse_box_line(line)
            except ValueError:
                try:
                    values = [float(v) for v in line.split()]
                except ValueError:
                    values = []
                if len(values) != 5 or values[0] != 0 or not all(map(math.isfinite, values)) \
                        or values[3] <= 0 or values[4] <= 0:
                    quarantined.append(frame)
                    break
                _, cx, cy, w, h = values
                x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
                x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                image = cv2.imread(str(frame.image))
                if image is None:
                    quarantined.append(frame)
                    break
                height, width = image.shape[:2]
                if (x2 - x1) * width < 2 or (y2 - y1) * height < 2:
                    quarantined.append(frame)  # nothing usable left; never invent a negative
                    break
                changed = True
            repaired.append(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
        else:
            if changed:
                clipped += 1
                if apply:
                    frame.label.write_text('\n'.join(repaired) + '\n')
    return clipped, quarantined


def repair(root, apply):
    root = Path(root)
    if not (root / 'images').is_dir() or not (root / 'labels').is_dir():
        raise ValueError(f'not a YOLO dataset: {root}')
    if apply:
        require_mounted_destination(root)
    frames = load_frames(root)
    report = collections.Counter()
    for name, planner in (('reserved', plan_reserved), ('duplicates', plan_duplicates),
                          ('sessions', plan_sessions)):
        for frame, destination in planner(frames):
            report[f'{name}: {frame.split} -> {destination}'] += 1
            frame.move(root, destination, apply)
    if apply:
        frames = load_frames(root)
    clipped, quarantined = clip_boxes([f for f in frames if f.split != QUARANTINE], apply)
    if clipped:
        report['boxes clipped'] += clipped
    for frame in quarantined:
        report[f'boxes unusable: {frame.split} -> {QUARANTINE}'] += 1
        frame.move(root, QUARANTINE, apply)
    if apply:
        write_dataset_yaml(root, splits={f.split for f in load_frames(root)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root')
    parser.add_argument('--apply', action='store_true', help='move files; without it only the plan is printed')
    args = parser.parse_args()
    report = repair(args.root, args.apply)
    print(f'{args.root} ({"applied" if args.apply else "dry run, nothing moved"})')
    for key, count in sorted(report.items()):
        print(f'  {count:5d}  {key}')
    if not report:
        print('  nothing to repair')


if __name__ == '__main__':
    main()
