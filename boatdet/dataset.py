"""YOLO boat dataset: naming, validated writes and label parsing."""
import math
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np

SPLITS = ('train', 'val', 'test')
REMOTE_MOUNTS = ('shipcaps_remote', 'shipcaps1_remote')
# Reserved for testing, including unlabeled tails; never mined into train/val.
TEST_VIDEOS = {'renkliTekneTekne.mp4', 'renksizTekneTekne.mp4'}


def session_key(stem):
    """Canonical capture timestamp across camera names and '-'/'_' separators."""
    recording = re.sub(r'_\d{5,}$', '', stem)
    stamp = re.search(r'(\d{8})[-_](\d{6})', recording)
    return '_'.join(stamp.groups()) if stamp else recording


def recording_tag(video, override=None):
    """Label filename prefix; ShipCam files get their recording directory appended."""
    path = Path(video)
    tag = override or path.stem.replace(',', '_').replace(' ', '_')
    # Server recordings reuse ShipCam0.mp4/ShipCam1.mp4 in every capture folder.
    if not override and re.fullmatch(r'ShipCam\d+', path.stem, re.I):
        if not path.parent.name:
            raise ValueError('Use --tag for a ShipCam file without a recording directory')
        tag = f'{path.stem}_{path.parent.name}'.replace('-', '_').replace(' ', '_')
    if not tag or tag in ('.', '..') or '/' in tag or '\\' in tag:
        raise ValueError('tag must be a nonempty filename prefix without path separators')
    return tag


def label_path(root, split, tag, idx):
    return Path(root) / 'labels' / split / f'{tag}_{idx:05d}.txt'


def require_mounted_destination(path):
    """Refuse writes into a dropped rclone mount, which silently becomes a local dir."""
    for parent in (Path(path).resolve(), *Path(path).resolve().parents):
        if parent.name in REMOTE_MOUNTS and not os.path.ismount(parent):
            raise OSError(f'Remote mount is unavailable: {parent}; refusing local dataset writes')


def check_training_sources(videos):
    conflicts = sorted({str(v) for v in videos if os.path.basename(v) in TEST_VIDEOS})
    if conflicts:
        raise ValueError('Reserved test recordings cannot be mined into training: ' + ', '.join(conflicts))


def parse_box_line(line):
    """Normalized xyxy from a 'cls cx cy w h' line; ValueError unless a valid in-frame boat box."""
    values = [float(v) for v in line.split()]
    if len(values) != 5:
        raise ValueError('expected five fields')
    cls, x, y, w, h = values
    eps = 1e-5
    if not (all(math.isfinite(v) for v in values) and cls == 0 and 0 < w <= 1 and 0 < h <= 1
            and x - w / 2 >= -eps and y - h / 2 >= -eps and x + w / 2 <= 1 + eps and y + h / 2 <= 1 + eps):
        raise ValueError('invalid boat box')
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def yolo_line(box_xywh, width, height):
    """Clip a pixel box to the image; YOLO label line."""
    x, y, w, h = map(float, box_xywh)
    if not np.isfinite((x, y, w, h)).all() or w <= 0 or h <= 0:
        raise ValueError('box must contain finite coordinates and positive dimensions')
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError('box does not contain a usable area inside the image')
    return f'0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n'


def _atomic_write(dest, payload):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest.parent, prefix='.annotation-', delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
        os.replace(temporary, dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_annotation(root, split, tag, idx, frame, box_xywh=None):
    """Write image then label; no box = reviewed negative. Label never precedes its image."""
    require_mounted_destination(root)
    if split not in SPLITS:
        raise ValueError('split must be train, val or test')
    recording_tag('', tag)
    height, width = frame.shape[:2]
    text = '' if box_xywh is None else yolo_line(box_xywh, width, height)
    label = label_path(root, split, tag, idx)
    image = Path(root) / 'images' / split / (label.stem + '.jpg')
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        raise OSError(f'could not encode {image}')
    _atomic_write(image, encoded.tobytes())
    _atomic_write(label, text.encode())


def write_dataset_yaml(root, splits=('train', 'val')):
    """Create the split directories and dataset.yaml for a YOLO detection dataset."""
    require_mounted_destination(root)
    root = Path(root)
    splits = [s for s in SPLITS if s in splits]
    for sub in ('images', 'labels'):
        for split in splits:
            (root / sub / split).mkdir(parents=True, exist_ok=True)
    entries = ''.join(f'{split}: images/{split}\n' for split in splits)
    (root / 'dataset.yaml').write_text(f'path: {root.resolve()}\n{entries}names:\n  0: boat\n')
