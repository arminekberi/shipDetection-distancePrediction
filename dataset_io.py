"""Validated, resumable writes for manually reviewed boat annotations."""
import os
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np


def recording_tag(video, override=None):
    path = Path(video)
    tag = override or path.stem.replace(',', '_').replace(' ', '_')
    # Server recordings reuse ShipCam0.mp4/ShipCam1.mp4 in every capture folder.
    if not override and re.fullmatch(r'ShipCam\d+', path.stem, re.I):
        parent = path.parent.name
        if not parent:
            raise ValueError('Use --tag for a ShipCam file without a recording directory')
        tag = f'{path.stem}_{parent}'.replace('-', '_').replace(' ', '_')
    if not tag or tag in ('.', '..') or '/' in tag or '\\' in tag:
        raise ValueError('tag must be a nonempty filename prefix without path separators')
    return tag


def require_mounted_destination(path):
    for parent in (Path(path).resolve(), *Path(path).resolve().parents):
        if parent.name in ('shipcaps_remote', 'shipcaps1_remote') and not os.path.ismount(parent):
            raise OSError(f'Remote mount is unavailable: {parent}; refusing local dataset writes')


def save_annotation(out_dir, split, tag, idx, frame, box=None):
    """Commit the label only after the image has been successfully encoded and saved."""
    require_mounted_destination(out_dir)
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    recording_tag('', tag)
    text = ''
    if box is not None:
        x, y, w, h = map(float, box)
        if not np.isfinite((x, y, w, h)).all() or w <= 0 or h <= 0:
            raise ValueError('box must contain finite coordinates and positive dimensions')
        height, width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError('box does not contain a usable area inside the image')
        text = f'0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n'
    stem = f'{tag}_{idx:05d}'
    image_path = Path(out_dir) / 'images' / split / (stem + '.jpg')
    label_path = Path(out_dir) / 'labels' / split / (stem + '.txt')
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        raise OSError(f'could not encode {image_path}')
    for dest, payload in ((image_path, encoded.tobytes()), (label_path, text.encode())):
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
