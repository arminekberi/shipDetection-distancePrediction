"""Video decoding/encoding helpers, for mp4 archives and raw rig frames."""
import math
import json
import re
from pathlib import Path

import cv2
import numpy as np

WORKING_SIZE = (640, 360)  # tracking/depth/label coordinates (width, height)
DEFAULT_FPS = 15.0


RAW_SUFFIX = '.bgr'
RAW_ASPECT = 16 / 9


def raw_geometry(size_bytes, aspect=RAW_ASPECT):
    """(width, height) of a headerless BGR frame of this size, assuming the sensor aspect."""
    pixels, height = size_bytes / 3, round((size_bytes / 3 / aspect) ** 0.5)
    width = round(height * aspect)
    if height < 1 or width * height * 3 != size_bytes:
        raise ValueError(f'{size_bytes} bytes is not a {aspect:.3f}:1 BGR frame ({pixels:.0f} pixels)')
    return width, height


class RawCapture:
    """VideoCapture-compatible reader for a directory of headerless BGR frames.

    The rig keeps full-resolution frames beside the preview mp4. That archive is a
    1280-wide ~1.2 Mbit/s encode of a 4608x2592 capture and drops most of the
    detail small targets are made of, so measurements belong on these files.
    """

    def __init__(self, directory):
        directory = Path(directory)
        self.files = sorted(p for p in directory.iterdir() if p.suffix in ('.bgr', '.gray'))
        if not self.files:
            raise FileNotFoundError(f'no .bgr/.gray frames in {directory}')
        manifest = directory / 'manifest.json'
        self.channels = 3
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            self.width, self.height = int(metadata['width']), int(metadata['height'])
            self.channels = int(metadata['channels'])
            if self.width < 1 or self.height < 1 or self.channels not in (1, 3):
                raise ValueError(f'invalid raw frame geometry: {manifest}')
            pixel_format = metadata.get('pix_fmt', 'gray' if self.channels == 1 else 'bgr24')
            if pixel_format != ('gray' if self.channels == 1 else 'bgr24'):
                raise ValueError(f'unsupported raw pixel format {pixel_format!r}: {manifest}')
            expected = self.width * self.height * self.channels
            if metadata.get('bytes_per_frame', expected) != expected:
                raise ValueError(f'inconsistent bytes_per_frame: {manifest}')
        else:
            if any(p.suffix == '.gray' for p in self.files):
                raise ValueError('a .gray capture needs manifest.json: legacy color frames also use this suffix')
            self.width, self.height = raw_geometry(self.files[0].stat().st_size)
        self.stamps = [int(m.group(1)) for f in self.files if (m := re.search(r'_(\d{13})', f.name))]
        if len(self.stamps) != len(self.files):
            self.stamps = []  # partial stamps would silently mix two time bases
        else:
            # A restarted capture may reset the frame counter inside the same
            # directory. Sort by capture time, not by the reused counter prefix.
            ordered = sorted(zip(self.stamps, self.files), key=lambda item: (item[0], item[1].name))
            self.stamps = [stamp for stamp, _ in ordered]
            self.files = [path for _, path in ordered]
        gaps = np.diff(self.stamps)
        gaps = gaps[gaps > 0] if len(gaps) else gaps
        self.fps = float(1000 / np.median(gaps)) if len(gaps) else DEFAULT_FPS
        self.index = 0

    def frame_time(self, index):
        """Capture time of a frame in seconds. Intervals are uneven, so this is
        the measured stamp rather than index / fps."""
        if not self.stamps or not 0 <= index < len(self.stamps):
            return index / self.fps
        return (self.stamps[index] - self.stamps[0]) / 1000

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.files):
            return False, None
        path = self.files[self.index]
        self.index += 1
        frame = np.fromfile(path, np.uint8)
        if frame.size != self.width * self.height * self.channels:
            raise ValueError(f'truncated raw frame: {path}')
        if self.channels == 1:
            return True, cv2.cvtColor(frame.reshape(self.height, self.width), cv2.COLOR_GRAY2BGR)
        return True, frame.reshape(self.height, self.width, 3)

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_COUNT: len(self.files),
                cv2.CAP_PROP_POS_FRAMES: self.index,
                cv2.CAP_PROP_FRAME_WIDTH: self.width,
                cv2.CAP_PROP_FRAME_HEIGHT: self.height}.get(prop, 0.0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.index = max(0, min(int(value), len(self.files)))
        return True

    def release(self):
        self.files = []


def open_video(source):
    """Opened capture for a camera index, video file, or raw frame directory."""
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        return RawCapture(source)
    if isinstance(source, (str, Path)) and not Path(source).is_file():
        # Dropped rclone mount leaves an empty local dir behind.
        raise FileNotFoundError(f'video not found: {source} '
                                '(under shipcaps*_remote/? rerun scripts/remount_shipcaps.sh)')
    cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'could not open source {source!r}')
    return cap


def video_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS


def read_frames(source, sample_every=1, max_frames=None):
    """Yield (index, frame) for every sample_every-th decoded frame."""
    if sample_every < 1:
        raise ValueError('sample_every must be positive')
    cap = open_video(source)
    try:
        index = 0
        while max_frames is None or index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % sample_every == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


def h264_writer(path, fps, size):
    """H.264 writer; mp4v output does not play in browsers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'avc1'), fps, size)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f'could not open H.264 writer: {path}')
    return writer


def frame_timer(cap, fps):
    """Measured capture time per frame for raw sources, even spacing otherwise."""
    return cap.frame_time if isinstance(cap, RawCapture) else (lambda index: index / fps)


def resize_working(frame):
    return cv2.resize(frame, WORKING_SIZE)


def color_fraction(frame, spread=12):
    """Share of pixels whose channels differ; a luma capture stays near zero.

    Rig recordings made with SHIP_CAM_PIXEL_FORMAT=gray decode as three equal
    channels, so they look like color video to every codec and player.
    """
    if frame.ndim < 3 or frame.shape[2] < 3:
        return 0.0
    return float((frame.max(axis=2).astype(int) - frame.min(axis=2) > spread).mean())
