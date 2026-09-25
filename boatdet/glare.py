"""Highlight overlap of a detection box: a review signal for specular glare.

Measured on the 2026-09-11 pool recordings, a light reflection mistaken for a
boat had 88% of its box pixels with some channel at 250 or above, real boats
at most 3%. The all-channels test misses it: the reflection is yellowish and
its blue channel stays below the threshold. One glare case is evidence for a
review flag, not for rejecting bright boxes, which would also erase a white hull
or a boat in sunlight (see docs/GLARE_PLAN.md).
"""
from boatdet.config import GLARE_CHANNEL_THRESHOLD


def highlight_fraction(frame, box, working_size, threshold=GLARE_CHANNEL_THRESHOLD):
    """Share of pixels inside a working-frame box with any channel >= threshold, or None.

    The box is mapped onto `frame`, which may be the native-resolution source.
    """
    if frame is None or frame.ndim != 3:
        return None
    height, width = frame.shape[:2]
    sx, sy = width / working_size[0], height / working_size[1]
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(width, int(x1 * sx))), max(0, min(width, int(round(x2 * sx))))))
    y1, y2 = sorted((max(0, min(height, int(y1 * sy))), max(0, min(height, int(round(y2 * sy))))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    crop = frame[y1:y2, x1:x2]
    step = max(1, min(crop.shape[:2]) // 64)  # subsample large native crops; the share is what matters
    return float((crop[::step, ::step] >= threshold).any(axis=2).mean())
