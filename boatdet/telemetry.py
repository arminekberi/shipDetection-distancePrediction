"""Per-stage timing and per-track logging.

Console output stays quiet unless debug logging is on: a per-frame print on a
live rig costs more than the stage it reports.
"""
import csv
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

TRACK_CSV_HEADER = ['timestamp', 'frame_id', 'track_id', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2',
                    'water_contact_u', 'water_contact_v', 'water_contact_source', 'distance_m',
                    'bearing_deg', 'relative_velocity_mps', 'vx', 'vy', 'ttc_s', 'apriltag_id',
                    'measurement_source', 'highlight_fraction', 'glare_suspect',
                    'missed_frames', 'measurement_age', 'tracking_status', 'seconds_since_detection',
                    'visual_bearing_deg', 'visual_depth_m', 'visual_center_u', 'visual_center_v',
                    # bearings-only motion analysis; empty without own-ship navigation.
                    # tma_range_m is meaningful only where tma_status is 'converged'; the
                    # interval and the parallax say how much the rest of the row is worth.
                    'tma_x_m', 'tma_y_m', 'tma_speed_mps', 'tma_course_deg', 'tma_range_m',
                    'tma_range_low_m', 'tma_range_high_m', 'tma_parallax', 'tma_status']
SEGMENTATION_CSV_HEADER = ['frame_id', 'inference_time_ms', 'obstacle_count', 'reused']


class StageTimer:
    """Milliseconds per pipeline stage plus the effective frame rate.

    Timings are kept per frame so a slow segmentation frame stays visible
    instead of being averaged away.
    """

    def __init__(self, window=30):
        self.stages = {}
        self.totals = {}
        self.frames = 0
        self._frame_times = deque(maxlen=window)
        self._frame_start = None

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000)

    def record(self, name, milliseconds):
        self.stages[name] = float(milliseconds)
        self.totals[name] = self.totals.get(name, 0.0) + float(milliseconds)

    def start_frame(self):
        self._frame_start = time.perf_counter()
        self.stages = {}

    def end_frame(self):
        """Total ms of the frame just finished."""
        if self._frame_start is None:
            return 0.0
        elapsed = (time.perf_counter() - self._frame_start) * 1000
        self.record('total', elapsed)
        self._frame_times.append(elapsed)
        self.frames += 1
        self._frame_start = None
        return elapsed

    @property
    def fps(self):
        """Effective frame rate over the recent window, 0 before the first frame."""
        if not self._frame_times:
            return 0.0
        average = sum(self._frame_times) / len(self._frame_times)
        return 1000.0 / average if average > 0 else 0.0

    def summary(self):
        """Mean ms per stage over the whole run, plus the frame count and rate."""
        frames = max(self.frames, 1)
        summary = {f'{name}_ms': round(total / frames, 2) for name, total in self.totals.items()}
        return dict(summary, frames=self.frames, fps=round(self.fps, 2))

    def format(self):
        return '  '.join(f'{name}: {value:.0f} ms' for name, value in self.stages.items()) + \
               f'  ({self.fps:.1f} fps)'


class TrackLogger:
    """CSV of every tracked target, and of the segmentation branch beside it."""

    def __init__(self, path, segmentation_path=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open('w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow(TRACK_CSV_HEADER)
        self._segmentation_file = None
        if segmentation_path:
            self._segmentation_file = Path(segmentation_path).open('w', newline='')
            self._segmentation_writer = csv.writer(self._segmentation_file)
            self._segmentation_writer.writerow(SEGMENTATION_CSV_HEADER)

    def write_tracks(self, frame_id, timestamp, tracks):
        for track in tracks:
            x1, y1, x2, y2 = track.bbox_xyxy
            contact = track.water_contact_px or (None, None)
            self._writer.writerow([
                f'{timestamp:.6f}', frame_id, track.track_id, track.class_name,
                f'{track.confidence:.4f}', f'{x1:.1f}', f'{y1:.1f}', f'{x2:.1f}', f'{y2:.1f}',
                '' if contact[0] is None else f'{contact[0]:.1f}',
                '' if contact[1] is None else f'{contact[1]:.1f}',
                track.water_contact_source or '',
                '' if track.distance_m is None else f'{track.distance_m:.3f}',
                '' if track.bearing_deg is None else f'{track.bearing_deg:.2f}',
                '' if track.relative_velocity_mps is None else f'{track.relative_velocity_mps:.3f}',
                '' if track.vx is None else f'{track.vx:.3f}',
                '' if track.vy is None else f'{track.vy:.3f}',
                '' if track.ttc_s is None else f'{track.ttc_s:.2f}',
                '' if track.apriltag_id is None else track.apriltag_id,
                track.measurement_source or '',
                '' if track.highlight_fraction is None else f'{track.highlight_fraction:.3f}',
                int(track.glare_suspect), track.missed_frames, track.measurement_age,
                'predicted' if track.missed_frames else 'observed',
                f'{track.seconds_since_detection:.3f}',
                '' if track.visual_bearing_deg is None else f'{track.visual_bearing_deg:.6f}',
                '' if track.visual_depth_m is None else f'{track.visual_depth_m:.6f}',
                '' if track.missed_frames else f'{(track.detection_bbox_xyxy[0]+track.detection_bbox_xyxy[2])/2:.6f}',
                '' if track.missed_frames else f'{(track.detection_bbox_xyxy[1]+track.detection_bbox_xyxy[3])/2:.6f}',
                *self._motion_columns(track.motion)])

    @staticmethod
    def _motion_columns(motion):
        """The bearings-only columns, blank for a track no navigation could place."""
        if motion is None:
            return [''] * 9
        return [f'{motion.x_m:.1f}', f'{motion.y_m:.1f}', f'{motion.speed_mps:.2f}',
                f'{motion.course_deg:.1f}', f'{motion.range_m:.1f}', f'{motion.range_low_m:.1f}',
                f'{motion.range_high_m:.1f}', f'{motion.peak_parallax_ratio:.5f}', motion.status]

    def write_segmentation(self, frame_id, result, obstacle_count):
        if self._segmentation_file is None or result is None:
            return
        self._segmentation_writer.writerow([frame_id, f'{result.inference_time_ms:.1f}',
                                            obstacle_count, int(result.reused)])

    def close(self):
        self._file.close()
        if self._segmentation_file is not None:
            self._segmentation_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
