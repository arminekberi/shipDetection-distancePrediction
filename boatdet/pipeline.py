"""Perception pipeline: detection, segmentation, fusion, tracking and range.

    frame -> detector ----\\
                           fusion -> tracker -> ego-motion -> state estimation -> tracks
    frame -> segmentation /

Every branch beyond the detector is optional. With segmentation off, fusion off
and no calibration, the tracker still produces identified, filtered tracks; each
extra input only adds measurements.
"""
import math
from dataclasses import dataclass, field

from boatdet.apriltag import NullAprilTagSource
from boatdet.config import FusionConfig, MotConfig, SegmentationConfig, TmaConfig
from boatdet.depth import distance_in_box
from boatdet.detection import as_detections
from boatdet.egomotion import NullEgoMotion
from boatdet.fusion import fuse
from boatdet.glare import highlight_fraction
from boatdet.geometry import (bearing_from_pixel, bearing_from_ray, position_from_range,
                              water_intersection)
from boatdet.mot import MultiObjectTracker, iou
from boatdet.segmentation import extract_obstacles
from boatdet.telemetry import StageTimer
from boatdet.tma import BearingObservation, MultiTargetTma, bearing_sigma


@dataclass
class PipelineResult:
    """Everything one frame produced; the overlay and the logs read only this."""
    frame_index: int
    timestamp: float
    tracks: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    obstacles: list = field(default_factory=list)
    unfused_obstacles: list = field(default_factory=list)
    segmentation: object = None
    orientation: object = None
    timer: StageTimer = None
    roi_band: tuple = None   # (top, bottom) rows of the detector search band, working coords
    own_ship: object = None  # OwnShipState the bearings were placed from, None without navigation


class PerceptionPipeline:
    """One frame in, tracks out; stages are timed separately."""

    def __init__(self, detector, working_size, mot_config=MotConfig(),
                 segmentation_runner=None, segmentation_config=SegmentationConfig(),
                 fusion_config=FusionConfig(), calibration=None, ego_motion=None,
                 tag_source=None, timer=None, depth_calibration=(1.0, 0.0),
                 own_ship=None, tma_config=TmaConfig()):
        if segmentation_runner is not None and fusion_config.enabled and \
                fusion_config.unknown_conf < mot_config.conf_threshold:
            raise ValueError(
                f'FUSION_UNKNOWN_CONF ({fusion_config.unknown_conf}) is below TRACK_CONF_THRESHOLD '
                f'({mot_config.conf_threshold}); segmentation-only obstacles could never open a track')
        self.detector = detector
        self.working_size = working_size
        self.mot_config = mot_config
        self.segmentation_runner = segmentation_runner
        self.segmentation_config = segmentation_config
        self.fusion_config = fusion_config
        # Calibration must already describe working_size; rescale with CameraCalibration.scaled.
        self.calibration = calibration
        self.depth_calibration = depth_calibration
        self.ego_motion = ego_motion or NullEgoMotion()
        self.tag_source = tag_source or NullAprilTagSource()
        self.timer = timer or StageTimer()
        self.tracker = MultiObjectTracker(
            working_size, mot_config,
            camera_matrix=None if calibration is None else calibration.camera_matrix)
        # Bearings-only motion analysis runs only with both of its inputs: a lens that
        # turns pixels into angles, and navigation that says where those angles were taken
        # from. Either one missing and the world-frame state is not derivable at all.
        if own_ship is not None and calibration is None:
            raise ValueError('own-ship navigation was supplied without a camera calibration; '
                             'bearings need intrinsics before they mean anything')
        self.own_ship = own_ship
        self.tma_config = tma_config
        self.tma = MultiTargetTma(tma_config) if own_ship is not None else None
        self.bearing_sigma_rad = None if own_ship is None else bearing_sigma(
            tma_config.pixel_sigma_px, calibration.camera_matrix[0, 0],
            math.radians(tma_config.heading_sigma_deg), math.radians(tma_config.mount_sigma_deg))

    def process(self, frame, native_frame=None, timestamp=0.0, frame_index=0, depth_m=None):
        """Run one frame through every enabled stage."""
        self.timer.start_frame()
        with self.timer.stage('detector'):
            candidates = self.detector.candidates(native_frame if native_frame is not None else frame,
                                                  self.working_size)
            detections = as_detections(candidates)
            source = native_frame if native_frame is not None else frame
            for detection in detections:
                detection.highlight_fraction = highlight_fraction(source, detection.bbox_xyxy, self.working_size)

        segmentation = None
        obstacles = []
        if self.segmentation_runner is not None:
            with self.timer.stage('segmentation'):
                segmentation = self.segmentation_runner.process(frame, frame_index)
                if segmentation is not None:
                    obstacles = extract_obstacles(segmentation, self.segmentation_config.min_obstacle_area)
                    self.timer.record('segmentation_inference',
                                      0.0 if segmentation.reused else segmentation.inference_time_ms)

        tags = self.tag_source.detect(frame, timestamp, frame_index)
        if self.fusion_config.apriltag_vessel_ids:
            tags = [tag for tag in tags if tag.tag_id in self.fusion_config.apriltag_vessel_ids]
        orientation = self.ego_motion.get_orientation(timestamp)

        with self.timer.stage('fusion'):
            detections = fuse(detections, segmentation, obstacles, tags, self.fusion_config)

        with self.timer.stage('tracking'):
            tracks = self.tracker.update(detections, timestamp, frame, tags, orientation)

        with self.timer.stage('ranging'):
            for track in tracks:
                track.visual_bearing_deg = None
                track.visual_depth_m = None
                if not track.missed_frames:
                    x1, y1, x2, y2 = track.detection_bbox_xyxy
                    track.visual_bearing_deg = bearing_from_ray(((x1+x2)/2, (y1+y2)/2), self.calibration, orientation)
                    if depth_m is not None:
                        track.visual_depth_m = distance_in_box(depth_m, tuple(int(round(v)) for v in (x1,y1,x2,y2)))
                self._measure(track, tags, orientation, depth_m, timestamp)

        own_ship = None
        if self.tma is not None:
            with self.timer.stage('tma'):
                own_ship = self._analyze_motion(tracks, orientation, timestamp)

        self.timer.end_frame()
        unfused = [o for o in obstacles
                   if not any(iou(o.bbox_xyxy, t.bbox_xyxy) >= self.fusion_config.unknown_iou
                              for t in tracks)]
        return PipelineResult(frame_index, timestamp, tracks, detections, obstacles, unfused,
                              segmentation, orientation, self.timer,
                              self._roi_band(native_frame, frame), own_ship)

    def _analyze_motion(self, tracks, orientation, timestamp):
        """World-frame state of every track from its bearing and our own navigation.

        The bearing is taken from the box center through the full ray, so platform
        roll does not leak into it. A frame the navigation does not cover produces
        no observation at all: an interpolated own position would be read
        downstream as target motion.
        """
        own_ship = self.own_ship.at(timestamp)
        for track_id in set(self.tma.estimators) - {track.track_id for track in tracks}:
            self.tma.drop(track_id)
        if own_ship is None:
            return None
        for track in tracks:
            if track.missed_frames:
                continue  # a coasted box is a prediction, not a new bearing
            bearing = bearing_from_ray(track.center_px, self.calibration, orientation)
            if bearing is None:
                continue
            track.motion = self.tma.observe(track.track_id, BearingObservation.from_relative(
                own_ship, bearing, self.bearing_sigma_rad, timestamp,
                self.tma_config.lever_forward_m, self.tma_config.lever_starboard_m))
        return own_ship

    def _measure(self, track, tags, orientation, depth_m, timestamp):
        """Best available range for one track, highest-confidence source first.

        A tag that carries a pose ends the search. A tag that only carries a
        bearing (what the deployed pipeline reports) refines the direction and
        leaves the range to the next source.
        """
        tag = next((t for t in tags if t.tag_id == track.apriltag_id), None)
        tag_bearing = tag.bearing_deg if tag is not None else None
        if tag is not None and tag.has_pose:
            position = tag.position_m or position_from_range(tag.distance_m, tag.bearing_deg)
            self.tracker.observe(track.track_id, position_m=position,
                                 distance_m=None if position else tag.distance_m,
                                 bearing_deg=tag.bearing_deg, source='apriltag',
                                 timestamp=timestamp)
            return
        # A predicted box/contact is not a fresh observation. Sampling background
        # there invents range updates and rewinds the metric filter to the last hit.
        if track.missed_frames:
            return
        if self.calibration is not None and track.water_contact_px is not None:
            estimate = water_intersection(track.water_contact_px, self.calibration, orientation)
            if estimate is not None:
                self.tracker.observe(track.track_id,
                                     position_m=(estimate.x_relative_m, estimate.y_relative_m),
                                     bearing_deg=tag_bearing if tag_bearing is not None
                                     else estimate.bearing_deg,
                                     source='water_plane' if tag is None else 'water_plane+apriltag',
                                     timestamp=timestamp,
                                     measurement_noise=self.mot_config.measurement_noise_m)
                return
        if depth_m is not None:
            raw = distance_in_box(depth_m, self._depth_box(track))
            if raw is not None:
                scale, offset = self.depth_calibration
                distance = scale * raw + offset
                if distance <= 0:
                    return
                bearing = tag_bearing if tag_bearing is not None else \
                    bearing_from_pixel(track.center_px[0], self.calibration)
                position = position_from_range(distance, bearing)
                self.tracker.observe(track.track_id, position_m=position,
                                     distance_m=None if position else distance,
                                     bearing_deg=bearing,
                                     source='depth' if tag is None else 'depth+apriltag',
                                     timestamp=timestamp, raw_distance_m=raw,
                                     measurement_noise=self.mot_config.depth_noise_m)

    def _roi_band(self, native_frame, frame):
        """Search band of a two-stage detector in working rows, or None for full-frame detection."""
        band = getattr(self.detector, 'last_band', None)
        if band is None:
            return None
        source = native_frame if native_frame is not None else frame
        scale = self.working_size[1] / source.shape[0]
        return band[0] * scale, band[1] * scale

    @staticmethod
    def _depth_box(track):
        """Depth is sampled above the waterline, so a reflection cannot pull the median."""
        x1, y1, x2, y2 = (int(round(v)) for v in track.bbox_xyxy)
        if track.water_contact_px is not None:
            contact = int(round(track.water_contact_px[1]))
            if contact - y1 >= 2:
                y2 = min(y2, contact)
        return x1, y1, x2, y2
