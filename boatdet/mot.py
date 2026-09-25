"""ByteTrack-style multi-object tracking with metric state estimation.

ByteTrack rather than BoT-SORT: no ReID weights ship with this project and the
target is a Jetson, so a deep appearance model would cost more than it returns.
Three additions carry the maritime case that plain IoU association does not:

* a second association stage over weak detections, so a boat that the detector
  briefly reports at low confidence (glare, wake, spray) keeps its identity;
* a motion gate over predicted position and scale, so a target crossing many box
  widths between frames, or growing quickly as it closes, still associates;
* an HSV histogram cost and AprilTag identity, which keep two boats apart while
  they cross. A visible tag is a hard constraint, appearance only a tie-break.
"""
import math

import cv2
import numpy as np

from boatdet.config import UNKNOWN_OBSTACLE_CLASS, MotConfig
from boatdet.egomotion import rotation_pixel_shift
from boatdet.motion import ConstantVelocityFilter, closing_speed, closing_speed_sigma, time_to_collision
from boatdet.tracking import ema

HIST_BINS = (8, 8)
HIST_RANGES = (0, 180, 0, 256)


def iou(box_a, box_b):
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def appearance_descriptor(frame, box):
    """Normalized hue/saturation histogram of a box, or None if the crop is empty."""
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([crop], [0, 1], None, list(HIST_BINS), list(HIST_RANGES))
    return cv2.normalize(histogram, histogram).flatten()


def appearance_similarity(left, right):
    """Correlation of two descriptors in [0, 1]; 0.5 (neutral) when either is missing."""
    if left is None or right is None:
        return 0.5
    return float(np.clip(cv2.compareHist(left, right, cv2.HISTCMP_CORREL), 0.0, 1.0))


def greedy_match(tracks, detections, cost):
    """Greedy lowest-cost assignment; cost returns None for a gated (forbidden) pair."""
    pairs = []
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            value = cost(track, detection)
            if value is not None:
                pairs.append((value, track_index, detection_index))
    matches, used_tracks, used_detections = [], set(), set()
    for _, track_index, detection_index in sorted(pairs, key=lambda item: item[0]):
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))
    return matches


class TrackState:
    """Everything published about one target; fields not yet measured stay None."""

    def __init__(self, track_id, detection, timestamp):
        self.track_id = track_id
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.bbox_xyxy = tuple(float(v) for v in detection.bbox_xyxy)
        self.detection_bbox_xyxy = self.bbox_xyxy
        self.confidence = float(detection.confidence)

        self.center_px = box_center(self.bbox_xyxy)
        self.water_contact_px = detection.water_contact_px
        self.water_contact_source = detection.water_contact_source

        self.age = 0
        self.hits = 0
        self.missed_frames = 0
        self.seconds_since_detection = 0.0

        self.distance_m = None
        self.bearing_deg = None
        self.position_m = None

        self.relative_velocity_mps = None
        self.vx = None
        self.vy = None
        self.ttc_s = None
        self.motion = None           # TargetEstimate: world-frame state from bearings-only analysis

        self.last_timestamp = timestamp
        self.source = detection.source
        self.apriltag_id = None
        self.measurement_source = None
        self.measurement_age = 0
        self.raw_distance_m = None   # measurement before distance calibration
        self.visual_bearing_deg = None  # fresh camera ray, before metric/TMA filtering
        self.visual_depth_m = None      # fresh raw monocular depth, independent of range fusion
        self.sky_fraction = detection.sky_fraction
        self.obstacle_fraction = detection.obstacle_fraction
        self.segmentation_confirmed = detection.segmentation_confirmed
        self.highlight_fraction = detection.highlight_fraction
        self.glare_suspect = False
        self.confirmed = False

    def as_record(self):
        """Flat dictionary for logging and for the viewer payloads."""
        return {'track_id': self.track_id, 'class_name': self.class_name, 'class_id': self.class_id,
                'confidence': round(self.confidence, 4), 'bbox_xyxy': [round(v, 2) for v in self.bbox_xyxy],
                'water_contact_px': None if self.water_contact_px is None else
                [round(v, 2) for v in self.water_contact_px],
                'water_contact_source': self.water_contact_source,
                'distance_m': self.distance_m, 'bearing_deg': self.bearing_deg,
                'relative_velocity_mps': self.relative_velocity_mps, 'vx': self.vx, 'vy': self.vy,
                'ttc_s': self.ttc_s, 'apriltag_id': self.apriltag_id,
                'measurement_source': self.measurement_source, 'source': self.source,
                'motion': None if self.motion is None else self.motion.as_dict(),
                'raw_distance_m': self.raw_distance_m,
                'visual_bearing_deg': self.visual_bearing_deg, 'visual_depth_m': self.visual_depth_m,
                'highlight_fraction': self.highlight_fraction, 'glare_suspect': self.glare_suspect,
                'age': self.age, 'hits': self.hits, 'missed_frames': self.missed_frames,
                'measurement_age': self.measurement_age,
                'tracking_status': 'predicted' if self.missed_frames else 'observed',
                'seconds_since_detection': self.seconds_since_detection,
                'last_timestamp': self.last_timestamp}


class Track:
    """A TrackState plus the filters and descriptors that maintain it."""

    def __init__(self, track_id, detection, timestamp, config):
        self.config = config
        self.state = TrackState(track_id, detection, timestamp)
        self.image_filter = ConstantVelocityFilter(2, config.measurement_noise_px,
                                                   config.process_noise_px, config.max_dt_s)
        self.image_filter.initialize(box_center(detection.bbox_xyxy), timestamp)
        self.metric_filter = None
        self.metric_dim = 0
        self.size = (detection.bbox_xyxy[2] - detection.bbox_xyxy[0],
                     detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
        self.descriptor = None
        self.predicted_center = box_center(detection.bbox_xyxy)

    @property
    def track_id(self):
        return self.state.track_id

    def predicted_box(self):
        cx, cy = self.predicted_center
        w, h = self.size
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def predict(self, timestamp, pixel_shift=None):
        """Advance the filters to `timestamp`; pixel_shift removes camera rotation."""
        center = self.image_filter.predict(timestamp)
        if center is not None and pixel_shift is not None:
            self.image_filter.shift(pixel_shift)
            center = self.image_filter.position
        if center is not None:
            self.predicted_center = (float(center[0]), float(center[1]))
        if self.metric_filter is not None:
            self.metric_filter.predict(timestamp)
            self._refresh_metric()
        self.state.age += 1

    def match(self, detection, timestamp, frame=None, descriptor=None):
        """Fold an associated detection into the track."""
        state = self.state
        state.detection_bbox_xyxy = tuple(float(v) for v in detection.bbox_xyxy)
        self.image_filter.correct(box_center(detection.bbox_xyxy), timestamp)
        self.predicted_center = tuple(float(v) for v in self.image_filter.position)
        size = (detection.bbox_xyxy[2] - detection.bbox_xyxy[0],
                detection.bbox_xyxy[3] - detection.bbox_xyxy[1])
        self.size = ema(self.config.box_smoothing, size, self.size)
        state.bbox_xyxy = self.box()
        state.center_px = self.predicted_center
        state.confidence = float(detection.confidence)
        state.last_timestamp = timestamp
        state.hits += 1
        state.missed_frames = 0
        state.seconds_since_detection = 0.0
        state.source = 'fused' if state.apriltag_id is not None else detection.source
        if detection.class_name and detection.source != 'segmentation':
            state.class_id, state.class_name = detection.class_id, detection.class_name
        state.sky_fraction = detection.sky_fraction
        state.obstacle_fraction = detection.obstacle_fraction
        state.segmentation_confirmed = detection.segmentation_confirmed
        self._update_highlight(detection)
        self._update_contact(detection)
        if state.hits >= self.config.min_hits:
            state.confirmed = True
        if descriptor is None and frame is not None:
            descriptor = appearance_descriptor(frame, detection.bbox_xyxy)
        if descriptor is not None:
            self.descriptor = descriptor if self.descriptor is None else \
                ema(self.config.appearance_smoothing, descriptor, self.descriptor)
            self.descriptor = np.asarray(self.descriptor, dtype=np.float32)

    def miss(self, timestamp):
        self.state.missed_frames += 1
        self.state.seconds_since_detection = max(0., timestamp - self.state.last_timestamp)
        self.state.bbox_xyxy = self.box()
        self.state.center_px = self.predicted_center
        self.state.water_contact_px = None
        self.state.water_contact_source = None

    def box(self, frame_size=None):
        box = self.predicted_box()
        if frame_size is None:
            return tuple(float(v) for v in box)
        width, height = frame_size
        w, h = self.size
        cx = min(max(self.predicted_center[0], w / 2), width - w / 2)
        cy = min(max(self.predicted_center[1], h / 2), height - h / 2)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def _update_highlight(self, detection):
        """Smoothed highlight share of the matched boxes, flagged for review above the threshold."""
        value = detection.highlight_fraction
        if value is None:
            return
        state = self.state
        state.highlight_fraction = value if state.highlight_fraction is None else \
            self.config.box_smoothing * value + (1 - self.config.box_smoothing) * state.highlight_fraction
        state.glare_suspect = state.highlight_fraction >= self.config.glare_suspect_fraction

    def _update_contact(self, detection):
        """Smooth the water-contact point; a changed source restarts the smoothing."""
        point = detection.water_contact_px
        state = self.state
        if point is None:
            state.water_contact_px = None
            state.water_contact_source = None
            return
        if state.water_contact_px is None or state.water_contact_source != detection.water_contact_source:
            state.water_contact_px = tuple(float(v) for v in point)
        else:
            state.water_contact_px = ema(self.config.box_smoothing, point, state.water_contact_px)
        state.water_contact_source = detection.water_contact_source

    def observe(self, position_m=None, distance_m=None, bearing_deg=None, source=None,
                timestamp=None, raw_distance_m=None, measurement_noise=None):
        """Fuse a metric measurement and refresh distance, velocity and TTC.

        A two component position is filtered in the water plane; a bare range is
        filtered on its own, which still yields a closing speed but no heading.
        """
        state = self.state
        if position_m is not None:
            measurement, dim = np.asarray(position_m, dtype=float).ravel(), 2
        elif distance_m is not None:
            measurement, dim = np.array([float(distance_m)]), 1
        else:
            return
        if not np.isfinite(measurement).all():
            return
        if self.metric_filter is None or self.metric_dim != dim:
            self.metric_filter = ConstantVelocityFilter(dim, self.config.measurement_noise_m,
                                                        self.config.process_noise_m, self.config.max_dt_s)
            self.metric_dim = dim
        self.metric_filter.correct(measurement, timestamp, measurement_noise)
        state.measurement_source = source
        state.measurement_age = 0
        state.raw_distance_m = raw_distance_m
        if bearing_deg is not None and math.isfinite(bearing_deg):
            state.bearing_deg = round(float(bearing_deg), 2)
        self._refresh_metric()

    def _refresh_metric(self):
        """Recompute published metric fields from the filter state."""
        state = self.state
        if self.metric_filter is None or not self.metric_filter.initialized:
            return
        position, velocity = self.metric_filter.position, self.metric_filter.velocity
        if self.metric_dim == 2:
            distance = float(np.linalg.norm(position))
            state.position_m = (round(float(position[0]), 3), round(float(position[1]), 3))
            state.vx, state.vy = round(float(velocity[0]), 3), round(float(velocity[1]), 3)
            state.bearing_deg = round(math.degrees(math.atan2(position[1], position[0])), 2)
            state.relative_velocity_mps = round(float(np.linalg.norm(velocity)), 3)
            closing = closing_speed(position, velocity)
            sigma = closing_speed_sigma(position, self.metric_filter.covariance, 2)
        else:
            distance = float(position[0])
            state.position_m = None
            state.vx = state.vy = None
            state.relative_velocity_mps = round(abs(float(velocity[0])), 3)
            closing = -float(velocity[0])
            sigma = math.sqrt(max(0.0, float(self.metric_filter.covariance[1, 1])))
        state.distance_m = round(distance, 3) if math.isfinite(distance) and distance > 0 else None
        state.ttc_s = time_to_collision(state.distance_m, closing,
                                        self.config.ttc_min_closing_speed,
                                        self.config.ttc_max_reasonable_value,
                                        sigma, self.config.ttc_min_sigma)
        if state.ttc_s is not None:
            state.ttc_s = round(state.ttc_s, 2)


class MultiObjectTracker:
    """Detections in, persistent TrackStates out.

    `update` is the only entry point per frame; metric measurements are attached
    afterwards through `observe`, which keeps range estimation out of tracking.
    """

    def __init__(self, working_size, config=MotConfig(), camera_matrix=None):
        self.config = config
        self.width, self.height = working_size
        self.camera_matrix = camera_matrix
        self.max_center_distance = min(working_size) * config.max_center_distance_frac
        self.tracks = []
        self.next_id = 1
        self.last_orientation = None
        self.tag_to_track = {}
        self._descriptors = {}

    def update(self, detections, timestamp, frame=None, tags=(), orientation=None):
        """Associate one frame of detections; returns the confirmed tracks."""
        detections = [d for d in detections if d.confidence >= self.config.low_conf_threshold]
        self._descriptors = {id(d): appearance_descriptor(frame, d.bbox_xyxy) for d in detections} \
            if self.config.appearance_weight > 0 else {}
        shift_source = self._ego_shift(orientation)
        for track in self.tracks:
            track.predict(timestamp, shift_source(track.predicted_center))
            track.state.measurement_age += 1
        # Expire extended-coast identities before association, including after a
        # capture gap. Preserve the legacy lifecycle when this mode is disabled.
        self._prune(timestamp, before_association=True)

        high = [d for d in detections if d.confidence >= self.config.conf_threshold]
        low = [d for d in detections if d.confidence < self.config.conf_threshold]
        open_tracks = list(self.tracks)
        matched = self._tag_matches(open_tracks, high + low, tags)
        for track, detection in matched:
            open_tracks.remove(track)
            (high if detection in high else low).remove(detection)

        # ByteTrack order: strong detections, then weak ones, then the motion gate.
        for pool, gate_cost in ((high, self._appearance_iou_cost(self.config.iou_threshold)),
                                (low, self._iou_cost(self.config.second_iou_threshold)),
                                (high, self._motion_cost())):
            pairs = greedy_match(open_tracks, pool, gate_cost)
            for track_index, detection_index in pairs:
                matched.append((open_tracks[track_index], pool[detection_index]))
            for track_index, detection_index in sorted(pairs, reverse=True):
                open_tracks.pop(track_index)
            for _, detection_index in sorted(pairs, key=lambda p: p[1], reverse=True):
                pool.pop(detection_index)

        for track, detection in matched:
            track.match(detection, timestamp, frame, self._descriptors.get(id(detection)))
        for track in open_tracks:
            track.miss(timestamp)
        for detection in high:
            self._start(detection, timestamp)

        self._assign_tags(tags)
        self._prune(timestamp)
        return self.confirmed()

    def confirmed(self):
        return [track.state for track in self.tracks if track.state.confirmed]

    def states(self):
        return [track.state for track in self.tracks]

    def observe(self, track_id, position_m=None, distance_m=None, bearing_deg=None,
                source=None, timestamp=None, raw_distance_m=None, measurement_noise=None):
        """Attach a metric measurement to one track by id."""
        for track in self.tracks:
            if track.track_id == track_id:
                track.observe(position_m, distance_m, bearing_deg, source, timestamp, raw_distance_m,
                              measurement_noise)
                return track.state
        return None

    def _ego_shift(self, orientation):
        """Per-point pixel shift caused by platform rotation since the last frame."""
        previous, self.last_orientation = self.last_orientation, orientation or self.last_orientation
        if orientation is None or previous is None or self.camera_matrix is None:
            return lambda point: None
        return lambda point: rotation_pixel_shift(previous, orientation, self.camera_matrix, point)

    def _appearance_iou_cost(self, gate):
        weight = self.config.appearance_weight

        def cost(track, detection):
            overlap = iou(track.predicted_box(), detection.bbox_xyxy)
            if overlap < gate or not self._class_compatible(track, detection) or not self._reacquire_allowed(track, detection):
                return None
            similarity = appearance_similarity(track.descriptor, self._descriptors.get(id(detection)))
            return (1 - overlap) + weight * (1 - similarity)
        return cost

    def _iou_cost(self, gate):
        def cost(track, detection):
            overlap = iou(track.predicted_box(), detection.bbox_xyxy)
            return 1 - overlap if overlap >= gate and self._class_compatible(track, detection) and self._reacquire_allowed(track, detection) else None
        return cost

    def _motion_cost(self):
        """Fallback for fast targets: predicted centers close and scale plausible."""
        weight = self.config.appearance_weight

        def cost(track, detection):
            if not self._class_compatible(track, detection) or not self._reacquire_allowed(track, detection):
                return None
            tx, ty = track.predicted_center
            dx, dy = box_center(detection.bbox_xyxy)
            distance = math.hypot(dx - tx, dy - ty)
            gate = self._center_gate(track)
            if distance > gate:
                return None
            track_area, detection_area = box_area(track.predicted_box()), box_area(detection.bbox_xyxy)
            if min(track_area, detection_area) <= 0:
                return None
            ratio = max(track_area, detection_area) / min(track_area, detection_area)
            if ratio > self.config.max_scale_ratio:
                return None
            similarity = appearance_similarity(track.descriptor, self._descriptors.get(id(detection)))
            return 1 + distance / max(gate, 1e-6) + weight * (1 - similarity)
        return cost

    def _extended_coast(self, track):
        return (self.config.occlusion_seconds > 0 and track.state.confirmed and
                track.state.class_name in ('boat', 'ship', 'vessel'))

    def _center_gate(self, track):
        if not self._extended_coast(track) or not track.state.missed_frames:
            return self.max_center_distance
        # Uncertainty expands the search, but never to the whole image.
        sigma = math.sqrt(max(0., float(np.linalg.eigvalsh(track.image_filter.covariance[:2, :2])[-1])))
        return min(2 * self.max_center_distance, max(self.max_center_distance, 3 * sigma))

    def _reacquire_allowed(self, track, detection):
        if not self._extended_coast(track) or not track.state.missed_frames:
            return True
        area_a, area_b = box_area(track.predicted_box()), box_area(detection.bbox_xyxy)
        if min(area_a, area_b) <= 0 or max(area_a, area_b) / min(area_a, area_b) > self.config.max_scale_ratio:
            return False
        if math.dist(track.predicted_center, box_center(detection.bbox_xyxy)) > self._center_gate(track):
            return False
        descriptor = self._descriptors.get(id(detection))
        if track.descriptor is None or descriptor is None:
            # With no appearance evidence, weak detections cannot reclaim a hidden ID.
            return detection.confidence >= self.config.conf_threshold
        return appearance_similarity(track.descriptor, descriptor) >= self.config.reacquire_appearance

    def _class_compatible(self, track, detection):
        """Named classes must agree; extended-coast vessels cannot become unknown obstacles."""
        if self._extended_coast(track) and detection.class_name not in ('boat', 'ship', 'vessel'):
            return False  # the occluding pier/obstacle must not become the hidden boat
        if detection.class_name in (None, UNKNOWN_OBSTACLE_CLASS) or \
                track.state.class_name == UNKNOWN_OBSTACLE_CLASS:
            return True
        return detection.class_name == track.state.class_name

    def _tag_matches(self, tracks, detections, tags):
        """Detections holding a tag already bound to a track are matched first."""
        pairs = []
        used_tracks, used_detections = set(), set()
        for tag in tags:
            track_id = self.tag_to_track.get(tag.tag_id)
            track = next((t for t in tracks if t.track_id == track_id), None)
            if track is None or track.track_id in used_tracks:
                continue
            inside = [d for d in detections if id(d) not in used_detections and self._contains(d, tag.center_px)]
            if len(inside) != 1:
                continue
            pairs.append((track, inside[0]))
            used_tracks.add(track.track_id)
            used_detections.add(id(inside[0]))
        return pairs

    def _assign_tags(self, tags):
        """Bind a visible tag to the track whose box contains it; identity survives its loss."""
        for tag in tags:
            owner = None
            for track in self.tracks:
                if self._point_in_box(tag.center_px, track.state.bbox_xyxy):
                    if owner is not None:
                        owner = None  # ambiguous during a crossing: leave the binding alone
                        break
                    owner = track
            if owner is None:
                continue
            previous = self.tag_to_track.get(tag.tag_id)
            if previous is not None and previous != owner.track_id:
                for track in self.tracks:
                    if track.track_id == previous and track.state.apriltag_id == tag.tag_id:
                        track.state.apriltag_id = None
            self.tag_to_track[tag.tag_id] = owner.track_id
            owner.state.apriltag_id = tag.tag_id
            owner.state.source = 'fused'

    @staticmethod
    def _point_in_box(point, box):
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    def _contains(self, detection, point):
        return self._point_in_box(point, detection.bbox_xyxy)

    def _start(self, detection, timestamp):
        track = Track(self.next_id, detection, timestamp, self.config)
        track.state.hits = 1
        track.state.confirmed = self.config.min_hits <= 1
        track._update_highlight(detection)
        track.descriptor = self._descriptors.get(id(detection))
        self.tracks.append(track)
        self.next_id += 1

    def _prune(self, timestamp, before_association=False):
        """Drop tracks unmatched for too many frames or seconds, or coasted out of view.

        Normally both frame and second limits apply. Extended vessel coasting
        uses its seconds limit alone so the occlusion window is independent of fps.
        """
        kept = []
        for track in self.tracks:
            extended = self._extended_coast(track)
            if before_association and not extended:
                kept.append(track)
                continue
            cx, cy = track.predicted_center
            margin = 0.25
            outside = not (-margin * self.width <= cx <= (1 + margin) * self.width and
                           -margin * self.height <= cy <= (1 + margin) * self.height)
            elapsed = timestamp - track.state.last_timestamp
            timeout = self.config.occlusion_seconds if extended else self.config.max_coast_s
            stale = (before_association or track.state.missed_frames > 0) and elapsed > timeout
            frames_expired = not extended and track.state.missed_frames > self.config.max_missed_frames
            if frames_expired or stale or elapsed < 0 or \
                    (outside and track.state.missed_frames > 0):
                self.tag_to_track = {tag: tid for tag, tid in self.tag_to_track.items()
                                     if tid != track.track_id}
                continue
            kept.append(track)
        self.tracks = kept
