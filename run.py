"""Track boats and estimate their metric distance with Depth Anything V2.

Source is a camera index (live, async depth) or a video file (every frame, synchronous).
`--tracker bytetrack` follows every visible target with persistent ids, relative
velocity and time-to-collision, optionally fused with maritime segmentation;
the other trackers keep following a single target exactly as before.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2

from boatdet import config as defaults
from boatdet import overlay as overlays
from boatdet.apriltag import ReplayAprilTagSource
from boatdet.config import (FusionConfig, MotConfig, OverlayConfig, RoiConfig, SegmentationConfig,
                            TmaConfig)
from boatdet.depth import DEFAULT_INFER_SIZE, DEFAULT_MODEL, DepthEstimator, DepthWorker, distance_in_box
from boatdet.detection import YoloDetector
from boatdet.device import get_device
from boatdet.egomotion import SampledEgoMotion
from boatdet.geometry import CameraCalibration
from boatdet.pipeline import PerceptionPipeline
from boatdet.tma import OwnShipTrack
from boatdet.proposal import TwoStageDetector, build_proposer
from boatdet.segmentation import SegmentationRunner, SegmentationUnavailable, UltralyticsSegmentation
from boatdet.telemetry import TrackLogger
from boatdet.tracking import (CsrtTracker, DetectionTracker, KalmanTracker, ManualCsrtTracker,
                              TrackerConfig)
from boatdet.video import (WORKING_SIZE, RawCapture, color_fraction, frame_timer, h264_writer,
                           open_video, video_fps)

TRACKERS = {'csrt': CsrtTracker, 'kalman': KalmanTracker, 'none': DetectionTracker}
MULTI_TRACKER = 'bytetrack'  # multi-object mode, handled by boatdet.pipeline
CSV_HEADER = ['frame', 'distance_raw_m', 'distance_smoothed_m', 'inference_ms', 'time_s', 'distance_model_m']
WINDOW = 'Depth Anything V2 - distance to tracked object'


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def parse_box(value, width, height):
    x1, y1, x2, y2 = (int(v) for v in value.split(','))
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= 2 and y2 - y1 >= 2):
        raise ValueError('target must be a nonempty box inside the working frame')
    return x1, y1, x2, y2


def parse_grid(value):
    rows, cols = (int(v) for v in value.lower().split('x'))
    if min(rows, cols) < 1:
        raise ValueError
    return rows, cols


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=parse_source, default=0, help='camera index or video path')
    p.add_argument('--model', default=DEFAULT_MODEL, help='Hugging Face depth model id')
    p.add_argument('--width', type=int, default=WORKING_SIZE[0], help='working/display width')
    p.add_argument('--height', type=int, default=WORKING_SIZE[1], help='working/display height')
    p.add_argument('--infer-size', type=int, default=DEFAULT_INFER_SIZE, help='depth model input size')
    p.add_argument('--fp16', action='store_true', help='half-precision depth, faster, less accurate')
    p.add_argument('--depth-range', help='fixed NEAR,FAR meters for the depth colors and the colorbar; '
                                         'by default the range follows the scene')
    p.add_argument('--smoothing', type=float, default=0.3, help='distance EMA factor, 1 = none')
    p.add_argument('--calib-scale', type=float, default=1.0,
                   help='corrected = scale * model + offset (calibrate.py); identity by default, '
                        'fit per camera/model from independent measurements')
    p.add_argument('--calib-offset', type=float, default=0.0, help='calibration offset, meters')
    p.add_argument('--no-display', action='store_true', help='headless; needs --target or --yolo-weights')
    p.add_argument('--target', help='manual x1,y1,x2,y2 box in working coords, tracked by CSRT')
    p.add_argument('--output', help='annotated mp4 (H.264): camera and depth map, boxes on both, legend')
    p.add_argument('--log', help='per-frame distance CSV')
    p.add_argument('--max-frames', type=int, help='stop after N frames')
    p.add_argument('--sync', action='store_true', help='blocking depth per frame; implied for files')

    y = p.add_argument_group('YOLO detection (with --yolo-weights)')
    y.add_argument('--yolo-weights', help='boat detector .pt; enables automatic tracking')
    y.add_argument('--tracker', default='csrt', choices=(*TRACKERS, MULTI_TRACKER),
                   help='csrt: appearance bridge; kalman: motion gate; none: per-frame detection only; '
                        'bytetrack: all targets, persistent ids, velocity and TTC')
    y.add_argument('--yolo-conf', type=float, default=TrackerConfig.conf, help='trusted detection')
    y.add_argument('--yolo-low-conf', type=float, default=TrackerConfig.low_conf, help='weak detection, last resort')
    y.add_argument('--yolo-imgsz', type=int, default=960, help='YOLO inference size')
    y.add_argument('--box-smoothing', type=float, default=TrackerConfig.box_smoothing, help='box EMA factor')
    y.add_argument('--max-jump-frac', type=float, default=TrackerConfig.max_jump_frac,
                   help='reject center jumps above this fraction of the short side')
    y.add_argument('--grace-frames', type=int, default=TrackerConfig.grace_frames, help='frames coasted before loss')
    y.add_argument('--reacquire-frames', type=int, default=TrackerConfig.reacquire_frames,
                   help='unanchored frames before a confident hit may relocate the track')
    y.add_argument('--tile-grid', help='e.g. 2x2: detect on overlapping native-resolution tiles')
    y.add_argument('--tile-overlap', type=float, default=0.2, help='tile overlap fraction')
    y.add_argument('--augment', action='store_true', help='YOLO test-time augmentation')
    y.add_argument('--clahe', action='store_true', help='experimental CLAHE before detection')
    y.add_argument('--roi', default=defaults.ROI_MODE, choices=('none', 'band', 'horizon', 'segmentation'),
                   help='detect only inside a horizontal band: fixed, around the detected horizon, '
                        'or above the segmented water; combines with --tile-grid')
    y.add_argument('--roi-band', default=','.join(map(str, defaults.ROI_BAND)),
                   help='fixed band (and horizon fallback) as TOP,BOTTOM fractions of frame height')

    m = p.add_argument_group('multi-object tracking (--tracker bytetrack)')
    m.add_argument('--track-max-missed-frames', type=int, default=defaults.TRACK_MAX_MISSED_FRAMES,
                   help='frames a track is predicted through missed detections before it is dropped')
    m.add_argument('--track-min-hits', type=int, default=defaults.TRACK_MIN_HITS,
                   help='matched frames before a track is reported')
    m.add_argument('--track-occlusion-seconds', type=float, default=defaults.TRACK_OCCLUSION_SECONDS,
                   help='keep confirmed vessel IDs through this many seconds without detection; '
                        'overrides frame/coast limits for vessels, 0 disables (try 8)')
    m.add_argument('--track-iou-threshold', type=float, default=defaults.TRACK_IOU_THRESHOLD,
                   help='minimum IoU for the first association stage')
    m.add_argument('--track-center-frac', type=float, default=defaults.TRACK_MAX_CENTER_DISTANCE_FRAC,
                   help='motion gate: max center jump per frame, fraction of the short side')
    m.add_argument('--track-appearance-weight', type=float, default=defaults.TRACK_APPEARANCE_WEIGHT,
                   help='weight of the histogram descriptor in the match cost, 0 disables it')
    m.add_argument('--ttc-min-closing-speed', type=float, default=defaults.TTC_MIN_CLOSING_SPEED,
                   help='m/s below which a closing speed is treated as noise and TTC is None')
    m.add_argument('--ttc-max-value', type=float, default=defaults.TTC_MAX_REASONABLE_VALUE,
                   help='seconds above which a TTC is reported as None')
    m.add_argument('--calibration', help='camera calibration JSON (CAMERA_MATRIX, CAMERA_HEIGHT_M, ...); '
                                         'without it range falls back to the depth model')
    m.add_argument('--imu-log', help='recorded orientation samples JSON for ego-motion compensation')
    m.add_argument('--apriltag-log', help='per-frame AprilTag detections JSON from the tag pipeline')
    m.add_argument('--track-log', help='per-track CSV (id, distance, velocity, TTC, sources)')

    t = p.add_argument_group('bearings-only target motion analysis (--tracker bytetrack)')
    t.add_argument('--own-ship-log', help='own-ship navigation JSON/CSV (timestamp, x_m/y_m or '
                                          'latitude/longitude, heading, speed); enables the world-frame '
                                          'state estimate. Needs --calibration')
    t.add_argument('--tma-min-range', type=float, default=defaults.TMA_MIN_RANGE_M,
                   help='nearest range the hypotheses cover; nothing outside the window is reportable')
    t.add_argument('--tma-max-range', type=float, default=defaults.TMA_MAX_RANGE_M,
                   help='furthest range the hypotheses cover')
    t.add_argument('--tma-hypotheses', type=int, default=defaults.TMA_HYPOTHESES,
                   help='range slices seeded on the first bearing of a track')
    t.add_argument('--tma-pixel-sigma', type=float, default=defaults.TMA_PIXEL_SIGMA_PX,
                   help='jitter of the bearing reference point in working pixels; measure it with '
                        'bearing_noise.py rather than guessing')
    t.add_argument('--tma-heading-sigma-deg', type=float, default=defaults.TMA_HEADING_SIGMA_DEG,
                   help='own-ship heading uncertainty; it enters every bearing one for one')
    t.add_argument('--tma-mount-sigma-deg', type=float, default=defaults.TMA_MOUNT_SIGMA_DEG,
                   help='camera mounting angle uncertainty relative to the hull')
    t.add_argument('--tma-max-speed', type=float, default=defaults.TMA_MAX_TARGET_SPEED_MPS,
                   help='plausible target speed bound, m/s; a soft prior, never a veto')
    t.add_argument('--tma-lever', default=f'{defaults.TMA_LEVER_FORWARD_M},{defaults.TMA_LEVER_STARBOARD_M}',
                   help='camera offset FORWARD,STARBOARD in meters from the navigation reference')
    t.add_argument('--own-ship-max-age-s', type=float, default=1.0,
                   help='frames further than this from a navigation sample are left unplaced')

    s = p.add_argument_group('maritime segmentation')
    s.add_argument('--segmentation-weights', default=defaults.SEGMENTATION_MODEL_PATH,
                   help='water/obstacle/sky segmentation model; no weights ship with this repository')
    s.add_argument('--segmentation-input-size', type=int, default=defaults.SEGMENTATION_INPUT_SIZE,
                   help='segmentation inference size')
    s.add_argument('--segmentation-every', type=int, default=defaults.SEGMENTATION_EVERY_N_FRAMES,
                   help='run segmentation every N frames and reuse the mask in between')
    s.add_argument('--min-obstacle-area', type=int, default=defaults.SEGMENTATION_MIN_OBSTACLE_AREA,
                   help='smallest segmented obstacle in working-frame pixels')
    s.add_argument('--no-fusion', action='store_true', help='track detector output without segmentation fusion')
    s.add_argument('--segmentation-log', help='per-frame segmentation CSV (latency, obstacle count)')

    o = p.add_argument_group('debug overlays')
    o.add_argument('--no-track-overlay', action='store_true', help='hide track boxes and labels')
    o.add_argument('--no-segmentation-overlay', action='store_true', help='hide the semantic mask')
    o.add_argument('--no-water-contact-overlay', action='store_true', help='hide water-contact markers')
    o.add_argument('--no-distance-overlay', action='store_true', help='hide distance and velocity labels')
    o.add_argument('--no-ttc-overlay', action='store_true', help='hide TTC labels')
    o.add_argument('--no-motion-overlay', action='store_true', help='hide bearings-only state labels')
    o.add_argument('--debug', action='store_true', help='per-frame stage timings on the console')
    return p


def validate(parser, args):
    if args.no_display and not (args.target or args.yolo_weights):
        parser.error('headless processing requires --target or --yolo-weights; a center box is not a detection')
    if args.target and args.yolo_weights:
        parser.error('choose either manual --target or automatic --yolo-weights')
    if min(args.width, args.height, args.infer_size, args.yolo_imgsz) < 2:
        parser.error('image dimensions must be at least 2')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if not (0 < args.smoothing <= 1 and 0 < args.box_smoothing <= 1 and 0 < args.max_jump_frac <= 1):
        parser.error('smoothing and jump fractions must be in (0, 1]')
    if not (math.isfinite(args.calib_scale) and args.calib_scale > 0 and math.isfinite(args.calib_offset)):
        parser.error('calibration requires a positive finite scale and finite offset')
    if args.depth_range:
        try:
            near, far = (float(v) for v in str(args.depth_range).split(','))
            if not 0 <= near < far or not math.isfinite(far):
                raise ValueError
            args.depth_range = (near, far)
        except ValueError:
            parser.error('--depth-range must be NEAR,FAR meters with 0 <= NEAR < FAR')
    if args.tracker == MULTI_TRACKER and not args.yolo_weights:
        parser.error(f'--tracker {MULTI_TRACKER} needs --yolo-weights; it tracks detections, not a manual box')
    if args.segmentation_weights and args.tracker != MULTI_TRACKER:
        parser.error(f'--segmentation-weights applies to --tracker {MULTI_TRACKER}')
    if args.track_min_hits < 1 or args.track_max_missed_frames < 0:
        parser.error('--track-min-hits must be positive and --track-max-missed-frames nonnegative')
    if not math.isfinite(args.track_occlusion_seconds) or args.track_occlusion_seconds < 0:
        parser.error('--track-occlusion-seconds must be finite and nonnegative')
    if args.track_occlusion_seconds and args.tracker != MULTI_TRACKER:
        parser.error('--track-occlusion-seconds requires --tracker bytetrack')
    if not 0 <= args.track_iou_threshold <= 1 or not 0 < args.track_center_frac <= 1:
        parser.error('--track-iou-threshold must be in [0, 1] and --track-center-frac in (0, 1]')
    if args.track_appearance_weight < 0:
        parser.error('--track-appearance-weight must be nonnegative')
    if args.ttc_min_closing_speed <= 0 or args.ttc_max_value <= 0:
        parser.error('TTC thresholds must be positive')
    if args.segmentation_every < 1 or args.segmentation_input_size < 2 or args.min_obstacle_area < 1:
        parser.error('--segmentation-every, --segmentation-input-size and --min-obstacle-area must be positive')
    multi_only = ('track_log', 'segmentation_log', 'calibration', 'imu_log', 'apriltag_log',
                  'own_ship_log')
    used = [f'--{name.replace("_", "-")}' for name in multi_only if getattr(args, name)]
    if used and args.tracker != MULTI_TRACKER:
        parser.error(f'{", ".join(used)} applies to --tracker {MULTI_TRACKER}')
    if args.grace_frames < 0 or args.reacquire_frames < 1:
        parser.error('--grace-frames must be nonnegative and --reacquire-frames must be positive')
    if not 0 <= args.yolo_low_conf <= args.yolo_conf <= 1:
        parser.error('require 0 <= --yolo-low-conf <= --yolo-conf <= 1')
    if args.own_ship_log:
        if not args.calibration:
            parser.error('--own-ship-log needs --calibration: without intrinsics a pixel is not a bearing')
        if not 0 < args.tma_min_range < args.tma_max_range:
            parser.error('require 0 < --tma-min-range < --tma-max-range')
        if args.tma_hypotheses < 2 or args.tma_max_speed <= 0 or args.own_ship_max_age_s <= 0:
            parser.error('--tma-hypotheses must be at least 2, --tma-max-speed and '
                         '--own-ship-max-age-s positive')
        try:
            forward, starboard = (float(v) for v in str(args.tma_lever).split(','))
            args.tma_lever = (forward, starboard)
        except ValueError:
            parser.error('--tma-lever takes FORWARD,STARBOARD meters')
    if args.roi == 'segmentation' and not args.segmentation_weights:
        parser.error('--roi segmentation needs --segmentation-weights')
    if args.roi != 'none' and not args.yolo_weights:
        parser.error('--roi applies to automatic detection (--yolo-weights)')
    try:
        top, bottom = (float(v) for v in str(args.roi_band).split(','))
        args.roi_band = RoiConfig(band=(top, bottom)).band
    except ValueError:
        parser.error('--roi-band must be TOP,BOTTOM with 0 <= TOP < BOTTOM <= 1')
    try:
        if args.target:
            args.target = parse_box(args.target, args.width, args.height)
        if args.tile_grid:
            args.tile_grid = parse_grid(args.tile_grid)
            if not 0 <= args.tile_overlap < 1:
                raise ValueError
    except ValueError as exc:
        parser.error(str(exc) or '--tile-grid must be positive ROWSxCOLS and overlap must be in [0,1)')
    # File replay must pair depth and tracking from the same frame.
    args.sync = args.sync or isinstance(args.camera, str)


def select_box(frame):
    """Interactive ROI; ESC falls back to a centered box."""
    print('Draw a box around the object, then ENTER/SPACE. ESC tracks the frame center.')
    x, y, w, h = cv2.selectROI('select object to track', frame, showCrosshair=True)
    cv2.destroyWindow('select object to track')
    if w < 2 or h < 2:
        height, width = frame.shape[:2]
        w, h = width // 5, height // 5
        x, y = (width - w) // 2, (height - h) // 2
    print(f'--target "{x},{y},{x + w},{y + h}"')
    return x, y, x + w, y + h


def build_detector(args, segmentation_runner=None):
    """YOLO candidates, wrapped in a region proposer when --roi is set."""
    from ultralytics import YOLO
    detector = YoloDetector(YOLO(args.yolo_weights), conf=args.yolo_low_conf, imgsz=args.yolo_imgsz,
                            augment=args.augment, clahe=args.clahe,
                            tile_grid=args.tile_grid, tile_overlap=args.tile_overlap)
    proposer = build_proposer(RoiConfig(mode=args.roi, band=args.roi_band), segmentation_runner)
    return detector if proposer is None else TwoStageDetector(proposer, detector)


def build_overlay_config(args):
    """Overlays are off entirely for a headless run that writes no video."""
    if args.no_display and not args.output:
        return OverlayConfig.disabled()
    return OverlayConfig(show_tracks=not args.no_track_overlay,
                         show_segmentation=not args.no_segmentation_overlay,
                         show_water_contact=not args.no_water_contact_overlay,
                         show_distance=not args.no_distance_overlay,
                         show_ttc=not args.no_ttc_overlay,
                         show_motion=not args.no_motion_overlay)


def build_segmentation(args, size, device):
    """(runner, config); (None, defaults) while no segmentation weights are configured."""
    if not args.segmentation_weights:
        return None, SegmentationConfig()
    config = SegmentationConfig(enabled=True, model_path=args.segmentation_weights,
                                input_size=args.segmentation_input_size,
                                every_n_frames=args.segmentation_every,
                                min_obstacle_area=args.min_obstacle_area, fp16=args.fp16)
    return SegmentationRunner(UltralyticsSegmentation.load(config, size, device), config), config


def build_calibration(args, size):
    """Calibration rescaled to the working frame, or None when none is configured."""
    if not args.calibration:
        return None
    calibration = CameraCalibration.load(args.calibration)
    if calibration.image_size and tuple(calibration.image_size) != tuple(size):
        return calibration.scaled(calibration.image_size, size)
    return calibration


def build_tag_source(args, size, cap):
    """Tag replay: the rig detections.jsonl, or the generic per-frame JSON."""
    if not args.apriltag_log:
        return None
    if args.apriltag_log.endswith('.jsonl'):
        raw = isinstance(cap, RawCapture)
        return ReplayAprilTagSource.from_rig_log(
            args.apriltag_log, size, frame_names=[f.name for f in cap.files] if raw else None,
            native_size=(cap.width, cap.height) if raw else None)
    return ReplayAprilTagSource.load(args.apriltag_log)


def build_pipeline(args, size, device, cap=None):
    """Multi-object perception pipeline for --tracker bytetrack."""
    mot = MotConfig(conf_threshold=args.yolo_conf, low_conf_threshold=args.yolo_low_conf,
                    iou_threshold=args.track_iou_threshold,
                    max_missed_frames=args.track_max_missed_frames, min_hits=args.track_min_hits,
                    occlusion_seconds=args.track_occlusion_seconds,
                    max_center_distance_frac=args.track_center_frac,
                    appearance_weight=args.track_appearance_weight,
                    box_smoothing=args.box_smoothing,
                    ttc_min_closing_speed=args.ttc_min_closing_speed,
                    ttc_max_reasonable_value=args.ttc_max_value)
    runner, segmentation_config = build_segmentation(args, size, device)
    own_ship = OwnShipTrack.load(args.own_ship_log, args.own_ship_max_age_s) \
        if args.own_ship_log else None
    return PerceptionPipeline(
        build_detector(args, runner), size, mot, runner, segmentation_config,
        FusionConfig(enabled=runner is not None and not args.no_fusion),
        calibration=build_calibration(args, size),
        ego_motion=SampledEgoMotion.load(args.imu_log) if args.imu_log else None,
        tag_source=build_tag_source(args, size, cap),
        depth_calibration=(args.calib_scale, args.calib_offset),
        own_ship=own_ship, tma_config=build_tma_config(args))


def build_tma_config(args):
    """Range window, bank size and the bearing error budget from the flags."""
    forward, starboard = args.tma_lever if isinstance(args.tma_lever, tuple) else (0.0, 0.0)
    return TmaConfig(min_range_m=args.tma_min_range, max_range_m=args.tma_max_range,
                     hypotheses=args.tma_hypotheses, max_target_speed_mps=args.tma_max_speed,
                     pixel_sigma_px=args.tma_pixel_sigma,
                     heading_sigma_deg=args.tma_heading_sigma_deg,
                     mount_sigma_deg=args.tma_mount_sigma_deg,
                     lever_forward_m=forward, lever_starboard_m=starboard)


def build_tracker(args, first_frame):
    if args.target:
        return ManualCsrtTracker(first_frame, args.target)
    if not args.yolo_weights:
        return ManualCsrtTracker(first_frame, select_box(first_frame))
    detector = build_detector(args)
    config = TrackerConfig(args.yolo_conf, args.yolo_low_conf, args.box_smoothing, args.max_jump_frac,
                           args.grace_frames, args.reacquire_frames)
    return TRACKERS[args.tracker](detector, (args.width, args.height), config)


class DistanceFilter:
    """Calibrated distance plus EMA; resets whenever the measurement is lost."""

    def __init__(self, scale, offset, smoothing):
        self.scale, self.offset, self.smoothing = scale, offset, smoothing
        self.smoothed = None

    def update(self, model_m):
        """(model_m, calibrated_m, smoothed_m); None where unavailable."""
        calibrated = None if model_m is None else self.scale * model_m + self.offset
        if calibrated is None or calibrated <= 0 or not math.isfinite(calibrated):
            self.smoothed = None
            return model_m, None, None
        self.smoothed = calibrated if self.smoothed is None else \
            self.smoothing * calibrated + (1 - self.smoothing) * self.smoothed
        return model_m, calibrated, self.smoothed


def fmt(value, digits=4):
    return '' if value is None else f'{value:.{digits}f}'


def draw(frame, depth, box, calibrated, smoothed, calibration=(1.0, 0.0)):
    """Manual-target view: the same box on the camera frame and on the depth map."""
    camera, depth_panel = frame.copy(), depth.color.copy()
    if box is None:
        overlays.draw_label_block(camera, ['TRACKING LOST', 'press r to reselect'],
                                  (12, overlays.HEADER_HEIGHT), overlays.WARNING_COLOR)
    else:
        x1, y1, x2, y2 = box
        for image in (camera, depth_panel):
            cv2.rectangle(image, (x1, y1), (x2, y2), overlays.TRACK_COLOR, 2, overlays.LINE)
        if smoothed is not None:
            lines = [f'{smoothed:.2f} m', f'raw {calibrated:.2f} m']
            overlays.draw_label_block(camera, lines, (x1, y1 - 34), overlays.TRACK_COLOR)
            overlays.draw_label_block(depth_panel, [f'{smoothed:.2f} m'], (x1, y2 + 2), overlays.TRACK_COLOR)
    overlays.draw_caption(camera, 'CAMERA  manual target')
    if depth.span is not None:
        overlays.draw_colorbar(depth_panel, depth.span, calibration)
    overlays.draw_caption(depth_panel, 'DEPTH  metric, monocular')
    return camera, depth_panel


def nearest_track(tracks):
    """Closest measured track, else the most confident one; None when nothing is tracked."""
    tracks = [t for t in tracks if not t.missed_frames]
    measured = [t for t in tracks if t.distance_m is not None]
    if measured:
        return min(measured, key=lambda t: t.distance_m)
    return max(tracks, key=lambda t: t.confidence, default=None)


def track_distances(track, scale, offset):
    """(model_m, calibrated_m, filtered_m) of one track for the per-frame CSV.

    A depth measurement is reported before and after calibration; a metric
    source (AprilTag pose, water plane) is already in meters and is not scaled.
    """
    if track is None:
        return None, None, None
    if track.measurement_age:
        return None, None, track.distance_m
    model = track.raw_distance_m
    calibrated = scale * model + offset if model is not None else track.distance_m
    return model, calibrated, track.distance_m


def write_metadata(path, metadata):
    if path:
        path.write_text(json.dumps(metadata, indent=2) + '\n')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    size = (args.width, args.height)
    device = get_device()
    print(f'device: {device}, fp16: {args.fp16 and device != "cpu"}')

    cap = open_video(args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # live: drop stale frames
    worker = writer = log_file = track_log = pipeline = None
    try:
        ok, native = cap.read()
        if not ok:
            raise RuntimeError('source opened but decoded no frames')
        fps = video_fps(cap)
        estimator = DepthEstimator.load(args.model, device, size, args.infer_size, args.fp16,
                                        span=args.depth_range or None)
        worker = DepthWorker(estimator)
        multi = args.tracker == MULTI_TRACKER
        pipeline = build_pipeline(args, size, device, cap) if multi else None
        tracker = None if multi else build_tracker(args, cv2.resize(native, size))
        overlay_config = build_overlay_config(args)
        distance = DistanceFilter(args.calib_scale, args.calib_offset, args.smoothing)
        calibration = (args.calib_scale, args.calib_offset)
        title = f'BOAT DETECTION  ·  {Path(str(args.camera)).name}' if isinstance(args.camera, str) \
            else f'BOAT DETECTION  ·  camera {args.camera}'
        track_log = TrackLogger(args.track_log, args.segmentation_log) \
            if (args.track_log or args.segmentation_log) else None

        writer = h264_writer(args.output, fps, overlays.composed_size(size)) if args.output else None
        if args.log:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log_file = open(args.log, 'w', newline='')
            log = csv.writer(log_file)
            log.writerow(CSV_HEADER)
        if not args.sync:
            worker.start()

        # The detectors are trained on color; luma input costs nearly all recall.
        color = color_fraction(native)
        if color < 0.05 and args.yolo_weights:
            print(f'WARNING: source looks monochrome ({color:.1%} color pixels). The boat '
                  f'detectors are color-trained and lose nearly all recall on luma input; '
                  f'record with pixel_format=bgr. See docs/EVALUATION.md')

        metadata_path = Path(args.output or args.log).with_suffix('.meta.json') if args.output or args.log else None
        metadata = dict(vars(args), fps=fps, calibrated=(args.calib_scale != 1 or args.calib_offset != 0),
                        segmentation_enabled=bool(pipeline and pipeline.segmentation_runner),
                        source_color_fraction=round(color, 4), source_is_color=color >= 0.05,
                        source_raw_frames=isinstance(cap, RawCapture),
                        target_selection=f'Automatic (YOLO + {args.tracker})' if args.yolo_weights else 'Manual CSRT',
                        source_video=str(args.camera), frames=0, status='running')
        write_metadata(metadata_path, metadata)

        frame_time = frame_timer(cap, fps)
        frame_idx = 0
        while ok and (args.max_frames is None or frame_idx < args.max_frames):
            frame = cv2.resize(native, size)
            if args.sync:
                depth = estimator.infer(frame)
            else:
                worker.submit_frame(frame.copy())
                depth = worker.snapshot()
            depth_m, infer_ms = depth.depth_m, depth.infer_ms

            if multi:
                result = pipeline.process(frame, native, frame_time(frame_idx), frame_idx, depth_m)
                primary = nearest_track(result.tracks)
                model_m, calibrated, smoothed = track_distances(primary, args.calib_scale, args.calib_offset)
                if track_log:
                    track_log.write_tracks(frame_idx, frame_time(frame_idx), result.tracks)
                    track_log.write_segmentation(frame_idx, result.segmentation, len(result.obstacles))
                if args.debug:
                    print(f'frame {frame_idx}: {len(result.tracks)} tracks  {pipeline.timer.format()}')
                panels = (overlays.render(frame, result, overlay_config, status=False),
                          overlays.render_depth(depth.color, result, depth.span, calibration, overlay_config))
                combined = overlays.compose(panels, title=title,
                                            subtitle=f'frame {frame_idx}  ·  {len(result.tracks)} tracks',
                                            status=overlays.status_text(result.timer, result.segmentation))
            else:
                tracked, (x, y, w, h) = tracker.update(frame, native)
                box = (x, y, x + w, y + h) if tracked else None
                model_m = distance_in_box(depth_m, box) if box and depth_m is not None else None
                model_m, calibrated, smoothed = distance.update(model_m)
                if smoothed is not None:
                    print(f'frame {frame_idx}: {smoothed:.2f} m (inference: {infer_ms:.0f} ms)', end='\r')
                combined = overlays.compose(draw(frame, depth, box, calibrated, smoothed, calibration),
                                            title=title, subtitle=f'frame {frame_idx}',
                                            status=f'depth: {infer_ms:.0f} ms  ·  r: reselect  q: quit')
            if writer:
                writer.write(combined)
            if log_file:
                log.writerow([frame_idx, fmt(calibrated), fmt(smoothed), f'{infer_ms:.1f}',
                              f'{frame_time(frame_idx):.6f}', fmt(model_m)])
            frame_idx += 1

            if not args.no_display:
                cv2.imshow(WINDOW, combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r') and not multi:
                    tracker = ManualCsrtTracker(frame, select_box(frame))
                    distance.smoothed = None
            ok, native = cap.read()

        metadata.update(frames=frame_idx, status='complete')
        if pipeline is not None:
            metadata.update(timings=pipeline.timer.summary(), tracks_created=pipeline.tracker.next_id - 1)
            print(f'\n{frame_idx} frames, {pipeline.tracker.next_id - 1} tracks, {pipeline.timer.summary()}')
        write_metadata(metadata_path, metadata)
    finally:
        if worker is not None and worker.is_alive():
            worker.stop()
            worker.join(timeout=2.0)
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if track_log is not None:
            track_log.close()
        if not args.no_display:
            cv2.destroyAllWindows()
            for _ in range(4):  # macOS needs extra event pumps to close windows
                cv2.waitKey(1)


if __name__ == '__main__':
    try:
        main()
    except SegmentationUnavailable as exc:
        raise SystemExit(f'segmentation error: {exc}')
    except KeyboardInterrupt:
        print('\ninterrupted, shutting down')
