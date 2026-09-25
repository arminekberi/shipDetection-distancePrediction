"""Defaults for multi-object tracking, segmentation, fusion, geometry and overlays.

Module constants are the single source of truth; the frozen config objects take
their defaults from them and run.py flags override them per run.
"""
from dataclasses import dataclass
import math

# --- multi-object tracking ---
TRACKER_TYPE = 'bytetrack'
TRACK_CONF_THRESHOLD = 0.45        # high-confidence stage, also starts new tracks
TRACK_LOW_CONF_THRESHOLD = 0.15    # weak detections, second association stage only
TRACK_IOU_THRESHOLD = 0.30
TRACK_SECOND_IOU_THRESHOLD = 0.50  # weak detections must overlap more convincingly
TRACK_MAX_MISSED_FRAMES = 15
TRACK_MIN_HITS = 3
TRACK_MAX_CENTER_DISTANCE_FRAC = 0.25  # motion gate, fraction of the frame short side
TRACK_MAX_SCALE_RATIO = 4.0            # motion gate, allowed box area change per match
TRACK_APPEARANCE_WEIGHT = 0.25         # 0 disables the histogram descriptor
TRACK_APPEARANCE_SMOOTHING = 0.3       # EMA on the descriptor of a matched track
TRACK_BOX_SMOOTHING = 0.5              # EMA on box size and water contact, 1 = raw
TRACK_MAX_DT_S = 1.0                   # longer gaps reset velocity instead of extrapolating
TRACK_MEASUREMENT_NOISE_PX = 9.0
TRACK_PROCESS_NOISE_PX = 25.0
TRACK_MEASUREMENT_NOISE_M = 0.5       # m^2, water-plane and tag ranges
TRACK_DEPTH_NOISE_M = 4.0             # m^2, monocular depth model ranges
TRACK_PROCESS_NOISE_M = 2.0
TRACK_MAX_COAST_S = 2.0               # seconds without a match before a track is dropped
TRACK_OCCLUSION_SECONDS = 0.0         # opt in to longer, seconds-based vessel coasting
TRACK_REACQUIRE_APPEARANCE = 0.5      # minimum histogram correlation after a disappearance

# --- time to collision ---
TTC_MIN_CLOSING_SPEED = 0.2        # m/s below this the estimate is noise
TTC_MAX_REASONABLE_VALUE = 120.0   # s, longer horizons are reported as None
TTC_MIN_SIGMA = 2.0                # closing speed must exceed this many filter std devs

# --- segmentation ---
SEGMENTATION_ENABLED = False
SEGMENTATION_MODEL_PATH = None
SEGMENTATION_INPUT_SIZE = 512
SEGMENTATION_EVERY_N_FRAMES = 2
SEGMENTATION_MIN_OBSTACLE_AREA = 120   # px in working coordinates
SEGMENTATION_FP16 = True
SEGMENTATION_CONF = 0.25
# Model class name -> semantic category. Names not listed keep their own category.
SEGMENTATION_CLASS_MAP = {
    'water': 'water', 'sea': 'water', 'river': 'water', 'lake': 'water',
    'sky': 'sky',
    'obstacle': 'obstacle', 'static_obstacle': 'obstacle', 'dynamic_obstacle': 'obstacle',
    'boat': 'boat', 'ship': 'boat', 'vessel': 'boat', 'buoy': 'buoy',
    'shore': 'shore', 'land': 'shore', 'pier': 'pier', 'dock': 'pier',
}
# Extra categories folded into the obstacle mask exposed to the safety logic.
SEGMENTATION_OBSTACLE_CATEGORIES = ('obstacle', 'boat', 'buoy', 'shore', 'pier')

# --- detector / segmentation fusion ---
FUSION_ENABLED = True
FUSION_SKY_FRACTION = 0.80         # box mostly in sky: suspicious detection
FUSION_SKY_CONF_SCALE = 0.5        # confidence factor applied instead of dropping it
FUSION_OBSTACLE_FRACTION = 0.20    # obstacle coverage that confirms a detection
FUSION_UNKNOWN_IOU = 0.30          # segmented blob above this IoU belongs to a detection
FUSION_UNKNOWN_CONF = 0.50             # reported confidence of a segmentation-only obstacle;
                                       # must reach TRACK_CONF_THRESHOLD or it can never open a track
UNKNOWN_OBSTACLE_CLASS = 'unknown_obstacle'
UNKNOWN_OBSTACLE_CLASS_ID = -1
# Tags are also mounted on infrastructure (the test hall has them on the roof
# trusses), so only listed ids are vessel identities. Empty: every tag may
# bind to a detected vessel, none may create a track on its own.
APRILTAG_VESSEL_IDS = ()
# A listed tag the detector misses may open an identity-and-bearing-only track.
APRILTAG_SEEDS_TRACKS = False
TAGGED_VESSEL_CLASS = 'tagged_vessel'
TAGGED_VESSEL_CLASS_ID = -2

# --- glare review signal (logged and drawn, never used to drop a detection) ---
GLARE_CHANNEL_THRESHOLD = 250     # a pixel counts as highlight if any BGR channel reaches this
GLARE_SUSPECT_FRACTION = 0.30     # highlight share of a box that flags it for review

# --- camera geometry ---
CAMERA_CALIBRATION_PATH = None     # JSON with CAMERA_MATRIX, DIST_COEFFS, CAMERA_HEIGHT_M, ...
CAMERA_MATRIX = None
DIST_COEFFS = None
CAMERA_HEIGHT_M = None
CAMERA_TO_IMU_TRANSFORM = None     # 3x3 rotation, camera -> IMU/vessel body frame (FRD)

# --- bearings-only target motion analysis ---
# Scaled for the covered test basin, which is the only scene this rig records: tens
# of meters, model speeds, a camera at ~4 fps. At sea every one of these is an order
# of magnitude out, and `tma_sim.py` keeps open-water scenarios with their own
# overrides so the method stays exercised at that scale too.
#
# The window the range hypotheses tile; nothing outside it can ever be reported,
# so it must cover the scenario rather than the expected answer. Replace the
# assumed basin extent below with the measured tank dimensions.
TMA_MIN_RANGE_M = 1.0
TMA_MAX_RANGE_M = 120.0
TMA_HYPOTHESES = 12                # log-spaced range slices seeded on the first bearing
TMA_MIN_HYPOTHESES = 3             # never pruned below this: a single mode would hide ambiguity
TMA_MIN_WEIGHT = 1e-5              # weight under which a hypothesis stops being carried
TMA_MAX_TARGET_SPEED_MPS = 3.0     # a model under its own power; bounds the velocity prior
TMA_SPEED_PRIOR_SLACK_MPS = 1.0    # softness of that bound, it penalises and never vetoes
TMA_PROCESS_NOISE_PSD = 0.005      # target acceleration PSD, m^2/s^3. Raising it buys
                                   # manoeuvre response and costs range accuracy directly.
TMA_MAX_DT_S = 2.0                 # longer prediction steps are clamped, not extrapolated
TMA_MIN_UPDATES = 10               # before this the estimate is reported as initializing
TMA_CONVERGED_RANGE_FRAC = 0.20    # range sigma above this share of the range stays ambiguous.
                                   # A range known to plus-minus a quarter of itself is a bracket,
                                   # not a fix, and the geometric floor puts a basin weave right at
                                   # that line: 0.25 let marginal cases through as converged.
TMA_PARALLAX_SIGMAS = 3.0          # own-ship cross-LOS parallax must beat the bearing noise by this
TMA_MANEUVER_WINDOW_S = 40.0       # window the parallax is measured over; a basin run is short
TMA_WEIGHT_FORGET_S = 20.0         # hypothesis weights flatten with this time constant, so a bank
                                   # cannot collapse onto one range by accumulating negligible
                                   # likelihood differences; 0 disables the flattening
TMA_CREDIBLE_LEVEL = 0.90          # reported range interval
# Bearing error budget. Measure the first one for your detector and reference point
# (bearing_noise.py); detection confidence is not an angular variance.
TMA_PIXEL_SIGMA_PX = 1.5           # jitter of the bearing reference point, working pixels.
                                   # Measured raw on 6 rig recordings: 0.67 px RMS, 0.41 robust
                                   # (bearing_noise.py --video). Kept above that because the
                                   # measurement cannot see the box center drifting with aspect.
TMA_HEADING_SIGMA_DEG = 1.0        # own-ship heading uncertainty; it enters every bearing directly
TMA_MOUNT_SIGMA_DEG = 0.3          # camera mounting angles relative to the hull
TMA_LEVER_FORWARD_M = 0.0          # camera offset from the navigation reference point
TMA_LEVER_STARBOARD_M = 0.0
# A bearing no hypothesis can explain is more likely a bad detection or a navigation
# glitch than evidence: without a gate one such sample collapses the whole bank onto
# whatever range makes the impossible bearing rate arithmetically possible.
TMA_GATE_SIGMAS = 5.0              # 0 disables the gate
TMA_GATE_MAX_CONSECUTIVE = 5       # rejections in a row after which the bank is reseeded: a
                                   # persistent disagreement means our state is wrong, not the
                                   # bearings, and a permanent lockout would be worse than a restart
TMA_RANGE_PRIOR_FACTOR = 2.0       # how far outside the range window a hypothesis may drift

# --- overlays ---
SHOW_TRACKS = True
SHOW_SEGMENTATION = True
SHOW_WATER_CONTACT = True
SHOW_DISTANCE = True
SHOW_TTC = True
SHOW_MOTION = True        # bearings-only state, drawn with its status so a guess cannot read as a fix
SEGMENTATION_OVERLAY_ALPHA = 0.35

# --- logging ---
TRACK_LOG_PATH = None
DEBUG_LOGGING = False


@dataclass(frozen=True)
class MotConfig:
    """Association, coasting and filtering parameters of the multi-object tracker."""
    conf_threshold: float = TRACK_CONF_THRESHOLD
    low_conf_threshold: float = TRACK_LOW_CONF_THRESHOLD
    iou_threshold: float = TRACK_IOU_THRESHOLD
    second_iou_threshold: float = TRACK_SECOND_IOU_THRESHOLD
    max_missed_frames: int = TRACK_MAX_MISSED_FRAMES
    min_hits: int = TRACK_MIN_HITS
    max_center_distance_frac: float = TRACK_MAX_CENTER_DISTANCE_FRAC
    max_scale_ratio: float = TRACK_MAX_SCALE_RATIO
    appearance_weight: float = TRACK_APPEARANCE_WEIGHT
    appearance_smoothing: float = TRACK_APPEARANCE_SMOOTHING
    box_smoothing: float = TRACK_BOX_SMOOTHING
    max_dt_s: float = TRACK_MAX_DT_S
    measurement_noise_px: float = TRACK_MEASUREMENT_NOISE_PX
    process_noise_px: float = TRACK_PROCESS_NOISE_PX
    measurement_noise_m: float = TRACK_MEASUREMENT_NOISE_M
    depth_noise_m: float = TRACK_DEPTH_NOISE_M
    process_noise_m: float = TRACK_PROCESS_NOISE_M
    max_coast_s: float = TRACK_MAX_COAST_S
    occlusion_seconds: float = TRACK_OCCLUSION_SECONDS
    reacquire_appearance: float = TRACK_REACQUIRE_APPEARANCE
    ttc_min_closing_speed: float = TTC_MIN_CLOSING_SPEED
    ttc_max_reasonable_value: float = TTC_MAX_REASONABLE_VALUE
    ttc_min_sigma: float = TTC_MIN_SIGMA
    glare_suspect_fraction: float = GLARE_SUSPECT_FRACTION

    def __post_init__(self):
        if not 0 <= self.low_conf_threshold <= self.conf_threshold <= 1:
            raise ValueError('require 0 <= low_conf_threshold <= conf_threshold <= 1')
        if self.max_missed_frames < 0 or self.min_hits < 1:
            raise ValueError('max_missed_frames must be nonnegative and min_hits positive')
        if not 0 < self.box_smoothing <= 1:
            raise ValueError('box_smoothing must be in (0, 1]')
        if self.max_dt_s <= 0 or self.max_scale_ratio < 1:
            raise ValueError('max_dt_s must be positive and max_scale_ratio at least 1')
        if not math.isfinite(self.occlusion_seconds) or self.occlusion_seconds < 0:
            raise ValueError('occlusion_seconds must be finite and nonnegative')
        if not 0 <= self.reacquire_appearance <= 1:
            raise ValueError('reacquire_appearance must be in [0, 1]')


@dataclass(frozen=True)
class SegmentationConfig:
    """Segmentation branch; disabled by default because no weights ship with the repository."""
    enabled: bool = SEGMENTATION_ENABLED
    model_path: str = SEGMENTATION_MODEL_PATH
    input_size: int = SEGMENTATION_INPUT_SIZE
    every_n_frames: int = SEGMENTATION_EVERY_N_FRAMES
    min_obstacle_area: int = SEGMENTATION_MIN_OBSTACLE_AREA
    fp16: bool = SEGMENTATION_FP16
    conf: float = SEGMENTATION_CONF
    obstacle_categories: tuple = SEGMENTATION_OBSTACLE_CATEGORIES

    def __post_init__(self):
        if self.enabled and not self.model_path:
            raise ValueError('segmentation is enabled but no model path is configured')
        if self.every_n_frames < 1 or self.input_size < 2 or self.min_obstacle_area < 1:
            raise ValueError('every_n_frames, input_size and min_obstacle_area must be positive')


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = FUSION_ENABLED
    sky_fraction: float = FUSION_SKY_FRACTION
    sky_conf_scale: float = FUSION_SKY_CONF_SCALE
    obstacle_fraction: float = FUSION_OBSTACLE_FRACTION
    unknown_iou: float = FUSION_UNKNOWN_IOU
    unknown_conf: float = FUSION_UNKNOWN_CONF
    apriltag_seeds_tracks: bool = APRILTAG_SEEDS_TRACKS
    apriltag_vessel_ids: tuple = APRILTAG_VESSEL_IDS

    def __post_init__(self):
        if not all(0 <= v <= 1 for v in (self.sky_fraction, self.sky_conf_scale,
                                         self.obstacle_fraction, self.unknown_iou, self.unknown_conf)):
            raise ValueError('fusion thresholds must be in [0, 1]')


@dataclass(frozen=True)
class OverlayConfig:
    show_tracks: bool = SHOW_TRACKS
    show_segmentation: bool = SHOW_SEGMENTATION
    show_water_contact: bool = SHOW_WATER_CONTACT
    show_distance: bool = SHOW_DISTANCE
    show_ttc: bool = SHOW_TTC
    show_motion: bool = SHOW_MOTION
    segmentation_alpha: float = SEGMENTATION_OVERLAY_ALPHA

    @classmethod
    def disabled(cls):
        return cls(False, False, False, False, False, False)


@dataclass(frozen=True)
class TmaConfig:
    """Bearings-only target motion analysis: range window, bank size and honesty thresholds."""
    min_range_m: float = TMA_MIN_RANGE_M
    max_range_m: float = TMA_MAX_RANGE_M
    hypotheses: int = TMA_HYPOTHESES
    min_hypotheses: int = TMA_MIN_HYPOTHESES
    min_weight: float = TMA_MIN_WEIGHT
    max_target_speed_mps: float = TMA_MAX_TARGET_SPEED_MPS
    speed_prior_slack_mps: float = TMA_SPEED_PRIOR_SLACK_MPS
    process_noise_psd: float = TMA_PROCESS_NOISE_PSD
    max_dt_s: float = TMA_MAX_DT_S
    min_updates: int = TMA_MIN_UPDATES
    converged_range_frac: float = TMA_CONVERGED_RANGE_FRAC
    parallax_sigmas: float = TMA_PARALLAX_SIGMAS
    maneuver_window_s: float = TMA_MANEUVER_WINDOW_S
    weight_forget_s: float = TMA_WEIGHT_FORGET_S
    pixel_sigma_px: float = TMA_PIXEL_SIGMA_PX
    heading_sigma_deg: float = TMA_HEADING_SIGMA_DEG
    mount_sigma_deg: float = TMA_MOUNT_SIGMA_DEG
    lever_forward_m: float = TMA_LEVER_FORWARD_M
    lever_starboard_m: float = TMA_LEVER_STARBOARD_M
    gate_sigmas: float = TMA_GATE_SIGMAS
    gate_max_consecutive: int = TMA_GATE_MAX_CONSECUTIVE
    range_prior_factor: float = TMA_RANGE_PRIOR_FACTOR
    credible_level: float = TMA_CREDIBLE_LEVEL

    def __post_init__(self):
        if not 0 < self.min_range_m < self.max_range_m:
            raise ValueError('require 0 < min_range_m < max_range_m')
        if self.hypotheses < 2 or not 1 <= self.min_hypotheses <= self.hypotheses:
            raise ValueError('need at least two hypotheses and 1 <= min_hypotheses <= hypotheses')
        if self.process_noise_psd < 0 or self.max_dt_s <= 0 or self.max_target_speed_mps <= 0:
            raise ValueError('process noise must be nonnegative, max_dt_s and max speed positive')
        if self.weight_forget_s < 0 or self.maneuver_window_s <= 0:
            raise ValueError('weight_forget_s must be nonnegative and maneuver_window_s positive')
        if not 0 < self.min_weight < 1 or not 0 < self.credible_level < 1:
            raise ValueError('min_weight and credible_level must lie in (0, 1)')
        if self.gate_max_consecutive < 1:
            raise ValueError('gate_max_consecutive must be positive or the bank could never recover')
        if self.gate_sigmas < 0 or self.range_prior_factor < 1:
            raise ValueError('gate_sigmas must be nonnegative and range_prior_factor at least 1')
        if min(self.pixel_sigma_px, self.heading_sigma_deg, self.mount_sigma_deg) < 0:
            raise ValueError('the bearing error budget cannot contain a negative sigma')
        if self.pixel_sigma_px == 0 and self.heading_sigma_deg == 0 and self.mount_sigma_deg == 0:
            raise ValueError('a bearing with no uncertainty at all would be trusted absolutely')


# --- two-stage detection: region proposal before the detector ---
ROI_MODE = 'none'                 # none | band | horizon | segmentation
ROI_BAND = (0.35, 0.78)           # fixed band, fractions of frame height (label centers: 0.40-0.73)
HORIZON_SEARCH = (0.15, 0.85)     # rows where the horizon may lie, fractions of height
HORIZON_MARGIN_ABOVE = 0.08       # band above the horizon line, fraction of height
HORIZON_MARGIN_BELOW = 0.35       # band below it, where near boats extend
HORIZON_WORK_WIDTH = 320          # edge search runs on a downscaled frame
HORIZON_MIN_INLIER_FRAC = 0.40    # weaker line fits fall back to the fixed band
HORIZON_MAX_TILT_DEG = 20.0       # roll beyond this is treated as a failed fit
HORIZON_SMOOTHING = 0.5           # EMA on the line between frames, 1 = raw
HORIZON_MIN_EDGE = 8.0            # weakest vertical gradient (grey levels/px) counted as an edge


@dataclass(frozen=True)
class RoiConfig:
    mode: str = ROI_MODE
    band: tuple = ROI_BAND
    search: tuple = HORIZON_SEARCH
    margin_above: float = HORIZON_MARGIN_ABOVE
    margin_below: float = HORIZON_MARGIN_BELOW
    work_width: int = HORIZON_WORK_WIDTH
    min_inlier_frac: float = HORIZON_MIN_INLIER_FRAC
    max_tilt_deg: float = HORIZON_MAX_TILT_DEG
    smoothing: float = HORIZON_SMOOTHING
    min_edge: float = HORIZON_MIN_EDGE

    def __post_init__(self):
        if self.mode not in ('none', 'band', 'horizon', 'segmentation'):
            raise ValueError(f'unknown ROI mode {self.mode!r}')
        for low, high in (self.band, self.search):
            if not 0 <= low < high <= 1:
                raise ValueError('ROI fractions must satisfy 0 <= low < high <= 1')
