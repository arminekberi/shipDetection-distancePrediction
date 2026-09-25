"""Debug and presentation overlays for tracks, segmentation and the depth map.

Drawing is optional everywhere: the pipeline computes the same state with every
overlay off, which is how a production run should be configured.

Color carries one meaning only. Track boxes wear the reserved status scale
(tracked / unknown obstacle / collision risk) and every state is also written as
a word and a line style, because good-green and critical-red collapse into each
other under deuteranopia. The depth map is a single-hue sequential ramp, so a
map color is never mistaken for a track state, and it ships with a labelled
colorbar in meters.
"""
import cv2
import numpy as np

from boatdet.config import UNKNOWN_OBSTACLE_CLASS, OverlayConfig
from boatdet.segmentation import OBSTACLE, SKY, WATER

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


def bgr(hex_color):
    """'#rrggbb' -> OpenCV BGR tuple."""
    return tuple(int(hex_color[i:i + 2], 16) for i in (5, 3, 1))


# Reserved status scale: a state never travels as color alone (see module docstring).
TRACK_COLOR = bgr('#0ca30c')      # good: a tracked vessel
UNKNOWN_COLOR = bgr('#fab219')    # warning: segmentation-only obstacle, unclassified
WARNING_COLOR = bgr('#d03b3b')    # critical: closing, TTC available
GLARE_COLOR = bgr('#ec835a')      # serious: measurement suspect, review flag
# Chrome and ink.
INK = bgr('#ffffff')
INK_SECONDARY = bgr('#c3c2b7')
INK_MUTED = bgr('#898781')
SURFACE = bgr('#1a1a19')
PLANE = bgr('#0d0d0d')
HAIRLINE = bgr('#2c2c2a')
CONTACT_COLOR = INK               # waterline contact is geometry, not a state
ROI_COLOR = INK_MUTED             # the search band is chrome, and stays recessive
TEXT_COLOR = INK_SECONDARY
# BGR tint per semantic category.
CATEGORY_COLORS = {WATER: (255, 140, 0), OBSTACLE: (0, 0, 255), SKY: (180, 180, 180)}

LABEL_SCALE = 0.4
CAPTION_SCALE = 0.42
PAD = 4
HEADER_HEIGHT = 30
LEGEND_HEIGHT = 46


def track_color(track):
    if getattr(track, 'missed_frames', 0):
        return UNKNOWN_COLOR
    if track.ttc_s is not None:
        return WARNING_COLOR
    return UNKNOWN_COLOR if track.class_name == UNKNOWN_OBSTACLE_CLASS else TRACK_COLOR


def track_state(track):
    """Short word for the track's status, so the color is never the only cue."""
    if getattr(track, 'missed_frames', 0):
        return 'PREDICTED'
    if track.ttc_s is not None:
        return 'RISK'
    return 'UNKNOWN' if track.class_name == UNKNOWN_OBSTACLE_CLASS else 'TRACKED'


def text_size(text, scale=LABEL_SCALE, thickness=1):
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return width, height + baseline


def draw_dashed_rect(image, corner1, corner2, color, thickness=1, dash=6, line_type=LINE):
    """A dashed box, the line style that separates an unconfirmed obstacle from a track."""
    (x1, y1), (x2, y2) = corner1, corner2
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness, line_type)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness, line_type)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness, line_type)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness, line_type)


def draw_panel(image, corner1, corner2, alpha=0.72, color=SURFACE):
    """Translucent backdrop; labels stay readable over sky, water and glare alike."""
    x1, y1 = (max(0, v) for v in corner1)
    x2, y2 = min(image.shape[1], corner2[0]), min(image.shape[0], corner2[1])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return
    region = image[y1:y2, x1:x2]
    cv2.addWeighted(np.full(region.shape, color, np.uint8), alpha, region, 1 - alpha, 0, region)


def overlaps(box, other):
    return not (box[2] <= other[0] or other[2] <= box[0] or box[3] <= other[1] or other[3] <= box[1])


def draw_label_block(image, lines, anchor, accent, scale=LABEL_SCALE, right_limit=None, occupied=None):
    """Label card with an accent bar, clamped inside the image. Returns its box.

    `right_limit` keeps the card clear of the colorbar gutter; `occupied` is a
    list of cards already placed, which this one slides below rather than cover.
    """
    sizes = [text_size(line, scale) for line in lines]
    width = max(w for w, _ in sizes) + 3 * PAD + 2
    height = sum(h for _, h in sizes) + PAD * (len(lines) + 1)
    limit = image.shape[1] if right_limit is None else min(image.shape[1], right_limit)
    x = int(max(0, min(limit - width, anchor[0])))
    y = int(max(0, min(image.shape[0] - height, anchor[1])))
    for _ in range(len(occupied or ())):
        if not any(overlaps((x, y, x + width, y + height), box) for box in occupied):
            break
        y = int(min(image.shape[0] - height, y + height + 2))
    if occupied is not None:
        occupied.append((x, y, x + width, y + height))
    draw_panel(image, (x, y), (x + width, y + height))
    cv2.rectangle(image, (x, y), (x + 2, y + height - 1), accent, -1)
    cursor = y + PAD
    for index, (line, (_, line_height)) in enumerate(zip(lines, sizes)):
        cursor += line_height
        cv2.putText(image, line, (x + 2 * PAD + 2, cursor - PAD // 2), FONT, scale,
                    INK if index == 0 else INK_SECONDARY, 1, LINE)
        cursor += PAD
    return x, y, x + width, y + height


def draw_caption(image, text, accent=INK_MUTED):
    """Panel name in the top-left corner: which view the viewer is looking at."""
    width, height = text_size(text, CAPTION_SCALE)
    draw_panel(image, (0, 0), (width + 3 * PAD + 2, height + 2 * PAD))
    cv2.rectangle(image, (0, 0), (2, height + 2 * PAD), accent, -1)
    cv2.putText(image, text, (2 * PAD + 2, height + PAD // 2), FONT, CAPTION_SCALE, INK_SECONDARY, 1, LINE)


def draw_segmentation(frame, result, config=OverlayConfig()):
    """Blend the semantic mask into the frame; returns the same array."""
    if result is None or not config.show_segmentation:
        return frame
    tint = np.zeros_like(frame)
    for value, color in CATEGORY_COLORS.items():
        tint[result.mask == value] = color
    painted = result.mask != 255
    if painted.any():
        blended = cv2.addWeighted(frame, 1 - config.segmentation_alpha, tint, config.segmentation_alpha, 0)
        frame[painted] = blended[painted]
    return frame


def track_labels(track, config=OverlayConfig()):
    """Label lines for one track, shortest first."""
    if getattr(track, 'missed_frames', 0):
        return [f'#{track.track_id} {track.class_name} PREDICTED',
                f'last seen {track.seconds_since_detection:.1f}s ago']
    lines = [f'#{track.track_id} {track.class_name} {track.confidence:.2f}']
    if track.apriltag_id is not None:
        lines[0] += f' tag{track.apriltag_id}'
    if config.show_distance and track.distance_m is not None:
        measured = f'{track.distance_m:.1f} m'
        if track.relative_velocity_mps is not None:
            measured += f'  {track.relative_velocity_mps:+.1f} m/s'
        if track.measurement_source:
            measured += f'  [{track.measurement_source}]'
        lines.append(measured)
    if config.show_ttc and track.ttc_s is not None:
        lines.append(f'RISK  TTC {track.ttc_s:.1f} s')
    if config.show_motion and track.motion is not None:
        lines.append(motion_label(track.motion))
    if track.glare_suspect:
        lines.append('glare suspect')
    return lines


def motion_label(motion):
    """One line of bearings-only state, never a range the geometry has not earned.

    Anything short of 'converged' is drawn as the interval it really is, so a
    watchstander cannot read an ambiguous point estimate as a fix.
    """
    if motion.status == 'converged':
        return f'TMA  {motion.range_m:.0f} m  {motion.speed_mps:.1f} m/s  {motion.course_deg:03.0f}deg'
    if motion.status == 'initializing':
        return 'TMA  initializing'
    return f'TMA  {motion.range_low_m:.0f}-{motion.range_high_m:.0f} m  ambiguous'


def draw_track_box(image, track, color):
    """Box in the track's status color, with the line style that repeats the status.

    A dark ring rides just outside the box: the status colors are stepped for a
    dark surface, and bright water or the near end of the depth ramp is not one.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in track.bbox_xyxy)
    state = track_state(track)
    if state in ('UNKNOWN', 'PREDICTED'):
        draw_dashed_rect(image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), PLANE, 1)
        draw_dashed_rect(image, (x1, y1), (x2, y2), color, 1)
    else:
        thickness = 3 if state == 'RISK' else 2
        cv2.rectangle(image, (x1 - thickness, y1 - thickness), (x2 + thickness, y2 + thickness),
                      PLANE, 1, LINE)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, LINE)
    return x1, y1, x2, y2


def draw_tracks(frame, tracks, config=OverlayConfig()):
    """Boxes, ids, class, confidence, range, velocity, TTC and the water-contact point."""
    if not config.show_tracks:
        return frame
    placed = []
    for track in tracks:
        color = track_color(track)
        x1, y1, x2, y2 = draw_track_box(frame, track, color)
        lines = track_labels(track, config)
        block_height = sum(text_size(line)[1] for line in lines) + PAD * (len(lines) + 1)
        draw_label_block(frame, lines, (x1, y1 - block_height - 4), color, occupied=placed)
        if track.glare_suspect:
            cv2.drawMarker(frame, (x2, y1), GLARE_COLOR, cv2.MARKER_TRIANGLE_UP, 9, 2)
        if config.show_water_contact and track.water_contact_px is not None:
            u, v = (int(round(value)) for value in track.water_contact_px)
            cv2.drawMarker(frame, (u, v), PLANE, cv2.MARKER_TILTED_CROSS, 11, 3)
            cv2.drawMarker(frame, (u, v), CONTACT_COLOR, cv2.MARKER_TILTED_CROSS, 10, 1)
    return frame


def draw_obstacle_contours(frame, obstacles, config=OverlayConfig()):
    """Segmented regions that no track covers, so unfused hazards stay visible."""
    if not (config.show_segmentation and config.show_tracks):
        return frame
    for obstacle in obstacles:
        x1, y1, x2, y2 = (int(round(v)) for v in obstacle.bbox_xyxy)
        draw_dashed_rect(frame, (x1, y1), (x2, y2), UNKNOWN_COLOR, 1, dash=4)
    return frame


def status_text(timer, segmentation=None):
    """One line with per-stage timings and the effective frame rate."""
    text = timer.format() if timer is not None else ''
    if segmentation is not None:
        text += f'  seg: {segmentation.inference_time_ms:.0f} ms' + (' (reused)' if segmentation.reused else '')
    return text


def draw_status(frame, timer, segmentation=None, config=OverlayConfig()):
    """Timings in the frame's bottom-left; compose() puts them in the footer instead."""
    if not (config.show_tracks or config.show_segmentation):
        return frame
    cv2.putText(frame, status_text(timer, segmentation), (10, frame.shape[0] - 10), FONT,
                LABEL_SCALE, TEXT_COLOR, 1, LINE)
    return frame


def draw_roi(frame, band, config=OverlayConfig()):
    """Dashed lines at the edges of the detector search band."""
    if band is None or not config.show_tracks:
        return frame
    width = frame.shape[1]
    for row in band:
        y = int(round(min(max(row, 0), frame.shape[0] - 1)))
        for x in range(0, width, 16):
            cv2.line(frame, (x, y), (min(x + 8, width - 1), y), ROI_COLOR, 1, LINE)
    return frame


def render(frame, result, config=OverlayConfig(), status=True):
    """Full debug view of one PipelineResult, drawn on a copy of the frame.

    `status=False` for a panel handed to compose(), which prints the timings in
    its footer instead of over the image.
    """
    canvas = frame.copy()
    draw_segmentation(canvas, result.segmentation, config)
    draw_roi(canvas, result.roi_band, config)
    draw_obstacle_contours(canvas, result.unfused_obstacles, config)
    draw_tracks(canvas, result.tracks, config)
    if status:
        draw_status(canvas, result.timer, result.segmentation, config)
    if config.show_tracks or config.show_segmentation:
        draw_caption(canvas, 'CAMERA  detection and tracking')
    return canvas


# --- depth panel ---------------------------------------------------------

COLORBAR_WIDTH = 9
COLORBAR_MARGIN = 10
COLORBAR_TICKS = 5
COLORBAR_GUTTER = 62   # bar, margin and the widest tick label; track cards stay clear of it


def depth_lut():
    """The map's own ramp, imported lazily so drawing an overlay never loads torch."""
    from boatdet.depth import DEPTH_LUT

    return DEPTH_LUT


def calibrated(value, calibration=(1.0, 0.0)):
    scale, offset = calibration
    return scale * value + offset


def draw_colorbar(panel, span, calibration=(1.0, 0.0), ticks=COLORBAR_TICKS):
    """Vertical ramp with meter ticks: the depth map's legend.

    Tick labels are the calibrated meters the track labels report, so the two
    readings on screen cannot disagree.
    """
    height, width = panel.shape[:2]
    top = HEADER_HEIGHT if height > 3 * HEADER_HEIGHT else 2
    bottom = height - max(18, height // 12)
    if bottom - top < 24 or width < 80:
        return panel
    right = width - COLORBAR_MARGIN
    left = right - COLORBAR_WIDTH
    draw_panel(panel, (width - COLORBAR_GUTTER, top - 14), (width, bottom + 12), 0.55, PLANE)
    # Top of the bar is the near end, matching colorize_depth (near = light).
    ramp = cv2.applyColorMap(np.linspace(255, 0, bottom - top).astype(np.uint8).reshape(-1, 1), depth_lut())
    panel[top:bottom, left:right] = cv2.resize(ramp, (COLORBAR_WIDTH, bottom - top),
                                               interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (left - 1, top - 1), (right, bottom), INK_MUTED, 1)
    cv2.putText(panel, 'depth', (width - COLORBAR_GUTTER + PAD, top - 4), FONT,
                LABEL_SCALE, INK_MUTED, 1, LINE)
    near, far = (calibrated(v, calibration) for v in span)
    for index in range(ticks):
        fraction = index / (ticks - 1)
        y = int(round(top + fraction * (bottom - top - 1)))
        label = f'{near + fraction * (far - near):.0f} m'
        label_width, label_height = text_size(label, LABEL_SCALE)
        x = left - 6 - label_width
        cv2.putText(panel, label, (x, y + label_height // 2 - 2), FONT, LABEL_SCALE, INK_SECONDARY, 1, LINE)
        cv2.line(panel, (left - 3, y), (left - 1, y), INK_MUTED, 1)
    return panel


def render_depth(depth_color, result=None, span=None, calibration=(1.0, 0.0), config=OverlayConfig()):
    """The depth map with the tracked vessels drawn on it and a colorbar legend.

    The same boxes as the camera panel, so a viewer can check that what the
    detector calls a ship is the thing the depth map puts at that range.
    """
    panel = depth_color.copy()
    tracks = getattr(result, 'tracks', result) or []
    placed = []
    if config.show_tracks:
        for track in tracks:
            color = track_color(track)
            x1, y1, x2, y2 = draw_track_box(panel, track, color)
            lines = [f'#{track.track_id}']
            if track.missed_frames:
                lines[0] += ' PREDICTED'
            elif config.show_distance and track.distance_m is not None:
                lines[0] += f'  {track.distance_m:.1f} m'
            if track.measurement_source and not track.missed_frames:
                lines[0] += f'  [{track.measurement_source}]'
            draw_label_block(panel, lines, (x1, y2 + 4), color, occupied=placed,
                             right_limit=panel.shape[1] - COLORBAR_GUTTER if span is not None else None)
    if span is not None:
        draw_colorbar(panel, span, calibration)
    draw_caption(panel, 'DEPTH  metric, monocular')
    return panel


# --- composed frame ------------------------------------------------------

LEGEND_ITEMS = (
    ('box', TRACK_COLOR, 'Tracked vessel'),
    ('dashed', UNKNOWN_COLOR, 'Unknown obstacle'),
    ('thick', WARNING_COLOR, 'Collision risk (TTC)'),
    ('triangle', GLARE_COLOR, 'Glare suspect'),
    ('marker', CONTACT_COLOR, 'Waterline contact'),
    ('dotted', ROI_COLOR, 'Detector search band'),
    ('ramp', None, 'Depth: near to far'),
)
LEGEND_ROWS = 2
LEGEND_ROW_HEIGHT = 18


def draw_swatch(strip, kind, color, x, y, width=20, height=11):
    """The legend swatch repeats the mark's real line style, not just its hue."""
    top, bottom = y - height // 2, y + height // 2
    if kind == 'box':
        cv2.rectangle(strip, (x, top), (x + width, bottom), color, 2, LINE)
    elif kind == 'thick':
        cv2.rectangle(strip, (x, top), (x + width, bottom), color, 3, LINE)
    elif kind == 'dashed':
        draw_dashed_rect(strip, (x, top), (x + width, bottom), color, 1, dash=4, line_type=cv2.LINE_8)
    elif kind == 'marker':
        cv2.drawMarker(strip, (x + width // 2, y), color, cv2.MARKER_TILTED_CROSS, 10, 1)
    elif kind == 'triangle':
        cv2.drawMarker(strip, (x + width // 2, y), color, cv2.MARKER_TRIANGLE_UP, 9, 2)
    elif kind == 'dotted':
        for dash_x in range(x, x + width, 8):
            cv2.line(strip, (dash_x, y), (min(dash_x + 4, x + width), y), color, 1, LINE)
    elif kind == 'ramp':
        ramp = cv2.applyColorMap(np.linspace(255, 0, width).astype(np.uint8).reshape(1, -1), depth_lut())
        strip[top:bottom, x:x + width] = cv2.resize(ramp, (width, bottom - top),
                                                    interpolation=cv2.INTER_NEAREST)
    return x + width + PAD + 2


def draw_legend(width, height=LEGEND_HEIGHT, status=''):
    """Footer strip: what every mark on the two panels means, plus the timings.

    Items wrap onto a second row rather than run into the status text, so a
    narrower working size drops nothing from the legend.
    """
    strip = np.full((height, width, 3), PLANE, np.uint8)
    cv2.line(strip, (0, 0), (width, 0), HAIRLINE, 1)
    status_width = text_size(status, LABEL_SCALE)[0] if status else 0
    if status:
        cv2.putText(strip, status, (max(12, width - status_width - 12), 19), FONT,
                    LABEL_SCALE, INK_MUTED, 1, LINE)
    x, row = 12, 0
    for kind, color, label in LEGEND_ITEMS:
        label_width, _ = text_size(label, LABEL_SCALE)
        budget = width - 12 - (status_width + 20 if row == 0 else 0)
        if x + 26 + label_width > budget:
            row, x = row + 1, 12
            if row >= LEGEND_ROWS:
                break
        y = 15 + row * LEGEND_ROW_HEIGHT
        x = draw_swatch(strip, kind, color, x, y)
        cv2.putText(strip, label, (x, y + 4), FONT, LABEL_SCALE, INK_SECONDARY, 1, LINE)
        x += label_width + 16
    return strip


def draw_header(width, title, subtitle='', height=HEADER_HEIGHT):
    strip = np.full((height, width, 3), PLANE, np.uint8)
    cv2.putText(strip, title, (12, height - 10), FONT, CAPTION_SCALE, INK, 1, LINE)
    if subtitle:
        subtitle_width, _ = text_size(subtitle, LABEL_SCALE)
        cv2.putText(strip, subtitle, (max(12, width - subtitle_width - 12), height - 10), FONT,
                    LABEL_SCALE, INK_MUTED, 1, LINE)
    cv2.line(strip, (0, height - 1), (width, height - 1), HAIRLINE, 1)
    return strip


def composed_size(panel_size, panels=2):
    """(width, height) of the frame compose() returns for panels of this size."""
    width, height = panel_size
    return width * panels, height + HEADER_HEIGHT + LEGEND_HEIGHT


def compose(panels, title='BOAT DETECTION', subtitle='', status=''):
    """Header, the panels side by side, and the legend footer as one frame."""
    body = cv2.hconcat(list(panels))
    width = body.shape[1]
    return cv2.vconcat([draw_header(width, title, subtitle), body, draw_legend(width, status=status)])
