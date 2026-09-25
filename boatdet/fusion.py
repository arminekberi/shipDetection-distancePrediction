"""Detector and segmentation fusion, and the water-contact point.

Segmentation is a confidence and safety signal here, not a veto: a detection
that segmentation disagrees with is weakened and flagged, never dropped. The
detector is the trained, evaluated component; the mask is the wider net.
"""
from boatdet.config import (TAGGED_VESSEL_CLASS, TAGGED_VESSEL_CLASS_ID, UNKNOWN_OBSTACLE_CLASS,
                            UNKNOWN_OBSTACLE_CLASS_ID, FusionConfig)
from boatdet.detection import Detection
from boatdet.mot import iou
from boatdet.segmentation import water_contact_point

# Highest available water-contact source wins; see select_water_contact. A tag is
# deliberately absent: it is mounted above the waterline, so it localizes identity
# and bearing, never the hull's contact with the water.
CONTACT_SOURCES = ('keypoint', 'segmentation', 'bbox_bottom')


def crop_mask(mask, box):
    """(view, x_offset, y_offset) of a working-frame xyxy box inside a mask."""
    height, width = mask.shape
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, min(width, int(x1))), max(0, min(width, int(x2)))))
    y1, y2 = sorted((max(0, min(height, int(y1))), max(0, min(height, int(y2)))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None, 0, 0
    return mask[y1:y2, x1:x2], x1, y1


def select_water_contact(detection, segmentation=None, keypoint=None):
    """((u, v), source) for range estimation, by descending source quality.

    A keypoint model localizes the hull better than a mask, and a mask beats the
    bottom of the box, which reflections and wake push below the real waterline.
    """
    if keypoint is not None:
        return tuple(float(v) for v in keypoint), 'keypoint'
    if segmentation is not None:
        component, x0, y0 = crop_mask(segmentation.obstacle_mask, detection.bbox_xyxy)
        if component is not None and component.any():
            water, _, _ = crop_mask(segmentation.water_mask, detection.bbox_xyxy)
            point = water_contact_point(component, water)
            if point is not None:
                return (point[0] + x0, point[1] + y0), 'segmentation'
    return detection.bottom_center_px, 'bbox_bottom'


def annotate(detections, segmentation, config=FusionConfig()):
    """Attach segmentation agreement to detections, adjusting confidence only downward."""
    if segmentation is None:
        return detections
    for detection in detections:
        detection.sky_fraction = segmentation.category_fraction(detection.bbox_xyxy, segmentation.sky_mask)
        detection.obstacle_fraction = segmentation.category_fraction(detection.bbox_xyxy,
                                                                     segmentation.obstacle_mask)
        detection.segmentation_confirmed = detection.obstacle_fraction >= config.obstacle_fraction
        # Mostly sky and nothing solid in the box: probably a horizon artifact.
        if detection.sky_fraction >= config.sky_fraction and not detection.segmentation_confirmed:
            detection.confidence *= config.sky_conf_scale
    return detections


def unknown_obstacles(obstacles, detections, config=FusionConfig()):
    """Segmented regions no detection explains, as unknown_obstacle detections.

    They are never relabelled as boats: an unrecognized hazard has to stay
    unrecognized in the log and in the UI.
    """
    unmatched = []
    for obstacle in obstacles:
        if any(iou(obstacle.bbox_xyxy, d.bbox_xyxy) >= config.unknown_iou for d in detections):
            continue
        unmatched.append(Detection(
            bbox_xyxy=obstacle.bbox_xyxy,
            confidence=config.unknown_conf,
            class_id=UNKNOWN_OBSTACLE_CLASS_ID,
            class_name=UNKNOWN_OBSTACLE_CLASS,
            source='segmentation',
            water_contact_px=obstacle.water_contact_px,
            water_contact_source='segmentation' if obstacle.water_contact_px else None,
            obstacle_fraction=1.0,
            segmentation_confirmed=True))
    return unmatched


def tagged_vessel_detections(tags, detections, config=FusionConfig()):
    """Tags that no detection covers, as vessel detections carrying identity only.

    The box is the tag, not the hull, so no water-contact point is derived from
    it: a range would be measured to a marker above the waterline.
    """
    if not config.apriltag_seeds_tracks:
        return []
    seeded = []
    for tag in tags:
        if config.apriltag_vessel_ids and tag.tag_id not in config.apriltag_vessel_ids:
            continue
        if any(tag_for_detection(detection, [tag]) is not None for detection in detections):
            continue
        seeded.append(Detection(bbox_xyxy=tag.bbox_xyxy, confidence=float(tag.confidence),
                                class_id=TAGGED_VESSEL_CLASS_ID, class_name=TAGGED_VESSEL_CLASS,
                                source='apriltag'))
    return seeded


def tag_for_detection(detection, tags):
    """The single tag inside this box, or None when zero or several are visible."""
    x1, y1, x2, y2 = detection.bbox_xyxy
    inside = [tag for tag in tags if x1 <= tag.center_px[0] <= x2 and y1 <= tag.center_px[1] <= y2]
    return inside[0] if len(inside) == 1 else None


def fuse(detections, segmentation=None, obstacles=(), tags=(), config=FusionConfig()):
    """Detector output plus segmentation-only obstacles, each with a water-contact point."""
    fused = list(detections)
    if config.enabled:
        fused = annotate(fused, segmentation, config)
        fused += unknown_obstacles(obstacles, fused, config)
    fused += tagged_vessel_detections(tags, fused, config)
    for detection in fused:
        if detection.source == 'apriltag':
            continue  # the box is the marker; the hull waterline is unknown
        point, source = select_water_contact(detection, segmentation if config.enabled else None)
        # A segmentation-derived contact from the obstacle blob itself is already set.
        if detection.water_contact_px is None or CONTACT_SOURCES.index(source) < \
                CONTACT_SOURCES.index(detection.water_contact_source or 'bbox_bottom'):
            detection.water_contact_px, detection.water_contact_source = point, source
    return fused
