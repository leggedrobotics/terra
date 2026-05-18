import cv2
import numpy as np


def _to_abc_from_points(p1, p2):
    """Return line coefficients A,B,C for Ax + By + C = 0 from two points."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return {"A": float(a), "B": float(b), "C": float(c)}


def _point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def _get_border_pixels(dig_mask: np.ndarray):
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(dig_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    border = np.logical_and(dig_mask, np.logical_not(eroded))
    rows, cols = np.where(border)
    return np.stack([cols, rows], axis=1).astype(np.float32)


def _polygon_covers_border(pts: np.ndarray, border_pixels: np.ndarray, max_dist: float):
    if len(pts) < 3 or len(border_pixels) == 0:
        return False
    max_seen_dist = 0.0
    for pixel in border_pixels:
        nearest = min(
            _point_to_segment_distance(pixel, pts[i], pts[(i + 1) % len(pts)])
            for i in range(len(pts))
        )
        max_seen_dist = max(max_seen_dist, nearest)
        if max_seen_dist > max_dist:
            return False
    return True


def _approx_foundation_border_polygon(dig_mask: np.ndarray):
    """
    Approximate the rasterized foundation border with the fewest useful segments.
    """
    mask_u8 = (dig_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    border_pixels = _get_border_pixels(dig_mask)
    approx = None
    max_border_error_px = 1.8
    for eps_fraction in (0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01):
        candidate = cv2.approxPolyDP(contour, eps_fraction * peri, True).reshape(-1, 2)
        if _polygon_covers_border(candidate, border_pixels, max_border_error_px):
            approx = candidate
            break
    if approx is None:
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True).reshape(-1, 2)
    return approx.reshape(-1, 2).astype(np.float32)


def build_foundation_border_axes_from_mask(dig_mask: np.ndarray):
    """
    Build border axes from the rasterized foundation mask.
    Uses contour extraction and adaptive polygon approximation to get the fewest
    edge segments that still cover the raster border.
    """
    pts = _approx_foundation_border_polygon(dig_mask)
    if pts.shape[0] < 2:
        return []
    axes = []
    for i in range(pts.shape[0]):
        p1 = pts[i]
        p2 = pts[(i + 1) % pts.shape[0]]
        if np.all(p1 == p2):
            continue
        axes.append(_to_abc_from_points(p1, p2))
    return axes


def build_foundation_border_lines_from_mask(dig_mask: np.ndarray):
    pts = _approx_foundation_border_polygon(dig_mask)
    if pts.shape[0] < 2:
        return []
    lines = []
    for i in range(pts.shape[0]):
        p1 = pts[i]
        p2 = pts[(i + 1) % pts.shape[0]]
        if np.all(p1 == p2):
            continue
        lines.append(
            [
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
            ]
        )
    return lines


def build_foundation_border_metadata(dig_mask: np.ndarray):
    """
    Build border-axis metadata for the final Terra foundation mask.

    Source metadata may be in the pre-downsampled/pre-padded image frame. Rebuild
    from the final raster mask so the generated axes match the playable map.
    """
    return {
        "foundation_border_axes_ABC": build_foundation_border_axes_from_mask(dig_mask),
        "foundation_border_lines_pts": build_foundation_border_lines_from_mask(dig_mask),
    }
