"""
Shared red-dot detection utility for ok_tasks.

HSV-based red dot detection with morphological cleanup and connected-component
filtering. Extracted from HomeRedDotTask so daily tasks can reuse the same
detection logic without duplicating it ("不要重复造轮子").

Usage:
    from ok_tasks._red_dot import detect_red_dot, roi_center, roi_to_pixels
    dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
"""

from typing import Optional, Tuple

import cv2
import numpy as np

# Built-in fallbacks so detect_red_dot works correctly even when called without
# explicit defaults. Mirrors the DEFAULT_CONFIG values used by the daily tasks.
_BUILTIN_DEFAULTS = {
    "hsv_lower1": [0, 120, 120],
    "hsv_upper1": [10, 255, 255],
    "hsv_lower2": [170, 120, 120],
    "hsv_upper2": [180, 255, 255],
    "min_red_pixels": 20,
    "morph_open_size": 3,
    "min_component_area": 16,
    "max_component_area": 1200,
    "max_component_aspect": 1.8,
    "min_component_circularity": 0.45,
    "min_component_fill_ratio": 0.45,
}


def roi_to_pixels(roi, w, h) -> Tuple[int, int, int, int]:
    """Convert a relative roi ``[x1, y1, x2, y2]`` (0-1) to pixel coords."""
    x1 = max(0, int(w * roi[0]))
    y1 = max(0, int(h * roi[1]))
    x2 = min(w, int(w * roi[2]))
    y2 = min(h, int(h * roi[3]))
    return x1, y1, x2, y2


def roi_center(roi, w, h) -> Tuple[int, int]:
    """Return the pixel center of a relative roi ``[x1, y1, x2, y2]`` (0-1)."""
    x1, y1, x2, y2 = roi_to_pixels(roi, w, h)
    return (x1 + x2) // 2, (y1 + y2) // 2


def detect_red_dot(frame: np.ndarray, roi, config=None, defaults=None) -> Optional[Tuple[int, int]]:
    """
    Detect a red dot within ``roi`` of ``frame``.

    Args:
        frame: BGR image (numpy array).
        roi: ``[x1, y1, x2, y2]`` in relative coords (0-1).
        config: dict-like with ``.get(key, default)`` for per-task overrides. May be None.
        defaults: dict of default values (e.g. DEFAULT_CONFIG). May be None.

    Returns:
        ``(cx, cy)`` in frame pixel coords, or ``None`` if no red dot found.
    """

    def g(key):
        # Mirrors: self.config.get(key, DEFAULT_CONFIG[key])
        if config is not None:
            v = config.get(key, None)
            if v is not None:
                return v
        if defaults is not None:
            d = defaults.get(key)
            if d is not None:
                return d
        return _BUILTIN_DEFAULTS.get(key)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_to_pixels(roi, w, h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower1 = np.array(g("hsv_lower1") or [0, 120, 120])
    upper1 = np.array(g("hsv_upper1") or [10, 255, 255])
    lower2 = np.array(g("hsv_lower2") or [170, 120, 120])
    upper2 = np.array(g("hsv_upper2") or [180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    ksize = int(g("morph_open_size") or 0)
    if ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    pixels = int(cv2.countNonZero(mask))
    if pixels < int(g("min_red_pixels") or 0):
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(g("min_component_area") or 0)
    max_area = float(g("max_component_area") or 0)
    max_aspect = float(g("max_component_aspect") or 0)
    min_circularity = float(g("min_component_circularity") or 0)
    min_fill = float(g("min_component_fill_ratio") or 0)

    best = None
    best_score = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue
        aspect = (bw / float(bh)) if bw >= bh else (bh / float(bw))
        if aspect > max_aspect:
            continue

        perimeter = float(cv2.arcLength(cnt, True))
        if perimeter <= 0:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        fill_ratio = area / float(bw * bh)
        if fill_ratio < min_fill:
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        score = area * circularity
        if score > best_score:
            best_score = score
            cx = int(m["m10"] / m["m00"]) + x1
            cy = int(m["m01"] / m["m00"]) + y1
            best = (cx, cy)

    return best
