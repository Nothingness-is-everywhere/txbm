"""
Shared red-dot detection utility for ok_tasks.

HSV-based red dot detection with morphological cleanup and connected-component
filtering. Extracted from HomeRedDotTask so daily tasks can reuse the same
detection logic without duplicating it ("不要重复造轮子").

Usage:
    from ok_tasks._red_dot import detect_red_dot, roi_center, roi_to_pixels
    dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
"""

from typing import Callable, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Breathing-light (pulsing red dot) retry defaults
# ---------------------------------------------------------------------------
# Many in-game red dots pulse: fully opaque red -> semi-transparent -> fully red.
# A single frame grab may catch the low-saturation phase, causing a false
# "no dot" result even when the dot is visually present. Running detection
# across multiple frames with a short inter-frame delay covers multiple
# breathing cycles and reliably eliminates this false-negative.
#
# Tasks that use these defaults override at their own config level via the
# keys below. See AlchemyDispatchTask DEFAULT_CONFIG for a real example.
BREATHE_DEFAULT_RETRY_TIMES = 5
BREATHE_DEFAULT_RETRY_INTERVAL_MS = 200
# Config key names. Tasks should mirror these keys in their DEFAULT_CONFIG so
# the shared helper below can read them uniformly.
CFG_RETRY_TIMES = "red_dot_retry_times"
CFG_RETRY_INTERVAL_MS = "red_dot_retry_interval_ms"


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


# ---------------------------------------------------------------------------
# Breathing-light retry loop (shared across tasks)
# ---------------------------------------------------------------------------

def detect_red_dot_breathing(
    fetch_frame: Callable[[], Optional[np.ndarray]],
    sleep_s: Callable[[float], None],
    roi,
    config=None,
    defaults=None,
    retry_times: Optional[int] = None,
    retry_interval_ms: Optional[int] = None,
    retry_times_cfg_key: str = CFG_RETRY_TIMES,
    retry_interval_cfg_key: str = CFG_RETRY_INTERVAL_MS,
    logger_fn: Optional[Callable[[str], None]] = None,
    label: str = "breathing",
) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray], int]:
    """Multi-frame retry wrapper around ``detect_red_dot`` for breathing dots.

    Game UI red dots often fade in and out. A single-frame detection can miss
    them because the HSV saturation/value thresholds drop below the configured
    limits during the semi-transparent phase. This helper runs detection up to
    ``retry_times`` frames with a short delay between attempts, giving the
    breathing cycle enough time to swing back into full-opacity red.

    Args:
        fetch_frame: Callable that returns the latest frame (BGR uint8), or
            ``None`` if no frame is available. Usually ``lambda: self.next_frame()``
            or similar. The helper calls this on every retry so each attempt
            observes a new frame.
        sleep_s: Callable that sleeps for the given seconds between attempts.
            Usually ``self.sleep``. Must not raise; if it does the exception
            propagates out (caller's responsibility to handle).
        roi: Relative ROI passed through to ``detect_red_dot``.
        config: Per-task config dict forwarded to ``detect_red_dot``. Also read
            for retry parameters when ``retry_times`` / ``retry_interval_ms``
            are ``None``.
        defaults: Defaults dict forwarded to ``detect_red_dot``.
        retry_times: Explicit retry count. When ``None`` (recommended), read
            from ``config[retry_times_cfg_key]`` or fall back to
            ``BREATHE_DEFAULT_RETRY_TIMES``.
        retry_interval_ms: Explicit interval in ms. When ``None`` (recommended),
            read from ``config[retry_interval_cfg_key]`` or fall back to
            ``BREATHE_DEFAULT_RETRY_INTERVAL_MS``.
        retry_times_cfg_key: Config key name used to look up ``retry_times``.
            Override when a single task has *multiple* breathing-dot locations
            that need independent parameters (e.g. entry dot vs per-step dot).
        retry_interval_cfg_key: Same as above but for the interval.
        logger_fn: Optional logger callable (e.g. ``logger.info``) used to
            record per-attempt progress. When ``None`` no progress logs are
            emitted; all failures still produce a "no dot after N retries"
            message via ``logger_fn`` when available.
        label: Short label prepended to progress/diagnostic logs (e.g. "step2",
            "home-entry", "friend-stamina").

    Returns:
        ``(dot, last_frame, attempts)`` where:
          - ``dot`` is the detected ``(cx, cy)`` or ``None``.
          - ``last_frame`` is the most recent non-``None`` frame the helper
            observed. Callers often need the frame for downstream shape-based
            centre calculations even when detection missed.
          - ``attempts`` is how many frames were actually inspected (counting
            stops on the first success or when ``retry_times`` is reached).
            If every ``fetch_frame()`` returned ``None``, ``attempts`` equals
            the number of tries made before bailing.

    Behaviour notes (matches AlchemyDispatchTask step2 proven fix):
        - Returns on the FIRST frame that produces a non-None dot.
        - ``attempts`` is capped at the resolved retry count.
        - ``fetch_frame`` returning ``None`` is treated as "no frame this
          tick"; the helper continues to the next retry so transient frame
          unavailability does not abort detection.
    """

    # Resolve retry params with explicit > config > global default priority.
    if retry_times is None:
        if config is not None:
            retry_times = config.get(retry_times_cfg_key, None)
        if retry_times is None:
            retry_times = BREATHE_DEFAULT_RETRY_TIMES
    retry_times = max(1, int(retry_times))

    if retry_interval_ms is None:
        if config is not None:
            retry_interval_ms = config.get(retry_interval_cfg_key, None)
        if retry_interval_ms is None:
            retry_interval_ms = BREATHE_DEFAULT_RETRY_INTERVAL_MS
    retry_interval_s = max(0.0, float(retry_interval_ms) / 1000.0)

    last_frame = None
    attempts = 0

    for i in range(1, retry_times + 1):
        attempts = i
        frame = fetch_frame()
        if frame is None:
            if logger_fn and i == retry_times:
                logger_fn(f"[{label}] 重检 {i}/{retry_times}：仍无画面，放弃检测")
            if i < retry_times and retry_interval_s > 0:
                sleep_s(retry_interval_s)
            continue

        last_frame = frame
        dot = detect_red_dot(frame, roi, config, defaults)
        if dot is not None:
            if logger_fn:
                logger_fn(f"[{label}] 第 {i}/{retry_times} 帧命中红点 {dot}")
            return dot, last_frame, attempts

        if i < retry_times and retry_interval_s > 0:
            sleep_s(retry_interval_s)

    # All attempts exhausted.
    if logger_fn:
        logger_fn(
            f"[{label}] {retry_times} 帧均未检测到红点（可能确实无红点 or 呼吸灯相位窗口不足）"
        )
    return None, last_frame, attempts
