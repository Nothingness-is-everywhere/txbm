"""
Home red-dot detection task for 天下布魔 (Tianxia Bumo).

Detects the red notification dot on the profile (personal info) button on the
game home screen and taps it.

Flow:
  1. Template-match the profile button to confirm we are on the home screen.
  2. If on home, run HSV red-color detection on the button ROI.
  3. If red pixels are found, tap the red-dot centroid.
  4. (后续步骤待补充) Subsequent steps after tapping the red dot — to be added
     once the user specifies the follow-up flow.

Configuration is loaded from configs/HomeRedDotTask.json
(managed by the task config system).
"""

import time
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ok import TriggerTask

logger = logging.getLogger("HomeRedDotTask")

# ---------------------------------------------------------------------------
# Project root resolution (all paths relative to repo root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(relative_path: str) -> Path:
    """Resolve a repository-relative path to an absolute path."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
# Selection: x=37, y=69, w=279, h=107  (rx=0.0343, ry=0.0359, rw=0.2583, rh=0.0557)
# ROI format: [x_start, y_start, x_end, y_end] as ratios of screen size.
_PROFILE_ROI = [0.0343, 0.0359, 0.2926, 0.0916]

DEFAULT_CONFIG = {
    "_enabled": True,
    # Home-screen detection via profile-button template match
    "home_template_path": "templates/home_profile_button.png",
    "home_template_roi": list(_PROFILE_ROI),
    "home_threshold": 0.70,
    # Red-dot detection (HSV) — searched inside the same profile-button ROI
    "red_dot_roi": list(_PROFILE_ROI),
    # HSV red range. Red wraps around hue 0/180, so two intervals are used.
    "hsv_lower1": [0, 80, 80],
    "hsv_upper1": [10, 255, 255],
    "hsv_lower2": [170, 80, 80],
    "hsv_upper2": [180, 255, 255],
    # Minimum red pixels to count as a dot (avoids noise)
    "min_red_pixels": 8,
    # Morphological cleanup
    "morph_open_size": 3,
    # Cooldown to avoid repeated clicks on the same dot
    "cooldown_seconds": 5.0,
    # Pause after a click before any follow-up step
    "post_click_sleep": 1.0,
}


class HomeRedDotTask(TriggerTask):
    """
    Detects the red notification dot on the home-screen profile button and taps it.

    定时触发的主页红点检测任务。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "主页红点点击"
        self.description = "检测主页个人信息按钮上的红点并点击"
        self.trigger_interval = 3
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        self._home_tpl: Optional[np.ndarray] = None
        self._last_click_time: float = 0.0

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self._load_template()

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------
    def _load_template(self):
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        abs_path = str(resolve_repo_path(rel))
        tpl = cv2.imread(abs_path, cv2.IMREAD_COLOR)
        if tpl is None:
            logger.warning(f"Home template not found, home detection disabled: {abs_path}")
            self._home_tpl = None
        else:
            self._home_tpl = tpl
            h, w = tpl.shape[:2]
            logger.info(f"Home template loaded: {abs_path} ({w}x{h})")

    # ------------------------------------------------------------------
    # Main entry point (called by the framework)
    # ------------------------------------------------------------------
    def run(self):
        """Main trigger entry point called periodically by the framework."""
        frame = self.executor.frame
        if frame is None:
            logger.debug("No frame available, skipping")
            return False

        # 1. Confirm we are on the home screen
        if not self._is_home(frame):
            return False

        # 2. Detect the red dot
        dot = self._detect_red_dot(frame)
        if dot is None:
            logger.debug("No red dot detected on profile button")
            return False

        # 3. Cooldown check
        cooldown = self.config.get("cooldown_seconds", DEFAULT_CONFIG["cooldown_seconds"])
        now = time.time()
        if now - self._last_click_time < cooldown:
            logger.debug(f"Red dot found but in cooldown ({now - self._last_click_time:.1f}s)")
            return False

        # 4. Tap the red dot
        cx, cy = dot
        logger.info(f"Red dot detected, tapping at ({cx}, {cy})")
        self.click(cx, cy)
        self._last_click_time = time.time()

        # 5. Wait before any follow-up step
        self.sleep(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        # TODO: 后续步骤待用户确认 —— 在此处补充点击红点之后的流程
        # e.g. enter the profile menu, claim rewards, then return to home.

        return True

    # ------------------------------------------------------------------
    # Home-screen detection
    # ------------------------------------------------------------------
    def _is_home(self, frame: np.ndarray) -> bool:
        """Template-match the profile button to confirm we are on the home screen."""
        if self._home_tpl is None:
            return False

        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_to_pixels(roi, w, h)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return False

        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(self._home_tpl, cv2.COLOR_BGR2GRAY)
        th, tw = tpl_gray.shape[:2]
        if region_gray.shape[0] < th or region_gray.shape[1] < tw:
            return False

        res = cv2.matchTemplate(region_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        threshold = self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"])
        if max_val >= threshold:
            logger.debug(f"Home screen confirmed (conf={max_val:.3f})")
            return True
        logger.debug(f"Not on home screen (conf={max_val:.3f} < {threshold})")
        return False

    # ------------------------------------------------------------------
    # Red-dot detection (HSV)
    # ------------------------------------------------------------------
    def _detect_red_dot(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red pixels inside the profile-button ROI and return the centroid."""
        roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_to_pixels(roi, w, h)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower1 = np.array(self.config.get("hsv_lower1", DEFAULT_CONFIG["hsv_lower1"]))
        upper1 = np.array(self.config.get("hsv_upper1", DEFAULT_CONFIG["hsv_upper1"]))
        lower2 = np.array(self.config.get("hsv_lower2", DEFAULT_CONFIG["hsv_lower2"]))
        upper2 = np.array(self.config.get("hsv_upper2", DEFAULT_CONFIG["hsv_upper2"]))

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )

        # Morphological open to remove specks
        ksize = self.config.get("morph_open_size", DEFAULT_CONFIG["morph_open_size"])
        if ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        pixels = int(cv2.countNonZero(mask))
        min_pix = self.config.get("min_red_pixels", DEFAULT_CONFIG["min_red_pixels"])
        if pixels < min_pix:
            return None

        # Centroid of the red region
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"]) + x1
        cy = int(moments["m01"] / moments["m00"]) + y1
        logger.info(f"Red dot found: {pixels} pixels, centroid ({cx}, {cy})")
        return (cx, cy)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _roi_to_pixels(roi, w, h) -> Tuple[int, int, int, int]:
        x1 = max(0, int(w * roi[0]))
        y1 = max(0, int(h * roi[1]))
        x2 = min(w, int(w * roi[2]))
        y2 = min(h, int(h * roi[3]))
        return x1, y1, x2, y2
