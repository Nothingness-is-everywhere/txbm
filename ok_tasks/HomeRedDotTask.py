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

from ok.task.task import BaseTask

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
    # 是否每次启动自动执行（配置面板仅显示此项，其他技术配置项隐藏）
    "enable_after_start": False,
    # Home-screen detection via profile-button template match
    "home_template_path": "templates/home_profile_button.png",
    "home_template_roi": list(_PROFILE_ROI),
    "home_threshold": 0.70,
    # Red-dot detection (HSV) — searched inside the same profile-button ROI
    "red_dot_roi": list(_PROFILE_ROI),
    # Follow-up red-dot ROI: after tapping the profile button, a second red
    # dot may appear on the opened page at this region.
    # Selection: x=404,y=270,w=276,h=99 (rx=0.3741, ry=0.1406, rw=0.2556, rh=0.0516)
    "followup_red_dot_roi": [0.3741, 0.1406, 0.6297, 0.1922],
    # Reward red-dot ROI: after tapping the follow-up red dot, a reward
    # button may light up here. Tap only if a red dot is present.
    # Selection: x=597,y=1591,w=306,h=99 (rx=0.5528, ry=0.8286, rw=0.2833, rh=0.0500)
    "reward_red_dot_roi": [0.5528, 0.8286, 0.8361, 0.8786],
    # Close-button ROI: tap the centre to return to the home screen.
    # Selection: x=479,y=1772,w=121,h=118 (rx=0.4435, ry=0.9229, rw=0.1120, rh=0.0615)
    "close_button_roi": [0.4435, 0.9229, 0.5555, 0.9844],
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


class HomeRedDotTask(BaseTask):
    """
    周常日常任务：检测主页个人信息按钮红点并完成领取流程。

    One-time task (shown under the "周常日常" tab). Run flow:
      1. Confirm we are on the home screen.
      2. Detect the red dot on the profile button and tap it.
      3. Detect the follow-up red dot and tap it.
      4. Detect the reward red dot and tap it (if present).
      5. Tap the close button to return to the home screen.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "获取好友体力"
        self.description = "检测主页红点，依次点击红点领取奖励，最后关闭返回主页"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板只显示"是否每次启动执行"，其他技术配置项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'enable_after_start'}
        self.config_description = {"enable_after_start": "是否每次启动执行"}
        self._home_tpl: Optional[np.ndarray] = None
        self._last_click_time: float = 0.0

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self.enable_after_start = self.config.get("enable_after_start", False)
        self._load_template()

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------
    def _load_template(self):
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        abs_path = str(resolve_repo_path(rel))
        # cv2.imread does not support non-ASCII paths on Windows; use
        # np.fromfile + cv2.imdecode instead.
        try:
            tpl = cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Failed to read home template: {abs_path} ({e})")
            tpl = None
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

        # 6. 后续步骤：点击个人信息红点后，检测第二个选区红点并点击
        followup_roi = self.config.get(
            "followup_red_dot_roi", DEFAULT_CONFIG["followup_red_dot_roi"])
        frame2 = self.next_frame()
        if frame2 is None:
            logger.debug("No frame for follow-up red-dot detection")
            return True
        dot2 = self._detect_red_dot(frame2, followup_roi)
        if dot2 is None:
            logger.debug("No follow-up red dot detected")
            return True
        fx, fy = dot2
        logger.info(f"Follow-up red dot detected, tapping at ({fx}, {fy})")
        self.click(fx, fy)
        self.sleep(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        # 7. 后续步骤：检测奖励红点（选区A），有就点击，没有就跳过
        reward_roi = self.config.get(
            "reward_red_dot_roi", DEFAULT_CONFIG["reward_red_dot_roi"])
        frame3 = self.next_frame()
        if frame3 is not None:
            dot3 = self._detect_red_dot(frame3, reward_roi)
            if dot3 is not None:
                # 点击选区A中心（按钮中心）而非红点质心，避免点歪
                x1, y1, x2, y2 = self._roi_to_pixels(
                    reward_roi, frame3.shape[1], frame3.shape[0])
                rx, ry = (x1 + x2) // 2, (y1 + y2) // 2
                logger.info(f"Reward red dot detected, tapping button centre at ({rx}, {ry})")
                self.click(rx, ry)
                self.sleep(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        # 8. 点击关闭按钮（选区B）回到主界面
        close_roi = self.config.get(
            "close_button_roi", DEFAULT_CONFIG["close_button_roi"])
        frame4 = self.next_frame()
        if frame4 is not None:
            x1, y1, x2, y2 = self._roi_to_pixels(
                close_roi, frame4.shape[1], frame4.shape[0])
            close_x = (x1 + x2) // 2
            close_y = (y1 + y2) // 2
            logger.info(f"Closing back to home, tapping close button at ({close_x}, {close_y})")
            self.click(close_x, close_y)
            self.sleep(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

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
    def _detect_red_dot(self, frame: np.ndarray, roi=None) -> Optional[Tuple[int, int]]:
        """Detect red pixels inside the given ROI and return the centroid."""
        if roi is None:
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
