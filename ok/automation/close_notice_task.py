"""
Close notice popup task for 天下布魔 (Tianxia Bumo).

Two-phase flow:
  Phase 1: Detect notice popup (by title template) and click close button.
  Phase 2: Detect post-close screen (by TAP TO START template) and click once.

Uses template matching with OpenCV for feature detection and
adbutils for ADB-based screen capture and input simulation.
"""

import time
import logging
import random
from pathlib import Path
from typing import Optional, Tuple

import adbutils
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("close_notice")

ADB_SERIAL = "127.0.0.1:16384"

NOTICE_TITLE_TPL = "templates/notice_title_v2.png"
CLOSE_BTN_TPL = "templates/close_button_v3.png"
POST_CLOSE_TPL = "templates/tap_to_start.png"

NOTICE_THRESHOLD = 0.70
CLOSE_BTN_THRESHOLD = 0.70
POST_CLOSE_THRESHOLD = 0.65

PHASE1_DELAY = 1.2
PHASE2_DELAY = 0.3

NOTICE_ROI = (0.10, 0.04, 0.90, 0.25)
CLOSE_BTN_ROI = (0.35, 0.80, 0.65, 0.95)
POST_CLOSE_ROI = (0.20, 0.75, 0.80, 0.92)

DEFAULT_CLOSE_POS = (0.50, 0.90)
DEFAULT_TAP_POS = (0.50, 0.85)


class NoticeCloser:
    """Two-phase notice popup handler with post-close navigation."""

    def __init__(self, serial: str = ADB_SERIAL):
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(serial)
        self.width = 1080
        self.height = 1920

        self.notice_title_tpl = self._load_template(NOTICE_TITLE_TPL)
        self.close_btn_tpl = self._load_template(CLOSE_BTN_TPL)
        self.post_close_tpl = self._load_template(POST_CLOSE_TPL)

    def _load_template(self, path: str) -> np.ndarray:
        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"Failed to load template: {path}")
        h, w = template.shape[:2]
        logger.debug(f"Template loaded: {path} ({w}x{h})")
        return template

    def capture_screen(self) -> np.ndarray:
        png_bytes = self.device.shell("screencap -p", encoding=None, timeout=10)
        if not png_bytes or len(png_bytes) == 0:
            raise RuntimeError("Failed to capture screen")
        image_data = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode screenshot")
        self.height, self.width = image.shape[:2]
        return image

    def _match_template(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        roi: Tuple[float, float, float, float],
        threshold: float,
        label: str = "feature",
    ) -> Optional[Tuple[int, int, float]]:
        img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        rx1, ry1, rx2, ry2 = roi
        roi_x1 = int(self.width * rx1)
        roi_y1 = int(self.height * ry1)
        roi_x2 = int(self.width * rx2)
        roi_y2 = int(self.height * ry2)

        roi_region = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]

        result = cv2.matchTemplate(roi_region, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            tpl_h, tpl_w = tpl_gray.shape
            center_x = roi_x1 + max_loc[0] + tpl_w // 2
            center_y = roi_y1 + max_loc[1] + tpl_h // 2
            logger.info(
                f"[{label}] Found at ({center_x}, {center_y}) "
                f"conf={max_val:.3f}"
            )
            return (center_x, center_y, max_val)
        else:
            logger.info(
                f"[{label}] Not found. "
                f"Max conf={max_val:.3f} (threshold={threshold})"
            )
            return None

    def find_notice_popup(
        self, screen: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        return self._match_template(
            screen, self.notice_title_tpl, NOTICE_ROI, NOTICE_THRESHOLD,
            "notice_popup"
        )

    def find_close_button(
        self, screen: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        return self._match_template(
            screen, self.close_btn_tpl, CLOSE_BTN_ROI, CLOSE_BTN_THRESHOLD,
            "close_button"
        )

    def find_post_close(
        self, screen: np.ndarray
    ) -> bool:
        """Detect if current screen is in post-close state (for state recognition only)."""
        result = self._match_template(
            screen, self.post_close_tpl, POST_CLOSE_ROI, POST_CLOSE_THRESHOLD,
            "post_close_state"
        )
        return result is not None

    def find_confirm_button(
        self, screen: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        """Find the confirm/continue button in post-close screen (for clicking)."""
        return self._match_template(
            screen, self.post_close_tpl, POST_CLOSE_ROI, POST_CLOSE_THRESHOLD,
            "confirm_button"
        )

    def click(self, x: int, y: int) -> None:
        self.device.shell(f"input tap {x} {y}", timeout=5)
        logger.info(f"Click at ({x}, {y})")

    def _jitter(self, base_x: int, base_y: int, r: int = 10) -> Tuple[int, int]:
        return (base_x + random.randint(-r, r), base_y + random.randint(-r, r))

    def _delay(self, base: float) -> float:
        v = random.uniform(-0.15 * base, 0.15 * base)
        return max(0.05, base + v)

    def close_notice_popup(self) -> bool:
        """
        Close notice popup if detected (Step B).
        
        Returns:
            True if notice was detected and closed, or no notice detected (skip).
            False on critical error.
        """
        logger.info("=" * 50)
        logger.info("Step B: Detect & Close Notice Popup")
        logger.info("=" * 50)

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return False

        notice_result = self.find_notice_popup(screen)

        if notice_result is None:
            logger.info("skip: No notice popup detected")
            logger.info("=" * 50)
            logger.info("close_notice_popup finished (skipped)")
            logger.info("=" * 50)
            return True

        logger.info(f"Notice popup detected (conf={notice_result[2]:.3f}), proceeding to close")

        close_result = self.find_close_button(screen)

        if close_result is None:
            logger.warning("Close button not found via template, skipping close to avoid misclick")
            logger.info("=" * 50)
            logger.info("close_notice_popup finished (close button not found)")
            logger.info("=" * 50)
            return True

        cx, cy, _ = close_result
        jx, jy = self._jitter(cx, cy, r=12)
        logger.info(f"Clicking close button at ({jx}, {jy})")
        self.click(jx, jy)

        d1 = self._delay(0.5)
        logger.info(f"Waiting {d1:.2f}s for popup to dismiss...")
        time.sleep(d1)

        # Verify close
        try:
            screen2 = self.capture_screen()
            notice_still_there = self.find_notice_popup(screen2)
            if notice_still_there is not None:
                logger.warning(f"Notice popup still visible after close click (conf={notice_still_there[2]:.3f}), retrying once...")
                close_result2 = self.find_close_button(screen2)
                if close_result2 is not None:
                    cx2, cy2, _ = close_result2
                    jx2, jy2 = self._jitter(cx2, cy2, r=12)
                    logger.info(f"Retry clicking close button at ({jx2}, {jy2})")
                    self.click(jx2, jy2)
                    time.sleep(self._delay(0.3))
                else:
                    logger.warning("Close button not found during retry, giving up")
        except Exception as e:
            logger.warning(f"Notice close verification failed: {e}")

        logger.info("=" * 50)
        logger.info("close_notice_popup finished")
        logger.info("=" * 50)
        return True

    def _detect_checkin_screen(self, screen: np.ndarray) -> bool:
        """Detect if we're on the check-in screen."""
        try:
            from ok.automation.checkin_close_task import CheckInCloser
            checkin_closer = CheckInCloser()
            return checkin_closer.find_checkin_screen(screen) is not None
        except Exception as e:
            logger.debug(f"Check-in detection failed: {e}")
            return False

    def _detect_main_screen(self, screen: np.ndarray) -> bool:
        """Detect if we're already on the main screen."""
        try:
            # Check for stamina values or other main screen features
            from ok.automation.stamina_reader import get_stamina, update_global_stamina
            update_global_stamina()
            state = get_stamina(force_refresh=True)
            if state.stamina1.value > 0 or state.stamina2.value > 0:
                return True
            
            # Check for back button (indicates navigable screen)
            from ok.automation.back_navigation_task import BackButtonNavigator
            navigator = BackButtonNavigator()
            return navigator.find_back_button(screen) is not None
        except Exception as e:
            logger.debug(f"Main screen detection failed: {e}")
            return False

    def process_post_close_screen(self, 
                                  short_retries: int = 3, 
                                  short_delay: float = 0.5,
                                  long_wait: float = 3.0,
                                  long_poll_interval: float = 0.5) -> str:
        """
        Process post-close screen (Step C - mandatory).
        Two-layer retry strategy:
          Layer 1: Short retries (3 attempts, 0.5s interval)
          Layer 2: Long wait window (3s, 0.5s poll interval) - also detects check-in/main screen
        
        Returns:
            'success': Confirm button clicked successfully
            'bypass': Skipped because check-in or main screen detected
            'warning': Timed out but proceeding conservatively (no blind clicks)
            'failed': Critical error (screenshot consistently failed)
        """
        logger.info("=" * 50)
        logger.info("Step C: Process Post-Close Screen (Mandatory)")
        logger.info("=" * 50)

        # Layer 1: Short retries
        logger.info(f"Layer 1: Short retries ({short_retries} attempts, {short_delay}s interval)")
        for attempt in range(short_retries):
            try:
                screen = self.capture_screen()
                
                # Check if already at check-in or main screen (bypass)
                if self._detect_checkin_screen(screen):
                    logger.info(f"Bypass: Check-in screen detected, Step C completed")
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished (bypass - check-in detected)")
                    logger.info("=" * 50)
                    return 'bypass'
                
                if self._detect_main_screen(screen):
                    logger.info(f"Bypass: Main screen detected, Step C completed")
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished (bypass - main screen detected)")
                    logger.info("=" * 50)
                    return 'bypass'
                
                # Try to find confirm button
                confirm_result = self.find_confirm_button(screen)
                if confirm_result is not None:
                    px, py, conf = confirm_result
                    jx, jy = self._jitter(px, py, r=10)
                    logger.info(f"Confirm button found (conf={conf:.3f}) on attempt {attempt+1}, clicking at ({jx}, {jy})")
                    self.click(jx, jy)
                    time.sleep(self._delay(0.3))
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished successfully")
                    logger.info("=" * 50)
                    return 'success'
                
                logger.info(f"Confirm button not found on attempt {attempt+1}, retrying in {short_delay}s...")
                
            except Exception as e:
                logger.warning(f"Process post-close attempt {attempt+1} failed: {e}")
            
            if attempt < short_retries - 1:
                time.sleep(short_delay)

        # Layer 2: Long wait window
        logger.info(f"Layer 2: Long wait window ({long_wait}s, {long_poll_interval}s poll interval)")
        long_wait_end = time.time() + long_wait
        long_attempt = 0
        
        while time.time() < long_wait_end:
            long_attempt += 1
            try:
                screen = self.capture_screen()
                
                # Check if already at check-in or main screen (bypass)
                if self._detect_checkin_screen(screen):
                    logger.info(f"Bypass on long wait attempt {long_attempt}: Check-in screen detected")
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished (bypass - check-in detected)")
                    logger.info("=" * 50)
                    return 'bypass'
                
                if self._detect_main_screen(screen):
                    logger.info(f"Bypass on long wait attempt {long_attempt}: Main screen detected")
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished (bypass - main screen detected)")
                    logger.info("=" * 50)
                    return 'bypass'
                
                # Try to find confirm button
                confirm_result = self.find_confirm_button(screen)
                if confirm_result is not None:
                    px, py, conf = confirm_result
                    jx, jy = self._jitter(px, py, r=10)
                    logger.info(f"Confirm button found on long wait attempt {long_attempt} (conf={conf:.3f}), clicking at ({jx}, {jy})")
                    self.click(jx, jy)
                    time.sleep(self._delay(0.3))
                    logger.info("=" * 50)
                    logger.info("process_post_close_screen finished successfully (found in long wait)")
                    logger.info("=" * 50)
                    return 'success'
                
                logger.debug(f"Long wait attempt {long_attempt}: No confirm button, no check-in, no main screen")
                
            except Exception as e:
                logger.warning(f"Long wait attempt {long_attempt} failed: {e}")
            
            time.sleep(long_poll_interval)

        # All attempts exhausted - return warning (not failed, proceed conservatively)
        logger.warning("Step C timeout: No confirm button found after short retries + long wait")
        logger.warning("Proceeding conservatively without blind clicks")
        logger.info("=" * 50)
        logger.info("process_post_close_screen finished (WARNING)")
        logger.info("=" * 50)
        return 'warning'


def run_task():
    try:
        closer = NoticeCloser()
        success = closer.close_notice_popup()
        if success:
            logger.info("Task completed successfully")
        else:
            logger.warning("Task completed with issues")
        return success
    except Exception as e:
        logger.error(f"Task failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_task()