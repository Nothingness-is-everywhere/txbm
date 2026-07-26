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
from typing import Optional, Tuple, List

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
    ) -> Optional[Tuple[int, int, float]]:
        return self._match_template(
            screen, self.post_close_tpl, POST_CLOSE_ROI, POST_CLOSE_THRESHOLD,
            "post_close"
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
        Two-phase notice handling:
          1. Detect notice popup → click close button
          2. Detect post-close screen → click once

        Returns:
            True if both phases completed successfully.
        """
        logger.info("=" * 50)
        logger.info("Phase 1: Detect & Close Notice Popup")
        logger.info("=" * 50)

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return False

        notice_result = self.find_notice_popup(screen)

        if notice_result is None:
            logger.info(
                "No notice popup detected - "
                "skipping Phase 1 (popup not visible)"
            )
        else:
            logger.info("Notice popup detected - searching for close button...")
            close_result = self.find_close_button(screen)

            if close_result is not None:
                cx, cy, _ = close_result
            else:
                cx = int(self.width * DEFAULT_CLOSE_POS[0])
                cy = int(self.height * DEFAULT_CLOSE_POS[1])
                logger.info(
                    "Close button not found via template, "
                    f"using default position ({cx}, {cy})"
                )

            jx, jy = self._jitter(cx, cy, r=12)
            logger.info(f"Clicking close button at ({jx}, {jy})")
            self.click(jx, jy)

            d1 = self._delay(PHASE1_DELAY)
            logger.info(f"Waiting {d1:.2f}s for popup to dismiss...")
            time.sleep(d1)

            logger.info("Phase 1 complete - close button clicked")

        logger.info("=" * 50)
        logger.info("Phase 2: Detect Post-Close Screen & Click")
        logger.info("=" * 50)

        try:
            screen2 = self.capture_screen()
        except Exception as e:
            logger.warning(f"Phase 2 screen capture failed: {e}")
            screen2 = None

        if screen2 is not None:
            post_result = self.find_post_close(screen2)

            if post_result is not None:
                px, py, conf = post_result
                jx2, jy2 = self._jitter(px, py, r=10)
                logger.info(
                    f"Post-close screen detected (conf={conf:.3f}) - "
                    f"clicking at ({jx2}, {jy2})"
                )
                self.click(jx2, jy2)

                d2 = self._delay(PHASE2_DELAY)
                logger.info(f"Waiting {d2:.2f}s...")
                time.sleep(d2)

                logger.info("Phase 2 complete - post-close click done")
            else:
                logger.info(
                    "Post-close screen not detected - "
                    "skipping Phase 2"
                )
        else:
            logger.info("No screen available for Phase 2 - skipping")

        logger.info("=" * 50)
        logger.info("close_notice_popup task finished")
        logger.info("=" * 50)
        return True


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