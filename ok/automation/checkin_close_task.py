"""
Check-in screen close task for 天下布魔 (Tianxia Bumo).

Detects the check-in screen and clicks the bottom blank area
to dismiss it.

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
logger = logging.getLogger("checkin_close")

ADB_SERIAL = "127.0.0.1:16384"

CHECKIN_TITLE_TPL = "templates/checkin_title.png"
CHECKIN_MSG_TPL = "templates/checkin_message.png"

CHECKIN_THRESHOLD = 0.70
CLICK_THRESHOLD = 0.65

CLICK_POS = (0.50, 0.92)

CHECKIN_TITLE_ROI = (0.30, 0.04, 0.70, 0.15)
CHECKIN_MSG_ROI = (0.20, 0.65, 0.80, 0.80)


class CheckInCloser:
    """Detects and closes the check-in screen."""

    def __init__(self, serial: str = ADB_SERIAL):
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(serial)
        self.width = 1080
        self.height = 1920

        self.checkin_title_tpl = self._load_template(CHECKIN_TITLE_TPL)
        self.checkin_msg_tpl = self._load_template(CHECKIN_MSG_TPL)

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

    def find_checkin_screen(
        self, screen: np.ndarray
    ) -> Optional[Tuple[int, int, float]]:
        title_result = self._match_template(
            screen, self.checkin_title_tpl, CHECKIN_TITLE_ROI,
            CHECKIN_THRESHOLD, "checkin_title"
        )

        if title_result is not None:
            return title_result

        msg_result = self._match_template(
            screen, self.checkin_msg_tpl, CHECKIN_MSG_ROI,
            CHECKIN_THRESHOLD, "checkin_message"
        )
        return msg_result

    def click(self, x: int, y: int) -> None:
        self.device.shell(f"input tap {x} {y}", timeout=5)
        logger.info(f"Click at ({x}, {y})")

    def _jitter(self, base_x: int, base_y: int, r: int = 10) -> Tuple[int, int]:
        return (base_x + random.randint(-r, r), base_y + random.randint(-r, r))

    def _delay(self, base: float) -> float:
        v = random.uniform(-0.15 * base, 0.15 * base)
        return max(0.05, base + v)

    def close_checkin_screen(self) -> bool:
        """
        Detect check-in screen and click bottom blank area to close.

        Returns:
            True if the task completed successfully.
        """
        logger.info("=" * 50)
        logger.info("Starting close_checkin_screen task")
        logger.info("=" * 50)

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return False

        checkin_result = self.find_checkin_screen(screen)

        if checkin_result is None:
            logger.info("Check-in screen not detected - nothing to close")
            return True

        logger.info("Check-in screen detected - clicking bottom area...")

        click_x = int(self.width * CLICK_POS[0])
        click_y = int(self.height * CLICK_POS[1])

        jx, jy = self._jitter(click_x, click_y, r=15)
        logger.info(f"Clicking at ({jx}, {jy}) to close check-in screen")
        self.click(jx, jy)

        d = self._delay(0.5)
        logger.info(f"Waiting {d:.2f}s for screen to close...")
        time.sleep(d)

        try:
            screen2 = self.capture_screen()
            result2 = self.find_checkin_screen(screen2)
            if result2 is None:
                logger.info(
                    "Verification: check-in screen no longer visible "
                    "- screen closed successfully"
                )
            else:
                logger.warning(
                    "Verification: check-in screen still visible "
                    f"(conf={result2[2]:.3f})"
                )
        except Exception as e:
            logger.warning(f"Verification failed: {e}")

        logger.info("close_checkin_screen task completed")
        return True


def run_task() -> bool:
    try:
        closer = CheckInCloser()
        success = closer.close_checkin_screen()
        if success:
            logger.info("Task completed: check-in screen handled")
        else:
            logger.warning("Task completed with issues")
        return success
    except Exception as e:
        logger.error(f"Task failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    exit(0 if run_task() else 1)