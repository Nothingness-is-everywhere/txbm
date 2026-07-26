"""
Back button navigation task for 天下布魔 (Tianxia Bumo).

Detects the back button (←) in the top-left corner of the screen
and clicks it to navigate back to the previous screen.

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
logger = logging.getLogger("back_nav")

ADB_SERIAL = "127.0.0.1:16384"
TEMPLATE_PATH = "templates/back_button.png"
THRESHOLD = 0.75
CLICK_DELAY = 0.3
MATCH_ROI = (0.0, 0.0, 0.20, 0.15)
DEFAULT_BTN_POS = (0.065, 0.068)


class BackButtonNavigator:
    """Detects and clicks the back button to navigate backwards."""

    def __init__(self, serial: str = ADB_SERIAL):
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(serial)
        self.template = self._load_template()
        self.width = 1080
        self.height = 1920

    def _load_template(self) -> np.ndarray:
        template_path = Path(TEMPLATE_PATH)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {TEMPLATE_PATH}"
            )
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"Failed to load template: {TEMPLATE_PATH}")
        logger.info(f"Back button template loaded: {template.shape}")
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

    def find_back_button(
        self, screen: np.ndarray, threshold: float = THRESHOLD
    ) -> Optional[Tuple[int, int, float]]:
        img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        rx1, ry1, rx2, ry2 = MATCH_ROI
        roi_x1 = int(self.width * rx1)
        roi_y1 = int(self.height * ry1)
        roi_x2 = int(self.width * rx2)
        roi_y2 = int(self.height * ry2)

        roi = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]

        result = cv2.matchTemplate(roi, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            tpl_h, tpl_w = tpl_gray.shape
            center_x = roi_x1 + max_loc[0] + tpl_w // 2
            center_y = roi_y1 + max_loc[1] + tpl_h // 2
            logger.info(
                f"Back button found at ({center_x}, {center_y}) "
                f"with confidence {max_val:.3f}"
            )
            return (center_x, center_y, max_val)
        else:
            logger.warning(
                f"Back button not found. "
                f"Max confidence: {max_val:.3f}"
            )
            return None

    def click(self, x: int, y: int) -> bool:
        result = self.device.shell(f"input tap {x} {y}", timeout=5)
        logger.info(f"Click sent at ({x}, {y})")
        return True

    def _throttle_delay(self, base_delay: float) -> float:
        variation = random.uniform(-0.15 * base_delay, 0.15 * base_delay)
        return max(0.05, base_delay + variation)

    def navigate_back(self) -> bool:
        """
        Detect and click the back button to navigate backwards.

        Returns:
            True if the back button was clicked successfully.
        """
        logger.info("=" * 50)
        logger.info("Starting navigate_back task")
        logger.info("=" * 50)

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return False

        result = self.find_back_button(screen, threshold=THRESHOLD)

        if result is not None:
            cx, cy, conf = result
            logger.info(
                f"Back button detected (conf={conf:.3f}) at ({cx}, {cy})"
            )
        else:
            cx = int(self.width * DEFAULT_BTN_POS[0])
            cy = int(self.height * DEFAULT_BTN_POS[1])
            logger.info(
                f"Back button not detected, using default position "
                f"({cx}, {cy})"
            )

        click_x = cx + random.randint(-8, 8)
        click_y = cy + random.randint(-8, 8)

        logger.info(f"Clicking back button at ({click_x}, {click_y})")
        self.click(click_x, click_y)

        delay = self._throttle_delay(CLICK_DELAY)
        logger.info(f"Waiting {delay:.2f}s for navigation...")
        time.sleep(delay)

        try:
            screen2 = self.capture_screen()
            result2 = self.find_back_button(screen2, threshold=0.65)
            if result2 is not None:
                logger.info(
                    "Back button still visible after click "
                    "- navigation may still be in progress"
                )
            else:
                logger.info(
                    "Back button no longer visible "
                    "- navigation completed"
                )
        except Exception as e:
            logger.warning(f"Verification screenshot failed: {e}")

        logger.info("navigate_back task completed successfully")
        return True


def run_task():
    try:
        navigator = BackButtonNavigator()
        success = navigator.navigate_back()
        if success:
            logger.info("Task completed: back navigation handled")
        else:
            logger.warning("Task completed with issues")
        return success
    except Exception as e:
        logger.error(f"Task failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_task()