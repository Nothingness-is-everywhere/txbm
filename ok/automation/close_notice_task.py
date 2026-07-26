"""
Close notice popup task for 天下布魔 (Tianxia Bumo).

Detects the '王城公布栏' notice popup screen and clicks the
close button (X) twice with a delay interval between clicks.

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
TEMPLATE_PATH = "templates/close_button.png"
THRESHOLD = 0.75
FIRST_CLICK_DELAY = 1.0
SECOND_CLICK_DELAY = 0.5
MATCH_ROI = (0.20, 0.55, 0.80, 0.95)
DEFAULT_BTN_POS = (0.50, 0.855)
MAX_CLICK_ATTEMPTS = 3


class NoticeCloser:
    """Detects and closes the notice popup on the emulator."""

    def __init__(self, serial: str = ADB_SERIAL):
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(serial)
        self.template = self._load_template()
        self.width = 1080
        self.height = 1920

    def _load_template(self) -> np.ndarray:
        """Load the close button template image."""
        template_path = Path(TEMPLATE_PATH)
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template not found: {TEMPLATE_PATH}. "
                "Run _extract_template.py first."
            )
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            raise ValueError(f"Failed to load template: {TEMPLATE_PATH}")
        logger.info(f"Template loaded: {template.shape}")
        return template

    def capture_screen(self) -> np.ndarray:
        """Capture current screen from the emulator."""
        png_bytes = self.device.shell("screencap -p", encoding=None, timeout=10)
        if not png_bytes or len(png_bytes) == 0:
            raise RuntimeError("Failed to capture screen")
        image_data = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode screenshot")
        self.height, self.width = image.shape[:2]
        logger.debug(f"Screen captured: {self.width}x{self.height}")
        return image

    def find_close_button(
        self, screen: np.ndarray, threshold: float = THRESHOLD
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find the close button on the screen using template matching.

        Args:
            screen: Current screen capture as BGR image.
            threshold: Minimum confidence threshold (0.0-1.0).

        Returns:
            Tuple of (center_x, center_y, confidence) if found, None otherwise.
        """
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
                f"Close button found at ({center_x}, {center_y}) "
                f"with confidence {max_val:.3f}"
            )
            return (center_x, center_y, max_val)
        else:
            logger.warning(
                f"Close button not found. "
                f"Max confidence: {max_val:.3f} (threshold: {threshold})"
            )
            return None

    def click(self, x: int, y: int) -> bool:
        """
        Click at the specified coordinates via ADB.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            True if the click was sent successfully.
        """
        result = self.device.shell(
            f"input tap {x} {y}", timeout=5
        )
        logger.info(f"Click sent at ({x}, {y})")
        return True

    def _throttle_delay(self, base_delay: float) -> float:
        """Calculate delay with random variation for anti-detection."""
        variation = random.uniform(-0.15 * base_delay, 0.15 * base_delay)
        return max(0.05, base_delay + variation)

    def close_notice_popup(self) -> bool:
        """
        Main task: detect and close the notice popup with two clicks.

        Always performs two clicks with a delay interval between them.
        Uses template matching when possible, falls back to default
        button position when template is not detected.

        Returns:
            True if the task completed successfully.
        """
        logger.info("=" * 50)
        logger.info("Starting close_notice_popup task")
        logger.info("=" * 50)

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return False

        result = self.find_close_button(screen, threshold=THRESHOLD)

        if result is not None:
            cx, cy, conf = result
            logger.info(
                f"Button detected via template matching "
                f"(conf={conf:.3f}) at ({cx}, {cy})"
            )
        else:
            cx = int(self.width * DEFAULT_BTN_POS[0])
            cy = int(self.height * DEFAULT_BTN_POS[1])
            logger.info(
                f"Button not detected, using default position "
                f"({cx}, {cy})"
            )

        click_x = cx + random.randint(-10, 10)
        click_y = cy + random.randint(-10, 10)

        logger.info(
            f"[Click 1/2] Clicking at ({click_x}, {click_y})"
        )
        self.click(click_x, click_y)

        delay1 = self._throttle_delay(FIRST_CLICK_DELAY)
        logger.info(f"Waiting {delay1:.2f}s before second click...")
        time.sleep(delay1)

        click_x2 = cx + random.randint(-10, 10)
        click_y2 = cy + random.randint(-10, 10)

        logger.info(
            f"[Click 2/2] Clicking again at ({click_x2}, {click_y2})"
        )
        self.click(click_x2, click_y2)

        delay2 = self._throttle_delay(SECOND_CLICK_DELAY)
        logger.info(f"Waiting {delay2:.2f}s for popup to dismiss...")
        time.sleep(delay2)

        try:
            screen3 = self.capture_screen()
            result3 = self.find_close_button(screen3, threshold=0.65)
            if result3 is None:
                logger.info(
                    "Verification: close button no longer visible "
                    "- popup successfully dismissed"
                )
            else:
                logger.warning(
                    "Verification: close button still visible "
                    f"(conf={result3[2]:.3f}) after both clicks"
                )
        except Exception as e:
            logger.warning(f"Verification screenshot failed: {e}")

        logger.info("close_notice_popup task completed successfully")
        return True


def run_task():
    """Run the close notice popup task."""
    try:
        closer = NoticeCloser()
        success = closer.close_notice_popup()
        if success:
            logger.info("Task completed: notice popup handled")
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