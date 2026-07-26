"""
Stamina reader module for 天下布魔 (Tianxia Bumo).

Reads stamina/energy values from the main interface using:
  1. Template matching to locate stamina bar regions
  2. Digit extraction via contour analysis + template matching
  3. EasyOCR as enhanced option (when available)

Usage:
  from ok.automation.stamina_reader import StaminaReader, get_stamina
  reader = StaminaReader()
  values = reader.read_all()
  stamina1_current = values.stamina1.current
  stamina1_max = values.stamina1.max
"""

import time
import re
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import adbutils
import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("stamina_reader")

ADB_SERIAL = "127.0.0.1:16384"

STAMINA1_TPL = "templates/stamina1_bar.png"
STAMINA2_TPL = "templates/stamina2_bar.png"

STAMINA_THRESHOLD = 0.65

STAMINA1_ROI = (0.65, 0.62, 0.95, 0.72)
STAMINA2_ROI = (0.70, 0.74, 0.95, 0.83)

STAMINA1_NUMBERS_OFFSET = (0.02, 0.10, 0.85, 0.45)
STAMINA2_NUMBERS_OFFSET = (0.05, 0.15, 0.80, 0.45)

DIGIT_TEMPLATES_DIR = "templates/digit_templates"

DIGIT_SIZE = (25, 40)


@dataclass
class StaminaValue:
    """Represents a stamina/energy value."""
    current: int = 0
    max: int = 0
    name: str = ""

    @property
    def ratio(self) -> float:
        """Get current/max ratio."""
        if self.max == 0:
            return 0.0
        return self.current / self.max

    @property
    def is_full(self) -> bool:
        """Check if stamina is full."""
        return self.current >= self.max

    @property
    def is_empty(self) -> bool:
        """Check if stamina is empty."""
        return self.current <= 0

    def __str__(self) -> str:
        return f"{self.current}/{self.max}"


@dataclass
class StaminaState:
    """Container for all stamina values."""
    stamina1: StaminaValue = field(default_factory=lambda: StaminaValue(name="stamina1"))
    stamina2: StaminaValue = field(default_factory=lambda: StaminaValue(name="stamina2"))
    last_update: float = 0.0

    def to_dict(self) -> Dict[str, StaminaValue]:
        return {
            "stamina1": self.stamina1,
            "stamina2": self.stamina2,
        }

    def summary(self) -> str:
        lines = []
        lines.append(f"Stamina 1: {self.stamina1} ({self.stamina1.name})")
        lines.append(f"Stamina 2: {self.stamina2} ({self.stamina2.name})")
        return "\n".join(lines)


class StaminaReader:
    """Reads stamina values from the game's main interface."""

    _instance: Optional['StaminaReader'] = None
    _cached_state: Optional[StaminaState] = None
    _cache_time: float = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, serial: str = ADB_SERIAL):
        if self._initialized:
            return
        self._initialized = True
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(serial)
        self.width = 1080
        self.height = 1920

        self.stamina1_tpl = self._load_template(STAMINA1_TPL)
        self.stamina2_tpl = self._load_template(STAMINA2_TPL)

        self._digit_templates: Dict[str, np.ndarray] = {}
        self._load_digit_templates()

        self._ocr_reader = None
        self._ocr_initialized = False
        self._ocr_available = False

    def _load_digit_templates(self):
        """Load digit templates from templates/digit_templates directory."""
        templates_path = Path(DIGIT_TEMPLATES_DIR)
        if not templates_path.exists():
            logger.info(f"Digit templates directory not found: {DIGIT_TEMPLATES_DIR}")
            return

        count = 0
        for img_file in templates_path.glob("*.png"):
            name = img_file.stem
            img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img is not None:
                self._digit_templates[name] = img
                count += 1

        if count > 0:
            logger.info(f"Loaded {count} digit templates from {DIGIT_TEMPLATES_DIR}")

    def _extract_digit_regions(
        self, number_region: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """Extract individual digit regions from a number display area."""
        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        digit_regions = []
        for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if ch >= 25 and 8 <= cw <= 45 and area >= 100:
                aspect = cw / ch
                if 0.15 <= aspect <= 1.2:
                    pad = 2
                    dx1 = max(0, x - pad)
                    dy1 = max(0, y - pad)
                    dx2 = min(number_region.shape[1], x + cw + pad)
                    dy2 = min(number_region.shape[0], y + ch + pad)

                    digit_img = number_region[dy1:dy2, dx1:dx2]
                    digit_regions.append((x, y, cw, ch, digit_img))

        return digit_regions

    def _recognize_digit_via_templates(
        self, digit_img: np.ndarray
    ) -> Optional[str]:
        """Recognize a digit using template matching."""
        if not self._digit_templates:
            return None

        digit_gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_score = 0

        for name, tpl in self._digit_templates.items():
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

            th, tw = tpl_gray.shape[:2]
            digit_resized = cv2.resize(digit_gray, (tw, th))

            result = cv2.matchTemplate(
                digit_resized, tpl_gray, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = name

        if best_score >= 0.5:
            logger.debug(
                f"Digit matched as '{best_match}' "
                f"(score: {best_score:.3f})"
            )
            return best_match

        return None

    def _read_numbers_via_templates(
        self, number_region: np.ndarray
    ) -> Optional[str]:
        """Read numbers using template-based digit recognition."""
        if not self._digit_templates:
            return None

        digit_regions = self._extract_digit_regions(number_region)

        if len(digit_regions) < 2:
            logger.debug(
                f"Not enough digit regions found: {len(digit_regions)}"
            )
            return None

        recognized_parts = []
        prev_x_end = None

        for i, (x, y, cw, ch, digit_img) in enumerate(digit_regions):
            if i > 0 and prev_x_end is not None:
                gap = x - prev_x_end
                avg_width = (digit_regions[i - 1][2] + cw) / 2
                if gap > avg_width * 0.8:
                    recognized_parts.append('/')

            digit_name = self._recognize_digit_via_templates(digit_img)

            if digit_name is not None:
                digit_value = self._map_template_to_digit(digit_name)
                recognized_parts.append(digit_value)
            else:
                recognized_parts.append('?')

            prev_x_end = x + cw

        result = ''.join(recognized_parts)
        logger.debug(f"Template-based recognition: '{result}'")
        return result

    def _map_template_to_digit(self, template_name: str) -> str:
        """Map a template name to its digit value."""
        name_lower = template_name.lower()

        mapping = {
            'd0': '0', 'd1': '1', 'd2': '2', 'd3': '3', 'd4': '4',
            'd5': '5', 'd6': '6', 'd7': '7', 'd8': '8', 'd9': '9',
        }

        for key, value in mapping.items():
            if key in name_lower:
                return value

        digit_pattern = re.search(r'd(\d+)', name_lower)
        if digit_pattern:
            return digit_pattern.group(1)

        return '?'

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

    def _init_ocr(self):
        """Initialize EasyOCR reader lazily."""
        if self._ocr_initialized:
            return

        self._ocr_initialized = True

        try:
            import easyocr
            logger.info("Initializing EasyOCR reader...")
            self._ocr_reader = easyocr.Reader(['en'], gpu=False)
            self._ocr_available = True
            logger.info("EasyOCR reader initialized successfully")
        except ImportError:
            logger.info("EasyOCR not installed, using fallback recognition")
            self._ocr_available = False
        except Exception as e:
            logger.warning(f"EasyOCR init failed: {e}")
            self._ocr_available = False

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
    ) -> Optional[Tuple[int, int, int, int]]:
        """Match template and return bounding box (x, y, w, h)."""
        img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        rx1, ry1, rx2, ry2 = roi
        roi_x1 = int(self.width * rx1)
        roi_y1 = int(self.height * ry1)
        roi_x2 = int(self.width * rx2)
        roi_y2 = int(self.height * ry2)

        roi_region = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi_region.shape[0] < tpl_gray.shape[0] or roi_region.shape[1] < tpl_gray.shape[1]:
            logger.warning(f"ROI too small for template matching: {label}")
            return None

        result = cv2.matchTemplate(roi_region, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            tpl_h, tpl_w = tpl_gray.shape
            abs_x = roi_x1 + max_loc[0]
            abs_y = roi_y1 + max_loc[1]
            logger.info(
                f"[{label}] Found at ({abs_x}, {abs_y}) "
                f"size={tpl_w}x{tpl_h} conf={max_val:.3f}"
            )
            return (abs_x, abs_y, tpl_w, tpl_h)
        else:
            logger.info(
                f"[{label}] Not found. "
                f"Max conf={max_val:.3f} (threshold={threshold})"
            )
            return None

    def _read_numbers_ocr(
        self, roi: np.ndarray
    ) -> Optional[str]:
        """Read numbers using EasyOCR."""
        self._init_ocr()

        if not self._ocr_available or self._ocr_reader is None:
            return None

        try:
            result = self._ocr_reader.readtext(roi, detail=0)
            if result:
                text = " ".join(result)
                logger.debug(f"OCR result: '{text}'")
                return text
            return None
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            return None

    def _read_numbers_fallback(
        self, roi: np.ndarray
    ) -> Optional[str]:
        """Fallback: count digit regions as a simple heuristic."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        digit_count = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 20 and w > 8:
                digit_count += 1

        if digit_count == 0:
            return None

        logger.debug(f"Fallback recognition: found {digit_count} regions")
        return f"digits:{digit_count}"

    def _read_numbers_from_region(
        self, image: np.ndarray, region: Tuple[int, int, int, int]
    ) -> Optional[str]:
        """
        Read numbers from a region using template matching, then OCR as fallback.

        Args:
            image: Full screen image
            region: (x, y, w, h) bounding box

        Returns:
            Recognized text string or None
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return None

        if self._digit_templates:
            template_result = self._read_numbers_via_templates(roi)
            if template_result is not None:
                return template_result

        ocr_result = self._read_numbers_ocr(roi)
        if ocr_result is not None:
            return ocr_result

        fallback_result = self._read_numbers_fallback(roi)
        return fallback_result

    def _parse_stamina_text(
        self, text: Optional[str]
    ) -> Tuple[int, int]:
        """
        Parse stamina text in format "current/max".

        Args:
            text: Recognition result string

        Returns:
            Tuple of (current, max)
        """
        if not text:
            return (0, 0)

        text = text.strip()

        numbers = re.findall(r'\d+', text)

        if len(numbers) >= 2:
            current = int(numbers[0])
            max_val = int(numbers[1])
            logger.debug(f"Parsed stamina: {current}/{max_val}")
            return (current, max_val)
        elif len(numbers) == 1:
            current = int(numbers[0])
            logger.debug(f"Parsed stamina (only current): {current}")
            return (current, 0)
        else:
            logger.warning(f"Could not parse stamina from: '{text}'")
            return (0, 0)

    def read_stamina1(self, screen: np.ndarray) -> StaminaValue:
        """Read stamina 1 value from screen."""
        result = self._match_template(
            screen, self.stamina1_tpl, STAMINA1_ROI,
            STAMINA_THRESHOLD, "stamina1"
        )

        if result is None:
            logger.warning("Could not find stamina1 bar")
            return StaminaValue(name="stamina1")

        x, y, w, h = result
        ox1, oy1, ow, oh = STAMINA1_NUMBERS_OFFSET
        number_x = x + int(w * ox1)
        number_y = y + int(h * oy1)
        number_w = int(w * ow)
        number_h = int(h * oh)

        number_region = (number_x, number_y, number_w, number_h)

        text = self._read_numbers_from_region(screen, number_region)
        current, max_val = self._parse_stamina_text(text)

        return StaminaValue(
            current=current,
            max=max_val,
            name="stamina1"
        )

    def read_stamina2(self, screen: np.ndarray) -> StaminaValue:
        """Read stamina 2 value from screen."""
        result = self._match_template(
            screen, self.stamina2_tpl, STAMINA2_ROI,
            STAMINA_THRESHOLD, "stamina2"
        )

        if result is None:
            logger.warning("Could not find stamina2 bar")
            return StaminaValue(name="stamina2")

        x, y, w, h = result
        ox1, oy1, ow, oh = STAMINA2_NUMBERS_OFFSET
        number_x = x + int(w * ox1)
        number_y = y + int(h * oy1)
        number_w = int(w * ow)
        number_h = int(h * oh)

        number_region = (number_x, number_y, number_w, number_h)

        text = self._read_numbers_from_region(screen, number_region)
        current, max_val = self._parse_stamina_text(text)

        return StaminaValue(
            current=current,
            max=max_val,
            name="stamina2"
        )

    def read_all(
        self, force_refresh: bool = False
    ) -> StaminaState:
        """
        Read all stamina values from the current screen.

        Args:
            force_refresh: If True, re-read from device even if cached

        Returns:
            StaminaState with current values
        """
        cache_duration = 5.0

        if (
            not force_refresh
            and self._cached_state is not None
            and (time.time() - self._cache_time) < cache_duration
        ):
            logger.debug("Returning cached stamina state")
            return self._cached_state

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return self._cached_state or StaminaState()

        stamina1 = self.read_stamina1(screen)
        stamina2 = self.read_stamina2(screen)

        state = StaminaState(
            stamina1=stamina1,
            stamina2=stamina2,
            last_update=time.time(),
        )

        self._cached_state = state
        self._cache_time = time.time()

        logger.info(state.summary())
        return state

    @classmethod
    def get_state(
        cls, force_refresh: bool = False
    ) -> StaminaState:
        """
        Get current stamina state (convenience method).

        Args:
            force_refresh: If True, force re-read from device

        Returns:
            Current StaminaState
        """
        reader = cls()
        return reader.read_all(force_refresh=force_refresh)


def get_stamina() -> StaminaState:
    """
    Get current stamina values (global accessor).

    This is the recommended way to access stamina values
    from any part of the application.

    Returns:
        StaminaState with current stamina values
    """
    return StaminaReader.get_state()


if __name__ == "__main__":
    state = get_stamina()
    print(state.summary())
    print(f"\nStamina 1 ratio: {state.stamina1.ratio:.1%}")
    print(f"Stamina 2 ratio: {state.stamina2.ratio:.1%}")