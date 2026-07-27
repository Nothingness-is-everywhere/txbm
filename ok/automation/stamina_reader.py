"""
Stamina reader module for automatic game stamina value recognition.

Provides two public stamina variables that are automatically updated
by reading the game's home screen:

    EXPEDITION_STAMINA   - 当前远征体力值 (current expedition stamina)
    TRAINING_STAMINA     - 当前训练体力值 (current training stamina)
    EXPEDITION_STAMINA_VALUE - 远征体力完整值对象 (StaminaValue)
    TRAINING_STAMINA_VALUE   - 训练体力完整值对象 (StaminaValue)

Usage:
    from ok.automation.stamina_reader import (
        EXPEDITION_STAMINA, TRAINING_STAMINA,
        EXPEDITION_STAMINA_VALUE, TRAINING_STAMINA_VALUE,
        update_global_stamina, get_stamina,
    )

    update_global_stamina()  # 从屏幕读取并更新所有体力值
    print(EXPEDITION_STAMINA)  # 输出: 60
    print(TRAINING_STAMINA)    # 输出: 25
"""

from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stamina_reader")

STAMINA_ROI_CONFIG = {
    "expedition": {
        "x_start": 0.738,
        "x_end": 0.850,
        "y_start": 0.728,
        "y_end": 0.762,
    },
    "training": {
        "x_start": 0.800,
        "x_end": 0.920,
        "y_start": 0.822,
        "y_end": 0.862,
    },
}

ADB_SERIAL = "emulator-5554"

_ENGINE = None
_stamina_cache: Dict[str, Tuple[int, int, float]] = {}
_cache_ttl = 5.0
_last_update_time = 0.0


@dataclass
class StaminaValue:
    """Stamina value with current/max and derived properties."""

    current: int = 0
    max_val: int = 0
    source: str = "unknown"

    @property
    def ratio(self) -> float:
        if self.max_val <= 0:
            return 0.0
        return float(self.current) / float(self.max_val)

    @property
    def is_full(self) -> bool:
        return self.current >= self.max_val > 0

    @property
    def is_empty(self) -> bool:
        return self.current <= 0

    def __str__(self) -> str:
        return f"{self.current}/{self.max_val}"

    def __repr__(self) -> str:
        return f"StaminaValue(current={self.current}, max_val={self.max_val})"


class StaminaState:
    """Enumeration of stamina states."""

    UNKNOWN = "unknown"
    NORMAL = "normal"
    FULL = "full"
    EMPTY = "empty"
    LOW = "low"

    def __init__(self, state: str = "unknown"):
        self._state = state

    @property
    def state(self) -> str:
        return self._state

    def __str__(self) -> str:
        return self._state

    def __repr__(self) -> str:
        return f"StaminaState({self._state})"


def save_image(img, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ext = filepath.suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(str(filepath), 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False


def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)


class DigitClassifier:
    """Feature-based digit classifier for game UI numbers."""

    def __init__(self):
        self.template_dir = Path("templates/digits")
        self.templates: Dict[str, List[np.ndarray]] = {}
        self._load_templates()

    def _load_templates(self):
        """Load digit templates."""
        self.templates = {}
        if not self.template_dir.exists():
            return

        for template_path in self.template_dir.glob("digit_*.png"):
            digit = template_path.stem.replace("digit_", "")
            img = read_image(str(template_path))
            if img is not None:
                if digit not in self.templates:
                    self.templates[digit] = []
                self.templates[digit].append(img)

    def extract_features(self, digit_img: np.ndarray) -> np.ndarray:
        """Extract features from a digit image."""
        h, w = digit_img.shape[:2]

        gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        num_contours = len(contours)

        has_hole = False
        num_holes = 0
        if hierarchy is not None and len(hierarchy) > 0:
            for hier in hierarchy[0]:
                if hier[2] != -1:
                    has_hole = True
                    num_holes += 1

        aspect_ratio = w / h if h > 0 else 1.0

        upper_half = binary[:h//2, :]
        lower_half = binary[h//2:, :]
        left_half = binary[:, :w//2]
        right_half = binary[:, w//2:]

        total_pixels = binary.size + 1
        upper_ratio = np.sum(upper_half > 0) / total_pixels * 2
        lower_ratio = np.sum(lower_half > 0) / total_pixels * 2
        left_ratio = np.sum(left_half > 0) / total_pixels * 2
        right_ratio = np.sum(right_half > 0) / total_pixels * 2

        x, y, bw, bh = cv2.boundingRect(binary)
        fill_ratio = (bw * bh) / (binary.size + 1)

        features = np.array([
            num_contours / 10.0,
            num_holes / 5.0,
            float(aspect_ratio),
            float(upper_ratio),
            float(lower_ratio),
            float(left_ratio),
            float(right_ratio),
            float(fill_ratio),
        ])

        return features

    def classify(self, digit_img: np.ndarray) -> Tuple[str, float]:
        """Classify a digit using template matching and feature analysis."""
        h, w = digit_img.shape[:2]

        if h < 10 or w < 3:
            return "?", 0.0

        if self.templates:
            best_match, best_score = self._template_match(digit_img)
            if best_score > 0.7:
                return best_match, best_score

        features = self.extract_features(digit_img)
        digit, confidence = self._rule_based_classify(features)
        return digit, confidence

    def _template_match(self, digit_img: np.ndarray) -> Tuple[str, float]:
        """Match digit against templates."""
        best_match = "?"
        best_score = 0.0

        digit_gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        for digit, templates in self.templates.items():
            for template in templates:
                tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                if tpl_gray.shape[0] < 10:
                    continue

                digit_resized = cv2.resize(digit_gray, (30, 50))
                tpl_resized = cv2.resize(tpl_gray, (30, 50))

                result = cv2.matchTemplate(
                    digit_resized, tpl_resized, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_score:
                    best_score = max_val
                    best_match = digit

        return best_match, best_score

    def _rule_based_classify(self, features: np.ndarray) -> Tuple[str, float]:
        """Rule-based classification using extracted features."""
        num_contours = features[0] * 10
        num_holes = features[1] * 5
        aspect_ratio = features[2]
        upper_ratio = features[3]
        lower_ratio = features[4]
        left_ratio = features[5]
        right_ratio = features[6]
        fill_ratio = features[7]

        if num_holes >= 2 and aspect_ratio > 0.5:
            return "8", 0.85

        if num_holes == 1:
            if aspect_ratio > 0.65:
                return "0", 0.85
            elif upper_ratio > lower_ratio * 1.3 and aspect_ratio < 0.5:
                return "9", 0.75
            elif lower_ratio > upper_ratio * 1.2:
                return "6", 0.75
            else:
                return "4", 0.60

        if num_holes == 0:
            if aspect_ratio < 0.3:
                return "1", 0.90

            if upper_ratio > lower_ratio * 1.5 and right_ratio > left_ratio:
                return "7", 0.70

            if abs(upper_ratio - lower_ratio) < 0.15:
                if aspect_ratio > 0.6:
                    return "3", 0.65
                else:
                    return "5", 0.65

            if upper_ratio > lower_ratio and left_ratio > right_ratio:
                return "2", 0.65

            if fill_ratio > 0.5 and aspect_ratio > 0.7:
                return "0", 0.50

        return "?", 0.30


class StaminaReader:
    """Reads stamina values from the game main screen."""

    def __init__(self, adb_serial: str = ADB_SERIAL):
        self.adb_serial = adb_serial
        self.classifier = DigitClassifier()
        self._easyocr_reader = None
        self._init_easyocr()

    def _init_easyocr(self):
        """Initialize EasyOCR if available."""
        global _ENGINE
        if _ENGINE is not None and _ENGINE != "template_only":
            self._easyocr_reader = _ENGINE
            return

        if _ENGINE == "template_only":
            self._easyocr_reader = None
            return

        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            _ENGINE = self._easyocr_reader
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.info(f"EasyOCR not available: {e}")
            logger.info("Using feature-based recognition")
            self._easyocr_reader = None
            _ENGINE = "template_only"

    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screen via ADB."""
        try:
            import adbutils
            adb = adbutils.AdbClient()
            device = adb.device(self.adb_serial)
            img = device.screenshot()

            img_array = np.array(img)
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:
                    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return img_array
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None

    def _crop_region(
        self,
        img: np.ndarray,
        x_ratio_start: float,
        x_ratio_end: float,
        y_ratio_start: float,
        y_ratio_end: float,
    ) -> np.ndarray:
        """Crop a region from the image using relative coordinates."""
        h, w = img.shape[:2]
        x1 = int(w * x_ratio_start)
        x2 = int(w * x_ratio_end)
        y1 = int(h * y_ratio_start)
        y2 = int(h * y_ratio_end)
        return img[y1:y2, x1:x2]

    def _segment_digits(self, image: np.ndarray) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """Segment individual digits from a stamina value image."""
        h, w = image.shape[:2]
        area = h * w

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 100, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        digit_regions = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cnt_area = bw * bh

            if cnt_area > area * 0.25:
                continue

            if bh < 12 or bw < 3:
                continue

            aspect = bw / bh
            if aspect > 3.5 or aspect < 0.15:
                continue

            digit_roi = image[y:y+bh, x:x+bw]
            digit_regions.append((x, y, bw, bh, digit_roi))

        digit_regions.sort(key=lambda r: (r[0], r[1]))

        return digit_regions

    def _recognize_stamina_text(self, region: np.ndarray) -> Optional[str]:
        """Recognize stamina text using available methods."""
        if self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(region)
                if results:
                    text = " ".join([r[1] for r in results])
                    logger.debug(f"EasyOCR result: '{text}'")
                    return text
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")

        digits = self._segment_digits(region)

        if not digits:
            return None

        recognized_digits = []
        for x, y, w, h, roi in digits:
            digit, confidence = self.classifier.classify(roi)
            recognized_digits.append(digit)

        text = "".join(recognized_digits)
        logger.debug(f"Feature-based result: '{text}'")
        return text

    def _parse_stamina_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse stamina text in 'current/max' format."""
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            try:
                current = int(numbers[0])
                max_val = int(numbers[1])
                return (current, max_val)
            except (ValueError, IndexError):
                pass
        elif len(numbers) == 1:
            try:
                val = int(numbers[0])
                return (val, val)
            except ValueError:
                pass
        return None

    def read_stamina(
        self, stamina_type: str = "expedition", force_refresh: bool = False
    ) -> Optional[StaminaValue]:
        """Read stamina value from the game screen."""
        global _last_update_time

        if not force_refresh and stamina_type in _stamina_cache:
            current, max_val, timestamp = _stamina_cache[stamina_type]
            if time.time() - timestamp < _cache_ttl:
                return StaminaValue(current=current, max_val=max_val, source="cache")

        if stamina_type not in STAMINA_ROI_CONFIG:
            logger.error(f"Unknown stamina type: {stamina_type}")
            return None

        config = STAMINA_ROI_CONFIG[stamina_type]
        img = self._capture_screen()
        if img is None:
            logger.error("Failed to capture screen")
            return None

        region = self._crop_region(
            img,
            config["x_start"],
            config["x_end"],
            config["y_start"],
            config["y_end"],
        )

        text = self._recognize_stamina_text(region)
        if text is None:
            logger.warning(f"Failed to recognize {stamina_type} stamina")
            return None

        parsed = self._parse_stamina_text(text)
        if parsed is None:
            logger.warning(f"Failed to parse stamina text: '{text}'")
            return None

        current, max_val = parsed
        _stamina_cache[stamina_type] = (current, max_val, time.time())
        _last_update_time = time.time()

        logger.info(
            f"{stamina_type} stamina: {current}/{max_val} "
            f"(ratio: {float(current)/max_val:.2f})"
        )

        return StaminaValue(
            current=current, max_val=max_val, source="recognition"
        )

    def read_all_stamina(
        self, force_refresh: bool = False
    ) -> Dict[str, Optional[StaminaValue]]:
        """Read all stamina values from the game screen."""
        result = {}
        for stamina_type in STAMINA_ROI_CONFIG:
            result[stamina_type] = self.read_stamina(
                stamina_type, force_refresh=force_refresh
            )
        return result


_expedition_stamina: Optional[StaminaValue] = None
_training_stamina: Optional[StaminaValue] = None
_reader_instance: Optional[StaminaReader] = None


def init_stamina(force_refresh: bool = False) -> Dict[str, Optional[StaminaValue]]:
    """Initialize stamina readings."""
    global _expedition_stamina, _training_stamina, _reader_instance

    if _reader_instance is None:
        _reader_instance = StaminaReader()

    values = _reader_instance.read_all_stamina(force_refresh=force_refresh)

    _expedition_stamina = values.get("expedition")
    _training_stamina = values.get("training")

    return values


def get_stamina(
    stamina_type: str = "expedition", force_refresh: bool = False
) -> Optional[StaminaValue]:
    """Get stamina value for the specified type."""
    global _expedition_stamina, _training_stamina, _reader_instance

    if _reader_instance is None:
        _reader_instance = StaminaReader()

    if stamina_type == "expedition":
        if force_refresh or _expedition_stamina is None:
            _expedition_stamina = _reader_instance.read_stamina(
                "expedition", force_refresh=True
            )
        return _expedition_stamina
    elif stamina_type == "training":
        if force_refresh or _training_stamina is None:
            _training_stamina = _reader_instance.read_stamina(
                "training", force_refresh=True
            )
        return _training_stamina
    else:
        logger.error(f"Unknown stamina type: {stamina_type}")
        return None


EXPEDITION_STAMINA_MAX: int = 150
TRAINING_STAMINA_MAX: int = 50

EXPEDITION_STAMINA: int = 0
TRAINING_STAMINA: int = 0

EXPEDITION_STAMINA_VALUE: StaminaValue = StaminaValue(current=0, max_val=EXPEDITION_STAMINA_MAX)
TRAINING_STAMINA_VALUE: StaminaValue = StaminaValue(current=0, max_val=TRAINING_STAMINA_MAX)

__all__ = [
    "STAMINA_ROI_CONFIG",
    "ADB_SERIAL",
    "StaminaValue",
    "StaminaState",
    "DigitClassifier",
    "StaminaReader",
    "init_stamina",
    "get_stamina",
    "update_global_stamina",
    "EXPEDITION_STAMINA",
    "TRAINING_STAMINA",
    "EXPEDITION_STAMINA_MAX",
    "TRAINING_STAMINA_MAX",
    "EXPEDITION_STAMINA_VALUE",
    "TRAINING_STAMINA_VALUE",
]


def update_global_stamina(force_refresh: bool = False) -> Dict[str, Optional[StaminaValue]]:
    """从屏幕读取并更新所有公共体力值变量。

    Args:
        force_refresh: 是否强制刷新（忽略缓存）

    Returns:
        包含所有体力值的字典
    """
    global EXPEDITION_STAMINA, TRAINING_STAMINA
    global EXPEDITION_STAMINA_VALUE, TRAINING_STAMINA_VALUE

    values = init_stamina(force_refresh=force_refresh)

    exp = values.get("expedition")
    if exp is not None:
        EXPEDITION_STAMINA_VALUE.current = exp.current
        EXPEDITION_STAMINA_VALUE.max_val = exp.max_val
        EXPEDITION_STAMINA = exp.current

    train = values.get("training")
    if train is not None:
        TRAINING_STAMINA_VALUE.current = train.current
        TRAINING_STAMINA_VALUE.max_val = train.max_val
        TRAINING_STAMINA = train.current

    return values


if __name__ == "__main__":
    print("=" * 60)
    print("Stamina Reader Test")
    print("=" * 60)

    reader = StaminaReader()

    print("\nReading all stamina values...")
    values = reader.read_all_stamina(force_refresh=True)

    for stamina_type, value in values.items():
        if value:
            print(f"\n{stamina_type}:")
            print(f"  Current: {value.current}")
            print(f"  Max: {value.max_val}")
            print(f"  Ratio: {value.ratio:.2%}")
            print(f"  Is Full: {value.is_full}")
            print(f"  Is Empty: {value.is_empty}")
        else:
            print(f"\n{stamina_type}: Failed to read")

    print("\n" + "=" * 60)
    print("Test complete!")