"""
Stamina reader module for game stamina values.

Provides automatic recognition of expedition and training stamina values
from the main game screen, with configurable ROI positions and support
for multiple recognition engines (EasyOCR and template-based).
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

from ok.automation.digit_recognizer import DigitRecognizer

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


@dataclass
class StaminaValue:
    """Stamina value with current/max and derived properties."""

    current: int = 0
    max_val: int = 0
    source: str = "unknown"

    @property
    def ratio(self) -> float:
        """Return the stamina ratio as a float between 0 and 1."""
        if self.max_val <= 0:
            return 0.0
        return float(self.current) / float(self.max_val)

    @property
    def is_full(self) -> bool:
        """Check if stamina is full."""
        return self.current >= self.max_val > 0

    @property
    def is_empty(self) -> bool:
        """Check if stamina is empty."""
        return self.current <= 0

    def __str__(self) -> str:
        return f"{self.current}/{self.max_val}"

    def __repr__(self) -> str:
        return f"StaminaValue(current={self.current}, max_val={self.max_val})"


class StaminaReader:
    """Reads stamina values from the game main screen."""

    def __init__(self, adb_serial: str = ADB_SERIAL):
        self.adb_serial = adb_serial
        self.digit_recognizer = DigitRecognizer()
        self._easyocr_reader = None
        self._init_easyocr()

    def _init_easyocr(self):
        """Initialize EasyOCR if available."""
        global _ENGINE
        if _ENGINE is not None:
            self._easyocr_reader = _ENGINE
            return

        try:
            import torch
            logger.info("PyTorch is available")
        except ImportError:
            logger.warning("PyTorch not available")

        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            _ENGINE = self._easyocr_reader
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not available, using template-based recognition")
            self._easyocr_reader = None
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            logger.warning("Falling back to template-based recognition")
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

    def _recognize_with_easyocr(self, region: np.ndarray) -> Optional[str]:
        """Recognize text using EasyOCR."""
        if self._easyocr_reader is None:
            return None

        try:
            results = self._easyocr_reader.readtext(region)
            if results:
                text = " ".join([r[1] for r in results])
                logger.debug(f"EasyOCR result: '{text}'")
                return text
        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {e}")

        return None

    def _recognize_with_templates(self, region: np.ndarray) -> Optional[str]:
        """Recognize text using template matching."""
        number = self.digit_recognizer.read_number(region)
        if number:
            logger.debug(f"Template recognition result: '{number}'")
        return number

    def _parse_stamina_text(self, text: str) -> Optional[Tuple[int, int]]:
        """Parse stamina text in 'current/max' format."""
        import re

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
                return (int(numbers[0]), 0)
            except ValueError:
                pass

        return None

    def read_stamina(
        self, stamina_type: str = "expedition", force_refresh: bool = False
    ) -> Optional[StaminaValue]:
        """
        Read stamina value from the game screen.

        Args:
            stamina_type: Type of stamina ('expedition' or 'training')
            force_refresh: Force refresh instead of using cache

        Returns:
            StaminaValue object or None if reading fails
        """
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

        text = self._recognize_with_easyocr(region)
        if text is None:
            text = self._recognize_with_templates(region)

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
        """
        Read all stamina values from the game screen.

        Args:
            force_refresh: Force refresh instead of using cache

        Returns:
            Dictionary mapping stamina type to StaminaValue
        """
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
    """
    Initialize stamina readings by reading both expedition and training values.

    Args:
        force_refresh: Force fresh reading instead of using cache

    Returns:
        Dictionary with 'expedition' and 'training' stamina values
    """
    global _expedition_stamina, _training_stamina, _reader_instance

    if _reader_instance is None:
        _reader_instance = StaminaReader()

    values = _reader_instance.read_all_stamina(force_refresh=force_refresh)

    _expedition_stamina = values.get("expedition")
    _training_stamina = values.get("training")

    return values


def get_stamina(stamina_type: str = "expedition", force_refresh: bool = False) -> Optional[StaminaValue]:
    """
    Get stamina value for the specified type.

    Args:
        stamina_type: 'expedition' or 'training'
        force_refresh: Force fresh reading

    Returns:
        StaminaValue object or None

    Example:
        >>> from ok.automation.stamina_reader import get_stamina, init_stamina
        >>> init_stamina()
        >>> exp = get_stamina('expedition')
        >>> print(f"Expedition: {exp.current}/{exp.max_val}")
        >>> train = get_stamina('training')
        >>> print(f"Training: {train.current}/{train.max_val}")
    """
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


EXPEDITION_STAMINA = 0
TRAINING_STAMINA = 150

TRAINING_STAMINA_VALUE = StaminaValue(current=0, max_val=0)
EXPEDITION_STAMINA_VALUE = StaminaValue(current=0, max_val=0)


def update_global_stamina(force_refresh: bool = False):
    """
    Update global stamina variables.

    This is the main entry point for refreshing stamina values.
    After calling this, EXPEDITION_STAMINA, TRAINING_STAMINA,
    EXPEDITION_STAMINA_VALUE, and TRAINING_STAMINA_VALUE will be updated.

    Args:
        force_refresh: Force fresh reading
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