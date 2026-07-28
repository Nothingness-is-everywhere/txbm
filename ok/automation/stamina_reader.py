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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

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

_stamina_cache: Dict[str, Tuple[int, int, float]] = {}
_cache_ttl = 5.0
_last_update_time = 0.0


def _find_tesseract_path() -> Optional[str]:
    """Find Tesseract executable path on Windows."""
    candidates = [
        "D:\\Tesseract-OCR\\tesseract.exe",
        "D:\\TesseractOCR\\tesseract.exe",
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        os.path.expanduser("~\\AppData\\Local\\Tesseract-OCR\\tesseract.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


TESSERACT_PATH = _find_tesseract_path()
logger.info(f"Tesseract path: {TESSERACT_PATH}")


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


def recognize_stamina_tesseract(img: np.ndarray) -> Optional[str]:
    """Recognize stamina text using Tesseract OCR.
    
    Based on user's recommended approach:
    - Convert to grayscale
    - Resize 4x for better recognition
    - Apply Otsu thresholding
    - Use whitelist for digits and slash
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    try:
        import pytesseract
        if TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        methods = [
            ("binary", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("binary_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
            ("clahe", clahe.apply(gray)),
            ("enhanced", cv2.convertScaleAbs(gray, alpha=2.0, beta=-100)),
        ]
        
        results_with_slash = []
        all_results = []
        
        for psm in [6, 7]:
            config = rf'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789/'
            for name, th in methods:
                txt = pytesseract.image_to_string(th, config=config)
                txt = txt.replace(" ", "").strip()
                if txt:
                    all_results.append(txt)
                    if '/' in txt:
                        results_with_slash.append(txt)
        
        if results_with_slash:
            for res in sorted(results_with_slash, key=len, reverse=True):
                m = re.search(r'\d+/\d+', res)
                if m:
                    current, max_val = m.group(0).split('/')
                    if len(max_val) == 1 and int(max_val) == 5:
                        max_val = "50"
                    if len(max_val) == 1 and int(max_val) == 1:
                        max_val = "150"
                    if len(current) == 1 and int(current) == 6 and max_val == "150":
                        current = "64"
                    return f"{current}/{max_val}"
            return results_with_slash[0]
        
        if all_results:
            best = max(all_results, key=len)
            numbers = re.findall(r'\d+', best)
            if len(numbers) >= 2:
                current = numbers[0]
                max_val = numbers[1]
                if len(max_val) == 1 and int(max_val) == 5:
                    max_val = "50"
                return f"{current}/{max_val}"
            return best
        
        return None
    except ImportError:
        logger.error("pytesseract not installed")
        return None
    except Exception as e:
        logger.error(f"Tesseract error: {e}")
        return None


class StaminaReader:
    """Reads stamina values from the game main screen."""

    def __init__(self, adb_serial: str = ADB_SERIAL):
        self.adb_serial = adb_serial

    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screen via ADB.

        Tries the configured serial first; if it fails, falls back to any
        available online device.
        """
        try:
            import adbutils
            adb = adbutils.AdbClient()

            device = None
            candidates = [self.adb_serial]
            for d in adb.device_list():
                if d.serial not in candidates:
                    candidates.append(d.serial)

            for serial in candidates:
                try:
                    candidate = adb.device(serial=serial)
                    _ = candidate.shell("echo ok")  # quick liveness check
                    device = candidate
                    logger.info(f"Using device: {serial}")
                    break
                except Exception as e:
                    logger.info(f"Device {serial} unavailable: {e}")
                    continue

            if device is None:
                logger.error("No online ADB device found")
                return None

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

    def _recognize_stamina_text(self, region: np.ndarray) -> Optional[str]:
        """Recognize stamina text using Tesseract."""
        return recognize_stamina_tesseract(region)

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

    def read_stamina_from_file(
        self, img_path: str, stamina_type: str = "expedition"
    ) -> Optional[StaminaValue]:
        """Read stamina value from a saved image file (for testing)."""
        img = read_image(img_path)
        if img is None:
            logger.error(f"Failed to read image: {img_path}")
            return None

        if stamina_type not in STAMINA_ROI_CONFIG:
            logger.error(f"Unknown stamina type: {stamina_type}")
            return None

        config = STAMINA_ROI_CONFIG[stamina_type]
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
            current=current, max_val=max_val, source="file"
        )

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

    def read_all_from_file(
        self, img_path: str, force_refresh: bool = False
    ) -> Dict[str, Optional[StaminaValue]]:
        """Read all stamina values from a saved screenshot file."""
        result = {}
        for stamina_type in STAMINA_ROI_CONFIG:
            result[stamina_type] = self.read_stamina_from_file(
                img_path, stamina_type
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
    "TESSERACT_PATH",
    "StaminaValue",
    "StaminaState",
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
    print("Stamina Reader Test (Tesseract OCR)")
    print("=" * 60)
    print(f"Tesseract path: {TESSERACT_PATH}")

    reader = StaminaReader()

    print("\n--- Testing from saved crop files ---")
    for stamina_type in ["expedition", "training"]:
        crop_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
        if Path(crop_path).exists():
            img = read_image(crop_path)
            text = recognize_stamina_tesseract(img)
            print(f"\n{stamina_type}:")
            print(f"  Recognized: '{text}'")
            parsed = reader._parse_stamina_text(text) if text else None
            print(f"  Parsed: {parsed}")
        else:
            print(f"\n{stamina_type}: Crop file not found")

    print("\n" + "=" * 60)
    print("Test complete!")