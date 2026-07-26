"""
Stamina reader module for 天下布魔 (Tianxia Bumo).

Reads stamina/energy values from the main interface using:
  1. Bar template matching (with limited ROI) for robust localization
  2. HSV color-space white text segmentation
  3. Connected-component analysis for digit extraction
  4. Template-based digit recognition via XOR comparison

This implementation is optimized for speed and accuracy without OCR.

Usage:
  from ok.automation.stamina_reader import StaminaReader, get_stamina
  reader = StaminaReader()
  values = reader.read_all()
  stamina1_current = values.stamina1.current
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

def _imread(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read image file, supporting Unicode paths (e.g., Chinese characters)."""
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None


ADB_SERIAL = "127.0.0.1:16384"

STAMINA_CONFIG = {
    "stamina1": {
        "name": "stamina1",
        "label": "出征",
        "bar_template": "templates/stamina1_bar.png",
        "bar_search_roi": (0.55, 0.60, 0.95, 0.78),
        "number_relative_offset": (0.05, 0.30, 0.80, 0.55),
        "fallback_number_roi": (800, 1275, 130, 40),
    },
    "stamina2": {
        "name": "stamina2",
        "label": "调教",
        "bar_template": "templates/stamina2_bar.png",
        "bar_search_roi": (0.60, 0.72, 0.95, 0.88),
        "number_relative_offset": (0.05, 0.30, 0.80, 0.55),
        "fallback_number_roi": (840, 1510, 120, 50),
    },
}

STAMINA_THRESHOLD = 0.60

DIGIT_TEMPLATE_SIZE = (28, 40)

WHITE_HSV_LOWER = np.array([0, 0, 150])
WHITE_HSV_UPPER = np.array([180, 100, 255])

CONNECTIVITY = 8
MIN_DIGIT_AREA = 5
MAX_DIGIT_AREA = 1200
MIN_DIGIT_HEIGHT = 5
MAX_DIGIT_HEIGHT = 80
MIN_DIGIT_WIDTH = 2
MAX_DIGIT_WIDTH = 70
NARROW_ASPECT_RATIO = 0.25

TEMPLATE_SIMILARITY_THRESHOLD = 0.50

DIGIT_TEMPLATES_DIRS = [
    "templates/digits",
    "templates/digit_templates",
    "templates",
]

CACHE_DURATION = 5.0

SCALE_FACTOR = 1.0


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
    stamina1: StaminaValue = field(
        default_factory=lambda: StaminaValue(name="stamina1")
    )
    stamina2: StaminaValue = field(
        default_factory=lambda: StaminaValue(name="stamina2")
    )
    last_update: float = 0.0

    def to_dict(self) -> Dict[str, StaminaValue]:
        return {
            "stamina1": self.stamina1,
            "stamina2": self.stamina2,
        }

    def summary(self) -> str:
        lines = []
        lines.append(f"Stamina 1 (出征): {self.stamina1}")
        lines.append(f"Stamina 2 (调教): {self.stamina2}")
        return "\n".join(lines)


class DigitTemplateLibrary:
    """Manages digit templates for fast template matching."""

    def __init__(self):
        self._templates: Dict[str, List[np.ndarray]] = {}
        self._load_all_templates()

    def _load_all_templates(self) -> None:
        """Load digit templates from all template directories."""
        loaded = 0
        for dir_path in DIGIT_TEMPLATES_DIRS:
            templates_path = Path(dir_path)
            if not templates_path.exists():
                continue

            for img_file in templates_path.rglob("*.png"):
                digit_char = None
                parent_dir = img_file.parent.name

                if parent_dir in '0123456789':
                    digit_char = parent_dir
                else:
                    digit_char = self._extract_digit(img_file.stem)

                if digit_char is None:
                    continue

                img = _imread(str(img_file), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                processed = self._preprocess_template(img)
                if processed is not None and processed.size > 0:
                    if digit_char not in self._templates:
                        self._templates[digit_char] = []
                    self._templates[digit_char].append(processed)
                    loaded += 1

        if loaded > 0:
            logger.info(f"Loaded {loaded} digit templates "
                        f"({len(self._templates)} digits: "
                        f"{''.join(sorted(self._templates.keys()))})")
        else:
            logger.warning("No digit templates found")

    def _extract_digit(self, name: str) -> Optional[str]:
        """Extract digit character from template filename."""
        name_lower = name.lower()

        for d in '0123456789':
            if f'd{d}' in name_lower or f'_{d}' in name_lower or f'digit_{d}' in name_lower:
                return d

        match = re.search(r'(\d)', name)
        if match:
            return match.group(1)

        return None

    def _preprocess_template(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess a template image for matching."""
        if img is None or img.size == 0:
            return None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        if not contours:
            return None

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)

        padding = 3
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img.shape[1], x_max + padding)
        y_max = min(img.shape[0], y_max + padding)

        cropped = mask[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            return None

        resized = cv2.resize(cropped, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return binary

    def recognize(
        self, digit_img: np.ndarray, threshold: float = TEMPLATE_SIMILARITY_THRESHOLD
    ) -> Tuple[Optional[str], float]:
        """
        Recognize a digit by comparing against all templates.

        Args:
            digit_img: Image of a single digit (white on dark background)
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Tuple of (recognized_digit, confidence) or (best_match, best_score)
        """
        if digit_img is None or digit_img.size == 0:
            return None, 0.0

        processed = self._preprocess_digit_for_matching(digit_img)

        if processed is None or processed.size == 0:
            return None, 0.0

        best_digit = None
        best_score = 0.0

        for digit_char, templates in self._templates.items():
            max_for_digit = 0.0
            for template in templates:
                score = self._compute_similarity(processed, template)
                if score > max_for_digit:
                    max_for_digit = score

            if max_for_digit > best_score:
                best_score = max_for_digit
                best_digit = digit_char

        if best_score >= threshold:
            return best_digit, best_score

        return best_digit, best_score

    def _preprocess_digit_for_matching(self, digit_img: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess a digit image for template matching."""
        if digit_img is None or digit_img.size == 0:
            return None

        hsv = cv2.cvtColor(digit_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        if not contours:
            return None

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + cw)
            y_max = max(y_max, y + ch)

        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(digit_img.shape[1], x_max + padding)
        y_max = min(digit_img.shape[0], y_max + padding)

        cropped = mask[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            return None

        resized = cv2.resize(cropped, DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        return binary

    def _compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute similarity between two binary images using XOR + SSIM."""
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        xor_result = cv2.bitwise_xor(img1, img2)
        total_pixels = img1.shape[0] * img1.shape[1]
        different_pixels = np.sum(xor_result > 0)
        xor_similarity = 1.0 - (different_pixels / total_pixels)

        if total_pixels > 10:
            try:
                ssim_score = self._compute_ssim(img1, img2)
                combined = 0.6 * xor_similarity + 0.4 * ssim_score
                return float(combined)
            except Exception:
                pass

        return float(xor_similarity)

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM-like structural similarity."""
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return float(np.mean(ssim_map))

    def get_template_count(self) -> int:
        """Return the number of loaded templates."""
        return sum(len(templates) for templates in self._templates.values())

    def get_supported_digits(self) -> List[str]:
        """Return list of digits that have templates."""
        return sorted(self._templates.keys())


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

        self._template_lib = DigitTemplateLibrary()
        self._bar_templates: Dict[str, np.ndarray] = {}
        self._load_bar_templates()

        self._ocr_reader = None
        self._ocr_initialized = False
        self._ocr_available = False

    def _load_bar_templates(self) -> None:
        """Load stamina bar templates for localization."""
        for key, cfg in STAMINA_CONFIG.items():
            tpl_path = Path(cfg["bar_template"])
            if tpl_path.exists():
                img = _imread(str(tpl_path), cv2.IMREAD_COLOR)
                if img is not None:
                    self._bar_templates[key] = img
                    logger.info(f"Loaded bar template for {key}: "
                                f"{img.shape[1]}x{img.shape[0]}")

    def capture_screen(self) -> np.ndarray:
        """Capture the current screen from the device."""
        png_bytes = self.device.shell("screencap -p", encoding=None, timeout=10)
        if not png_bytes or len(png_bytes) == 0:
            raise RuntimeError("Failed to capture screen")
        image_data = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode screenshot")
        self.height, self.width = image.shape[:2]
        return image

    def _locate_bar(
        self, screen: np.ndarray, stamina_key: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate a stamina bar on screen using template matching.

        Args:
            screen: Full screen image
            stamina_key: Key in STAMINA_CONFIG

        Returns:
            Tuple of (x, y, w, h) bounding box or None
        """
        cfg = STAMINA_CONFIG[stamina_key]
        tpl = self._bar_templates.get(stamina_key)

        if tpl is None:
            logger.warning(f"No bar template for {stamina_key}")
            return None

        rx1, ry1, rx2, ry2 = cfg["bar_search_roi"]
        roi_x1 = int(self.width * rx1)
        roi_y1 = int(self.height * ry1)
        roi_x2 = int(self.width * rx2)
        roi_y2 = int(self.height * ry2)

        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(self.width, roi_x2)
        roi_y2 = min(self.height, roi_y2)

        if roi_x2 - roi_x1 < tpl.shape[1] or roi_y2 - roi_y1 < tpl.shape[0]:
            logger.warning(f"Search ROI too small for {stamina_key}")
            return None

        img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        roi_region = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]

        result = cv2.matchTemplate(roi_region, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= STAMINA_THRESHOLD:
            tpl_h, tpl_w = tpl_gray.shape
            abs_x = roi_x1 + max_loc[0]
            abs_y = roi_y1 + max_loc[1]
            logger.debug(
                f"[{stamina_key}] Bar found at ({abs_x}, {abs_y}) "
                f"size={tpl_w}x{tpl_h} conf={max_val:.3f}"
            )
            return (abs_x, abs_y, tpl_w, tpl_h)
        else:
            logger.debug(
                f"[{stamina_key}] Bar not found. "
                f"Max conf={max_val:.3f} (threshold={STAMINA_THRESHOLD})"
            )
            return None

    def _extract_digit_regions(
        self, number_region: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray, float]]:
        """
        Extract individual digit regions from a number display area.

        Uses white color segmentation with adaptive thresholding and
        watershed-based splitting for merged digits.

        Returns:
            List of (x, y, w, h, digit_image, aspect_ratio) tuples
        """
        if number_region is None or number_region.size == 0:
            return []

        hsv = cv2.cvtColor(number_region, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

        kernel_small = np.ones((1, 1), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_small)

        digit_regions = self._extract_from_mask(white_mask, number_region)

        if len(digit_regions) >= 3:
            return digit_regions

        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
        for thresh_val in [160, 180, 200, 220]:
            _, binary = cv2.threshold(
                gray, thresh_val, 255, cv2.THRESH_BINARY
            )

            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            regions = self._extract_from_mask(binary, number_region)
            regions = self._split_merged_regions(regions, number_region)

            if len(regions) > len(digit_regions):
                digit_regions = regions

            if len(digit_regions) >= 3:
                break

        digit_regions = self._split_merged_regions(digit_regions, number_region)

        if len(digit_regions) >= 1:
            return digit_regions

        logger.debug("All extraction strategies failed for digit extraction")
        return []

    def _split_merged_regions(
        self,
        regions: List[Tuple[int, int, int, int, np.ndarray, float]],
        number_region: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray, float]]:
        """Split large merged regions into individual digits."""
        if len(regions) < 2:
            return regions

        split_regions = []
        for x, y, w, h, digit_img, aspect in regions:
            area = w * h
            is_large = (w > h * 1.8 and w > 25) or area > 1500

            if not is_large:
                split_regions.append((x, y, w, h, digit_img, aspect))
                continue

            sub_regions = self._split_region_watershed(x, y, w, h, number_region)
            if len(sub_regions) >= 2:
                split_regions.extend(sub_regions)
            else:
                split_regions.append((x, y, w, h, digit_img, aspect))

        split_regions.sort(key=lambda r: r[0])
        return split_regions

    def _split_region_watershed(
        self,
        x: int, y: int, w: int, h: int,
        number_region: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray, float]]:
        """Split a large region using distance transform + watershed."""
        roi = number_region[y:y+h, x:x+w]

        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary_roi = cv2.threshold(gray_roi, 180, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(mask, binary_roi)

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        _, peaks = cv2.threshold(
            dist_transform, 0.3 * dist_transform.max(), 255, cv2.THRESH_BINARY
        )
        peaks = peaks.astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            peaks, connectivity=8
        )

        if num_labels <= 2:
            return []

        markers = np.zeros_like(mask)
        for lid in range(1, num_labels):
            cx = int(stats[lid, cv2.CC_STAT_LEFT] + stats[lid, cv2.CC_STAT_WIDTH] / 2)
            cy = int(stats[lid, cv2.CC_STAT_TOP] + stats[lid, cv2.CC_STAT_HEIGHT] / 2)
            cv2.circle(markers, (cx, cy), 3, int(lid), -1)

        markers = cv2.watershed(roi, markers)

        unique_labels = set(markers.flatten())
        unique_labels.discard(-1)
        unique_labels.discard(0)

        sub_regions = []
        for label in unique_labels:
            label_mask = np.zeros(markers.shape[:2], dtype=np.uint8)
            label_mask[markers == label] = 255

            contours, _ = cv2.findContours(
                label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            lx, ly, lw, lh = cv2.boundingRect(contours[0])

            if lw < 2 or lh < 3:
                continue

            sub_x = x + lx
            sub_y = y + ly
            sub_regions.append(
                (sub_x, sub_y, lw, lh,
                 roi[ly:ly+lh, lx:lx+lw],
                 lw / lh if lh > 0 else 0)
            )

        return sub_regions

    def _extract_from_mask(
        self, mask: np.ndarray, number_region: np.ndarray
    ) -> List[Tuple[int, int, int, int, np.ndarray, float]]:
        """Extract digit regions from a binary mask."""
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=CONNECTIVITY
        )

        digit_regions = []
        for label_id in range(1, num_labels):
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]

            if area < MIN_DIGIT_AREA:
                continue

            if w < MIN_DIGIT_WIDTH or h < MIN_DIGIT_HEIGHT:
                continue

            aspect = w / h if h > 0 else 0

            if aspect < 0.08 or aspect > 4.5:
                continue

            pad = 2
            dx1 = max(0, x - pad)
            dy1 = max(0, y - pad)
            dx2 = min(number_region.shape[1], x + w + pad)
            dy2 = min(number_region.shape[0], y + h + pad)

            digit_img = number_region[dy1:dy2, dx1:dx2]
            digit_regions.append((x, y, w, h, digit_img, aspect))

        digit_regions.sort(key=lambda r: r[0])
        return digit_regions

    def _read_numbers_from_region(
        self, number_region: np.ndarray
    ) -> Optional[str]:
        """
        Read numbers from a region using connected components + template matching.

        Args:
            number_region: Image region containing stamina text

        Returns:
            Recognized text string or None
        """
        if number_region is None or number_region.size == 0:
            return None

        digit_regions = self._extract_digit_regions(number_region)

        if len(digit_regions) < 1:
            logger.debug(
                f"No digit regions found in region "
                f"({number_region.shape[1]}x{number_region.shape[0]})"
            )
            return None

        recognized_parts = []
        prev_x_end = None
        img_height = number_region.shape[0]

        for i, (x, y, w, h, digit_img, aspect) in enumerate(digit_regions):
            if i > 0 and prev_x_end is not None:
                gap = x - prev_x_end
                avg_height = (digit_regions[i - 1][3] + h) / 2
                avg_width = (digit_regions[i - 1][2] + w) / 2

                if gap > avg_height * 0.4 and gap > avg_width * 0.5:
                    recognized_parts.append('/')

            if aspect < NARROW_ASPECT_RATIO and h > 5:
                recognized_parts.append('/')
                prev_x_end = x + w
                continue

            region_area = w * h
            avg_digit_area = (img_height * 0.4) ** 2

            if region_area > avg_digit_area * 2.5:
                sub_digits = self._split_and_recognize(
                    x, y, w, h, number_region
                )
                if sub_digits:
                    recognized_parts.extend(sub_digits)
                    prev_x_end = x + w
                    continue

            digit_char, confidence = self._template_lib.recognize(digit_img)

            if digit_char is not None and confidence >= TEMPLATE_SIMILARITY_THRESHOLD:
                recognized_parts.append(digit_char)
                logger.debug(
                    f"Digit[{i}] matched: '{digit_char}' "
                    f"(confidence: {confidence:.3f}, aspect: {aspect:.2f})"
                )
            else:
                recognized_parts.append('?')
                logger.debug(
                    f"Digit[{i}] unknown "
                    f"(best: {digit_char}, conf: {confidence:.3f})"
                )

            prev_x_end = x + w

        result = ''.join(recognized_parts)
        logger.debug(f"Recognition result: '{result}'")
        return result

    def _split_and_recognize(
        self, x: int, y: int, w: int, h: int,
        number_region: np.ndarray
    ) -> List[str]:
        """Split a large region and recognize each sub-digit."""
        sub_regions = self._split_region_watershed(x, y, w, h, number_region)

        if len(sub_regions) < 2:
            return []

        result = []
        for sx, sy, sw, sh, sub_img, sa in sub_regions:
            digit_char, confidence = self._template_lib.recognize(sub_img)
            if digit_char is not None and confidence >= TEMPLATE_SIMILARITY_THRESHOLD:
                result.append(digit_char)
            else:
                result.append('?')

        return result

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

        cleaned = text.replace('?', '')
        numbers = re.findall(r'\d+', cleaned)

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

    def _get_number_region(
        self, screen: np.ndarray, stamina_key: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the number region for a stamina value.

        Uses bar template matching for robust localization,
        falls back to fixed ROI if bar matching fails.

        Returns:
            Tuple of (x, y, w, h) or None
        """
        cfg = STAMINA_CONFIG[stamina_key]

        bar_pos = self._locate_bar(screen, stamina_key)

        if bar_pos is not None:
            bx, by, bw, bh = bar_pos
            ox, oy, ow, oh = cfg["number_relative_offset"]
            nx = bx + int(bw * ox)
            ny = by + int(bh * oy)
            nw = int(bw * ow)
            nh = int(bh * oh)

            if nx >= 0 and ny >= 0 and nx + nw <= self.width and ny + nh <= self.height:
                logger.debug(
                    f"[{stamina_key}] Number region from bar: "
                    f"({nx},{ny}) {nw}x{nh}"
                )
                return (nx, ny, nw, nh)

        fx, fy, fw, fh = cfg["fallback_number_roi"]
        if fx + fw <= self.width and fy + fh <= self.height:
            logger.debug(
                f"[{stamina_key}] Using fallback ROI: "
                f"({fx},{fy}) {fw}x{fh}"
            )
            return (fx, fy, fw, fh)

        logger.warning(f"[{stamina_key}] No valid region found")
        return None

    def read_stamina1(self, screen: np.ndarray) -> StaminaValue:
        """Read stamina 1 (出征/Expedition) value from screen."""
        region = self._get_number_region(screen, "stamina1")

        if region is None:
            logger.warning("Could not get stamina1 number region")
            return StaminaValue(name="stamina1")

        nx, ny, nw, nh = region
        number_region = screen[ny:ny + nh, nx:nx + nw]

        if number_region.size == 0:
            logger.warning("Stamina1 number region is empty")
            return StaminaValue(name="stamina1")

        text = self._read_numbers_from_region(number_region)
        current, max_val = self._parse_stamina_text(text)

        return StaminaValue(
            current=current,
            max=max_val,
            name="stamina1"
        )

    def read_stamina2(self, screen: np.ndarray) -> StaminaValue:
        """Read stamina 2 (调教/Training) value from screen."""
        region = self._get_number_region(screen, "stamina2")

        if region is None:
            logger.warning("Could not get stamina2 number region")
            return StaminaValue(name="stamina2")

        nx, ny, nw, nh = region
        number_region = screen[ny:ny + nh, nx:nx + nw]

        if number_region.size == 0:
            logger.warning("Stamina2 number region is empty")
            return StaminaValue(name="stamina2")

        text = self._read_numbers_from_region(number_region)
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
        if (
            not force_refresh
            and self._cached_state is not None
            and (time.time() - self._cache_time) < CACHE_DURATION
        ):
            logger.debug("Returning cached stamina state")
            return self._cached_state

        start_time = time.time()

        try:
            screen = self.capture_screen()
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return self._cached_state or StaminaState()

        stamina1 = self.read_stamina1(screen)
        stamina2 = self.read_stamina2(screen)

        elapsed = time.time() - start_time
        logger.info(
            f"Recognition took {elapsed:.3f}s - "
            f"stamina1={stamina1}, stamina2={stamina2}"
        )

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