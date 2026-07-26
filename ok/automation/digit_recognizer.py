"""
Digit recognition module for stamina values.

Provides template-based digit recognition as a fallback
when EasyOCR is not available.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digit_recognizer")

DIGIT_TEMPLATES_DIR = "templates/digits"

DIGIT_SIZE = (30, 50)


class DigitRecognizer:
    """Template-based digit recognizer for game UI numbers."""

    def __init__(self):
        self.digit_templates: dict = {}
        self._load_templates()

    def _load_templates(self):
        """Load digit templates from templates/digits directory."""
        templates_path = Path(DIGIT_TEMPLATES_DIR)

        self.digit_templates = {}

        if not templates_path.exists():
            logger.warning(f"Digit templates directory not found: {DIGIT_TEMPLATES_DIR}")
            return

        for digit_dir in templates_path.iterdir():
            if digit_dir.is_dir() or not digit_dir.suffix == '.png':
                continue

            name = digit_dir.stem
            img = cv2.imread(str(digit_dir), cv2.IMREAD_COLOR)
            if img is not None:
                self.digit_templates[name] = img
                logger.debug(f"Loaded digit template: {name}")

    def recognize_digit(
        self, digit_img: np.ndarray, threshold: float = 0.6
    ) -> Optional[str]:
        """
        Recognize a single digit from an image.

        Args:
            digit_img: Image of a single digit
            threshold: Minimum matching confidence

        Returns:
            Recognized digit character or None
        """
        if digit_img is None or digit_img.size == 0:
            return None

        digit_img = cv2.resize(digit_img, DIGIT_SIZE)
        gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_confidence = 0

        for name, template in self.digit_templates.items():
            tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            tpl_resized = cv2.resize(tpl_gray, DIGIT_SIZE)

            result = cv2.matchTemplate(gray, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                best_match = name

        if best_confidence >= threshold:
            logger.debug(
                f"Recognized digit as '{best_match}' "
                f"(confidence: {best_confidence:.3f})"
            )
            return best_match
        else:
            logger.debug(
                f"Digit recognition failed. "
                f"Best match: '{best_match}' "
                f"confidence: {best_confidence:.3f}"
            )
            return None

    def extract_digits(
        self,
        number_region: np.ndarray,
        min_height: int = 15,
    ) -> List[Tuple[str, int]]:
        """
        Extract and recognize digits from a number region.

        Args:
            number_region: Image region containing numbers
            min_height: Minimum height for digit contours

        Returns:
            List of (recognized_digit, x_position) tuples
        """
        if number_region is None or number_region.size == 0:
            return []

        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        digit_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= min_height and w >= 5:
                digit_roi = number_region[y:y+h, x:x+w]
                digit_regions.append((x, digit_roi))

        digit_regions.sort(key=lambda r: r[0])

        recognized = []
        for x, roi in digit_regions:
            digit = self.recognize_digit(roi)
            if digit is not None:
                recognized.append((digit, x))

        return recognized

    def read_number(
        self, number_region: np.ndarray
    ) -> Optional[str]:
        """
        Read a complete number string from a region.

        Args:
            number_region: Image region containing the number

        Returns:
            Recognized number string or None
        """
        digits = self.extract_digits(number_region)

        if not digits:
            return None

        return "".join(d[0] for d in digits)

    def read_stamina_text(
        self, number_region: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Read stamina text in "current/max" format.

        Args:
            number_region: Image region containing stamina text

        Returns:
            Tuple of (current, max) or None
        """
        text = self.read_number(number_region)

        if text is None:
            return None

        parts = text.split('/')
        if len(parts) >= 2:
            try:
                current = int(parts[0])
                max_val = int(parts[1])
                return (current, max_val)
            except ValueError:
                pass

        import re
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            return (int(numbers[0]), int(numbers[1]))
        elif len(numbers) == 1:
            return (int(numbers[0]), 0)

        return None


def create_digit_template(
    digit_img: np.ndarray, digit_value: str, output_dir: str = DIGIT_TEMPLATES_DIR
) -> bool:
    """
    Create a template for a specific digit.

    Args:
        digit_img: Image of the digit
        digit_value: The digit value (e.g., '0', '1', etc.)
        output_dir: Directory to save the template

    Returns:
        True if saved successfully
    """
    os.makedirs(output_dir, exist_ok=True)

    digit_img = cv2.resize(digit_img, DIGIT_SIZE)
    filepath = os.path.join(output_dir, f"digit_{digit_value}.png")
    cv2.imwrite(filepath, digit_img)
    logger.info(f"Saved digit template: {filepath}")
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        action = sys.argv[1]

        if action == "read":
            if len(sys.argv) > 2:
                img_path = sys.argv[2]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    recognizer = DigitRecognizer()
                    result = recognizer.read_number(img)
                    print(f"Recognized: {result}")
                else:
                    print(f"Failed to load: {img_path}")
            else:
                print("Usage: python digit_recognizer.py read <image_path>")
        elif action == "templates":
            recognizer = DigitRecognizer()
            print(f"Loaded {len(recognizer.digit_templates)} templates:")
            for name in recognizer.digit_templates:
                print(f"  - {name}")
    else:
        print("Usage:")
        print("  python digit_recognizer.py read <image_path>")
        print("  python digit_recognizer.py templates")