"""Utility to generate and organize digit templates for stamina reader.

This script:
1. Reads existing digit images from screenshots/ and templates/
2. Extracts and normalizes them into templates/fonts/ directory
3. Generates complete template set for digits 0-9

Usage:
  python _generate_digit_templates.py
  python _generate_digit_templates.py --generate-synthetic
"""

import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

TEMPLATES_DIR = "templates"
FONTS_DIR = "templates/fonts"
DIGIT_SIZE = (28, 40)

WHITE_HSV_LOWER = np.array([0, 0, 180])
WHITE_HSV_UPPER = np.array([180, 80, 255])


def extract_digit_value(filename: str) -> Optional[str]:
    """Extract digit value from filename."""
    name_lower = filename.lower()

    digit_map = {
        'd0': '0', '_0': '0', 'digit_0': '0', '0.png': '0',
        'd1': '1', '_1': '1', 'digit_1': '1', '1.png': '1',
        'd2': '2', '_2': '2', 'digit_2': '2', '2.png': '2',
        'd3': '3', '_3': '3', 'digit_3': '3', '3.png': '3',
        'd4': '4', '_4': '4', 'digit_4': '4', '4.png': '4',
        'd5': '5', '_5': '5', 'digit_5': '5', '5.png': '5',
        'd6': '6', '_6': '6', 'digit_6': '6', '6.png': '6',
        'd7': '7', '_7': '7', 'digit_7': '7', '7.png': '7',
        'd8': '8', '_8': '8', 'digit_8': '8', '8.png': '8',
        'd9': '9', '_9': '9', 'digit_9': '9', '9.png': '9',
    }

    for pattern, value in digit_map.items():
        if pattern in name_lower:
            return value

    match = re.search(r'(\d)', filename)
    if match:
        return match.group(1)

    return None


def preprocess_digit(img: np.ndarray) -> Optional[np.ndarray]:
    """Preprocess a digit image into a normalized template."""
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

    resized = cv2.resize(cropped, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    return binary


def collect_existing_templates() -> List[Tuple[str, str, np.ndarray]]:
    """Collect existing digit images from all template directories."""
    sources = [
        "templates/digits",
        "templates/digit_templates",
        "screenshots",
        "debug",
    ]

    templates = []

    for source_dir in sources:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"  Directory not found: {source_dir}")
            continue

        for img_file in sorted(source_path.glob("*.png")):
            digit = extract_digit_value(img_file.stem)
            if digit is None:
                continue

            img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue

            processed = preprocess_digit(img)
            if processed is not None and processed.size > 0:
                templates.append((digit, img_file.stem, processed))

    return templates


def generate_synthetic_templates() -> List[Tuple[str, str, np.ndarray]]:
    """Generate synthetic digit templates using OpenCV fonts."""
    templates = []
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2

    for digit_char in '0123456789':
        canvas = np.zeros((60, 50), dtype=np.uint8)
        canvas.fill(0)

        text_size = cv2.getTextSize(digit_char, font, font_scale, thickness)[0]
        x = (canvas.shape[1] - text_size[0]) // 2
        y = (canvas.shape[0] + text_size[1]) // 2

        cv2.putText(canvas, digit_char, (x, y), font, font_scale, 255, thickness)

        contours, _ = cv2.findContours(
            canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                x_min = min(x_min, bx)
                y_min = min(y_min, by)
                x_max = max(x_max, bx + bw)
                y_max = max(y_max, by + bh)

            padding = 3
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(canvas.shape[1], x_max + padding)
            y_max = min(canvas.shape[0], y_max + padding)

            cropped = canvas[y_min:y_max, x_min:x_max]
            resized = cv2.resize(cropped, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

            templates.append((digit_char, f"synthetic_{digit_char}", binary))

    return templates


def save_templates(templates: List[Tuple[str, str, np.ndarray]], output_dir: str):
    """Save templates organized by digit value."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    digit_counts: dict = {}

    for digit, name, img in templates:
        digit_dir = out_path / digit
        digit_dir.mkdir(exist_ok=True)

        if digit not in digit_counts:
            digit_counts[digit] = 0

        digit_counts[digit] += 1
        count = digit_counts[digit]

        filename = f"{name}_{count}.png"
        filepath = digit_dir / filename
        cv2.imwrite(str(filepath), img)

    for digit in sorted(digit_counts.keys()):
        print(f"  Digit '{digit}': {digit_counts[digit]} templates")

    return digit_counts


def merge_templates():
    """Merge templates from all source directories into fonts/."""
    print("=" * 60)
    print("DIGIT TEMPLATE GENERATOR")
    print("=" * 60)

    print("\n[1] Collecting existing templates...")
    existing = collect_existing_templates()
    print(f"    Found {len(existing)} existing templates")

    print("\n[2] Generating synthetic templates...")
    synthetic = generate_synthetic_templates()
    print(f"    Generated {len(synthetic)} synthetic templates")

    print("\n[3] Merging into fonts/ directory...")
    all_templates = existing + synthetic
    counts = save_templates(all_templates, FONTS_DIR)

    print("\n[4] Creating index file...")
    index = {}
    for digit in sorted(counts.keys()):
        digit_dir = Path(FONTS_DIR) / digit
        files = list(digit_dir.glob("*.png"))
        index[digit] = {
            "count": len(files),
            "files": [str(f.name) for f in files[:5]],
        }

    import json
    index_path = Path(FONTS_DIR) / "index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"    Saved index to {index_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(counts.values())
    digits_covered = [d for d in '0123456789' if d in counts and counts[d] > 0]
    print(f"  Total templates: {total}")
    print(f"  Digits covered: {''.join(digits_covered)}")
    print(f"  Missing digits: {''.join(set('0123456789') - set(digits_covered))}")
    print("=" * 60)


if __name__ == "__main__":
    merge_templates()