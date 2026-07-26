"""Direct test of updated stamina reader with all fixes."""
import sys
import re
import cv2
import numpy as np
from pathlib import Path

WHITE_HSV_LOWER = np.array([0, 0, 150])
WHITE_HSV_UPPER = np.array([180, 100, 255])

DIGIT_SIZE = (28, 40)

test_images = [
    "screenshots/stamina1_full_numbers.png",
    "screenshots/stamina2_full_numbers.png",
    "screenshots/stamina1_numbers_20260726_201455.png",
    "screenshots/stamina2_numbers_20260726_201455.png",
]

def extract_digits_robust(image):
    """Extract digits with multi-strategy approach."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    all_regions = []

    for thresh_val in [120, 140, 160, 180, 200]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary_close, connectivity=8
        )

        regions = []
        for lid in range(1, num_labels):
            x = stats[lid, cv2.CC_STAT_LEFT]
            y = stats[lid, cv2.CC_STAT_TOP]
            w = stats[lid, cv2.CC_STAT_WIDTH]
            h = stats[lid, cv2.CC_STAT_HEIGHT]
            area = stats[lid, cv2.CC_STAT_AREA]

            if area < 5 or w < 2 or h < 3:
                continue

            aspect = w / h if h > 0 else 0
            if aspect < 0.08 or aspect > 5.0:
                continue

            pad = 2
            dx1 = max(0, x - pad)
            dy1 = max(0, y - pad)
            dx2 = min(image.shape[1], x + w + pad)
            dy2 = min(image.shape[0], y + h + pad)

            digit_img = image[dy1:dy2, dx1:dx2]
            regions.append((x, y, w, h, digit_img, aspect, thresh_val))

        regions.sort(key=lambda r: r[0])

        if len(regions) >= 2:
            print(f"    Threshold {thresh_val}: {len(regions)} digit regions ✅")
            for i, (x, y, w, h, _, aspect, tv) in enumerate(regions):
                print(f"      [{i}] x={x}, y={y}, w={w}, h={h}, aspect={aspect:.2f}")
            return regions

        print(f"    Threshold {thresh_val}: {len(regions)} regions (need more)")

    return all_regions


print("=" * 60)
print("DIRECT DIGIT EXTRACTION TEST")
print("=" * 60)

for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"Image: {img_path}")
    print(f"{'='*60}")

    img_file = Path(img_path)
    if not img_file.exists():
        print("  File not found, skipping")
        continue

    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if img is None:
        print("  Cannot read, skipping")
        continue

    h, w = img.shape[:2]
    print(f"  Size: {w}x{h}")

    regions = extract_digits_robust(img)

    if len(regions) >= 2:
        print(f"\n  Result: Found {len(regions)} digit regions")
        for i, (x, y, ww, hh, digit_img, aspect, tv) in enumerate(regions):
            char_hint = f" (narrow '/')" if aspect < 0.3 else ""
            print(f"    [{i}] x={x}, y={y}, w={ww}, h={hh}, aspect={aspect:.2f}{char_hint}")

            digit_out = Path(f"screenshots/test_digit_{img_file.stem}_{i}.png")
            cv2.imwrite(str(digit_out), digit_img)
    else:
        print(f"\n  ❌ Failed to extract enough digits ({len(regions)} found)")

print("\n" + "=" * 60)
print("TEST COMPLETE")