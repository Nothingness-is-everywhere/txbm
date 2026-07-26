"""Debug digit extraction and recognition for stamina values."""
import sys
import cv2
import numpy as np
from pathlib import Path

WHITE_HSV_LOWER = np.array([0, 0, 180])
WHITE_HSV_UPPER = np.array([180, 80, 255])

test_images = [
    "screenshots/stamina1_full_numbers.png",
    "screenshots/stamina2_full_numbers.png",
]

for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"Analyzing: {img_path}")
    print(f"{'='*60}")

    img_file = Path(img_path)
    if not img_file.exists():
        print(f"  File not found!")
        continue

    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  Cannot read image!")
        continue

    h, w = img.shape[:2]
    print(f"  Size: {w}x{h}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

    white_ratio = np.sum(white_mask > 0) / white_mask.size
    print(f"  White pixel ratio (HSV): {white_ratio:.3f}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ratio = np.sum(gray > 180) / gray.size
    print(f"  Bright pixel ratio (gray>180): {gray_ratio:.3f}")

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    print(f"\n  Connected components (gray threshold 180):")
    print(f"    Total labels: {num_labels}")

    valid_count = 0
    for label_id in range(1, min(num_labels, 50)):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        ww = stats[label_id, cv2.CC_STAT_WIDTH]
        hh = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        is_digit = (hh >= 10 and hh <= 70 and
                    ww >= 3 and ww <= 60 and
                    area >= 10 and area <= 900)

        if is_digit:
            valid_count += 1
            aspect = ww / hh if hh > 0 else 0
            print(f"    [{label_id}] x={x}, y={y}, w={ww}, h={hh}, "
                  f"area={area:.0f}, aspect={aspect:.2f} ✅ DIGIT")

    print(f"    Valid digits found: {valid_count}")

    out_dir = Path("screenshots/debug_digits")
    out_dir.mkdir(exist_ok=True)

    for thresh_val in [150, 180, 200]:
        _, bin_img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        n_labels, _, _, _ = cv2.connectedComponentsWithStats(
            bin_img, connectivity=8
        )

        valid = 0
        for lid in range(1, min(n_labels, 50)):
            st = stats[lid] if lid < len(stats) else None
            if st is not None:
                hh = st[cv2.CC_STAT_HEIGHT]
                ww = st[cv2.CC_STAT_WIDTH]
                ar = st[cv2.CC_STAT_AREA]
                if 10 <= hh <= 70 and 3 <= ww <= 60 and 10 <= ar <= 900:
                    valid += 1

        print(f"\n  Threshold {thresh_val}: {n_labels - 1} components, {valid} valid digits")

    debug_out = img.copy()
    for lid in range(1, min(num_labels, 50)):
        x = stats[lid, cv2.CC_STAT_LEFT]
        y = stats[lid, cv2.CC_STAT_TOP]
        ww = stats[lid, cv2.CC_STAT_WIDTH]
        hh = stats[lid, cv2.CC_STAT_HEIGHT]
        ar = stats[lid, cv2.CC_STAT_AREA]

        if 10 <= hh <= 70 and 3 <= ww <= 60 and 10 <= ar <= 900:
            cv2.rectangle(debug_out, (x, y), (x+ww, y+hh), (0, 255, 0), 2)

    out_path = out_dir / f"debug_{img_file.stem}.png"
    cv2.imwrite(str(out_path), debug_out)
    print(f"\n  Debug image saved: {out_path}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")