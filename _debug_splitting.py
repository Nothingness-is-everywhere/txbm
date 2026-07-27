"""Debug the column-based splitting."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import (
    StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER
)

screenshots = PROJECT / "screenshots"
reader = StaminaReader()

# Load stamina1 image
img1 = _imread(str(screenshots / "stamina1_full_numbers.png"), cv2.IMREAD_COLOR)
h, w = img1.shape[:2]

# Find the digit regions
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
kernel = np.ones((1, 1), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

print(f"Connected components: {num_labels - 1}")
for lid in range(1, num_labels):
    x = stats[lid, cv2.CC_STAT_LEFT]
    y = stats[lid, cv2.CC_STAT_TOP]
    ww = stats[lid, cv2.CC_STAT_WIDTH]
    hh = stats[lid, cv2.CC_STAT_HEIGHT]
    area = stats[lid, cv2.CC_STAT_AREA]
    aspect = ww / hh if hh > 0 else 0
    print(f"  [{lid}] x={x}, y={y}, w={ww}, h={hh}, area={area}, aspect={aspect:.2f}")

# Test splitting on the large merged region
print("\n" + "=" * 60)
print("Testing column-based splitting on merged region")
print("=" * 60)

# The large region: x=66, y=0, w=50, h=53
lx, ly, lw, lh = 66, 0, 50, 53
roi = img1[ly:ly+lh, lx:lx+lw]

print(f"ROI: {roi.shape}")

# Apply the same logic as _split_by_columns
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask_roi = cv2.inRange(hsv_roi, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary_roi = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
mask_roi = cv2.bitwise_or(mask_roi, binary_roi)

# Column-wise density
col_counts = np.sum(mask_roi > 0, axis=0)
print(f"\nColumn counts (0-{lw-1}):")
for i, count in enumerate(col_counts):
    bar = '#' * int(count / 2)
    print(f"  col {i:2d}: {count:3d} {bar}")

smoothed = np.convolve(col_counts, np.ones(3)/3, mode='same')
print(f"\nSmoothed: {smoothed}")

threshold = lh * 0.15
print(f"Threshold: {threshold}")

gap_columns = smoothed < threshold
print(f"Gap columns: {gap_columns}")

# Find segments
segments = []
start = 0
in_segment = False
for col in range(lw):
    if not gap_columns[col]:
        if not in_segment:
            start = col
            in_segment = True
    else:
        if in_segment:
            segments.append((start, col - 1))
            in_segment = False
if in_segment:
    segments.append((start, lw - 1))

print(f"\nSegments: {segments}")

# Visualize
vis = img1.copy()
for seg_start, seg_end in segments:
    sw = seg_end - seg_start + 1
    sx = lx + seg_start
    cv2.rectangle(vis, (sx, ly), (sx + sw, ly + lh), (0, 255, 0), 2)
    print(f"  Segment: x={sx}, w={sw}")

# Save visualization
cv2.imwrite(str(screenshots / "debug_split.png"), vis)
print(f"\nDebug visualization saved")

print("\nDone!")