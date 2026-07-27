import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER, STAMINA_CONFIG

reader = StaminaReader()
template_lib = reader._template_lib
screenshots = PROJECT / 'screenshots'

img = _imread(str(screenshots / 'full_screen_20260726_203459.png'), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
reader.height, reader.width = h, w

config = STAMINA_CONFIG['stamina1']
bar_region = reader._locate_bar(img, 'stamina1')
bx, by, bw, bh = bar_region

ox, oy, ow, oh = config['number_relative_offset']
nx = bx + int(bw * ox)
ny = by + int(bh * oy)
nw = int(bw * ow)
nh = int(bh * oh)

number_region = img[ny:ny+nh, nx:nx+nw]

x, y, w, h = 29, 0, 112, 35

roi = number_region[y:y+h, x:x+w]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
mask = cv2.bitwise_or(mask, binary)

col_counts = np.sum(mask > 0, axis=0)

print("列密度分析:")
for col in range(0, w, 5):
    densities = col_counts[col:min(col+5, w)]
    print(f"  列 {col}-{min(col+4, w-1)}: {list(densities)}")

all_candidates = []
for smooth_window in [1, 3, 5, 7, 9]:
    if smooth_window > 1:
        smoothed = np.convolve(col_counts, np.ones(smooth_window)/smooth_window, mode='same')
    else:
        smoothed = col_counts.astype(np.float64)

    max_density = np.max(smoothed)
    if max_density < 3:
        continue

    for threshold_factor in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        rel_threshold = max_density * threshold_factor
        gap_columns = smoothed < rel_threshold

        segments = []
        start = 0
        in_segment = False

        for col in range(w):
            if not gap_columns[col]:
                if not in_segment:
                    start = col
                    in_segment = True
            else:
                if in_segment:
                    segments.append((start, col - 1))
                    in_segment = False

        if in_segment:
            segments.append((start, w - 1))

        valid_segments = [(s, e) for s, e in segments if e - s + 1 >= 2]

        if 2 <= len(valid_segments) <= 8:
            score = reader._evaluate_segment_quality(valid_segments, roi)
            all_candidates.append((valid_segments, score))

            if smooth_window == 5 and threshold_factor in [0.30, 0.40, 0.50]:
                print(f"\n  平滑={smooth_window}, 阈值={threshold_factor:.2f}: {len(valid_segments)} 段, score={score:.3f}")
                for i, (s, e) in enumerate(valid_segments):
                    sub_roi = roi[:, s:e+1]
                    sw = e - s + 1
                    char, conf = template_lib.recognize(sub_roi)
                    print(f"    段 {i}: [{s},{e}] w={sw}, 识别='{char}' (conf={conf:.3f})")

print(f"\n共 {len(all_candidates)} 个候选方案")
all_candidates.sort(key=lambda x: x[1], reverse=True)

if all_candidates:
    best_segments = all_candidates[0][0]
    best_score = all_candidates[0][1]
    print(f"\n最佳方案: score={best_score:.3f}")
    for i, (s, e) in enumerate(best_segments):
        print(f"  段 {i}: [{s},{e}] w={e-s+1}")
else:
    print("\n没有找到有效的候选方案!")

print("\n完成!")