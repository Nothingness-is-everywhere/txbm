import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER

reader = StaminaReader()
template_lib = reader._template_lib
screenshots = PROJECT / 'screenshots'

region = _imread(str(screenshots / 'stamina1_full_numbers.png'), cv2.IMREAD_COLOR)
h, w = region.shape[:2]

x, y, rw, rh = 66, 0, 50, 53

print("=" * 60)
print(f"分析区域 [{x}, {y}, {rw}, {rh}]")
print("=" * 60)

roi = region[y:y+rh, x:x+rw]
print(f"ROI 尺寸: {roi.shape[1]}x{roi.shape[0]}")

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
mask_combined = cv2.bitwise_or(mask, binary)

col_counts = np.sum(mask_combined > 0, axis=0)

print("\n列密度分析:")
for col in range(rw):
    density = col_counts[col]
    bar = '#' * min(int(density / 3), 30) if density > 0 else ''
    print(f"  列 {col:2d}: 密度={density:3d} {bar}")

smoothed = np.convolve(col_counts, np.ones(5)/5, mode='same')
max_density = np.max(smoothed)
print(f"\n平滑后最大密度: {max_density:.1f}")

print("\n尝试不同阈值:")
for threshold_factor in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    rel_threshold = max_density * threshold_factor
    gap_columns = smoothed < rel_threshold
    
    segments = []
    start = 0
    in_segment = False
    for col in range(rw):
        if not gap_columns[col]:
            if not in_segment:
                start = col
                in_segment = True
        else:
            if in_segment:
                segments.append((start, col - 1))
                in_segment = False
    if in_segment:
        segments.append((start, rw - 1))
    
    valid = [(s, e) for s, e in segments if e - s + 1 >= 2]
    print(f"  阈值 {threshold_factor:.2f}: {len(segments)} 段, {len(valid)} 有效")
    for i, (s, e) in enumerate(valid):
        sub_x = x + s
        sub_roi = roi[0:rh, s:e+1]
        sub_area = sub_roi.shape[0] * sub_roi.shape[1]
        sub_char, sub_conf = template_lib.recognize(sub_roi)
        print(f"    子区域 {i}: x={sub_x}, w={e-s+1}, area={sub_area}, 识别='{sub_char}' (conf={sub_conf:.3f})")

print("\n" + "=" * 60)
print("直接用提取的分割结果测试")
print("=" * 60)

sub_regions = reader._split_by_columns(x, y, rw, rh, region)
print(f"\n_split_by_columns 返回 {len(sub_regions)} 个子区域:")
for i, (sx, sy, sw, sh, simg, sa) in enumerate(sub_regions):
    char, conf = template_lib.recognize(simg)
    print(f"  [{i}] x={sx}, y={sy}, w={sw}, h={sh}, aspect={sa:.3f}, 识别='{char}' (conf={conf:.3f})")
    
    cv2.imwrite(str(screenshots / f'split_sub_{i}.png'), simg)

print("\n完成!")