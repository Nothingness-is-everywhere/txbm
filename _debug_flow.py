import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

# 详细分析 stamina1_full_numbers.png 的提取流程
name = 'stamina1_full_numbers.png'
path = screenshots / name
region = _imread(str(path), cv2.IMREAD_COLOR)
h, w = region.shape[:2]
print(f'分析: {name} ({w}x{h})')

# 步骤1: HSV白色分割
hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

kernel = np.ones((1, 1), np.uint8)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

# 提取初始区域
num_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
initial_regions = []
for lid in range(1, num_labels):
    x = stats[lid, cv2.CC_STAT_LEFT]
    y = stats[lid, cv2.CC_STAT_TOP]
    ww = stats[lid, cv2.CC_STAT_WIDTH]
    hh = stats[lid, cv2.CC_STAT_HEIGHT]
    area = stats[lid, cv2.CC_STAT_AREA]
    aspect = ww / hh if hh > 0 else 0
    
    pad = 2
    dx1, dy1 = max(0, x - pad), max(0, y - pad)
    dx2, dy2 = min(w, x + ww + pad), min(h, y + hh + pad)
    digit_img = region[dy1:dy2, dx1:dx2]
    
    initial_regions.append((x, y, ww, hh, digit_img, aspect))

print(f'\n初始区域 (HSV掩码): {len(initial_regions)} 个')
for i, (x, y, ww, hh, img, aspect) in enumerate(initial_regions):
    area = ww * hh
    print(f'  [{i}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, area={area:4d}, aspect={aspect:.2f}')

# 步骤2: 对大区域进行分割
print(f'\n对大区域进行分割:')
final_regions = []
for x, y, ww, hh, img, aspect in initial_regions:
    area = ww * hh
    is_large = (ww > hh * 1.3 and ww > 20) or area > 800
    
    if not is_large:
        final_regions.append((x, y, ww, hh, img, aspect))
        print(f'  保留: x={x}, y={y}, w={ww}, h={hh}')
        continue
    
    print(f'  分割: x={x}, y={y}, w={ww}, h={hh}, area={area}')
    
    # 尝试列分割
    sub_regions = reader._split_by_columns(x, y, ww, hh, region)
    print(f'    列分割结果: {len(sub_regions)} 个')
    
    if len(sub_regions) < 2:
        # 尝试分水岭
        sub_regions = reader._split_region_watershed(x, y, ww, hh, region)
        print(f'    分水岭结果: {len(sub_regions)} 个')
    
    if len(sub_regions) >= 2:
        for j, (sx, sy, sw, sh, simg, sa) in enumerate(sub_regions):
            print(f'      子[{j}]: x={sx}, y={sy}, w={sw}, h={sh}, aspect={sa:.2f}')
            final_regions.append((sx, sy, sw, sh, simg, sa))
    else:
        print(f'    无法分割，保留原区域')
        final_regions.append((x, y, ww, hh, img, aspect))

# 按x坐标排序
final_regions.sort(key=lambda r: r[0])

print(f'\n最终区域: {len(final_regions)} 个')
for i, (x, y, ww, hh, img, aspect) in enumerate(final_regions):
    area = ww * hh
    char, conf = reader._template_lib.recognize(img)
    is_slash = (aspect > 2.0 and hh >= 3) or (aspect < 0.20 and hh >= 3 and ww >= 2)
    
    print(f'  [{i}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, area={area:4d}, aspect={aspect:.2f} => {"slash" if is_slash else char} ({conf:.3f})')

print('\n完成!')