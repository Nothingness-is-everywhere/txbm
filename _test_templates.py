import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

# 测试模板库对已知数字的识别能力
template_lib = reader._template_lib

print('模板库统计:')
print(f'  总模板数: {template_lib.get_template_count()}')
print(f'  支持的数字: {template_lib.get_supported_digits()}')

# 对每个模板做自匹配测试
print('\n模板自匹配测试:')
for digit in template_lib.get_supported_digits():
    templates = template_lib._templates.get(digit, [])
    if templates:
        scores = []
        for tpl in templates:
            matched_char, conf = template_lib.recognize(tpl)
            scores.append(conf)
        avg_score = np.mean(scores) if scores else 0
        min_score = np.min(scores) if scores else 0
        print(f'  数字 {digit}: {len(templates)} 个模板, 平均自匹配分数={avg_score:.3f}, 最低={min_score:.3f}')

# 检查保存的数字裁剪区域
print('\n检查数字裁剪区域:')
for name in sorted(screenshots.glob('digit_stamina1_full_numbers_*.png')):
    img = _imread(str(name), cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        char, conf = template_lib.recognize(img)
        print(f'  {name.name} ({w}x{h}): 识别为 "{char}" (置信度: {conf:.3f})')

# 测试 stamina1 的各个组成部分
print('\n测试 stamina1 图像:')
name = 'stamina1_full_numbers.png'
region = _imread(str(screenshots / name), cv2.IMREAD_COLOR)
h, w = region.shape[:2]

# 手动分割成单个数字
# 根据之前的分析，应该是 "1050/1100" 这样的格式
# 让我手动提取每个数字并测试

# 保存整个区域以便分析
cv2.imwrite(str(screenshots / 'test_region.png'), region)

# 手动分析
hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

# 膨胀一下，连接断裂的部分
kernel = np.ones((2, 2), np.uint8)
white_mask = cv2.dilate(white_mask, kernel, iterations=1)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

print(f'  连通域数量: {num_labels - 1}')

# 显示每个连通域
for lid in range(1, num_labels):
    x = stats[lid, cv2.CC_STAT_LEFT]
    y = stats[lid, cv2.CC_STAT_TOP]
    ww = stats[lid, cv2.CC_STAT_WIDTH]
    hh = stats[lid, cv2.CC_STAT_HEIGHT]
    area = stats[lid, cv2.CC_STAT_AREA]
    
    # 提取这个区域
    pad = 3
    dx1, dy1 = max(0, x - pad), max(0, y - pad)
    dx2, dy2 = min(w, x + ww + pad), min(h, y + hh + pad)
    digit_img = region[dy1:dy2, dx1:dx2]
    
    char, conf = template_lib.recognize(digit_img)
    
    # 保存这个区域
    cv2.imwrite(str(screenshots / f'comp_{lid}.png'), digit_img)
    
    print(f'    [{lid}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, area={area:4d} => "{char}" (conf={conf:.3f})')

print('\n完成!')