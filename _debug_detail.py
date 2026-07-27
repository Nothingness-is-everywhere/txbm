import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

# 详细分析 stamina1_full_numbers.png
name = 'stamina1_full_numbers.png'
path = screenshots / name

region = _imread(str(path), cv2.IMREAD_COLOR)
h, w = region.shape[:2]
print(f'分析: {name} ({w}x{h})')

# 提取数字区域
digit_regions = reader._extract_digit_regions(region)
print(f'\n提取的数字区域 ({len(digit_regions)} 个):')

recognized_parts = []
for i, (x, y, ww, hh, digit_img, aspect) in enumerate(digit_regions):
    is_slash = False
    
    if aspect > 2.0 and hh >= 3:
        is_slash = True
    if aspect < 0.20 and hh >= 3 and ww >= 2:
        is_slash = True
    
    if i > 0:
        prev = digit_regions[i-1]
        gap = x - (prev[0] + prev[2])
        if gap > 0 and prev[3] > 0 and gap > prev[3] * 0.35 and gap > prev[2] * 0.4:
            is_slash = True
    
    if is_slash:
        recognized_parts.append('/')
        print(f'  [{i}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, aspect={aspect:.2f} => "/" (分隔符)')
    else:
        char, conf = reader._template_lib.recognize(digit_img)
        recognized_parts.append(char or '?')
        print(f'  [{i}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, aspect={aspect:.2f} => "{char}" (置信度: {conf:.3f})')

result = ''.join(recognized_parts)
current, max_val = reader._parse_stamina_text(result)
print(f'\n识别结果: "{result}"')
print(f'解析: {current}/{max_val}')

# 保存每个数字区域的图像
for i, (x, y, ww, hh, digit_img, aspect) in enumerate(digit_regions):
    cv2.imwrite(str(screenshots / f'digit_{name.replace(".png", "")}_{i}.png'), digit_img)

print(f'\n已保存数字区域图像')