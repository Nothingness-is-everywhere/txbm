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

# 加载 stamina1 图像
region = _imread(str(screenshots / 'stamina1_full_numbers.png'), cv2.IMREAD_COLOR)
h, w = region.shape[:2]

# 手动提取每个数字区域（根据之前的分析）
# 区域 [0] x=2, y=44, w=7, h=9 - 小区域
# 区域 [1] x=66, y=0, w=50, h=53 - 大区域（应该是 "10"）
# 区域 [2] x=123, y=0, w=19, h=53 - "3"
# 区域 [3] x=154, y=0, w=18, h=53 - "4"
# 区域 [4] x=240, y=0, w=3, h=53 - "/"

regions_info = [
    (2, 44, 7, 9, "small_noise"),
    (66, 0, 50, 53, "big_area"),
    (123, 0, 19, 53, "digit_3"),
    (154, 0, 18, 53, "digit_4"),
    (240, 0, 3, 53, "slash"),
]

print("=" * 60)
print("详细分析每个区域的识别过程")
print("=" * 60)

for rx, ry, rw, rh, name in regions_info:
    pad = 3
    dx1, dy1 = max(0, rx - pad), max(0, ry - pad)
    dx2, dy2 = min(w, rx + rw + pad), min(h, ry + rh + pad)
    digit_img = region[dy1:dy2, dx1:dx2]
    
    print(f"\n--- 区域 {name} ({rw}x{rh}) ---")
    
    # 保存原始裁剪
    cv2.imwrite(str(screenshots / f'raw_{name}.png'), digit_img)
    
    # 手动预处理
    processed = template_lib._preprocess_digit_for_matching(digit_img)
    
    if processed is not None:
        cv2.imwrite(str(screenshots / f'processed_{name}.png'), processed)
        print(f"  预处理后尺寸: {processed.shape[1]}x{processed.shape[0]}")
    
    # 识别
    char, conf = template_lib.recognize(digit_img)
    print(f"  识别结果: '{char}' (置信度: {conf:.3f})")
    
    # 显示最佳匹配的模板
    if char is not None and char in template_lib._templates:
        templates = template_lib._templates[char]
        print(f"  数字 '{char}' 共有 {len(templates)} 个模板")
        
        # 找出最佳匹配的模板
        best_tpl = None
        best_score = 0
        for tpl in templates:
            score = template_lib._compute_similarity(processed if processed is not None else digit_img, tpl)
            if score > best_score:
                best_score = score
                best_tpl = tpl
        
        if best_tpl is not None:
            cv2.imwrite(str(screenshots / f'best_tpl_{name}.png'), best_tpl)
            print(f"  最佳模板匹配分数: {best_score:.3f}")

print("\n完成!")