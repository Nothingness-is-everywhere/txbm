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

digit_regions = reader._extract_digit_regions(number_region)

for i, (dx, dy, dw, dh, dimg, da) in enumerate(digit_regions):
    area = dw * dh
    is_slash = (da < 0.12 and dh >= 15) or (dw < 5 and dh >= 20)
    is_very_large = (dw > dh * 1.3 and dw > 25) or area > 2200
    
    print(f"\n--- 区域 [{i}] ({dw}x{dh}, area={area}, aspect={da:.3f}) ---")
    print(f"  is_slash={is_slash}, is_very_large={is_very_large}")
    
    if is_very_large and not is_slash:
        print(f"  尝试分割...")
        
        sub_regions = reader._split_by_columns(dx, dy, dw, dh, number_region)
        print(f"  _split_by_columns 返回 {len(sub_regions)} 个子区域:")
        
        valid_count = 0
        for j, (sx, sy, sw, sh, simg, sa) in enumerate(sub_regions):
            sub_area = sw * sh
            char, conf = template_lib.recognize(simg)
            is_valid = sub_area >= 30 and sw >= 5 and conf >= 0.45
            print(f"    [{j}] x={sx}, y={sy}, w={sw}, h={sh}, area={sub_area}, 识别='{char}' (conf={conf:.3f}), 有效={is_valid}")
            if is_valid:
                valid_count += 1
        
        print(f"  有效子区域数: {valid_count}")
        
        if valid_count >= 2:
            print(f"  => 分割成功，使用 {valid_count} 个子区域")
        else:
            char, conf = template_lib.recognize(dimg)
            print(f"  => 分割失败，保留原区域，识别='{char}' (conf={conf:.3f})")

print("\n完成!")