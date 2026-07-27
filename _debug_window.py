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
    is_very_large = (dw > dh * 1.3 and dw > 25) or area > 2200
    
    print(f"\n--- 区域 [{i}] ({dw}x{dh}, area={area}, aspect={da:.3f}) ---")
    print(f"  is_very_large={is_very_large}")
    
    if is_very_large:
        window_subs = reader._split_by_sliding_window(dx, dy, dw, dh, number_region)
        print(f"  滑动窗口返回 {len(window_subs)} 个子区域:")
        for j, (sx, sy, sw, sh, simg, sa) in enumerate(window_subs):
            char, conf = template_lib.recognize(simg)
            print(f"    [{j}] x={sx}, w={sw}, 识别='{char}' (conf={conf:.3f})")
    else:
        char, conf = template_lib.recognize(dimg)
        print(f"  直接识别: '{char}' (conf={conf:.3f})")

print("\n完成!")