import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER, STAMINA_CONFIG

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

img = _imread(str(screenshots / 'full_screen_20260726_203459.png'), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
print(f"截图尺寸: {w}x{h}")

reader.height, reader.width = h, w

for stamina_key in ['stamina1', 'stamina2']:
    print(f"\n{'='*60}")
    print(f"处理: {stamina_key}")
    print(f"{'='*60}")
    
    config = STAMINA_CONFIG[stamina_key]
    
    bar_region = reader._locate_bar(img, stamina_key)
    if bar_region is None:
        bar_region = config.get('fallback_bar_roi')
        print(f"  模板匹配失败，使用回退 bar 区域: {bar_region}")
    else:
        bx, by, bw, bh = bar_region
        print(f"  模板匹配成功: {bar_region}")
    
    if bar_region:
        bx, by, bw, bh = bar_region
        
        bar_img = img[by:by+bh, bx:bx+bw]
        cv2.imwrite(str(screenshots / f'debug_{stamina_key}_bar.png'), bar_img)
        print(f"  保存 bar 图像 ({bar_img.shape[1]}x{bar_img.shape[0]})")
        
        ox, oy, ow, oh = config['number_relative_offset']
        nx = bx + int(bw * ox)
        ny = by + int(bh * oy)
        nw = int(bw * ow)
        nh = int(bh * oh)
        
        print(f"  计算数字区域(相对bar): offset=({ox},{oy},{ow},{oh}) => 绝对坐标 ({nx},{ny},{nw},{nh})")
        
        number_region = img[ny:ny+nh, nx:nx+nw]
        cv2.imwrite(str(screenshots / f'debug_{stamina_key}_numbers.png'), number_region)
        print(f"  保存数字区域 ({number_region.shape[1]}x{number_region.shape[0]})")
        
        digit_regions = reader._extract_digit_regions(number_region)
        print(f"  提取到 {len(digit_regions)} 个数字区域:")
        for i, (dx, dy, dw, dh, dimg, da) in enumerate(digit_regions):
            char, conf = reader._template_lib.recognize(dimg)
            print(f"    [{i}] x={dx}, y={dy}, w={dw}, h={dh}, aspect={da:.3f}, area={dw*dh}, 识别='{char}' (conf={conf:.3f})")
            
            cv2.imwrite(str(screenshots / f'debug_{stamina_key}_digit_{i}.png'), dimg)
        
        result_text = reader._read_numbers_from_region(number_region)
        current, max_val = reader._parse_stamina_text(result_text)
        print(f"  最终识别: '{result_text}' => {current}/{max_val}")

print("\n完成!")