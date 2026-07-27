import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER, MAX_DIGIT_AREA

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

# 详细分析分割过程
for name in ['stamina1_full_numbers.png', 'stamina2_full_numbers.png']:
    path = screenshots / name
    if path.exists():
        region = _imread(str(path), cv2.IMREAD_COLOR)
        if region is not None:
            h, w = region.shape[:2]
            print(f'\n{"="*60}')
            print(f'分析分割过程: {name} ({w}x{h})')
            print(f'{"="*60}')
            
            # HSV白色分割
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
            
            kernel = np.ones((1, 1), np.uint8)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # 提取原始区域
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
            
            all_regions = []
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
                
                all_regions.append((x, y, ww, hh, digit_img, aspect))
            
            print(f'\n原始连通域 (共 {len(all_regions)} 个):')
            for i, (x, y, ww, hh, img, aspect) in enumerate(all_regions):
                area = ww * hh
                is_large = (ww > hh * 1.5 and ww > 25) or area > 1500
                print(f'  [{i}] x={x:3d}, y={y:3d}, w={ww:3d}, h={hh:3d}, area={area:4d}, aspect={aspect:.2f} {"[待分割]" if is_large else ""}')
            
            # 尝试分割
            print(f'\n尝试分割大区域:')
            for i, (x, y, ww, hh, img, aspect) in enumerate(all_regions):
                area = ww * hh
                is_large = (ww > hh * 1.5 and ww > 25) or area > 1500
                
                if is_large:
                    print(f'\n  分割 [{i}] (x={x}, y={y}, w={ww}, h={hh}):')
                    
                    # 尝试列分割
                    sub_regions = reader._split_by_columns(x, y, ww, hh, region)
                    print(f'    列分割结果: {len(sub_regions)} 个子区域')
                    
                    if len(sub_regions) < 2:
                        print(f'    尝试分水岭分割...')
                        sub_regions2 = reader._split_region_watershed(x, y, ww, hh, region)
                        print(f'    分水岭结果: {len(sub_regions2)} 个子区域')
                        
                        if sub_regions2:
                            for j, (sx, sy, sw, sh, simg, sa) in enumerate(sub_regions2):
                                print(f'      子[{j}] x={sx}, y={sy}, w={sw}, h={sh}, aspect={sa:.2f}')
                        else:
                            print(f'      分水岭也失败!')
                    else:
                        for j, (sx, sy, sw, sh, simg, sa) in enumerate(sub_regions):
                            char, conf = reader._template_lib.recognize(simg)
                            print(f'      子[{j}] x={sx}, y={sy}, w={sw}, h={sh}, aspect={sa:.2f} => "{char}" (conf={conf:.3f})')

print('\n完成!')