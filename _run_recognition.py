import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread

reader = StaminaReader()
screenshots = PROJECT / 'screenshots'

# 使用现有的截图进行测试
screen_path = screenshots / 'full_screen_20260726_203459.png'

if screen_path.exists():
    screen = _imread(str(screen_path), cv2.IMREAD_COLOR)
    if screen is not None:
        reader.height, reader.width = screen.shape[:2]
        print(f'截图分辨率: {reader.width}x{reader.height}')
        
        # 识别体力值
        s1 = reader.read_stamina1(screen)
        s2 = reader.read_stamina2(screen)
        
        print(f'')
        print(f'===== 全图识别结果 =====')
        print(f'体力1 (出征): {s1.current}/{s1.max}')
        print(f'体力2 (调教): {s2.current}/{s2.max}')
        print(f'====================')
    else:
        print('无法读取截图')
else:
    print(f'截图文件不存在: {screen_path}')

# 直接测试完整数字区域
print('')
print('===== 直接数字区域测试 =====')
for name in ['stamina1_full_numbers.png', 'stamina2_full_numbers.png']:
    path = screenshots / name
    if path.exists():
        region = _imread(str(path), cv2.IMREAD_COLOR)
        if region is not None:
            h, w = region.shape[:2]
            print(f'\n--- {name} ({w}x{h}) ---')
            
            result = reader._read_numbers_from_region(region)
            current, max_val = reader._parse_stamina_text(result)
            
            print(f'识别文本: "{result}"')
            print(f'解析结果: {current}/{max_val}')