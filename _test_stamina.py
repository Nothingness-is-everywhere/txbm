"""Test stamina reader functionality."""
import sys
from pathlib import Path

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

import ok.automation.stamina_reader as sr
sr._ENGINE = None

from ok.automation.stamina_reader import StaminaReader, StaminaValue, get_stamina, init_stamina

def main():
    print("=" * 60)
    print("Stamina Reader 测试")
    print("=" * 60)
    
    reader = StaminaReader()
    print(f"\n1. StaminaReader 初始化成功")
    print(f"   - EasyOCR: {'可用' if reader._easyocr_reader else '不可用(使用模板匹配)'}")
    print(f"   - 模板数量: {len(reader.digit_recognizer.digit_templates)}")
    
    print("\n2. 测试截图...")
    img = reader._capture_screen()
    if img is not None:
        h, w = img.shape[:2]
        print(f"   - 截图成功: {w}x{h}")
    else:
        print("   - 截图失败!")
        return 1
    
    print("\n3. 裁剪体力值区域...")
    exp_region = reader._crop_region(img, 0.738, 0.850, 0.728, 0.762)
    train_region = reader._crop_region(img, 0.800, 0.920, 0.822, 0.862)
    print(f"   - 出征区域: {exp_region.shape[1]}x{exp_region.shape[0]}")
    print(f"   - 调教区域: {train_region.shape[1]}x{train_region.shape[0]}")
    
    out_dir = PROJECT / 'screenshots' / 'stamina_test'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    import cv2
    for name, region in [('expedition', exp_region), ('training', train_region)]:
        ok, buf = cv2.imencode('.png', region)
        if ok:
            filepath = out_dir / f'{name}_region.png'
            with open(str(filepath), 'wb') as f:
                f.write(buf.tobytes())
            print(f"   - 保存: {filepath}")
    
    print("\n4. 测试模板匹配识别...")
    exp_text = reader._recognize_with_templates(exp_region)
    train_text = reader._recognize_with_templates(train_region)
    print(f"   - 出征识别: '{exp_text}'")
    print(f"   - 调教识别: '{train_text}'")
    
    print("\n5. 测试完整读取流程...")
    values = reader.read_all_stamina(force_refresh=True)
    
    for stamina_type, value in values.items():
        if value:
            print(f"\n   {stamina_type}:")
            print(f"     - 当前: {value.current}")
            print(f"     - 最大: {value.max_val}")
            print(f"     - 比例: {value.ratio:.2%}")
            print(f"     - 满体力: {value.is_full}")
            print(f"     - 空体力: {value.is_empty}")
        else:
            print(f"\n   {stamina_type}: 识别失败")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
