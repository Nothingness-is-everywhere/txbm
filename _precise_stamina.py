"""Precisely crop stamina values from main screen."""
import sys
import time
from pathlib import Path

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

import cv2
import numpy as np

ADB_SERIAL = "emulator-5554"

def save_image(img, filepath):
    """Save image using imencode to avoid Chinese path issues."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ext = filepath.suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(str(filepath), 'wb') as f:
            f.write(buf.tobytes())
        return True
    return False

def capture_screen():
    """Capture screen via ADB."""
    import adbutils
    adb = adbutils.AdbClient()
    device = adb.device(ADB_SERIAL)
    img = device.screenshot()
    
    img_array = np.array(img)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:
            return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_array

def main():
    print("="*60)
    print("精确裁剪体力值")
    print("="*60)
    
    out_dir = PROJECT / 'screenshots' / 'stamina_crops'
    
    print("\n1. 获取当前截图...")
    img = capture_screen()
    h, w = img.shape[:2]
    print(f"屏幕尺寸: {w}x{h}")
    save_image(img, out_dir / 'main_full.png')
    
    result = img.copy()
    
    stamina_crops = [
        {
            'key': 'expedition',
            'name': '出征体力值',
            'y_ratio_start': 0.71,
            'y_ratio_end': 0.745,
            'x_ratio_start': 0.72,
            'x_ratio_end': 0.82,
        },
        {
            'key': 'training',
            'name': '调教体力值',
            'y_ratio_start': 0.79,
            'y_ratio_end': 0.825,
            'x_ratio_start': 0.76,
            'x_ratio_end': 0.84,
        },
    ]
    
    print("\n" + "="*60)
    print("裁剪体力值数字区域")
    print("="*60)
    
    for crop in stamina_crops:
        x1 = int(w * crop['x_ratio_start'])
        y1 = int(h * crop['y_ratio_start'])
        x2 = int(w * crop['x_ratio_end'])
        y2 = int(h * crop['y_ratio_end'])
        
        cropped = img[y1:y2, x1:x2]
        
        crop_path = out_dir / f"{crop['key']}_value.png"
        save_image(cropped, crop_path)
        
        print(f"\n{crop['name']}:")
        print(f"  区域: x={x1}-{x2}, y={y1}-{y2}")
        print(f"  尺寸: {cropped.shape[1]}x{cropped.shape[0]}")
        print(f"  保存: {crop_path}")
        
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(result, crop['name'], (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    annotated_path = out_dir / 'stamina_annotated.png'
    save_image(result, annotated_path)
    print(f"\n带标注截图: {annotated_path}")
    
    print("\n完成!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
