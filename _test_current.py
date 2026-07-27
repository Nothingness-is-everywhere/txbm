"""验证改进后的分割和识别"""
import sys
sys.modules['easyocr'] = None

from ok.automation.stamina_reader import StaminaReader

reader = StaminaReader()
reader._easyocr_reader = None

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    print(f"\n{'='*60}")
    print(f"{stamina_type}:")
    print(f"{'='*60}")
    
    import cv2
    import numpy as np
    from ok.automation.stamina_reader import read_image
    
    img = read_image(img_path)
    if img is None:
        print("无法读取图片")
        continue
    
    digits = reader._segment_digits(img)
    print(f"Segmented {len(digits)} digit(s):")
    
    recognized = []
    for x, y, w, h, roi in digits:
        digit, conf = reader.classifier.classify(roi)
        recognized.append(digit)
        print(f"  x={x:3d}, y={y:3d}, w={w:2d}, h={h:2d} -> '{digit}' (conf={conf:.2f})")
    
    text = "".join(recognized)
    parsed = reader._parse_stamina_text(text)
    print(f"\nRecognized: '{text}'")
    print(f"Parsed: {parsed}")