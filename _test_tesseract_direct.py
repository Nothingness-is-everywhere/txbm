"""Test Tesseract directly on cropped stamina images"""
import cv2
import numpy as np
from pathlib import Path

def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

print("Testing Tesseract...")

try:
    import pytesseract
    print("pytesseract imported successfully")
    
    # Check if tesseract binary is available
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except Exception as e:
        print(f"Tesseract binary check: {e}")
        
except Exception as e:
    print(f"pytesseract import failed: {e}")
    import sys
    sys.exit(1)

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        print(f"\n{stamina_type}: cannot read image")
        continue
    
    print(f"\n{'='*60}")
    print(f"{stamina_type}: {img.shape}")
    print(f"{'='*60}")
    
    # Try different configs
    configs = [
        ('--psm 7 -c tessedit_char_whitelist=0123456789/', 'single line, digits only'),
        ('--psm 6 -c tessedit_char_whitelist=0123456789/', 'uniform block, digits only'),
        ('--psm 13 -c tessedit_char_whitelist=0123456789/', 'raw line, digits only'),
        ('--psm 7', 'single line, default chars'),
    ]
    
    for config, desc in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            print(f"\n  [{desc}]: '{text}'")
        except Exception as e:
            print(f"\n  [{desc}]: Error: {e}")
    
    # Try with preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try different thresholds
    for thresh_val in [100, 120, 140, 160, 180]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        try:
            text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789/').strip()
            if text and text != '':
                print(f"\n  [thresh={thresh_val}]: '{text}'")
        except Exception as e:
            pass

print("\nDone!")