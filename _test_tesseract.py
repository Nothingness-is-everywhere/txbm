"""测试pytesseract识别体力值"""
import sys
import cv2
import numpy as np
import pytesseract
from ok.automation.stamina_reader import read_image

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        print(f"{stamina_type}: 无法读取图片")
        continue

    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"{stamina_type}: shape={img.shape}")
    print(f"{'='*60}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try pytesseract with different configs
    for psm in [7, 8, 10, 13]:
        try:
            text = pytesseract.image_to_string(
                gray, 
                config=f'--psm {psm} -c tessedit_char_whitelist=0123456789/'
            )
            print(f"  PSM {psm}: '{text.strip()}'")
        except Exception as e:
            print(f"  PSM {psm}: Error - {e}")
    
    # Try with preprocessing
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text_bin = pytesseract.image_to_string(
        binary, config='--psm 7 -c tessedit_char_whitelist=0123456789/'
    )
    print(f"  Binary (thresh=150): '{text_bin.strip()}'")
    
    _, binary_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text_inv = pytesseract.image_to_string(
        binary_inv, config='--psm 7 -c tessedit_char_whitelist=0123456789/'
    )
    print(f"  Binary Inv (thresh=150): '{text_inv.strip()}'")
    
    # Resize and try
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    text_resized = pytesseract.image_to_string(
        resized, config='--psm 7 -c tessedit_char_whitelist=0123456789/'
    )
    print(f"  Resized 3x: '{text_resized.strip()}'")