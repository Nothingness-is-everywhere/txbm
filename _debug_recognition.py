"""查看图像像素分布以确定正确的分割方法"""
import sys
sys.modules['easyocr'] = None

import cv2
import numpy as np
from ok.automation.stamina_reader import read_image

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        print(f"{stamina_type}: 无法读取图片")
        continue

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"\n{'='*60}")
    print(f"{stamina_type}: shape={img.shape}")
    print(f"{'='*60}")

    print(f"Gray stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
    print(f"Percentiles: 5%={np.percentile(gray,5):.0f}, 25%={np.percentile(gray,25):.0f}, "
          f"50%={np.percentile(gray,50):.0f}, 75%={np.percentile(gray,75):.0f}, "
          f"95%={np.percentile(gray,95):.0f}")

    # Try different thresholds
    for thresh in [120, 150, 180, 200]:
        _, mask_bin = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        _, mask_inv = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        
        contours_bin, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_inv, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_bin = sum(1 for c in contours_bin if cv2.boundingRect(c)[2] >= 3 and cv2.boundingRect(c)[3] >= 10)
        valid_inv = sum(1 for c in contours_inv if cv2.boundingRect(c)[2] >= 3 and cv2.boundingRect(c)[3] >= 10)
        
        print(f"\n  Threshold={thresh}:")
        print(f"    BINARY:  {len(contours_bin)} contours, {valid_bin} valid (w>=3,h>=10)")
        print(f"    INV:     {len(contours_inv)} contours, {valid_inv} valid (w>=3,h>=10)")

    # Try Otsu
    _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_otsu, _ = cv2.findContours(mask_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_otsu = sum(1 for c in contours_otsu if cv2.boundingRect(c)[2] >= 3 and cv2.boundingRect(c)[3] >= 10)
    print(f"\n  Otsu: {len(contours_otsu)} contours, {valid_otsu} valid")
    
    # Save all masks for comparison
    cv2.imwrite(f"debug_output/{stamina_type}_gray.png", gray)
    _, mask_180 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"debug_output/{stamina_type}_mask_180.png", mask_180)
    _, mask_180_inv = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f"debug_output/{stamina_type}_mask_180_inv.png", mask_180_inv)
    cv2.imwrite(f"debug_output/{stamina_type}_mask_otsu.png", mask_otsu)
    
    print(f"  Saved comparison masks to debug_output/")