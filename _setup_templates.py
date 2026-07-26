"""
Extract and label digit templates for stamina recognition.

This script:
1. Captures the current screen
2. Extracts individual digit regions
3. Saves them as labeled templates for the stamina reader
"""
import os
import sys
import cv2
import numpy as np
import adbutils
from pathlib import Path

adb = adbutils.AdbClient()
d = adb.device('127.0.0.1:16384')

print("Capturing screen...")
png_bytes = d.shell("screencap -p", encoding=None, timeout=10)
if not png_bytes:
    print("Failed to capture screen")
    sys.exit(1)

image_data = np.frombuffer(png_bytes, dtype=np.uint8)
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
h, w = image.shape[:2]

DIGITS_DIR = Path("templates/digit_templates")
DIGITS_DIR.mkdir(parents=True, exist_ok=True)

STAMINA1_ROI = (0.02, 0.04, 0.32, 0.10)
STAMINA2_ROI = (0.58, 0.04, 0.88, 0.10)

def find_bar(screen, template_path, roi):
    tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if tpl is None:
        return None
    
    rx1, ry1, rx2, ry2 = roi
    roi_x1 = int(w * rx1)
    roi_y1 = int(h * ry1)
    roi_x2 = int(w * rx2)
    roi_y2 = int(h * ry2)
    
    img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    
    roi_region = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    result = cv2.matchTemplate(roi_region, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= 0.70:
        tpl_h, tpl_w = tpl_gray.shape
        return (roi_x1 + max_loc[0], roi_y1 + max_loc[1], tpl_w, tpl_h)
    return None

def extract_digits(number_region):
    gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if ch >= 25 and 8 <= cw <= 45 and area >= 100:
            aspect = cw / ch
            if 0.15 <= aspect <= 1.2:
                pad = 2
                dx1 = max(0, x - pad)
                dy1 = max(0, y - pad)
                dx2 = min(number_region.shape[1], x + cw + pad)
                dy2 = min(number_region.shape[0], y + ch + pad)
                
                digit_img = number_region[dy1:dy2, dx1:dx2]
                digit_regions.append((x, y, cw, ch, digit_img))
    
    return digit_regions

print("\n" + "="*60)
print("Extracting digit templates...")
print("="*60)

templates_info = []

for name, tpl_path, roi in [
    ('stamina1', 'templates/stamina1_bar.png', STAMINA1_ROI),
    ('stamina2', 'templates/stamina2_bar.png', STAMINA2_ROI),
]:
    result = find_bar(image, tpl_path, roi)
    
    if result is None:
        print(f"{name}: NOT FOUND")
        continue
    
    x, y, tw, th = result
    print(f"\n{name}: FOUND at ({x},{y})")
    
    number_y = y + int(th * 0.28)
    number_h = int(th * 0.70)
    number_region = image[number_y:number_y+number_h, x:x+tw]
    
    cv2.imwrite(str(DIGITS_DIR / f"{name}_full_numbers.png"), number_region)
    
    digits = extract_digits(number_region)
    print(f"  Found {len(digits)} digit regions")
    
    for i, (dx, dy, dw, dh, digit_img) in enumerate(digits):
        h2, w2 = digit_img.shape[:2]
        target_h = 40
        scale = target_h / h2
        target_w = int(w2 * scale)
        resized = cv2.resize(digit_img, (target_w, target_h))
        
        filename = f"{name}_d{i}.png"
        filepath = str(DIGITS_DIR / filename)
        cv2.imwrite(filepath, resized)
        
        templates_info.append({
            'file': filename,
            'label': name,
            'index': i,
            'position': f"x={dx}, y={dy}, w={dw}, h={dh}"
        })
        
        print(f"    [{i}] Saved as {filename} ({target_w}x{target_h})")

print("\n" + "="*60)
print("Templates saved. Please label them correctly.")
print("Based on visual inspection:")
print()
print("stamina1 digits (70/60):")
print("  stamina1_d0.png = '7'")
print("  stamina1_d1.png = '0'")
print("  stamina1_d2.png = '6'")
print("  stamina1_d3.png = '0'")
print()
print("stamina2 digits (120/150):")
print("  stamina2_d0.png = '1'")
print("  stamina2_d1.png = '2'")
print("  stamina2_d2.png = '0'")
print("  stamina2_d3.png = '1'")
print("  stamina2_d4.png = '5'")
print("  stamina2_d5.png = '0'")
print("="*60)

print("\nNow run: python -m ok.automation.stamina_reader")
print("to test the updated reader with template matching")