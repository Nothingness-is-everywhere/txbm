"""Precise stamina digit extraction and recognition."""
import sys
import re
import cv2
import numpy as np
import adbutils

adb = adbutils.AdbClient()
d = adb.device('127.0.0.1:16384')

print("Capturing screen...")
png_bytes = d.shell("screencap -p", encoding=None, timeout=10)
image_data = np.frombuffer(png_bytes, dtype=np.uint8)
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
h, w = image.shape[:2]
print(f"Screen: {w}x{h}")

def extract_stamina_numbers(image, sx, sy, sw, sh):
    region = image[sy:sy+sh, sx:sx+sw]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    print(f"\nRegion ({sx},{sy}) {sw}x{sh}:")
    print(f"  Gray range: {gray.min()}-{gray.max()}, mean={gray.mean():.1f}")
    
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    kernel_small = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel_small, iterations=1)
    binary = cv2.erode(binary, kernel_small, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = cw / ch if ch > 0 else 0
        
        if 5 < ch < 50 and 3 < cw < 40 and area > 15 and aspect > 0.15:
            digit_boxes.append((x, y, cw, ch, area))
    
    digit_boxes.sort(key=lambda b: b[0])
    
    print(f"  Found {len(digit_boxes)} digit-like boxes:")
    for i, (bx, by, bw, bh, ba) in enumerate(digit_boxes):
        print(f"    [{i}] local({bx},{by}) {bw}x{bh} area={ba:.0f}")
        digit_img = region[by:by+bh, bx:bx+bw]
        digit_gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
        print(f"        gray range: {digit_gray.min()}-{digit_gray.max()}, white_ratio={np.mean(digit_gray > 200):.2f}")
    
    return digit_boxes

stamina1_region = extract_stamina_numbers(image, 770, 1255, 180, 65)
stamina2_region = extract_stamina_numbers(image, 820, 1485, 160, 80)

print("\n" + "="*60)
print("CURRENT STAMINA VALUES (from visual inspection):")
print("  出征 (Expedition): 84/150")
print("  调教 (Tame): 1/50")
print("="*60)

print("\nSaving individual digit crops for analysis...")

for i, (bx, by, bw, bh, ba) in enumerate(stamina1_region):
    crop = image[1255+by:1255+by+bh, 770+bx:770+bx+bw]
    cv2.imwrite(f"screenshots/s1_d{i}.png", crop)

for i, (bx, by, bw, bh, ba) in enumerate(stamina2_region):
    crop = image[1485+by:1485+by+bh, 820+bx:820+bx+bw]
    cv2.imwrite(f"screenshots/s2_d{i}.png", crop)

print("\nSaved digit crops:")
print("  s1_d*.png - stamina1 digits")
print("  s2_d*.png - stamina2 digits")