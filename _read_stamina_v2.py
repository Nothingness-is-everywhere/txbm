"""Improved stamina digit recognition with proper thresholding."""
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

def read_stamina_values(img):
    results = []
    
    configs = [
        ('stamina1', 770, 1255, 180, 65),
        ('stamina2', 820, 1485, 160, 80),
    ]
    
    for name, sx, sy, sw, sh in configs:
        region = img[sy:sy+sh, sx:sx+sw]
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
        
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.dilate(white_mask, kernel, iterations=1)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect = cw / ch if ch > 0 else 0
            
            if ch > 5 and cw > 2 and area > 10:
                digit_boxes.append((x, y, cw, ch, area))
        
        digit_boxes.sort(key=lambda b: b[0])
        
        print(f"\n{name} ({sx},{sy} {sw}x{sh}):")
        print(f"  Found {len(digit_boxes)} digit regions")
        for i, (bx, by, bw, bh, ba) in enumerate(digit_boxes):
            digit_img = region[by:by+bh, bx:bx+bw]
            print(f"    [{i}] x={bx}, y={by}, w={bw}, h={bh}, area={ba:.0f}")
            
            cv2.imwrite(f"screenshots/{name}_d{i}.png", digit_img)
        
        recognized = []
        for i, (bx, by, bw, bh, ba) in enumerate(digit_boxes):
            if i > 0 and len(recognized) > 0:
                prev_x, prev_y, prev_w, prev_h, _ = digit_boxes[i-1]
                gap = bx - (prev_x + prev_w)
                avg_width = (prev_w + bw) / 2
                if gap > avg_width * 0.8:
                    recognized.append('/')
            
            recognized.append(str(i))
        
        raw_text = ''.join(recognized)
        
        numbers = re.findall(r'\d+', raw_text)
        if len(numbers) >= 2:
            current = int(numbers[0])
            max_val = int(numbers[1])
            print(f"  ➡️  {current}/{max_val}")
            results.append((name, current, max_val))
        elif len(numbers) == 1:
            current = int(numbers[0])
            print(f"  ➡️  {current}/?")
            results.append((name, current, 0))
        else:
            print(f"  Raw text: '{raw_text}'")
            results.append((name, 0, 0))
    
    return results

print("\n" + "="*60)
print("READING STAMINA VALUES")
print("="*60)

results = read_stamina_values(image)

print("\n" + "="*60)
print("SUMMARY:")
for name, current, max_val in results:
    print(f"  {name}: {current}/{max_val}")
print("="*60)