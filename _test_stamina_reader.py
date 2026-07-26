"""Test the updated stamina reader with simple OCR fallback."""
import sys
import re
import cv2
import numpy as np
import adbutils

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
print(f"Screen: {w}x{h}")

STAMINA1_TPL = cv2.imread('templates/stamina1_bar.png', cv2.IMREAD_COLOR)
STAMINA2_TPL = cv2.imread('templates/stamina2_bar.png', cv2.IMREAD_COLOR)

STAMINA1_ROI = (0.65, 0.62, 0.95, 0.72)
STAMINA2_ROI = (0.70, 0.74, 0.95, 0.83)

STAMINA_THRESHOLD = 0.65

STAMINA1_NUMBERS_OFFSET = (0.02, 0.10, 0.85, 0.45)
STAMINA2_NUMBERS_OFFSET = (0.05, 0.15, 0.80, 0.45)

def find_bar(screen, template, roi, name):
    rx1, ry1, rx2, ry2 = roi
    roi_x1 = int(w * rx1)
    roi_y1 = int(h * ry1)
    roi_x2 = int(w * rx2)
    roi_y2 = int(h * ry2)
    
    img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    roi_region = img_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi_region.shape[0] < tpl_gray.shape[0]:
        roi_region = img_gray[roi_y1:min(roi_y2, roi_y1+tpl_gray.shape[0]), roi_x1:roi_x2]
    
    result = cv2.matchTemplate(roi_region, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= STAMINA_THRESHOLD:
        tpl_h, tpl_w = tpl_gray.shape
        return (roi_x1 + max_loc[0], roi_y1 + max_loc[1], tpl_w, tpl_h, max_val)
    return None

def read_numbers_simple(number_region):
    gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if ch > 8 and cw > 3 and area > 10:
            digit_regions.append((x, y, cw, ch, area))
    
    print(f"  Found {len(digit_regions)} digit regions")
    for i, (dx, dy, dw, dh, da) in enumerate(digit_regions):
        print(f"    [{i}] x={dx}, y={dy}, w={dw}, h={dh}, area={da:.0f}")
    
    if len(digit_regions) < 2:
        return None
    
    recognized = []
    for i, (dx, dy, dw, dh, da) in enumerate(digit_regions):
        digit_roi = number_region[dy:dy+dh, dx:dx+dw]
        digit_gray = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
        
        avg_val = np.mean(digit_gray)
        white_ratio = np.sum(digit_gray > 200) / digit_gray.size
        
        if i > 0 and len(recognized) > 0:
            prev_x, prev_y, prev_w, prev_h, _ = digit_regions[i-1]
            gap = dx - (prev_x + prev_w)
            if gap > dw * 0.5:
                recognized.append('/')
        
        recognized.append(str(i))
    
    return ''.join(recognized)

print("\n" + "="*60)
print("TESTING STAMINA READER")
print("="*60)

for name, tpl, roi, offset in [
    ('STAMINA 1 (出征)', STAMINA1_TPL, STAMINA1_ROI, STAMINA1_NUMBERS_OFFSET),
    ('STAMINA 2 (调教)', STAMINA2_TPL, STAMINA2_ROI, STAMINA2_NUMBERS_OFFSET),
]:
    result = find_bar(image, tpl, roi, name)
    
    if result is None:
        print(f"\n{name}: NOT FOUND")
        continue
    
    x, y, tw, th, conf = result
    print(f"\n{name}: FOUND at ({x},{y}) conf={conf:.3f}")
    
    ox1, oy1, ow, oh = offset
    num_x = x + int(tw * ox1)
    num_y = y + int(th * oy1)
    num_w = int(tw * ow)
    num_h = int(th * oh)
    
    print(f"  Numbers region: ({num_x},{num_y}) {num_w}x{num_h}")
    
    number_region = image[num_y:num_y+num_h, num_x:num_x+num_w]
    
    cv2.imwrite(f"screenshots/test_{name.replace(' ', '_')}.png", number_region)
    
    raw_result = read_numbers_simple(number_region)
    print(f"  Raw recognition: '{raw_result}'")
    
    if raw_result:
        numbers = re.findall(r'\d+', raw_result)
        if len(numbers) >= 2:
            print(f"  ➡️  {numbers[0]}/{numbers[1]}")
        elif len(numbers) == 1:
            print(f"  ➡️  {numbers[0]}/?")

print("\n" + "="*60)
print("NOTE: Simple recognition uses region counting")
print("      For accurate OCR, EasyOCR models need to be downloaded")
print("="*60)