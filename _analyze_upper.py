"""Search more broadly for stamina regions - look at upper portion of screen."""
import cv2
import numpy as np

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

print(f"Image size: {w}x{h}")

upper_region = image[0:500, :, :]

gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)

print("\nAnalyzing upper region (y: 0-500)...")

for threshold_name, thresh_val in [('dark', 50), ('very-dark', 30), ('black', 15)]:
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect_area = cw * ch
        fill = area / rect_area if rect_area > 0 else 0
        
        if cw > 30 and ch > 10 and fill > 0.5 and area > 300:
            boxes.append((x, y, cw, ch, area, fill))
    
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    print(f"\n  Threshold {threshold_name} (<{thresh_val}): found {len(boxes)} regions")
    for i, (bx, by, bw, bh, ba, bf) in enumerate(boxes[:15]):
        print(f"    [{i}] ({bx},{by}) {bw}x{bh} area={ba:.0f} fill={bf:.2f}")

result = image.copy()

_, binary_all = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
kernel_big = np.ones((7, 7), np.uint8)
binary_all = cv2.morphologyEx(binary_all, cv2.MORPH_CLOSE, kernel_big)
binary_all = cv2.morphologyEx(binary_all, cv2.MORPH_OPEN, kernel_big)

contours_all, _ = cv2.findContours(binary_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours_all:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = cw * ch
    fill = area / rect_area if rect_area > 0 else 0
    
    if cw > 50 and ch > 20 and fill > 0.4 and area > 1000 and y < 500:
        cv2.rectangle(result, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
        cv2.putText(result, f'({x},{y}) {cw}x{ch}', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite('screenshots/upper_regions_analysis.png', result)
print('\nSaved: screenshots/upper_regions_analysis.png')

print('\n' + '='*60)
print('Saving crops of potential stamina regions...')
print('='*60)

crops_dir = 'screenshots/stamina_crops'
import os
os.makedirs(crops_dir, exist_ok=True)

key_regions = [
    ('top_left', 0, 150, 200, 350),
    ('top_right', 700, 150, 380, 350),
    ('below_buttons_left', 0, 280, 250, 200),
    ('below_buttons_right', 700, 280, 380, 200),
    ('center_top', 200, 100, 680, 400),
    ('full_upper', 0, 100, 1080, 400),
]

for name, x, y, w, h in key_regions:
    crop = image[y:y+h, x:x+w]
    cv2.imwrite(f'{crops_dir}/{name}.png', crop)
    print(f'  Saved {name}: ({x},{y}) {w}x{h}')