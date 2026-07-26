"""Look for stamina regions in middle and lower sections."""
import cv2
import numpy as np

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

print(f"Full screen: {w}x{h}")

crops_dir = 'screenshots/stamina_crops'

regions = [
    ('middle_left', 0, 400, 300, 400),
    ('middle_center', 250, 400, 600, 400),
    ('middle_right', 750, 400, 330, 400),
    ('lower_left', 0, 800, 300, 400),
    ('lower_center', 250, 800, 600, 400),
    ('lower_right', 750, 800, 330, 400),
    ('bottom', 0, 1200, 1080, 720),
    ('middle_full', 0, 400, 1080, 500),
    ('middle_band', 0, 600, 1080, 200),
]

import os
os.makedirs(crops_dir, exist_ok=True)

for name, x, y, rw, rh in regions:
    crop = image[y:y+rh, x:x+rw]
    cv2.imwrite(f'{crops_dir}/{name}.png', crop)
    print(f'Saved {name}: ({x},{y}) {rw}x{rh}')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((7, 7), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = image.copy()

boxes_found = []
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = cw * ch
    fill = area / rect_area if rect_area > 0 else 0
    
    if cw > 80 and ch > 30 and fill > 0.4 and area > 3000:
        boxes_found.append((x, y, cw, ch, fill))
        cv2.rectangle(result, (x, y), (x+cw, y+ch), (0, 255, 0), 3)
        label = f'({x},{y}) {cw}x{ch}'
        cv2.putText(result, label, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

boxes_found.sort(key=lambda b: (b[1], b[0]))

print(f'\nFound {len(boxes_found)} dark rectangular regions:')
for i, (bx, by, bw, bh, bf) in enumerate(boxes_found):
    print(f'  [{i}] ({bx},{by}) {bw}x{bh} fill={bf:.2f}')

cv2.imwrite(f'{crops_dir}/all_boxes_annotated.png', result)
print(f'\nSaved annotated image with {len(boxes_found)} boxes')

if len(boxes_found) >= 2:
    print('\nPossible stamina bars (looking for pairs):')
    for i in range(min(5, len(boxes_found))):
        for j in range(i+1, min(5, len(boxes_found))):
            bx1, by1, bw1, bh1, _ = boxes_found[i]
            bx2, by2, bw2, bh2, _ = boxes_found[j]
            
            y_diff = abs(by1 - by2)
            x_overlap = min(bx1+bw1, bx2+bw2) - max(bx1, bx2)
            
            if y_diff < 50 and x_overlap < 0:
                print(f'  Pair [{i}]+[{j}]: y_diff={y_diff}, separate_x=True')