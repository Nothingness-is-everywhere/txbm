"""Analyze screen to find stamina value regions marked with black boxes."""
import cv2
import numpy as np

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

print(f"Image size: {w}x{h}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

black_mask = cv2.inRange(gray, 0, 30)

kernel = np.ones((5, 5), np.uint8)
black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print('Searching for black rectangular regions...')
print()

candidates = []
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    
    if area > 5000 and area < 200000:
        rect_area = cw * ch
        fill_ratio = area / rect_area if rect_area > 0 else 0
        
        aspect = cw / ch if ch > 0 else 0
        
        if 0.3 <= aspect <= 5.0 and fill_ratio > 0.6:
            candidates.append({
                'x': x, 'y': y, 'w': cw, 'h': ch,
                'area': area, 'fill': fill_ratio, 'aspect': aspect
            })
            print(f'  Black region at ({x},{y}): {cw}x{ch} area={area:.0f} fill={fill_ratio:.2f} aspect={aspect:.2f}')

print(f'\nTotal black box candidates: {len(candidates)}')

candidates.sort(key=lambda c: (c['y'], c['x']))

print('\nTop candidates (sorted by position):')
for i, c in enumerate(candidates[:10]):
    x, y, cw, ch = c['x'], c['y'], c['w'], c['h']
    print(f'  [{i}] ({x},{y}) {cw}x{ch} fill={c["fill"]:.2f}')

result_img = image.copy()
for c in candidates:
    cv2.rectangle(result_img, (c['x'], c['y']), (c['x']+c['w'], c['y']+c['h']), (0, 255, 0), 2)

cv2.imwrite('screenshots/black_boxes_analysis.png', result_img)
print('\nSaved annotated image: screenshots/black_boxes_analysis.png')

print('\n' + '='*60)
print('NOTE: Looking for stamina regions near "出征" and "调教" buttons')
print('These should be in the upper portion of the screen')
print('='*60)