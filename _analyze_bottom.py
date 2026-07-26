"""Analyze bottom-right area for stamina values."""
import cv2
import numpy as np
import os

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

crops_dir = 'screenshots/stamina_crops'
os.makedirs(crops_dir, exist_ok=True)

print(f"Full screen: {w}x{h}")

bottom_regions = [
    ('bottom_right', 700, 1200, 380, 720),
    ('bottom_full', 0, 1200, 1080, 720),
    ('bottom_right_small', 750, 1400, 330, 520),
    ('bottom_right_tiny', 800, 1500, 280, 420),
    ('bottom_quarter', 540, 1440, 540, 480),
]

for name, x, y, rw, rh in bottom_regions:
    crop = image[y:y+rh, x:x+rw]
    cv2.imwrite(f'{crops_dir}/{name}.png', crop)
    print(f"Saved {name}: ({x},{y}) {rw}x{rh}")

bottom_half = image[960:1920, :, :]
gray_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

print("\nAnalyzing bottom half for text/digits...")

_, binary_bottom = cv2.threshold(gray_bottom, 200, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
binary_bottom = cv2.dilate(binary_bottom, kernel, iterations=2)

contours, _ = cv2.findContours(binary_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

text_regions = []
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area > 50 and ch > 8 and cw > 5:
        text_regions.append((x, y, cw, ch, area))

text_regions.sort(key=lambda r: (r[1], r[0]))

print(f"\nFound {len(text_regions)} white text regions in bottom half:")
for i, (rx, ry, rw, rh, ra) in enumerate(text_regions[:40]):
    abs_y = ry + 960
    print(f"  [{i}] ({rx},{abs_y}) {rw}x{rh} area={ra:.0f}")

result = image.copy()
for rx, ry, rw, rh, ra in text_regions:
    abs_y = ry + 960
    if 5 < rw < 200 and 8 < rh < 80:
        cv2.rectangle(result, (rx, abs_y), (rx+rw, abs_y+rh), (0, 0, 255), 2)

cv2.imwrite(f'{crops_dir}/bottom_text_annotated.png', result)
print("\nSaved bottom text annotated image")

right_bottom = image[1200:1920, 540:1080, :]
gray_rb = cv2.cvtColor(right_bottom, cv2.COLOR_BGR2GRAY)

_, binary_rb = cv2.threshold(gray_rb, 200, 255, cv2.THRESH_BINARY)
binary_rb = cv2.dilate(binary_rb, kernel, iterations=2)

contours_rb, _ = cv2.findContours(binary_rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rb_text = []
for cnt in contours_rb:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area > 30 and ch > 5:
        rb_text.append((x, y, cw, ch, area))

rb_text.sort(key=lambda r: (r[1], r[0]))

print(f"\nRight-bottom ({540}-{1080}, {1200}-{1920}): found {len(rb_text)} regions")
for i, (rx, ry, rw, rh, ra) in enumerate(rb_text[:50]):
    abs_x = rx + 540
    abs_y = ry + 1200
    print(f"  [{i}] ({abs_x},{abs_y}) {rw}x{rh} area={ra:.0f}")

result2 = image.copy()
for rx, ry, rw, rh, ra in rb_text:
    abs_x = rx + 540
    abs_y = ry + 1200
    if 5 < rw < 150 and 5 < rh < 60:
        cv2.rectangle(result2, (abs_x, abs_y), (abs_x+rw, abs_y+rh), (0, 255, 0), 2)

cv2.imwrite(f'{crops_dir}/right_bottom_text.png', result2)
print("\nSaved right-bottom text annotated image")