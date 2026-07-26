"""Detailed search for stamina value displays - looking at button areas."""
import cv2
import numpy as np
import os

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

crops_dir = 'screenshots/stamina_crops'
os.makedirs(crops_dir, exist_ok=True)

print(f"Image: {w}x{h}")

for y_start in range(0, h, 200):
    y_end = min(y_start + 300, h)
    crop = image[y_start:y_end, :, :]
    cv2.imwrite(f'{crops_dir}/row_{y_start}_{y_end}.png', crop)

print("Saved rows")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for brightness_region in [
    ('top', 0, 200),
    ('upper', 200, 400),
    ('mid', 400, 700),
    ('lower_mid', 700, 1000),
    ('lower', 1000, 1300),
    ('bottom', 1300, 1600),
    ('very_bottom', 1600, 1920),
]:
    region = gray[brightness_region[1]:brightness_region[2], :]
    avg_brightness = np.mean(region)
    dark_pct = np.sum(region < 50) / region.size * 100
    print(f"  {brightness_region[0]} ({brightness_region[1]}-{brightness_region[2]}): avg={avg_brightness:.1f} dark={dark_pct:.1f}%")

print("\nLooking for white/light text regions...")

_, white_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
white_mask = cv2.dilate(white_mask, kernel, iterations=2)
white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

white_regions = []
for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area > 100 and ch > 10:
        white_regions.append((x, y, cw, ch, area))

white_regions.sort(key=lambda r: (r[1], r[0]))

print(f"\nFound {len(white_regions)} white text regions:")
for i, (rx, ry, rw, rh, ra) in enumerate(white_regions[:30]):
    print(f"  [{i}] ({rx},{ry}) {rw}x{rh} area={ra:.0f}")

result = image.copy()
for rx, ry, rw, rh, ra in white_regions:
    if 50 < rw < 300 and 20 < rh < 100:
        cv2.rectangle(result, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        cv2.putText(result, f'({rx},{ry})', (rx, ry-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

cv2.imwrite(f'{crops_dir}/white_text_regions.png', result)
print(f"\nSaved annotated white text regions")