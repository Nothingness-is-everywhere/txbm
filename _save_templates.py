"""Save correct stamina region templates."""
import cv2
import numpy as np
import os

image = cv2.imread('screenshots/full_screen_20260726_203459.png')

templates_dir = 'templates'
os.makedirs(templates_dir, exist_ok=True)

out_dir = 'screenshots/stamina_extraction'
os.makedirs(out_dir, exist_ok=True)

stamina1_bar = image[1255:1320, 770:950]
cv2.imwrite(f'{templates_dir}/stamina1_bar.png', stamina1_bar)
print(f"Saved stamina1_bar.png: {stamina1_bar.shape[1]}x{stamina1_bar.shape[0]} (出征体力)")

stamina2_bar = image[1485:1565, 820:980]
cv2.imwrite(f'{templates_dir}/stamina2_bar.png', stamina2_bar)
print(f"Saved stamina2_bar.png: {stamina2_bar.shape[1]}x{stamina2_bar.shape[0]} (调教体力)")

expedition_numbers = image[1275:1315, 800:930]
cv2.imwrite(f'{out_dir}/stamina1_numbers_clean.png', expedition_numbers)

tame_numbers = image[1510:1555, 840:960]
cv2.imwrite(f'{out_dir}/stamina2_numbers_clean.png', tame_numbers)

print("\nSaved number regions for inspection")

print("\n" + "="*60)
print("FINAL COORDINATES (normalized 0-1 range):")
print("="*60)
print(f"  STAMINA1_ROI = ({770/1080:.4f}, {1255/1920:.4f}, {950/1080:.4f}, {1320/1920:.4f})")
print(f"  STAMINA2_ROI = ({820/1080:.4f}, {1485/1920:.4f}, {980/1080:.4f}, {1565/1920:.4f})")
print()
print(f"  Stamina1 absolute (1080x1920):")
print(f"    Bar region: x=770-950, y=1255-1320")
print(f"    Numbers: x=800-930, y=1275-1315")
print()
print(f"  Stamina2 absolute (1080x1920):")
print(f"    Bar region: x=820-980, y=1485-1565")
print(f"    Numbers: x=840-960, y=1510-1555")
print("="*60)