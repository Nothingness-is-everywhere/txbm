"""Extract precise stamina value regions from the screen."""
import cv2
import numpy as np
import os

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

print(f"Full screen: {w}x{h}")

templates_dir = 'templates'
os.makedirs(templates_dir, exist_ok=True)

out_dir = 'screenshots/stamina_extraction'
os.makedirs(out_dir, exist_ok=True)

expedition_region = image[1260:1320, 780:940]
cv2.imwrite(f'{out_dir}/expedition_stamina.png', expedition_region)
print(f"Expedition stamina (出征): saved ({780},{1260}) 160x60")
print(f"  Expected: 84/150")

tame_region = image[1490:1560, 830:970]
cv2.imwrite(f'{out_dir}/tame_stamina.png', tame_region)
print(f"Tame stamina (调教): saved ({830},{1490}) 140x70")
print(f"  Expected: 1/50")

expedition_stamina = image[1275:1310, 810:920]
cv2.imwrite(f'{out_dir}/expedition_numbers.png', expedition_stamina)

tame_stamina = image[1510:1550, 850:950]
cv2.imwrite(f'{out_dir}/tame_numbers.png', tame_stamina)

print("\nSaved number regions for OCR:")
print(f"  Expedition numbers: (810,1275) 110x35")
print(f"  Tame numbers: (850,1510) 100x40")

result = image.copy()

cv2.rectangle(result, (780, 1260), (940, 1320), (0, 255, 0), 3)
cv2.putText(result, 'Stamina1: 84/150', (780, 1255),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.rectangle(result, (830, 1490), (970, 1560), (0, 255, 0), 3)
cv2.putText(result, 'Stamina2: 1/50', (830, 1485),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite(f'{out_dir}/annotated_stamina.png', result)
print("\nSaved annotated full screen with stamina markers")

print("\n" + "="*60)
print("REGION COORDINATES (for stamina_reader.py):")
print("="*60)
print(f"  Stamina1 (出征/Expedition):")
print(f"    ROI: x={780}-{940}, y={1260}-{1320}")
print(f"    Numbers ROI: x={810}-{920}, y={1275}-{1310}")
print()
print(f"  Stamina2 (调教/Tame):")
print(f"    ROI: x={830}-{970}, y={1490}-{1560}")
print(f"    Numbers ROI: x={850}-{950}, y={1510}-{1550}")
print()
print("NORMALIZED (0-1):")
print(f"  Stamina1 ROI: ({780/w:.4f}, {1260/h:.4f}, {940/w:.4f}, {1320/h:.4f})")
print(f"  Stamina2 ROI: ({830/w:.4f}, {1490/h:.4f}, {970/w:.4f}, {1560/h:.4f})")