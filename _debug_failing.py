"""Debug failing digit images."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER

failing = [
    "templates/digits/s1_d1.png",
    "templates/digits/stamina1_1.png",
    "templates/digit_templates/stamina1_d1.png",
    "templates/digit_templates/stamina1_d4.png",
]

for path_str in failing:
    img_path = PROJECT / path_str
    print(f"\n{'='*60}")
    print(f"Analyzing: {path_str}")
    
    img = _imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print("  CANNOT READ IMAGE!")
        continue
    
    h, w = img.shape[:2]
    print(f"  Size: {w}x{h}")
    
    # Check pixel stats
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Gray min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
    
    # HSV analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    white_pct = cv2.countNonZero(white_mask) / (w * h) * 100
    print(f"  White pixels (HSV): {white_pct:.1f}%")
    
    # Check contours on white mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  Contours (white mask): {len(contours)}")
    
    if contours:
        for i, cnt in enumerate(contours):
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            print(f"    [{i}] x={x}, y={y}, w={cw}, h={ch}, area={area}")
    
    # Try grayscale thresholds
    for thresh in [100, 120, 140, 160, 180, 200]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            x, y, cw, ch = cv2.boundingRect(contours2[0])
            print(f"  Threshold {thresh}: {len(contours2)} contours, first=({x},{y},{cw},{ch})")
        else:
            print(f"  Threshold {thresh}: 0 contours")
    
    # Save debug visualization
    debug_out = img_path.parent / f"debug_{img_path.name}"
    vis = img.copy()
    if contours:
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x+cw, y+ch), (0, 255, 0), 1)
    cv2.imwrite(str(debug_out), vis)
    print(f"  Debug saved: {debug_out.name}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")