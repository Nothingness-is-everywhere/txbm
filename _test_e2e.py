"""End-to-end pipeline test on actual game screenshots."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import (
    StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER,
    DIGIT_TEMPLATE_SIZE,
)

screenshots = PROJECT / "screenshots"

reader = StaminaReader()

full_screens = list(screenshots.glob("full_screen*.png")) + list(screenshots.glob("main_final*.png"))
number_regions = list(screenshots.glob("*full_numbers*.png"))

print("=" * 60)
print("END-TO-END STAMINA RECOGNITION TEST")
print("=" * 60)

for screen_path in sorted(full_screens):
    print(f"\n{'='*60}")
    print(f"FULL SCREEN: {screen_path.name}")
    
    screen = _imread(str(screen_path), cv2.IMREAD_COLOR)
    if screen is None:
        print("  Cannot read!")
        continue
    
    reader.height, reader.width = screen.shape[:2]
    print(f"  Resolution: {reader.width}x{reader.height}")
    
    s1 = reader.read_stamina1(screen)
    s2 = reader.read_stamina2(screen)
    
    print(f"  Stamina1 (出征): {s1.current}/{s1.max}")
    print(f"  Stamina2 (调教): {s2.current}/{s2.max}")

print("\n" + "=" * 60)
print("DIRECT NUMBER REGION TEST")
print("=" * 60)

lib = reader._template_lib

for region_path in sorted(number_regions):
    print(f"\n{'='*60}")
    print(f"REGION: {region_path.name}")
    
    region = _imread(str(region_path), cv2.IMREAD_COLOR)
    if region is None:
        print("  Cannot read!")
        continue
    
    h, w = region.shape[:2]
    print(f"  Size: {w}x{h}")
    
    # Step 1: Extract digits
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)
    
    kernel = np.ones((1, 1), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    
    print(f"  White mask components: {num_labels - 1}")
    
    # Try with grayscale thresholds
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    best_regions = []
    
    for thresh in [140, 160, 180, 200]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        num_labels2, _, stats2, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        regions = []
        for lid in range(1, num_labels2):
            x = stats2[lid, cv2.CC_STAT_LEFT]
            y = stats2[lid, cv2.CC_STAT_TOP]
            ww = stats2[lid, cv2.CC_STAT_WIDTH]
            hh = stats2[lid, cv2.CC_STAT_HEIGHT]
            area = stats2[lid, cv2.CC_STAT_AREA]
            
            if area < 5 or ww < 2 or hh < 3:
                continue
            
            aspect = ww / hh if hh > 0 else 0
            if aspect < 0.08 or aspect > 4.5:
                continue
            
            regions.append((x, y, ww, hh, aspect))
        
        print(f"  Threshold {thresh}: {len(regions)} valid regions")
        if len(regions) >= 2 and len(regions) > len(best_regions):
            best_regions = regions
    
    if not best_regions:
        print("  ⚠️  No digit regions found!")
        continue
    
    print(f"\n  Best ({len(best_regions)} regions):")
    recognized = []
    for i, (x, y, ww, hh, aspect) in enumerate(best_regions):
        pad = 2
        dx1, dy1 = max(0, x - pad), max(0, y - pad)
        dx2, dy2 = min(w, x + ww + pad), min(h, y + hh + pad)
        digit_img = region[dy1:dy2, dx1:dx2]
        
        char, conf = lib.recognize(digit_img)
        
        is_narrow = aspect < 0.25 and hh > 5
        
        if is_narrow:
            recognized.append('/')
            print(f"    [{i}] x={x}, y={y}, w={ww}, h={hh}, aspect={aspect:.2f} => '/' (narrow)")
        else:
            recognized.append(char or '?')
            print(f"    [{i}] x={x}, y={y}, w={ww}, h={hh}, aspect={aspect:.2f} => '{char}' (conf={conf:.3f})")
    
    result = ''.join(recognized)
    print(f"\n  RECOGNIZED: '{result}'")
    
    import re
    cleaned = result.replace('?', '')
    numbers = re.findall(r'\d+', cleaned)
    if len(numbers) >= 2:
        print(f"  PARSED: {numbers[0]}/{numbers[1]}")
    elif len(numbers) == 1:
        print(f"  PARSED: {numbers[0]}/?")
    else:
        print(f"  PARSED: unable to parse")

print("\n" + "=" * 60)
print("TEST COMPLETE")