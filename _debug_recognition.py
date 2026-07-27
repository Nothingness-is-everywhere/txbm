"""Debug the stamina recognition process step by step."""
import cv2
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ok.automation.stamina_reader import StaminaReader, _imread, DigitTemplateLibrary

screenshot_path = "screenshots/full_screen_20260726_203459.png"
img = _imread(screenshot_path)

if img is None:
    print("Failed to load screenshot!")
    exit(1)

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

reader = StaminaReader()

# Test with both stamina regions
for stamina_key in ["stamina1", "stamina2"]:
    print("\n" + "="*60)
    print(f"Processing {stamina_key}")
    print("="*60)
    
    # Get ROI
    roi = reader.get_roi_coordinates(stamina_key, w, h)
    print(f"ROI: {roi}")
    
    if roi is None:
        continue
    
    x, y, rw, rh = roi
    region = img[y:y+rh, x:x+rw]
    
    print(f"Region size: {region.shape[1]}x{region.shape[0]}")
    
    # Save the region
    cv2.imwrite(f"debug_{stamina_key}_region.png", region)
    
    # Step 1: HSV color space segmentation
    print("\n--- Step 1: HSV Segmentation ---")
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 100, 255]))
    cv2.imwrite(f"debug_{stamina_key}_hsv_mask.png", mask)
    
    white_pct = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
    print(f"White pixel percentage: {white_pct:.1f}%")
    
    # Step 2: Connected components analysis
    print("\n--- Step 2: Connected Components ---")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"Found {num_labels - 1} connected components")
    
    # Filter and show each component
    valid_components = []
    for label_id in range(1, num_labels):
        cx = stats[label_id, cv2.CC_STAT_LEFT]
        cy = stats[label_id, cv2.CC_STAT_TOP]
        cw = stats[label_id, cv2.CC_STAT_WIDTH]
        ch = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        valid = area >= 15 and cw >= 2 and ch >= 4 and cw <= rw * 0.8 and ch <= rh * 0.9
        print(f"  Component {label_id}: pos=({cx},{cy}), size={cw}x{ch}, area={area:.0f}, valid={valid}")
        
        if valid:
            pad = 4
            dx1 = max(0, cx - pad)
            dy1 = max(0, cy - pad)
            dx2 = min(rw, cx + cw + pad)
            dy2 = min(rh, cy + ch + pad)
            
            digit_img = region[dy1:dy2, dx1:dx2]
            valid_components.append((cx, cy, cw, ch, digit_img))
            
            cv2.imwrite(f"debug_{stamina_key}_comp_{label_id}.png", digit_img)
    
    print(f"\nValid components: {len(valid_components)}")
    
    # Step 3: Recognize each component
    print("\n--- Step 3: Template Recognition ---")
    lib = DigitTemplateLibrary()
    
    for i, (cx, cy, cw, ch, digit_img) in enumerate(valid_components):
        aspect = cw / ch if ch > 0 else 0
        print(f"\n  Component {i}: pos=({cx},{cy}), size={cw}x{ch}, aspect={aspect:.2f}")
        
        # Check if it might be a slash
        is_slash = False
        if aspect < 0.3 and ch >= 10 and 2 <= cw <= 20:
            is_slash = True
            print(f"    -> Likely a slash '/' (aspect={aspect:.2f}, h={ch})")
        
        if not is_slash:
            char, confidence = lib.recognize(digit_img)
            print(f"    Recognized: '{char}' with confidence {confidence:.3f}")
            
            # Try to show what the digit looks like
            processed = lib._preprocess_digit_for_matching(digit_img)
            if processed is not None:
                cv2.imwrite(f"debug_{stamina_key}_digit_{i}_processed.png", processed)
                print(f"    Processed shape: {processed.shape}")
    
    # Step 4: Try different thresholds
    print("\n--- Step 4: Trying different thresholds ---")
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    for thresh in [100, 120, 140, 160, 180, 200]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        n_labels, _, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Count valid components
        valid_count = 0
        for lid in range(1, n_labels):
            la = cv2.CC_STAT_AREA
            lw = cv2.CC_STAT_WIDTH
            lh = cv2.CC_STAT_HEIGHT
            area = cv2.connectedComponentsWithStats(binary, connectivity=8)[2][lid, la]
            cw_val = cv2.connectedComponentsWithStats(binary, connectivity=8)[2][lid, lw]
            ch_val = cv2.connectedComponentsWithStats(binary, connectivity=8)[2][lid, lh]
            if area >= 15 and cw_val >= 2 and ch_val >= 4:
                valid_count += 1
        
        print(f"  Threshold {thresh}: {n_labels-1} components, {valid_count} valid")
    
    # Final recognition
    print("\n--- Final Recognition Result ---")
    text = reader.recognize_numbers(region)
    current, max_val = reader.parse_stamina_text(text)
    print(f"  Raw text: '{text}'")
    print(f"  Parsed: {current}/{max_val}")

print("\n" + "="*60)
print("Debug complete! Check debug_*.png files.")
print("="*60)