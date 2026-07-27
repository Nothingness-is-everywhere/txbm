"""Final approach: for each position, match against ALL templates and pick best"""
import cv2
import numpy as np
from pathlib import Path
import re

def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

TEMPLATE_DIR = Path("templates/digits")

# Load all templates
all_templates = {}
for pattern in ["digit_*.png", "exp_digit_*.png", "train_digit_*.png"]:
    for tpl_path in TEMPLATE_DIR.glob(pattern):
        name = tpl_path.stem
        if name.startswith("digit_"):
            digit = name.replace("digit_", "")
        elif name.startswith("exp_digit_"):
            digit = name.replace("exp_digit_", "")
        elif name.startswith("train_digit_"):
            digit = name.replace("train_digit_", "")
        else:
            continue
        img = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if digit not in all_templates:
                all_templates[digit] = []
            all_templates[digit].append(img)

print(f"Loaded {sum(len(v) for v in all_templates.values())} templates for digits: {sorted(all_templates.keys())}")

def find_digit_candidates(image_gray, templates):
    """Find candidate positions using template matching"""
    h, w = image_gray.shape[:2]
    candidates = []
    
    # Use a narrower scale range since we know the approximate digit size
    for digit, tpl_list in templates.items():
        for tpl in tpl_list:
            th, tw = tpl.shape[:2]
            
            for scale in np.linspace(0.35, 1.0, 14):
                scaled_w = int(tw * scale)
                scaled_h = int(th * scale)
                
                if scaled_w > w or scaled_h > h:
                    continue
                if scaled_w < 4 or scaled_h < 10:
                    continue
                
                tpl_scaled = cv2.resize(tpl, (scaled_w, scaled_h))
                result = cv2.matchTemplate(image_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.80:
                    candidates.append({
                        'x': max_loc[0],
                        'y': max_loc[1],
                        'w': scaled_w,
                        'h': scaled_h,
                        'template_digit': digit,
                        'confidence': max_val,
                    })
    
    return candidates

def robust_nms(candidates, iou_threshold=0.4):
    """Robust NMS that also considers y-position grouping"""
    if not candidates:
        return []
    
    candidates.sort(key=lambda c: c['confidence'], reverse=True)
    
    kept = []
    for c in candidates:
        is_dup = False
        for k in kept:
            # Calculate IoU
            x1 = max(c['x'], k['x'])
            y1 = max(c['y'], k['y'])
            x2 = min(c['x'] + c['w'], k['x'] + k['w'])
            y2 = min(c['y'] + c['h'], k['y'] + k['h'])
            
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                c_area = c['w'] * c['h']
                k_area = k['w'] * k['h']
                iou = intersection / min(c_area, k_area)
                
                if iou > iou_threshold:
                    is_dup = True
                    break
        
        if not is_dup:
            kept.append(c)
    
    return kept

def identify_digit_at_position(image_gray, x, y, w, h, templates):
    """At a given position, match against ALL templates and return best match"""
    ih, iw = image_gray.shape[:2]
    
    # Add padding around the position
    pad_x = max(5, w // 2)
    pad_y = max(5, h // 2)
    roi_x1 = max(0, x - pad_x)
    roi_y1 = max(0, y - pad_y)
    roi_x2 = min(iw, x + w + pad_x)
    roi_y2 = min(ih, y + h + pad_y)
    
    roi = image_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return '?', 0.0
    
    best_digit = '?'
    best_score = 0
    
    for digit, tpl_list in templates.items():
        for tpl in tpl_list:
            th, tw = tpl.shape[:2]
            
            # Try scales from small to large
            for scale in np.linspace(0.3, 1.5, 13):
                scaled_w = max(2, int(tw * scale))
                scaled_h = max(4, int(th * scale))
                
                if scaled_w > roi.shape[1] or scaled_h > roi.shape[0]:
                    continue
                
                tpl_scaled = cv2.resize(tpl, (scaled_w, scaled_h))
                
                result = cv2.matchTemplate(roi, tpl_scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_digit = digit
    
    return best_digit, best_score

# Process
for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{'='*60}")
    print(f"{stamina_type}: {img.shape}")
    print(f"{'='*60}")
    
    # Step 1: Find candidate positions
    candidates = find_digit_candidates(gray, all_templates)
    print(f"\nCandidates found: {len(candidates)}")
    
    # Step 2: NMS
    filtered = robust_nms(candidates, iou_threshold=0.4)
    print(f"After NMS: {len(filtered)}")
    
    # Step 3: Sort by x position
    filtered.sort(key=lambda c: (c['x'] + c['w']/2))
    
    # Step 4: For each position, re-classify using ALL templates
    print(f"\nRe-classification:")
    recognized = []
    
    for i, c in enumerate(filtered):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        digit, score = identify_digit_at_position(gray, x, y, w, h, all_templates)
        recognized.append(digit)
        
        print(f"  [{i}] x={x:3d}, y={y:2d}, w={w:2d}, h={h:2d} -> "
              f"'{digit}' (score={score:.3f})")
    
    text = "".join(recognized)
    print(f"\nRecognized: '{text}'")
    
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        print(f"Parsed: {numbers[0]}/{numbers[1]}")
    elif len(numbers) == 1:
        print(f"Parsed: {numbers[0]}")

print("\nDone!")