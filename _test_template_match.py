"""改进的模板匹配 - 限制范围 + 位置过滤"""
import sys
sys.modules['easyocr'] = None

import cv2
import numpy as np
from pathlib import Path
from ok.automation.stamina_reader import read_image

TEMPLATE_DIR = Path("templates/digits")

def load_digit_templates():
    templates = {}
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
                if digit not in templates:
                    templates[digit] = []
                templates[digit].append(img)
    return templates

def find_text_baseline(matches):
    """找到文本基线(y坐标的主要聚类)"""
    if not matches:
        return None, None
    
    y_values = [m['y'] + m['h']/2 for m in matches]
    
    hist = {}
    for y in y_values:
        bucket = round(y / 5) * 5
        hist[bucket] = hist.get(bucket, 0) + 1
    
    best_bucket = max(hist, key=hist.get)
    
    baseline_y = best_bucket
    tolerance = 15
    
    on_baseline = [m for m, y in zip(matches, y_values) 
                   if abs(y - baseline_y) < tolerance]
    
    return on_baseline, baseline_y

def match_stamina_text(image_gray, templates):
    """在图像中匹配体力值文本"""
    h, w = image_gray.shape[:2]
    
    all_matches = []
    
    for digit, tpl_list in templates.items():
        for tpl in tpl_list:
            th, tw = tpl.shape[:2]
            
            digit_matches = []
            for scale in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]:
                scaled_w = int(tw * scale)
                scaled_h = int(th * scale)
                
                if scaled_w > w or scaled_h > h:
                    continue
                
                tpl_scaled = cv2.resize(tpl, (scaled_w, scaled_h))
                
                result = cv2.matchTemplate(
                    image_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED
                )
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.70:
                    digit_matches.append({
                        'digit': digit,
                        'confidence': max_val,
                        'x': max_loc[0],
                        'y': max_loc[1],
                        'w': scaled_w,
                        'h': scaled_h,
                    })
            
            if digit_matches:
                best = max(digit_matches, key=lambda m: m['confidence'])
                all_matches.append(best)
    
    baseline_matches, baseline_y = find_text_baseline(all_matches)
    
    if not baseline_matches:
        baseline_matches = all_matches
    
    baseline_matches.sort(key=lambda m: m['x'])
    
    # Remove duplicates (same position, different digit)
    final = []
    for m in baseline_matches:
        is_dup = False
        for f in final:
            if abs(m['x'] - f['x']) < 10 and abs(m['y'] - f['y']) < 10:
                if m['confidence'] > f['confidence']:
                    final.remove(f)
                    final.append(m)
                is_dup = True
                break
        if not is_dup:
            final.append(m)
    
    return final

templates = load_digit_templates()
print(f"Loaded templates: {sorted(templates.keys())}")

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        print(f"\n{stamina_type}: 无法读取")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{'='*60}")
    print(f"{stamina_type}: {img.shape}")
    print(f"{'='*60}")
    
    matches = match_stamina_text(gray, templates)
    
    recognized = []
    print(f"\nMatches ({len(matches)}):")
    for m in matches:
        print(f"  x={m['x']:3d}, y={m['y']:2d}, w={m['w']:2d}, h={m['h']:2d} -> "
              f"'{m['digit']}' (conf={m['confidence']:.2f})")
        recognized.append(m['digit'])
    
    text = "".join(recognized)
    print(f"\nRecognized: '{text}'")
    
    # Parse stamina
    import re
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        print(f"Parsed: {numbers[0]}/{numbers[1]}")
    
    # Visualize
    vis = img.copy()
    for m in matches:
        cx, cy, cw, ch = m['x'], m['y'], m['w'], m['h']
        cv2.rectangle(vis, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 1)
        cv2.putText(vis, f"{m['digit']}({m['confidence']:.1f})", 
                    (cx, cy-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(f"debug_output/{stamina_type}_matched_v2.png", vis)