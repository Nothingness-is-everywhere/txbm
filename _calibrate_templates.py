"""Calibrate templates: extract digits at known positions with correct labels"""
import cv2
import numpy as np
from pathlib import Path

def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(str(path), 'wb') as f:
            f.write(buf.tobytes())

# Known stamina values and expected digits:
# Expedition: "64/150" -> digits: 6, 4, 1, 5, 0 (ignoring '/')
# Training: "24/50" -> digits: 2, 4, 5, 0

# Based on the template matching results, the text baseline is around y=40-46 for expedition
# and y=52-58 for training. Let me extract ROIs at the known positions.

KNOWN_POSITIONS = {
    "expedition": {
        # (x_center, y_center, width, height, expected_digit)
        # Positions of digits in "64/150"
        "digits": [
            (24, 40, 15, 25, "6"),   # First digit of "64"
            (41, 39, 15, 25, "4"),   # Second digit of "64"  
            (58, 45, 11, 20, "1"),   # First digit of "150" (after '/')
            (70, 45, 11, 20, "5"),   # Second digit of "150"
            (84, 46, 11, 20, "0"),   # Third digit of "150"
        ],
        "crop": "screenshots/stamina_crops/expedition_final_v1.png",
    },
    "training": {
        # Positions of digits in "24/50"
        "digits": [
            (65, 52, 13, 22, "2"),   # First digit of "24"
            (82, 51, 15, 25, "4"),   # Second digit of "24"
            (101, 58, 12, 19, "5"),  # First digit of "50" (after '/')
            (109, 58, 10, 17, "0"),  # Second digit of "50"
        ],
        "crop": "screenshots/stamina_crops/training_final_v1.png",
    },
}

output_dir = Path("templates/digits")
output_dir.mkdir(parents=True, exist_ok=True)

for stamina_type, config in KNOWN_POSITIONS.items():
    img = read_image(config["crop"])
    if img is None:
        print(f"Cannot read {config['crop']}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{'='*60}")
    print(f"Calibrating {stamina_type}: {img.shape}")
    print(f"{'='*60}")
    
    for xc, yc, w, h, expected_digit in config["digits"]:
        # Extract ROI
        x1 = max(0, xc - w//2)
        y1 = max(0, yc - h//2)
        x2 = min(img.shape[1], xc + w//2)
        y2 = min(img.shape[0], yc + h//2)
        
        roi = img[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            print(f"  [{expected_digit}] at ({xc},{xc}): EMPTY ROI, skipping")
            continue
        
        # Save as calibrated template
        template_path = output_dir / f"calib_{stamina_type}_digit_{expected_digit}.png"
        save_image(roi_gray, template_path)
        
        # Also resize to standard size for better matching
        standard = cv2.resize(roi_gray, (30, 50))
        standard_path = output_dir / f"calib_std_{stamina_type}_digit_{expected_digit}.png"
        save_image(standard, standard_path)
        
        print(f"  [{expected_digit}] at ({xc},{xc}) -> saved to {template_path}")

# Now test with calibrated templates
print(f"\n{'='*60}")
print("Testing with calibrated templates...")
print(f"{'='*60}")

# Load calibrated templates
calib_templates = {}
for pattern in ["calib_*_digit_*.png"]:
    for tpl_path in output_dir.glob(pattern):
        name = tpl_path.stem
        # Parse: calib_{type}_digit_{digit}
        parts = name.split("_")
        if len(parts) >= 4:
            digit = parts[-1]
            img = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if digit not in calib_templates:
                    calib_templates[digit] = []
                calib_templates[digit].append(img)

print(f"Loaded calibrated templates for: {sorted(calib_templates.keys())}")

def match_with_templates(image_gray, templates):
    """Match image against all templates and return best sequence"""
    h, w = image_gray.shape[:2]
    
    # Find all possible digit positions
    all_matches = []
    
    for digit, tpl_list in templates.items():
        for tpl in tpl_list:
            th, tw = tpl.shape[:2]
            
            for scale in np.linspace(0.3, 1.2, 10):
                sw = max(3, int(tw * scale))
                sh = max(6, int(th * scale))
                
                if sw > w or sh > h:
                    continue
                
                tpl_scaled = cv2.resize(tpl, (sw, sh))
                result = cv2.matchTemplate(image_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.75:
                    all_matches.append({
                        'x': max_loc[0],
                        'y': max_loc[1],
                        'w': sw,
                        'h': sh,
                        'digit': digit,
                        'score': max_val,
                    })
    
    # NMS
    all_matches.sort(key=lambda m: m['score'], reverse=True)
    kept = []
    for m in all_matches:
        is_dup = False
        for k in kept:
            x1 = max(m['x'], k['x'])
            y1 = max(m['y'], k['y'])
            x2 = min(m['x']+m['w'], k['x']+k['w'])
            y2 = min(m['y']+m['h'], k['y']+k['h'])
            if x1 < x2 and y1 < y2:
                inter = (x2-x1)*(y2-y1)
                if inter / min(m['w']*m['h'], k['w']*k['h']) > 0.4:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(m)
    
    kept.sort(key=lambda m: m['x'])
    return kept

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{stamina_type}: {img.shape}")
    
    matches = match_with_templates(gray, calib_templates)
    
    print(f"  Found {len(matches)} matches:")
    recognized = []
    for m in matches:
        recognized.append(m['digit'])
        print(f"    x={m['x']:3d}, y={m['y']:2d}, w={m['w']:2d}, h={m['h']:2d} -> "
              f"'{m['digit']}' (score={m['score']:.3f})")
    
    text = "".join(recognized)
    print(f"\n  Recognized: '{text}'")
    
    import re
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        print(f"  Parsed: {numbers[0]}/{numbers[1]}")
    elif len(numbers) == 1:
        print(f"  Parsed: {numbers[0]}")

print("\nDone!")