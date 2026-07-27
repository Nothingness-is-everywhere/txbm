"""从裁剪图中提取数字模板并测试识别"""
import cv2
import numpy as np
from pathlib import Path
from ok.automation.stamina_reader import read_image

TEMPLATE_DIR = Path("templates/digits")
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

def extract_digits_adaptive(image, name_prefix):
    """使用自适应方法提取数字"""
    h, w = image.shape[:2]
    area = h * w
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    best_mask = None
    best_count = 0
    
    for thresh in [120, 140, 160, 180, 200]:
        for mode in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, mask = cv2.threshold(gray, thresh, 255, mode)
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                cnt_area = bw * bh
                if bh < 8 or bw < 2:
                    continue
                if cnt_area > area * 0.25:
                    continue
                aspect = bw / bh if bh > 0 else 0
                if aspect > 4.0 or aspect < 0.1:
                    continue
                valid.append((x, y, bw, bh))
            
            if 2 <= len(valid) <= 8 and len(valid) > best_count:
                best_count = len(valid)
                best_mask = mask.copy()
    
    if best_mask is None:
        _, best_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        cnt_area = bw * bh
        if cnt_area > area * 0.25:
            continue
        if bh < 8 or bw < 2:
            continue
        aspect = bw / bh if bh > 0 else 0
        if aspect > 4.0 or aspect < 0.1:
            continue
        
        roi = image[y:y+bh, x:x+bw]
        digit_regions.append((x, y, bw, bh, roi))
    
    digit_regions.sort(key=lambda r: (r[0], r[1]))
    
    print(f"  Found {len(digit_regions)} digit regions:")
    for i, (x, y, bw, bh, roi) in enumerate(digit_regions):
        print(f"    [{i}] x={x:3d}, y={y:3d}, w={bw:2d}, h={bh:2d}, aspect={bw/bh:.2f}")
        
        digit_resized = cv2.resize(roi, (30, 50))
        cv2.imwrite(str(TEMPLATE_DIR / f"{name_prefix}_digit_{i}.png"), digit_resized)
    
    return digit_regions

# Process expedition
print("=" * 60)
print("Expedition (64/150)")
print("=" * 60)
exp_img = read_image("screenshots/stamina_crops/expedition_final_v1.png")
exp_digits = extract_digits_adaptive(exp_img, "exp")

# Process training  
print("\n" + "=" * 60)
print("Training (24/50)")
print("=" * 60)
train_img = read_image("screenshots/stamina_crops/training_final_v1.png")
train_digits = extract_digits_adaptive(train_img, "train")

# Now let's manually map digits based on what we expect
# Expedition should be: 6, 4, /, 1, 5, 0
# Training should be: 2, 4, /, 5, 0

print("\n" + "=" * 60)
print("Templates saved to templates/digits/")
print("Files:")
for f in sorted(TEMPLATE_DIR.glob("*.png")):
    img = cv2.imread(str(f))
    if img is not None:
        print(f"  {f.name}: {img.shape}")