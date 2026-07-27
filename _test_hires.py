"""从全屏截图高分辨率裁剪并识别体力值"""
import sys
sys.modules['easyocr'] = None

import cv2
import numpy as np
from ok.automation.stamina_reader import STAMINA_ROI_CONFIG, read_image

full_img = read_image("screenshots/main_final.png")
if full_img is None:
    full_img = read_image("screenshots/full_screen_20260726_203459.png")

if full_img is None:
    print("无法读取全屏截图")
    sys.exit(1)

h, w = full_img.shape[:2]
print(f"Full screenshot: {w}x{h}")

for stamina_type in ["expedition", "training"]:
    config = STAMINA_ROI_CONFIG[stamina_type]
    x1 = int(w * config["x_start"])
    x2 = int(w * config["x_end"])
    y1 = int(h * config["y_start"])
    y2 = int(h * config["y_end"])
    
    region = full_img[y1:y2, x1:x2]
    rh, rw = region.shape[:2]
    print(f"\n{stamina_type}: crop region [{x1}:{x2}, {y1}:{y2}] = {rw}x{rh}")
    
    # Save high-res crop
    cv2.imwrite(f"debug_output/{stamina_type}_hires.png", region)
    
    # Try upscaling
    upscaled = cv2.resize(region, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"debug_output/{stamina_type}_upscaled.png", upscaled)
    
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    
    # Try multiple thresholds for inverted text
    for thresh in [140, 160, 180, 200]:
        _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if ch < 20 or cw < 6:
                continue
            if area > gray.size * 0.2:
                continue
            aspect = cw / ch if ch > 0 else 0
            if aspect > 4.0 or aspect < 0.15:
                continue
            valid.append((cx, cy, cw, ch))
        
        valid.sort(key=lambda r: r[0])
        
        if 2 <= len(valid) <= 10:
            print(f"  Threshold {thresh} (INV): {len(valid)} valid contours")
            for i, (cx, cy, cw, ch) in enumerate(valid):
                digit_roi = upscaled[cy:cy+ch, cx:cx+cw]
                digit_small = cv2.resize(digit_roi, (30, 50))
                cv2.imwrite(f"debug_output/{stamina_type}_t{thresh}_d{i}.png", digit_small)
                print(f"    [{i}] x={cx:4d}, y={cy:4d}, w={cw:3d}, h={ch:3d}, aspect={cw/ch:.2f}")
            
            # Save the mask with contours drawn
            vis = upscaled.copy()
            for cx, cy, cw, ch in valid:
                cv2.rectangle(vis, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
            cv2.imwrite(f"debug_output/{stamina_type}_t{thresh}_annotated.png", vis)
            break
    else:
        print(f"  No good threshold found")

print("\nDone! Check debug_output/ for results.")