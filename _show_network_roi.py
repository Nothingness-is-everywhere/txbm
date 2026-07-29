"""Generate annotated screenshot showing network error detection areas."""
import cv2
import numpy as np
import os

image = cv2.imread('screenshots/full_screen_20260726_203459.png')
h, w = image.shape[:2]

print(f"Screen size: {w}x{h}")

result = image.copy()

# Network error detection ROI from config:
# popup_roi: [0.10, 0.25, 0.90, 0.85]
roi_x1 = int(w * 0.10)  # 108
roi_y1 = int(h * 0.25)  # 480
roi_x2 = int(w * 0.90)  # 972
roi_y2 = int(h * 0.85)  # 1632

# Draw ROI
cv2.rectangle(result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 4)
cv2.putText(result, f'Popup ROI [0.10,0.25,0.90,0.85]', 
           (roi_x1, roi_y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(result, f'({roi_x1},{roi_y1})-({roi_x2},{roi_y2})', 
           (roi_x1, roi_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Button search area (within popup, lower portion)
# From _match_confirm_button: margin_x=8%, margin_top=55%, margin_bottom=5%
btn_x1 = roi_x1 + int((roi_x2 - roi_x1) * 0.08)
btn_y1 = roi_y1 + int((roi_y2 - roi_y1) * 0.55)
btn_x2 = roi_x2 - int((roi_x2 - roi_x1) * 0.08)
btn_y2 = roi_y2 - int((roi_y2 - roi_y1) * 0.05)

cv2.rectangle(result, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 0, 0), 3)
cv2.putText(result, 'Button search area', (btn_x1, btn_y2 + 25),
           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

# Draw center of ROI
cx, cy = (roi_x1 + roi_x2) // 2, (roi_y1 + roi_y2) // 2
cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
cv2.putText(result, f'Center: ({cx},{cy})', (cx + 12, cy - 8),
           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

# Screen info
cv2.putText(result, f'Screen: {w}x{h}', (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(result, 'Network Error Detection Areas:', (10, 65),
           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Legend
legend_x, legend_y = 10, h - 160
cv2.rectangle(result, (legend_x - 5, legend_y - 25), (legend_x + 350, legend_y + 140), (50, 50, 50), -1)
cv2.putText(result, 'Legend:', (legend_x, legend_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
cv2.rectangle(result, (legend_x, legend_y + 15), (legend_x + 25, legend_y + 35), (0, 255, 0), 2)
cv2.putText(result, 'Popup detection ROI (10%-90% x, 25%-85% y)', (legend_x + 35, legend_y + 32),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.rectangle(result, (legend_x, legend_y + 45), (legend_x + 25, legend_y + 65), (255, 0, 0), 2)
cv2.putText(result, 'Confirm button search area', (legend_x + 35, legend_y + 62),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.circle(result, (legend_x + 12, legend_y + 88), 6, (0, 0, 255), -1)
cv2.putText(result, 'ROI center point', (legend_x + 35, legend_y + 92),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(result, 'popup_threshold: 0.80 | button_threshold: 0.85', (legend_x, legend_y + 118),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
cv2.putText(result, 'Templates: network_popup.png, network_confirm_button.png', (legend_x, legend_y + 138),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

cv2.imwrite('screenshots/network_detection_areas.png', result)
print("Saved: screenshots/network_detection_areas.png")

# Also analyze template quality
print("\n" + "="*60)
print("TEMPLATE QUALITY ANALYSIS:")
print("="*60)

for tpl_name in ['templates/network_popup.png', 'templates/network_confirm_button.png']:
    if os.path.exists(tpl_name):
        tpl = cv2.imread(tpl_name, cv2.IMREAD_COLOR)
        if tpl is not None:
            gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            h, w = tpl.shape[:2]
            var = float(np.var(gray))
            min_val, max_val = float(gray.min()), float(gray.max())
            print(f"\n  {tpl_name}:")
            print(f"    Size: {w}x{h}")
            print(f"    Gray range: {min_val}-{max_val}")
            print(f"    Variance: {var:.2f}")
            if var < 5.0:
                print(f"    ⚠️  WARNING: Variance too low! Template is likely a placeholder.")
                print(f"    ⚠️  The template needs to be a real screenshot of the network error popup.")
            else:
                print(f"    ✅ Template looks valid")
        else:
            print(f"\n  {tpl_name}: FAILED to load")
    else:
        print(f"\n  {tpl_name}: NOT FOUND")

print("\n" + "="*60)
print("DETECTION CONFIGURATION:")
print("="*60)
print(f"  popup_roi: [0.10, 0.25, 0.90, 0.85]")
print(f"    -> pixels: ({roi_x1},{roi_y1}) to ({roi_x2},{roi_y2})")
print(f"  popup_threshold: 0.80")
print(f"  button_threshold: 0.85")
print(f"  match_scales: [0.8, 0.9, 1.0, 1.1, 1.2]")
print(f"  fallback_to_ocr: True")
print(f"  ocr_texts: ['网络不好', '网络异常', '连接失败', '网络连接', '无法连接', '确认', '确定']")
print(f"  cooldown_seconds: 5.0")