"""Show current vs corrected detection ROI on screenshots."""
import cv2
import numpy as np

IMAGES = ['screenshots/1.png', 'screenshots/2.png']

old_roi = [0.10, 0.25, 0.90, 0.85]
new_roi = [0.00, 0.15, 1.00, 0.70]

for img_path in IMAGES:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        continue
    
    h, w = image.shape[:2]
    result = image.copy()
    
    ox1, oy1 = int(w * old_roi[0]), int(h * old_roi[1])
    ox2, oy2 = int(w * old_roi[2]), int(h * old_roi[3])
    cv2.rectangle(result, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
    cv2.putText(result, 'OLD ROI [0.10,0.25,0.90,0.85]', (ox1, oy1 - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    nx1, ny1 = int(w * new_roi[0]), int(h * new_roi[1])
    nx2, ny2 = int(w * new_roi[2]), int(h * new_roi[3])
    cv2.rectangle(result, (nx1, ny1), (nx2, ny2), (0, 255, 0), 4)
    cv2.putText(result, 'NEW ROI [0.00,0.15,1.00,0.70]', (nx1, ny2 + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    if ox1 > nx1:
        cv2.rectangle(result, (nx1, oy1), (ox1, oy2), (0, 0, 255), 2)
        cv2.putText(result, 'CUTOFF', (nx1 + 5, oy2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
    
    cv2.rectangle(result, (ox1, ny1), (ox2, oy1), (255, 0, 0), 2)
    cv2.rectangle(result, (ox1, oy2), (ox2, ny2), (255, 0, 0), 2)
    
    ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
    ncx, ncy = (nx1 + nx2) // 2, (ny1 + ny2) // 2
    cv2.circle(result, (ocx, ocy), 6, (0, 0, 255), -1)
    cv2.circle(result, (ncx, ncy), 6, (0, 255, 0), -1)
    
    label = img_path.split('/')[-1]
    cv2.putText(result, label + ' - ' + str(w) + 'x' + str(h), (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(result, 'Red=OLD (cuts off popup) | Green=NEW (full coverage)', 
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    popup_info = ""
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 30000 and ch > 100:
            popup_info = "Actual popup: (" + str(x) + "," + str(y) + ") " + str(cw) + "x" + str(ch)
            inside_old = (x >= ox1 and x + cw <= ox2 and y >= oy1 and y + ch <= oy2)
            inside_new = (x >= nx1 and x + cw <= nx2 and y >= ny1 and y + ch <= ny2)
            popup_info += " | Inside OLD: " + str(inside_old) + " | Inside NEW: " + str(inside_new)
            cv2.rectangle(result, (x, y), (x + cw, y + ch), (255, 255, 0), 2)
    
    cv2.putText(result, popup_info, (10, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    
    out_path = img_path.replace('.png', '_comparison.png')
    cv2.imwrite(out_path, result)
    print("Saved: " + out_path)

print("")
print("=" * 60)
print("PROBLEM ANALYSIS:")
print("=" * 60)
print("  Current popup_roi: [0.10, 0.25, 0.90, 0.85]")
print("    -> Leaves 10% margin on each side")
print("    -> But the popup spans nearly full screen width")
print("    -> Left margin (10%) cuts off popup's left edge")
print()
print("  Suggested popup_roi: [0.00, 0.15, 1.00, 0.70]")
print("    -> 0% left margin, 0% right margin")
print("    -> Wider vertical coverage (15%-70%)")
print("    -> Captures the full popup")