"""Draw network detection ROI on user's screenshots 1.png and 2.png."""
import cv2
import numpy as np

IMAGES = ['screenshots/1.png', 'screenshots/2.png']

# Current config from NetworkErrorHandler.py
popup_roi = [0.10, 0.25, 0.90, 0.85]  # normalized
popup_threshold = 0.80
button_threshold = 0.85

for img_path in IMAGES:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Cannot read {img_path}")
        continue
    
    h, w = image.shape[:2]
    result = image.copy()
    
    # Convert normalized ROI to pixel coordinates
    rx1 = int(w * popup_roi[0])  # 0.10 -> left
    ry1 = int(h * popup_roi[1])  # 0.25 -> top
    rx2 = int(w * popup_roi[2])  # 0.90 -> right
    ry2 = int(h * popup_roi[3])  # 0.85 -> bottom
    
    # Draw popup detection ROI (green)
    cv2.rectangle(result, (rx1, ry1), (rx2, ry2), (0, 255, 0), 4)
    cv2.putText(result, f'Popup ROI [0.10,0.25,0.90,0.85]', 
               (rx1, ry1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(result, f'({rx1},{ry1})-({rx2},{ry2})', 
               (rx1, ry1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    # Draw button search area (red) - within popup, lower 55%-95% height
    btn_x1 = rx1 + int((rx2 - rx1) * 0.08)
    btn_y1 = ry1 + int((ry2 - ry1) * 0.55)
    btn_x2 = rx2 - int((rx2 - rx1) * 0.08)
    btn_y2 = ry2 - int((ry2 - ry1) * 0.05)
    
    cv2.rectangle(result, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 0, 255), 3)
    cv2.putText(result, 'Button search area (55%-95% height)', 
               (btn_x1, btn_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # Center point of ROI
    cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
    cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
    cv2.putText(result, f'Center: ({cx},{cy})', (cx + 12, cy - 8),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Label image
    label = img_path.split('/')[-1]
    cv2.putText(result, f'{label} - Screen: {w}x{h}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(result, 'Network Error Detection Areas:', (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(result, f'popup_threshold: {popup_threshold} | button_threshold: {button_threshold}', 
               (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Check if the actual popup position is within our ROI
    # Analyze the image to find the popup
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    popup_regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 30000 and ch > 100:
            popup_regions.append((x, y, cw, ch, area))
    
    popup_regions.sort(key=lambda r: r[1])
    
    print(f"\n{'='*60}")
    print(f"Analyzing {img_path}:")
    print(f"  Screen: {w}x{h}")
    print(f"  ROI: ({rx1},{ry1}) -> ({rx2},{ry2})")
    
    # Find the actual popup
    for i, (px, py, pw, ph, pa) in enumerate(popup_regions):
        print(f"  Possible popup [{i}]: ({px},{py}) {pw}x{ph} area={pa:.0f}")
        
        # Check if inside ROI
        inside_x = px >= rx1 and px + pw <= rx2
        inside_y = py >= ry1 and py + ph <= ry2
        inside = inside_x and inside_y
        
        print(f"    Inside ROI: {inside} (x:{inside_x}, y:{inside_y})")
        
        # Draw actual popup bounding box
        cv2.rectangle(result, (px, py), (px + pw, py + ph), (255, 255, 0), 2)
        cv2.putText(result, f'Actual popup [{i}]', (px, py - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)
    
    # Save result
    out_path = img_path.replace('.png', '_annotated.png')
    cv2.imwrite(out_path, result)
    print(f"  Saved: {out_path}")

print("\nDone! Check the annotated images:")
print("  screenshots/1_annotated.png")
print("  screenshots/2_annotated.png")