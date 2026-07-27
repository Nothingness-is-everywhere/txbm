import sys
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage

PROJECT = Path(r'd:\学习\python\ok-script')
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread, WHITE_HSV_LOWER, WHITE_HSV_UPPER, STAMINA_CONFIG

reader = StaminaReader()
template_lib = reader._template_lib
screenshots = PROJECT / 'screenshots'

img = _imread(str(screenshots / 'full_screen_20260726_203459.png'), cv2.IMREAD_COLOR)
h, w = img.shape[:2]
reader.height, reader.width = h, w

config = STAMINA_CONFIG['stamina1']
bar_region = reader._locate_bar(img, 'stamina1')
bx, by, bw, bh = bar_region

ox, oy, ow, oh = config['number_relative_offset']
nx = bx + int(bw * ox)
ny = by + int(bh * oy)
nw = int(bw * ow)
nh = int(bh * oh)

number_region = img[ny:ny+nh, nx:nx+nw]

x, y, w, h = 29, 0, 112, 35
roi = number_region[y:y+h, x:x+w]

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_or(mask, binary)

cv2.imwrite(str(screenshots / 'debug_fused_mask.png'), combined)

print("=" * 60)
print("方法 1: 距离变换 + 分水岭")
print("=" * 60)

dist_transform = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
print(f"距离变换最大值: {dist_transform.max():.1f}")

_, peaks = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, cv2.THRESH_BINARY)
peaks = peaks.astype(np.uint8)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
print(f"找到 {num_labels - 1} 个峰值区域")

if num_labels > 1:
    markers = np.zeros_like(combined)
    for i in range(1, num_labels):
        markers[labels == i] = i
    
    print(f"标记数量: {num_labels - 1}")
    
    markers = markers.astype(np.int32)
    roi_bgr = roi if len(roi.shape) == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    
    if len(roi_bgr.shape) == 2:
        roi_bgr = cv2.cvtColor(roi_bgr, cv2.COLOR_GRAY2BGR)
    
    try:
        cv2.watershed(roi_bgr, markers)
        
        unique_labels = np.unique(markers)
        valid_labels = [l for l in unique_labels if l > 0]
        print(f"分水岭分割产生 {len(valid_labels)} 个区域")
        
        for label_idx, label_val in enumerate(valid_labels):
            label_mask = (markers == label_val).astype(np.uint8)
            moments = cv2.moments(label_mask)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                ys, xs = np.where(label_mask > 0)
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                w_seg = x_max - x_min + 1
                h_seg = y_max - y_min + 1
                
                if w_seg >= 3 and h_seg >= 5:
                    seg_roi = roi[y_min:y_max+1, x_min:x_max+1]
                    char, conf = template_lib.recognize(seg_roi)
                    print(f"  区域 {label_idx}: x=[{x_min},{x_max}], y=[{y_min},{y_max}], {w_seg}x{h_seg}, 识别='{char}' (conf={conf:.3f})")
    except Exception as e:
        print(f"分水岭失败: {e}")

print("\n" + "=" * 60)
print("方法 2: 形态学细化（骨架提取）")
print("=" * 60)

kernel = np.ones((2, 2), np.uint8)
eroded = cv2.erode(combined, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)
difference = cv2.subtract(combined, dilated)

print(f"形态学差值非零像素: {np.count_nonzero(difference)}")

print("\n" + "=" * 60)
print("方法 3: 垂直投影分析")
print("=" * 60)

col_counts = np.sum(combined > 0, axis=0)
col_projection = col_counts.astype(np.float64)

for i in range(1, len(col_projection) - 1):
    if col_projection[i] < col_projection[i-1] and col_projection[i] < col_projection[i+1]:
        if col_projection[i] < 30:
            print(f"  列 {i}: 局部最小值 = {col_projection[i]}")

smoothed = np.convolve(col_projection, np.ones(5)/5, mode='same')
print("\n平滑后:")
for i in range(1, len(smoothed) - 1):
    if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
        print(f"  列 {i}: 局部最小值 = {smoothed[i]:.1f}")

print("\n" + "=" * 60)
print("方法 4: 分析每个数字的轮廓特征")
print("=" * 60)

contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"找到 {len(contours)} 个轮廓")

for i, cnt in enumerate(contours):
    x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
    area_c = cv2.contourArea(cnt)
    if area_c > 10:
        aspect_c = w_c / h_c if h_c > 0 else 0
        print(f"  轮廓 {i}: bbox=[{x_c},{y_c},{w_c},{h_c}], area={area_c:.0f}, aspect={aspect_c:.3f}")

print("\n完成!")