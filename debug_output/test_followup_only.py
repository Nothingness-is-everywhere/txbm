# -*- coding: utf-8 -*-
"""Test the follow-up red-dot step on the current (post-profile-tap) screen."""
import subprocess
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

ADB = r"D:\MuMu\nx_main\adb.exe"
DEV = "emulator-5554"
ROI = [0.3741, 0.1406, 0.6297, 0.1922]  # follow-up red-dot region
MIN_PIX = 8

r = subprocess.run([ADB, "-s", DEV, "exec-out", "screencap", "-p"],
                   capture_output=True, timeout=10)
img = Image.open(BytesIO(r.stdout)).convert("RGB")
frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
h, w = frame.shape[:2]
print(f"Screenshot: {w}x{h}")

# Save current screen for the user to confirm which page we are on
cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_followup_current.png", frame)

x1, y1 = int(w * ROI[0]), int(h * ROI[1])
x2, y2 = int(w * ROI[2]), int(h * ROI[3])
region = frame[y1:y2, x1:x2]
hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
mask = cv2.bitwise_or(
    cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
    cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255])),
)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
pixels = int(cv2.countNonZero(mask))
print(f"Red pixels in follow-up ROI: {pixels} (min={MIN_PIX})")

cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_followup_mask.png", mask)
cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_followup_region.png", region)

if pixels >= MIN_PIX:
    M = cv2.moments(mask)
    if M["m00"] == 0:
        print("Zero-area centroid, no click")
    else:
        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        print(f"Follow-up red dot at ({cx}, {cy}), tapping...")
        subprocess.run([ADB, "-s", DEV, "shell", "input", "tap", str(cx), str(cy)],
                       timeout=5)
        print("Tapped. Check the emulator.")
else:
    print("No follow-up red dot on current screen.")
