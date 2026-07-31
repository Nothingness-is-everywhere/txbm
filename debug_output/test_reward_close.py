# -*- coding: utf-8 -*-
"""Test the reward red-dot (selection A) + close button (selection B) step."""
import time
import subprocess
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

ADB = r"D:\MuMu\nx_main\adb.exe"
DEV = "emulator-5554"
ROI_REWARD = [0.5528, 0.8286, 0.8361, 0.8786]  # selection A
ROI_CLOSE = [0.4435, 0.9229, 0.5555, 0.9844]   # selection B
MIN_PIX = 8


def screenshot():
    r = subprocess.run([ADB, "-s", DEV, "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=10)
    img = Image.open(BytesIO(r.stdout)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def tap(x, y):
    subprocess.run([ADB, "-s", DEV, "shell", "input", "tap", str(x), str(y)],
                   timeout=5)
    print(f"  -> ADB tap ({x}, {y})")


def detect_red(frame, roi):
    h, w = frame.shape[:2]
    x1, y1 = int(w * roi[0]), int(h * roi[1])
    x2, y2 = int(w * roi[2]), int(h * roi[3])
    region = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255])),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    pixels = int(cv2.countNonZero(mask))
    cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_reward_mask.png", mask)
    cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_reward_region.png", region)
    print(f"  reward red pixels: {pixels} (min={MIN_PIX})")
    if pixels < MIN_PIX:
        return None
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]) + x1, int(M["m01"] / M["m00"]) + y1)


def main():
    frame = screenshot()
    h, w = frame.shape[:2]
    print(f"[1] Screenshot: {w}x{h}")
    cv2.imwrite(r"d:\学习\python\ok-script\debug_output\test_reward_current.png", frame)

    # Step A: detect reward red dot, tap if present
    print("[2] Detecting reward red dot (selection A)...")
    dot = detect_red(frame, ROI_REWARD)
    if dot is None:
        print("[2] No reward red dot -> skip tapping A")
    else:
        print(f"[2] Reward red dot at {dot}, tapping...")
        tap(dot[0], dot[1])
        time.sleep(1.5)

    # Step B: tap close button centre to return home
    frame2 = screenshot()
    x1, y1 = int(w * ROI_CLOSE[0]), int(h * ROI_CLOSE[1])
    x2, y2 = int(w * ROI_CLOSE[2]), int(h * ROI_CLOSE[3])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    print(f"[3] Tapping close button (selection B) at ({cx}, {cy})...")
    tap(cx, cy)
    print("[done] Close button tapped. Check the emulator — should be back on home.")


if __name__ == "__main__":
    main()
