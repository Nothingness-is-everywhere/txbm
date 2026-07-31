# -*- coding: utf-8 -*-
"""End-to-end test of HomeRedDotTask: profile red-dot -> tap -> follow-up red-dot -> tap."""
import time
import subprocess
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

ADB = r"D:\MuMu\nx_main\adb.exe"
DEV = "emulator-5554"

ROI_PROFILE = [0.0343, 0.0359, 0.2926, 0.0916]   # profile-button region
ROI_FOLLOWUP = [0.3741, 0.1406, 0.6297, 0.1922]  # follow-up red-dot region
THRESHOLD = 0.70
MIN_PIX = 8
TPL_PATH = r"d:\学习\python\ok-script\templates\home_profile_button.png"


def screenshot() -> np.ndarray:
    r = subprocess.run([ADB, "-s", DEV, "exec-out", "screencap", "-p"],
                       capture_output=True, timeout=10)
    img = Image.open(BytesIO(r.stdout)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def tap(x, y):
    subprocess.run([ADB, "-s", DEV, "shell", "input", "tap", str(x), str(y)],
                   timeout=5)
    print(f"  -> ADB tap ({x}, {y})")


def detect_red_dot(frame, roi, tag):
    h, w = frame.shape[:2]
    x1, y1 = int(w * roi[0]), int(h * roi[1])
    x2, y2 = int(w * roi[2]), int(h * roi[3])
    region = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 80]); upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80]); upper2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    pixels = int(cv2.countNonZero(mask))
    # save visual
    cv2.imwrite(rf"d:\学习\python\ok-script\debug_output\test_{tag}_mask.png", mask)
    cv2.imwrite(rf"d:\学习\python\ok-script\debug_output\test_{tag}_region.png", region)
    print(f"  [{tag}] red pixels: {pixels} (min={MIN_PIX})")
    if pixels < MIN_PIX:
        return None
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + x1
    cy = int(M["m01"] / M["m00"]) + y1
    return (cx, cy)


def main():
    # ---- Step 1: screenshot + home check ----
    frame = screenshot()
    h, w = frame.shape[:2]
    print(f"[1] Screenshot: {w}x{h}")

    tpl = cv2.imdecode(np.fromfile(TPL_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    x1, y1 = int(w * ROI_PROFILE[0]), int(h * ROI_PROFILE[1])
    x2, y2 = int(w * ROI_PROFILE[2]), int(h * ROI_PROFILE[3])
    region = frame[y1:y2, x1:x2]
    res = cv2.matchTemplate(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    print(f"[2] Home match: conf={max_val:.4f} -> {'HOME' if max_val >= THRESHOLD else 'NOT HOME'}")
    if max_val < THRESHOLD:
        print("[abort] Not on home screen")
        return

    # ---- Step 2: detect + tap profile red dot ----
    print("[3] Detecting profile-button red dot...")
    dot1 = detect_red_dot(frame, ROI_PROFILE, "profile")
    if dot1 is None:
        print("[3] No profile red dot -> nothing to do")
        return
    print(f"[3] Profile red dot at {dot1}, tapping...")
    tap(dot1[0], dot1[1])

    # ---- Step 3: wait for the next page to load ----
    wait = 1.5
    print(f"[4] Waiting {wait}s for page load...")
    time.sleep(wait)

    # ---- Step 4: detect + tap follow-up red dot ----
    frame2 = screenshot()
    print(f"[5] Follow-up screenshot: {frame2.shape[1]}x{frame2.shape[0]}")
    print("[6] Detecting follow-up red dot...")
    dot2 = detect_red_dot(frame2, ROI_FOLLOWUP, "followup")
    if dot2 is None:
        print("[6] No follow-up red dot -> done")
        return
    print(f"[6] Follow-up red dot at {dot2}, tapping...")
    tap(dot2[0], dot2[1])
    print("[done] Follow-up click performed. Check the emulator.")


if __name__ == "__main__":
    main()
