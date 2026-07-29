"""
Test script for NetworkErrorHandler template matching.

Usage:
  python _test_network_error_template.py --image PATH_TO_SCREENSHOT

Required: Real template images must exist:
  templates/network_popup.png
  templates/network_confirm_button.png

Use _extract_network_templates.py to crop real templates from a screenshot first.

The script will:
  - Load the screenshot and templates
  - Run template matching for popup and confirm button
  - Output match scores, thresholds, and coordinates
  - Save debug images with bounding boxes to artifacts/network_error/
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(relative_path: str) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def safe_imread(path: str) -> Optional[np.ndarray]:
    """Read image safely, handling Unicode paths."""
    p = Path(path)
    if not p.exists():
        print(f"[WARN] File not found: {path}")
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to decode: {path}")
        return img
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


def validate_template_image(img: np.ndarray, label: str) -> bool:
    """Validate template is real (not solid-color placeholder)."""
    if img is None or img.size == 0:
        print(f"[{label}] FAIL: empty or None")
        return False
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        print(f"[{label}] FAIL: too small ({w}x{h})")
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    if var < 5.0:
        print(f"[{label}] FAIL: variance too low ({var:.2f}), likely a placeholder. REPLACE with real template!")
        return False
    print(f"[{label}] OK: {w}x{h}, var={var:.2f}")
    return True


def normalize_roi(roi: List[float], screen_w: int, screen_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Normalize ROI: auto-detect normalized [0,1] vs pixel coordinates."""
    try:
        vals = [float(v) for v in roi]
    except (TypeError, ValueError):
        return None
    if len(vals) != 4:
        return None

    is_pixel = any(v > 1.5 for v in vals)
    if is_pixel:
        ax1, ay1, ax2, ay2 = [int(round(v)) for v in vals]
    else:
        vals = [max(0.0, min(1.0, v)) for v in vals]
        ax1 = int(round(screen_w * vals[0]))
        ay1 = int(round(screen_h * vals[1]))
        ax2 = int(round(screen_w * vals[2]))
        ay2 = int(round(screen_h * vals[3]))

    ax1 = max(0, min(screen_w - 1, ax1))
    ay1 = max(0, min(screen_h - 1, ay1))
    ax2 = max(0, min(screen_w, ax2))
    ay2 = max(0, min(screen_h, ay2))

    if ax1 >= ax2 or ay1 >= ay2:
        return None
    return (ax1, ay1, ax2, ay2)


def multi_scale_match(
    screen: np.ndarray,
    template: np.ndarray,
    roi: List[float],
    threshold: float,
    scales: List[float],
    label: str,
) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
    """Multi-scale template matching with ROI auto-detection."""
    if template is None:
        print(f"[{label}] Template is None")
        return None

    screen_h, screen_w = screen.shape[:2]
    tpl_h, tpl_w = template.shape[:2]

    pixel_roi = normalize_roi(roi, screen_w, screen_h)
    if pixel_roi is None:
        print(f"[{label}] Invalid ROI")
        return None

    roi_x1, roi_y1, roi_x2, roi_y2 = pixel_roi
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    print(f"\n[{label}] Screen: {screen_w}x{screen_h}, ROI: ({roi_x1},{roi_y1})-({roi_x2},{roi_y2}) [{roi_w}x{roi_h}]")
    print(f"[{label}] Template: {tpl_w}x{tpl_h}, Scales: {scales}, Threshold: {threshold}")

    if roi_w < tpl_w or roi_h < tpl_h:
        print(f"[{label}] ROI too small for template")
        return None

    roi_region = screen[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_val = 0.0
    best_loc = None
    best_scale = 1.0
    best_size = (tpl_w, tpl_h)

    for scale in scales:
        scaled_w = int(tpl_w * scale)
        scaled_h = int(tpl_h * scale)
        if scaled_w > roi_gray.shape[1] or scaled_h > roi_gray.shape[0]:
            print(f"  scale {scale:.2f}: SKIP (too large)")
            continue
        if scaled_w < 4 or scaled_h < 4:
            continue

        tpl_scaled = cv2.resize(tpl_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(roi_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"  scale {scale:.2f}: conf={max_val:.4f} at local({max_loc[0]},{max_loc[1]})")

        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_size = (scaled_w, scaled_h)

    if best_val >= threshold and best_loc is not None:
        cx = roi_x1 + best_loc[0] + best_size[0] // 2
        cy = roi_y1 + best_loc[1] + best_size[1] // 2
        bx1 = roi_x1 + best_loc[0]
        by1 = roi_y1 + best_loc[1]
        bx2 = bx1 + best_size[0]
        by2 = by1 + best_size[1]
        print(f"\n[{label}] FOUND!")
        print(f"  Center: ({cx}, {cy}), Confidence: {best_val:.4f}, Scale: {best_scale:.2f}")
        print(f"  BBox: ({bx1},{by1}) - ({bx2},{by2})")
        return (cx, cy, best_val, (bx1, by1, bx2, by2))
    else:
        print(f"\n[{label}] NOT FOUND. Max conf={best_val:.4f} (threshold={threshold})")
        return None


def run_test(image_path: str, config: dict):
    """Run template matching test on a real screenshot."""
    print("=" * 60)
    print("NetworkErrorHandler Template Matching Test")
    print("=" * 60)

    screen = safe_imread(image_path)
    if screen is None:
        print(f"[ERROR] Cannot load screenshot: {image_path}")
        sys.exit(1)

    screen_h, screen_w = screen.shape[:2]
    print(f"\nScreenshot: {image_path}")
    print(f"Dimensions: {screen_w}x{screen_h}")

    popup_abs = str(resolve_repo_path(config["popup_template_path"]))
    button_abs = str(resolve_repo_path(config["confirm_button_template_path"]))

    popup_tpl = safe_imread(popup_abs)
    btn_tpl = safe_imread(button_abs)

    print("\n--- Template Validation ---")
    popup_valid = validate_template_image(popup_tpl, "popup")
    btn_valid = validate_template_image(btn_tpl, "confirm_button")

    if not popup_valid or not btn_valid:
        print("\n[ERROR] Templates invalid. Please use _extract_network_templates.py to create real templates.")
        print("  python _extract_network_templates.py --image <screenshot_with_popup>")
        sys.exit(1)

    # Test 1: Popup detection
    print("\n" + "-" * 60)
    print("Test 1: Network Popup Detection")
    print("-" * 60)
    popup_result = None
    if popup_tpl is not None:
        popup_result = multi_scale_match(screen, popup_tpl, config["popup_roi"], config["popup_threshold"], config["match_scales"], "popup")

    # Test 2: Button detection
    print("\n" + "-" * 60)
    print("Test 2: Confirm Button Detection")
    print("-" * 60)
    btn_result = None
    if popup_result is not None and btn_tpl is not None:
        _, _, _, popup_bbox = popup_result
        bx1, by1, bx2, by2 = popup_bbox
        margin_x = int((bx2 - bx1) * 0.08)
        margin_top = int((by2 - by1) * 0.55)
        margin_bottom = int((by2 - by1) * 0.05)
        btn_roi = [
            max(0.0, (bx1 + margin_x) / screen_w),
            max(0.0, (by1 + margin_top) / screen_h),
            min(1.0, (bx2 - margin_x) / screen_w),
            min(1.0, (by2 - margin_bottom) / screen_h),
        ]
        print("Searching for confirm button within popup region...")
        btn_result = multi_scale_match(screen, btn_tpl, btn_roi, config["button_threshold"], config["match_scales"], "confirm_button")
    elif popup_result is None:
        print("[SKIP] Popup not detected, cannot search for button")
    else:
        print("[SKIP] No button template available")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    popup_found = popup_result is not None
    btn_found = btn_result is not None

    print(f"  Popup detected: {'YES' if popup_found else 'NO'}")
    if popup_found:
        print(f"    Confidence: {popup_result[2]:.4f}, Center: ({popup_result[0]}, {popup_result[1]})")

    print(f"  Confirm button found: {'YES' if btn_found else 'NO'}")
    if btn_found:
        print(f"    Confidence: {btn_result[2]:.4f}, Center: ({btn_result[0]}, {btn_result[1]})")

    # Save debug
    out_dir = resolve_repo_path("artifacts/network_error")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = screen.copy()
    if popup_result:
        _, _, _, bbox = popup_result
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis, f"popup ({popup_result[2]:.2f})", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if btn_result:
        _, _, _, bbox = btn_result
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, f"confirm ({btn_result[2]:.2f})", (x1, min(screen_h - 20, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = str(out_dir / f"network_error_test_{timestamp}.png")
    cv2.imwrite(out_path, vis)
    print(f"\nDebug image saved: {out_path}")

    print("\n" + "=" * 60)
    if popup_found and btn_found:
        print("TEST PASSED: Both popup and confirm button detected!")
        print(f"  Would click confirm button at ({btn_result[0]}, {btn_result[1]})")
        return 0
    elif popup_found and not btn_found:
        print("PARTIAL: Popup found but confirm button not detected.")
        print("  Suggestion: adjust button template or lower button_threshold")
        return 1
    else:
        print("FAILED: Neither popup nor button detected.")
        print("  Suggestion: verify template images, ROI, and thresholds")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Test NetworkErrorHandler template matching")
    parser.add_argument("--image", type=str, required=True, help="Path to game screenshot showing network error popup")
    parser.add_argument("--popup-threshold", type=float, default=None)
    parser.add_argument("--button-threshold", type=float, default=None)
    args = parser.parse_args()

    config = {
        "popup_template_path": "templates/network_popup.png",
        "confirm_button_template_path": "templates/network_confirm_button.png",
        "popup_threshold": 0.80,
        "button_threshold": 0.85,
        "match_scales": [0.8, 0.9, 1.0, 1.1, 1.2],
        "popup_roi": [0.10, 0.25, 0.90, 0.85],
    }

    if args.popup_threshold is not None:
        config["popup_threshold"] = args.popup_threshold
    if args.button_threshold is not None:
        config["button_threshold"] = args.button_threshold

    return run_test(args.image, config)


if __name__ == "__main__":
    sys.exit(main())