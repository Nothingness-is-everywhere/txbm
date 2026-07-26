"""Test stamina reader with existing screenshots (offline test)."""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ok.automation.stamina_reader import (
    StaminaReader,
    DigitTemplateLibrary,
    StaminaValue,
    StaminaState,
)


def test_template_library():
    """Test the digit template library."""
    print("=" * 60)
    print("TEST 1: DigitTemplateLibrary")
    print("=" * 60)

    lib = DigitTemplateLibrary()
    count = lib.get_template_count()
    digits = lib.get_supported_digits()

    print(f"  Templates loaded: {count}")
    print(f"  Supported digits: {''.join(digits)}")

    missing = set('0123456789') - set(digits)
    if missing:
        print(f"  ⚠️  Missing digits: {''.join(sorted(missing))}")
    else:
        print(f"  ✅ All digits 0-9 covered!")

    return count > 0 and len(missing) == 0


def test_recognition_on_screenshot():
    """Test recognition using existing screenshot images."""
    print("\n" + "=" * 60)
    print("TEST 2: Recognition on screenshot images")
    print("=" * 60)

    test_images = [
        ("screenshots/stamina1_full_numbers.png", "stamina1"),
        ("screenshots/stamina2_full_numbers.png", "stamina2"),
        ("screenshots/stamina1_numbers_20260726_201455.png", "stamina1_old"),
        ("screenshots/stamina2_numbers_20260726_201455.png", "stamina2_old"),
    ]

    lib = DigitTemplateLibrary()
    reader = StaminaReader.__new__(StaminaReader)
    reader._initialized = True
    reader._template_lib = lib

    for img_path, label in test_images:
        img_file = Path(img_path)
        if not img_file.exists():
            print(f"  ⚠️  Skipping {img_path} (file not found)")
            continue

        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            print(f"  ⚠️  Skipping {img_path} (unreadable)")
            continue

        print(f"\n  [{label}] {img_path}: {img.shape[1]}x{img.shape[0]}")

        result = reader._read_numbers_from_region(img)
        if result:
            print(f"    Recognized: '{result}'")
            current, max_val = reader._parse_stamina_text(result)
            print(f"    Parsed: {current}/{max_val}")
        else:
            print(f"    ❌ Recognition failed")

    return True


def test_roi_extraction():
    """Test extracting number regions using fixed ROIs."""
    print("\n" + "=" * 60)
    print("TEST 3: ROI-based extraction from full screen")
    print("=" * 60)

    full_screen = Path("screenshots/full_screen_20260726_203459.png")
    if not full_screen.exists():
        print("  ⚠️  Full screen screenshot not found, skipping")
        return True

    img = cv2.imread(str(full_screen), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"  Screen size: {w}x{h}")

    from ok.automation.stamina_reader import STAMINA_CONFIG

    lib = DigitTemplateLibrary()

    for stamina_key, cfg in STAMINA_CONFIG.items():
        nx, ny, nw, nh = cfg["number_roi"]
        print(f"\n  [{cfg['label']}] ROI: ({nx},{ny}) {nw}x{nh}")

        if nx + nw > w or ny + nh > h:
            print(f"    ⚠️  ROI exceeds screen bounds")
            continue

        number_region = img[ny:ny + nh, nx:nx + nw]

        roi_out = Path(f"screenshots/test_roi_{stamina_key}.png")
        cv2.imwrite(str(roi_out), number_region)
        print(f"    Saved ROI to {roi_out}")

        reader = StaminaReader.__new__(StaminaReader)
        reader._initialized = True
        reader._template_lib = lib

        result = reader._read_numbers_from_region(number_region)
        if result:
            current, max_val = reader._parse_stamina_text(result)
            print(f"    ✅ Recognized: {result} → {current}/{max_val}")
        else:
            print(f"    ❌ Recognition failed")

    return True


def main():
    print("\n" + "=" * 60)
    print("STAMINA READER TEST SUITE")
    print("=" * 60)

    results = {}

    results["template_library"] = test_template_library()
    results["recognition"] = test_recognition_on_screenshot()
    results["roi_extraction"] = test_roi_extraction()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())