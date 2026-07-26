"""Full end-to-end test of the stamina reader pipeline."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import (
    StaminaReader, DigitTemplateLibrary,
    WHITE_HSV_LOWER, WHITE_HSV_UPPER,
    DIGIT_TEMPLATE_SIZE,
)

def test_template_library():
    """Test that template library can recognize digits."""
    print("=" * 60)
    print("TEST 1: Template Library Verification")
    print("=" * 60)

    lib = DigitTemplateLibrary()

    total = sum(len(t) for t in lib._templates.values())
    print(f"  Total templates loaded: {total}")

    for digit in '0123456789':
        templates = lib._templates.get(digit, [])
        print(f"  Digit '{digit}': {len(templates)} templates")

    print()
    return lib

def test_recognition_on_saved_images():
    """Test digit recognition on pre-saved digit images."""
    print("=" * 60)
    print("TEST 2: Direct Digit Recognition")
    print("=" * 60)

    lib = DigitTemplateLibrary()

    font_dir = PROJECT / "templates" / "fonts"
    results = []
    total = 0
    correct = 0

    for digit_dir in sorted(font_dir.iterdir()):
        if not digit_dir.is_dir():
            continue
        expected_digit = digit_dir.name

        for img_file in sorted(digit_dir.glob("*.png")):
            img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
            if img is None:
                continue

            total += 1
            char, conf = lib.recognize(img)

            is_correct = char == expected_digit
            if is_correct:
                correct += 1
            if not is_correct:
                results.append((img_file.name, expected_digit, char, conf))

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  Total templates tested: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")

    if results:
        print(f"\n  Misrecognitions ({len(results)}):")
        for name, expected, got, conf in results[:20]:
            print(f"    {name}: expected '{expected}', got '{got}' (conf={conf:.3f})")

    print()
    return accuracy

def test_end_to_end_with_images():
    """Test full pipeline on saved number region images."""
    print("=" * 60)
    print("TEST 3: End-to-End Digit Extraction + Recognition")
    print("=" * 60)

    screenshots_dir = PROJECT / "screenshots"
    if not screenshots_dir.exists():
        print("  No screenshots directory, skipping")
        return

    test_images = list(screenshots_dir.glob("*full_numbers*.png"))
    test_images += list(screenshots_dir.glob("*numbers_*.png"))

    if not test_images:
        print("  No test images found, skipping")
        return

    lib = DigitTemplateLibrary()

    for img_path in sorted(test_images):
        print(f"\n  Image: {img_path.name}")
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print("    Cannot read")
            continue

        h, w = img.shape[:2]
        print(f"    Size: {w}x{h}")

        # Step 1: Extract white regions
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, WHITE_HSV_LOWER, WHITE_HSV_UPPER)

        # Step 2: Try multiple thresholds
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best_regions = []

        for thresh in [120, 140, 160, 180, 200]:
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            regions = []
            for lid in range(1, num_labels):
                x = stats[lid, cv2.CC_STAT_LEFT]
                y = stats[lid, cv2.CC_STAT_TOP]
                ww = stats[lid, cv2.CC_STAT_WIDTH]
                hh = stats[lid, cv2.CC_STAT_HEIGHT]
                area = stats[lid, cv2.CC_STAT_AREA]

                if area < 5 or ww < 2 or hh < 3:
                    continue

                aspect = ww / hh if hh > 0 else 0
                if aspect < 0.08 or aspect > 5.0:
                    continue

                regions.append((x, y, ww, hh, aspect))

            if len(regions) >= 2 and len(regions) > len(best_regions):
                best_regions = regions

            if len(regions) >= 4:
                break

        print(f"    Found {len(best_regions)} digit regions")

        # Step 3: Recognize each region
        recognized = []
        for i, (x, y, ww, hh, aspect) in enumerate(best_regions):
            pad = 2
            dx1, dy1 = max(0, x - pad), max(0, y - pad)
            dx2 = min(w, x + ww + pad)
            dy2 = min(h, y + hh + pad)
            digit_img = img[dy1:dy2, dx1:dx2]

            char, conf = lib.recognize(digit_img)

            aspect_narrow = aspect < 0.25 and hh > 5

            if aspect_narrow:
                recognized.append('/')
                print(f"      [{i}] x={x}, y={y}, w={ww}, h={hh}, aspect={aspect:.2f} => '/' (narrow)")
            else:
                recognized.append(char or '?')
                print(f"      [{i}] x={x}, y={y}, w={ww}, h={hh}, aspect={aspect:.2f} => '{char}' (conf={conf:.3f})")

        result = ''.join(recognized)
        print(f"    Recognized: '{result}'")

        # Step 4: Try splitting large regions
        for i, (x, y, ww, hh, aspect) in enumerate(best_regions):
            area = ww * hh
            if (ww > hh * 1.8 and ww > 25) or area > 1500:
                print(f"    ⚠️  Region [{i}] might be merged (area={area}, aspect={aspect:.2f})")

    print()

def test_full_pipeline():
    """Test the full StaminaReader pipeline."""
    print("=" * 60)
    print("TEST 4: Full StaminaReader Pipeline")
    print("=" * 60)

    screenshots_dir = PROJECT / "screenshots"
    if not screenshots_dir.exists():
        print("  No screenshots directory, skipping")
        return

    screen_files = sorted(screenshots_dir.glob("*stamina1_full*"))
    screen_files += sorted(screenshots_dir.glob("*stamina2_full*"))

    if not screen_files:
        print("  No screen images found, skipping")
        return

    reader = StaminaReader()

    for screen_file in screen_files[:2]:
        print(f"\n  Processing: {screen_file.name}")
        screen = cv2.imread(str(screen_file), cv2.IMREAD_COLOR)
        if screen is None:
            print("    Cannot read")
            continue

        reader.height, reader.width = screen.shape[:2]

        stamina1 = reader.read_stamina1(screen)
        stamina2 = reader.read_stamina2(screen)

        print(f"    Stamina1: {stamina1.current}/{stamina1.max} (conf={stamina1.confidence:.3f})")
        print(f"    Stamina2: {stamina2.current}/{stamina2.max} (conf={stamina2.confidence:.3f})")

    print()

if __name__ == "__main__":
    test_template_library()
    test_recognition_on_saved_images()
    test_end_to_end_with_images()
    test_full_pipeline()
    print("=" * 60)
    print("ALL TESTS COMPLETE")