"""Quick test of template recognition accuracy using existing digit images."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import DigitTemplateLibrary, _imread

lib = DigitTemplateLibrary()

total = sum(len(t) for t in lib._templates.values())
print(f"Total templates loaded: {total}")
for d in '0123456789':
    n = len(lib._templates.get(d, []))
    print(f"  Digit '{d}': {n} templates")

print("\n" + "=" * 60)
print("Testing recognition accuracy on saved digit images")
print("=" * 60)

test_dirs = [
    PROJECT / "templates" / "digits",
    PROJECT / "templates" / "digit_templates",
]

correct = 0
total = 0
errors = []

for test_dir in test_dirs:
    if not test_dir.exists():
        continue

    for img_file in sorted(test_dir.glob("*.png")):
        img = _imread(str(img_file), cv2.IMREAD_COLOR)
        if img is None:
            continue

        name_lower = img_file.name.lower()

        expected = None
        for d in '0123456789':
            if f'_d{d}' in name_lower or f'digit_{d}' in name_lower or f'_{d}.' in name_lower:
                expected = d
                break

        if expected is None:
            continue

        total += 1
        char, conf = lib.recognize(img)

        if char == expected:
            correct += 1
        else:
            errors.append((img_file.name, expected, char or 'None', conf))

if total > 0:
    accuracy = correct / total * 100
    print(f"\n  Tested: {total} images")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1f}%")

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for name, exp, got, conf in errors:
            print(f"    {name}: expected '{exp}', got '{got}' (conf={conf:.3f})")
else:
    print("\n  No test images found with identifiable digit names")

print("\n" + "=" * 60)
print("Template recognition test complete")