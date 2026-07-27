"""Test actual pipeline methods on saved images."""
import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

from ok.automation.stamina_reader import StaminaReader, _imread

screenshots = PROJECT / "screenshots"

reader = StaminaReader()

# Test with stamina1_full_numbers.png
img1 = _imread(str(screenshots / "stamina1_full_numbers.png"), cv2.IMREAD_COLOR)
print(f"stamina1_full_numbers: {img1.shape}")
result1 = reader._read_numbers_from_region(img1)
print(f"  Result: '{result1}'")
if result1:
    current, max_val = reader._parse_stamina_text(result1)
    print(f"  Parsed: {current}/{max_val}")

# Test with stamina2_full_numbers.png
img2 = _imread(str(screenshots / "stamina2_full_numbers.png"), cv2.IMREAD_COLOR)
print(f"\nstamina2_full_numbers: {img2.shape}")
result2 = reader._read_numbers_from_region(img2)
print(f"  Result: '{result2}'")
if result2:
    current, max_val = reader._parse_stamina_text(result2)
    print(f"  Parsed: {current}/{max_val}")

# Test with full screen
img3 = _imread(str(screenshots / "full_screen_20260726_203459.png"), cv2.IMREAD_COLOR)
print(f"\nfull_screen: {img3.shape}")
reader.height, reader.width = img3.shape[:2]

s1 = reader.read_stamina1(img3)
print(f"  Stamina1: {s1.current}/{s1.max}")

s2 = reader.read_stamina2(img3)
print(f"  Stamina2: {s2.current}/{s2.max}")

print("\nDone!")