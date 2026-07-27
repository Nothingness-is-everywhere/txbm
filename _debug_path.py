"""Debug path resolution."""
from pathlib import Path
import os

p = Path('ok/automation/stamina_reader.py').resolve()
print(f"File: {p}")
print(f"File exists: {p.exists()}")

root = p.parent.parent.parent.parent
print(f"PROJECT_ROOT: {root}")
print(f"PROJECT_ROOT exists: {root.exists()}")

fonts_dir = root / "templates" / "fonts"
print(f"\nfonts_dir: {fonts_dir}")
print(f"fonts_dir exists: {fonts_dir.exists()}")

if fonts_dir.exists():
    for digit_dir in sorted(fonts_dir.iterdir()):
        if digit_dir.is_dir():
            files = list(digit_dir.glob("*.png"))
            print(f"  {digit_dir.name}/: {len(files)} png files")
            if files:
                print(f"    First: {files[0].name}")

# Check the actual code's PROJECT_ROOT
print("\n" + "="*50)
print("Checking actual stamina_reader.py logic...")

from ok.automation.stamina_reader import PROJECT_ROOT as ACTUAL_ROOT
print(f"ACTUAL_PROJECT_ROOT: {ACTUAL_ROOT}")
print(f"ACTUAL_PROJECT_ROOT exists: {ACTUAL_ROOT.exists()}")

fonts_dir2 = ACTUAL_ROOT / "templates" / "fonts"
print(f"fonts_dir (from module): {fonts_dir2}")
print(f"fonts_dir exists: {fonts_dir2.exists()}")

if fonts_dir2.exists():
    for digit_dir in sorted(fonts_dir2.iterdir()):
        if digit_dir.is_dir() and digit_dir.name in '0123456789':
            files = list(digit_dir.rglob("*.png"))
            print(f"  {digit_dir.name}/: {len(files)} png files (rglob)")