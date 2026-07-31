# -*- coding: utf-8 -*-
"""Capture the profile-button region as the home template."""
import subprocess
from io import BytesIO
from PIL import Image

ADB = r"D:\MuMu\nx_main\adb.exe"
r = subprocess.run([ADB, "-s", "emulator-5554", "exec-out", "screencap", "-p"],
                   capture_output=True, timeout=10)
img = Image.open(BytesIO(r.stdout))
print("Full screen:", img.size)

# Selection: x=37,y=69,w=279,h=107  ->  (37,69)-(316,176)
crop = img.crop((37, 69, 316, 176))
crop.save(r"d:\学习\python\ok-script\templates\home_profile_button.png")
print("Template saved:", crop.size)

# Save current full screen for the user to confirm we are on the home page
img.save(r"d:\学习\python\ok-script\debug_output\home_current.png")
print("Current screen saved to debug_output/home_current.png")
