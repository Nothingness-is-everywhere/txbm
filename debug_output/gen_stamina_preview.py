# -*- coding: utf-8 -*-
"""Generate a preview image with the two stamina selection regions annotated."""
import subprocess
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# From stamina_reader.STAMINA_ROI_CONFIG (rx, ry, rw, rh)
regions = [
    ("Expedition Stamina (max=150)", 0.7435, 0.7474, 0.0500, 0.0172, (255, 0, 0)),
    ("Training Stamina (max=50)",    0.7981, 0.8396, 0.0944, 0.0234, (0, 255, 0)),
]

img = None
# Try a real ADB screenshot first
ADB = r"D:\MuMu\nx_main\adb.exe"
for cmd in ([ADB, "-s", "emulator-5554", "exec-out", "screencap", "-p"],
            [ADB, "-s", "127.0.0.1:5555", "exec-out", "screencap", "-p"],
            [ADB, "-s", "127.0.0.1:16384", "exec-out", "screencap", "-p"]):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=10)
        if r.returncode == 0 and len(r.stdout) > 5000:
            img = Image.open(BytesIO(r.stdout))
            print("ADB screenshot OK", cmd[1], img.size)
            break
        else:
            print("ADB no output", cmd[1], r.returncode, len(r.stdout), r.stderr[:200])
    except Exception as e:
        print("ADB failed:", cmd[1], e)

if img is None:
    img = Image.new("RGB", (1080, 1920), (50, 50, 50))
    print("Using placeholder 1080x1920")

W, H = img.size
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 32)
    font_small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
except Exception:
    font = ImageFont.load_default()
    font_small = font

for name, rx, ry, rw, rh, color in regions:
    x1 = int(rx * W)
    y1 = int(ry * H)
    x2 = int((rx + rw) * W)
    y2 = int((ry + rh) * H)
    # thick rectangle outline
    for t in range(5):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
    # labels
    draw.text((x1, max(0, y1 - 42)), name, fill=color, font=font)
    draw.text((x1, min(H - 30, y2 + 8)),
              f"({x1},{y1})-({x2},{y2})  {W}x{H}", fill=color, font=font_small)

out = r"d:\学习\python\ok-script\debug_output\stamina_regions_preview.png"
img.save(out)
print("Saved:", out, img.size)
