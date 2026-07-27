"""Test EasyOCR directly on cropped stamina images"""
import cv2
import numpy as np
from pathlib import Path

def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

print("Initializing EasyOCR reader...")
try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR initialized successfully!")
except Exception as e:
    print(f"EasyOCR init failed: {e}")
    sys.exit(1)

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = read_image(img_path)
    if img is None:
        print(f"\n{stamina_type}: cannot read image")
        continue
    
    print(f"\n{'='*60}")
    print(f"{stamina_type}: {img.shape}")
    print(f"{'='*60}")
    
    # Try EasyOCR
    try:
        results = reader.readtext(img)
        print(f"\nEasyOCR results ({len(results)}):")
        for bbox, text, conf in results:
            print(f"  Text: '{text}' (conf={conf:.2f})")
            print(f"  BBox: {[[int(c) for c in pt] for pt in bbox]}")
    except Exception as e:
        print(f"EasyOCR failed: {e}")
    
    # Also try grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        results2 = reader.readtext(gray)
        print(f"\nGrayscale EasyOCR results ({len(results2)}):")
        for bbox, text, conf in results2:
            print(f"  Text: '{text}' (conf={conf:.2f})")
    except Exception as e:
        print(f"Grayscale EasyOCR failed: {e}")
    
    # Try enlarged image
    scale = 3
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
    try:
        results3 = reader.readtext(enlarged)
        print(f"\nEnlarged ({scale}x) EasyOCR results ({len(results3)}):")
        for bbox, text, conf in results3:
            print(f"  Text: '{text}' (conf={conf:.2f})")
    except Exception as e:
        print(f"Enlarged EasyOCR failed: {e}")

print("\nDone!")