"""使用EasyOCR识别"""
import easyocr
import cv2
import numpy as np

print("Initializing EasyOCR (first time may download models)...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("EasyOCR initialized!")

for stamina_type in ["expedition", "training"]:
    img_path = f"screenshots/stamina_crops/{stamina_type}_final_v1.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"{stamina_type}: 无法读取图片")
        continue
    
    results = reader.readtext(img)
    print(f"\n{stamina_type}:")
    for bbox, text, conf in results:
        print(f"  Text: '{text}', Confidence: {conf:.2f}")
    
    if results:
        full_text = " ".join([r[1] for r in results])
        print(f"  Full text: '{full_text}'")