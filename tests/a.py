import cv2
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

def read_image(path):
    with open(str(path), 'rb') as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def recognize_fraction_tesseract(img_path):
    img = read_image(img_path)
    if img is None:
        return "Image not found"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/'
    txt = pytesseract.image_to_string(th, config=config)

    m = re.search(r'\d+/\d+', txt.replace(" ", ""))
    return m.group(0) if m else txt.strip()

if __name__ == "__main__":
    print(recognize_fraction_tesseract("test.png"))
