try:
    import pytesseract
    print("pytesseract available")
except ImportError:
    print("pytesseract NOT available")

try:
    from PIL import Image
    print("PIL available")
except ImportError:
    print("PIL NOT available")

try:
    import easyocr
    print("easyocr available")
except ImportError:
    print("easyocr NOT available")

try:
    import numpy
    print("numpy available")
except ImportError:
    print("numpy NOT available")

try:
    import cv2
    print(f"opencv available: {cv2.__version__}")
except ImportError:
    print("opencv NOT available")