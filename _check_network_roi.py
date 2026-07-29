"""Connect to ADB device and show network error detection ROI."""
import sys
import time
import cv2
import numpy as np
import adbutils

def main():
    # Connect to ADB
    client = adbutils.AdbClient()
    
    # Try multiple ports
    connected = False
    for port in [5555, 5554, 16384]:
        try:
            client.connect(f"127.0.0.1:{port}")
            print(f"Connected to 127.0.0.1:{port}")
            connected = True
            break
        except Exception as e:
            print(f"Port {port}: {e}")
    
    if not connected:
        print("Could not connect to any port, trying emulator-5554...")
        try:
            client.connect("emulator-5554")
            connected = True
        except Exception as e:
            print(f"emulator-5554: {e}")
    
    time.sleep(1.0)
    
    # Get device
    try:
        device = client.device()
        print(f"Device: {device}")
        print(f"State: {device.get_state()}")
    except Exception as e:
        print(f"No device found: {e}")
        print("\nTrying to list all devices...")
        devices = client.device_list()
        print(f"Device list: {devices}")
        
        # Try to get by serial
        for port in [5555, 5554, 16384]:
            try:
                dev = client.device(f"127.0.0.1:{port}")
                state = dev.get_state()
                print(f"  127.0.0.1:{port} -> state={state}")
                if state == 'device':
                    device = dev
                    break
            except Exception as e:
                print(f"  127.0.0.1:{port} -> {e}")
    
    if 'device' not in dir() or device is None:
        print("\nERROR: No device available. Cannot take screenshot.")
        print("Please ensure the emulator is running and accessible via ADB.")
        sys.exit(1)
    
    # Take screenshot
    print("\nTaking screenshot...")
    try:
        png_bytes = device.shell("screencap -p", encoding=None, timeout=15)
        if not png_bytes:
            print("Failed to capture screen (empty result)")
            sys.exit(1)
        
        image_data = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        print(f"Screen captured: {w}x{h}")
        
        # Draw network error detection ROI
        # popup_roi: [0.10, 0.25, 0.90, 0.85]
        roi_x1 = int(w * 0.10)
        roi_y1 = int(h * 0.25)
        roi_x2 = int(w * 0.90)
        roi_y2 = int(h * 0.85)
        
        result = image.copy()
        
        # Draw ROI rectangle
        cv2.rectangle(result, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
        cv2.putText(result, f'Popup ROI [0.10,0.25,0.90,0.85]', 
                   (roi_x1, roi_y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center point
        cx, cy = (roi_x1 + roi_x2) // 2, (roi_y1 + roi_y2) // 2
        cv2.circle(result, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(result, f'Center: ({cx},{cy})', (cx + 12, cy - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Also draw button search area (within popup, lower half)
        btn_x1 = roi_x1 + int((roi_x2 - roi_x1) * 0.08)
        btn_y1 = roi_y1 + int((roi_y2 - roi_y1) * 0.55)
        btn_x2 = roi_x2 - int((roi_x2 - roi_x1) * 0.08)
        btn_y2 = roi_y2 - int((roi_y2 - roi_y1) * 0.05)
        cv2.rectangle(result, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 0, 0), 2)
        cv2.putText(result, 'Button search area', (btn_x1, btn_y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show coordinates on screen
        cv2.putText(result, f'Screen: {w}x{h}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = f'screenshots/network_detection_{timestamp}.png'
        cv2.imwrite(path, result)
        print(f"Saved annotated screenshot: {path}")
        
        # Print ROI details
        print(f"\nNetwork Error Detection ROI Details:")
        print(f"  popup_roi (normalized): [0.10, 0.25, 0.90, 0.85]")
        print(f"  popup_roi (pixels): ({roi_x1}, {roi_y1}) -> ({roi_x2}, {roi_y2})")
        print(f"  popup_threshold: 0.80")
        print(f"  button_threshold: 0.85")
        print(f"  match_scales: [0.8, 0.9, 1.0, 1.1, 1.2]")
        print(f"  Button search area (pixels): ({btn_x1}, {btn_y1}) -> ({btn_x2}, {btn_y2})")
        
        # Check if popup templates exist and are valid
        import os
        for tpl_name in ['network_popup.png', 'network_confirm_button.png']:
            tpl_path = f'templates/{tpl_name}'
            if os.path.exists(tpl_path):
                tpl_img = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
                if tpl_img is not None:
                    gray = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY)
                    var = np.var(gray)
                    h, w = tpl_img.shape[:2]
                    print(f"\n  Template {tpl_name}: {w}x{h}, variance={var:.2f}")
                    if var < 5.0:
                        print(f"    ⚠️  WARNING: Template variance too low! Likely a placeholder.")
                        print(f"    ⚠️  Replace with actual screenshot of the network popup.")
                else:
                    print(f"\n  Template {tpl_name}: FAILED to load")
            else:
                print(f"\n  Template {tpl_name}: NOT FOUND")
        
    except Exception as e:
        print(f"Error during screenshot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()