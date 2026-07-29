"""
Template extraction utility for NetworkErrorHandler.

Allows interactively cropping the network popup and confirm button
from a game screenshot to create template images.

Usage:
  python _extract_network_templates.py --image PATH_TO_SCREENSHOT

The script opens the screenshot in a window. Draw rectangles to select:
  1. First rectangle: the full popup dialog → saved as templates/network_popup.png
  2. Second rectangle: the "确定" button     → saved as templates/network_confirm_button.png

After drawing each rectangle:
  - Press 's' to save the selection as a template
  - Press 'c' to cancel and redraw
  - Press 'q' to quit
  - Press 'h' for help
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np


# Default output paths
POPUP_TEMPLATE_PATH = "templates/network_popup.png"
BUTTON_TEMPLATE_PATH = "templates/network_confirm_button.png"


class TemplateExtractor:
    """Interactive template extractor using OpenCV window."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = self._load_image(image_path)
        if self.image is None:
            sys.exit(1)

        self.display = self.image.copy()
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.selections = []
        self.current_selection = None
        self.step = 0  # 0: select popup, 1: select button
        self.popup_template = None
        self.button_template = None

        self.window_name = "Template Extractor - Network Error Handler"

    def _load_image(self, path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            print(f"[ERROR] File not found: {path}")
            return None
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[ERROR] Failed to decode: {path}")
                return None
            print(f"✓ Loaded: {path} ({img.shape[1]}x{img.shape[0]})")
            return img
        except Exception as e:
            print(f"[ERROR] {e}")
            return None

    def _draw_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing rectangles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.fx, self.fy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.display = self.image.copy()
                # Redraw previous selections
                for sel, label, color in self.selections:
                    x1, y1, x2, y2 = sel
                    cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(self.display, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Draw current rectangle
                cv2.rectangle(self.display, (self.ix, self.iy), (x, y), (0, 255, 255), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            self.current_selection = (min(self.ix, self.fx), min(self.iy, self.fy),
                                     max(self.ix, self.fx), max(self.iy, self.fy))
            # Redraw with all selections
            self.display = self.image.copy()
            for sel, label, color in self.selections:
                x1, y1, x2, y2 = sel
                cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Draw current
            if self.current_selection:
                x1, y1, x2, y2 = self.current_selection
                cv2.rectangle(self.display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label = "popup" if self.step == 0 else "confirm_button"
                cv2.putText(self.display, f"[{label}] Press 's' to save",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_instructions(self):
        """Draw instruction overlay on the image."""
        overlay = self.display.copy()
        h, w = overlay.shape[:2]

        # Semi-transparent instruction panel at top
        panel_h = 90
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, self.display, 0.4, 0, self.display)

        instructions = [
            f"Step {self.step + 1}/2: {'Select the FULL POPUP dialog' if self.step == 0 else 'Select the CONFIRM (确定) button'}",
            "  Drag a rectangle with left mouse button",
            "  's' = save selection  |  'c' = cancel  |  'h' = help  |  'q' = quit",
        ]

        for i, line in enumerate(instructions):
            cv2.putText(self.display, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(self.window_name, self.display)

    def _save_current(self):
        """Save the current selection as a template."""
        if self.current_selection is None:
            print("No selection to save")
            return

        x1, y1, x2, y2 = self.current_selection
        region = self.image[y1:y2, x1:x2]

        if region.size == 0:
            print("Empty selection, try again")
            self.current_selection = None
            return

        if self.step == 0:
            # Save as popup template
            Path("templates").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(POPUP_TEMPLATE_PATH, region)
            print(f"✓ Popup template saved: {POPUP_TEMPLATE_PATH} ({region.shape[1]}x{region.shape[0]})")
            self.selections.append((self.current_selection, "popup", (0, 255, 0)))
            self.popup_template = region
            self.step = 1
            self.current_selection = None
            self.display = self.image.copy()
            for sel, label, color in self.selections:
                sx1, sy1, sx2, sy2 = sel
                cv2.rectangle(self.display, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(self.display, label, (sx1, sy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif self.step == 1:
            # Save as button template
            Path("templates").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(BUTTON_TEMPLATE_PATH, region)
            print(f"✓ Confirm button template saved: {BUTTON_TEMPLATE_PATH} ({region.shape[1]}x{region.shape[0]})")
            self.selections.append((self.current_selection, "confirm_button", (0, 0, 255)))
            self.button_template = region
            self.current_selection = None
            self.display = self.image.copy()
            for sel, label, color in self.selections:
                sx1, sy1, sx2, sy2 = sel
                cv2.rectangle(self.display, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(self.display, label, (sx1, sy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print("\n✓ All templates extracted successfully!")
            print(f"  Popup: {POPUP_TEMPLATE_PATH}")
            print(f"  Button: {BUTTON_TEMPLATE_PATH}")
            print("\nRun the test to verify: python _test_network_error_template.py --image <screenshot>")

    def _cancel_current(self):
        """Cancel current selection."""
        self.current_selection = None
        self.display = self.image.copy()
        for sel, label, color in self.selections:
            sx1, sy1, sx2, sy2 = sel
            cv2.rectangle(self.display, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(self.display, label, (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def run(self):
        """Run the interactive extraction loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._draw_callback)

        print("\n" + "=" * 60)
        print("Template Extractor for NetworkErrorHandler")
        print("=" * 60)
        print(f"\nImage: {self.image_path}")
        print(f"Step 1: Draw rectangle around the FULL POPUP dialog")
        print(f"Step 2: Draw rectangle around the '确定' (confirm) button")
        print(f"\nControls:")
        print(f"  's' = save current selection as template")
        print(f"  'c' = cancel current selection")
        print(f"  'h' = show help")
        print(f"  'q' = quit\n")

        while True:
            self._draw_instructions()
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                self._save_current()
                if self.step > 1:
                    break
            elif key == ord('c'):
                self._cancel_current()
            elif key == ord('h'):
                print("\n--- Help ---")
                print("1. Click and drag to draw a rectangle")
                print("2. Press 's' to save the selected region as template")
                print("3. First save = popup template, Second save = button template")
                print("4. Press 'q' to quit at any time")
                print("-------------\n")
            elif key == ord('q'):
                print("Quit. No templates saved.")
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Extract network error popup templates from a screenshot"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the game screenshot showing the network error popup"
    )
    args = parser.parse_args()

    extractor = TemplateExtractor(args.image)
    extractor.run()


if __name__ == "__main__":
    main()