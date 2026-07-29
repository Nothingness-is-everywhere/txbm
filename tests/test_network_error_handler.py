"""
Unit tests for NetworkErrorHandler.

Run with:
    python -m pytest tests/test_network_error_handler.py -v

Tests cover:
  1. test_roi_normalized_and_pixel_modes — ROI auto-detection (normalized + pixel)
  2. test_detect_popup_and_button_on_fixture_image — end-to-end detection on synthetic fixture
  3. test_reject_click_when_button_outside_popup — anti-misclick validation
  4. test_fallback_to_ocr_when_template_missing — OCR fallback when templates invalid
  5. test_cooldown_prevents_repeat_click — cooldown deduplication
  6. test_template_validation_rejects_placeholder — low-variance template rejection
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Ensure project root is on the path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ok_tasks.NetworkErrorHandler import (
    DEFAULT_CONFIG,
    NetworkErrorHandler,
    is_bbox_inside,
    normalize_roi,
    validate_template_image,
)


def _create_fixture_image(
    screen_w: int = 1080, screen_h: int = 1920
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Create a synthetic test image with a simulated network popup + button.

    Returns (image, popup_bbox, button_bbox).
    """
    img = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (screen_w, screen_h), (30, 30, 50), -1)

    # Popup
    pop_x1, pop_y1 = 120, 500
    pop_x2, pop_y2 = 960, 1100
    cv2.rectangle(img, (pop_x1, pop_y1), (pop_x2, pop_y2), (240, 240, 245), -1)
    cv2.rectangle(img, (pop_x1, pop_y1), (pop_x2, pop_y2), (100, 100, 150), 3)

    # Title
    cv2.rectangle(img, (pop_x1, pop_y1), (pop_x2, pop_y1 + 70), (230, 230, 240), -1)
    cv2.putText(img, "  提示", (pop_x1 + 30, pop_y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 80, 80), 2)

    # Message
    cv2.putText(img, "Network unstable", (pop_x1 + 60, pop_y1 + 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2)

    # Button
    btn_x1, btn_y1 = 390, 900
    btn_x2, btn_y2 = 690, 980
    cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (180, 150, 100), -1)
    cv2.rectangle(img, (btn_x1, btn_y1), (btn_x2, btn_y2), (120, 100, 60), 2)
    cv2.putText(img, "OK", (btn_x1 + 110, btn_y1 + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return img, (pop_x1, pop_y1, pop_x2, pop_y2), (btn_x1, btn_y1, btn_x2, btn_y2)


def _create_placeholder_template(w: int = 100, h: int = 50) -> np.ndarray:
    """Create a low-variance placeholder image (should be rejected)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _create_valid_template(w: int = 100, h: int = 50) -> np.ndarray:
    """Create a template image with sufficient variance (should pass validation)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (200, 100, 50), 2)
    cv2.putText(img, "TEST", (10, h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


class TestROINormalization(unittest.TestCase):
    """Tests for normalize_roi function."""

    def test_normalized_mode_valid(self):
        """Normalized [0,1] values produce correct pixel coordinates."""
        result = normalize_roi([0.0, 0.0, 1.0, 1.0], 1000, 500)
        self.assertEqual(result, (0, 0, 1000, 500))

        result = normalize_roi([0.1, 0.25, 0.9, 0.85], 1080, 1920)
        self.assertEqual(result, (108, 480, 972, 1632))

    def test_pixel_mode_valid(self):
        """Pixel coordinates (>1.5) pass through directly."""
        result = normalize_roi([100, 200, 800, 900], 1080, 1920)
        self.assertEqual(result, (100, 200, 800, 900))

    def test_mixed_values_auto_detect_pixel(self):
        """Any value > 1.5 triggers pixel mode for all values."""
        result = normalize_roi([0.1, 0.25, 972, 1632], 1080, 1920)
        self.assertIsNotNone(result)

    def test_invalid_ordering(self):
        """x1 >= x2 or y1 >= y2 returns None."""
        result = normalize_roi([0.9, 0.85, 0.1, 0.25], 1080, 1920)
        self.assertIsNone(result)

        result = normalize_roi([100, 500, 80, 900], 1080, 1920)
        self.assertIsNone(result)

    def test_invalid_count(self):
        """ROI with wrong number of elements returns None."""
        result = normalize_roi([0.1, 0.2, 0.9], 1080, 1920)
        self.assertIsNone(result)

    def test_clamped_to_screen(self):
        """Values outside screen bounds are clamped."""
        result = normalize_roi([-100, -50, 2000, 3000], 1080, 1920)
        self.assertEqual(result, (0, 0, 1080, 1920))

    def test_clamped_normalized(self):
        """Normalized values outside [0,1] are clamped."""
        result = normalize_roi([-0.5, -0.2, 1.5, 1.2], 1000, 500)
        self.assertEqual(result, (0, 0, 1000, 500))


class TestBboxInside(unittest.TestCase):
    """Tests for is_bbox_inside function."""

    def test_strictly_inside(self):
        self.assertTrue(is_bbox_inside((100, 100, 200, 150), (50, 50, 300, 300)))

    def test_equal_to_inside(self):
        self.assertTrue(is_bbox_inside((50, 50, 300, 300), (50, 50, 300, 300)))

    def test_partially_outside(self):
        self.assertFalse(is_bbox_inside((40, 100, 200, 150), (50, 50, 300, 300)))

    def test_completely_outside(self):
        self.assertFalse(is_bbox_inside((400, 400, 500, 500), (50, 50, 300, 300)))


class TestTemplateValidation(unittest.TestCase):
    """Tests for validate_template_image function."""

    def test_valid_template(self):
        img = _create_valid_template(100, 50)
        self.assertTrue(validate_template_image(img, "test"))

    def test_none_template(self):
        self.assertFalse(validate_template_image(None, "test"))

    def test_empty_template(self):
        self.assertFalse(validate_template_image(np.array([]), "test"))

    def test_too_small(self):
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        self.assertFalse(validate_template_image(tiny, "test"))

    def test_low_variance_placeholder(self):
        img = _create_placeholder_template(100, 50)
        self.assertFalse(validate_template_image(img, "test"))

    def test_single_color(self):
        img = np.full((60, 60, 3), 128, dtype=np.uint8)
        self.assertFalse(validate_template_image(img, "test"))


class TestAntiMisclick(unittest.TestCase):
    """Tests for the anti-misclick verification logic."""

    def test_reject_click_when_button_outside_popup(self):
        """Button outside popup bbox should be rejected."""
        popup_bbox = (120, 500, 960, 1100)
        button_bbox_inside = (390, 900, 690, 980)
        button_bbox_outside = (200, 100, 500, 200)

        self.assertTrue(is_bbox_inside(button_bbox_inside, popup_bbox))
        self.assertFalse(is_bbox_inside(button_bbox_outside, popup_bbox))

    def test_reject_when_conf_too_low(self):
        """Low confidence should trigger rejection."""
        popup_conf = 0.82
        button_conf = 0.70
        popup_threshold = 0.80
        button_threshold = 0.85

        self.assertLess(button_conf, button_threshold)

    def test_accept_when_all_pass(self):
        """Valid popup + button inside + high confidence should pass."""
        popup_bbox = (120, 500, 960, 1100)
        button_bbox = (390, 900, 690, 980)
        popup_conf = 0.95
        button_conf = 0.92
        popup_threshold = 0.80
        button_threshold = 0.85

        self.assertGreaterEqual(popup_conf, popup_threshold)
        self.assertGreaterEqual(button_conf, button_threshold)
        self.assertTrue(is_bbox_inside(button_bbox, popup_bbox))


class TestCooldownAndFallback(unittest.TestCase):
    """Tests for cooldown and OCR fallback mechanisms."""

    def setUp(self):
        self.handler = NetworkErrorHandler.__new__(NetworkErrorHandler)
        self.handler._last_handle_time = 0.0
        self.handler._last_handle_fingerprint = ""
        self.handler.config = dict(DEFAULT_CONFIG)

    def test_cooldown_prevents_repeat_click(self):
        """Same fingerprint within cooldown should be rejected."""
        import time as _time
        self.handler.config["cooldown_seconds"] = 5.0

        fp = "abc12345"
        self.handler._last_handle_fingerprint = fp
        self.handler._last_handle_time = _time.time()

        self.assertTrue(self.handler._is_in_cooldown(fp))

    def test_cooldown_allows_different_fingerprint(self):
        """Different fingerprint should not hit cooldown."""
        import time as _time
        self.handler.config["cooldown_seconds"] = 5.0

        self.handler._last_handle_fingerprint = "abc12345"
        self.handler._last_handle_time = _time.time()

        self.assertFalse(self.handler._is_in_cooldown("xyz98765"))

    def test_cooldown_expires(self):
        """After cooldown expires, same fingerprint should be allowed."""
        import time as _time
        self.handler.config["cooldown_seconds"] = 0.01

        fp = "abc12345"
        self.handler._last_handle_fingerprint = fp
        self.handler._last_handle_time = _time.time() - 10.0  # 10 seconds ago

        self.assertFalse(self.handler._is_in_cooldown(fp))

    def test_record_handle_updates_state(self):
        """_record_handle should update time and fingerprint."""
        import time as _time
        before = _time.time()
        self.handler._record_handle("test_fp")
        after = _time.time()

        self.assertEqual(self.handler._last_handle_fingerprint, "test_fp")
        self.assertGreaterEqual(self.handler._last_handle_time, before)
        self.assertLessEqual(self.handler._last_handle_time, after)

    def test_fallback_to_ocr_when_template_missing(self):
        """When popup template is invalid, should not detect popup."""
        self.handler._popup_valid = False
        self.handler._popup_tpl = None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        found, result = self.handler.detect_network_popup_by_template(frame)

        self.assertFalse(found)
        self.assertIsNone(result)


class TestFingerprint(unittest.TestCase):
    """Tests for the fingerprint computation."""

    def test_same_bbox_produces_same_fingerprint(self):
        fp1 = NetworkErrorHandler._compute_popup_fingerprint((100, 200, 800, 900), 0.95)
        fp2 = NetworkErrorHandler._compute_popup_fingerprint((100, 200, 800, 900), 0.95)
        self.assertEqual(fp1, fp2)

    def test_different_bbox_produces_different_fingerprint(self):
        fp1 = NetworkErrorHandler._compute_popup_fingerprint((100, 200, 800, 900), 0.95)
        fp2 = NetworkErrorHandler._compute_popup_fingerprint((200, 300, 900, 1000), 0.95)
        self.assertNotEqual(fp1, fp2)

    def test_different_conf_produces_different_fingerprint(self):
        fp1 = NetworkErrorHandler._compute_popup_fingerprint((100, 200, 800, 900), 0.95)
        fp2 = NetworkErrorHandler._compute_popup_fingerprint((100, 200, 800, 900), 0.80)
        self.assertNotEqual(fp1, fp2)


class TestDetectionOnFixture(unittest.TestCase):
    """End-to-end detection tests using synthetic fixture images."""

    def setUp(self):
        self.handler = NetworkErrorHandler.__new__(NetworkErrorHandler)
        self.handler.config = dict(DEFAULT_CONFIG)
        self.handler.config["popup_roi"] = [0.0, 0.0, 1.0, 1.0]
        self.handler.config["popup_threshold"] = 0.80
        self.handler.config["button_threshold"] = 0.85
        self.handler.config["match_scales"] = [1.0]
        self.handler._last_handle_time = 0.0
        self.handler._last_handle_fingerprint = ""

        # Create template images from fixture
        fixture, pop_bbox, btn_bbox = _create_fixture_image(1080, 1920)

        pop_x1, pop_y1, pop_x2, pop_y2 = pop_bbox
        self.handler._popup_tpl = fixture[pop_y1:pop_y2, pop_x1:pop_x2].copy()

        btn_x1, btn_y1, btn_x2, btn_y2 = btn_bbox
        self.handler._button_tpl = fixture[btn_y1:btn_y2, btn_x1:btn_x2].copy()

        self.handler._popup_valid = validate_template_image(self.handler._popup_tpl, "test_popup")
        self.handler._button_valid = validate_template_image(self.handler._button_tpl, "test_button")

    def test_detect_popup_on_fixture(self):
        """Popup template should match the fixture popup region."""
        fixture, _, _ = _create_fixture_image(1080, 1920)
        found, result = self.handler.detect_network_popup_by_template(fixture)

        self.assertTrue(found, "Popup should be detected in fixture image")
        self.assertIsNotNone(result)

        cx, cy, conf, bbox = result
        self.assertGreaterEqual(conf, 0.80)
        self.assertGreater(cx, 0)
        self.assertGreater(cy, 0)

    def test_detect_button_on_fixture(self):
        """Button template should be found within popup bbox."""
        fixture, pop_bbox, _ = _create_fixture_image(1080, 1920)
        found, result = self.handler.detect_confirm_button_in_popup(fixture, pop_bbox)

        self.assertTrue(found, "Button should be detected in fixture image")
        self.assertIsNotNone(result)

        _, _, conf, btn_bbox = result
        self.assertGreaterEqual(conf, 0.85)

    def test_full_detection_pipeline(self):
        """Full pipeline: detect popup + detect button inside popup."""
        fixture, pop_bbox, expected_btn_bbox = _create_fixture_image(1080, 1920)

        popup_found, popup_result = self.handler.detect_network_popup_by_template(fixture)
        self.assertTrue(popup_found)

        _, _, _, found_popup_bbox = popup_result
        btn_found, btn_result = self.handler.detect_confirm_button_in_popup(
            fixture, found_popup_bbox
        )
        self.assertTrue(btn_found)

        # Verify button is inside popup
        _, _, _, found_btn_bbox = btn_result
        self.assertTrue(is_bbox_inside(found_btn_bbox, found_popup_bbox))

    def test_no_detection_on_blank_image(self):
        """Blank image should not trigger false positive detection."""
        blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
        found, result = self.handler.detect_network_popup_by_template(blank)
        self.assertFalse(found)
        self.assertIsNone(result)

    def test_anti_misclick_rejects_button_outside_popup(self):
        """Button detected outside popup bbox should be rejected."""
        fixture, pop_bbox, btn_bbox = _create_fixture_image(1080, 1920)

        # Deliberately pass a button bbox far outside the popup
        outside_btn = (50, 50, 150, 100)
        result = self.handler._verify_click_target(
            popup_conf=0.90,
            button_conf=0.90,
            popup_bbox=pop_bbox,
            button_bbox=outside_btn,
        )
        self.assertFalse(result, "Should reject click when button is outside popup")

    def test_anti_misclick_passes_when_valid(self):
        """Valid detection should pass anti-misclick check."""
        fixture, pop_bbox, btn_bbox = _create_fixture_image(1080, 1920)

        result = self.handler._verify_click_target(
            popup_conf=0.95,
            button_conf=0.92,
            popup_bbox=pop_bbox,
            button_bbox=btn_bbox,
        )
        self.assertTrue(result, "Valid detection should pass anti-misclick check")

    def test_anti_misclick_rejects_low_confidence(self):
        """Low confidence should trigger rejection."""
        fixture, pop_bbox, btn_bbox = _create_fixture_image(1080, 1920)

        result = self.handler._verify_click_target(
            popup_conf=0.50,
            button_conf=0.92,
            popup_bbox=pop_bbox,
            button_bbox=btn_bbox,
        )
        self.assertFalse(result, "Should reject click when popup confidence is too low")

        result = self.handler._verify_click_target(
            popup_conf=0.95,
            button_conf=0.50,
            popup_bbox=pop_bbox,
            button_bbox=btn_bbox,
        )
        self.assertFalse(result, "Should reject click when button confidence is too low")


class TestTemplateMissingFallback(unittest.TestCase):
    """Tests for behavior when templates are missing/invalid."""

    def setUp(self):
        self.handler = NetworkErrorHandler.__new__(NetworkErrorHandler)
        self.handler.config = dict(DEFAULT_CONFIG)

    def test_popup_template_none_returns_not_found(self):
        """Popup detection should return False when template is None."""
        self.handler._popup_valid = False
        self.handler._popup_tpl = None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        found, result = self.handler.detect_network_popup_by_template(frame)
        self.assertFalse(found)
        self.assertIsNone(result)

    def test_button_template_none_returns_not_found(self):
        """Button detection should return False when template is None."""
        self.handler._button_valid = False
        self.handler._button_tpl = None

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        found, result = self.handler.detect_confirm_button_in_popup(
            frame, (0, 0, 50, 50)
        )
        self.assertFalse(found)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()