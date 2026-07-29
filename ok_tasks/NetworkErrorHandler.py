"""
Network Error Handler for 天下布魔 (Tianxia Bumo).

Detects and dismisses the "网络不稳定，将重新启动游戏" popup
using template matching (primary) with OCR fallback.

Detection flow:
  1. Template-match the full popup on the screenshot
  2. If popup found, search the "确定" button within popup ROI
  3. Anti-misclick: verify button bbox is inside popup bbox
  4. Click button center, verify dismissal, retry if needed
  5. If template matching fails, fall back to OCR-based detection
  6. Cooldown prevents re-clicking the same popup within a window

Configuration is loaded from configs/NetworkErrorHandler.json
(managed by the task config system).
"""

import hashlib
import time
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from ok import TriggerTask

logger = logging.getLogger("NetworkErrorHandler")

# ---------------------------------------------------------------------------
# Project root resolution (all paths relative to repo root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_path(relative_path: str) -> Path:
    """
    Resolve a repository-relative path to an absolute path.

    将仓库相对路径解析为绝对路径，确保跨机器可运行。
    """
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Default configuration (merged into task config at runtime)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "_enabled": False,
    "popup_template_path": "templates/network_popup.png",
    "confirm_button_template_path": "templates/network_confirm_button.png",
    "popup_threshold": 0.80,
    "button_threshold": 0.85,
    "match_scales": [0.8, 0.9, 1.0, 1.1, 1.2],
    "max_retries": 3,
    "retry_delay_min": 0.5,
    "retry_delay_max": 1.0,
    "post_click_wait_min": 0.5,
    "post_click_wait_max": 1.0,
    "fallback_to_ocr": True,
    "debug_save_images": True,
    "debug_output_dir": "artifacts/network_error",
    "popup_roi": [0.10, 0.25, 0.90, 0.85],
    "ocr_texts": ["网络不好", "网络异常", "连接失败", "网络连接", "无法连接", "确认", "确定"],
    "cooldown_seconds": 5.0,
    "template_min_var": 5.0,
    "template_min_size": 20,
}


def normalize_roi(
    roi: List[float], screen_w: int, screen_h: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    Normalize ROI to absolute pixel coordinates.

    Supports two formats:
      - Normalized [0, 1]: values like [0.10, 0.25, 0.90, 0.85]
      - Pixel coordinates [>1]: values like [108, 480, 972, 1632]

    Returns (x1, y1, x2, y2) in absolute pixels, or None on invalid input.
    最终保证 x1 < x2, y1 < y2。

    Args:
        roi: [x1, y1, x2, y2] either normalized (0-1) or pixel (>1).
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.

    Returns:
        (ax1, ay1, ax2, ay2) clamped to screen bounds, or None if invalid.
    """
    try:
        vals = [float(v) for v in roi]
    except (TypeError, ValueError):
        logger.error(f"Invalid ROI: cannot parse as float list: {roi}")
        return None

    if len(vals) != 4:
        logger.error(f"ROI must have exactly 4 elements, got {len(vals)}: {roi}")
        return None

    # Auto-detect format: if any value > 1.5, treat all as pixel coordinates
    is_pixel = any(v > 1.5 for v in vals)

    if is_pixel:
        ax1, ay1, ax2, ay2 = [int(round(v)) for v in vals]
    else:
        # Clamp normalized values to [0, 1]
        vals = [max(0.0, min(1.0, v)) for v in vals]
        ax1 = int(round(screen_w * vals[0]))
        ay1 = int(round(screen_h * vals[1]))
        ax2 = int(round(screen_w * vals[2]))
        ay2 = int(round(screen_h * vals[3]))

    # Clamp to screen bounds
    ax1 = max(0, min(screen_w - 1, ax1))
    ay1 = max(0, min(screen_h - 1, ay1))
    ax2 = max(0, min(screen_w, ax2))
    ay2 = max(0, min(screen_h, ay2))

    if ax1 >= ax2 or ay1 >= ay2:
        logger.error(
            f"Invalid ROI after normalization: ({ax1},{ay1},{ax2},{ay2}), "
            f"must have x1<x2 and y1<y2 (raw={roi}, is_pixel={is_pixel})"
        )
        return None

    return (ax1, ay1, ax2, ay2)


def validate_template_image(img: np.ndarray, label: str) -> bool:
    """
    Validate that a template image is suitable for production matching.

    验证模板图像的有效性：
      - 非空
      - 最小尺寸 >= template_min_size
      - 方差 >= template_min_var (排除纯色图)
      - 至少有一个通道的标准差足够

    Args:
        img: Template image in BGR format.
        label: Human-readable label for logging.

    Returns:
        True if valid, False otherwise.
    """
    if img is None or img.size == 0:
        logger.error(f"[{label}] Template is empty or None")
        return False

    h, w = img.shape[:2]
    min_size = DEFAULT_CONFIG["template_min_size"]
    if h < min_size or w < min_size:
        logger.error(f"[{label}] Template too small: {w}x{h}, min={min_size}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    min_var = DEFAULT_CONFIG["template_min_var"]
    if var < min_var:
        logger.error(
            f"[{label}] Template variance too low: {var:.2f} < {min_var:.1f}, "
            f"likely a solid-color placeholder. Replace with a real template."
        )
        return False

    logger.info(f"[{label}] Template valid: {w}x{h}, var={var:.2f}")
    return True


def is_bbox_inside(
    inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]
) -> bool:
    """
    Check if inner bbox is strictly inside outer bbox.

    检查内部 bbox 是否完全位于外部 bbox 内。
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


class NetworkErrorHandler(TriggerTask):
    """
    Detects and dismisses the network instability popup via template matching.

    定时触发的网络错误弹窗处理任务。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "网络错误处理"
        self.description = "检测并关闭网络不稳定弹窗（模板匹配优先，OCR回退）"
        self.trigger_interval = 2
        self.visible = True
        self.default_config = DEFAULT_CONFIG
        self._popup_tpl = None
        self._button_tpl = None
        self._popup_valid = False
        self._button_valid = False
        self._last_debug_path = None

        # Cooldown state: prevent re-clicking same popup within cooldown window
        self._last_handle_time: float = 0.0
        self._last_handle_fingerprint: str = ""

    def on_create(self):
        self._enabled = self.config.get("_enabled", False)
        self._load_templates()

    # ------------------------------------------------------------------
    # Template loading & validation
    # ------------------------------------------------------------------

    def _load_templates(self):
        """
        Load and validate popup and button template images.

        加载并验证模板图像的有效性。
        """
        popup_rel = self.config.get("popup_template_path", DEFAULT_CONFIG["popup_template_path"])
        button_rel = self.config.get(
            "confirm_button_template_path", DEFAULT_CONFIG["confirm_button_template_path"]
        )

        popup_abs = str(resolve_repo_path(popup_rel))
        button_abs = str(resolve_repo_path(button_rel))

        self._popup_tpl = self._safe_imread(popup_abs)
        self._button_tpl = self._safe_imread(button_abs)

        self._popup_valid = self._popup_tpl is not None and validate_template_image(
            self._popup_tpl, "popup"
        )
        self._button_valid = self._button_tpl is not None and validate_template_image(
            self._button_tpl, "confirm_button"
        )

        if not self._popup_valid:
            logger.error(
                "Popup template invalid or missing. "
                "Place a real template at '%s' or enable OCR fallback.",
                popup_rel,
            )
        if not self._button_valid:
            logger.error(
                "Confirm button template invalid or missing. "
                "Place a real template at '%s' or enable OCR fallback.",
                button_rel,
            )

    @staticmethod
    def _safe_imread(path: str) -> Optional[np.ndarray]:
        """
        Read an image file safely, handling Unicode paths on Windows.

        安全读取图片，处理 Windows 下的 Unicode 路径问题。
        """
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Fingerprint for cooldown
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_popup_fingerprint(
        bbox: Tuple[int, int, int, int], conf: float
    ) -> str:
        """Compute a short fingerprint for a popup detection result."""
        raw = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{conf:.2f}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def _is_in_cooldown(self, fingerprint: str) -> bool:
        """Check if the same popup was already handled within the cooldown window."""
        cooldown = self.config.get(
            "cooldown_seconds", DEFAULT_CONFIG["cooldown_seconds"]
        )
        now = time.time()
        if (
            fingerprint == self._last_handle_fingerprint
            and now - self._last_handle_time < cooldown
        ):
            logger.info(
                f"Cooldown hit: same popup fingerprint={fingerprint} "
                f"handled {now - self._last_handle_time:.1f}s ago, skipping"
            )
            return True
        return False

    def _record_handle(self, fingerprint: str):
        """Record that a popup was successfully handled."""
        self._last_handle_time = time.time()
        self._last_handle_fingerprint = fingerprint

    # ------------------------------------------------------------------
    # Template matching helpers
    # ------------------------------------------------------------------

    def _multi_scale_match(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        roi: List[float],
        threshold: float,
        scales: List[float],
        label: str,
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """
        Multi-scale template matching within an ROI (normalized or pixel).

        在 ROI 内进行多尺度模板匹配（支持归一化和像素两种模式）。

        Args:
            screen: Full screenshot in BGR format.
            template: Template image in BGR format.
            roi: ROI values [x1, y1, x2, y2], auto-detected format.
            threshold: Minimum confidence to accept.
            scales: List of scale factors to try.
            label: Human-readable label for logging.

        Returns:
            Tuple (center_x, center_y, confidence, (x1, y1, x2, y2)) or None.
            Coordinates are in absolute screen pixels.
        """
        if template is None:
            logger.warning(f"[{label}] Template is None, skipping")
            return None

        screen_h, screen_w = screen.shape[:2]
        tpl_h, tpl_w = template.shape[:2]

        pixel_roi = normalize_roi(roi, screen_w, screen_h)
        if pixel_roi is None:
            logger.warning(f"[{label}] Invalid ROI, skipping")
            return None

        roi_x1, roi_y1, roi_x2, roi_y2 = pixel_roi
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1

        if roi_w < tpl_w or roi_h < tpl_h:
            logger.warning(
                f"[{label}] ROI too small for template: "
                f"ROI={roi_w}x{roi_h}, tpl={tpl_w}x{tpl_h}"
            )
            return None

        roi_region = screen[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_gray = cv2.cvtColor(roi_region, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        best_val = 0.0
        best_loc = None
        best_scale = 1.0
        best_size = (tpl_w, tpl_h)

        for scale in scales:
            scaled_w = int(tpl_w * scale)
            scaled_h = int(tpl_h * scale)

            if scaled_w > roi_gray.shape[1] or scaled_h > roi_gray.shape[0]:
                continue
            if scaled_w < 4 or scaled_h < 4:
                continue

            tpl_scaled = cv2.resize(
                tpl_gray, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA
            )
            result = cv2.matchTemplate(
                roi_gray, tpl_scaled, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
                best_size = (scaled_w, scaled_h)

        if best_val >= threshold and best_loc is not None:
            cx = roi_x1 + best_loc[0] + best_size[0] // 2
            cy = roi_y1 + best_loc[1] + best_size[1] // 2
            bx1 = roi_x1 + best_loc[0]
            by1 = roi_y1 + best_loc[1]
            bx2 = bx1 + best_size[0]
            by2 = by1 + best_size[1]

            logger.info(
                f"[{label}] Found at ({cx}, {cy}) "
                f"conf={best_val:.3f} scale={best_scale:.2f} "
                f"bbox=({bx1},{by1})-({bx2},{by2})"
            )
            return (cx, cy, best_val, (bx1, by1, bx2, by2))
        else:
            logger.info(
                f"[{label}] Not found. "
                f"Max conf={best_val:.3f} (threshold={threshold})"
            )
            return None

    def _match_popup(
        self, screen: np.ndarray
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """Detect the full network error popup on screen."""
        roi = self.config.get("popup_roi", DEFAULT_CONFIG["popup_roi"])
        threshold = self.config.get("popup_threshold", DEFAULT_CONFIG["popup_threshold"])
        scales = self.config.get("match_scales", DEFAULT_CONFIG["match_scales"])

        return self._multi_scale_match(
            screen, self._popup_tpl, roi, threshold, scales, "network_popup"
        )

    def _match_confirm_button(
        self,
        screen: np.ndarray,
        popup_bbox: Tuple[int, int, int, int],
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """
        Find the "确定" button within the popup bounding box.

        在弹窗 bbox 内查找"确定"按钮。
        """
        bx1, by1, bx2, by2 = popup_bbox
        screen_h, screen_w = screen.shape[:2]

        margin_x = int((bx2 - bx1) * 0.08)
        margin_top = int((by2 - by1) * 0.55)
        margin_bottom = int((by2 - by1) * 0.05)

        # Build ROI in normalized form (auto-detected)
        btn_roi = [
            max(0.0, (bx1 + margin_x) / screen_w),
            max(0.0, (by1 + margin_top) / screen_h),
            min(1.0, (bx2 - margin_x) / screen_w),
            min(1.0, (by2 - margin_bottom) / screen_h),
        ]

        threshold = self.config.get("button_threshold", DEFAULT_CONFIG["button_threshold"])
        scales = self.config.get("match_scales", DEFAULT_CONFIG["match_scales"])

        return self._multi_scale_match(
            screen, self._button_tpl, btn_roi, threshold, scales, "confirm_button"
        )

    # ------------------------------------------------------------------
    # Detection interface (public, testable)
    # ------------------------------------------------------------------

    def detect_network_popup_by_template(
        self, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]]:
        """
        Detect network popup using template matching.

        使用模板匹配检测网络错误弹窗。

        Args:
            frame: Screenshot in BGR format.

        Returns:
            Tuple (found, result).
            found=True  => result is (center_x, center_y, confidence, bbox).
            found=False => result is None.
        """
        if not self._popup_valid or self._popup_tpl is None:
            logger.warning("Popup template not loaded or invalid, skipping detection")
            return False, None

        result = self._match_popup(frame)
        if result is not None:
            return True, result
        return False, None

    def detect_confirm_button_in_popup(
        self,
        frame: np.ndarray,
        popup_bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]]:
        """
        Find the "确定" button within the popup region.

        在弹窗区域内查找"确定"按钮。
        """
        if not self._button_valid or self._button_tpl is None:
            logger.warning("Button template not loaded or invalid, skipping detection")
            return False, None

        result = self._match_confirm_button(frame, popup_bbox)
        if result is not None:
            return True, result
        return False, None

    # ------------------------------------------------------------------
    # Anti-misclick: verify button is inside popup
    # ------------------------------------------------------------------

    def _verify_click_target(
        self,
        popup_conf: float,
        button_conf: float,
        popup_bbox: Tuple[int, int, int, int],
        button_bbox: Tuple[int, int, int, int],
    ) -> bool:
        """
        Anti-misclick verification: reject click if conditions not met.

        防误点二次确认：
          1. popup_conf >= popup_threshold
          2. button_conf >= button_threshold
          3. button bbox is strictly inside popup bbox

        Returns:
            True if all checks pass, False if click should be rejected.
        """
        popup_threshold = self.config.get(
            "popup_threshold", DEFAULT_CONFIG["popup_threshold"]
        )
        button_threshold = self.config.get(
            "button_threshold", DEFAULT_CONFIG["button_threshold"]
        )

        checks_ok = True

        if popup_conf < popup_threshold:
            logger.error(
                f"REJECT click: popup conf {popup_conf:.3f} < threshold {popup_threshold:.3f}"
            )
            checks_ok = False

        if button_conf < button_threshold:
            logger.error(
                f"REJECT click: button conf {button_conf:.3f} < threshold {button_threshold:.3f}"
            )
            checks_ok = False

        if not is_bbox_inside(button_bbox, popup_bbox):
            logger.error(
                f"REJECT click: button bbox {button_bbox} is NOT inside popup bbox {popup_bbox}"
            )
            checks_ok = False

        if checks_ok:
            logger.info(
                f"Anti-misclick check PASSED: popup_conf={popup_conf:.3f}, "
                f"button_conf={button_conf:.3f}, button_inside_popup=True"
            )
        return checks_ok

    # ------------------------------------------------------------------
    # Core handler logic
    # ------------------------------------------------------------------

    def handle_network_popup(self, frame: np.ndarray) -> bool:
        """
        Attempt to detect and dismiss the network popup.

        尝试检测并关闭网络错误弹窗。

        Detection order:
          1. Template matching (popup + confirm button + anti-misclick check)
          2. If template fails and fallback enabled => OCR detection
          3. Click button center, verify dismissal, retry
          4. Cooldown prevents rapid repeat clicks

        Args:
            frame: Screenshot in BGR format.

        Returns:
            True if popup was found and handled (or not found at all).
            False if popup found but could not be dismissed.
        """
        max_retries = self.config.get("max_retries", DEFAULT_CONFIG["max_retries"])
        post_wait_min = self.config.get(
            "post_click_wait_min", DEFAULT_CONFIG["post_click_wait_min"]
        )
        post_wait_max = self.config.get(
            "post_click_wait_max", DEFAULT_CONFIG["post_click_wait_max"]
        )
        fallback = self.config.get("fallback_to_ocr", DEFAULT_CONFIG["fallback_to_ocr"])
        save_debug = self.config.get(
            "debug_save_images", DEFAULT_CONFIG["debug_save_images"]
        )
        debug_dir = self.config.get(
            "debug_output_dir", DEFAULT_CONFIG["debug_output_dir"]
        )

        # --- 1. Template matching path ---
        popup_found, popup_result = self.detect_network_popup_by_template(frame)

        if popup_found and popup_result is not None:
            cx, cy, conf, popup_bbox = popup_result
            fingerprint = self._compute_popup_fingerprint(popup_bbox, conf)

            # Cooldown check
            if self._is_in_cooldown(fingerprint):
                return True

            logger.info(f"Popup detected via template (conf={conf:.3f})")

            btn_found, btn_result = self.detect_confirm_button_in_popup(
                frame, popup_bbox
            )

            if btn_found and btn_result is not None:
                bx, by, bconf, btn_bbox = btn_result

                # --- Anti-misclick verification ---
                if not self._verify_click_target(conf, bconf, popup_bbox, btn_bbox):
                    logger.error(
                        "Anti-misclick check failed, refusing to click. "
                        "Will try OCR fallback if enabled."
                    )
                    if fallback:
                        return self._handle_network_popup_by_ocr(frame)
                    return False

                logger.info(
                    f"Confirm button found (conf={bconf:.3f}), clicking..."
                )

                if save_debug:
                    self._save_debug_image(
                        frame, popup_bbox, btn_bbox, debug_dir, "popup_found"
                    )

                self._record_handle(fingerprint)
                return self._click_and_verify(
                    bx, by, post_wait_min, post_wait_max, max_retries, debug_dir, frame
                )
            else:
                logger.warning(
                    "Popup found but confirm button not detected via template"
                )
                # Anti-misclick: do NOT click popup center blindly
                # Try OCR fallback instead
                if fallback:
                    logger.info("Falling back to OCR...")
                    ocr_result = self._handle_network_popup_by_ocr(frame)
                    if ocr_result:
                        self._record_handle(fingerprint)
                        return True
                logger.warning(
                    "Cannot safely dismiss popup: button not found and OCR fallback "
                    "failed or disabled. Skipping to avoid misclick."
                )
                return False
        else:
            logger.info("Popup not found via template matching")

        # --- 2. OCR fallback (only if templates failed entirely) ---
        if fallback:
            logger.info("Attempting OCR fallback...")
            ocr_handled = self._handle_network_popup_by_ocr(frame)
            if ocr_handled:
                return True
            logger.info("OCR fallback did not detect network error popup")

        return False

    def _click_and_verify(
        self,
        x: int,
        y: int,
        wait_min: float,
        wait_max: float,
        max_retries: int,
        debug_dir: str,
        frame: np.ndarray,
    ) -> bool:
        """
        Click a point, wait, verify popup dismissal, and retry if needed.

        点击后等待并验证弹窗是否消失，必要时重试。
        """
        for attempt in range(max_retries):
            jitter_x = x + random.randint(-10, 10)
            jitter_y = y + random.randint(-10, 10)
            logger.info(
                f"Clicking confirm button at ({jitter_x}, {jitter_y}) "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            self.click(jitter_x, jitter_y)

            wait = random.uniform(wait_min, wait_max)
            logger.info(f"Waiting {wait:.2f}s for popup to dismiss...")
            time.sleep(wait)

            try:
                new_frame = self.next_frame()
                if new_frame is None:
                    logger.warning("No new frame after click, assuming dismissed")
                    return True

                popup_found, _ = self.detect_network_popup_by_template(new_frame)
                if not popup_found:
                    logger.info("Popup dismissed successfully")
                    return True
                else:
                    logger.warning(
                        f"Popup still visible after click (attempt {attempt + 1})"
                    )

                    if self.config.get(
                        "debug_save_images", DEFAULT_CONFIG["debug_save_images"]
                    ):
                        self._save_debug_image(
                            new_frame,
                            None,
                            None,
                            debug_dir,
                            f"popup_still_visible_attempt{attempt + 1}",
                        )

                    if attempt < max_retries - 1:
                        retry_delay = random.uniform(
                            self.config.get(
                                "retry_delay_min",
                                DEFAULT_CONFIG["retry_delay_min"],
                            ),
                            self.config.get(
                                "retry_delay_max",
                                DEFAULT_CONFIG["retry_delay_max"],
                            ),
                        )
                        logger.info(
                            f"Waiting {retry_delay:.2f}s before retry..."
                        )
                        time.sleep(retry_delay)

            except Exception as e:
                logger.warning(f"Verification failed: {e}, assuming dismissed")
                return True

        logger.warning(
            f"Popup could not be dismissed after {max_retries} attempts"
        )
        return False

    def _handle_network_popup_by_ocr(self, frame: np.ndarray) -> bool:
        """
        Legacy OCR-based network popup detection (fallback).

        基于 OCR 的网络弹窗检测（回退方案）。
        """
        ocr_texts = self.config.get("ocr_texts", DEFAULT_CONFIG["ocr_texts"])
        for text in ocr_texts:
            boxes = self.ocr(match=text)
            if boxes:
                logger.info(f"[OCR] Detected network error text: '{text}'")
                self.click_box(boxes[0])
                logger.info(f"[OCR] Clicked detected box for '{text}'")
                return True
        return False

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _save_debug_image(
        self,
        frame: np.ndarray,
        popup_bbox: Optional[Tuple[int, int, int, int]],
        button_bbox: Optional[Tuple[int, int, int, int]],
        output_dir: str,
        tag: str,
    ):
        """Save a debug image with bounding boxes drawn."""
        try:
            out_path = resolve_repo_path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            vis = frame.copy()
            if popup_bbox:
                x1, y1, x2, y2 = popup_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    vis,
                    "popup",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            if button_bbox:
                x1, y1, x2, y2 = button_bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    vis,
                    "confirm",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"network_error_{tag}_{timestamp}.png"
            full_path = str(out_path / filename)
            cv2.imwrite(full_path, vis)
            logger.debug(f"Debug image saved: {full_path}")
            self._last_debug_path = full_path
        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")

    # ------------------------------------------------------------------
    # Main entry point (called by the framework)
    # ------------------------------------------------------------------

    def run(self):
        """
        Main trigger entry point called periodically by the framework.

        框架周期性调用的主入口。

        Returns:
            True if a popup was found and handled (or not found),
            False if found but could not be dismissed.
        """
        frame = self.executor.frame
        if frame is None:
            logger.debug("No frame available, skipping")
            return False

        logger.debug(f"Frame acquired: {frame.shape}")
        return self.handle_network_popup(frame)