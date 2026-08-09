"""
Network Error Handler for 天下布魔 (Tianxia Bumo).

Detects and dismisses the "网络不稳定，将重新启动游戏" popup
using template matching.

Detection flow:
  1. Template-match the full popup on the screenshot
  2. If popup found, search the "确定" button within popup ROI
  3. Anti-misclick: verify button bbox is inside popup bbox
  4. Click button center, verify dismissal, retry if needed
  5. Cooldown prevents re-clicking the same popup within a window

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

from ok.trigger.base import ManagedTriggerTask, TriggerContext
from ok.trigger.categories import TriggerCategory
from ok.trigger.decision import TriggerDecision, TriggerResult

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
    "_enabled": True,
    "popup_template_path": "templates/network_popup.png",
    "popup_template_paths": [
        "templates/network_popup.png",
        "templates/network_popup_v2.png",
    ],
    "confirm_button_template_path": "templates/network_confirm_button.png",
    "confirm_button_template_paths": [
        "templates/network_confirm_button.png",
        "templates/network_confirm_button_v2.png",
    ],
    "popup_threshold": 0.55,
    "button_threshold": 0.55,
    # Reduced from 9 scales to 4: covers the common resolution variance while
    # cutting matchTemplate cost by ~55%. The unified scheduler's cooldown also
    # skips detection entirely during cooldown, so net CPU drops further.
    "match_scales": [0.8, 0.9, 1.0, 1.1],
    "max_retries": 3,
    "retry_delay_min": 0.5,
    "retry_delay_max": 1.0,
    "post_click_wait_min": 0.5,
    "post_click_wait_max": 1.0,
    "debug_save_images": True,
    "debug_output_dir": "artifacts/network_error",
    "popup_roi": [0.00, 0.37, 1.00, 0.63],
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


class NetworkErrorHandler(ManagedTriggerTask):
    """
    Detects and dismisses the network instability popup via template matching.

    定时触发的网络错误弹窗处理任务（受管于统一调度器）。
    """

    # --- ManagedTriggerTask classification ---
    category = TriggerCategory.NETWORK
    priority = 80
    trigger_mode = "polling"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "网络错误处理"
        self.description = "检测并关闭网络不稳定弹窗（模板匹配）"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        self._popup_tpl = None
        self._button_tpl = None
        self._popup_tpls = []  # list of (template, valid) tuples
        self._button_tpls = []  # list of (template, valid) tuples
        self._popup_valid = False
        self._button_valid = False
        self._last_debug_path = None

        # Cooldown state: prevent re-clicking same popup within cooldown window.
        # Retained for backward compatibility with direct unit tests; the
        # framework throttle also enforces cooldown/dedup centrally.
        self._last_handle_time: float = 0.0
        self._last_handle_fingerprint: str = ""

        # Pending detection result handed from check() to handle().
        self._pending_popup: Optional[Tuple[int, int, float, Tuple[int, int, int, int]]] = None

    def on_create(self):
        super().on_create()  # ManagedTriggerTask: refresh params from config
        # Default to enabled so the network popup handler is on by default.
        self._enabled = self.config.get("_enabled", True)
        self._load_templates()

    # ------------------------------------------------------------------
    # Template loading & validation
    # ------------------------------------------------------------------

    def _load_templates(self):
        """
        Load and validate popup and button template images.

        支持多模板：依次尝试 popup_template_paths 列表中的所有模板，
        任一匹配成功即判定为检测到弹窗。

        加载并验证模板图像的有效性。
        """
        # Load popup templates (support multiple)
        popup_paths = self.config.get(
            "popup_template_paths", DEFAULT_CONFIG["popup_template_paths"]
        )
        # Fallback to single path for backward compatibility
        if not popup_paths:
            single = self.config.get(
                "popup_template_path", DEFAULT_CONFIG["popup_template_path"]
            )
            popup_paths = [single]

        self._popup_tpls = []
        for rel_path in popup_paths:
            abs_path = str(resolve_repo_path(rel_path))
            tpl = self._safe_imread(abs_path)
            if tpl is not None:
                valid = validate_template_image(tpl, f"popup[{rel_path}]")
                self._popup_tpls.append((tpl, valid))
            else:
                logger.warning(f"Popup template not found or invalid: {rel_path}")

        # Keep backward-compatible single template (first valid one)
        self._popup_tpl = None
        self._popup_valid = False
        for tpl, valid in self._popup_tpls:
            if valid:
                self._popup_tpl = tpl
                self._popup_valid = True
                break

        # Load button templates (support multiple)
        button_paths = self.config.get(
            "confirm_button_template_paths", DEFAULT_CONFIG["confirm_button_template_paths"]
        )
        if not button_paths:
            single = self.config.get(
                "confirm_button_template_path", DEFAULT_CONFIG["confirm_button_template_path"]
            )
            button_paths = [single]

        self._button_tpls = []
        for rel_path in button_paths:
            abs_path = str(resolve_repo_path(rel_path))
            tpl = self._safe_imread(abs_path)
            if tpl is not None:
                valid = validate_template_image(tpl, f"button[{rel_path}]")
                self._button_tpls.append((tpl, valid))
            else:
                logger.warning(f"Button template not found or invalid: {rel_path}")

        # Keep backward-compatible single template (first valid one)
        self._button_tpl = None
        self._button_valid = False
        for tpl, valid in self._button_tpls:
            if valid:
                self._button_tpl = tpl
                self._button_valid = True
                break

        if not self._popup_valid:
            logger.error(
                "No valid popup template loaded. "
                "Place real templates in templates/."
            )
        if not self._button_valid:
            logger.error(
                "No valid confirm button template loaded. "
                "Place real templates in templates/."
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
        counter: Optional[list] = None,
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
            counter: Optional ``[int]`` list incremented once per
                ``cv2.matchTemplate`` call, for budget/metrics accounting.

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
            if counter is not None:
                counter[0] += 1
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
        self, screen: np.ndarray, counter: Optional[list] = None
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """Detect the full network error popup on screen using all loaded templates."""
        roi = self.config.get("popup_roi", DEFAULT_CONFIG["popup_roi"])
        threshold = self.config.get("popup_threshold", DEFAULT_CONFIG["popup_threshold"])
        scales = self.config.get("match_scales", DEFAULT_CONFIG["match_scales"])

        # Try all loaded popup templates, return the best match
        best_result = None
        best_conf = 0.0
        for tpl, valid in self._popup_tpls:
            if not valid or tpl is None:
                continue
            result = self._multi_scale_match(
                screen, tpl, roi, threshold, scales, "network_popup", counter
            )
            if result is not None and result[2] > best_conf:
                best_conf = result[2]
                best_result = result

        return best_result

    def _match_confirm_button(
        self,
        screen: np.ndarray,
        popup_bbox: Tuple[int, int, int, int],
        counter: Optional[list] = None,
    ) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """
        Find the "确定" button within the popup bounding box.

        在弹窗 bbox 内查找"确定"按钮（尝试所有已加载的按钮模板）。
        """
        bx1, by1, bx2, by2 = popup_bbox
        screen_h, screen_w = screen.shape[:2]

        margin_x = int((bx2 - bx1) * 0.20)
        margin_top = int((by2 - by1) * 0.50)
        margin_bottom = int((by2 - by1) * 0.01)

        # Build ROI in normalized form (auto-detected)
        btn_roi = [
            max(0.0, (bx1 + margin_x) / screen_w),
            max(0.0, (by1 + margin_top) / screen_h),
            min(1.0, (bx2 - margin_x) / screen_w),
            min(1.0, (by2 - margin_bottom) / screen_h),
        ]

        threshold = self.config.get("button_threshold", DEFAULT_CONFIG["button_threshold"])
        scales = self.config.get("match_scales", DEFAULT_CONFIG["match_scales"])

        # Try all loaded button templates, return the best match
        best_result = None
        best_conf = 0.0
        for tpl, valid in self._button_tpls:
            if not valid or tpl is None:
                continue
            result = self._multi_scale_match(
                screen, tpl, btn_roi, threshold, scales, "confirm_button", counter
            )
            if result is not None and result[2] > best_conf:
                best_conf = result[2]
                best_result = result

        return best_result

    # ------------------------------------------------------------------
    # Detection interface (public, testable)
    # ------------------------------------------------------------------

    def detect_network_popup_by_template(
        self, frame: np.ndarray, counter: Optional[list] = None
    ) -> Tuple[bool, Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]]:
        """
        Detect network popup using template matching.

        使用模板匹配检测网络错误弹窗（遍历所有已加载模板）。

        Args:
            frame: Screenshot in BGR format.
            counter: Optional ``[int]`` list incremented per matchTemplate call.

        Returns:
            Tuple (found, result).
            found=True  => result is (center_x, center_y, confidence, bbox).
            found=False => result is None.
        """
        if not self._popup_valid:
            logger.warning("No valid popup template loaded, skipping detection")
            return False, None

        result = self._match_popup(frame, counter)
        if result is not None:
            return True, result
        return False, None

    def detect_confirm_button_in_popup(
        self,
        frame: np.ndarray,
        popup_bbox: Tuple[int, int, int, int],
        counter: Optional[list] = None,
    ) -> Tuple[bool, Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]]:
        """
        Find the "确定" button within the popup region.

        在弹窗区域内查找"确定"按钮。
        """
        if not self._button_valid or self._button_tpl is None:
            logger.warning("Button template not loaded or invalid, skipping detection")
            return False, None

        result = self._match_confirm_button(frame, popup_bbox, counter)
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

    # ------------------------------------------------------------------
    # ManagedTriggerTask two-phase contract
    # ------------------------------------------------------------------
    def check(self, context: TriggerContext) -> TriggerDecision:
        """Cheap-ish detection phase.

        检测阶段：使用模板匹配检测弹窗，并将结果暂存于 _pending_popup，供 handle() 使用。
        """
        frame = context.frame
        if frame is None:
            frame = self._current_frame()
        if frame is None:
            self._pending_popup = None
            return TriggerDecision(should_handle=False, reason="no_frame")

        # Template matching (count matchTemplate calls for budget/metrics).
        match_counter = [0]
        popup_found, popup_result = self.detect_network_popup_by_template(frame, match_counter)
        context.match_calls += match_counter[0]

        if popup_found and popup_result is not None:
            _, _, conf, popup_bbox = popup_result
            fingerprint = self._compute_popup_fingerprint(popup_bbox, conf)
            self._pending_popup = popup_result
            logger.info(f"Popup detected via template (conf={conf:.3f})")
            return TriggerDecision(
                should_handle=True, reason="popup_template",
                fingerprint=fingerprint, cost_estimate=0.5,
            )

        self._pending_popup = None
        return TriggerDecision(should_handle=False, reason="no_popup")

    def handle(self, context: TriggerContext) -> TriggerResult:
        """Action phase: dismiss the popup detected by ``check()``.

        处理阶段：根据 check() 暂存的结果点击确认按钮，并验证关闭。
        """
        frame = context.frame
        if frame is None:
            frame = self._current_frame()

        max_retries = self.config.get("max_retries", DEFAULT_CONFIG["max_retries"])
        post_wait_min = self.config.get("post_click_wait_min", DEFAULT_CONFIG["post_click_wait_min"])
        post_wait_max = self.config.get("post_click_wait_max", DEFAULT_CONFIG["post_click_wait_max"])
        save_debug = self.config.get("debug_save_images", DEFAULT_CONFIG["debug_save_images"])
        debug_dir = self.config.get("debug_output_dir", DEFAULT_CONFIG["debug_output_dir"])

        if self._pending_popup is not None and frame is not None:
            _, _, conf, popup_bbox = self._pending_popup
            fingerprint = self._compute_popup_fingerprint(popup_bbox, conf)
            button_counter = [0]
            btn_found, btn_result = self.detect_confirm_button_in_popup(
                frame, popup_bbox, button_counter
            )
            context.match_calls += button_counter[0]

            if btn_found and btn_result is not None:
                bx, by, bconf, btn_bbox = btn_result
                if not self._verify_click_target(conf, bconf, popup_bbox, btn_bbox):
                    logger.error("Anti-misclick check failed, refusing to click.")
                    return TriggerResult.fail("anti_misclick", match_calls=context.match_calls)

                logger.info(f"Confirm button found (conf={bconf:.3f}), clicking...")
                if save_debug:
                    self._save_debug_image(frame, popup_bbox, btn_bbox, debug_dir, "popup_found")
                self._record_handle(fingerprint)
                ok = self._click_and_verify(
                    bx, by, post_wait_min, post_wait_max, max_retries, debug_dir, frame, context
                )
                return TriggerResult(
                    handled=True, success=ok,
                    match_calls=context.match_calls,
                    retries=context.retries,
                )

            logger.warning("Popup found but confirm button not detected via template")
            logger.warning("Cannot safely dismiss popup; skipping to avoid misclick.")
            return TriggerResult.fail("button_not_found", match_calls=context.match_calls)

        return TriggerResult.skip("no_pending")

    def handle_network_popup(self, frame: np.ndarray) -> bool:
        """Legacy entry point. Delegates to ``check()`` + ``handle()``.

        旧入口（保留向后兼容）：在临时上下文上执行 check+handle。
        """
        ctx = TriggerContext(frame=frame, executor=getattr(self, "_executor", None),
                             now=time.time())
        decision = self.check(ctx)
        if not decision.should_handle:
            return False
        result = self.handle(ctx)
        return result.handled and result.success

    def _click_and_verify(
        self,
        x: int,
        y: int,
        wait_min: float,
        wait_max: float,
        max_retries: int,
        debug_dir: str,
        frame: np.ndarray,
        context: Optional[TriggerContext] = None,
    ) -> bool:
        """
        Click a point, wait, verify popup dismissal, and retry if needed.

        点击后等待并验证弹窗是否消失，必要时重试。``context`` 用于协作式超时
        与重试/匹配次数统计（受管模式下由 handle() 传入）。
        """
        for attempt in range(max_retries):
            # Cooperative timeout: bail if the framework deadline has passed.
            if context is not None and context.timed_out():
                logger.warning(f"click_and_verify timed out at attempt {attempt + 1}")
                return False
            if context is not None:
                context.retries = attempt

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

                verify_counter = [0] if context is not None else None
                popup_found, _ = self.detect_network_popup_by_template(new_frame, verify_counter)
                if context is not None and verify_counter is not None:
                    context.match_calls += verify_counter[0]
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
    # Main entry point
    # ------------------------------------------------------------------
    # run() is provided by ManagedTriggerTask: it calls check() -> handle()
    # through the shared throttle/metrics/timeout, and never raises, so a
    # transient detection error can no longer disable this trigger.
