"""
Shared home-screen detection utility for ok_tasks / ok.automation.

模板匹配检测当前是否在主界面。提取自 HomeRedDotTask / AlchemyDispatchTask /
GameStartupTask 中重复的实现，统一 ROI / 阈值 / 模板路径默认值，避免“检查是否在
主页”的代码到处复制（与 ``ok_tasks/_red_dot.py`` 同样的共享工具思路）。

Usage:
    from ok_tasks._home import (
        is_on_home, match_template_in_roi, load_template_image,
        DEFAULT_HOME_TEMPLATE_PATH, DEFAULT_HOME_ROI, DEFAULT_HOME_THRESHOLD,
    )
    tpl = load_template_image(DEFAULT_HOME_TEMPLATE_PATH)
    if is_on_home(frame, tpl, DEFAULT_HOME_ROI, DEFAULT_HOME_THRESHOLD):
        ...
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 仓库根目录（ok_tasks/ 的上一级），用于解析相对模板路径。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 主页模板 / ROI / 阈值默认值（与各任务原有实现保持一致）。
# 模板：templates/home_profile_button.png；ROI：左上角个人信息按钮区域；阈值 0.70。
DEFAULT_HOME_TEMPLATE_PATH = "templates/home_profile_button.png"
DEFAULT_HOME_ROI = [0.0343, 0.0359, 0.2926, 0.0916]
DEFAULT_HOME_THRESHOLD = 0.70


def _resolve_path(rel_path: str, project_root: Optional[Path] = None) -> Path:
    """将相对路径解析为绝对路径；已是绝对路径则原样返回。"""
    p = Path(rel_path)
    return p if p.is_absolute() else (project_root or _PROJECT_ROOT) / p


def load_template_image(
    rel_path: str, project_root: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    安全加载模板图像（使用 ``np.fromfile`` + ``cv2.imdecode`` 处理 Windows 中文路径）。

    Args:
        rel_path: 仓库相对路径或绝对路径。
        project_root: 仓库根目录，为 None 时使用 ``ok_tasks`` 上一级。

    Returns:
        BGR 模板图像，失败返回 None。
    """
    abs_path = _resolve_path(rel_path, project_root)
    if not abs_path.exists():
        return None
    try:
        data = np.fromfile(str(abs_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def match_template_in_roi(
    frame: np.ndarray,
    template: np.ndarray,
    roi,
    threshold: float,
) -> Tuple[bool, float]:
    """
    在 ``frame`` 的 ``roi`` 区域内灰度匹配 ``template``，返回 (是否达到阈值, 最高置信度)。

    Args:
        frame: BGR 图像。
        template: BGR 模板图像。
        roi: ``[x1, y1, x2, y2]`` 相对坐标 (0-1)。
        threshold: 置信度阈值。

    Returns:
        ``(matched, confidence)``。
    """
    if frame is None or template is None:
        return False, 0.0
    h, w = frame.shape[:2]
    x1 = max(0, int(w * roi[0]))
    y1 = max(0, int(h * roi[1]))
    x2 = min(w, int(w * roi[2]))
    y2 = min(h, int(h * roi[3]))
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False, 0.0
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    th, tw = tpl_gray.shape[:2]
    if region_gray.shape[0] < th or region_gray.shape[1] < tw:
        return False, 0.0
    res = cv2.matchTemplate(region_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val) >= threshold, float(max_val)


def is_on_home(
    frame: np.ndarray,
    template: np.ndarray,
    roi=DEFAULT_HOME_ROI,
    threshold: float = DEFAULT_HOME_THRESHOLD,
) -> bool:
    """
    检测当前画面是否在主界面。

    Args:
        frame: BGR 图像。
        template: 主页模板图像（由 ``load_template_image`` 加载）。
        roi: ``[x1, y1, x2, y2]`` 相对坐标 (0-1)，默认为主页按钮区域。
        threshold: 置信度阈值。

    Returns:
        True 表示当前在主界面。
    """
    if frame is None or template is None:
        return False
    matched, _ = match_template_in_roi(frame, template, roi, threshold)
    return matched
