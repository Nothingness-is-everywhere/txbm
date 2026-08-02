"""
Home red-dot gated fixed-click task for 天下布魔 (Tianxia Bumo).

流程：
  1. 模板匹配确认在主页。
  2. 仅检测第一个主页红点；无红点则结束，有红点继续。
  3. 第 2/3/4/5 步按固定位置直接点击（不再做额外红点检测）。
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ok.task.task import BaseTask

from ok_tasks._red_dot import detect_red_dot, roi_center, roi_to_pixels

logger = logging.getLogger("HomeRedDotTask")


# Selection: x=37, y=69, w=279, h=107  (rx=0.0343, ry=0.0359, rw=0.2583, rh=0.0557)
_PROFILE_ROI = [0.0343, 0.0359, 0.2926, 0.0916]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大启动按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 主页检测（模板匹配）
    "home_template_path": "templates/home_profile_button.png",
    "home_template_roi": list(_PROFILE_ROI),
    "home_threshold": 0.70,
    # 仅用于“第一个主页红点”检测
    "red_dot_roi": list(_PROFILE_ROI),
    # 第 2 步固定点击区域（个人信息入口）
    "profile_click_roi": list(_PROFILE_ROI),
    # 第 3 步固定点击区域（第二个入口）
    # Selection: x=133,y=692,w=92,h=96 (rx=0.1231, ry=0.3604, rw=0.0852, rh=0.0500)
    "followup_click_roi": [0.1231, 0.3604, 0.2083, 0.4104],
    # 第 4 步固定点击区域（奖励按钮）
    # Selection: x=597,y=1591,w=306,h=99 (rx=0.5528, ry=0.8286, rw=0.2833, rh=0.0500)
    "reward_click_roi": [0.5528, 0.8286, 0.8361, 0.8786],
    # 第 5 步固定点击区域（关闭按钮）
    # Selection: x=479,y=1772,w=121,h=118 (rx=0.4435, ry=0.9229, rw=0.1120, rh=0.0615)
    "close_button_click_roi": [0.4435, 0.9229, 0.5555, 0.9844],
    # 每步点击后的等待秒数
    "post_click_sleep": 1.0,
    # HSV red range. Red wraps around hue 0/180, so two intervals are used.
    "hsv_lower1": [0, 120, 120],
    "hsv_upper1": [10, 255, 255],
    "hsv_lower2": [170, 120, 120],
    "hsv_upper2": [180, 255, 255],
    # Minimum red pixels to count as a dot (avoids noise)
    "min_red_pixels": 20,
    # Morphological cleanup
    "morph_open_size": 3,
    # Connected-component filtering to reduce false positives
    "min_component_area": 16,
    "max_component_area": 1200,
    "max_component_aspect": 1.8,
    "min_component_circularity": 0.45,
    "min_component_fill_ratio": 0.45,
}


class HomeRedDotTask(BaseTask):
    """
    周常日常任务：仅检测第一个主页红点，后续固定点击。

    One-time task flow:
      1. Confirm home screen by template match.
      2. Detect first home red dot; exit if absent.
      3. Click profile fixed position.
      4. Click follow-up fixed position.
      5. Click reward fixed position.
      6. Click close fixed position.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "获取好友体力"
        self.description = "固定位置依次点击第2~5步（无红点检测/模板匹配）"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        # 单独启动本任务时不连带启用 enable_after_start 任务（如 GameStartupTask），保持单独执行
        self.standalone_start = True
        self._home_tpl: Optional[np.ndarray] = None

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_template()

    def run(self):
        """
        仅检测第一个主页红点后，执行固定点击流程（每步间隔 1 秒）。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("获取好友体力：无画面可用，终止")
            return False

        # 1. 确认在主页（恢复模板匹配检测）
        if not self._is_home(frame):
            logger.info("获取好友体力：不在主页，终止")
            return False

        # 仅检测第一个主页红点：无红点直接结束，有红点继续
        dot = self._detect_red_dot(frame)
        if dot is None:
            logger.info("获取好友体力：第一个主页红点不存在，直接结束")
            return False
        logger.info(f"获取好友体力：检测到第一个主页红点 {dot}，继续执行固定点击")

        h, w = frame.shape[:2]

        # 2. 点击个人信息入口固定位置
        p_x, p_y = roi_center(self.config.get("profile_click_roi", DEFAULT_CONFIG["profile_click_roi"]), w, h)
        logger.info(f"[步骤2/5] 点击个人信息入口固定位置 ({p_x}, {p_y})")
        self.click(p_x, p_y)
        self.sleep(step_sleep)

        # 3. 点击第二入口固定位置
        f_x, f_y = roi_center(self.config.get("followup_click_roi", DEFAULT_CONFIG["followup_click_roi"]), w, h)
        logger.info(f"[步骤3/5] 点击第二入口固定位置 ({f_x}, {f_y})")
        self.click(f_x, f_y)
        self.sleep(step_sleep)

        # 4. 点击奖励按钮固定位置
        r_x, r_y = roi_center(self.config.get("reward_click_roi", DEFAULT_CONFIG["reward_click_roi"]), w, h)
        logger.info(f"[步骤4/5] 点击奖励按钮固定位置 ({r_x}, {r_y})")
        self.click(r_x, r_y)
        self.sleep(step_sleep)

        # 5. 点击关闭按钮固定位置
        c_x, c_y = roi_center(
            self.config.get("close_button_click_roi", DEFAULT_CONFIG["close_button_click_roi"]), w, h)
        logger.info(f"[步骤5/5] 点击关闭按钮固定位置 ({c_x}, {c_y})")
        self.click(c_x, c_y)
        self.sleep(step_sleep)

        logger.info("获取好友体力：固定点击流程执行完毕")
        return True

    def _load_template(self):
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        p = Path(rel)
        abs_path = str(p if p.is_absolute() else (_PROJECT_ROOT / p))
        try:
            tpl = cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Failed to read home template: {abs_path} ({e})")
            tpl = None
        if tpl is None:
            logger.warning(f"Home template not found, home detection disabled: {abs_path}")
            self._home_tpl = None
        else:
            self._home_tpl = tpl

    def _is_home(self, frame: np.ndarray) -> bool:
        if self._home_tpl is None:
            return False
        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_to_pixels(roi, w, h)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return False

        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        tpl_gray = cv2.cvtColor(self._home_tpl, cv2.COLOR_BGR2GRAY)
        th, tw = tpl_gray.shape[:2]
        if region_gray.shape[0] < th or region_gray.shape[1] < tw:
            return False

        res = cv2.matchTemplate(region_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        threshold = self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"])
        return max_val >= threshold

    def _detect_red_dot(self, frame: np.ndarray, roi=None) -> Optional[Tuple[int, int]]:
        if roi is None:
            roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        return detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
