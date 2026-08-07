"""
Home red-dot gated fixed-click task for 天下布魔 (Tianxia Bumo).

流程：
  1. 模板匹配确认在主页。
  2. 仅检测第一个主页红点；无红点则结束，有红点继续。
  3. 第 2/3/4/5 步按固定位置直接点击（不再做额外红点检测）。
  4. 步骤4前会做一次"是否进入奖励页面"的模板匹配，未匹配则补点第二入口固定位置。
"""

import logging
from typing import Optional, Tuple

import numpy as np
from ok.task.task import BaseTask

from ok_tasks._home import is_on_home, load_template_image
from ok_tasks._red_dot import detect_red_dot, roi_center, roi_to_pixels

logger = logging.getLogger("HomeRedDotTask")


# Selection: x=37, y=69, w=279, h=107  (rx=0.0343, ry=0.0359, rw=0.2583, rh=0.0557)
_PROFILE_ROI = [0.0343, 0.0359, 0.2926, 0.0916]

# 步骤4前：奖励页面模板匹配选区 (x=600, y=492, w=148, h=44 @ 1080×1920) 中心 (674, 514)
# 模板文件 templates/reward_page_indicator.png
_REWARD_PAGE_ROI = [0.5556, 0.2562, 0.6926, 0.2792]

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
    # 步骤4前：奖励页面模板匹配（未匹配则补点第二入口固定位置）
    "reward_page_template_path": "templates/reward_page_indicator.png",
    "reward_page_template_roi": list(_REWARD_PAGE_ROI),
    "reward_page_threshold": 0.70,
    "reward_page_check_max_attempts": 3,
    # 每步点击后的等待秒数
    "post_click_sleep": 1.0,
    # 点击关闭按钮后未回主页时的最大补点次数
    "home_check_max_attempts": 3,
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
        self._reward_page_tpl: Optional[np.ndarray] = None

    def on_create(self):
        # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
        # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
        self._enabled = False
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_template()
        self._load_reward_page_template()

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

        # 3.5 模板匹配校验是否进入奖励页面；未进入则补点第二入口固定位置
        self._ensure_reward_page(self.config.get("followup_click_roi", DEFAULT_CONFIG["followup_click_roi"]))

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

        # 点击关闭按钮后校验是否回到主页，未回主页则补点几次
        self._ensure_home(self.config.get("close_button_click_roi", DEFAULT_CONFIG["close_button_click_roi"]))

        logger.info("获取好友体力：固定点击流程执行完毕")
        return True

    def _load_template(self):
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"Home template not found, home detection disabled: {rel}")
        self._home_tpl = tpl

    def _is_home(self, frame: np.ndarray) -> bool:
        if self._home_tpl is None:
            return False
        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        threshold = float(self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"]))
        return is_on_home(frame, self._home_tpl, roi, threshold)

    def _load_reward_page_template(self):
        """加载步骤4前的奖励页面模板；模板缺失时校验将被跳过（不阻塞流程）。"""
        rel = self.config.get(
            "reward_page_template_path", DEFAULT_CONFIG["reward_page_template_path"]
        )
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"获取好友体力：奖励页模板未找到，步骤4前页面校验将跳过: {rel}")
        self._reward_page_tpl = tpl

    def _is_on_reward_page(self, frame: np.ndarray) -> bool:
        """检测当前画面是否已进入奖励页面；模板缺失时返回 True（跳过校验，向后兼容）。"""
        if self._reward_page_tpl is None:
            return True
        roi = self.config.get(
            "reward_page_template_roi", DEFAULT_CONFIG["reward_page_template_roi"]
        )
        threshold = float(
            self.config.get("reward_page_threshold", DEFAULT_CONFIG["reward_page_threshold"])
        )
        return is_on_home(frame, self._reward_page_tpl, roi, threshold)

    def _ensure_reward_page(self, click_roi) -> bool:
        """步骤4前校验是否进入奖励页面；未进入则按 click_roi（第二入口）补点几次。"""
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(
            self.config.get(
                "reward_page_check_max_attempts",
                DEFAULT_CONFIG["reward_page_check_max_attempts"],
            )
        )
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning("获取好友体力：校验奖励页面时无画面可用")
                return False
            if self._is_on_reward_page(frame):
                logger.info(f"获取好友体力：已进入奖励页面（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(
                f"获取好友体力：未进入奖励页面，第 {attempt} 次补点第二入口 ({cx}, {cy})"
            )
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"获取好友体力：{max_attempts} 次补点后仍未进入奖励页面")
        return False

    def _ensure_home(self, click_roi) -> bool:
        """点击关闭/返回按钮后校验是否回到主页；未回主页则按 click_roi 补点几次。"""
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(self.config.get("home_check_max_attempts", DEFAULT_CONFIG["home_check_max_attempts"]))
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning("获取好友体力：校验主页时无画面可用")
                return False
            if self._is_home(frame):
                logger.info(f"获取好友体力：已回到主页（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(f"获取好友体力：未回主页，第 {attempt} 次补点关闭按钮 ({cx}, {cy})")
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"获取好友体力：{max_attempts} 次补点后仍未回主页")
        return False

    def _detect_red_dot(self, frame: np.ndarray, roi=None) -> Optional[Tuple[int, int]]:
        if roi is None:
            roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        return detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
