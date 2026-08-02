"""
招募任务（日常分类）。

流程（步骤增量添加中，当前仅第 1 步）：
  1. 检测招募入口选区是否有红色（红点）；有红点则点击该选区，无红点则结束。

红点检测复用 ok_tasks._red_dot 中的共享实现（与 HomeRedDotTask /
AlchemyDispatchTask 同一套 HSV 双区间 + 形态学开运算 + 连通域过滤逻辑）。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ok.task.task import BaseTask

from ok_tasks._red_dot import detect_red_dot, roi_center

logger = logging.getLogger("RecruitTask")


# 选区信息: x=676, y=1452, w=68, h=72  (图片尺寸 1080x1920)
# rx=0.6259, ry=0.7562, rw=0.0630, rh=0.0375  中心 (0.6574, 0.7750) = (710, 1488)
_RECRUIT_ROI = [0.6259, 0.7562, 0.6889, 0.7937]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大开始按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 红点检测区域（相对坐标 [x1, y1, x2, y2]）
    "red_dot_roi": list(_RECRUIT_ROI),
    # 检测到红点后点击的区域，点击其中心
    "click_roi": list(_RECRUIT_ROI),
    # 点击后等待秒数
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


class RecruitTask(BaseTask):
    """
    日常任务：招募。

    步骤1：检测招募入口选区红点，有红点则点击该选区；无红点则结束。
    （后续步骤待补充）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "招募"
        self.description = "检测招募入口红点并点击（后续步骤待补充）"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        # 单独启动本任务时不连带启用 enable_after_start 任务（如 GameStartupTask），保持单独执行
        self.standalone_start = True

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self.follow_batch_start = self.config.get("follow_batch_start", True)

    def run(self) -> bool:
        """
        招募任务主流程：检测选区红点，有则点击。

        Returns:
            True 表示检测到红点并点击；False 表示无红点或无画面。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("招募：无画面可用，终止")
            return False

        # 1. 检测招募入口选区红点；无红点直接结束，有红点继续点击
        roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
        if dot is None:
            logger.info("招募：选区无红点，结束")
            return False
        logger.info(f"招募：检测到红点 {dot}，点击选区")

        h, w = frame.shape[:2]
        cx, cy = roi_center(self.config.get("click_roi", DEFAULT_CONFIG["click_roi"]), w, h)
        logger.info(f"[步骤1] 点击招募选区中心 ({cx}, {cy})")
        self.click(cx, cy)
        self.sleep(step_sleep)

        # 后续步骤待补充
        logger.info("招募：第1步执行完毕")
        return True
