"""
炼金和派遣任务（日常分类）。

流程（第一步）：
  1. 检测指定入口区域是否存在红点。
  2. 无红点则结束；有红点则点击区域内的 》 按钮进入对应界面。

后续步骤将在本任务中陆续追加。

红点检测复用 ok_tasks._red_dot 中的共享实现（与 HomeRedDotTask 同一套方法）。
"""

import logging

from ok.task.task import BaseTask

from ok_tasks._red_dot import detect_red_dot, roi_center

logger = logging.getLogger("AlchemyDispatchTask")


# 选区信息: x=0, y=877, w=96, h=131  (图片尺寸 1080x1920)
# rx=0.0000, ry=0.4568, rw=0.0889, rh=0.0682  中心 (0.0444, 0.4906)
_ENTRY_ROI = [0.0, 0.4568, 0.0889, 0.5250]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大启动按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 红点检测区域（相对坐标 [x1, y1, x2, y2]）
    "red_dot_roi": list(_ENTRY_ROI),
    # 检测到红点后点击的区域（》按钮所在区域），点击其中心
    "click_roi": list(_ENTRY_ROI),
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


class AlchemyDispatchTask(BaseTask):
    """
    日常任务：炼金和派遣。

    第一步：检测炼金/派遣入口区域的红点，存在则点击 》 按钮进入。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "炼金和派遣"
        self.description = "检测炼金/派遣入口红点，有红点则点击 》 按钮进入"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self.follow_batch_start = self.config.get("follow_batch_start", True)

    def run(self):
        """
        第一步：检测入口红点 → 点击 》 按钮。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("炼金和派遣：无画面可用，终止")
            return False

        roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
        if dot is None:
            logger.info("炼金和派遣：入口无红点，直接结束")
            return False

        logger.info(f"炼金和派遣：检测到入口红点 {dot}，点击 》 按钮进入")

        h, w = frame.shape[:2]
        click_roi = self.config.get("click_roi", DEFAULT_CONFIG["click_roi"])
        cx, cy = roi_center(click_roi, w, h)
        logger.info(f"[步骤1] 点击 》 按钮 ({cx}, {cy})")
        self.click(cx, cy)
        self.sleep(step_sleep)

        # TODO: 后续步骤在此追加

        logger.info("炼金和派遣：第一步执行完毕")
        return True
