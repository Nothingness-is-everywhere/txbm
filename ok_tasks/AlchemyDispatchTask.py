"""
炼金和派遣任务（日常分类）。

流程：
  1. 检测主页入口区域红点，有红点则点击 》 按钮进入炼金/派遣界面；无红点则结束。
  2. 进入界面后检测指定区域红点，有红点则点击该区域。
  3. 步骤2点击后，点击"需求"按钮（固定位置）。
  4. 步骤3后加载"保存"按钮模板文件，循环匹配并点击，直到匹配不到为止。
  5. 点击返回按钮回到主界面。

后续步骤将在本任务中陆续追加。

红点检测复用 ok_tasks._red_dot 中的共享实现（与 HomeRedDotTask 同一套方法）。
模板匹配复用 cv2.matchTemplate（与 HomeRedDotTask._is_home 同一套技术）。
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ok.task.task import BaseTask

from ok_tasks._red_dot import detect_red_dot, roi_center, roi_to_pixels

logger = logging.getLogger("AlchemyDispatchTask")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 选区信息: x=0, y=877, w=96, h=131  (图片尺寸 1080x1920)
# rx=0.0000, ry=0.4568, rw=0.0889, rh=0.0682  中心 (0.0444, 0.4906)
_ENTRY_ROI = [0.0, 0.4568, 0.0889, 0.5250]

# 步骤2选区信息: x=128, y=676, w=97, h=112  (图片尺寸 1080x1920)
# rx=0.1185, ry=0.3521, rw=0.0898, rh=0.0583  中心 (0.1630, 0.3812)
_STEP2_ROI = [0.1185, 0.3521, 0.2083, 0.4104]

# 步骤3选区信息: x=592, y=1164, w=288, h=72  (图片尺寸 1080x1920)
# rx=0.5481, ry=0.6062, rw=0.2667, rh=0.0375  中心 (0.6815, 0.6250)
_STEP3_ROI = [0.5481, 0.6062, 0.8148, 0.6437]

# 步骤4选区信息（模板匹配区域，运行时截图剪切为模板）: x=52, y=1340, w=244, h=60  (图片尺寸 1080x1920)
# rx=0.0481, ry=0.6979, rw=0.2259, rh=0.0312  中心 (0.1611, 0.7135)
_STEP4_ROI = [0.0481, 0.6979, 0.2740, 0.7291]

# 步骤5选区信息（返回主界面按钮）: x=8, y=76, w=180, h=80  (图片尺寸 1080x1920)
# rx=0.0074, ry=0.0396, rw=0.1667, rh=0.0417  中心 (0.0907, 0.0604)
_STEP5_ROI = [0.0074, 0.0396, 0.1741, 0.0813]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大启动按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 红点检测区域（相对坐标 [x1, y1, x2, y2]）
    "red_dot_roi": list(_ENTRY_ROI),
    # 检测到红点后点击的区域（》按钮所在区域），点击其中心
    "click_roi": list(_ENTRY_ROI),
    # 步骤2：进入界面后的红点检测区域
    "step2_red_dot_roi": list(_STEP2_ROI),
    # 步骤2：检测到红点后点击的区域，点击其中心
    "step2_click_roi": list(_STEP2_ROI),
    # 步骤3：步骤2点击后点击的"需求"按钮区域（固定位置点击），点击其中心
    "step3_click_roi": list(_STEP3_ROI),
    # 步骤4：保存按钮模板文件路径
    "step4_template_path": "templates/alchemy_save_button.png",
    # 步骤4：模板匹配搜索区域（相对坐标 [x1, y1, x2, y2]）
    "step4_template_roi": list(_STEP4_ROI),
    # 步骤4：模板匹配置信度阈值
    "step4_threshold": 0.70,
    # 步骤4：匹配成功后点击的区域，点击其中心
    "step4_click_roi": list(_STEP4_ROI),
    # 步骤4：循环点击的间隔秒数（每次点击后等待再重新匹配）
    "step4_click_interval": 2.0,
    # 步骤4：循环点击安全上限（防止异常时无限循环，0 表示不限制）
    "step4_max_iters": 50,
    # 步骤5：返回主界面按钮区域（固定位置点击），点击其中心
    "step5_click_roi": list(_STEP5_ROI),
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

    步骤1：检测主页入口红点，有红点则点击 》 按钮进入；无红点则结束。
    步骤2：进入界面后检测指定区域红点，有红点则点击该区域。
    步骤3：步骤2点击后，点击"需求"按钮固定位置。
    步骤4：加载"保存"按钮模板文件，循环匹配并点击直到匹配不到。
    步骤5：点击返回按钮回到主界面。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "炼金和派遣"
        self.description = "检测炼金/派遣红点并依次点击进入与领取"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        self._step4_tpl: Optional[np.ndarray] = None

    def on_create(self):
        self._enabled = self.config.get("_enabled", True)
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_step4_template()

    def _load_step4_template(self):
        """加载步骤4保存按钮模板文件到内存。"""
        rel = self.config.get("step4_template_path", DEFAULT_CONFIG["step4_template_path"])
        p = Path(rel)
        abs_path = p if p.is_absolute() else (_PROJECT_ROOT / p)
        try:
            tpl = cv2.imdecode(np.fromfile(str(abs_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"炼金和派遣：读取步骤4模板失败: {abs_path} ({e})")
            tpl = None
        if tpl is None:
            logger.warning(f"炼金和派遣：步骤4模板未找到，步骤4将无法匹配: {abs_path}")
        self._step4_tpl = tpl

    def _match_template(self, frame: np.ndarray, template: np.ndarray, roi, threshold: float):
        """在 frame 的 roi 区域内匹配 template，返回 (是否匹配, 最高置信度)。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi_to_pixels(roi, w, h)
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

    def run(self):
        """
        步骤1：检测入口红点 → 点击 》 按钮（无红点则结束）。
        步骤2：进入界面后检测指定区域红点 → 有红点则点击该区域。
        步骤3：步骤2点击后 → 点击"需求"按钮固定位置。
        步骤4：加载"保存"按钮模板 → 循环匹配并点击直到匹配不到。
        步骤5：点击返回按钮回到主界面。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("炼金和派遣：无画面可用，终止")
            return False

        # 步骤1：入口红点检测（无红点直接结束，有红点点击 》 按钮进入）
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

        # 步骤2：进入界面后取新帧，检测指定区域红点，有则点击该区域
        step2_clicked = False
        frame2 = self.next_frame()
        if frame2 is None:
            logger.warning("炼金和派遣：步骤2无画面可用，跳过")
        else:
            roi2 = self.config.get("step2_red_dot_roi", DEFAULT_CONFIG["step2_red_dot_roi"])
            dot2 = detect_red_dot(frame2, roi2, self.config, DEFAULT_CONFIG)
            if dot2 is None:
                logger.info("炼金和派遣：步骤2区域无红点，跳过")
            else:
                logger.info(f"炼金和派遣：步骤2检测到红点 {dot2}，点击该区域")
                h2, w2 = frame2.shape[:2]
                click_roi2 = self.config.get("step2_click_roi", DEFAULT_CONFIG["step2_click_roi"])
                cx2, cy2 = roi_center(click_roi2, w2, h2)
                logger.info(f"[步骤2] 点击区域 ({cx2}, {cy2})")
                self.click(cx2, cy2)
                self.sleep(step_sleep)
                step2_clicked = True

        # 步骤3：步骤2点击后，点击"需求"按钮固定位置（仅步骤2确实点击时执行）
        step3_done = False
        if step2_clicked:
            h3, w3 = frame2.shape[:2]
            click_roi3 = self.config.get("step3_click_roi", DEFAULT_CONFIG["step3_click_roi"])
            cx3, cy3 = roi_center(click_roi3, w3, h3)
            logger.info(f"[步骤3] 点击 需求 按钮 ({cx3}, {cy3})")
            self.click(cx3, cy3)
            self.sleep(step_sleep)
            step3_done = True
        else:
            logger.info("炼金和派遣：步骤2未点击，跳过步骤3")

        # 步骤4：加载/自动生成"保存"按钮模板，循环匹配并点击直到匹配不到
        if step3_done:
            frame4 = self.next_frame()
            if frame4 is None:
                logger.warning("炼金和派遣：步骤4无画面可用，跳过")
            else:
                h4, w4 = frame4.shape[:2]
                tpl_roi4 = self.config.get("step4_template_roi", DEFAULT_CONFIG["step4_template_roi"])
                template = self._step4_tpl
                if template is None:
                    logger.warning("炼金和派遣：步骤4模板不可用，跳过")
                else:
                    threshold = float(self.config.get("step4_threshold", DEFAULT_CONFIG["step4_threshold"]))
                    click_roi4 = self.config.get("step4_click_roi", DEFAULT_CONFIG["step4_click_roi"])
                    cx4, cy4 = roi_center(click_roi4, w4, h4)
                    interval = float(self.config.get("step4_click_interval", DEFAULT_CONFIG["step4_click_interval"]))
                    max_iters = int(self.config.get("step4_max_iters", DEFAULT_CONFIG["step4_max_iters"]))
                    clicked = 0
                    current = frame4
                    while max_iters <= 0 or clicked < max_iters:
                        matched, conf = self._match_template(current, template, tpl_roi4, threshold)
                        if not matched:
                            logger.info(f"炼金和派遣：步骤4未匹配到保存按钮 (置信度 {conf:.3f} < {threshold})，结束循环")
                            break
                        logger.info(f"[步骤4] 匹配到保存按钮 (置信度 {conf:.3f})，第 {clicked + 1} 次点击 ({cx4}, {cy4})")
                        self.click(cx4, cy4)
                        clicked += 1
                        self.sleep(interval)
                        current = self.next_frame()
                        if current is None:
                            logger.warning("炼金和派遣：步骤4循环中无画面可用，结束循环")
                            break
                    logger.info(f"炼金和派遣：步骤4共点击 {clicked} 次保存按钮")
        else:
            logger.info("炼金和派遣：步骤3未执行，跳过步骤4")

        # 步骤5：点击返回按钮回到主界面（进入炼金/派遣界面后总要返回）
        frame5 = self.next_frame()
        if frame5 is None:
            frame5 = frame
        if frame5 is not None:
            h5, w5 = frame5.shape[:2]
            click_roi5 = self.config.get("step5_click_roi", DEFAULT_CONFIG["step5_click_roi"])
            cx5, cy5 = roi_center(click_roi5, w5, h5)
            logger.info(f"[步骤5] 点击 返回主界面 按钮 ({cx5}, {cy5})")
            self.click(cx5, cy5)
            self.sleep(step_sleep)
        else:
            logger.warning("炼金和派遣：步骤5无画面可用，跳过")

        # TODO: 后续步骤在此追加

        logger.info("炼金和派遣：流程执行完毕")
        return True
