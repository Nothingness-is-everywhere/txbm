"""
炼金和派遣任务（日常分类）。

流程：
  === 炼金 ===
  1. 检测主页炼金/派遣入口红点，有红点则点击 》 按钮进入"炼金和派遣"页面；无红点则结束。
  2. 进入页面后检测指定区域红点，有红点则点击该区域。
  3. 步骤2点击后，点击"需求"按钮（固定位置）。
  4. 步骤3后加载"保存"按钮模板文件，循环匹配并点击，直到匹配不到为止。
  5. 点击返回按钮回到主界面。
  === 派遣 ===
  6. 返回主界面后再次检测入口红点并点击 》 重新进入"炼金和派遣"页面。
  7. 检测派遣入口模板（templates/dispatch_entry.png），匹配则点击进入派遣页面。
  8-10. 依次执行3个派遣子任务，每个子任务序列：点选区 → 点继续 → 点选区自身 → 点确认。
  11. 点击与炼金相同的返回按钮位置回到主界面。

红点检测复用 ok_tasks._red_dot 中的共享实现（与 HomeRedDotTask 同一套方法）。
主页检测 / 模板匹配复用 ok_tasks._home 中的共享实现（is_on_home / match_template_in_roi）。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ok.task.task import BaseTask

from ok_tasks._home import is_on_home, load_template_image, match_template_in_roi
from ok_tasks._red_dot import detect_red_dot, roi_center

logger = logging.getLogger("AlchemyDispatchTask")


# 选区信息: x=0, y=877, w=96, h=131  (图片尺寸 1080x1920)
# rx=0.0000, ry=0.4568, rw=0.0889, rh=0.0682  中心 (0.0444, 0.4906)
_ENTRY_ROI = [0.0, 0.4568, 0.0889, 0.5250]

# 步骤2选区信息: x=128, y=676, w=97, h=112  (图片尺寸 1080x1920)
# rx=0.1185, ry=0.3521, rw=0.0898, rh=0.0583  中心 (0.1630, 0.3812)
_STEP2_ROI = [0.1185, 0.3521, 0.2083, 0.4104]

# 步骤3选区信息: x=592, y=1164, w=288, h=72  (图片尺寸 1080x1920)
# rx=0.5481, ry=0.6062, rw=0.2667, rh=0.0375  中心 (0.6815, 0.6250)
_STEP3_ROI = [0.5481, 0.6062, 0.8148, 0.6437]

# 步骤4选区信息（保存按钮模板匹配区域）: x=52, y=1340, w=244, h=60  (图片尺寸 1080x1920)
# rx=0.0481, ry=0.6979, rw=0.2259, rh=0.0312  中心 (0.1611, 0.7135)
_STEP4_ROI = [0.0481, 0.6979, 0.2740, 0.7291]

# 步骤5选区信息（返回主界面按钮）: x=8, y=76, w=180, h=80  (图片尺寸 1080x1920)
# rx=0.0074, ry=0.0396, rw=0.1667, rh=0.0417  中心 (0.0907, 0.0604)
_STEP5_ROI = [0.0074, 0.0396, 0.1741, 0.0813]

# 步骤7选区信息（派遣入口，模板匹配）: x=42, y=853, w=183, h=64  (图片尺寸 1080x1920)
# rx=0.0389, ry=0.4443, rw=0.1694, rh=0.0333  中心 (0.1231, 0.4609)
_DISPATCH_ENTRY_ROI = [0.0389, 0.4443, 0.2083, 0.4776]

# 步骤8派遣子任务1选区: x=493, y=1331, w=201, h=86  中心 (0.5491, 0.7156)
_DISPATCH_ITEM_1_ROI = [0.4565, 0.6932, 0.6426, 0.7380]
# 步骤9派遣子任务2选区: x=468, y=1536, w=176, h=56  中心 (0.5148, 0.8146)
_DISPATCH_ITEM_2_ROI = [0.4333, 0.8000, 0.5963, 0.8292]
# 步骤10派遣子任务3选区: x=516, y=1700, w=56, h=92  中心 (0.5037, 0.9094)
_DISPATCH_ITEM_3_ROI = [0.4778, 0.8854, 0.5297, 0.9333]
# 派遣"继续"按钮选区（每子任务第2次点击，固定位置）: x=472, y=1572, w=148, h=52  中心 (0.5056, 0.8323)
_DISPATCH_CONTINUE_ROI = [0.4370, 0.8187, 0.5740, 0.8458]
# 派遣"确认"按钮选区（每子任务第4次点击，固定位置）: x=484, y=1676, w=160, h=144  中心 (0.5222, 0.9104)
_DISPATCH_CONFIRM_ROI = [0.4481, 0.8729, 0.5962, 0.9479]

# 主页检测区域（与 HomeRedDotTask 一致，用于返回主界面后校验）: x=37, y=69, w=279, h=107
_HOME_ROI = [0.0343, 0.0359, 0.2926, 0.0916]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大开始按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 红点检测区域（相对坐标 [x1, y1, x2, y2]）
    "red_dot_roi": list(_ENTRY_ROI),
    # 检测到红点后点击的区域（》按钮所在区域），点击其中心
    "click_roi": list(_ENTRY_ROI),
    # 步骤2：进入页面后的红点检测区域
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
    # 步骤5：返回主界面按钮区域（固定位置点击），点击其中心（派遣步骤11复用此位置）
    "step5_click_roi": list(_STEP5_ROI),
    # 步骤7：派遣入口模板路径（使用项目已有 templates/dispatch_entry.png）
    "dispatch_entry_template_path": "templates/dispatch_entry.png",
    # 步骤7：派遣入口模板匹配搜索区域（相对坐标 [x1, y1, x2, y2]）
    "dispatch_entry_template_roi": list(_DISPATCH_ENTRY_ROI),
    # 步骤7：派遣入口匹配置信度阈值
    "dispatch_entry_threshold": 0.70,
    # 步骤7：匹配成功后点击的区域，点击其中心
    "dispatch_entry_click_roi": list(_DISPATCH_ENTRY_ROI),
    # 步骤8-10：3个派遣子任务选区（每个子任务序列：点选区→点继续→点选区自身→点确认）
    "dispatch_item_1_roi": list(_DISPATCH_ITEM_1_ROI),
    "dispatch_item_2_roi": list(_DISPATCH_ITEM_2_ROI),
    "dispatch_item_3_roi": list(_DISPATCH_ITEM_3_ROI),
    # 派遣"继续"按钮区域（固定位置，所有子任务共用）
    "dispatch_continue_roi": list(_DISPATCH_CONTINUE_ROI),
    # 派遣"确认"按钮区域（固定位置，所有子任务共用）
    "dispatch_confirm_roi": list(_DISPATCH_CONFIRM_ROI),
    # 主页检测（返回主界面后校验用，复用 HomeRedDotTask 的主页模板）
    "home_template_path": "templates/home_profile_button.png",
    "home_template_roi": list(_HOME_ROI),
    "home_threshold": 0.70,
    # 返回主界面后未回主页时的最大补点次数
    "home_check_max_attempts": 3,
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

    步骤1：检测入口红点，点击 》 进入"炼金和派遣"页面；无红点则结束。
    步骤2：进入页面后检测指定区域红点，有红点则点击该区域。
    步骤3：步骤2点击后，点击"需求"按钮固定位置。
    步骤4：加载"保存"按钮模板文件，循环匹配并点击直到匹配不到。
    步骤5：点击返回按钮回到主界面。
    步骤6：返回主界面后再次检测入口红点并点击 》 重新进入"炼金和派遣"页面。
    步骤7：检测派遣入口模板，匹配则点击进入派遣页面。
    步骤8-10：依次执行3个派遣子任务，每个序列：点选区→点继续→点选区自身→点确认。
    步骤11：点击与炼金相同的返回按钮位置回到主界面。
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
        # 单独启动本任务时不连带启用 enable_after_start 任务（如 GameStartupTask），保持单独执行
        self.standalone_start = True
        self._step4_tpl: Optional[np.ndarray] = None
        self._dispatch_tpl: Optional[np.ndarray] = None
        self._home_tpl: Optional[np.ndarray] = None

    def on_create(self):
        # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
        # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
        self._enabled = False
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_step4_template()
        self._load_dispatch_template()
        self._load_home_template()

    def _load_step4_template(self):
        """加载步骤4保存按钮模板文件到内存。"""
        rel = self.config.get("step4_template_path", DEFAULT_CONFIG["step4_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"炼金和派遣：步骤4模板未找到，步骤4将无法匹配: {rel}")
        self._step4_tpl = tpl

    def _load_dispatch_template(self):
        """加载派遣入口模板文件到内存（使用项目已有 templates/dispatch_entry.png）。"""
        rel = self.config.get("dispatch_entry_template_path", DEFAULT_CONFIG["dispatch_entry_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"派遣：入口模板未找到，步骤7将无法匹配: {rel}")
        self._dispatch_tpl = tpl

    def _load_home_template(self):
        """加载主页模板文件到内存（返回主界面校验用）。"""
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"炼金和派遣：主页模板未找到，返回主界面校验将跳过: {rel}")
        self._home_tpl = tpl

    def _is_home(self, frame: np.ndarray) -> bool:
        """通过主页模板匹配判断当前是否在主界面。"""
        if self._home_tpl is None:
            return False
        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        threshold = float(self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"]))
        return is_on_home(frame, self._home_tpl, roi, threshold)

    def _ensure_home(self, click_roi, label: str) -> bool:
        """点击返回按钮后校验是否回到主界面；未回主页则按 click_roi 补点几次。

        label 用于日志标识（如"炼金步骤5"、"派遣步骤11"）。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(self.config.get("home_check_max_attempts", DEFAULT_CONFIG["home_check_max_attempts"]))
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning(f"炼金和派遣：{label} 校验主页时无画面可用")
                return False
            if self._is_home(frame):
                logger.info(f"炼金和派遣：{label} 已回到主界面（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(f"炼金和派遣：{label} 未回主界面，第 {attempt} 次补点返回 ({cx}, {cy})")
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"炼金和派遣：{label} {max_attempts} 次补点后仍未回主界面")
        return False

    def _match_template(self, frame: np.ndarray, template: np.ndarray, roi, threshold: float) -> Tuple[bool, float]:
        """在 frame 的 roi 区域内匹配 template，返回 (是否匹配, 最高置信度)。"""
        return match_template_in_roi(frame, template, roi, threshold)

    def _enter_combined_page(self, frame: np.ndarray, label: str, step_sleep: float) -> bool:
        """检测入口红点并点击 》 按钮进入"炼金和派遣"页面。返回是否进入。"""
        roi = self.config.get("red_dot_roi", DEFAULT_CONFIG["red_dot_roi"])
        dot = detect_red_dot(frame, roi, self.config, DEFAULT_CONFIG)
        if dot is None:
            logger.info(f"{label}：入口无红点")
            return False
        logger.info(f"{label}：检测到入口红点 {dot}，点击 》 按钮进入")
        h, w = frame.shape[:2]
        click_roi = self.config.get("click_roi", DEFAULT_CONFIG["click_roi"])
        cx, cy = roi_center(click_roi, w, h)
        logger.info(f"[{label}] 点击 》 按钮 ({cx}, {cy})")
        self.click(cx, cy)
        self.sleep(step_sleep)
        return True

    def run(self):
        """
        === 炼金 ===
        步骤1：检测入口红点 → 点击 》 进入"炼金和派遣"页面（无红点则结束）。
        步骤2：进入页面后检测指定区域红点 → 有红点则点击该区域。
        步骤3：步骤2点击后 → 点击"需求"按钮固定位置。
        步骤4：加载"保存"按钮模板 → 循环匹配并点击直到匹配不到。
        步骤5：点击返回按钮回到主界面。
        === 派遣 ===
        步骤6：返回主界面后再次检测入口红点 → 点击 》 重新进入"炼金和派遣"页面。
        步骤7：检测派遣入口模板 → 匹配则点击进入派遣页面。
        步骤8-10：3个派遣子任务，每个序列：点选区→点继续→点选区自身→点确认。
        步骤11：点击与炼金相同的返回按钮位置回到主界面。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None or  not self._is_home(frame):
            logger.warning("炼金和派遣：无画面可用，终止")
            return False

        # === 炼金流程 ===
        # 步骤1：检测入口红点，点击 》 进入"炼金和派遣"页面；无红点则结束
        if not self._enter_combined_page(frame, "炼金", step_sleep):
            logger.info("炼金和派遣：入口无红点，结束")
            return False

        # 步骤2：进入页面后取新帧，检测指定区域红点，有则点击该区域
        step2_clicked = False
        frame2 = self.next_frame()
        if frame2 is None:
            logger.warning("炼金：步骤2无画面可用，跳过")
        else:
            roi2 = self.config.get("step2_red_dot_roi", DEFAULT_CONFIG["step2_red_dot_roi"])
            dot2 = detect_red_dot(frame2, roi2, self.config, DEFAULT_CONFIG)
            if dot2 is None:
                logger.info("炼金：步骤2区域无红点，跳过")
            else:
                logger.info(f"炼金：步骤2检测到红点 {dot2}，点击该区域")
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
            logger.info("炼金：步骤2未点击，跳过步骤3")

        # 步骤4：加载"保存"按钮模板，循环匹配并点击直到匹配不到
        if step3_done:
            frame4 = self.next_frame()
            if frame4 is None:
                logger.warning("炼金：步骤4无画面可用，跳过")
            else:
                h4, w4 = frame4.shape[:2]
                tpl_roi4 = self.config.get("step4_template_roi", DEFAULT_CONFIG["step4_template_roi"])
                template = self._step4_tpl
                if template is None:
                    logger.warning("炼金：步骤4模板不可用，跳过")
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
                            logger.info(f"炼金：步骤4未匹配到保存按钮 (置信度 {conf:.3f} < {threshold})，结束循环")
                            break
                        logger.info(f"[步骤4] 匹配到保存按钮 (置信度 {conf:.3f})，第 {clicked + 1} 次点击 ({cx4}, {cy4})")
                        self.click(cx4, cy4)
                        clicked += 1
                        self.sleep(interval)
                        current = self.next_frame()
                        if current is None:
                            logger.warning("炼金：步骤4循环中无画面可用，结束循环")
                            break
                    logger.info(f"炼金：步骤4共点击 {clicked} 次保存按钮")
        else:
            logger.info("炼金：步骤3未执行，跳过步骤4")

        # 步骤5：点击返回按钮回到主界面
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
            logger.warning("炼金：步骤5无画面可用，跳过")

        # 步骤5后校验是否回到主界面，未回主页则按返回按钮补点几次
        self._ensure_home(self.config.get("step5_click_roi", DEFAULT_CONFIG["step5_click_roi"]), "炼金步骤5")

        # === 派遣流程 ===
        # 步骤6：返回主界面后再次检测入口红点并点击 》 重新进入"炼金和派遣"页面
        frame6 = self.next_frame()
        if frame6 is None:
            logger.warning("派遣：步骤6无画面可用，跳过派遣流程")
        elif not self._enter_combined_page(frame6, "派遣", step_sleep):
            logger.info("派遣：入口无红点，跳过派遣流程")
        else:
            # 步骤7：检测派遣入口模板，匹配则点击进入派遣页面
            dispatch_entered = False
            frame7 = self.next_frame()
            if frame7 is None:
                logger.warning("派遣：步骤7无画面可用，跳过")
            else:
                dispatch_roi = self.config.get("dispatch_entry_template_roi", DEFAULT_CONFIG["dispatch_entry_template_roi"])
                template7 = self._dispatch_tpl
                if template7 is None:
                    logger.warning("派遣：入口模板不可用，跳过")
                else:
                    threshold7 = float(self.config.get("dispatch_entry_threshold", DEFAULT_CONFIG["dispatch_entry_threshold"]))
                    matched, conf = self._match_template(frame7, template7, dispatch_roi, threshold7)
                    if matched:
                        h7, w7 = frame7.shape[:2]
                        click_roi7 = self.config.get("dispatch_entry_click_roi", DEFAULT_CONFIG["dispatch_entry_click_roi"])
                        cx7, cy7 = roi_center(click_roi7, w7, h7)
                        logger.info(f"[步骤7] 匹配到派遣入口 (置信度 {conf:.3f})，点击进入派遣页面 ({cx7}, {cy7})")
                        self.click(cx7, cy7)
                        self.sleep(step_sleep)
                        dispatch_entered = True
                    else:
                        logger.info(f"派遣：步骤7未匹配到派遣入口 (置信度 {conf:.3f} < {threshold7})，跳过派遣子任务")

            # 步骤8-10：3个派遣子任务，每个序列：点选区→点继续→点选区自身→点确认
            if dispatch_entered:
                frame8 = self.next_frame()
                if frame8 is None:
                    frame8 = frame7
                if frame8 is None:
                    logger.warning("派遣：步骤8无画面可用，跳过子任务")
                else:
                    h8, w8 = frame8.shape[:2]
                    items = [
                        self.config.get("dispatch_item_1_roi", DEFAULT_CONFIG["dispatch_item_1_roi"]),
                        self.config.get("dispatch_item_2_roi", DEFAULT_CONFIG["dispatch_item_2_roi"]),
                        self.config.get("dispatch_item_3_roi", DEFAULT_CONFIG["dispatch_item_3_roi"]),
                    ]
                    continue_roi = self.config.get("dispatch_continue_roi", DEFAULT_CONFIG["dispatch_continue_roi"])
                    confirm_roi = self.config.get("dispatch_confirm_roi", DEFAULT_CONFIG["dispatch_confirm_roi"])
                    cont_x, cont_y = roi_center(continue_roi, w8, h8)
                    conf_x, conf_y = roi_center(confirm_roi, w8, h8)
                    for idx, item_roi in enumerate(items, start=1):
                        ix, iy = roi_center(item_roi, w8, h8)
                        step_no = 7 + idx  # 步骤8/9/10
                        logger.info(f"[步骤{step_no}] 派遣子任务{idx}：点击选区 ({ix}, {iy})")
                        self.click(ix, iy)
                        self.sleep(step_sleep)
                        logger.info(f"[步骤{step_no}] 派遣子任务{idx}：点击继续 ({cont_x}, {cont_y})")
                        self.click(cont_x, cont_y)
                        self.sleep(step_sleep)
                        logger.info(f"[步骤{step_no}] 派遣子任务{idx}：点击自身 ({ix}, {iy})")
                        self.click(ix, iy)
                        self.sleep(step_sleep)
                        logger.info(f"[步骤{step_no}] 派遣子任务{idx}：点击确认 ({conf_x}, {conf_y})")
                        self.click(conf_x, conf_y)
                        self.sleep(step_sleep)

                    # 步骤11：点击与炼金相同的返回按钮位置回到主界面
                    ret_roi = self.config.get("step5_click_roi", DEFAULT_CONFIG["step5_click_roi"])
                    rx, ry = roi_center(ret_roi, w8, h8)
                    logger.info(f"[步骤11] 点击 返回主界面 按钮 ({rx}, {ry})")
                    self.click(rx, ry)
                    self.sleep(step_sleep)

                    # 步骤11后校验是否回到主界面，未回主页则补点几次
                    self._ensure_home(ret_roi, "派遣步骤11")

        logger.info("炼金和派遣：流程执行完毕")
        return True
