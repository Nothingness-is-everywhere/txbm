"""
日常升级任务：日常升两级（天下布魔 Tianxia Bumo）。

流程：
  1. 模板匹配确认在主页，不在则结束；在则点击角色入口进入角色界面。
  2. 模板匹配校验是否进入角色界面，未进入则补点角色入口（最多重试若干次）。
  3. 模板匹配升级目标指示，未匹配则点击切换按钮，直到匹配（最多重试若干次）。
  4. 点击升级按钮。
  5. 模板匹配校验升级页面，未匹配则补点升级按钮（最多重试若干次）。
  6. 运行两次：点击确认按钮 → 再点击确认按钮；两次运行之间单独间隔 10 秒。

主页检测 / 模板匹配复用 ok_tasks._home 中的共享实现（is_on_home / match_template_in_roi）。
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ok.task.task import BaseTask

from ok_tasks._home import is_on_home, load_template_image, match_template_in_roi
from ok_tasks._red_dot import roi_center

logger = logging.getLogger("DailyLevelUpTask")


# 步骤1：角色入口选区 x=304, y=1768, w=132, h=60 @ 1080x1920 中心 (370, 1798)
# rx=0.2815, ry=0.9208, rw=0.1222, rh=0.0312
_ENTRY_ROI = [0.2815, 0.9208, 0.4037, 0.9520]

# 步骤2：角色界面模板匹配选区 x=108, y=804, w=344, h=76 @ 1080x1920 中心 (280, 842)
# rx=0.1000, ry=0.4188, rw=0.3185, rh=0.0396
_CHARACTER_PAGE_ROI = [0.1000, 0.4188, 0.4185, 0.4583]

# 步骤3：升级目标指示模板匹配选区 x=112, y=1468, w=248, h=80 @ 1080x1920 中心 (236, 1508)
# rx=0.1037, ry=0.7646, rw=0.2296, rh=0.0417
_TARGET_INDICATOR_ROI = [0.1037, 0.7646, 0.3333, 0.8062]

# 步骤3：未匹配时点击的切换按钮选区 x=924, y=404, w=52, h=76 @ 1080x1920 中心 (950, 442)
# rx=0.8556, ry=0.2104, rw=0.0481, rh=0.0396
_SWITCH_CLICK_ROI = [0.8556, 0.2104, 0.9037, 0.2500]

# 步骤4：升级按钮选区 x=884, y=1480, w=188, h=68 @ 1080x1920 中心 (978, 1514)
# rx=0.8185, ry=0.7708, rw=0.1741, rh=0.0354
_UPGRADE_BUTTON_ROI = [0.8185, 0.7708, 0.9926, 0.8062]

# 步骤5：升级页面模板匹配选区 x=24, y=172, w=580, h=96 @ 1080x1920 中心 (314, 220)
# rx=0.0222, ry=0.0896, rw=0.5370, rh=0.0500
_UPGRADE_INDICATOR_ROI = [0.0222, 0.0896, 0.5593, 0.1396]

# 步骤6：确认按钮选区 x=188, y=1224, w=88, h=84 @ 1080x1920 中心 (232, 1266)
# rx=0.1741, ry=0.6375, rw=0.0815, rh=0.0437
_CONFIRM_CLICK_ROI = [0.1741, 0.6375, 0.2556, 0.6812]

# 主页检测区域（与 HomeRedDotTask / AlchemyDispatchTask 一致）
_HOME_ROI = [0.0343, 0.0359, 0.2926, 0.0916]

DEFAULT_CONFIG = {
    "_enabled": True,
    # 是否跟随"周常日常"大开始按钮一起执行（配置面板仅显示此项，其他技术配置项隐藏）
    "follow_batch_start": True,
    # 主页检测（模板匹配）
    "home_template_path": "templates/home_profile_button.png",
    "home_template_roi": list(_HOME_ROI),
    "home_threshold": 0.70,
    # 步骤1：角色入口点击区域（点击其中心）
    "entry_click_roi": list(_ENTRY_ROI),
    # 步骤2：角色界面模板匹配校验
    "character_page_template_path": "templates/levelup_character_page.png",
    "character_page_template_roi": list(_CHARACTER_PAGE_ROI),
    "character_page_threshold": 0.70,
    "character_page_check_max_attempts": 3,
    # 步骤3：升级目标指示模板匹配
    "target_indicator_template_path": "templates/levelup_target_indicator.png",
    "target_indicator_template_roi": list(_TARGET_INDICATOR_ROI),
    "target_indicator_threshold": 0.70,
    # 步骤3：循环点击切换按钮直到匹配的安全上限（0 表示不限制）
    "target_indicator_max_iters": 30,
    # 步骤3：未匹配时点击的切换按钮区域（点击其中心）
    "switch_click_roi": list(_SWITCH_CLICK_ROI),
    # 步骤4：升级按钮点击区域（点击其中心）
    "upgrade_button_roi": list(_UPGRADE_BUTTON_ROI),
    # 步骤5：升级页面模板匹配校验
    "upgrade_indicator_template_path": "templates/levelup_upgrade_indicator.png",
    "upgrade_indicator_template_roi": list(_UPGRADE_INDICATOR_ROI),
    "upgrade_indicator_threshold": 0.70,
    "upgrade_indicator_check_max_attempts": 3,
    # 步骤6：确认按钮点击区域（点击其中心）
    "confirm_click_roi": list(_CONFIRM_CLICK_ROI),
    # 步骤6：两次运行之间的单独间隔（秒）
    "levelup_run_interval": 10.0,
    # 每步点击后的等待秒数（让游戏反应）
    "post_click_sleep": 1.0,
}


class DailyLevelUpTask(BaseTask):
    """
    日常任务：日常升两级。

    流程：
      1. 确认在主页 → 点击角色入口进入角色界面（不在主页则结束）。
      2. 模板匹配校验角色界面 → 未进入则补点角色入口。
      3. 模板匹配升级目标指示 → 未匹配则点击切换按钮，直到匹配。
      4. 点击升级按钮。
      5. 模板匹配校验升级页面 → 未匹配则补点升级按钮。
      6. 运行两次：点击确认按钮 → 再点击确认按钮（两次之间间隔 10 秒）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "日常升两级"
        self.description = "进入角色界面后点击升级，确认两次完成升两级"
        self.visible = True
        self.default_config = dict(DEFAULT_CONFIG)
        # 配置面板仅显示 follow_batch_start，其他技术项隐藏
        self.config_type = {k: {'hidden': True} for k in DEFAULT_CONFIG
                            if not k.startswith('_') and k != 'follow_batch_start'}
        self.config_description = {"follow_batch_start": "是否跟随大开始启动"}
        self.enable_after_start = False
        # 单独启动本任务时不连带启用 enable_after_start 任务（如 GameStartupTask），保持单独执行
        self.standalone_start = True
        self._home_tpl: Optional[np.ndarray] = None
        self._character_page_tpl: Optional[np.ndarray] = None
        self._target_indicator_tpl: Optional[np.ndarray] = None
        self._upgrade_indicator_tpl: Optional[np.ndarray] = None

    def on_create(self):
        # 永不自动启动；用户必须点"批量启动"或单个"Start"按钮才会执行。
        # 忽略 config 中可能残留的 _enabled=True（旧版本持久化的值）。
        self._enabled = False
        self.follow_batch_start = self.config.get("follow_batch_start", True)
        self._load_home_template()
        self._load_character_page_template()
        self._load_target_indicator_template()
        self._load_upgrade_indicator_template()

    def run(self):
        """
        日常升两级主流程：每步间隔 post_click_sleep 让游戏反应。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))

        frame = self.executor.frame
        if frame is None:
            frame = self.next_frame()
        if frame is None:
            logger.warning("日常升两级：无画面可用，终止")
            return False

        # 1. 确认在主页，不在则结束
        if not self._is_home(frame):
            logger.info("日常升两级：不在主页，终止")
            return False

        h, w = frame.shape[:2]

        # 点击角色入口进入角色界面
        e_x, e_y = roi_center(self.config.get("entry_click_roi", DEFAULT_CONFIG["entry_click_roi"]), w, h)
        logger.info(f"[步骤1] 点击角色入口 ({e_x}, {e_y})")
        self.click(e_x, e_y)
        self.sleep(step_sleep)

        # 2. 模板匹配校验是否进入角色界面；未进入则补点角色入口
        if not self._ensure_character_page(self.config.get("entry_click_roi", DEFAULT_CONFIG["entry_click_roi"])):
            logger.warning("日常升两级：未能进入角色界面，终止")
            return False

        # 3. 模板匹配升级目标指示；未匹配则点击切换按钮，直到匹配
        if not self._ensure_target_indicator():
            logger.warning("日常升两级：未能匹配到升级目标指示，终止")
            return False

        # 4. 点击升级按钮
        frame4 = self.next_frame()
        if frame4 is None:
            frame4 = frame
        h4, w4 = frame4.shape[:2]
        u_x, u_y = roi_center(self.config.get("upgrade_button_roi", DEFAULT_CONFIG["upgrade_button_roi"]), w4, h4)
        logger.info(f"[步骤4] 点击升级按钮 ({u_x}, {u_y})")
        self.click(u_x, u_y)
        self.sleep(step_sleep)

        # 5. 模板匹配校验升级页面；未匹配则补点升级按钮
        if not self._ensure_upgrade_page(self.config.get("upgrade_button_roi", DEFAULT_CONFIG["upgrade_button_roi"])):
            logger.warning("日常升两级：未能进入升级页面，终止")
            return False

        # 6. 运行两次：点击确认按钮 → 再点击确认按钮（两次之间间隔 10 秒）
        run_interval = float(self.config.get("levelup_run_interval", DEFAULT_CONFIG["levelup_run_interval"]))
        frame6 = self.next_frame()
        if frame6 is None:
            frame6 = frame4
        h6, w6 = frame6.shape[:2]
        c_x, c_y = roi_center(self.config.get("confirm_click_roi", DEFAULT_CONFIG["confirm_click_roi"]), w6, h6)
        for run_idx in range(1, 3):
            logger.info(f"[步骤6] 第 {run_idx} 次运行：点击确认按钮 ({c_x}, {c_y})")
            self.click(c_x, c_y)
            self.sleep(step_sleep)
            logger.info(f"[步骤6] 第 {run_idx} 次运行：再次点击确认按钮 ({c_x}, {c_y})")
            self.click(c_x, c_y)
            if run_idx < 2:
                logger.info(f"[步骤6] 两次运行之间等待 {run_interval} 秒")
                self.sleep(run_interval)
            else:
                self.sleep(step_sleep)

        logger.info("日常升两级：流程执行完毕")
        return True

    # ---------- 模板加载 ----------

    def _load_home_template(self):
        """加载主页模板文件到内存。"""
        rel = self.config.get("home_template_path", DEFAULT_CONFIG["home_template_path"])
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"日常升两级：主页模板未找到，主页检测将失效: {rel}")
        self._home_tpl = tpl

    def _load_character_page_template(self):
        """加载步骤2角色界面模板；模板缺失时校验将被跳过（不阻塞流程）。"""
        rel = self.config.get(
            "character_page_template_path", DEFAULT_CONFIG["character_page_template_path"]
        )
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"日常升两级：角色界面模板未找到，步骤2校验将跳过: {rel}")
        self._character_page_tpl = tpl

    def _load_target_indicator_template(self):
        """加载步骤3升级目标指示模板。"""
        rel = self.config.get(
            "target_indicator_template_path", DEFAULT_CONFIG["target_indicator_template_path"]
        )
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"日常升两级：升级目标指示模板未找到，步骤3将无法匹配: {rel}")
        self._target_indicator_tpl = tpl

    def _load_upgrade_indicator_template(self):
        """加载步骤5升级页面模板；模板缺失时校验将被跳过（不阻塞流程）。"""
        rel = self.config.get(
            "upgrade_indicator_template_path", DEFAULT_CONFIG["upgrade_indicator_template_path"]
        )
        tpl = load_template_image(rel)
        if tpl is None:
            logger.warning(f"日常升两级：升级页面模板未找到，步骤5校验将跳过: {rel}")
        self._upgrade_indicator_tpl = tpl

    # ---------- 模板匹配辅助 ----------

    def _is_home(self, frame: np.ndarray) -> bool:
        """通过主页模板匹配判断当前是否在主界面。"""
        if self._home_tpl is None:
            return False
        roi = self.config.get("home_template_roi", DEFAULT_CONFIG["home_template_roi"])
        threshold = float(self.config.get("home_threshold", DEFAULT_CONFIG["home_threshold"]))
        return is_on_home(frame, self._home_tpl, roi, threshold)

    def _is_on_character_page(self, frame: np.ndarray) -> bool:
        """检测当前画面是否已进入角色界面；模板缺失时返回 True（跳过校验，向后兼容）。"""
        if self._character_page_tpl is None:
            return True
        roi = self.config.get(
            "character_page_template_roi", DEFAULT_CONFIG["character_page_template_roi"]
        )
        threshold = float(
            self.config.get("character_page_threshold", DEFAULT_CONFIG["character_page_threshold"])
        )
        return is_on_home(frame, self._character_page_tpl, roi, threshold)

    def _is_target_matched(self, frame: np.ndarray) -> Tuple[bool, float]:
        """检测升级目标指示是否匹配；返回 (是否匹配, 最高置信度)。"""
        if self._target_indicator_tpl is None:
            return False, 0.0
        roi = self.config.get(
            "target_indicator_template_roi", DEFAULT_CONFIG["target_indicator_template_roi"]
        )
        threshold = float(
            self.config.get("target_indicator_threshold", DEFAULT_CONFIG["target_indicator_threshold"])
        )
        return match_template_in_roi(frame, self._target_indicator_tpl, roi, threshold)

    def _is_on_upgrade_page(self, frame: np.ndarray) -> bool:
        """检测当前画面是否已进入升级页面；模板缺失时返回 True（跳过校验，向后兼容）。"""
        if self._upgrade_indicator_tpl is None:
            return True
        roi = self.config.get(
            "upgrade_indicator_template_roi", DEFAULT_CONFIG["upgrade_indicator_template_roi"]
        )
        threshold = float(
            self.config.get("upgrade_indicator_threshold", DEFAULT_CONFIG["upgrade_indicator_threshold"])
        )
        return is_on_home(frame, self._upgrade_indicator_tpl, roi, threshold)

    # ---------- 流程校验 ----------

    def _ensure_character_page(self, click_roi) -> bool:
        """步骤2：校验是否进入角色界面；未进入则按 click_roi（角色入口）补点几次。
        模板缺失时 _is_on_character_page 返回 True（跳过校验，与炼金派遣一致）。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(
            self.config.get(
                "character_page_check_max_attempts",
                DEFAULT_CONFIG["character_page_check_max_attempts"],
            )
        )
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning("日常升两级：校验角色界面时无画面可用")
                return False
            if self._is_on_character_page(frame):
                logger.info(f"日常升两级：已进入角色界面（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(f"日常升两级：未进入角色界面，第 {attempt} 次补点入口 ({cx}, {cy})")
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"日常升两级：{max_attempts} 次补点后仍未进入角色界面")
        return False

    def _ensure_target_indicator(self) -> bool:
        """步骤3：模板匹配升级目标指示；未匹配则点击切换按钮，直到匹配或达到上限。
        模板缺失时 _is_target_matched 返回 (False, 0.0)，因此早退出并直接失败。
        """
        if self._target_indicator_tpl is None:
            logger.warning(
                "日常升两级：步骤3升级目标指示模板未找到 (templates/levelup_target_indicator.png)，"
                "步骤3将无法匹配"
            )
            return False
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_iters = int(
            self.config.get("target_indicator_max_iters", DEFAULT_CONFIG["target_indicator_max_iters"])
        )
        switch_roi = self.config.get("switch_click_roi", DEFAULT_CONFIG["switch_click_roi"])
        clicked = 0
        while max_iters <= 0 or clicked < max_iters:
            frame = self.next_frame()
            if frame is None:
                logger.warning("日常升两级：步骤3匹配时无画面可用")
                return False
            matched, conf = self._is_target_matched(frame)
            if matched:
                logger.info(
                    f"[步骤3 #{clicked + 1}] 匹配到升级目标指示 (conf={conf:.3f})"
                )
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(switch_roi, w, h)
            logger.info(
                f"[步骤3 #{clicked + 1}] 未匹配到升级目标指示 (最高 conf={conf:.3f})，"
                f"点击切换按钮 ({cx}, {cy})"
            )
            self.click(cx, cy)
            clicked += 1
            self.sleep(step_sleep)
        logger.warning(f"日常升两级：步骤3点击切换按钮 {clicked} 次后仍未匹配到升级目标指示")
        return False

    def _ensure_upgrade_page(self, click_roi) -> bool:
        """步骤5：校验是否进入升级页面；未进入则按 click_roi（升级按钮）补点几次。
        模板缺失时 _is_on_upgrade_page 返回 True（跳过校验，与炼金派遣一致）。
        """
        step_sleep = float(self.config.get("post_click_sleep", DEFAULT_CONFIG["post_click_sleep"]))
        max_attempts = int(
            self.config.get(
                "upgrade_indicator_check_max_attempts",
                DEFAULT_CONFIG["upgrade_indicator_check_max_attempts"],
            )
        )
        for attempt in range(1, max_attempts + 1):
            frame = self.next_frame()
            if frame is None:
                logger.warning("日常升两级：校验升级页面时无画面可用")
                return False
            if self._is_on_upgrade_page(frame):
                logger.info(f"日常升两级：已进入升级页面（第 {attempt} 次确认）")
                return True
            h, w = frame.shape[:2]
            cx, cy = roi_center(click_roi, w, h)
            logger.info(f"日常升两级：未进入升级页面，第 {attempt} 次补点升级按钮 ({cx}, {cy})")
            self.click(cx, cy)
            self.sleep(step_sleep)
        logger.warning(f"日常升两级：{max_attempts} 次补点后仍未进入升级页面")
        return False

    # ---------- 辅助 ----------

    def _match_template_direct(
        self, frame: np.ndarray, template: np.ndarray, roi, threshold: float
    ) -> Tuple[bool, float]:
        """在 frame 的 roi 区域内匹配 template，返回 (是否匹配, 最高置信度)。
        与 ok_tasks._home.match_template_in_roi 逻辑相同，此处内联以便在需要时保存诊断。
        """
        return match_template_in_roi(frame, template, roi, threshold)
