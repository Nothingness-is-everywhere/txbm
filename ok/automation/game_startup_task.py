"""
Game Startup Task for 天下布魔 (Tianxia Bumo).

This task:
  1. Runs after game is started
  2. Closes notice popup
  3. Closes check-in screen
  4. Navigates back to main screen
  5. Reads stamina values
  6. Completes once all steps finish

Used with enable_after_start=True for automatic execution after game launch.
"""

import time
import logging

from ok.task.task import BaseTask

logger = logging.getLogger("GameStartupTask")


class GameStartupTask(BaseTask):
    """
    Task that runs the full game startup sequence:
      - Close notice popup
      - Close check-in screen
      - Navigate back to main screen
      - Read stamina values
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "GameStartup"
        self.description = "自动处理游戏启动弹窗（公告、签到）并读取体力"
        self.enable_after_start = True

    def on_create(self):
        self._enabled = True

    def run(self):
        """
        Execute the full startup sequence:
          1. Close notice popup
          2. Close check-in screen
          3. Navigate back to main screen
          4. Read stamina values
        """
        logger.info("=" * 60)
        logger.info(f"Running {self.name}")
        logger.info("=" * 60)

        try:
            success = self._execute_startup_sequence()
            
            if success:
                logger.info(f"{self.name} completed successfully")
                return True
            else:
                logger.info(f"{self.name} finished with some steps skipped")
                return True

        except Exception as e:
            logger.error(f"{self.name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_startup_sequence(self) -> bool:
        """Execute the full startup sequence."""
        step_results = []

        logger.info("Step 1: Close notice popup")
        step_results.append(self._close_notice_popup())

        logger.info("Step 2: Close check-in screen")
        step_results.append(self._close_checkin_screen())

        logger.info("Step 3: Navigate back to main screen")
        step_results.append(self._navigate_back_if_needed())

        logger.info("Step 4: Read stamina values")
        step_results.append(self._read_stamina())

        return all(step_results)

    def _close_notice_popup(self) -> bool:
        """Detect and close notice popup using ADB automation."""
        try:
            from ok.automation.close_notice_task import NoticeCloser
            closer = NoticeCloser()
            result = closer.close_notice_popup()
            logger.info(f"_close_notice_popup: {result}")
            return result
        except Exception as e:
            logger.error(f"_close_notice_popup error: {e}")
            return False

    def _close_checkin_screen(self) -> bool:
        """Detect and close check-in screen using ADB automation."""
        try:
            from ok.automation.checkin_close_task import CheckInCloser
            closer = CheckInCloser()
            result = closer.close_checkin_screen()
            logger.info(f"_close_checkin_screen: {result}")
            return result
        except Exception as e:
            logger.error(f"_close_checkin_screen error: {e}")
            return False

    def _navigate_back_if_needed(self) -> bool:
        """Navigate back to main screen using ADB automation."""
        try:
            from ok.automation.back_navigation_task import BackButtonNavigator
            navigator = BackButtonNavigator()
            result = navigator.navigate_back()
            logger.info(f"_navigate_back_if_needed: {result}")
            return result
        except Exception as e:
            logger.error(f"_navigate_back_if_needed error: {e}")
            return False

    def _read_stamina(self) -> bool:
        """Read and store stamina values as public variables."""
        try:
            from ok.automation.stamina_reader import get_stamina, update_global_stamina
            update_global_stamina()
            state = get_stamina(force_refresh=True)
            logger.info("=" * 40)
            logger.info("STAMINA VALUES (public variables):")
            logger.info(f"  stamina1: {state.stamina1} (ratio: {state.stamina1.ratio:.1%})")
            logger.info(f"  stamina2: {state.stamina2} (ratio: {state.stamina2.ratio:.1%})")
            logger.info("=" * 40)
            return True
        except Exception as e:
            logger.error(f"_read_stamina error: {e}")
            return False
