"""
Game Startup Task for 天下布魔 (Tianxia Bumo).

This task:
  1. Waits for game to become ready (state-driven, not fixed delay)
  2. Closes notice popup
  3. Closes check-in screen
  4. Navigates back to main screen
  5. Reads stamina values
  6. Completes once all steps finish

Used with enable_after_start=True for automatic execution after game launch.

Configuration (environment variable priority):
  - POST_START_MIN_DELAY: Minimum wait before polling starts (default: 3s)
  - POST_START_MAX_WAIT: Maximum wait before giving up (default: 20s)
  - POST_START_POLL_INTERVAL: Polling interval (default: 0.5s)
  - POST_START_TASK_DELAY: Legacy, overrides min_delay if set (compatible)
"""

import os
import time
import logging

import numpy as np

from ok.task.task import BaseTask

logger = logging.getLogger("GameStartupTask")

# Default configuration values
DEFAULT_MIN_DELAY = 15  # Changed from 3s to 15s for longer game loading wait
DEFAULT_MAX_WAIT = 20
DEFAULT_POLL_INTERVAL = 0.5


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
        self.visible = False  # Hide from GUI task list, this is a startup-only task

    def on_create(self):
        self._enabled = True

    def _get_config_value(self, env_key: str, default: float) -> float:
        """Read config value from environment variable."""
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                logger.warning(f"Invalid {env_key} value '{env_val}', using default {default}")
        return default

    def _get_min_delay(self) -> float:
        """Get minimum delay before polling starts.
        
        Priority: POST_START_TASK_DELAY (legacy) > POST_START_MIN_DELAY > default (3s)
        """
        # Legacy support: POST_START_TASK_DELAY overrides min_delay
        legacy_val = os.environ.get('POST_START_TASK_DELAY')
        if legacy_val is not None:
            try:
                return float(legacy_val)
            except ValueError:
                logger.warning(f"Invalid POST_START_TASK_DELAY value '{legacy_val}', using POST_START_MIN_DELAY")
        
        return self._get_config_value('POST_START_MIN_DELAY', DEFAULT_MIN_DELAY)

    def _get_max_wait(self) -> float:
        """Get maximum wait time before giving up."""
        return self._get_config_value('POST_START_MAX_WAIT', DEFAULT_MAX_WAIT)

    def _get_poll_interval(self) -> float:
        """Get polling interval for checking game readiness."""
        return self._get_config_value('POST_START_POLL_INTERVAL', DEFAULT_POLL_INTERVAL)

    def _is_loading_in_progress(self, screen: np.ndarray) -> bool:
        """Check if loading/progress bar is visible.
        
        Returns True if loading is detected (we should keep waiting).
        Returns False if no loading detected (could be ready).
        """
        try:
            # Check for loading indicators using template matching
            from ok.automation.close_notice_task import NoticeCloser
            closer = NoticeCloser()
            
            # If notice popup is already visible, loading is done
            if closer.find_notice_popup(screen) is not None:
                logger.debug("Notice popup detected, loading complete")
                return False
            
            # If check-in screen is visible, loading is done
            from ok.automation.checkin_close_task import CheckInCloser
            checkin_closer = CheckInCloser()
            if checkin_closer.find_checkin_screen(screen) is not None:
                logger.debug("Check-in screen detected, loading complete")
                return False
            
            # If back button is visible (on a navigable screen), loading is done
            from ok.automation.back_navigation_task import BackButtonNavigator
            navigator = BackButtonNavigator()
            if navigator.find_back_button(screen) is not None:
                logger.debug("Back button detected, loading complete")
                return False
            
            # If post-close state is detected (TAP TO START), loading is done
            if closer.find_post_close(screen):
                logger.debug("Post-close state detected, loading complete")
                return False
            
            # Otherwise, might be loading
            logger.debug("No interactable state detected, assuming loading in progress")
            return True
            
        except Exception as e:
            logger.warning(f"_is_loading_in_progress detection failed: {e}")
            # If detection fails, be conservative - assume still loading
            return True

    def _is_interactable_state(self, screen: np.ndarray) -> bool:
        """Check if game is in an interactable state.
        
        Returns True if we can proceed with startup tasks.
        """
        try:
            from ok.automation.close_notice_task import NoticeCloser
            closer = NoticeCloser()
            
            # Notice popup visible = interactable
            if closer.find_notice_popup(screen) is not None:
                logger.debug("Interactable: notice popup visible")
                return True
            
            # Post-close state (TAP TO START) = interactable
            if closer.find_post_close(screen):
                logger.debug("Interactable: post-close state")
                return True
            
            # Check-in screen visible = interactable
            from ok.automation.checkin_close_task import CheckInCloser
            checkin_closer = CheckInCloser()
            if checkin_closer.find_checkin_screen(screen) is not None:
                logger.debug("Interactable: check-in screen visible")
                return True
            
            # Back button visible = interactable
            from ok.automation.back_navigation_task import BackButtonNavigator
            navigator = BackButtonNavigator()
            if navigator.find_back_button(screen) is not None:
                logger.debug("Interactable: back button visible")
                return True
            
            # Stamina values visible = interactable (on main screen)
            try:
                from ok.automation.stamina_reader import get_stamina
                state = get_stamina(force_refresh=True)
                if state.stamina1.value > 0 or state.stamina2.value > 0:
                    logger.debug("Interactable: stamina values detected")
                    return True
            except:
                pass
            
            logger.debug("Not in interactable state")
            return False
            
        except Exception as e:
            logger.warning(f"_is_interactable_state detection failed: {e}")
            return False

    def wait_for_game_ready(self) -> bool:
        """Wait for game to become ready using state-driven polling.
        
        Returns True when game is ready, False on timeout (but continues anyway).
        """
        min_delay = self._get_min_delay()
        max_wait = self._get_max_wait()
        poll_interval = self._get_poll_interval()
        
        logger.info(f"Waiting for game to become ready: min_delay={min_delay}s, max_wait={max_wait}s, poll_interval={poll_interval}s")
        
        # Minimum delay first
        logger.info(f"Waiting minimum delay: {min_delay}s")
        time.sleep(min_delay)
        
        # Polling loop
        start_time = time.time()
        attempts = 0
        
        while time.time() - start_time < max_wait:
            attempts += 1
            
            try:
                from ok.automation.close_notice_task import NoticeCloser
                closer = NoticeCloser()
                screen = closer.capture_screen()
                
                is_loading = self._is_loading_in_progress(screen)
                is_interactable = self._is_interactable_state(screen)
                
                if not is_loading or is_interactable:
                    logger.info(f"Game ready detected after {time.time() - start_time:.1f}s ({attempts} attempts)")
                    return True
                
                logger.debug(f"Game not ready yet (attempt {attempts}, elapsed {time.time() - start_time:.1f}s)")
                
            except Exception as e:
                logger.warning(f"Wait poll attempt {attempts} failed: {e}")
            
            time.sleep(poll_interval)
        
        # Timeout
        logger.warning(f"Game readiness timeout after {max_wait}s, proceeding conservatively (no blind clicks)")
        return False

    def run(self):
        """
        Execute the full startup sequence with state-driven waiting:
          1. Wait for game to become ready (min delay + state polling)
          2. Close notice popup
          3. Close check-in screen
          4. Navigate back to main screen
          5. Read stamina values
        """
        logger.info("=" * 60)
        logger.info(f"Running {self.name}")
        logger.info("=" * 60)

        try:
            # Wait for game to become ready (state-driven)
            self.wait_for_game_ready()

            logger.info("post-start startup-task begin: close_notice_popup")
            success = self._execute_startup_sequence()
            
            if success:
                logger.info(f"startup-task result: success")
                logger.info(f"{self.name} completed successfully")
                return True
            else:
                logger.info(f"startup-task result: skip/warning (no action needed)")
                logger.info(f"{self.name} finished with some steps skipped")
                return True

        except Exception as e:
            logger.error(f"startup-task result: failed - {e}")
            logger.error(f"{self.name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _execute_startup_sequence(self) -> bool:
        """Execute the full startup sequence following real flow:
        Step A: Wait for game ready (done before this method)
        Step B: Detect & Close Notice (conditional)
        Step C: Process Post-Close Screen (mandatory)
        Step D: Detect & Close Check-in (conditional)
        Step E: Verify Main Screen
        """
        logger.info("=" * 60)
        logger.info("Executing startup sequence...")
        logger.info("=" * 60)

        # Step B: Detect & Close Notice (conditional)
        logger.info("Step B: Detect & Close Notice Popup")
        notice_result = self._close_notice_popup()
        if not notice_result:
            logger.warning("Step B failed (critical error), but continuing to Step C")

        # Step C: Process Post-Close Screen (MANDATORY - must execute)
        logger.info("Step C: Process Post-Close Screen (Mandatory)")
        post_close_result = self._process_post_close_screen()
        
        # Handle Step C result semantics
        if post_close_result == 'success':
            logger.info("Step C: confirm button clicked successfully")
        elif post_close_result == 'bypass':
            logger.info("Step C: bypassed (check-in or main screen detected)")
        elif post_close_result == 'warning':
            logger.warning("Step C: timeout, proceeding conservatively")
        elif post_close_result == 'failed':
            logger.error("Step C: critical error, cannot proceed")
            return False
        else:
            # Fallback for unknown return values
            logger.warning(f"Step C: unknown result '{post_close_result}', proceeding conservatively")

        # Step D: Detect & Close Check-in (conditional)
        logger.info("Step D: Detect & Close Check-in Screen")
        if self._should_close_checkin():
            checkin_result = self._close_checkin_screen()
            if not checkin_result:
                logger.warning("Step D failed, but continuing")
        else:
            logger.info("  skip: no check-in screen detected (assumed at main screen)")

        # Step E: Verify Main Screen
        logger.info("Step E: Verify Main Screen")
        self._verify_main_screen()

        # Read stamina values (optional, for info)
        logger.info("Reading stamina values...")
        self._read_stamina()

        logger.info("=" * 60)
        logger.info("Startup sequence completed successfully")
        logger.info("=" * 60)
        return True

    def _process_post_close_screen(self) -> str:
        """Process post-close screen (Step C - mandatory).
        
        Returns:
            'success': Confirm button clicked successfully
            'bypass': Skipped because check-in or main screen detected
            'warning': Timed out but proceeding conservatively
            'failed': Critical error
        """
        try:
            from ok.automation.close_notice_task import NoticeCloser
            closer = NoticeCloser()
            return closer.process_post_close_screen()
        except Exception as e:
            logger.error(f"_process_post_close_screen error: {e}")
            return 'failed'

    def _verify_main_screen(self) -> bool:
        """Verify we've reached the main screen (Step E)."""
        try:
            from ok.automation.stamina_reader import get_stamina, update_global_stamina
            update_global_stamina()
            state = get_stamina(force_refresh=True)
            
            # Check if stamina values are present (indicates main screen)
            if state.stamina1.value > 0 or state.stamina2.value > 0:
                logger.info(f"Main screen verified: stamina1={state.stamina1.value}, stamina2={state.stamina2.value}")
                return True
            else:
                logger.warning("Main screen verification: stamina values not found, state may be uncertain")
                return False
        except Exception as e:
            logger.warning(f"Main screen verification failed: {e}")
            return False

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

    def _should_close_notice(self) -> bool:
        """Check if notice popup is visible and needs to be closed."""
        try:
            from ok.automation.close_notice_task import NoticeCloser
            closer = NoticeCloser()
            screen = closer.capture_screen()
            result = closer.find_notice_popup(screen)
            if result is not None:
                logger.info(f"  notice popup detected (conf={result[2]:.3f}), need to close")
                return True
            else:
                logger.info("  no notice popup detected")
                return False
        except Exception as e:
            logger.warning(f"_should_close_notice detection failed: {e}")
            return True

    def _should_close_checkin(self) -> bool:
        """Check if check-in screen is visible and needs to be closed."""
        try:
            from ok.automation.checkin_close_task import CheckInCloser
            closer = CheckInCloser()
            screen = closer.capture_screen()
            result = closer.find_checkin_screen(screen)
            if result is not None:
                logger.info(f"  check-in screen detected (conf={result[2]:.3f}), need to close")
                return True
            else:
                logger.info("  no check-in screen detected")
                return False
        except Exception as e:
            logger.warning(f"_should_close_checkin detection failed: {e}")
            return True

    def _should_navigate_back(self) -> bool:
        """Check if back button is visible (indicating not at main screen)."""
        try:
            from ok.automation.back_navigation_task import BackButtonNavigator
            navigator = BackButtonNavigator()
            screen = navigator.capture_screen()
            result = navigator.find_back_button(screen)
            if result is not None:
                logger.info(f"  back button detected (conf={result[2]:.3f}), need to navigate back")
                return True
            else:
                logger.info("  back button not detected, assume already at main screen")
                return False
        except Exception as e:
            logger.warning(f"_should_navigate_back detection failed: {e}")
            return True
