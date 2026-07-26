"""
Unified Automation Task Runner for 天下布魔 (Tianxia Bumo).

Orchestrates multiple automation tasks in sequence with:
  - Auto-detection of game state
  - Configurable task chain
  - Logging and reporting
  - Error handling and retry logic

Usage:
  python -m ok.automation.runner
  python -m ok.automation.runner --task close_notice       # single task
  python -m ok.automation.runner --task start_game        # launch + auto
"""

import time
import logging
import argparse
from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import adbutils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("automation_runner")

ADB_SERIAL = "127.0.0.1:16384"
PACKAGE_NAME = "com.pinkcore.tkfm"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_name: str
    success: bool
    duration: float
    message: str = ""


@dataclass
class AutomationRunner:
    """Unified task runner for 天下布魔 automation."""

    adb_serial: str = ADB_SERIAL
    package_name: str = PACKAGE_NAME
    max_retries: int = 1
    retry_delay: float = 1.0
    results: List[TaskResult] = field(default_factory=list)

    def __post_init__(self):
        self.adb = adbutils.AdbClient()
        self.device = self.adb.device(self.adb_serial)

    def check_device(self) -> bool:
        """Check if the emulator/device is connected."""
        try:
            devices = self.adb.device_list()
            if not devices:
                logger.error("No ADB devices found")
                return False
            logger.info(f"Device connected: {len(devices)} device(s)")
            return True
        except Exception as e:
            logger.error(f"Device check failed: {e}")
            return False

    def launch_game(self, wait_time: float = 8.0) -> bool:
        """Launch the game application."""
        logger.info(f"Launching game: {self.package_name}")
        try:
            self.device.shell(
                f"monkey -p {self.package_name} "
                f"-c android.intent.category.LAUNCHER 1",
                timeout=15
            )
            logger.info(f"Waiting {wait_time}s for game to load...")
            time.sleep(wait_time)
            logger.info("Game launched successfully")
            return True
        except Exception as e:
            logger.error(f"Game launch failed: {e}")
            return False

    def force_stop_game(self) -> bool:
        """Force stop the game application."""
        logger.info(f"Force stopping: {self.package_name}")
        try:
            self.device.shell(
                f"am force-stop {self.package_name}",
                timeout=5
            )
            time.sleep(1)
            logger.info("Game stopped")
            return True
        except Exception as e:
            logger.error(f"Force stop failed: {e}")
            return False

    def run_task(
        self,
        task_name: str,
        task_func: Callable[[], bool],
        **kwargs: Any
    ) -> TaskResult:
        """Run a single task with retry logic."""
        logger.info("=" * 60)
        logger.info(f"Running task: {task_name}")
        logger.info("=" * 60)

        start_time = time.time()
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                logger.info(
                    f"Retry attempt {attempt}/{self.max_retries}"
                )
                time.sleep(self.retry_delay)

            try:
                success = task_func(**kwargs)
                duration = time.time() - start_time

                if success:
                    result = TaskResult(
                        task_name=task_name,
                        success=True,
                        duration=duration,
                        message="Task completed successfully"
                    )
                    self.results.append(result)
                    logger.info(
                        f"[SUCCESS] {task_name} completed in {duration:.2f}s"
                    )
                    return result
                else:
                    last_error = "Task returned False"
                    logger.warning(
                        f"[WARNING] {task_name} returned False "
                        f"(attempt {attempt})"
                    )
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"[ERROR] {task_name} failed with exception: {e} "
                    f"(attempt {attempt})"
                )

        duration = time.time() - start_time
        result = TaskResult(
            task_name=task_name,
            success=False,
            duration=duration,
            message=f"Failed after {self.max_retries} attempts: {last_error}"
        )
        self.results.append(result)
        logger.error(
            f"[FAILED] {task_name} failed after "
            f"{self.max_retries} attempts in {duration:.2f}s"
        )
        return result

    def run_pipeline(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[TaskResult]:
        """
        Run a pipeline of tasks sequentially.

        Args:
            tasks: List of task configurations, each with:
                - 'name': Task name (str)
                - 'func': Callable that returns bool
                - 'kwargs': Optional dict of keyword arguments
                - 'required': If True, pipeline stops on failure (default: True)
        """
        self.results.clear()

        for i, task_config in enumerate(tasks):
            name = task_config.get("name", f"task_{i}")
            func = task_config.get("func")
            kwargs = task_config.get("kwargs", {})
            required = task_config.get("required", True)

            if func is None:
                logger.error(f"Task {name} has no 'func' defined")
                if required:
                    logger.error("Required task missing function, stopping pipeline")
                    break
                continue

            result = self.run_task(name, func, **kwargs)

            if not result.success and required:
                logger.error(
                    f"Required task '{name}' failed, stopping pipeline"
                )
                break

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary of all tasks."""
        total = len(self.results)
        if total == 0:
            return {"total": 0, "success": 0, "failed": 0}

        success_count = sum(1 for r in self.results if r.success)
        failed_count = total - success_count
        total_duration = sum(r.duration for r in self.results)

        return {
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "total_duration": total_duration,
            "tasks": [
                {
                    "name": r.task_name,
                    "success": r.success,
                    "duration": r.duration,
                    "message": r.message
                }
                for r in self.results
            ]
        }

    def print_summary(self):
        """Print execution summary."""
        summary = self.get_summary()
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {summary['total']}")
        logger.info(f"Successful:  {summary['success']}")
        logger.info(f"Failed:      {summary['failed']}")
        if summary['total_duration'] > 0:
            logger.info(f"Total time:  {summary['total_duration']:.2f}s")
        logger.info("-" * 60)

        for task in summary.get("tasks", []):
            status = "✅" if task["success"] else "❌"
            logger.info(
                f"{status} {task['name']} "
                f"({task['duration']:.2f}s) - {task['message']}"
            )

        logger.info("=" * 60)


def close_notice_task() -> bool:
    """Task 1: Close notice popup with two-phase detection."""
    try:
        from ok.automation.close_notice_task import NoticeCloser
        closer = NoticeCloser()
        return closer.close_notice_popup()
    except Exception as e:
        logger.error(f"close_notice_task error: {e}")
        return False


def navigate_back_task() -> bool:
    """Task 2: Navigate back by clicking back button."""
    try:
        from ok.automation.back_navigation_task import BackButtonNavigator
        navigator = BackButtonNavigator()
        return navigator.navigate_back()
    except Exception as e:
        logger.error(f"navigate_back_task error: {e}")
        return False


def launch_and_setup_game() -> bool:
    """
    Full startup sequence:
      1. Launch game
      2. Close notice popup
      3. Navigate to main screen
      4. Read stamina values (public variables)
    """
    runner = AutomationRunner()

    if not runner.check_device():
        return False

    launch_ok = runner.launch_game(wait_time=8.0)
    if not launch_ok:
        return False

    time.sleep(2)

    tasks = [
        {
            "name": "close_notice_popup",
            "func": close_notice_task,
            "required": True
        },
        {
            "name": "navigate_to_main",
            "func": navigate_back_task,
            "required": False
        },
        {
            "name": "read_stamina",
            "func": read_stamina_task,
            "required": False
        },
    ]

    results = runner.run_pipeline(tasks)
    runner.print_summary()

    return all(r.success for r in results if r)


def read_stamina_task() -> bool:
    """Task: Read and store stamina values as public variables."""
    try:
        from ok.automation.stamina_reader import get_stamina
        state = get_stamina(force_refresh=True)
        logger.info("=" * 40)
        logger.info("STAMINA VALUES (public variables):")
        logger.info(f"  stamina1: {state.stamina1} (ratio: {state.stamina1.ratio:.1%})")
        logger.info(f"  stamina2: {state.stamina2} (ratio: {state.stamina2.ratio:.1%})")
        logger.info("=" * 40)
        return True
    except Exception as e:
        logger.error(f"read_stamina_task error: {e}")
        return False


def run_existing_tasks() -> bool:
    """
    Run the existing tasks without launching game.
    Assumes game is already running.
    """
    runner = AutomationRunner()

    if not runner.check_device():
        return False

    tasks = [
        {
            "name": "close_notice_popup",
            "func": close_notice_task,
            "required": False
        },
        {
            "name": "navigate_back",
            "func": navigate_back_task,
            "required": False
        },
    ]

    results = runner.run_pipeline(tasks)
    runner.print_summary()

    return all(r.success for r in results if r)


def main():
    parser = argparse.ArgumentParser(
        description="天下布魔 Automation Task Runner"
    )
    parser.add_argument(
        "--task",
        choices=["close_notice", "navigate_back", "start_game", "both"],
        default="both",
        help="Task to execute"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch game before executing tasks"
    )
    parser.add_argument(
        "--force-stop",
        action="store_true",
        help="Force stop game after tasks"
    )

    args = parser.parse_args()

    runner = AutomationRunner()

    try:
        if args.launch or args.task == "start_game":
            logger.info("=== Launch and Setup Mode ===")
            success = launch_and_setup_game()
        elif args.task == "close_notice":
            logger.info("=== Close Notice Task Only ===")
            success = runner.run_task(
                "close_notice", close_notice_task
            ).success
        elif args.task == "navigate_back":
            logger.info("=== Navigate Back Task Only ===")
            success = runner.run_task(
                "navigate_back", navigate_back_task
            ).success
        else:
            logger.info("=== Run Both Tasks ===")
            success = run_existing_tasks()

        if args.force_stop:
            runner.force_stop_game()

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())