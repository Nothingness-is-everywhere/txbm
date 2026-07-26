"""
Task state machine for game automation.

Implements a formal state machine with states INIT->NAVIGATE->EXECUTE->VERIFY->DONE/RETRY/FAIL,
supporting step timeouts, retry counts, checkpoint recovery, and structured transitions.
"""

import time
import logging
from enum import Enum
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass, field


class TaskState(str, Enum):
    """Enumeration of task lifecycle states."""

    INIT = "init"
    NAVIGATE = "navigate"
    EXECUTE = "execute"
    VERIFY = "verify"
    DONE = "done"
    RETRY = "retry"
    FAIL = "fail"


class TaskStatus(str, Enum):
    """Enumeration of task execution status."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


VALID_TRANSITIONS: Dict[TaskState, List[TaskState]] = {
    TaskState.INIT: [TaskState.NAVIGATE, TaskState.FAIL],
    TaskState.NAVIGATE: [TaskState.EXECUTE, TaskState.RETRY, TaskState.FAIL],
    TaskState.EXECUTE: [TaskState.VERIFY, TaskState.RETRY, TaskState.FAIL],
    TaskState.VERIFY: [TaskState.EXECUTE, TaskState.DONE, TaskState.RETRY, TaskState.FAIL],
    TaskState.RETRY: [TaskState.NAVIGATE, TaskState.EXECUTE, TaskState.FAIL],
    TaskState.DONE: [],
    TaskState.FAIL: [],
}

TERMINAL_STATES = {TaskState.DONE, TaskState.FAIL}


@dataclass
class StateTransition:
    """Record of a state transition."""

    from_state: TaskState
    to_state: TaskState
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


class GameStateMachine:
    """
    Formal state machine for game task orchestration.

    Manages task lifecycle through states: INIT -> NAVIGATE -> EXECUTE -> VERIFY -> DONE,
    with RETRY and FAIL as intermediate/terminal states.

    Features:
        - Validates state transitions
        - Tracks transition history
        - Supports checkpoint-based recovery
        - Provides hooks for state change callbacks

    Usage:
        sm = GameStateMachine()
        sm.add_transition_callback(lambda old, new: print(f"{old} -> {new}"))
        sm.navigate()
        sm.execute()
        sm.verify()
    """

    def __init__(
        self,
        initial_state: TaskState = TaskState.INIT,
        max_retries: int = 3,
        step_timeout: float = 30.0,
        checkpoint_interval: int = 1,
    ):
        """
        Initialize the state machine.

        Args:
            initial_state: Starting state of the machine.
            max_retries: Maximum number of retry attempts before failing.
            step_timeout: Default timeout for each state transition (seconds).
            checkpoint_interval: Save checkpoint every N transitions.
        """
        self._state = initial_state
        self._previous_state: Optional[TaskState] = None
        self._retries = 0
        self._max_retries = max_retries
        self._step_timeout = step_timeout
        self._transition_count = 0
        self._checkpoint_interval = checkpoint_interval
        self._history: List[StateTransition] = []
        self._checkpoints: List[Dict[str, Any]] = []
        self._status: TaskStatus = TaskStatus.RUNNING
        self._transition_callbacks: List[
            Callable[[TaskState, TaskState], None]
        ] = []
        self._logger = logging.getLogger(self.__class__.__name__)
        self._start_time = time.time()

    @property
    def state(self) -> TaskState:
        """Current state of the machine."""
        return self._state

    @property
    def status(self) -> TaskStatus:
        """Current execution status."""
        return self._status

    @property
    def retries(self) -> int:
        """Number of retry attempts so far."""
        return self._retries

    @property
    def history(self) -> List[StateTransition]:
        """Full history of state transitions."""
        return list(self._history)

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since initialization (seconds)."""
        return time.time() - self._start_time

    @property
    def is_terminal(self) -> bool:
        """Check if the machine is in a terminal state."""
        return self._state in TERMINAL_STATES

    def add_transition_callback(
        self, callback: Callable[[TaskState, TaskState], None]
    ) -> None:
        """
        Register a callback for state transitions.

        Args:
            callback: Function receiving (old_state, new_state).
        """
        self._transition_callbacks.append(callback)

    def can_transition_to(self, target: TaskState) -> bool:
        """
        Check if a transition to the target state is valid.

        Args:
            target: Target state to check.

        Returns:
            True if the transition is valid.
        """
        valid_next = VALID_TRANSITIONS.get(self._state, [])
        return target in valid_next

    def transition_to(
        self,
        target: TaskState,
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """
        Perform a state transition.

        Args:
            target: Target state.
            reason: Description of why this transition occurred.
            force: If True, skip validation (for recovery scenarios).

        Returns:
            True if transition was successful.

        Raises:
            ValueError: If the transition is invalid and not forced.
        """
        if self.is_terminal:
            self._logger.warning(
                f"Cannot transition from terminal state {self._state}"
            )
            return False

        if not force and not self.can_transition_to(target):
            valid_next = VALID_TRANSITIONS.get(self._state, [])
            raise ValueError(
                f"Invalid transition: {self._state} -> {target}. "
                f"Valid next states: {valid_next}"
            )

        old_state = self._state
        self._previous_state = old_state
        self._state = target
        self._transition_count += 1

        transition = StateTransition(
            from_state=old_state, to_state=target, reason=reason
        )
        self._history.append(transition)

        if target == TaskState.RETRY:
            self._retries += 1
            if self._retries > self._max_retries:
                self._logger.error(
                    f"Max retries ({self._max_retries}) exceeded. Transitioning to FAIL."
                )
                self._do_fail("Max retries exceeded")
                return False
            self._status = TaskStatus.RETRYING
        elif target == TaskState.DONE:
            self._status = TaskStatus.SUCCESS
        elif target == TaskState.FAIL:
            self._status = TaskStatus.FAILED
        else:
            self._status = TaskStatus.RUNNING

        if self._transition_count % self._checkpoint_interval == 0:
            self._save_checkpoint()

        self._logger.info(
            f"State transition: {old_state} -> {target} "
            f"(retries={self._retries}, reason={reason})"
        )

        for callback in self._transition_callbacks:
            try:
                callback(old_state, target)
            except Exception as e:
                self._logger.error(f"Transition callback error: {e}")

        return True

    def _do_fail(self, reason: str) -> None:
        """Force transition to FAIL state."""
        if self._state != TaskState.FAIL:
            self._state = TaskState.FAIL
            self._status = TaskStatus.FAILED
            transition = StateTransition(
                from_state=self._previous_state or TaskState.INIT,
                to_state=TaskState.FAIL,
                reason=reason,
            )
            self._history.append(transition)
            for callback in self._transition_callbacks:
                try:
                    callback(self._previous_state or TaskState.INIT, TaskState.FAIL)
                except Exception:
                    pass

    def navigate(self, reason: str = "") -> bool:
        """Transition to NAVIGATE state."""
        return self.transition_to(TaskState.NAVIGATE, reason)

    def execute(self, reason: str = "") -> bool:
        """Transition to EXECUTE state."""
        return self.transition_to(TaskState.EXECUTE, reason)

    def verify(self, reason: str = "") -> bool:
        """Transition to VERIFY state."""
        return self.transition_to(TaskState.VERIFY, reason)

    def complete(self, reason: str = "") -> bool:
        """Transition to DONE state."""
        return self.transition_to(TaskState.DONE, reason)

    def retry(self, reason: str = "") -> bool:
        """Transition to RETRY state or increment retry count if already retrying."""
        if self._state == TaskState.RETRY:
            self._retries += 1
            self._logger.info(
                f"Retry incremented: count={self._retries}, reason={reason}"
            )
            if self._retries > self._max_retries:
                self._logger.error(
                    f"Max retries ({self._max_retries}) exceeded during retry state"
                )
                self._do_fail("Max retries exceeded")
                return False
            return True
        return self.transition_to(TaskState.RETRY, reason)

    def fail(self, reason: str = "") -> bool:
        """Transition to FAIL state."""
        return self.transition_to(TaskState.FAIL, reason)

    def reset(self) -> None:
        """Reset the state machine to initial state."""
        self._state = TaskState.INIT
        self._previous_state = None
        self._retries = 0
        self._status = TaskStatus.RUNNING
        self._history.clear()
        self._checkpoints.clear()
        self._transition_count = 0
        self._start_time = time.time()

    def _save_checkpoint(self) -> None:
        """Save current state as a checkpoint for recovery."""
        checkpoint = {
            "state": self._state.value,
            "status": self._status.value,
            "retries": self._retries,
            "transition_count": self._transition_count,
            "timestamp": time.time(),
            "elapsed_time": self.elapsed_time,
            "history": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "reason": t.reason,
                    "timestamp": t.timestamp,
                }
                for t in self._history[-10:]
            ],
        }
        self._checkpoints.append(checkpoint)
        if len(self._checkpoints) > 20:
            self._checkpoints.pop(0)

    def get_last_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint data."""
        return self._checkpoints[-1] if self._checkpoints else None

    def recover_from_checkpoint(self) -> bool:
        """
        Attempt to recover from the previous checkpoint (rollback one step).

        Returns:
            True if recovery was successful.
        """
        if len(self._checkpoints) < 2:
            self._logger.warning(
                "Need at least 2 checkpoints for recovery"
            )
            return False

        self._checkpoints.pop()

        checkpoint = self._checkpoints[-1]
        try:
            self._state = TaskState(checkpoint["state"])
            self._status = TaskStatus(checkpoint["status"])
            self._retries = checkpoint["retries"]
            self._transition_count = checkpoint["transition_count"]
            self._logger.info(
                f"Recovered from checkpoint: {checkpoint['state']}"
            )
            return True
        except (ValueError, KeyError) as e:
            self._logger.error(f"Failed to recover from checkpoint: {e}")
            return False

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the state machine execution.

        Returns:
            Dictionary with execution metrics.
        """
        return {
            "final_state": self._state.value,
            "final_status": self._status.value,
            "total_retries": self._retries,
            "total_transitions": self._transition_count,
            "elapsed_time": round(self.elapsed_time, 2),
            "history_length": len(self._history),
            "checkpoints_saved": len(self._checkpoints),
            "success": self._state == TaskState.DONE,
        }
