"""
Tests for the GameStateMachine class.

Covers:
    - State initialization
    - Valid state transitions
    - Invalid state transition rejection
    - Retry and max retry logic
    - Terminal state handling
    - Checkpoint save and recovery
    - Transition callbacks
    - Execution summary generation
"""

import time
import pytest

from ok.automation.state_machine import (
    TaskState,
    TaskStatus,
    GameStateMachine,
    StateTransition,
    VALID_TRANSITIONS,
    TERMINAL_STATES,
)


class TestTaskState:
    """Tests for TaskState enumeration."""

    def test_state_values(self):
        assert TaskState.INIT == "init"
        assert TaskState.NAVIGATE == "navigate"
        assert TaskState.EXECUTE == "execute"
        assert TaskState.VERIFY == "verify"
        assert TaskState.DONE == "done"
        assert TaskState.RETRY == "retry"
        assert TaskState.FAIL == "fail"

    def test_all_states_covered(self):
        expected_states = {"init", "navigate", "execute", "verify", "done", "retry", "fail"}
        actual_states = {s.value for s in TaskState}
        assert actual_states == expected_states


class TestTaskStatus:
    """Tests for TaskStatus enumeration."""

    def test_status_values(self):
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.SUCCESS == "success"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.RETRYING == "retrying"
        assert TaskStatus.CANCELLED == "cancelled"
        assert TaskStatus.TIMEOUT == "timeout"


class TestGameStateMachine:
    """Tests for GameStateMachine class."""

    def test_initial_state(self):
        sm = GameStateMachine()
        assert sm.state == TaskState.INIT
        assert sm.status == TaskStatus.RUNNING
        assert sm.is_terminal is False
        assert sm.retries == 0

    def test_custom_initial_state(self):
        sm = GameStateMachine(initial_state=TaskState.NAVIGATE)
        assert sm.state == TaskState.NAVIGATE

    def test_valid_transitions(self):
        sm = GameStateMachine()
        assert sm.navigate("go") is True
        assert sm.state == TaskState.NAVIGATE
        assert sm.execute("run") is True
        assert sm.state == TaskState.EXECUTE
        assert sm.verify("check") is True
        assert sm.state == TaskState.VERIFY
        assert sm.complete("done") is True
        assert sm.state == TaskState.DONE
        assert sm.is_terminal is True

    def test_navigate_transitions(self):
        sm = GameStateMachine()
        sm.navigate("first")
        assert sm.state == TaskState.NAVIGATE
        
        sm.execute("second")
        assert sm.state == TaskState.EXECUTE

    def test_retry_transition(self):
        sm = GameStateMachine(max_retries=3)
        sm.navigate("start")
        sm.retry("first retry")
        assert sm.state == TaskState.RETRY
        assert sm.retries == 1
        assert sm.status == TaskStatus.RETRYING

    def test_max_retries_exceeded(self):
        sm = GameStateMachine(max_retries=2)
        sm.navigate("start")
        
        sm.retry("retry 1")
        assert sm.retries == 1
        assert sm.state == TaskState.RETRY

        sm.retry("retry 2")
        assert sm.retries == 2
        assert sm.state == TaskState.RETRY

        sm.retry("retry 3 - should fail")
        assert sm.state == TaskState.FAIL
        assert sm.status == TaskStatus.FAILED
        assert sm.is_terminal is True

    def test_invalid_transition_raises(self):
        sm = GameStateMachine()
        with pytest.raises(ValueError) as exc_info:
            sm.complete("invalid from init")
        assert "Invalid transition" in str(exc_info.value)

    def test_force_transition(self):
        sm = GameStateMachine()
        result = sm.transition_to(TaskState.NAVIGATE, "forced", force=True)
        assert result is True
        assert sm.state == TaskState.NAVIGATE

    def test_terminal_state_prevents_transition(self):
        sm = GameStateMachine()
        sm.navigate("start")
        sm.execute("run")
        sm.verify("check")
        sm.complete("done")
        
        result = sm.navigate("should fail")
        assert result is False
        assert sm.state == TaskState.DONE

    def test_fail_transition(self):
        sm = GameStateMachine()
        sm.navigate("start")
        sm.fail("error occurred")
        assert sm.state == TaskState.FAIL
        assert sm.status == TaskStatus.FAILED
        assert sm.is_terminal is True

    def test_reset(self):
        sm = GameStateMachine()
        sm.navigate("nav")
        sm.execute("exec")
        sm.retry("retry")
        
        sm.reset()
        assert sm.state == TaskState.INIT
        assert sm.status == TaskStatus.RUNNING
        assert sm.retries == 0
        assert len(sm.history) == 0

    def test_history_tracking(self):
        sm = GameStateMachine()
        sm.navigate("step1")
        sm.execute("step2")
        sm.verify("step3")

        history = sm.history
        assert len(history) == 3
        assert history[0].from_state == TaskState.INIT
        assert history[0].to_state == TaskState.NAVIGATE
        assert history[1].from_state == TaskState.NAVIGATE
        assert history[1].to_state == TaskState.EXECUTE
        assert history[2].from_state == TaskState.EXECUTE
        assert history[2].to_state == TaskState.VERIFY

    def test_transition_callbacks(self):
        sm = GameStateMachine()
        callbacks = []

        def on_transition(old, new):
            callbacks.append((old, new))

        sm.add_transition_callback(on_transition)
        sm.navigate("nav")
        sm.execute("exec")

        assert len(callbacks) == 2
        assert callbacks[0] == (TaskState.INIT, TaskState.NAVIGATE)
        assert callbacks[1] == (TaskState.NAVIGATE, TaskState.EXECUTE)

    def test_multiple_callbacks(self):
        sm = GameStateMachine()
        call_count = [0]

        def cb1(old, new):
            call_count[0] += 1

        def cb2(old, new):
            call_count[0] += 1

        sm.add_transition_callback(cb1)
        sm.add_transition_callback(cb2)
        sm.navigate("nav")

        assert call_count[0] == 2

    def test_callback_error_does_not_crash(self):
        sm = GameStateMachine()

        def bad_callback(old, new):
            raise RuntimeError("callback error")

        sm.add_transition_callback(bad_callback)
        result = sm.navigate("nav")
        assert result is True

    def test_elapsed_time(self):
        sm = GameStateMachine()
        time.sleep(0.1)
        elapsed = sm.elapsed_time
        assert elapsed >= 0.1

    def test_execution_summary(self):
        sm = GameStateMachine(max_retries=3)
        sm.navigate("start")
        sm.execute("run")
        sm.verify("check")
        sm.complete("done")

        summary = sm.get_execution_summary()
        assert summary["final_state"] == "done"
        assert summary["final_status"] == "success"
        assert summary["total_retries"] == 0
        assert summary["total_transitions"] == 4
        assert summary["success"] is True
        assert summary["elapsed_time"] >= 0

    def test_fail_summary(self):
        sm = GameStateMachine(max_retries=1)
        sm.navigate("start")
        sm.retry("retry 1")
        sm.retry("retry 2 - max exceeded")

        summary = sm.get_execution_summary()
        assert summary["final_state"] == "fail"
        assert summary["success"] is False

    def test_can_transition_to(self):
        sm = GameStateMachine()
        assert sm.can_transition_to(TaskState.NAVIGATE) is True
        assert sm.can_transition_to(TaskState.EXECUTE) is False
        assert sm.can_transition_to(TaskState.DONE) is False

        sm.navigate("nav")
        assert sm.can_transition_to(TaskState.EXECUTE) is True
        assert sm.can_transition_to(TaskState.VERIFY) is False

    def test_checkpoint_save_and_recover(self):
        sm = GameStateMachine(checkpoint_interval=1)
        sm.navigate("nav")
        sm.execute("exec")
        
        checkpoint = sm.get_last_checkpoint()
        assert checkpoint is not None
        assert checkpoint["state"] == "execute"

        sm.verify("check")
        sm.complete("done")
        
        recovered = sm.recover_from_checkpoint()
        assert recovered is True
        assert sm.state == TaskState.VERIFY

    def test_recover_without_checkpoint(self):
        sm = GameStateMachine()
        result = sm.recover_from_checkpoint()
        assert result is False

    def test_recover_with_single_checkpoint(self):
        sm = GameStateMachine(checkpoint_interval=1)
        sm.navigate("nav")
        result = sm.recover_from_checkpoint()
        assert result is False

    def test_previous_state_tracking(self):
        sm = GameStateMachine()
        sm.navigate("nav")
        sm.execute("exec")
        sm.verify("check")

        assert sm._previous_state == TaskState.EXECUTE


class TestStateTransition:
    """Tests for StateTransition dataclass."""

    def test_creation(self):
        transition = StateTransition(
            from_state=TaskState.INIT,
            to_state=TaskState.NAVIGATE,
            reason="test",
        )
        assert transition.from_state == TaskState.INIT
        assert transition.to_state == TaskState.NAVIGATE
        assert transition.reason == "test"
        assert transition.timestamp > 0

    def test_default_timestamp(self):
        before = time.time()
        transition = StateTransition(
            from_state=TaskState.NAVIGATE,
            to_state=TaskState.EXECUTE,
        )
        after = time.time()
        assert before <= transition.timestamp <= after


class TestValidTransitions:
    """Tests for VALID_TRANSITIONS mapping."""

    def test_all_states_in_map(self):
        for state in TaskState:
            assert state in VALID_TRANSITIONS

    def test_terminal_states_have_no_transitions(self):
        for state in TERMINAL_STATES:
            assert VALID_TRANSITIONS[state] == []

    def test_init_valid_next_states(self):
        assert TaskState.NAVIGATE in VALID_TRANSITIONS[TaskState.INIT]
        assert TaskState.FAIL in VALID_TRANSITIONS[TaskState.INIT]

    def test_navigate_valid_next_states(self):
        valid = VALID_TRANSITIONS[TaskState.NAVIGATE]
        assert TaskState.EXECUTE in valid
        assert TaskState.RETRY in valid
        assert TaskState.FAIL in valid

    def test_retry_valid_next_states(self):
        valid = VALID_TRANSITIONS[TaskState.RETRY]
        assert TaskState.NAVIGATE in valid
        assert TaskState.EXECUTE in valid
        assert TaskState.FAIL in valid
