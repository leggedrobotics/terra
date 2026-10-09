"""Structured history, API arguments and time-budget lifecycle checks."""

import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from terra.structured_actions import StructuredClock, StructuredTermination
from terra.tests.test_pull_viewer_session import PullSessionLifecycleTests
from terra.viewer3d.pull_session import PullSession
from terra.viewer3d.structured_session import StructuredPullSession


class StructuredSessionLifecycleTests(PullSessionLifecycleTests):
    # Inherit the trusted saved-state fixture, not the legacy action semantics.
    test_450_freezes_until_explicit_exploration_and_undo_restores_freeze = None
    test_failed_frame_does_not_partially_commit_state_or_action = None
    test_invalid_native_transition_is_not_committed = None
    test_export_preserves_exact_native_history_and_case = None

    def session(self, step=0):
        legacy = super().session(step)
        session = StructuredPullSession.__new__(StructuredPullSession)
        session.__dict__.update(legacy.__dict__)
        session.recorder.metadata["action_mode"] = "structured_v1"
        del session._frame
        session._structured_ready = True
        session.time_budget_s = 10.0
        session.decision_budget = 450
        session.elapsed_times = [0.0]
        session.clocks = [StructuredClock()]
        session._initial_clock = StructuredClock()
        session.initials = {("17411", True, 0): (None, self.initial)}
        session._structured_masks = lambda state: dict(
            move_mask=np.ones((2, 5), bool), turn_mask=np.ones((2, 6), bool),
            do_mask=np.ones(12, bool), action_mask=np.ones(8, bool))
        session._structured_termination = lambda state, elapsed: StructuredTermination(
            elapsed >= 10, False, elapsed >= 10, -1.0 if elapsed >= 10 else 0.0)
        # Parent frame detaches the fixture without running native geometry.
        parent_frame = legacy._frame
        def frame(_self, **kwargs):
            legacy.states = session.states
            legacy.exploring = session.exploring
            return parent_frame(**kwargs)
        mock = patch.object(PullSession, "_frame", frame)
        mock.start()
        self.addCleanup(mock.stop)

        def advance(state, action, clock):
            session.last_request = action
            return SimpleNamespace(
                state=state._replace(env_steps=np.asarray(int(state.env_steps) + 1, np.int32)),
                reward=np.float32(-0.2), duration_s=np.float32(6.0),
                info=dict(action_had_effect=True, target_mutation=False, obstacle_mutation=False,
                          transition_mass_residual=0, time_visit_open=True, time_moved=False))
        session._structured_advance = advance
        session.recorder.frames = [session._frame(reward=0, action=None, effect=None, message="ready")]
        return session

    def test_time_budget_terminal_reward_once_and_undo_restores_clock(self):
        session = self.session()
        session.step(0, amount=2)
        terminal = session.step(2, amount=3)
        self.assertTrue(terminal["done"])
        self.assertEqual(terminal["diagnostics"]["termination_reason"], "time_budget")
        self.assertAlmostEqual(terminal["reward"], -1.2)
        with self.assertRaisesRegex(RuntimeError, "Episode ended"):
            session.step(7)
        beyond = session.step(7, continue_after_timeout=True)
        self.assertAlmostEqual(beyond["reward"], -0.2)
        session.undo()
        self.assertFalse(session.exploring)
        session.undo()
        self.assertEqual(session.elapsed_times, [0.0, 6.0])
        self.assertEqual(session.actions, [dict(action=0, amount=2, heading=-1)])
        self.assertFalse(session.recorder.frames[-1]["done"])

    def test_arguments_and_masks_reject_without_history_mutation(self):
        session = self.session()
        for kwargs in (dict(action=True), dict(action=0, amount=6), dict(action=2, amount=0),
                       dict(action=6, heading=12), dict(action=0, heading=1), dict(action=6, amount=2)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                session.step(**kwargs)
        session.recorder.frames[-1]["diagnostics"]["structured_actions"]["do_mask"][4] = False
        with self.assertRaisesRegex(ValueError, "no executable outcome"):
            session.step(6, heading=4)
        self.assertEqual(session.actions, [])
        self.assertEqual(session.elapsed_times, [0.0])
        session.step(6, heading=3)
        self.assertEqual(int(session.last_request.heading), 3)

    def test_frame_failure_rolls_back_elapsed_clock_and_exploration(self):
        session = self.session()
        initial = session.state
        with patch.object(session, "_frame", side_effect=RuntimeError("frame failed")):
            with self.assertRaisesRegex(RuntimeError, "frame failed"):
                session.step(0, amount=1)
        self.assertIs(session.state, initial)
        self.assertEqual(session.elapsed_times, [0.0])
        self.assertEqual(session.clocks, [StructuredClock()])
        self.assertEqual(session.actions, [])

    def test_export_contains_arguments_and_recoverable_clock(self):
        session = self.session()
        session.step(0, amount=3)
        with TemporaryDirectory() as output:
            from pathlib import Path
            session.output = Path(output)
            result = session.export()
            actions = json.loads(Path(result["paths"]["actions"]).read_text())
            clock = json.loads(Path(result["paths"]["structured_clock"]).read_text())
            self.assertEqual(actions["actions"], [dict(action=0, amount=3, heading=-1)])
            self.assertEqual(clock["elapsed_time_s"], [0.0, 6.0])
            self.assertEqual(len(clock["clocks"]), len(session.states))

    def test_invalid_reset_preserves_current_game_and_clock(self):
        session = self.session()
        session.step(0, amount=2)
        before = session.__dict__.copy()
        for invalid in (dict(case_id="missing"), dict(precision="yes"), dict(start=True)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                session.reset(**invalid)
            for key, value in before.items():
                self.assertIs(session.__dict__[key], value, key)

    def test_native_recording_restore_preserves_actions_state_and_clock(self):
        session = self.session()
        session.step(0, amount=3)
        with TemporaryDirectory() as output:
            from pathlib import Path
            session.output = Path(output)
            exported = session.export()
            expected_state = session.state
            expected_actions = list(session.actions)
            with patch.object(session, "reset", return_value=None):
                session.restore_recording(Path(exported["paths"]["native_states"]).parent)
            self.assert_state_equal(session.state, expected_state)
            self.assertEqual(session.actions, expected_actions)
            self.assertEqual(session.elapsed_times, [0.0, 6.0])
            self.assertTrue(session.clocks[-1].visit_open)
