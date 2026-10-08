"""Session lifecycle checks; native cut/unload parity is covered by verify_http.py.

These tests use the saved diagnostic State but stub the expensive transition.
They exercise terminal freezing, transactional history and full-state export,
without compiling another interactive simulator.
"""

import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import jax
import numpy as np

from terra.viewer3d.pull_session import PullSession
from terra.viewer3d.replay import ReplayRecorder, load_replay
from terra.viewer3d.snapshots import snapshot_from_timestep


INITIAL_STATES = Path(os.environ.get("TERRA_PULL_TEST_INITIAL_STATES", "initial_states.pkl"))


@unittest.skipUnless(INITIAL_STATES.is_file(), "set TERRA_PULL_TEST_INITIAL_STATES to the saved diagnostic state bank")
class PullSessionLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with INITIAL_STATES.open("rb") as stream:
            saved = pickle.load(stream)
        index = next(i for i, case in enumerate(saved["episodes"])
                     if case["source_slot"] == 17411 and case["mode"] == "precision"
                     and case["start"] == 0 and case["decoder"] == "greedy")
        cls.initial = saved["initial"][index]

    def session(self, step=0):
        session = PullSession.__new__(PullSession)
        session.states = [self.initial._replace(env_steps=np.asarray(step, np.int32))]
        session.actions = []
        session.exploring = False
        session.selected_case = dict(case_id="17411", precision=True, start=0)
        session.cases = [dict(id="17411", modes=["bulk", "precision"], starts=[0, 1])]
        session.recorder = ReplayRecorder(metadata=dict(selected_case=session.selected_case))

        def frame(*, reward, action, effect, message):
            value = snapshot_from_timestep(SimpleNamespace(
                state=session.state, env_cfg=session.state.env_cfg, reward=float(reward),
                done=int(session.state.env_steps) >= 450, info={"task_done": False}),
                action=action, actor_id=0 if action is not None else None)
            value["diagnostics"] = dict(exploring=session.exploring, loaded=0,
                                        current_workspace_obstacle=False, current_dig_admitted=False,
                                        action_had_effect=effect, message=message)
            return value

        def advance(state, action):
            return (state._replace(env_steps=np.asarray(int(state.env_steps) + 1, np.int32)),
                    np.float32(0), np.bool_(False), np.bool_(False),
                    dict(action_had_effect=False, target_mutation=False,
                         obstacle_mutation=False, transition_mass_residual=0))

        session._advance = advance
        session._frame = frame
        session.recorder.frames.append(frame(reward=0, action=None, effect=None, message="fixture"))
        return session

    def assert_state_equal(self, actual, expected):
        self.assertEqual(jax.tree_util.tree_structure(actual), jax.tree_util.tree_structure(expected))
        for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(left, right)

    def test_450_freezes_until_explicit_exploration_and_undo_restores_freeze(self):
        session = self.session(step=449)
        at_449 = session.state
        session.step(7)
        at_450 = session.state
        self.assertTrue(session.recorder.frames[-1]["done"])
        with self.assertRaisesRegex(RuntimeError, "Episode ended"):
            session.step(7)
        self.assertIs(session.state, at_450)
        self.assertEqual(session.actions, [7])
        session.continue_exploring()
        session.step(7)
        self.assertEqual(int(session.state.env_steps), 451)
        self.assertEqual(int(session.state.env_cfg.max_steps_in_episode), 450)
        self.assertTrue(session.recorder.frames[-1]["diagnostics"]["exploring"])
        session.undo()
        self.assertFalse(session.exploring)
        self.assert_state_equal(session.state, at_450)
        with self.assertRaisesRegex(RuntimeError, "Episode ended"):
            session.step(7)
        session.undo()
        self.assert_state_equal(session.state, at_449)
        self.assertFalse(session.payload()["can_undo"])

    def test_failed_frame_does_not_partially_commit_state_or_action(self):
        session = self.session()
        initial = session.state
        frames = list(session.recorder.frames)

        def broken_frame(**kwargs):
            raise RuntimeError("diagnostic failure")

        session._frame = broken_frame
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            session.step(7)
        self.assertIs(session.state, initial)
        self.assertEqual(session.actions, [])
        self.assertEqual(session.recorder.frames, frames)

    def test_invalid_native_transition_is_not_committed(self):
        for field, value in [("target_mutation", True), ("obstacle_mutation", True),
                             ("transition_mass_residual", 1)]:
            with self.subTest(field=field):
                session = self.session()
                initial = session.state
                advance = session._advance

                def bad_advance(state, action):
                    result = list(advance(state, action))
                    result[-1][field] = value
                    return tuple(result)

                session._advance = bad_advance
                with self.assertRaisesRegex(RuntimeError, "integrity failed"):
                    session.step(7)
                self.assertIs(session.state, initial)
                self.assertEqual(session.actions, [])
                self.assertEqual(len(session.recorder.frames), 1)

    def test_export_preserves_exact_native_history_and_case(self):
        session = self.session()
        session.step(7)
        with TemporaryDirectory() as folder:
            session.output = Path(folder)
            exported = session.export()
            replay = load_replay(exported["paths"]["replay"])
            with open(exported["paths"]["native_states"], "rb") as stream:
                native = pickle.load(stream)
            actions = json.loads(Path(exported["paths"]["actions"]).read_text())
            self.assertEqual(native["actions"], [7])
            self.assertEqual(actions["actions"], [7])
            for payload in (native, actions, replay):
                self.assertEqual(payload["metadata"]["selected_case"], session.selected_case)
            self.assertEqual(replay, exported["replay"])
            self.assertEqual(len(native["states"]), 2)
            for actual, expected in zip(native["states"], session.states):
                self.assert_state_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
