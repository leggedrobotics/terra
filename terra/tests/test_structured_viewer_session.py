"""Structured history, API arguments and time-budget lifecycle checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from terra.structured_actions import StructuredClock, StructuredTermination, StructuredTimeConfig
from terra.structured_episode import (
    StructuredEpisodeConfig, structured_episode_limits, structured_episode_termination,
)
from terra.tests.test_pull_viewer_session import PullSessionLifecycleTests
from terra.viewer3d.pull_session import PullSession
from terra.viewer3d.structured_session import StructuredPullSession, load_structured_protocol


class StructuredProtocolTests(unittest.TestCase):
    def protocol(self):
        return dict(StructuredEpisodeConfig()._asdict(), timing=StructuredTimeConfig()._asdict())

    def load(self, values):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            path.write_text(json.dumps(values))
            return load_structured_protocol(path)

    def test_flat_and_training_config_have_same_explicit_protocol(self):
        flat = self.protocol()
        flat["no_time_limit"] = not flat.pop("time_limit")
        flat["time_budget_factor"] = 3
        flat["decisions_per_dig_unit"] = 1.2
        nested = dict(training=dict(flat, learning_rate=.0003))
        self.assertEqual(self.load(flat), self.load(nested))
        self.assertEqual(self.load(flat)[1].time_budget_factor, 3)

    def test_missing_old_trainer_dead_end_option_keeps_legacy_behavior(self):
        values = self.protocol()
        for name in ("terminate_loaded_dead_ends", "terminate_unloaded_dead_ends",
                     "precision_safety", "precision_edge_weight"):
            del values[name]
        episode = self.load(values)[1]
        self.assertFalse(episode.terminate_loaded_dead_ends)
        self.assertFalse(episode.terminate_unloaded_dead_ends)
        self.assertFalse(episode.precision_safety)
        self.assertEqual(episode.precision_edge_weight, -1.)

    def test_invalid_protocol_values_are_rejected(self):
        for name, value in [("decision_limit", 1.5), ("decision_limit", True),
                            ("time_budget_s", 0), ("time_budget_factor", float("inf")),
                            ("decisions_per_dig_unit", -1), ("time_limit", "false"),
                            ("success_time_bonus", float("nan")), ("precision_safety", 1),
                            ("terminate_unloaded_dead_ends", "false"),
                            ("precision_edge_weight", -.5), ("precision_edge_weight", 1.1),
                            ("precision_edge_weight", True), ("precision_edge_weight", float("nan"))]:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.load(dict(self.protocol(), **{name: value}))
        for name, value in [("nav_speed_mps", 0), ("setup_s", -1), ("setup_per_dig", 1),
                            ("dig_s_per_m3", True)]:
            values = self.protocol()
            values["timing"][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.load(values)
        values = self.protocol()
        del values["success_time_bonus"]
        with self.assertRaisesRegex(ValueError, "missing episode"):
            self.load(values)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.load(dict(self.protocol(), no_time_limit=True))


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
        session.timing = StructuredTimeConfig()
        session.episode_config = StructuredEpisodeConfig(time_budget_s=10.0)
        session.structured_protocol = dict(episode=session.episode_config._asdict(), timing=session.timing._asdict())
        session.recorder.metadata["structured_protocol"] = session.structured_protocol
        session._structured_limits = lambda state: (10.0, 450)
        session.elapsed_times = [0.0]
        session.clocks = [StructuredClock()]
        session._initial_clock = StructuredClock()
        session.initials = {("17411", True, 0): (None, self.initial)}
        session._structured_masks = lambda state: dict(
            move_mask=np.ones((2, 5), bool), turn_mask=np.ones((2, 6), bool),
            do_mask=np.ones(12, bool), action_mask=np.ones(8, bool))
        session._structured_termination = lambda state, elapsed, do_mask: StructuredTermination(
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

    def test_time_reference_never_freezes_an_unlimited_episode(self):
        session = self.session()
        session.episode_config = session.episode_config._replace(time_limit=False)
        session._structured_termination = lambda state, elapsed, do_mask: structured_episode_termination(
            state, elapsed, timing=session.timing, config=session.episode_config)
        session.step(0, amount=2)
        beyond = session.step(2, amount=3)
        self.assertGreater(beyond["diagnostics"]["elapsed_time_s"], session.time_budget_s)
        self.assertFalse(beyond["done"])
        self.assertIsNone(beyond["diagnostics"]["termination_reason"])
        self.assertFalse(beyond["diagnostics"]["time_limit"])
        self.assertFalse(session.step(7)["done"])

    def test_success_bonus_is_applied_once_and_undo_restores_it(self):
        session = self.session()
        session._structured_termination = lambda state, elapsed, do_mask: StructuredTermination(
            int(state.env_steps) >= 2, int(state.env_steps) >= 2, False,
            8.5 if int(state.env_steps) >= 2 else 0.0)
        session.step(7)
        terminal = session.step(7)
        self.assertAlmostEqual(terminal["reward"], 8.3)
        self.assertEqual(terminal["diagnostics"]["terminal_reward"], 8.5)
        self.assertAlmostEqual(session.step(7, continue_after_timeout=True)["reward"], -.2)
        session.undo()
        session.undo()
        self.assertAlmostEqual(session.step(7)["reward"], 8.3)

    def test_exploration_cannot_pay_a_second_terminal_bonus_after_reopening_task(self):
        session = self.session()
        session._structured_termination = lambda state, elapsed, do_mask: StructuredTermination(
            int(state.env_steps) in (1, 3), int(state.env_steps) in (1, 3), False,
            8.5 if int(state.env_steps) in (1, 3) else 0.0)
        self.assertAlmostEqual(session.step(7)["reward"], 8.3)
        self.assertFalse(session.step(7, continue_after_timeout=True)["done"])
        again = session.step(7)
        self.assertTrue(again["done"])
        self.assertAlmostEqual(again["reward"], -.2)

    def test_reset_recomputes_map_limits_and_exports_previous_protocol(self):
        from terra.viewer3d.replay import ReplayRecorder
        session = self.session()
        config = StructuredEpisodeConfig(time_budget_s=3600, time_budget_factor=3,
            time_budget_offset_s=7960, decision_limit=300, decisions_per_dig_unit=1.2,
            time_limit=False, success_time_bonus=2, success_decision_bonus=1, precision_safety=True)
        session.episode_config = config
        session.structured_protocol = dict(episode=config._asdict(), timing=session.timing._asdict())
        session._structured_limits = lambda state: structured_episode_limits(state, timing=session.timing, config=config)
        second = self.initial._replace(world=self.initial.world._replace(
            target_map=self.initial.world.target_map._replace(map=self.initial.world.target_map.map * 2)))
        session.initials[("second", True, 0)] = (None, second)
        exported = []
        def reset_parent(_self, **kwargs):
            key = (kwargs["case_id"], True, 0)
            _self.states, _self.actions = [_self.initials[key][1]], []
            _self.selected_case = dict(case_id=key[0], precision=True, start=0)
            _self.recorder = ReplayRecorder(metadata={})
            _self.recorder.frames.append(_self._frame(reward=0, action=None, effect=None, message="reset"))
        with patch.object(PullSession, "reset", reset_parent), patch.object(session, "export", side_effect=lambda: exported.append(session.time_budget_s)):
            session.reset(case_id="17411")
            original_budget, original_limit = session.time_budget_s, session.decision_budget
            session.step(7)
            session.reset(case_id="second")
        self.assertEqual(exported, [original_budget])
        self.assertGreater(session.time_budget_s, original_budget)
        self.assertEqual(session.decision_budget, 2 * original_limit)
        self.assertEqual(session.elapsed_times, [0.0])
        self.assertEqual(session.recorder.metadata["structured_protocol"], session.structured_protocol)
        self.assertTrue(bool(session.state.env_cfg.precision_safety))

    def test_unloaded_certificate_has_separate_reason_and_pays_failure_once(self):
        session = self.session()
        session.episode_config = session.episode_config._replace(
            time_limit=False, terminate_unloaded_dead_ends=True)
        session._structured_termination = lambda state, elapsed, do_mask: structured_episode_termination(
            state, elapsed, timing=session.timing, config=session.episode_config,
            unloaded_dead_end=bool(state.env_steps >= 1))
        terminal = session.step(7)
        self.assertEqual(terminal["diagnostics"]["termination_reason"], "unloaded_dead_end")
        self.assertTrue(terminal["diagnostics"]["unloaded_dead_end"])
        self.assertFalse(terminal["diagnostics"]["loaded_dead_end"])
        self.assertAlmostEqual(terminal["reward"], -1.2)
        with self.assertRaisesRegex(RuntimeError, "Episode ended"):
            session.step(7)
        self.assertAlmostEqual(session.step(7, continue_after_timeout=True)["reward"], -.2)
        session.undo()
        session.undo()
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

    def test_loaded_dead_end_freezes_once_and_undo_recovers(self):
        session = self.session()
        advance = session._structured_advance
        mask = session._structured_masks
        def dig(state, action, clock):
            result = advance(state, action, clock)
            if int(action.action) == 6:
                cur = result.state._get_current_agent_state()
                result.state = result.state._set_current_agent_state(cur._replace(
                    loaded=np.asarray([3], cur.loaded.dtype)))
            return result
        def masks(state):
            result = mask(state)
            if int(state._get_current_agent_state().loaded[0]) > 0:
                result['do_mask'] = np.zeros(12, bool)
            return result
        def termination(state, elapsed, do_mask):
            dead = int(state._get_current_agent_state().loaded[0]) > 0 and not np.any(do_mask)
            return StructuredTermination(dead, False, False, -1.0 if dead else 0.0, dead)
        session._structured_advance = dig
        session._structured_masks = masks
        session._structured_termination = termination
        terminal = session.step(6, heading=0)
        self.assertTrue(terminal['done'])
        self.assertFalse(terminal['task_done'])
        self.assertEqual(terminal['diagnostics']['termination_reason'], 'loaded_no_unload')
        self.assertTrue(terminal['diagnostics']['loaded_dead_end'])
        self.assertAlmostEqual(terminal['reward'], -1.2)
        with self.assertRaisesRegex(RuntimeError, 'Episode ended'):
            session.step(7)
        beyond = session.step(7, continue_after_timeout=True)
        self.assertAlmostEqual(beyond['reward'], -.2)
        session.undo()
        self.assertTrue(session.recorder.frames[-1]['done'])
        session.undo()
        self.assertEqual(int(session.state._get_current_agent_state().loaded[0]), 0)
        self.assertFalse(session.recorder.frames[-1]['done'])
        self.assertFalse(session.exploring)

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
            self.assertEqual(clock["structured_protocol"], session.structured_protocol)

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
        import gzip
        import pickle
        session = self.session()
        session.step(0, amount=3)
        with TemporaryDirectory() as output:
            from pathlib import Path
            session.output = Path(output)
            exported = session.export()
            expected_state = session.state
            expected_actions = list(session.actions)
            folder = Path(exported["paths"]["native_states"]).parent
            # A historical protocol predates all three precision options.
            # Normalizing missing keys must preserve legacy semantics.
            def legacy_protocol(protocol):
                for name in ("terminate_unloaded_dead_ends", "precision_safety", "precision_edge_weight"):
                    del protocol["episode"][name]
            with (folder / "states.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            legacy_protocol(saved["metadata"]["structured_protocol"])
            with (folder / "states.pkl").open("wb") as stream:
                pickle.dump(saved, stream)
            with gzip.open(folder / "replay.json.gz", "rt") as stream:
                replay = json.load(stream)
            legacy_protocol(replay["metadata"]["structured_protocol"])
            with gzip.open(folder / "replay.json.gz", "wt") as stream:
                json.dump(replay, stream)
            clock = json.loads((folder / "structured_clock.json").read_text())
            legacy_protocol(clock["structured_protocol"])
            (folder / "structured_clock.json").write_text(json.dumps(clock))
            with patch.object(session, "reset", return_value=None):
                session.restore_recording(folder)
            self.assert_state_equal(session.state, expected_state)
            self.assertEqual(session.actions, expected_actions)
            self.assertEqual(session.elapsed_times, [0.0, 6.0])
            self.assertTrue(session.clocks[-1].visit_open)

    def test_restore_rejects_protocol_mismatch_before_touching_history(self):
        session = self.session()
        session.step(0, amount=3)
        with TemporaryDirectory() as output:
            session.output = Path(output)
            exported = session.export()
            before = session.state
            session.structured_protocol = dict(session.structured_protocol,
                timing=dict(session.structured_protocol["timing"], setup_s=123.0))
            with self.assertRaisesRegex(ValueError, "preserve the exported structured protocol"):
                session.restore_recording(Path(exported["paths"]["native_states"]).parent)
            self.assertIs(session.state, before)

    def test_old_export_only_restores_under_legacy_settings_and_reexports_protocol(self):
        import gzip
        import pickle
        session = self.session()
        session.step(0, amount=3)
        with TemporaryDirectory() as output:
            session.output = Path(output)
            exported = session.export()
            folder = Path(exported["paths"]["native_states"]).parent
            with (folder / "states.pkl").open("rb") as stream:
                native = pickle.load(stream)
            native["metadata"].pop("structured_protocol")
            with (folder / "states.pkl").open("wb") as stream:
                pickle.dump(native, stream)
            with gzip.open(folder / "replay.json.gz", "rt") as stream:
                replay = json.load(stream)
            replay["metadata"].pop("structured_protocol")
            with gzip.open(folder / "replay.json.gz", "wt") as stream:
                json.dump(replay, stream)
            path = folder / "structured_clock.json"
            clock = json.loads(path.read_text())
            clock.pop("structured_protocol")
            path.write_text(json.dumps(clock))
            original = session.structured_protocol
            session.structured_protocol = dict(original, episode=dict(original["episode"], time_limit=False))
            with self.assertRaisesRegex(ValueError, "Old recordings require"):
                session.restore_recording(folder)
            session.structured_protocol = original
            with patch.object(session, "reset", return_value=None):
                session.restore_recording(folder)
            self.assertEqual(session.recorder.metadata["structured_protocol"], original)
            self.assertEqual(session.elapsed_times, [0.0, 6.0])
            second_export = session.export()
            with patch.object(session, "reset", return_value=None):
                session.restore_recording(Path(second_export["paths"]["native_states"]).parent)
            # Earlier recordings predate the dead-end diagnostic and must
            # retain the old option rather than acquiring terminal failures.
            for frame in replay["frames"]:
                frame["diagnostics"].pop("loaded_dead_end")
            with gzip.open(folder / "replay.json.gz", "wt") as stream:
                json.dump(replay, stream)
            with self.assertRaisesRegex(ValueError, "terminate_loaded_dead_ends=False"):
                session.restore_recording(folder)
            session.structured_protocol = dict(original,
                episode=dict(original["episode"], terminate_loaded_dead_ends=False))
            with patch.object(session, "reset", return_value=None):
                session.restore_recording(folder)
            self.assertFalse(session.recorder.metadata["structured_protocol"]["episode"]["terminate_loaded_dead_ends"])
