import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.agent import Agent
from terra.agent import AgentState
from terra.benchmark_direct_service import _candidate_rows
from terra.benchmark_direct_service import _classify_complete_dump
from terra.benchmark_direct_service import _dig_prefilter_batch
from terra.benchmark_direct_service import _iter_chunks
from terra.benchmark_direct_service import _pad_rows
from terra.benchmark_direct_service import _rotate_cabin_steps
from terra.benchmark_direct_service import _service_batch
from terra.benchmark_direct_service import _state_at_base_pose
from terra.benchmark_direct_service import _target_dig_progress_batch
from terra.benchmark_direct_service import _target_progress
from terra.benchmark_direct_service import _transition_state
from terra.benchmark_direct_service import compute_initial_direct_service
from terra.config import EnvConfig
from terra.env import TerraEnv


def _reference_service_for_candidate(state, candidate):
    """The pre-optimization loop, including wraps after every transition."""
    posed = _state_at_base_pose(state, candidate[:3])
    posed = _rotate_cabin_steps(posed, candidate[3])
    posed = TerraEnv.wrap_state(posed, update_reachability=jnp.bool_(False))

    do_action = TrackedAction.do()
    cabin_action = TrackedAction.cabin_anticlock()
    dug = _transition_state(posed, do_action)
    progress = _target_progress(posed, dug)
    dig_valid = jnp.logical_and(
        dug.agent.agent_states[0].loaded[0] > 0,
        jnp.sum(progress) > 0,
    )

    def try_dump_heading(_, carry):
        rotated, counts = carry
        dumped = _transition_state(rotated, do_action)
        counts = counts + jnp.asarray(
            _classify_complete_dump(rotated, dumped),
            dtype=jnp.int32,
        )
        return _transition_state(rotated, cabin_action), counts

    _, counts = jax.lax.fori_loop(
        0,
        state.env_cfg.agent.angles_cabin,
        try_dump_heading,
        (dug, jnp.zeros((4,), dtype=jnp.int32)),
    )
    workspace_progress = jnp.where(dig_valid, progress, jnp.zeros_like(progress))
    direct_progress = jnp.where(
        jnp.logical_and(dig_valid, counts[0] > 0),
        progress,
        jnp.zeros_like(progress),
    )
    diagnostics = jnp.concatenate(
        (
            jnp.asarray((dig_valid,), dtype=jnp.int32),
            counts,
        )
    )
    return workspace_progress, direct_progress, diagnostics


@jax.jit
def _reference_service_batch(state, candidates):
    return jax.vmap(
        _reference_service_for_candidate,
        in_axes=(None, 0),
    )(state, candidates)


class BenchmarkDirectServiceTest(unittest.TestCase):
    SHAPE = (18, 18)
    CENTER = (9, 9)

    @classmethod
    def _env_config(cls) -> EnvConfig:
        base = EnvConfig()
        return base._replace(
            tile_size=np.float32(36.5714285714 / 64),
            agent=base.agent._replace(width=7, height=11),
            maps=base.maps._replace(edge_length_px=cls.SHAPE[0]),
            agent_types=(0,),
            action_types=(0,),
            max_steps_in_episode=450,
            enable_reachability_obs=False,
        )

    @classmethod
    def _agent(cls) -> Agent:
        zero = AgentState(
            pos_base=jnp.zeros((2,), dtype=jnp.int16),
            angle_base=jnp.zeros((1,), dtype=jnp.int8),
            angle_cabin=jnp.zeros((1,), dtype=jnp.int8),
            wheel_angle=jnp.zeros((1,), dtype=jnp.int8),
            loaded=jnp.zeros((1,), dtype=jnp.int8),
            agent_type=jnp.zeros((1,), dtype=jnp.int8),
            action_type=jnp.zeros((1,), dtype=jnp.int8),
            shovel_lifted=jnp.zeros((1,), dtype=jnp.int8),
            carry_baseline_potential=jnp.float32(0.0),
            carry_potential_after_lift=jnp.float32(0.0),
        )
        active = zero._replace(pos_base=jnp.asarray(cls.CENTER, dtype=jnp.int16))
        return Agent(
            width=jnp.asarray(7, dtype=jnp.int32),
            height=jnp.asarray(11, dtype=jnp.int32),
            moving_dumped_dirt=jnp.bool_(False),
            agent_states=(active, zero, zero, zero),
            agent_active=jnp.asarray([1, 0, 0, 0], dtype=jnp.int8),
            num_agents=jnp.int32(1),
            current_agent=jnp.int32(0),
        )

    @classmethod
    def _state(cls, *, nearby_dump: bool):
        target = np.zeros(cls.SHAPE, dtype=np.int8)
        target[8:11, 16:18] = -1
        if nearby_dump:
            target[8:11, 0:2] = 1
        else:
            target[14:16, 8:11] = 1

        env = TerraEnv.new(maps_size_px=cls.SHAPE[0])
        timestep = env.reset(
            jax.random.PRNGKey(7),
            jnp.asarray(target),
            jnp.zeros(cls.SHAPE, dtype=jnp.int8),
            -97.0 * jnp.ones((3, 3), dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            -97.0 * jnp.ones((64, 3), dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.ones(cls.SHAPE, dtype=jnp.bool_),
            jnp.zeros(cls.SHAPE, dtype=jnp.int8),
            jnp.ones(cls.SHAPE, dtype=jnp.float32),
            cls._env_config(),
            cls._agent(),
        )
        return env, timestep.state

    def test_exact_service_unions_overlapping_digs_without_double_counting(self):
        _, state = self._state(nearby_dump=True)
        result = compute_initial_direct_service(state)

        self.assertEqual(result["required_volume"], 6)
        self.assertEqual(result["admissible_pose_count_initial"], 12)
        self.assertEqual(
            result["base_pose_cabin_heading_candidates_initial"],
            12 * state.env_cfg.agent.angles_cabin,
        )
        self.assertEqual(
            result["dump_do_attempts_logical"],
            result["service_dig_candidate_attempts_logical"]
            * state.env_cfg.agent.angles_cabin,
        )
        self.assertEqual(
            result["dump_do_attempts_logical"],
            result["legal_complete_dump_attempts"]
            + result["wrong_complete_dump_attempts"]
            + result["rejected_dump_attempts"],
        )
        self.assertEqual(result["workspace_serviceable_volume_initial"], 6)
        self.assertEqual(result["direct_serviceable_volume_initial"], 6)
        self.assertEqual(result["initial_workspace_coverage"], 1.0)
        self.assertEqual(result["direct_service_coverage_initial"], 1.0)
        self.assertTrue(result["any_direct_transfer_pose_exists_initial"])
        self.assertGreater(result["legal_complete_dump_attempts"], 0)
        self.assertGreater(result["wrong_complete_dump_attempts"], 0)

    def test_complete_off_zone_dump_is_diagnostic_not_direct_service(self):
        _, state = self._state(nearby_dump=False)
        result = compute_initial_direct_service(state)

        self.assertEqual(result["workspace_serviceable_volume_initial"], 6)
        self.assertEqual(result["direct_serviceable_volume_initial"], 0)
        self.assertEqual(result["initial_workspace_coverage"], 1.0)
        self.assertEqual(result["direct_service_coverage_initial"], 0.0)
        self.assertFalse(result["any_direct_transfer_pose_exists_initial"])
        self.assertEqual(result["legal_complete_dump_attempts"], 0)
        self.assertGreater(result["wrong_complete_dump_attempts"], 0)

    def test_prefilter_rejections_have_zero_real_target_progress(self):
        _, state = self._state(nearby_dump=True)
        base_orientations = int(state.env_cfg.agent.angles_base)
        cabin_orientations = int(state.env_cfg.agent.angles_cabin)
        poses = np.column_stack(
            (
                np.full(base_orientations, self.CENTER[0], dtype=np.int32),
                np.full(base_orientations, self.CENTER[1], dtype=np.int32),
                np.arange(base_orientations, dtype=np.int32),
            )
        )
        candidates = _candidate_rows(poses, cabin_orientations)

        rejected_count = 0
        accepted_count = 0
        for rows in _iter_chunks(candidates, 128):
            padded, valid_count = _pad_rows(rows, 128)
            padded = jnp.asarray(padded, dtype=jnp.int32)
            accepted = np.asarray(jax.device_get(_dig_prefilter_batch(state, padded)))[
                :valid_count
            ]
            actual_progress = np.asarray(
                jax.device_get(_target_dig_progress_batch(state, padded))
            )[:valid_count]
            rejected_count += int((~accepted).sum())
            accepted_count += int(accepted.sum())
            np.testing.assert_array_equal(
                actual_progress[~accepted],
                np.zeros_like(actual_progress[~accepted]),
            )

        self.assertEqual(len(candidates), base_orientations * cabin_orientations)
        self.assertGreater(rejected_count, 0)
        self.assertGreater(accepted_count, 0)

    def test_raw_counterfactual_steps_match_full_wrap_batch(self):
        candidates = jnp.asarray(
            (
                (9, 9, 0, 0),
                (9, 9, 0, 1),
                (9, 9, 0, 11),
                (9, 9, 0, 6),
            ),
            dtype=jnp.int32,
        )
        for nearby_dump in (True, False):
            with self.subTest(nearby_dump=nearby_dump):
                _, state = self._state(nearby_dump=nearby_dump)
                optimized = _service_batch(state, candidates)
                reference = _reference_service_batch(state, candidates)
                for optimized_value, reference_value in zip(
                    optimized,
                    reference,
                ):
                    np.testing.assert_array_equal(
                        np.asarray(jax.device_get(optimized_value)),
                        np.asarray(jax.device_get(reference_value)),
                    )

                diagnostics = np.asarray(jax.device_get(optimized[2]))
                np.testing.assert_array_equal(
                    diagnostics[:3, 0],
                    np.ones((3,), dtype=np.int32),
                )
                np.testing.assert_array_equal(
                    diagnostics[3],
                    np.asarray((0, 0, 0, 12, 0), dtype=np.int32),
                )
                self.assertGreater(int(diagnostics[:3, 2].sum()), 0)
                if nearby_dump:
                    self.assertGreater(int(diagnostics[:3, 1].sum()), 0)
                else:
                    self.assertEqual(int(diagnostics[:3, 1].sum()), 0)

    def test_transition_helper_matches_step_no_reset_state(self):
        env, state = self._state(nearby_dump=True)
        actions = (
            ("dig", TrackedAction.do()),
            ("forward", TrackedAction.forward()),
            ("base_clock", TrackedAction.clock()),
            ("cabin_anticlock", TrackedAction.cabin_anticlock()),
        )
        for action_name, tracked_action in actions:
            with self.subTest(action=action_name), jax.disable_jit():
                direct = _transition_state(state, tracked_action)
                reference = env.step_no_reset(
                    state,
                    tracked_action,
                    state.env_cfg,
                ).state
            direct_fields = (
                direct.agent,
                direct.world.action_map.map,
                direct.world.dumpability_mask.map,
                direct.world.last_dig_mask.map,
                direct.world.traversability_mask.map,
                direct.env_steps,
            )
            reference_fields = (
                reference.agent,
                reference.world.action_map.map,
                reference.world.dumpability_mask.map,
                reference.world.last_dig_mask.map,
                reference.world.traversability_mask.map,
                reference.env_steps,
            )
            for direct_leaf, reference_leaf in zip(
                jax.tree_util.tree_leaves(direct_fields),
                jax.tree_util.tree_leaves(reference_fields),
            ):
                np.testing.assert_array_equal(
                    np.asarray(direct_leaf),
                    np.asarray(reference_leaf),
                )

    def test_wrong_complete_dump_classifier_never_reports_legal(self):
        _, state = self._state(nearby_dump=False)
        loaded = state.agent.agent_states[0]._replace(
            loaded=jnp.asarray([5], dtype=jnp.int8)
        )
        state = state._replace(
            agent=state.agent._replace(
                agent_states=(loaded,) + state.agent.agent_states[1:]
            )
        )
        dumped = state._handle_dump()
        legal, wrong, rejected, violation = _classify_complete_dump(
            state,
            dumped,
        )

        self.assertFalse(bool(legal))
        self.assertTrue(bool(wrong))
        self.assertFalse(bool(rejected))
        self.assertFalse(bool(violation))


if __name__ == "__main__":
    unittest.main()
