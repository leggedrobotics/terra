import pickle
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.config import RewardStage
from terra.config import REWARD_V2_TIMING_BASELINE
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.state import CORRECTED_DENSE_CONTRACT
from terra.state import State


class ExactDumpContractTest(unittest.TestCase):
    SHAPE = (64, 64)

    @staticmethod
    def _env_config(
        *,
        enforce_edge: bool = False,
        reward_stage: int = RewardStage.DENSE_SKILL,
    ) -> EnvConfig:
        batch_env = object.__new__(TerraEnvBatch)
        batch_env.batch_cfg = BatchConfig()._replace(
            maps_dims=MapsDimsConfig(maps_edge_length=64)
        )
        base = EnvConfig()
        batched = base._replace(
            agent=base.agent._replace(
                dig_depth=jnp.ones((1,), dtype=jnp.int32),
            )
        )
        updated = batch_env.update_env_cfgs(batched)
        return base._replace(
            tile_size=float(np.asarray(updated.tile_size)[0]),
            agent=base.agent._replace(
                width=int(np.asarray(updated.agent.width)[0]),
                height=int(np.asarray(updated.agent.height)[0]),
            ),
            maps=base.maps._replace(
                edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0]),
            ),
            agent_types=(0,),
            action_types=(0,),
            reward_stage=reward_stage,
            max_steps_in_episode=450,
            enforce_foundation_border_alignment=enforce_edge,
            foundation_dump_min_free_fraction=0.0,
        )

    @classmethod
    def _state(
        cls,
        target: np.ndarray,
        *,
        action: np.ndarray | None = None,
        padding: np.ndarray | None = None,
        dumpability: np.ndarray | None = None,
        loaded: int = 0,
        enforce_edge: bool = False,
        reward_stage: int = RewardStage.DENSE_SKILL,
        env_steps: int = 0,
        productive_workspace_cycles: int = 0,
    ) -> State:
        if action is None:
            action = np.zeros(cls.SHAPE, dtype=np.int8)
        if padding is None:
            padding = np.zeros(cls.SHAPE, dtype=np.int8)
        if dumpability is None:
            dumpability = np.ones(cls.SHAPE, dtype=np.bool_)
        state = State.new(
            jax.random.PRNGKey(7),
            cls._env_config(
                enforce_edge=enforce_edge,
                reward_stage=reward_stage,
            ),
            target,
            padding,
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            dumpability,
            action,
            distance_map_override=np.ones(cls.SHAPE, dtype=np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.array([32, 32], dtype=jnp.int16),
            angle_base=jnp.array([0], dtype=jnp.int8),
            angle_cabin=jnp.array([0], dtype=jnp.int8),
            loaded=jnp.array([loaded], dtype=jnp.int8),
        )
        return state._set_current_agent_state(current)._replace(
            env_steps=env_steps,
            productive_workspace_cycles=productive_workspace_cycles,
        )

    @classmethod
    def _workspace_coordinates(cls) -> np.ndarray:
        empty_target = np.zeros(cls.SHAPE, dtype=np.int8)
        state = cls._state(empty_target)
        workspace = np.asarray(state._build_dig_dump_cone()).reshape(cls.SHAPE)
        coordinates = np.argwhere(workspace)
        if coordinates.size == 0:
            raise AssertionError("Expected a non-empty tracked-excavator workspace.")
        return coordinates

    @staticmethod
    def _completion(state: State) -> dict[str, np.ndarray]:
        result = state._get_task_completion(
            state.world.action_map.map,
            state.world.target_map.map,
        )
        return {key: np.asarray(value) for key, value in result.items()}

    def test_contract_is_named(self):
        self.assertEqual(CORRECTED_DENSE_CONTRACT, "exact_visible_dump_v1")

    def test_exact_zone_is_success_and_former_buffer_is_not(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1

        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[20:22, 20:22] = -1
        exact_action[40, 40] = 4
        exact = self._state(target, action=exact_action)
        exact_completion = self._completion(exact)
        self.assertEqual(float(exact_completion["absolute_completion"]), 1.0)
        self.assertTrue(
            bool(
                exact._is_done_task(
                    exact.world.action_map.map,
                    exact.world.target_map.map,
                )
            )
        )

        buffer_action = np.zeros(self.SHAPE, dtype=np.int8)
        buffer_action[20:22, 20:22] = -1
        buffer_action[39, 40] = 4
        former_buffer = self._state(target, action=buffer_action)
        buffer_completion = self._completion(former_buffer)
        self.assertEqual(float(buffer_completion["dump_purity"]), 0.0)
        self.assertEqual(float(buffer_completion["absolute_completion"]), 0.0)
        self.assertFalse(
            bool(
                former_buffer._is_done_task(
                    former_buffer.world.action_map.map,
                    former_buffer.world.target_map.map,
                )
            )
        )

    def test_all_prerequisites_gate_combined_completion(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        partial_action = np.zeros(self.SHAPE, dtype=np.int8)
        partial_action[20, 20:22] = -1
        partial_action[40, 40] = 2

        partial = self._state(target, action=partial_action)
        partial_completion = self._completion(partial)
        self.assertEqual(float(partial_completion["dig_completion_total"]), 0.5)
        self.assertEqual(
            float(partial_completion["dump_volume_completion"]),
            0.5,
        )
        self.assertEqual(float(partial_completion["dump_purity"]), 1.0)
        self.assertEqual(float(partial_completion["absolute_completion"]), 0.5)
        self.assertEqual(int(partial_action.astype(np.int32).sum()), 0)

        loaded = self._state(target, action=partial_action, loaded=1)
        loaded_completion = self._completion(loaded)
        self.assertEqual(float(loaded_completion["unloaded_completion"]), 0.0)
        self.assertEqual(float(loaded_completion["absolute_completion"]), 0.0)

    def test_relocation_only_requires_positive_soil_in_exact_zone(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[40:43, 40:43] = 1

        empty = self._state(target)
        self.assertEqual(
            float(self._completion(empty)["absolute_completion"]),
            0.0,
        )

        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[41, 41] = 3
        exact = self._state(target, action=exact_action)
        self.assertEqual(
            float(self._completion(exact)["absolute_completion"]),
            1.0,
        )

        mixed_action = exact_action.copy()
        mixed_action[30, 30] = 1
        mixed = self._state(target, action=mixed_action)
        mixed_completion = self._completion(mixed)
        self.assertEqual(float(mixed_completion["dump_purity"]), 0.75)
        self.assertEqual(float(mixed_completion["absolute_completion"]), 0.75)

    def test_empty_task_and_target_obstacle_overlap_do_not_succeed(self):
        empty_target = np.zeros(self.SHAPE, dtype=np.int8)
        empty = self._state(empty_target)
        empty_completion = self._completion(empty)
        self.assertEqual(float(empty_completion["task_present"]), 0.0)
        self.assertEqual(float(empty_completion["absolute_completion"]), 0.0)

        target = empty_target.copy()
        target[40, 40] = 1
        padding = np.zeros(self.SHAPE, dtype=np.int8)
        padding[40, 40] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[40, 40] = 1
        invalid = self._state(target, action=action, padding=padding)
        invalid_completion = self._completion(invalid)
        self.assertEqual(float(invalid_completion["dump_mask_integrity"]), 0.0)
        self.assertEqual(float(invalid_completion["accepted_dump_volume"]), 0.0)
        self.assertEqual(float(invalid_completion["absolute_completion"]), 0.0)

    def test_foundation_edge_requirement_is_reported_and_gated(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:27, 20:27] = -1
        incomplete_action = np.where(target < 0, -1, 0).astype(np.int8)
        incomplete_action[20, 23] = 0
        incomplete = self._state(
            target,
            action=incomplete_action,
            enforce_edge=True,
        )
        incomplete_completion = self._completion(incomplete)
        self.assertLess(float(incomplete_completion["dig_completion_edge"]), 1.0)
        self.assertLess(float(incomplete_completion["absolute_completion"]), 1.0)

        complete_action = np.where(target < 0, -1, 0).astype(np.int8)
        complete = self._state(
            target,
            action=complete_action,
            enforce_edge=True,
        )
        complete_completion = self._completion(complete)
        self.assertEqual(float(complete_completion["dig_completion_edge"]), 1.0)
        self.assertEqual(float(complete_completion["absolute_completion"]), 1.0)

    def test_completion_eager_jit_and_vmap_agree(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[20:22, 20:22] = -1
        action[40, 40] = 4
        state = self._state(target, action=action)

        completion_fn = lambda state_: state_._calculate_completion_percentage(
            state_.world.action_map.map,
            state_.world.target_map.map,
        )
        eager = completion_fn(state)
        compiled = jax.jit(completion_fn)(state)
        batched_state = jax.tree_util.tree_map(
            lambda value: jnp.stack([jnp.asarray(value), jnp.asarray(value)]),
            state,
        )
        vectorized = jax.vmap(completion_fn)(batched_state)

        np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager))
        np.testing.assert_allclose(
            np.asarray(vectorized),
            np.repeat(np.asarray(eager)[None], 2, axis=0),
        )

    def test_terminal_reward_uses_the_same_absolute_completion(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        old_state = self._state(target)

        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[20:22, 20:22] = -1
        exact_action[40, 40] = 4
        exact_state = self._state(target, action=exact_action)
        _, exact_components = old_state._get_reward(
            exact_state,
            TrackedAction.do_nothing(),
        )
        self.assertEqual(float(exact_components["absolute_completion"]), 1.0)
        self.assertGreater(float(exact_components["terminal"]), 0.0)

        buffer_action = np.zeros(self.SHAPE, dtype=np.int8)
        buffer_action[20:22, 20:22] = -1
        buffer_action[39, 40] = 4
        buffer_state = self._state(target, action=buffer_action)
        _, buffer_components = old_state._get_reward(
            buffer_state,
            TrackedAction.do_nothing(),
        )
        self.assertEqual(float(buffer_components["absolute_completion"]), 0.0)
        self.assertEqual(float(buffer_components["terminal"]), 0.0)

    def test_one_cell_fresh_target_executes_normal_dig(self):
        coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(coordinate)] = -1
        state = self._state(target)
        old_map = np.asarray(state.world.action_map.map).astype(np.int32)

        dug = state._handle_dig()
        new_map = np.asarray(dug.world.action_map.map).astype(np.int32)
        loaded = int(dug._get_current_agent_state().loaded[0])

        self.assertEqual(int(new_map[tuple(coordinate)]), -1)
        self.assertEqual(int(np.count_nonzero(new_map)), 1)
        self.assertEqual(loaded, 1)
        self.assertEqual(int(old_map.sum()), int(new_map.sum()) + loaded)

    def test_one_cell_positive_soil_executes_normal_relift(self):
        coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[0, 0] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[tuple(coordinate)] = 7
        state = self._state(target, action=action)
        old_map = np.asarray(state.world.action_map.map).astype(np.int32)

        lifted = state._handle_dig()
        new_map = np.asarray(lifted.world.action_map.map).astype(np.int32)
        loaded = int(lifted._get_current_agent_state().loaded[0])

        self.assertEqual(int(new_map[tuple(coordinate)]), 0)
        self.assertEqual(int(np.count_nonzero(new_map)), 0)
        self.assertEqual(loaded, 7)
        self.assertEqual(int(old_map.sum()), int(new_map.sum()) + loaded)

    def test_over_capacity_positive_soil_loads_127_and_leaves_exact_remainder(self):
        coordinates = self._workspace_coordinates()
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[0, 0] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[tuple(coordinates.T)] = 3
        state = self._state(
            target,
            action=action,
            reward_stage=RewardStage.REWARD_V2,
        )._replace(
            stall_age_steps=jnp.int32(9)
        )
        initial_potential = float(
            state._compute_relocation_potential(state.world.action_map.map)
        )

        lifted = state._step(TrackedAction.do(), turn=False)
        lifted_map = np.asarray(lifted.world.action_map.map).astype(np.int32)
        loaded = int(lifted._get_current_agent_state().loaded[0])
        remaining = int(lifted_map.clip(min=0).sum())
        carry_credit = float(
            lifted._get_current_agent_state().carry_relocation_credit
        )
        remaining_potential = float(
            lifted._compute_relocation_potential(lifted.world.action_map.map)
        )

        self.assertEqual(int(action.sum()), 135)
        self.assertEqual(loaded, 127)
        self.assertEqual(remaining, 8)
        self.assertEqual(int(action.sum()), remaining + loaded)
        self.assertTrue(np.all(lifted_map >= 0))
        self.assertEqual(int(lifted.stall_age_steps), 0)
        self.assertAlmostEqual(
            carry_credit,
            initial_potential - remaining_potential,
            places=5,
        )

    def test_repeated_capacity_pickups_conserve_mass_and_clear_the_source(self):
        empty_target = np.zeros(self.SHAPE, dtype=np.int8)
        probe = self._state(empty_target)
        pickup_mask = np.asarray(probe._build_dig_dump_cone()).reshape(self.SHAPE)
        pickup_coordinates = np.argwhere(pickup_mask)

        opposite_agent = probe._get_current_agent_state()._replace(
            angle_cabin=jnp.array([6], dtype=jnp.int8)
        )
        opposite = probe._set_current_agent_state(opposite_agent)
        dump_mask = np.asarray(opposite._build_dig_dump_cone()).reshape(self.SHAPE)
        self.assertFalse(bool(np.any(pickup_mask & dump_mask)))

        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[dump_mask] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[tuple(pickup_coordinates.T)] = 3
        state = self._state(
            target,
            action=action,
            reward_stage=RewardStage.REWARD_V2,
        )
        initial_mass = int(action.sum())

        first_lift = state._step(TrackedAction.do(), turn=False)
        self.assertEqual(int(first_lift._get_current_agent_state().loaded[0]), 127)

        dump_agent = first_lift._get_current_agent_state()._replace(
            angle_cabin=jnp.array([6], dtype=jnp.int8)
        )
        first_dump = first_lift._set_current_agent_state(dump_agent)._step(
            TrackedAction.do(), turn=False
        )
        self.assertEqual(int(first_dump._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(
            float(first_dump._get_current_agent_state().carry_relocation_credit),
            0.0,
        )

        pickup_agent = first_dump._get_current_agent_state()._replace(
            angle_cabin=jnp.array([0], dtype=jnp.int8)
        )
        second_lift = first_dump._set_current_agent_state(pickup_agent)._step(
            TrackedAction.do(), turn=False
        )
        second_load = int(second_lift._get_current_agent_state().loaded[0])
        second_map = np.asarray(second_lift.world.action_map.map).astype(np.int32)
        self.assertEqual(second_load, 8)
        self.assertEqual(int(second_map[pickup_mask].clip(min=0).sum()), 0)
        self.assertEqual(int(second_map.sum()) + second_load, initial_mass)

        dump_agent = second_lift._get_current_agent_state()._replace(
            angle_cabin=jnp.array([6], dtype=jnp.int8)
        )
        final = second_lift._set_current_agent_state(dump_agent)._step(
            TrackedAction.do(), turn=False
        )
        final_map = np.asarray(final.world.action_map.map).astype(np.int32)
        self.assertEqual(int(final._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(int(final_map[pickup_mask].clip(min=0).sum()), 0)
        self.assertEqual(int(final_map.sum()), initial_mass)
        self.assertEqual(
            float(final._get_current_agent_state().carry_relocation_credit),
            0.0,
        )

    def test_legal_dump_is_contained_and_conserves_mass(self):
        legal_coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        state = self._state(target, loaded=11)
        old_map = np.asarray(state.world.action_map.map).astype(np.int32)

        dumped = state._handle_dump()
        new_map = np.asarray(dumped.world.action_map.map).astype(np.int32)
        delta = new_map - old_map
        accepted = np.asarray(dumped._accepted_dump_mask())

        self.assertEqual(int(delta.sum()), 11)
        self.assertEqual(int(np.count_nonzero(delta[~accepted])), 0)
        self.assertEqual(int(dumped._get_current_agent_state().loaded[0]), 0)
        self.assertEqual(
            int(old_map.sum()) + 11,
            int(new_map.sum()) + int(dumped._get_current_agent_state().loaded[0]),
        )

    def test_legal_dump_contains_relaxation_that_would_cross_boundary(self):
        legal_coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        state = self._state(target, loaded=20)
        accepted = np.asarray(state._accepted_dump_mask())

        unconstrained_seed = np.zeros(self.SHAPE, dtype=np.int16)
        unconstrained_seed[tuple(legal_coordinate)] = 20
        unconstrained = state._apply_local_soil_mechanics(
            jnp.asarray(unconstrained_seed),
            jnp.asarray(accepted),
        )
        self.assertGreater(
            int(np.asarray(unconstrained)[~accepted].sum()),
            0,
        )

        dumped = state._handle_dump()
        dumped_map = np.asarray(dumped.world.action_map.map).astype(np.int32)
        self.assertEqual(int(dumped_map[~accepted].sum()), 0)
        self.assertEqual(int(dumped_map[accepted].sum()), 20)

    def test_workspace_overlap_prefers_accepted_cells(self):
        coordinates = self._workspace_coordinates()
        legal_coordinate = coordinates[0]
        neutral_coordinate = coordinates[-1]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        state = self._state(target, loaded=9)

        dumped = state._handle_dump()
        dumped_map = np.asarray(dumped.world.action_map.map).astype(np.int32)
        self.assertEqual(int(dumped_map[tuple(legal_coordinate)]), 9)
        self.assertEqual(int(dumped_map[tuple(neutral_coordinate)]), 0)

    def test_entirely_off_zone_dump_executes_despite_potential_increase(self):
        coordinates = self._workspace_coordinates()
        workspace = np.zeros(self.SHAPE, dtype=np.bool_)
        workspace[coordinates[:, 0], coordinates[:, 1]] = True
        outside_coordinate = np.argwhere(~workspace)[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(outside_coordinate)] = 1
        state = self._state(target, loaded=13)
        old_potential = float(
            state._compute_relocation_potential(state.world.action_map.map)
        )

        dumped = state._handle_dump()
        dumped_map = np.asarray(dumped.world.action_map.map).astype(np.int32)
        new_potential = float(
            dumped._compute_relocation_potential(dumped.world.action_map.map)
        )
        accepted = np.asarray(dumped._accepted_dump_mask())

        self.assertEqual(int(dumped_map.sum()), 13)
        self.assertEqual(int(dumped_map[accepted].sum()), 0)
        self.assertEqual(int(dumped._get_current_agent_state().loaded[0]), 0)
        self.assertGreater(new_potential, old_potential)

    def test_unrepresentable_dump_is_rejected_without_mass_loss(self):
        legal_coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        action = np.zeros(self.SHAPE, dtype=np.int8)
        action[tuple(legal_coordinate)] = np.iinfo(np.int8).max
        state = self._state(target, action=action, loaded=1)
        old_map = np.asarray(state.world.action_map.map).copy()

        rejected = state._handle_dump()

        np.testing.assert_array_equal(
            np.asarray(rejected.world.action_map.map),
            old_map,
        )
        self.assertEqual(int(rejected._get_current_agent_state().loaded[0]), 1)

    def test_obstacle_and_non_dumpable_neighbors_are_not_modified(self):
        coordinates = self._workspace_coordinates()
        legal_coordinate = coordinates[len(coordinates) // 2]
        neighbor_candidates = np.array(
            [
                legal_coordinate + np.array([-1, 0]),
                legal_coordinate + np.array([1, 0]),
                legal_coordinate + np.array([0, -1]),
                legal_coordinate + np.array([0, 1]),
            ]
        )
        valid_neighbors = neighbor_candidates[
            np.all(
                (neighbor_candidates >= 0)
                & (neighbor_candidates < np.array(self.SHAPE)),
                axis=1,
            )
        ]
        obstacle_coordinate = valid_neighbors[0]
        non_dumpable_coordinate = valid_neighbors[1]

        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        padding = np.zeros(self.SHAPE, dtype=np.int8)
        padding[tuple(obstacle_coordinate)] = 1
        dumpability = np.ones(self.SHAPE, dtype=np.bool_)
        dumpability[tuple(non_dumpable_coordinate)] = False
        state = self._state(
            target,
            padding=padding,
            dumpability=dumpability,
            loaded=20,
        )

        dumped = state._handle_dump()
        dumped_map = np.asarray(dumped.world.action_map.map).astype(np.int32)
        self.assertEqual(int(dumped_map[tuple(obstacle_coordinate)]), 0)
        self.assertEqual(int(dumped_map[tuple(non_dumpable_coordinate)]), 0)
        self.assertEqual(int(dumped_map.sum()), 20)

    def test_repeated_dump_relift_cycle_conserves_mass(self):
        legal_coordinates = self._workspace_coordinates()[:2]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[
            legal_coordinates[:, 0],
            legal_coordinates[:, 1],
        ] = 1
        state = self._state(target, loaded=6)

        first_dump = state._handle_dump()
        first_lift = first_dump._handle_dig()
        second_dump = first_lift._handle_dump()
        second_lift = second_dump._handle_dig()

        self.assertEqual(
            int(np.asarray(first_dump.world.action_map.map).sum()),
            6,
        )
        self.assertEqual(int(first_lift._get_current_agent_state().loaded[0]), 6)
        self.assertEqual(
            int(np.asarray(first_lift.world.action_map.map).sum()),
            0,
        )
        self.assertEqual(
            int(np.asarray(second_dump.world.action_map.map).sum()),
            6,
        )
        self.assertEqual(int(second_lift._get_current_agent_state().loaded[0]), 6)
        self.assertEqual(
            int(np.asarray(second_lift.world.action_map.map).sum()),
            0,
        )

    def test_transition_diagnostics_count_one_mass_conserving_load_cycle(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        old_state = self._state(target)
        action_map = old_state.world.action_map.map.at[30, 30].set(-5)
        new_agent = old_state._get_current_agent_state()._replace(
            loaded=jnp.array([5], dtype=jnp.int8)
        )
        new_state = old_state._replace(
            world=old_state.world._replace(
                action_map=old_state.world.action_map._replace(map=action_map)
            )
        )._set_current_agent_state(new_agent)

        diagnostics = TerraEnv._transition_diagnostics(old_state, new_state)

        self.assertTrue(bool(diagnostics["action_had_effect"]))
        self.assertEqual(int(diagnostics["productive_workspace_cycle"]), 1)
        self.assertEqual(int(diagnostics["transition_mass_residual"]), 0)
        self.assertFalse(bool(diagnostics["target_mutation"]))
        self.assertFalse(bool(diagnostics["obstacle_mutation"]))

    def test_productive_workspace_counter_counts_load_boundaries_and_resets(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        old_state = self._state(target)
        loaded_agent = old_state._get_current_agent_state()._replace(
            loaded=jnp.array([5], dtype=jnp.int8)
        )
        loaded_state = old_state._set_current_agent_state(loaded_agent)
        first_diagnostics = TerraEnv._transition_diagnostics(
            old_state,
            loaded_state,
        )
        first_cycle = TerraEnv._accumulate_productive_workspace_cycles(
            old_state,
            loaded_state,
            first_diagnostics,
        )
        self.assertEqual(int(first_cycle.productive_workspace_cycles), 1)

        same_load_diagnostics = TerraEnv._transition_diagnostics(
            first_cycle,
            first_cycle,
        )
        same_cycle = TerraEnv._accumulate_productive_workspace_cycles(
            first_cycle,
            first_cycle,
            same_load_diagnostics,
        )
        self.assertEqual(int(same_cycle.productive_workspace_cycles), 1)

        unloaded_agent = same_cycle._get_current_agent_state()._replace(
            loaded=jnp.array([0], dtype=jnp.int8)
        )
        unloaded_state = same_cycle._set_current_agent_state(unloaded_agent)
        unload_diagnostics = TerraEnv._transition_diagnostics(
            same_cycle,
            unloaded_state,
        )
        after_unload = TerraEnv._accumulate_productive_workspace_cycles(
            same_cycle,
            unloaded_state,
            unload_diagnostics,
        )
        self.assertEqual(int(after_unload.productive_workspace_cycles), 1)

        reloaded_agent = after_unload._get_current_agent_state()._replace(
            loaded=jnp.array([3], dtype=jnp.int8)
        )
        reloaded_state = after_unload._set_current_agent_state(reloaded_agent)
        reload_diagnostics = TerraEnv._transition_diagnostics(
            after_unload,
            reloaded_state,
        )
        second_cycle = TerraEnv._accumulate_productive_workspace_cycles(
            after_unload,
            reloaded_state,
            reload_diagnostics,
        )
        self.assertEqual(int(second_cycle.productive_workspace_cycles), 2)
        self.assertEqual(
            int(self._state(target).productive_workspace_cycles),
            0,
        )

        terminal_target = np.zeros(self.SHAPE, dtype=np.int8)
        terminal_target[20:22, 20:22] = -1
        terminal_target[40, 40] = 1
        terminal_action = np.zeros(self.SHAPE, dtype=np.int8)
        terminal_action[20:22, 20:22] = -1
        terminal_action[40, 40] = 4
        terminal_state = self._state(
            terminal_target,
            action=terminal_action,
            reward_stage=RewardStage.TERMINAL_OBJECTIVE,
            env_steps=100,
            productive_workspace_cycles=3,
        )
        env = TerraEnv.new(maps_size_px=64)
        step_no_reset = TerraEnv.step_no_reset.__wrapped__(
            env,
            terminal_state,
            TrackedAction.do_nothing(),
            terminal_state.env_cfg,
        )
        self.assertTrue(bool(step_no_reset.done))
        self.assertEqual(
            int(step_no_reset.info["productive_workspace_cycles"]),
            3,
        )
        self.assertEqual(
            int(step_no_reset.state.productive_workspace_cycles),
            3,
        )

        auto_reset = TerraEnv.step.__wrapped__(
            env,
            terminal_state,
            TrackedAction.do_nothing(),
            terminal_target,
            np.zeros(self.SHAPE, dtype=np.int8),
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(self.SHAPE, dtype=np.bool_),
            np.zeros(self.SHAPE, dtype=np.int8),
            np.ones(self.SHAPE, dtype=np.float32),
            terminal_state.env_cfg,
        )
        self.assertTrue(bool(auto_reset.done))
        self.assertEqual(
            int(auto_reset.info["productive_workspace_cycles"]),
            3,
        )
        self.assertEqual(
            int(auto_reset.state.productive_workspace_cycles),
            0,
        )

    def test_terminal_objective_is_terminal_only_and_orders_successes(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        old_state = self._state(
            target,
            reward_stage=RewardStage.TERMINAL_OBJECTIVE,
        )

        nonterminal = self._state(
            target,
            reward_stage=RewardStage.TERMINAL_OBJECTIVE,
            env_steps=1,
        )
        nonterminal_reward, nonterminal_components = old_state._get_reward(
            nonterminal,
            TrackedAction.do_nothing(),
        )
        self.assertEqual(float(nonterminal_reward), 0.0)
        self.assertEqual(float(nonterminal_components["terminal"]), 0.0)
        self.assertEqual(float(nonterminal_components["existence"]), 0.0)
        self.assertEqual(float(nonterminal_components["trench"]), 0.0)
        np.testing.assert_array_equal(
            np.asarray(nonterminal_components["agent_rewards"]),
            np.zeros((4,), dtype=np.float32),
        )

        timeout = nonterminal._replace(env_steps=450)
        timeout_reward, timeout_components = old_state._get_reward(
            timeout,
            TrackedAction.do_nothing(),
        )
        self.assertEqual(float(timeout_reward), -1.0)
        self.assertEqual(float(timeout_components["workspace_efficiency"]), 0.0)
        self.assertEqual(float(timeout_components["step_efficiency"]), 0.0)

        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[20:22, 20:22] = -1
        exact_action[40, 40] = 4
        efficient = self._state(
            target,
            action=exact_action,
            reward_stage=RewardStage.TERMINAL_OBJECTIVE,
            env_steps=100,
            productive_workspace_cycles=1,
        )
        extra_workspace = efficient._replace(productive_workspace_cycles=2)
        slower = efficient._replace(env_steps=200)

        efficient_reward, efficient_components = old_state._get_reward(
            efficient,
            TrackedAction.do_nothing(),
        )
        extra_workspace_reward, _ = old_state._get_reward(
            extra_workspace,
            TrackedAction.do_nothing(),
        )
        slower_reward, _ = old_state._get_reward(
            slower,
            TrackedAction.do_nothing(),
        )
        expected_step_efficiency = 1.0 - 100.0 / 450.0
        dense_success_base = 2.0 * 200.0 / 70.0
        expected_reward = dense_success_base * (
            1.0 + 0.15 + 0.05 * expected_step_efficiency
        )
        np.testing.assert_allclose(
            np.asarray(efficient_reward),
            np.asarray(expected_reward, dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(efficient_components["workspace_efficiency"]),
            np.asarray(1.0, dtype=np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(efficient_components["step_efficiency"]),
            np.asarray(expected_step_efficiency, dtype=np.float32),
        )
        self.assertGreater(float(efficient_reward), float(extra_workspace_reward))
        self.assertGreater(float(efficient_reward), float(slower_reward))
        self.assertGreater(float(slower_reward), float(timeout_reward))
        self.assertGreaterEqual(float(efficient_reward), dense_success_base)
        self.assertLessEqual(float(efficient_reward), dense_success_base * 1.2)

        reward_fn = lambda candidate: old_state._get_reward(
            candidate,
            TrackedAction.do_nothing(),
        )[0]
        compiled = jax.jit(reward_fn)(efficient)
        batched_states = jax.tree_util.tree_map(
            lambda value: jnp.stack([jnp.asarray(value), jnp.asarray(value)]),
            efficient,
        )
        vectorized = jax.vmap(reward_fn)(batched_states)
        np.testing.assert_allclose(np.asarray(compiled), np.asarray(efficient_reward))
        np.testing.assert_allclose(
            np.asarray(vectorized),
            np.repeat(np.asarray(efficient_reward)[None], 2, axis=0),
        )

    def test_dense_reward_golden_values_are_unchanged(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        old_state = self._state(target)

        nonterminal = self._state(target, env_steps=1)
        nonterminal_reward, nonterminal_components = old_state._get_reward(
            nonterminal,
            TrackedAction.do_nothing(),
        )
        np.testing.assert_allclose(
            np.asarray(nonterminal_reward),
            np.asarray(-0.0035714285913854837, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(nonterminal_components["existence"]),
            np.asarray(-0.0035714285714285713, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[20:22, 20:22] = -1
        exact_action[40, 40] = 4
        success = self._state(target, action=exact_action, env_steps=101)
        success_reward, success_components = old_state._get_reward(
            success,
            TrackedAction.do_nothing(),
        )
        np.testing.assert_allclose(
            np.asarray(success_reward),
            np.asarray(6.853571891784668, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(success_components["terminal"]),
            np.asarray(6.857143402099609, dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(float(success_components["workspace_efficiency"]), 0.0)
        self.assertEqual(float(success_components["step_efficiency"]), 0.0)

    def test_annealed_reward_endpoints_and_midpoint_are_exact(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        exact_action = np.zeros(self.SHAPE, dtype=np.int8)
        exact_action[20:22, 20:22] = -1
        exact_action[40, 40] = 4
        old_state = self._state(target)
        dense = self._state(target, action=exact_action, env_steps=100)
        terminal = dense._replace(
            env_cfg=dense.env_cfg._replace(
                reward_stage=RewardStage.TERMINAL_OBJECTIVE,
            )
        )

        dense_reward, _ = old_state._get_reward(
            dense,
            TrackedAction.do_nothing(),
        )
        terminal_reward, _ = old_state._get_reward(
            terminal,
            TrackedAction.do_nothing(),
        )
        observed = []
        for terminal_mix in (0.0, 0.5, 1.0):
            annealed = dense._replace(
                env_cfg=dense.env_cfg._replace(
                    reward_stage=RewardStage.ANNEALED_OBJECTIVE,
                    terminal_reward_mix=terminal_mix,
                )
            )
            reward, _ = old_state._get_reward(
                annealed,
                TrackedAction.do_nothing(),
            )
            observed.append(reward)
        np.testing.assert_allclose(observed[0], dense_reward, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            observed[1],
            0.5 * (dense_reward + terminal_reward),
            rtol=0.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(observed[2], terminal_reward, rtol=0.0, atol=0.0)

    def test_live_anneal_mix_applies_and_survives_auto_reset(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[20:22, 20:22] = -1
        target[40, 40] = 1
        completed_action = np.zeros(self.SHAPE, dtype=np.int8)
        completed_action[20:22, 20:22] = -1
        completed_action[40, 40] = 4
        state = self._state(
            target,
            action=completed_action,
            reward_stage=RewardStage.ANNEALED_OBJECTIVE,
            env_steps=100,
            productive_workspace_cycles=1,
        )
        live_cfg = state.env_cfg._replace(terminal_reward_mix=1.0)
        env = TerraEnv.new(maps_size_px=64)

        terminal = TerraEnv.step_no_reset.__wrapped__(
            env,
            state,
            TrackedAction.do_nothing(),
            live_cfg,
        )
        dense_success_base = 2.0 * 200.0 / 70.0
        expected_reward = dense_success_base * (
            1.0 + 0.15 + 0.05 * (1.0 - 101.0 / 450.0)
        )
        np.testing.assert_allclose(
            np.asarray(terminal.reward),
            np.asarray(expected_reward, dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(float(terminal.env_cfg.terminal_reward_mix), 1.0)
        self.assertEqual(float(terminal.state.env_cfg.terminal_reward_mix), 1.0)

        auto_reset = TerraEnv.step.__wrapped__(
            env,
            state,
            TrackedAction.do_nothing(),
            target,
            np.zeros(self.SHAPE, dtype=np.int8),
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            np.ones(self.SHAPE, dtype=np.bool_),
            np.zeros(self.SHAPE, dtype=np.int8),
            np.ones(self.SHAPE, dtype=np.float32),
            live_cfg,
        )
        self.assertTrue(bool(auto_reset.done))
        self.assertEqual(float(auto_reset.env_cfg.terminal_reward_mix), 1.0)
        self.assertEqual(float(auto_reset.state.env_cfg.terminal_reward_mix), 1.0)

    def test_reward_fields_are_appended_for_legacy_positional_checkpoints(self):
        current = EnvConfig()
        self.assertEqual(
            EnvConfig._fields[-4:],
            (
                "reward_stage",
                "terminal_reward_mix",
                "reward_v2_timing_variant",
                "reset_tier",
            ),
        )
        legacy_values = pickle.loads(pickle.dumps(tuple(current)[:-4]))
        restored = EnvConfig(*legacy_values)
        self.assertEqual(
            restored.reward_stage,
            RewardStage.DENSE_SKILL,
        )
        self.assertEqual(restored.terminal_reward_mix, 0.0)
        self.assertEqual(
            restored.reward_v2_timing_variant,
            REWARD_V2_TIMING_BASELINE,
        )
        self.assertEqual(restored.reset_tier, 0)
        # A pre-timing checkpoint (reward_stage + mix, no variant) also loads.
        pre_timing = EnvConfig(*tuple(current)[:-2])
        self.assertEqual(
            pre_timing.reward_v2_timing_variant,
            REWARD_V2_TIMING_BASELINE,
        )
        self.assertEqual(pre_timing.reset_tier, 0)
        for field_name in EnvConfig._fields[:-4]:
            self.assertEqual(
                getattr(restored, field_name),
                getattr(current, field_name),
            )

    def test_transition_diagnostics_ignore_bookkeeping_and_detect_integrity(self):
        target = np.zeros(self.SHAPE, dtype=np.int8)
        old_state = self._state(target)
        bookkeeping_only = old_state._replace(
            env_steps=old_state.env_steps + 1,
            agent=old_state.agent._replace(current_agent=1),
        )
        diagnostics = TerraEnv._transition_diagnostics(
            old_state,
            bookkeeping_only,
        )
        self.assertFalse(bool(diagnostics["action_had_effect"]))
        self.assertEqual(int(diagnostics["transition_mass_residual"]), 0)

        target_map = old_state.world.target_map.map.at[1, 1].set(-1)
        padding_map = old_state.world.padding_mask.map.at[2, 2].set(1)
        action_map = old_state.world.action_map.map.at[3, 3].set(1)
        corrupted = old_state._replace(
            world=old_state.world._replace(
                target_map=old_state.world.target_map._replace(map=target_map),
                padding_mask=old_state.world.padding_mask._replace(map=padding_map),
                action_map=old_state.world.action_map._replace(map=action_map),
            )
        )
        diagnostics = TerraEnv._transition_diagnostics(old_state, corrupted)
        self.assertTrue(bool(diagnostics["target_mutation"]))
        self.assertTrue(bool(diagnostics["obstacle_mutation"]))
        self.assertEqual(int(diagnostics["transition_mass_residual"]), 1)

    def test_dump_eager_jit_and_vmap_agree(self):
        legal_coordinate = self._workspace_coordinates()[0]
        target = np.zeros(self.SHAPE, dtype=np.int8)
        target[tuple(legal_coordinate)] = 1
        state = self._state(target, loaded=7)

        dump_fn = lambda state_: (
            state_._handle_dump().world.action_map.map,
            state_._handle_dump()._get_current_agent_state().loaded,
        )
        eager_map, eager_load = dump_fn(state)
        compiled_map, compiled_load = jax.jit(dump_fn)(state)
        batched_state = jax.tree_util.tree_map(
            lambda value: jnp.stack([jnp.asarray(value), jnp.asarray(value)]),
            state,
        )
        vectorized_map, vectorized_load = jax.vmap(dump_fn)(batched_state)

        np.testing.assert_array_equal(
            np.asarray(compiled_map),
            np.asarray(eager_map),
        )
        np.testing.assert_array_equal(
            np.asarray(compiled_load),
            np.asarray(eager_load),
        )
        np.testing.assert_array_equal(
            np.asarray(vectorized_map),
            np.repeat(np.asarray(eager_map)[None], 2, axis=0),
        )
        np.testing.assert_array_equal(
            np.asarray(vectorized_load),
            np.repeat(np.asarray(eager_load)[None], 2, axis=0),
        )


if __name__ == "__main__":
    unittest.main()
