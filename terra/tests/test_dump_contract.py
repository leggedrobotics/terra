import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnvBatch
from terra.state import CORRECTED_DENSE_CONTRACT
from terra.state import State


class ExactDumpContractTest(unittest.TestCase):
    SHAPE = (64, 64)

    @staticmethod
    def _env_config(*, enforce_edge: bool = False) -> EnvConfig:
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
    ) -> State:
        if action is None:
            action = np.zeros(cls.SHAPE, dtype=np.int8)
        if padding is None:
            padding = np.zeros(cls.SHAPE, dtype=np.int8)
        if dumpability is None:
            dumpability = np.ones(cls.SHAPE, dtype=np.bool_)
        state = State.new(
            jax.random.PRNGKey(7),
            cls._env_config(enforce_edge=enforce_edge),
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
        return state._set_current_agent_state(current)

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
