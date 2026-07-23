import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.env_generation.partial_completion import AGENT_HEIGHT_TILES
from terra.env_generation.partial_completion import AGENT_WIDTH_TILES
from terra.env_generation.partial_completion import PartialCompletionConfig
from terra.env_generation.partial_completion import PartialCompletionError
from terra.env_generation.partial_completion import TILE_SIZE_M
from terra.env_generation.partial_completion import compute_dynamic_dumpability_numpy
from terra.env_generation.partial_completion import generate_partial_dataset
from terra.map import GridWorld
from terra.maps_buffer import MapsBuffer
from terra.maps_buffer import actions_sanity_check
from terra.maps_buffer import load_maps_from_disk
from terra.state import State


class PartialActionLoadingTest(unittest.TestCase):
    @staticmethod
    def _current_training_env_config() -> EnvConfig:
        """Use Terra's production config update for one current 64x64 env."""
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
        )

    def test_generator_geometry_matches_current_training_config(self):
        env_cfg = self._current_training_env_config()
        self.assertAlmostEqual(env_cfg.tile_size, TILE_SIZE_M, places=6)
        self.assertEqual(env_cfg.agent.width, AGENT_WIDTH_TILES)
        self.assertEqual(env_cfg.agent.height, AGENT_HEIGHT_TILES)

    def test_action_sanity_accepts_heights_and_rejects_invalid_ranges(self):
        actions_sanity_check(np.array([[-1, 0, 2, 127]], dtype=np.int16))

        with self.assertRaisesRegex(RuntimeError, "integer dtype"):
            actions_sanity_check(np.array([[0.0, 1.0]], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "min=-2"):
            actions_sanity_check(np.array([[-2, 0]], dtype=np.int16))
        with self.assertRaisesRegex(RuntimeError, "max=128"):
            actions_sanity_check(np.array([[0, 128]], dtype=np.int16))

    def test_grid_world_recomputes_initial_dynamic_dumpability(self):
        shape = (16, 16)
        target = np.zeros(shape, dtype=np.int8)
        action = np.zeros(shape, dtype=np.int8)
        action[7, 7] = -1
        static_dumpability = np.ones(shape, dtype=np.bool_)
        world = GridWorld.new(
            target,
            np.zeros(shape, dtype=np.int8),
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            static_dumpability,
            action,
            relocation_distance_map_override=np.ones(shape, dtype=np.float32),
        )

        expected = compute_dynamic_dumpability_numpy(static_dumpability, action)
        np.testing.assert_array_equal(np.asarray(world.dumpability_mask.map), expected)
        np.testing.assert_array_equal(
            np.asarray(world.dumpability_mask_init.map),
            static_dumpability,
        )

        zero_world = GridWorld.new(
            target,
            np.zeros(shape, dtype=np.int8),
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            static_dumpability,
            np.zeros(shape, dtype=np.int8),
            relocation_distance_map_override=np.ones(shape, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(zero_world.dumpability_mask.map),
            static_dumpability,
        )

    @staticmethod
    def _write_source_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for folder in ("images", "occupancy", "dumpability", "distance", "metadata"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        target = np.zeros((64, 64), dtype=np.int8)
        target[18:31, 18:31] = -1
        target[42:55, 40:55] = 1
        occupancy = np.zeros_like(target, dtype=np.bool_)
        dumpability = np.ones_like(target, dtype=np.bool_)
        distance = np.ones_like(target, dtype=np.float32)
        np.save(root / "images" / "img_1.npy", target)
        np.save(root / "occupancy" / "img_1.npy", occupancy)
        np.save(root / "dumpability" / "img_1.npy", dumpability)
        np.save(root / "distance" / "img_1.npy", distance)
        with (root / "metadata" / "trench_1.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump({"axes_ABC": []}, stream)
        return target, occupancy, dumpability

    def test_generated_dataset_loads_through_maps_buffer_and_state_reset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source"
            output = temporary / "partial"
            target, occupancy, dumpability = self._write_source_dataset(source)
            config = PartialCompletionConfig(
                completion_fractions=(0.25,),
                variants_per_fraction=1,
                mode_weights=(("in_zone", 1.0),),
                min_spawn_centers=1,
                max_attempts_per_variant=30,
                seed=19,
            )
            generate_partial_dataset(source, output, config=config)

            generated_action = np.load(output / "actions" / "img_1.npy")
            self.assertEqual(generated_action.dtype, np.int8)
            self.assertGreater(int(generated_action.max()), 1)
            np.testing.assert_array_equal(
                np.load(output / "images" / "img_1.npy"), target
            )
            np.testing.assert_array_equal(
                np.load(output / "occupancy" / "img_1.npy"),
                occupancy,
            )
            np.testing.assert_array_equal(
                np.load(output / "dumpability" / "img_1.npy"),
                dumpability,
            )

            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                (
                    maps,
                    occupancies,
                    trench_axes,
                    trench_types,
                    foundation_axes,
                    foundation_types,
                    dumpability_masks,
                    actions,
                    distances,
                ) = load_maps_from_disk(str(output))

            buffer = MapsBuffer.new(
                maps=maps[None, ...],
                padding_mask=occupancies[None, ...],
                trench_axes=trench_axes[None, ...],
                trench_types=trench_types[None, ...],
                foundation_border_axes=foundation_axes[None, ...],
                foundation_border_types=foundation_types[None, ...],
                dumpability_masks_init=dumpability_masks[None, ...],
                action_maps=actions[None, ...],
                distance_maps=distances[None, ...],
            )
            self.assertEqual(buffer.action_maps.dtype, jnp.int8)
            self.assertGreater(int(np.asarray(buffer.action_maps).max()), 1)

            env_cfg = self._current_training_env_config()
            state = State.new(
                jax.random.PRNGKey(0),
                env_cfg,
                maps[0],
                occupancies[0],
                trench_axes[0],
                trench_types[0],
                foundation_axes[0],
                foundation_types[0],
                dumpability_masks[0],
                actions[0],
                distance_map_override=distances[0],
            )
            expected_dynamic = compute_dynamic_dumpability_numpy(
                dumpability,
                generated_action,
            )
            np.testing.assert_array_equal(
                np.asarray(state.world.dumpability_mask.map),
                expected_dynamic,
            )
            self.assertGreater(int(np.asarray(state.world.action_map.map).max()), 1)

            wrapped = TerraEnv.wrap_state(state, update_reachability=jnp.bool_(False))
            self.assertEqual(wrapped.world.action_map.map.dtype, jnp.int8)

    def test_runtime_excavator_accepts_127_and_rejects_128(self):
        shape = (64, 64)
        target = np.zeros(shape, dtype=np.int8)
        occupancy = np.zeros(shape, dtype=np.int8)
        dumpability = np.ones(shape, dtype=np.bool_)
        action = np.zeros(shape, dtype=np.int8)

        env_cfg = self._current_training_env_config()
        state = State.new(
            jax.random.PRNGKey(127),
            env_cfg,
            target,
            occupancy,
            -97.0 * np.ones((3, 3), dtype=np.float32),
            np.int32(-1),
            -97.0 * np.ones((64, 3), dtype=np.float32),
            np.int32(-1),
            dumpability,
            action,
            distance_map_override=np.ones(shape, dtype=np.float32),
        )
        current = state._get_current_agent_state()._replace(
            pos_base=jnp.array([32, 32], dtype=jnp.int16),
            angle_base=jnp.array([0], dtype=jnp.int8),
            angle_cabin=jnp.array([0], dtype=jnp.int8),
        )
        pose_state = state._set_current_agent_state(current)
        workspace = np.asarray(pose_state._build_dig_dump_cone()).reshape(shape)
        remaining = 127
        for x, y in np.argwhere(workspace):
            if remaining == 0:
                break
            amount = min(4, remaining)
            action[int(x), int(y)] = amount
            remaining -= amount
        self.assertEqual(remaining, 0)
        self.assertGreater(int(np.count_nonzero(action)), 1)
        pose_state = pose_state._replace(
            world=pose_state.world._replace(
                action_map=pose_state.world.action_map._replace(
                    map=jnp.asarray(action),
                )
            )
        )
        state_127 = pose_state._replace(
            world=pose_state.world._replace(
                action_map=pose_state.world.action_map._replace(
                    map=jnp.asarray(action),
                )
            )
        )
        lifted = state_127._handle_dig()

        self.assertEqual(
            int(lifted._get_current_agent_state().loaded[0]),
            127,
        )
        self.assertEqual(
            int(np.asarray(lifted.world.action_map.map).astype(np.int32).sum()),
            0,
        )

        action_128 = action.copy()
        first_workspace_tile = np.argwhere(workspace)[0]
        action_128[
            int(first_workspace_tile[0]),
            int(first_workspace_tile[1]),
        ] += 1
        state_128 = pose_state._replace(
            world=pose_state.world._replace(
                action_map=pose_state.world.action_map._replace(
                    map=jnp.asarray(action_128),
                )
            )
        )
        rejected = state_128._handle_dig()
        self.assertEqual(
            int(rejected._get_current_agent_state().loaded[0]),
            0,
        )
        self.assertEqual(
            int(np.asarray(rejected.world.action_map.map).astype(np.int32).sum()),
            128,
        )

    def test_dataset_generation_rejects_nonzero_source_actions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            source = temporary / "source"
            output = temporary / "partial"
            target, _, _ = self._write_source_dataset(source)
            (source / "actions").mkdir()
            source_action = np.zeros_like(target, dtype=np.int8)
            source_action[10, 10] = 1
            np.save(source / "actions" / "img_1.npy", source_action)
            config = PartialCompletionConfig(
                completion_fractions=(0.25,),
                mode_weights=(("in_zone", 1.0),),
                min_spawn_centers=1,
            )

            with self.assertRaisesRegex(
                PartialCompletionError,
                "already has a nonzero action map",
            ):
                generate_partial_dataset(source, output, config=config)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
