import unittest

import numpy as np

from terra.env_generation.partial_completion import PartialCompletionConfig
from terra.env_generation.partial_completion import RELAY_CENTER_MAX_ROUTE_EXCESS_TILES
from terra.env_generation.partial_completion import _maximum_workspace_load
from terra.env_generation.partial_completion import _relay_corridor_masks
from terra.env_generation.partial_completion import _runtime_sampling_domain
from terra.env_generation.partial_completion import _select_completed_mask
from terra.env_generation.partial_completion import compute_dynamic_dumpability_numpy
from terra.env_generation.partial_completion import generate_partial_action_map
from terra.env_generation.partial_completion import validate_partial_state


class PartialCompletionGenerationTest(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((64, 64), dtype=np.int8)
        self.target[18:31, 18:31] = -1
        self.target[42:55, 40:55] = 1
        self.occupancy = np.zeros_like(self.target, dtype=np.bool_)
        self.dumpability = np.ones_like(self.target, dtype=np.bool_)

    @staticmethod
    def _config(mode, **overrides):
        values = {
            "completion_fractions": (0.25,),
            "mode_weights": ((mode, 1.0),),
            "variants_per_fraction": 1,
            "min_spawn_centers": 1,
            "max_attempts_per_variant": 30,
            "seed": 7,
        }
        values.update(overrides)
        return PartialCompletionConfig(**values)

    def test_fixed_seed_is_deterministic_and_builds_multi_height_piles(self):
        config = self._config("in_zone")
        first = generate_partial_action_map(
            self.target,
            self.occupancy,
            self.dumpability,
            rng=np.random.default_rng(11),
            config=config,
        )
        second = generate_partial_action_map(
            self.target,
            self.occupancy,
            self.dumpability,
            rng=np.random.default_rng(11),
            config=config,
        )
        different_seed = generate_partial_action_map(
            self.target,
            self.occupancy,
            self.dumpability,
            rng=np.random.default_rng(12),
            config=config,
        )

        np.testing.assert_array_equal(first.action_map, second.action_map)
        self.assertEqual(first.manifest, second.manifest)
        self.assertFalse(np.array_equal(first.action_map, different_seed.action_map))
        self.assertGreater(int(first.action_map.max()), 1)
        self.assertEqual(int(first.action_map.astype(np.int64).sum()), 0)

    def test_near_and_mixed_modes_preserve_their_support_contracts(self):
        for mode in ("near_zone", "mixed"):
            with self.subTest(mode=mode):
                config = self._config(mode)
                result = generate_partial_action_map(
                    self.target,
                    self.occupancy,
                    self.dumpability,
                    rng=np.random.default_rng(2),
                    config=config,
                )
                diagnostics = validate_partial_state(
                    self.target,
                    self.occupancy,
                    self.dumpability,
                    result.action_map,
                    config=config,
                    expected_mode=mode,
                )
                self.assertEqual(
                    diagnostics["positive_volume"],
                    diagnostics["negative_volume"],
                )
                self.assertLessEqual(
                    result.manifest["pile_count"],
                    config.max_piles,
                )

    def test_single_remaining_tile_is_an_actionable_partial_state(self):
        dig_target = np.zeros((20, 20), dtype=np.bool_)
        dig_target[10, 10:13] = True
        requested_count = 2
        selected = _select_completed_mask(
            dig_target,
            requested_count,
            np.random.default_rng(91),
        )

        self.assertEqual(int(np.count_nonzero(selected)), requested_count)
        self.assertEqual(int(np.count_nonzero(dig_target & ~selected)), 1)

    def test_singleton_remaining_component_is_valid(self):
        target = np.zeros((64, 64), dtype=np.int8)
        target[20, 20:23] = -1
        target[42:55, 40:55] = 1
        action = np.zeros_like(target, dtype=np.int8)
        action[20, 20:22] = -1
        action[45, 45] = 1
        action[45, 46] = 1
        config = self._config("in_zone")

        diagnostics = validate_partial_state(
            target,
            self.occupancy,
            self.dumpability,
            action,
            config=config,
            expected_mode="in_zone",
        )
        self.assertEqual(diagnostics["remaining_component_sizes"], [1])

    def test_relay_corridor_is_deterministic_and_keeps_pile_on_natural_route(self):
        target = np.zeros((64, 64), dtype=np.int8)
        target[10:22, 8:20] = -1
        target[42:54, 44:58] = 1
        config = self._config(
            "relay_corridor",
            completion_fractions=(0.90,),
            min_piles=1,
            max_piles=1,
            max_attempts_per_variant=100,
        )
        first = generate_partial_action_map(
            target,
            self.occupancy,
            self.dumpability,
            rng=np.random.default_rng(7),
            config=config,
        )
        second = generate_partial_action_map(
            target,
            self.occupancy,
            self.dumpability,
            rng=np.random.default_rng(7),
            config=config,
        )

        np.testing.assert_array_equal(first.action_map, second.action_map)
        self.assertEqual(first.manifest, second.manifest)
        self.assertGreater(first.manifest["relay_handoff_support_count"], 0)
        self.assertEqual(first.manifest["positive_component_count"], 1)
        self.assertLessEqual(
            first.manifest["piles"][0]["route_excess_tiles"],
            RELAY_CENTER_MAX_ROUTE_EXCESS_TILES,
        )
        self.assertEqual(first.manifest["minimum_bucket_loads_for_staged_volume"], 2)

        dynamic = compute_dynamic_dumpability_numpy(
            self.dumpability,
            np.where(first.action_map < 0, first.action_map, 0),
        )
        corridor, _, _ = _relay_corridor_masks(
            target,
            self.occupancy,
            dynamic,
            first.action_map < 0,
        )
        self.assertFalse(np.any((first.action_map > 0) & ~corridor))

    def test_relay_corridor_bends_through_an_obstacle_gap(self):
        target = np.zeros((64, 64), dtype=np.int8)
        target[20:28, 8:16] = -1
        target[20:28, 48:56] = 1
        occupancy = np.zeros_like(target, dtype=np.bool_)
        occupancy[:, 32] = True
        occupancy[28:36, 32] = False
        completed = target < 0
        action = np.where(completed, -1, 0).astype(np.int8)
        dynamic = compute_dynamic_dumpability_numpy(self.dumpability, action)

        corridor, _, corridor_data = _relay_corridor_masks(
            target,
            occupancy,
            dynamic,
            completed,
        )

        self.assertTrue(np.any(corridor[28:36, 30:35]))
        self.assertTrue(np.any(corridor_data["route_core"][28:36, 30:35]))
        self.assertFalse(corridor_data["route_core"][10, 30])

    def test_workspace_load_boundary_accepts_127_and_detects_128(self):
        offsets = (((0, 0),),)
        action = np.zeros((8, 8), dtype=np.int16)
        action[4, 4] = 127
        maximum, _ = _maximum_workspace_load(action, offsets)
        self.assertEqual(maximum, 127)

        action[4, 4] = 128
        maximum, _ = _maximum_workspace_load(action, offsets)
        self.assertEqual(maximum, 128)

    def test_runtime_sampling_domain_uses_first_row_column_bounds(self):
        open_domain = _runtime_sampling_domain(self.occupancy)
        self.assertTrue(open_domain[8, 8])
        self.assertFalse(open_domain[7, 8])

        blocked_border = self.occupancy.copy()
        blocked_border[:, 0] = True
        restricted_domain = _runtime_sampling_domain(blocked_border)
        self.assertFalse(np.any(restricted_domain))


if __name__ == "__main__":
    unittest.main()
