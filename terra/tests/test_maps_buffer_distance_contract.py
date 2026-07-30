import hashlib
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
from terra.config import CurriculumGlobalConfig
from terra.config import EnvConfig
from terra.config import RewardsType
from terra.maps_buffer import init_maps_buffer
from terra.maps_buffer import LEGACY_SCENARIO_IDENTITY_CONTRACT
from terra.maps_buffer import load_maps_from_disk
from terra.maps_buffer import MapsBuffer


class MapsBufferDistanceContractTest(unittest.TestCase):
    @staticmethod
    def write_contract(root: Path, count: int = 1):
        registry = root / "source_registry.jsonl"
        registry.write_text(
            "\n".join(
                json.dumps(
                    {
                        "map_id": f"test-map-{index}",
                        "source_id": f"test-source-{index}",
                        "split": "test",
                    },
                    sort_keys=True,
                )
                for index in range(1, count + 1)
            )
            + "\n"
        )
        registry_sha256 = hashlib.sha256(registry.read_bytes()).hexdigest()
        (root / "manifest.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "slot_index": index,
                        "map_id": f"test-map-{index}",
                        "source_id": f"test-source-{index}",
                        "split": "test",
                        "family": "foundation",
                        "stratum": "fixture",
                        "primary_cell": "fixture",
                        "slot_weight": 1.0,
                        "identity_slot_multiplicity": 1,
                    },
                    sort_keys=True,
                )
                for index in range(1, count + 1)
            )
            + "\n"
        )
        (root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema": "terra_exact_map_dataset_v1",
                    "slot_count": count,
                    "unique_identity_count": count,
                    "shape": [64, 64],
                    "distance_metric": "fixture_geodesic",
                    "distance_normalization": "fixture_unit_interval",
                    "accepted_dump_contract": "exact_visible_dump_v1",
                    "scenario_identity_contract": LEGACY_SCENARIO_IDENTITY_CONTRACT,
                    "source_registry": "source_registry.jsonl",
                    "source_registry_sha256": registry_sha256,
                },
                sort_keys=True,
            )
            + "\n"
        )

    @staticmethod
    def write_map(root: Path, distance=None):
        for folder in (
            "images",
            "occupancy",
            "dumpability",
            "actions",
            "distance",
            "metadata",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
        target = np.zeros((64, 64), dtype=np.int8)
        target[30:34, 30:34] = -1
        np.save(root / "images" / "img_1.npy", target)
        np.save(
            root / "occupancy" / "img_1.npy",
            np.zeros_like(target, dtype=np.int8),
        )
        np.save(
            root / "dumpability" / "img_1.npy",
            np.ones_like(target, dtype=np.int8),
        )
        np.save(
            root / "actions" / "img_1.npy",
            np.zeros_like(target, dtype=np.int8),
        )
        (root / "metadata" / "trench_1.json").write_text(
            json.dumps({"axes_ABC": []}) + "\n"
        )
        if distance is not None:
            np.save(root / "distance" / "img_1.npy", distance)
        MapsBufferDistanceContractTest.write_contract(root)

    def test_missing_distance_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(root)
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(
                    RuntimeError, "Dataset sidecars in .*distance"
                ):
                    load_maps_from_disk(str(root))

    def test_nonfinite_or_unnormalized_distance_fails_loudly(self):
        for value, message in (
            (np.nan, "non-finite"),
            (1.1, r"normalized to \[0, 1\]"),
        ):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    distance = np.zeros((64, 64), dtype=np.float32)
                    distance[0, 0] = value
                    self.write_map(root, distance)
                    with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                        with self.assertRaisesRegex(RuntimeError, message):
                            load_maps_from_disk(str(root))

    def test_wrong_shape_or_dtype_fails_loudly(self):
        cases = (
            (
                np.zeros((32, 32), dtype=np.float32),
                "shape mismatch",
            ),
            (
                np.zeros((64, 64), dtype=np.int16),
                "floating dtype",
            ),
        )
        for distance, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.write_map(root, distance)
                    with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                        with self.assertRaisesRegex(RuntimeError, message):
                            load_maps_from_disk(str(root))

    def test_exact_contract_rejects_count_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(
                root,
                np.zeros((64, 64), dtype=np.float32),
            )
            with patch.dict(os.environ, {"DATASET_SIZE": "2"}):
                with self.assertRaisesRegex(RuntimeError, "slot count mismatch"):
                    load_maps_from_disk(str(root))

            rows = [
                json.loads(line)
                for line in (root / "manifest.jsonl").read_text().splitlines()
            ]
            rows[0]["identity_slot_multiplicity"] = 2
            (root / "manifest.jsonl").write_text(
                json.dumps(rows[0], sort_keys=True) + "\n"
            )
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "declares multiplicity"):
                    load_maps_from_disk(str(root))

    def test_exact_contract_requires_an_explicit_identity_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(
                root,
                np.zeros((64, 64), dtype=np.float32),
            )
            metadata = json.loads((root / "dataset.json").read_text())
            metadata.pop("scenario_identity_contract")
            (root / "dataset.json").write_text(
                json.dumps(metadata, sort_keys=True) + "\n"
            )
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "explicitly declare scenario_identity_contract",
                ):
                    load_maps_from_disk(str(root))

    def test_exact_contract_rejects_cross_split_source_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(
                root,
                np.zeros((64, 64), dtype=np.float32),
            )
            registry = root / "source_registry.jsonl"
            registry.write_text(
                registry.read_text()
                + json.dumps(
                    {
                        "map_id": "other-map",
                        "source_id": "test-source-1",
                        "split": "development",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            metadata = json.loads((root / "dataset.json").read_text())
            metadata["source_registry_sha256"] = hashlib.sha256(
                registry.read_bytes()
            ).hexdigest()
            (root / "dataset.json").write_text(
                json.dumps(metadata, sort_keys=True) + "\n"
            )
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "not split-disjoint"):
                    load_maps_from_disk(str(root))

    def test_exact_contract_enforces_declared_capacity_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(
                root,
                np.zeros((64, 64), dtype=np.float32),
            )
            target_path = root / "images" / "img_1.npy"
            target = np.load(target_path)
            target[0:2, 0:8] = 1
            target[4:6, 0:8] = 1
            np.save(target_path, target)
            metadata = json.loads((root / "dataset.json").read_text())
            metadata["minimum_dump_capacity_ratio"] = 3.0
            (root / "dataset.json").write_text(
                json.dumps(metadata, sort_keys=True) + "\n"
            )
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(
                    RuntimeError, "single-layer capacity ratio"
                ):
                    load_maps_from_disk(str(root))

    def test_init_exposes_exact_manifest_family_and_cell_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "fixture"
            self.write_map(
                dataset_root,
                np.zeros((64, 64), dtype=np.float32),
            )

            class FixtureCurriculum(CurriculumGlobalConfig):
                levels = [
                    {
                        "maps_path": "fixture",
                        "max_steps_in_episode": 450,
                        "rewards_type": RewardsType.DENSE,
                        "apply_trench_rewards": False,
                    }
                ]

            batch_cfg = BatchConfig(
                curriculum_global=FixtureCurriculum()
            )
            with patch.dict(
                os.environ,
                {
                    "DATASET_PATH": temporary,
                    "DATASET_SIZE": "1",
                },
            ):
                buffer, _ = init_maps_buffer(
                    batch_cfg,
                    shuffle_maps=False,
                )

            self.assertEqual(buffer.family_names, ("unknown", "foundation"))
            self.assertEqual(
                buffer.primary_cell_names,
                ("unknown", "fixture"),
            )
            np.testing.assert_array_equal(buffer.slot_indices, [[0]])
            np.testing.assert_array_equal(buffer.family_ids, [[1]])
            np.testing.assert_array_equal(buffer.primary_cell_ids, [[1]])

    def test_map_and_provenance_selection_share_the_exact_rng_path(self):
        count = 4
        maps = jnp.arange(count, dtype=jnp.int8).reshape(1, count, 1, 1)
        zeros_map = jnp.zeros_like(maps)
        zeros_axes = jnp.zeros((1, count, 3, 3), dtype=jnp.float32)
        zeros_foundation_axes = jnp.zeros(
            (1, count, 64, 3),
            dtype=jnp.float32,
        )
        zeros_types = jnp.zeros((1, count), dtype=jnp.int32)
        buffer = MapsBuffer.new(
            maps=maps,
            padding_mask=zeros_map,
            trench_axes=zeros_axes,
            trench_types=zeros_types,
            foundation_border_axes=zeros_foundation_axes,
            foundation_border_types=zeros_types,
            dumpability_masks_init=jnp.ones_like(maps, dtype=jnp.bool_),
            action_maps=zeros_map,
            distance_maps=zeros_map.astype(jnp.float32),
            slot_indices=jnp.arange(count, dtype=jnp.int32)[None, :],
            family_ids=(10 + jnp.arange(count, dtype=jnp.int32))[None, :],
            primary_cell_ids=(
                20 + jnp.arange(count, dtype=jnp.int32)
            )[None, :],
        )
        env_cfg = EnvConfig()
        for seed in range(8):
            key = jax.random.PRNGKey(seed)
            selected_map = buffer.get_map(key, env_cfg)[0]
            slot, family, cell, _ = buffer.get_map_provenance(
                key,
                env_cfg,
            )
            selected = int(np.asarray(selected_map)[0, 0])
            self.assertEqual(int(slot), selected)
            self.assertEqual(int(family), 10 + selected)
            self.assertEqual(int(cell), 20 + selected)


if __name__ == "__main__":
    unittest.main()
