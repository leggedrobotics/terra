from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import terra.env_generation.partial_reset_bank as partial_reset_bank_module
from terra.config import EnvConfig
from terra.env import TerraEnv
from terra.env import TerraEnvBatch
from terra.env_generation.partial_completion import PartialCompletionError
from terra.env_generation.partial_reset_bank import _declared_training_leaves
from terra.env_generation.partial_reset_bank import materialize_sparse_partial_reset_bank
from terra.maps_buffer import MapsBuffer
from terra.maps_buffer import PARTIAL_COMPLETION_CONFIG
from terra.maps_buffer import PARTIAL_COMPLETION_MANIFEST
from terra.maps_buffer import PARTIAL_COMPLETION_REJECTIONS
from terra.maps_buffer import PARTIAL_RESET_BANK_INDEX
from terra.maps_buffer import PARTIAL_RESET_BANK_SCHEMA
from terra.maps_buffer import PARTIAL_RESET_LEAF_SCHEMA
from terra.maps_buffer import PARTIAL_RESET_TRIPLET_CONTRACT
from terra.maps_buffer import load_partial_reset_action_sidecars
from terra.maps_buffer import partial_reset_action_sanity_check
from terra.maps_buffer import partial_reset_bank_sha256
from terra.state import State


SHAPE = (16, 16)


def _canonical_layers():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[2, 2:12] = -1
    target[11:15, 11:15] = 1
    maps = np.stack((target, target))
    occupancies = np.zeros_like(maps, dtype=np.int8)
    dumpability = np.ones_like(maps, dtype=np.bool_)
    return maps, occupancies, dumpability


def _partial_action(target: np.ndarray, fraction: float) -> np.ndarray:
    action = np.zeros_like(target, dtype=np.int8)
    completed = int(round(fraction * int(np.count_nonzero(target < 0))))
    for x, y in np.argwhere(target < 0)[:completed]:
        action[int(x), int(y)] = -1
    action[12, 12] = completed
    return action


def _write_sparse_bank(
    root: Path,
    *,
    include_tier3: bool = True,
    pile_mode: str = "relay_corridor",
    pile_mode_policy: tuple[str, ...] | None = None,
):
    if pile_mode_policy is None:
        pile_mode_policy = (pile_mode,)
    maps, occupancies, dumpability = _canonical_layers()
    leaf = root / "condition"
    actions_dir = leaf / "actions"
    actions_dir.mkdir(parents=True)
    rows = []
    # A sparse row means a complete triplet for one canonical source.
    variants = [(1, 0.90, 1), (2, 0.75, 1)]
    if include_tier3:
        variants.append((3, 0.50, 1))
    for sidecar_index, (tier, fraction, source_index) in enumerate(
        variants,
        start=1,
    ):
        action = _partial_action(maps[source_index - 1], fraction)
        action_path = actions_dir / f"img_{sidecar_index}.npy"
        np.save(action_path, action)
        rows.append(
            {
                "sidecar_index": sidecar_index,
                "maps_path": "condition",
                "source_index": source_index,
                "source_map_id": f"map-{source_index}",
                "source_scenario_id": str(source_index) * 64,
                "reset_tier": tier,
                "variant_seed": 1234,
                "requested_completion_fraction": fraction,
                "achieved_completion_fraction": (
                    int(np.count_nonzero(action < 0))
                    / int(np.count_nonzero(maps[source_index - 1] < 0))
                ),
                "pile_mode": pile_mode,
                "source_triplet_contract": PARTIAL_RESET_TRIPLET_CONTRACT,
                "action_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
            }
        )
    (leaf / PARTIAL_COMPLETION_CONFIG).write_text(
        json.dumps(
            {
                "schema": PARTIAL_RESET_LEAF_SCHEMA,
                "maps_path": "condition",
                "pile_mode_policy": list(pile_mode_policy),
                "source_triplet_contract": PARTIAL_RESET_TRIPLET_CONTRACT,
                "completion_fractions": [0.90, 0.75, 0.50],
                "canonical_slot_count": 2,
                "successful_variant_count": len(rows),
            },
            sort_keys=True,
        )
        + "\n"
    )
    (leaf / PARTIAL_COMPLETION_MANIFEST).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (leaf / PARTIAL_COMPLETION_REJECTIONS).write_text("")
    (root / PARTIAL_COMPLETION_REJECTIONS).write_text("")
    digest = partial_reset_bank_sha256(root)
    (root / PARTIAL_RESET_BANK_INDEX).write_text(
        json.dumps(
            {
                "schema": PARTIAL_RESET_BANK_SCHEMA,
                "source_triplet_contract": PARTIAL_RESET_TRIPLET_CONTRACT,
                "pile_mode_policy": list(pile_mode_policy),
                "supported_maps_paths": ["condition"],
                "bank_sha256": digest,
            },
            sort_keys=True,
        )
        + "\n"
    )
    canonical_rows = [
        {
            "map_id": f"map-{index}",
            "scenario_id": str(index) * 64,
        }
        for index in (1, 2)
    ]
    return maps, occupancies, dumpability, canonical_rows, digest


def test_materializer_uses_one_ordered_fallback_mode_for_the_whole_triplet(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "source"
    source_leaf = source_root / "train" / "condition"
    source_leaf.mkdir(parents=True)
    (source_root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "terra_curriculum_loader_bank_v1",
                "train": [
                    {
                        "level_index": 0,
                        "maps_path": "train/condition",
                    }
                ],
            }
        )
        + "\n"
    )
    (source_leaf / "dataset.json").write_text(
        json.dumps(
            {
                "scenario_identity_contract": "terra_reset_arrays_sha256_v1",
                "slot_count": 1,
            }
        )
        + "\n"
    )
    (source_leaf / "manifest.jsonl").write_text("fixture\n")
    canonical_row = {
        "map_id": "map-1",
        "scenario_id": "1" * 64,
    }
    target = np.zeros((64, 64), dtype=np.int8)
    target[2, 2:12] = -1
    target[48:52, 48:52] = 1
    occupancy = np.zeros_like(target, dtype=np.int8)
    dumpability = np.ones_like(target, dtype=np.bool_)

    monkeypatch.setattr(
        partial_reset_bank_module,
        "validate_exact_dataset_contract",
        lambda *_: ([canonical_row], (64, 64), None),
    )
    monkeypatch.setattr(
        partial_reset_bank_module,
        "_load_source_layers",
        lambda *_: (target, occupancy, dumpability),
    )

    def generate(*_, config, **__):
        mode = config.mode_weights[0][0]
        if mode == "relay_corridor":
            raise PartialCompletionError("fixture has no relay corridor")
        fraction = config.completion_fractions[0]
        completed = int(round(fraction * 10))
        action = np.zeros_like(target, dtype=np.int8)
        for x, y in np.argwhere(target < 0)[:completed]:
            action[int(x), int(y)] = -1
        action[49, 49] = completed
        return SimpleNamespace(
            action_map=action,
            manifest={
                "requested_completion_fraction": fraction,
                "achieved_completion_fraction": completed / 10,
                "pile_mode": mode,
            },
        )

    monkeypatch.setattr(
        partial_reset_bank_module,
        "generate_partial_action_map",
        generate,
    )

    output_root = tmp_path / "partial"
    receipt = materialize_sparse_partial_reset_bank(
        source_root,
        output_root,
        pile_modes=("relay_corridor", "in_zone"),
        include_maps_paths=("train/condition",),
        min_spawn_centers=1,
        max_source_triplets_per_condition=1,
        max_sources_scanned_per_condition=1,
    )
    assert receipt["pile_mode_policy"] == ["relay_corridor", "in_zone"]
    assert receipt["supported_maps_paths"] == ["train/condition"]
    assert receipt["max_source_triplets_per_condition"] == 1
    assert receipt["max_sources_scanned_per_condition"] == 1
    config = json.loads(
        (
            output_root
            / "train"
            / "condition"
            / PARTIAL_COMPLETION_CONFIG
        ).read_text()
    )
    assert config["selected_pile_mode_counts"] == {
        "relay_corridor": 0,
        "in_zone": 1,
    }
    rows = [
        json.loads(line)
        for line in (
            output_root
            / "train"
            / "condition"
            / PARTIAL_COMPLETION_MANIFEST
        ).read_text().splitlines()
    ]
    assert [row["reset_tier"] for row in rows] == [1, 2, 3]
    assert {row["pile_mode"] for row in rows} == {"in_zone"}
    rejected = (
        output_root
        / "train"
        / "condition"
        / PARTIAL_COMPLETION_REJECTIONS
    ).read_text()
    assert "fixture has no relay corridor" in rejected

    maps = np.stack((target,))
    occupancies = np.stack((occupancy,))
    dumpability_maps = np.stack((dumpability,))
    _, available, _, supported = load_partial_reset_action_sidecars(
        output_root,
        ["train/condition"],
        [maps],
        [occupancies],
        [dumpability_maps],
        [[canonical_row]],
    )
    assert available[:, 0, 0].tolist() == [True, True, True]
    assert supported[:, 0].tolist() == [True, True, True, True]


def test_sparse_sidecar_is_source_bound_and_samples_only_available_slots():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        maps, occupancies, dumpability, canonical_rows, digest = _write_sparse_bank(
            root
        )
        actions, available, observed_digest, supported = (
            load_partial_reset_action_sidecars(
                root,
                ["condition"],
                [maps],
                [occupancies],
                [dumpability],
                [canonical_rows],
            )
        )

        assert observed_digest == digest
        assert actions.shape == (3, 1, 2, *SHAPE)
        np.testing.assert_array_equal(
            available[:, 0],
            np.array([[True, False], [True, False], [True, False]]),
        )
        np.testing.assert_array_equal(
            supported,
            np.array([[True], [True], [True], [True]]),
        )

        zeros_axes = jnp.zeros((1, 2, 4, 3), dtype=jnp.float32)
        zeros_foundation_axes = jnp.zeros((1, 2, 64, 3), dtype=jnp.float32)
        buffer = MapsBuffer.new(
            maps=jnp.asarray(maps[None]),
            padding_mask=jnp.asarray(occupancies[None]),
            trench_axes=zeros_axes,
            trench_types=jnp.zeros((1, 2), dtype=jnp.int32),
            trench_axis_owners=jnp.zeros((1, 2, *SHAPE), dtype=jnp.uint8),
            foundation_border_axes=zeros_foundation_axes,
            foundation_border_types=jnp.zeros((1, 2), dtype=jnp.int32),
            dumpability_masks_init=jnp.asarray(dumpability[None]),
            action_maps=jnp.zeros((1, 2, *SHAPE), dtype=jnp.int8),
            distance_maps=jnp.zeros((1, 2, *SHAPE), dtype=jnp.float32),
            partial_action_maps=jnp.asarray(actions),
            partial_action_available=jnp.asarray(available),
            partial_reset_supported_levels=jnp.asarray(supported),
            partial_reset_bank_sha256=digest,
        )
        full_only_buffer = MapsBuffer.new(
            maps=jnp.asarray(maps[None]),
            padding_mask=jnp.asarray(occupancies[None]),
            trench_axes=zeros_axes,
            trench_types=jnp.zeros((1, 2), dtype=jnp.int32),
            trench_axis_owners=jnp.zeros((1, 2, *SHAPE), dtype=jnp.uint8),
            foundation_border_axes=zeros_foundation_axes,
            foundation_border_types=jnp.zeros((1, 2), dtype=jnp.int32),
            dumpability_masks_init=jnp.asarray(dumpability[None]),
            action_maps=jnp.zeros((1, 2, *SHAPE), dtype=jnp.int8),
            distance_maps=jnp.zeros((1, 2, *SHAPE), dtype=jnp.float32),
        )
        full_key = jax.random.PRNGKey(81)
        with_sidecar_full = buffer.sample_map(full_key, EnvConfig())
        without_sidecar_full = full_only_buffer.sample_map(full_key, EnvConfig())
        for observed, expected in zip(with_sidecar_full, without_sidecar_full):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))
        expected_key, expected_subkey = jax.random.split(full_key)
        expected_index = jax.random.randint(expected_subkey, (), 0, 2)
        _, index_only_selection, index_only_key = MapsBuffer._select_index(
            SimpleNamespace(n_maps=2),
            full_key,
            EnvConfig(),
        )
        np.testing.assert_array_equal(index_only_selection, expected_index)
        np.testing.assert_array_equal(index_only_key, expected_key)

        lane_keys = jax.random.split(jax.random.PRNGKey(82), 4)
        lane_tiers = jnp.arange(4, dtype=jnp.int32)

        def sample_device(keys, tiers):
            return jax.vmap(
                lambda key, tier: buffer.sample_map(
                    key,
                    EnvConfig()._replace(reset_tier=tier),
                )
            )(keys, tiers)

        pmapped = jax.pmap(sample_device)(lane_keys[None], lane_tiers[None])
        pmapped_actions = np.asarray(pmapped[8][0])
        np.testing.assert_array_equal(pmapped_actions[0], np.zeros(SHAPE))
        for tier in (1, 2, 3):
            np.testing.assert_array_equal(
                pmapped_actions[tier],
                actions[tier - 1, 0, 0],
            )
        batch_stub = SimpleNamespace(partial_reset_supported_levels=supported)
        with pytest.raises(RuntimeError, match="reset_tier must lie"):
            TerraEnvBatch.validate_reset_tiers(
                batch_stub,
                EnvConfig()._replace(reset_tier=4),
            )

        for tier, expected_source in ((1, 0), (2, 0), (3, 0)):
            selected = buffer.sample_map(
                jax.random.PRNGKey(9),
                EnvConfig()._replace(reset_tier=tier),
            )
            np.testing.assert_array_equal(
                np.asarray(selected[8]),
                actions[tier - 1, 0, expected_source],
            )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        maps, occupancies, dumpability, canonical_rows, _ = _write_sparse_bank(
            root,
            include_tier3=False,
        )
        with pytest.raises(RuntimeError, match="complete source triplets"):
            load_partial_reset_action_sidecars(
                root,
                ["condition"],
                [maps],
                [occupancies],
                [dumpability],
                [canonical_rows],
            )

    with tempfile.TemporaryDirectory() as temporary:
        accepted_root = Path(temporary)
        declared_paths = ["train/second", "train/first"]
        for maps_path in declared_paths + ["sealed"]:
            directory = accepted_root / maps_path
            directory.mkdir(parents=True)
            (directory / "dataset.json").write_text("{}\n")
        (accepted_root / "dataset.json").write_text(
            json.dumps(
                {
                    "schema": "terra_curriculum_loader_bank_v1",
                    "train": [
                        {"level_index": index, "maps_path": maps_path}
                        for index, maps_path in enumerate(declared_paths)
                    ],
                    "evaluation_panels": {"sealed": {"maps_path": "sealed"}},
                }
            )
            + "\n"
        )
        assert [
            maps_path
            for maps_path, _ in _declared_training_leaves(accepted_root)
        ] == declared_paths


def test_partial_contract_rejects_mass_error_and_source_identity_drift():
    maps, occupancies, dumpability = _canonical_layers()
    invalid = _partial_action(maps[0], 0.90)
    invalid[12, 12] -= 1
    with pytest.raises(RuntimeError, match="mass conservation"):
        partial_reset_action_sanity_check(
            maps[0],
            occupancies[0],
            dumpability[0],
            invalid,
            expected_fraction=0.90,
        )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        maps, occupancies, dumpability, canonical_rows, _ = _write_sparse_bank(root)
        canonical_rows[0]["scenario_id"] = "f" * 64
        with pytest.raises(RuntimeError, match="source identity"):
            load_partial_reset_action_sidecars(
                root,
                ["condition"],
                [maps],
                [occupancies],
                [dumpability],
                [canonical_rows],
            )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        maps, occupancies, dumpability, canonical_rows, _ = _write_sparse_bank(root)
        with (root / "condition" / "actions" / "img_1.npy").open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(RuntimeError, match="bank digest mismatch"):
            load_partial_reset_action_sidecars(
                root,
                ["condition"],
                [maps],
                [occupancies],
                [dumpability],
                [canonical_rows],
            )


def test_reset_tier_is_latched_until_the_next_reset():
    target = np.zeros((64, 64), dtype=np.int8)
    target[20, 20:24] = -1
    distance = np.ones_like(target, dtype=np.float32)
    partial = np.zeros_like(target, dtype=np.int8)
    partial[20, 20:22] = -1
    partial[40, 40] = 2
    base_cfg = EnvConfig()
    partial_cfg = base_cfg._replace(
        max_steps_in_episode=1,
        reset_tier=2,
        tile_size=36.5714285714 / 64,
        agent=base_cfg.agent._replace(width=7, height=11),
        maps=base_cfg.maps._replace(edge_length_px=64),
    )
    state = State.new(
        jax.random.PRNGKey(4),
        partial_cfg,
        target,
        np.zeros_like(target),
        -97.0 * np.ones((4, 3), dtype=np.float32),
        np.int32(-1),
        np.zeros_like(target, dtype=np.uint8),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones_like(target, dtype=np.bool_),
        partial,
        distance_map_override=distance,
    )
    assert int(state.reset_tier) == 2
    q, _, p = state._reward_v2_progress()
    np.testing.assert_array_equal(np.asarray([q, p]), np.zeros((2,), dtype=np.float32))
    np.testing.assert_array_equal(
        np.asarray(TerraEnv._state_to_obs_dict(state)["reward_v2_reset_context"]),
        np.asarray(
            [
                state.material_q_reset,
                state.material_h_reset / state._required_excavation_volume(),
            ],
            dtype=np.float32,
        ),
    )

    full_cfg = partial_cfg._replace(reset_tier=0)
    scheduled = state._replace(env_cfg=full_cfg)
    assert int(scheduled.reset_tier) == 2
    reset = scheduled._reset(
        full_cfg,
        target,
        np.zeros_like(target),
        -97.0 * np.ones((4, 3), dtype=np.float32),
        np.int32(-1),
        np.zeros_like(target, dtype=np.uint8),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        np.ones_like(target, dtype=np.bool_),
        np.zeros_like(target),
        distance_map_override=distance,
    )
    assert int(reset.reset_tier) == 0
