from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from terra.maps_buffer import validate_exact_dataset_contract
from terra.maps_buffer import RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
from terra.maps_buffer import reset_array_scenario_sha256
from tools.map_generation import materialize_loader_bank as loader

CONDITIONS = (
    ("fnd-anchor", "foundation", "0"),
    ("trn-anchor", "trench", "0"),
)
REQUESTED = {
    "train": 2,
    "promotion": 1,
    "development": 1,
    "sealed": 1,
}


def _arrays(sample_index: int) -> dict[str, np.ndarray]:
    target = np.zeros((8, 8), dtype=np.int8)
    target[3:5, 3:5] = -1
    target[0:2, 0:4] = 1
    occupancy = np.zeros_like(target)
    dumpability = np.ones_like(target)
    actions = np.zeros_like(target)
    distance = np.full(target.shape, sample_index / 100.0, dtype=np.float32)
    return {
        "images": target,
        "occupancy": occupancy,
        "dumpability": dumpability,
        "actions": actions,
        "distance": distance,
    }


def _write_split_bank(root: Path) -> Path:
    root.mkdir()
    conditions_summary = {
        condition: {
            split: {
                "pair_slots": count,
                "source_groups": count,
                "scenarios": count,
            }
            for split, count in REQUESTED.items()
        }
        for condition, _, _ in CONDITIONS
    }
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema": loader.SPLIT_BANK_SCHEMA,
                "requested_pair_slots_per_condition": REQUESTED,
                "conditions": conditions_summary,
                "assignment_sha256": "a" * 64,
            }
        )
        + "\n"
    )

    sample_index = 0
    for split, count in REQUESTED.items():
        dataset = root / split / "dataset"
        for folder in (
            *loader.ARRAY_FOLDERS,
            *loader.TRENCH_METADATA_ARRAY_FOLDERS,
            "metadata",
        ):
            (dataset / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for condition, family, tier in CONDITIONS:
            for local_index in range(count):
                arrays = _arrays(sample_index)
                for folder, array in arrays.items():
                    np.save(dataset / folder / f"img_{sample_index}.npy", array)
                np.save(
                    dataset
                    / "trench_axis_owners"
                    / f"img_{sample_index}.npy",
                    np.zeros_like(arrays["images"], dtype=np.uint8),
                )
                (dataset / "metadata" / f"trench_{sample_index}.json").write_text(
                    "{}\n"
                )
                rows.append(
                    {
                        "condition_id": condition,
                        "family": family,
                        "map_id": f"map-{sample_index}",
                        "pair_slot_id": f"{condition}:{local_index}:{split}",
                        "sample_index": str(sample_index),
                        "scenario_sha256": reset_array_scenario_sha256(arrays),
                        "source_group_id": f"source-{sample_index}",
                        "split": split,
                        "tier": tier,
                    }
                )
                sample_index += 1
        with (root / split / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_review_admission(root: Path, accepted=None) -> Path:
    path = root / "review_admission.json"
    path.write_text(
        json.dumps(
            {
                "schema": loader.REVIEW_ADMISSION_SCHEMA,
                "release": loader.REVIEW_RELEASE_ID,
                "manifest_sha256": loader.REVIEW_MANIFEST_SHA256,
                "review_data_sha256": loader.REVIEW_DATA_SHA256,
                "review_bundle_sha256": "d" * 64,
                "accepted_conditions": (
                    accepted
                    if accepted is not None
                    else sorted(condition for condition, _, _ in CONDITIONS)
                ),
            }
        )
        + "\n"
    )
    return path


def _materialize(split_bank: Path, output: Path, tmp_path: Path):
    return loader.materialize_loader_bank(
        split_bank,
        output,
        "terra-test-revision",
        _write_review_admission(tmp_path),
    )


def test_exact_reset_seeds_use_the_frozen_partitionable_prng_contract():
    loader.jax.config.update("jax_threefry_partitionable", False)
    seeds = loader._exact_reset_seeds(4)

    assert seeds == [1, 3, 0, 2]
    assert loader.jax.config.jax_default_prng_impl == "threefry2x32"
    assert bool(loader.jax.config.jax_threefry_partitionable) is True
    np.testing.assert_array_equal(
        loader._selected_map_indices(seeds, 4),
        np.arange(4),
    )


def test_materializes_exact_training_levels_and_evaluation_panels(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    output = tmp_path / "loader"
    review_admission = _write_review_admission(tmp_path)

    receipt = loader.materialize_loader_bank(
        split_bank,
        output,
        "terra-test-revision",
        review_admission,
    )

    assert receipt == json.loads((output / "dataset.json").read_text())
    assert receipt["scenario_identity_contract"] == (
        RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
    )
    assert [
        (row["condition_id"], row["family"], row["branch_depth"], row["map_count"])
        for row in receipt["train"]
    ] == [
        ("fnd-anchor", "foundation", "Anchor", 2),
        ("trn-anchor", "trench", "Anchor", 2),
    ]
    assert (output / receipt["source_registry"]).is_file()
    assert receipt["review_admission"] == "review_admission.json"
    assert receipt["review_admission_sha256"] == loader._sha256_file(review_admission)
    assert (output / receipt["review_admission"]).read_bytes() == (
        review_admission.read_bytes()
    )

    for level in receipt["train"]:
        directory = output / level["maps_path"]
        dataset = json.loads((directory / "dataset.json").read_text())
        assert dataset["scenario_identity_contract"] == (
            RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
        )
        rows, shape, _ = validate_exact_dataset_contract(
            directory,
            level["map_count"],
        )
        assert shape == (8, 8)
        assert all("reset_seed" not in row for row in rows)
        assert all("episode_id" not in row for row in rows)

    for split, expected_per_condition in REQUESTED.items():
        if split == "train":
            continue
        panel = receipt["evaluation_panels"][split]
        expected_count = expected_per_condition * len(CONDITIONS)
        rows, _, _ = validate_exact_dataset_contract(
            output / panel["maps_path"],
            expected_count,
        )
        assert [row["slot_index"] for row in rows] == list(range(1, expected_count + 1))
        assert all(0 <= row["reset_seed"] <= 2**32 - 1 for row in rows)
        assert len({row["episode_id"] for row in rows}) == expected_count
        np.testing.assert_array_equal(
            loader._selected_map_indices(
                [row["reset_seed"] for row in rows],
                expected_count,
            ),
            np.arange(expected_count),
        )
        for row in rows:
            assert row["episode_id"] == loader.episode_id(
                row["scenario_id"],
                row["reset_seed"],
                receipt["environment_protocol_sha256"],
            )


def test_materialization_is_deterministic(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    first = tmp_path / "first"
    second = tmp_path / "second"
    _materialize(split_bank, first, tmp_path)
    _materialize(split_bank, second, tmp_path)

    for relative in (
        "dataset.json",
        "environment_protocol.json",
        "source_registry.jsonl",
        "promotion/manifest.jsonl",
        "development/manifest.jsonl",
        "sealed/manifest.jsonl",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_rejects_review_admission_condition_mismatch(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    review_admission = _write_review_admission(tmp_path, ["fnd-anchor"])
    with pytest.raises(ValueError, match="do not match the split bank"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "unused",
            "terra-test-revision",
            review_admission,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("release", "stale-release"),
        ("manifest_sha256", "0" * 64),
        ("review_data_sha256", "1" * 64),
    ),
)
def test_rejects_stale_review_admission_identity(tmp_path, field, value):
    split_bank = _write_split_bank(tmp_path / "split")
    review_admission = _write_review_admission(tmp_path)
    receipt = json.loads(review_admission.read_text())
    receipt[field] = value
    review_admission.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ValueError, match=f"{field} does not match"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "unused",
            "terra-test-revision",
            review_admission,
        )


def test_rejects_review_only_banks(tmp_path):
    review_only = tmp_path / "review"
    review_only.mkdir()
    with pytest.raises(ValueError, match="Review-only or unsplit"):
        _materialize(review_only, tmp_path / "unused", tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("source_group_id", "source leakage"),
        ("pair_slot_id", "pair-slot leakage"),
    ),
)
def test_rejects_cross_split_source_and_pair_leakage(
    tmp_path,
    field,
    message,
):
    split_bank = _write_split_bank(tmp_path / "split")
    train_rows = _csv(split_bank / "train" / "manifest.csv")
    promotion_manifest = split_bank / "promotion" / "manifest.csv"
    promotion_rows = _csv(promotion_manifest)
    promotion_rows[0][field] = train_rows[0][field]
    _write_csv(promotion_manifest, promotion_rows)
    with pytest.raises(ValueError, match=message):
        _materialize(split_bank, tmp_path / "leaking", tmp_path)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("pair_slot_id", "observed distinct pair_slots=1"),
        ("source_group_id", "observed distinct source_groups=1"),
    ),
)
def test_rejects_duplicate_pair_or_source_with_unchanged_row_count(
    tmp_path,
    field,
    message,
):
    split_bank = _write_split_bank(tmp_path / "split")
    train_manifest = split_bank / "train" / "manifest.csv"
    rows = _csv(train_manifest)
    rows[1][field] = rows[0][field]
    _write_csv(train_manifest, rows)

    with pytest.raises(ValueError, match=message):
        _materialize(split_bank, tmp_path / "duplicate-identity", tmp_path)


def test_rejects_per_condition_count_mismatch(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    train_manifest = split_bank / "train" / "manifest.csv"
    rows = _csv(train_manifest)
    rows = [
        row
        for row in rows
        if not (row["condition_id"] == "fnd-anchor" and row["map_id"] == "map-0")
    ]
    _write_csv(train_manifest, rows)

    with pytest.raises(ValueError, match="condition counts do not match"):
        _materialize(split_bank, tmp_path / "count-mismatch", tmp_path)


def test_rejects_array_identity_and_truncated_seed_hash_collisions(
    tmp_path,
    monkeypatch,
):
    split_bank = _write_split_bank(tmp_path / "split")
    promotion_manifest = split_bank / "promotion" / "manifest.csv"
    with promotion_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["scenario_sha256"] = "b" * 64
    _write_csv(promotion_manifest, rows)
    with pytest.raises(ValueError, match="scenario hash mismatch"):
        _materialize(split_bank, tmp_path / "bad-identity", tmp_path)

    split_bank = _write_split_bank(tmp_path / "second-split")
    monkeypatch.setattr(
        loader,
        "_exact_reset_seeds",
        lambda count: [7] * count,
    )
    with pytest.raises(ValueError, match="reset-seed hash collision"):
        _materialize(split_bank, tmp_path / "seed-collision", tmp_path)


def test_consumer_rejects_published_array_and_manifest_identity_mutation(
    tmp_path,
):
    split_bank = _write_split_bank(tmp_path / "split")
    array_output = tmp_path / "array-output"
    _materialize(split_bank, array_output, tmp_path)
    image_path = array_output / "promotion" / "images" / "img_1.npy"
    target = np.load(image_path)
    target[-1, -1] = -1
    np.save(image_path, target)
    with pytest.raises(RuntimeError, match="Scenario identity mismatch"):
        validate_exact_dataset_contract(array_output / "promotion", 2)

    manifest_output = tmp_path / "manifest-output"
    _materialize(split_bank, manifest_output, tmp_path)
    manifest_path = manifest_output / "promotion" / "manifest.jsonl"
    rows = _jsonl(manifest_path)
    rows[0]["scenario_id"] = "b" * 64
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(RuntimeError, match="provenance does not match"):
        validate_exact_dataset_contract(manifest_output / "promotion", 2)
