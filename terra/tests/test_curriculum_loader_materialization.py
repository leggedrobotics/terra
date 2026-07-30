from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from terra.maps_buffer import validate_exact_dataset_contract
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


def _scenario_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in loader.ARRAY_FOLDERS:
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


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
        for folder in (*loader.ARRAY_FOLDERS, "metadata"):
            (dataset / folder).mkdir(parents=True, exist_ok=True)
        rows = []
        for condition, family, tier in CONDITIONS:
            for local_index in range(count):
                arrays = _arrays(sample_index)
                for folder, array in arrays.items():
                    np.save(dataset / folder / f"img_{sample_index}.npy", array)
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
                        "scenario_sha256": _scenario_sha256(arrays),
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


def test_materializes_exact_training_levels_and_evaluation_panels(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    output = tmp_path / "loader"

    receipt = loader.materialize_loader_bank(
        split_bank,
        output,
        "terra-test-revision",
    )

    assert receipt == json.loads((output / "dataset.json").read_text())
    assert [
        (row["condition_id"], row["family"], row["branch_depth"], row["map_count"])
        for row in receipt["train"]
    ] == [
        ("fnd-anchor", "foundation", "Anchor", 2),
        ("trn-anchor", "trench", "Anchor", 2),
    ]
    assert (output / receipt["source_registry"]).is_file()

    for level in receipt["train"]:
        directory = output / level["maps_path"]
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
    loader.materialize_loader_bank(split_bank, first, "terra-test-revision")
    loader.materialize_loader_bank(split_bank, second, "terra-test-revision")

    for relative in (
        "dataset.json",
        "environment_protocol.json",
        "source_registry.jsonl",
        "promotion/manifest.jsonl",
        "development/manifest.jsonl",
        "sealed/manifest.jsonl",
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_rejects_review_only_and_source_leaking_banks(tmp_path):
    review_only = tmp_path / "review"
    review_only.mkdir()
    with pytest.raises(ValueError, match="Review-only or unsplit"):
        loader.materialize_loader_bank(
            review_only,
            tmp_path / "unused",
            "terra-test-revision",
        )

    split_bank = _write_split_bank(tmp_path / "split")
    promotion_manifest = split_bank / "promotion" / "manifest.csv"
    with promotion_manifest.open(newline="") as handle:
        promotion_rows = list(csv.DictReader(handle))
    promotion_rows[0]["source_group_id"] = "source-0"
    with promotion_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(promotion_rows[0]))
        writer.writeheader()
        writer.writerows(promotion_rows)
    with pytest.raises(ValueError, match="source leakage"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "leaking",
            "terra-test-revision",
        )


def test_rejects_per_condition_count_mismatch(tmp_path):
    split_bank = _write_split_bank(tmp_path / "split")
    train_manifest = split_bank / "train" / "manifest.csv"
    with train_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if not (row["condition_id"] == "fnd-anchor" and row["map_id"] == "map-0")
    ]
    with train_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="condition counts do not match"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "count-mismatch",
            "terra-test-revision",
        )


def test_rejects_array_identity_and_truncated_seed_hash_collisions(
    tmp_path,
    monkeypatch,
):
    split_bank = _write_split_bank(tmp_path / "split")
    promotion_manifest = split_bank / "promotion" / "manifest.csv"
    with promotion_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["scenario_sha256"] = "b" * 64
    with promotion_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="scenario hash mismatch"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "bad-identity",
            "terra-test-revision",
        )

    split_bank = _write_split_bank(tmp_path / "second-split")
    monkeypatch.setattr(
        loader,
        "_exact_reset_seeds",
        lambda count: [7] * count,
    )
    with pytest.raises(ValueError, match="reset-seed hash collision"):
        loader.materialize_loader_bank(
            split_bank,
            tmp_path / "seed-collision",
            "terra-test-revision",
        )
