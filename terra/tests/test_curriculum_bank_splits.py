import csv
from pathlib import Path

import numpy as np
import pytest

from tools.map_generation import materialize_splits as splits


def _row(
    condition: str,
    sample_index: int,
    source_group: str,
    pair_slot: str | None = None,
    dig_sha256: str | None = None,
) -> dict[str, str]:
    return {
        "condition_id": condition,
        "dig_sha256": dig_sha256 or f"dig-{pair_slot or source_group}",
        "map_id": f"map-{sample_index}",
        "pair_slot_id": pair_slot or source_group,
        "sample_index": str(sample_index),
        "scenario_sha256": f"{sample_index:064x}",
        "source_group_id": source_group,
    }


def _write_bank(root: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    root.mkdir()
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = root / "dataset"
    for folder_index, folder in enumerate(splits.ARRAY_FOLDERS):
        (dataset / folder).mkdir(parents=True)
        for row in rows:
            sample_index = int(row["sample_index"])
            np.save(
                dataset / folder / f"img_{sample_index}.npy",
                np.full((2, 2), sample_index + folder_index, dtype=np.int16),
            )
    (dataset / splits.METADATA_FOLDER).mkdir()
    for row in rows:
        sample_index = int(row["sample_index"])
        (
            dataset
            / splits.METADATA_FOLDER
            / f"trench_{sample_index}.json"
        ).write_text("{}\n")
    return manifest, dataset


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_materializes_exact_splits_and_keeps_shared_sources_together(
    tmp_path,
):
    rows = [
        _row("fnd-a", 0, "foundation-source:7", "slab:0"),
        _row("fnd-a", 1, "foundation-source:8", "slab:1"),
        _row("fnd-b", 2, "foundation-source:7", "slab:0"),
        _row("fnd-b", 3, "foundation-source:8", "slab:1"),
        _row("trn-a", 4, "straight:0"),
        _row("trn-a", 5, "straight:1"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)
    output = tmp_path / "output"
    requested = {
        "train": 1,
        "promotion": 1,
        "development": 0,
        "sealed": 0,
    }

    summary = splits.materialize_splits(
        manifest, dataset, output, requested
    )

    materialized = _read_rows(output / "manifest.csv")
    split_by_source_group = {}
    for row in materialized:
        split_by_source_group.setdefault(row["source_group_id"], set()).add(
            row["split"]
        )
    assert all(len(values) == 1 for values in split_by_source_group.values())
    assert {
        next(iter(split_by_source_group[source]))
        for source in ("foundation-source:7", "foundation-source:8")
    } == {"train", "promotion"}

    for condition in ("fnd-a", "fnd-b", "trn-a"):
        assert summary["conditions"][condition]["train"]["pair_slots"] == 1
        assert (
            summary["conditions"][condition]["promotion"]["pair_slots"] == 1
        )
    assert len(_read_rows(output / "train" / "manifest.csv")) == 3
    assert len(_read_rows(output / "promotion" / "manifest.csv")) == 3
    for row in materialized:
        copied = (
            output
            / row["split"]
            / "dataset"
            / "images"
            / f"img_{row['sample_index']}.npy"
        )
        assert copied.is_file()


def test_materialization_is_deterministic(tmp_path):
    rows = [
        _row("condition", index, f"source:{index}") for index in range(4)
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)
    requested = {
        "train": 1,
        "promotion": 1,
        "development": 1,
        "sealed": 1,
    }

    splits.materialize_splits(
        manifest, dataset, tmp_path / "first", requested
    )
    splits.materialize_splits(
        manifest, dataset, tmp_path / "second", requested
    )

    for relative_path in (
        "manifest.csv",
        "summary.json",
        "train/manifest.csv",
        "promotion/manifest.csv",
        "development/manifest.csv",
        "sealed/manifest.csv",
    ):
        assert (tmp_path / "first" / relative_path).read_bytes() == (
            tmp_path / "second" / relative_path
        ).read_bytes()


def test_preserves_fixed_source_split_assignments(tmp_path):
    rows = [
        _row("fnd-a", 0, "source:shared-0", "slab:0"),
        _row("fnd-b", 1, "source:shared-0", "slab:0"),
        _row("fnd-a", 2, "source:shared-1", "slab:1"),
        _row("fnd-b", 3, "source:shared-1", "slab:1"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    summary = splits.materialize_splits(
        manifest,
        dataset,
        tmp_path / "output",
        {
            "train": 1,
            "promotion": 1,
            "development": 0,
            "sealed": 0,
        },
        {"source:shared-0": "promotion"},
    )

    materialized = _read_rows(tmp_path / "output" / "manifest.csv")
    assert {
        row["split"]
        for row in materialized
        if row["source_group_id"] == "source:shared-0"
    } == {"promotion"}
    assert summary["fixed_source_assignments"]["matched_sources"] == 1


def test_fails_if_fixed_sources_conflict_within_pair_slot(tmp_path):
    rows = [
        _row("fnd-a", 0, "source:a", "slab:0"),
        _row("fnd-b", 1, "source:b", "slab:0"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    with pytest.raises(RuntimeError, match="conflict within pair slot"):
        splits.materialize_splits(
            manifest,
            dataset,
            tmp_path / "output",
            {
                "train": 1,
                "promotion": 0,
                "development": 0,
                "sealed": 0,
            },
            {"source:a": "train", "source:b": "sealed"},
        )


def test_fails_when_exact_per_condition_counts_are_unavailable(tmp_path):
    rows = [_row("condition", 0, "source:0")]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    with pytest.raises(
        RuntimeError, match="only 1 exact pair slots remain"
    ):
        splits.materialize_splits(
            manifest,
            dataset,
            tmp_path / "output",
            {
                "train": 1,
                "promotion": 1,
                "development": 0,
                "sealed": 0,
            },
        )


def test_fails_if_source_group_is_empty(tmp_path):
    rows = [_row("fnd-a", 0, "", "slab:0")]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    with pytest.raises(ValueError, match="empty source_group_id"):
        splits.materialize_splits(
            manifest,
            dataset,
            tmp_path / "output",
            {
                "train": 1,
                "promotion": 0,
                "development": 0,
                "sealed": 0,
            },
        )


def test_fails_if_cross_level_source_conflicts_leave_too_few_slots(tmp_path):
    rows = [
        _row("fnd-a", 0, "foundation-source:7", "slab:0"),
        _row("fnd-a", 1, "foundation-source:8", "slab:1"),
        _row("fnd-b", 2, "foundation-source:7", "large:0"),
        _row("fnd-b", 3, "foundation-source:9", "large:1"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    with pytest.raises(
        RuntimeError, match="only 1 source-disjoint exact pair slots remain"
    ):
        splits.materialize_splits(
            manifest,
            dataset,
            tmp_path / "output",
            {
                "train": 1,
                "promotion": 1,
                "development": 0,
                "sealed": 0,
            },
        )


def test_skips_cross_level_source_conflicts_before_split_assignment(tmp_path):
    # The stable hash orders roomy:14 before the two non-conflicting slots.
    rows = [
        _row("scarce", 0, "source:shared", "scarce:0"),
        _row("scarce", 1, "source:scarce", "scarce:1"),
        _row("roomy", 2, "source:shared", "roomy:14"),
        _row("roomy", 3, "source:roomy-1", "roomy:1"),
        _row("roomy", 4, "source:roomy-2", "roomy:2"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    summary = splits.materialize_splits(
        manifest,
        dataset,
        tmp_path / "output",
        {
            "train": 1,
            "promotion": 1,
            "development": 0,
            "sealed": 0,
        },
    )

    materialized = _read_rows(tmp_path / "output" / "manifest.csv")
    assert {row["source_group_id"] for row in materialized} == {
        "source:shared",
        "source:scarce",
        "source:roomy-1",
        "source:roomy-2",
    }
    assert summary["pair_integrity"]["source_conflict_pair_slots"] == [
        "roomy:14"
    ]


def test_drops_rerolled_pair_slot_before_exact_selection(tmp_path):
    rows = [
        _row("fnd-a", 0, "source-0", "slab:0", "dig-0"),
        _row("fnd-b", 1, "source-0", "slab:0", "dig-0"),
        _row("fnd-a", 2, "source-1a", "slab:1", "dig-1a"),
        _row("fnd-b", 3, "source-1b", "slab:1", "dig-1b"),
        _row("fnd-a", 4, "source-2", "slab:2", "dig-2"),
        _row("fnd-b", 5, "source-2", "slab:2", "dig-2"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)
    summary = splits.materialize_splits(
        manifest,
        dataset,
        tmp_path / "output",
        {
            "train": 1,
            "promotion": 1,
            "development": 0,
            "sealed": 0,
        },
    )

    assert summary["pair_integrity"]["incomplete_pair_slots"] == ["slab:1"]
    materialized = _read_rows(tmp_path / "output" / "manifest.csv")
    assert {row["pair_slot_id"] for row in materialized} == {
        "slab:0",
        "slab:2",
    }


def test_fails_when_rerolls_leave_too_few_exact_pairs(tmp_path):
    rows = [
        _row("fnd-a", 0, "source-a", "slab:0", "dig-a"),
        _row("fnd-b", 1, "source-b", "slab:0", "dig-b"),
    ]
    manifest, dataset = _write_bank(tmp_path / "input", rows)

    with pytest.raises(RuntimeError, match="Generate a larger candidate bank"):
        splits.materialize_splits(
            manifest,
            dataset,
            tmp_path / "output",
            {
                "train": 1,
                "promotion": 0,
                "development": 0,
                "sealed": 0,
            },
        )
