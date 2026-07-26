import copy

import pytest

from tools.build_b0_trench_side_diversity import (
    CELLS,
    REFERENCE_FIELDS,
    SPLIT_COUNTS,
    validate_records,
    validate_reference_subset,
)


def fake_record(split, cell, index):
    map_id = f"b0a-{split}-{cell}-{index:02d}"
    pair = f"{split}:pair:{index:02d}"
    return {
        "map_id": map_id,
        "source_id": f"{split}:source:{index}",
        "split": split,
        "family": "trench",
        "stratum": "B0D",
        "primary_cell": cell,
        "geometry": "trench_straight",
        "dump_layout": "broad_side_cast",
        "distance_center_tiles": 2,
        "side_access": "both" if cell == CELLS[0] else "one",
        "topology": "straight",
        "generation_seed": index,
        "generation_attempt": 0,
        "paired_source_group_id": pair,
        "topology_match_group_id": pair if cell == CELLS[0] else None,
        "dig_identity_sha256": f"dig-{split}-{index}",
        "target_identity_sha256": f"target-{split}-{cell}-{index}",
        "maximum_within_cell_geometry_iou": 0.0,
        "validation": {"status": "passed"},
    }


def test_validate_records_requires_declared_counts(monkeypatch):
    monkeypatch.setattr(
        "tools.build_b0_trench_side_diversity.SPLIT_COUNTS",
        {"train": 2, "development": 1},
    )
    records = [
        fake_record(split, cell, index)
        for split, count in {"train": 2, "development": 1}.items()
        for index in range(count)
        for cell in CELLS
    ]
    validation = validate_records(records)
    assert validation["identity_count"] == 6
    records.pop()
    with pytest.raises(RuntimeError, match="generated"):
        validate_records(records)


def test_reference_subset_ignores_only_stratum(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tools.build_b0_trench_side_diversity.SPLIT_COUNTS",
        {"train": 8, "development": 8},
    )
    identity_manifest = tmp_path / "identities.jsonl"
    identity_manifest.write_text("{}\n")
    generated = [
        fake_record(split, cell, index)
        for split in ("train", "development")
        for index in range(8)
        for cell in CELLS
    ]
    reference = []
    for record in generated:
        item = copy.deepcopy(record)
        item["stratum"] = "B0a"
        item["_identity_manifest"] = str(identity_manifest)
        reference.append(item)
    result = validate_reference_subset(generated, reference)
    assert result["passed"]
    assert set(result["exact_fields"]) == set(REFERENCE_FIELDS)

    reference[0]["target_identity_sha256"] = "different"
    with pytest.raises(RuntimeError, match="reference subset mismatch"):
        validate_reference_subset(generated, reference)
