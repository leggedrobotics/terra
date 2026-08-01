import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.map_generation import generate_curriculum_bank as generator
from tools.map_generation import materialize_candidate_extension as extension


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical_level(level: str) -> list[str]:
    return [
        condition.id
        for condition in generator.MAIN_CONDITIONS
        if condition.dig_bank_level == level
    ]


def _write_bank(
    root: Path,
    condition_ids: list[str],
    maps: int,
    map_id_prefix: str,
    source_sha256: str = "a" * 64,
) -> Path:
    root.mkdir()
    conditions = tuple(generator.MAIN_CONDITIONS)
    by_id = {condition.id: condition for condition in conditions}
    index_by_id = {condition.id: index for index, condition in enumerate(conditions)}
    rows = []
    for condition_id in condition_ids:
        condition = by_id[condition_id]
        condition_index = index_by_id[condition_id]
        for map_index in range(maps):
            sample_index = generator.sample_index_of(condition_index, map_index)
            dig_sha256 = _identity(f"dig:{condition.dig_bank_level}:{map_index}")
            row = {
                "attempt": "0",
                "condition_id": condition_id,
                "condition_index": str(condition_index),
                "dig_sha256": dig_sha256,
                "map_id": f"{map_id_prefix}-{sample_index:04d}",
                "map_index": str(map_index),
                "pair_slot_id": f"{condition.dig_bank_level}:{map_index}",
                "sample_index": str(sample_index),
                "scenario_sha256": _identity(f"scenario:{condition_id}:{map_index}"),
                "schema": generator.SCHEMA,
                "seed_base": str(generator.SEED_BASE),
                "source_group_id": f"dig:{dig_sha256}",
            }
            rows.append(row)
            for folder_index, folder in enumerate(extension.ARRAY_FOLDERS):
                destination = root / "dataset" / folder
                destination.mkdir(parents=True, exist_ok=True)
                np.save(
                    destination / f"img_{sample_index}.npy",
                    np.full(
                        (2, 2),
                        sample_index + folder_index,
                        dtype=np.int64,
                    ),
                )
            metadata = root / "dataset" / extension.METADATA_FOLDER
            metadata.mkdir(parents=True, exist_ok=True)
            (metadata / f"trench_{sample_index}.json").write_text(
                json.dumps(
                    {
                        "condition_id": condition_id,
                        "map_id": f"{map_id_prefix}-{sample_index:04d}",
                        "map_index": map_index,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    fieldnames = sorted(rows[0])
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = {condition_id: maps for condition_id in condition_ids}
    summary = {
        "accepted_maps": len(rows),
        "condition_count": len(conditions),
        "conditions_built_this_run": sorted(condition_ids),
        "dataset": "main",
        "generator": "tools/map_generation/generate_curriculum_bank.py",
        "maps_per_condition": counts,
        "schema": generator.SCHEMA,
        "seed_base": generator.SEED_BASE,
        "source_foundations_images": 600,
        "source_foundations_sha256": source_sha256,
        "taxonomy_release": "v6-main",
        "taxonomy_version": generator.tax.TAXONOMY_VERSION,
        "tile_size_m": generator.TILE_SIZE_M,
    }
    (root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return root


@pytest.fixture
def candidate_banks(tmp_path):
    all_conditions = [condition.id for condition in generator.MAIN_CONDITIONS]
    base = _write_bank(tmp_path / "base", all_conditions, 1, "base-candidates")
    net3 = _canonical_level("net3")
    shard = _write_bank(tmp_path / "net3", net3, 2, "larger-net3")
    return base, shard, net3


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_materializes_whole_level_suffix_and_preserves_base_ids(
    tmp_path, candidate_banks
):
    base, shard, net3 = candidate_banks
    output = tmp_path / "extended"

    receipt = extension.materialize_candidate_extension(base, [shard], output)

    output_rows = _rows(output / "manifest.csv")
    base_rows = _rows(base / "manifest.csv")
    assert len(output_rows) == len(base_rows) + len(net3)
    output_by_sample = {row["sample_index"]: row for row in output_rows}
    assert all(output_by_sample[row["sample_index"]] == row for row in base_rows)
    base_samples = {row["sample_index"] for row in base_rows}
    appended_rows = [
        row for row in output_rows if row["sample_index"] not in base_samples
    ]
    assert {row["map_id"] for row in appended_rows} == {
        f"base-candidates-{generator.sample_index_of(index, 1):04d}"
        for index, condition in enumerate(generator.MAIN_CONDITIONS)
        if condition.id in net3
    }
    assert receipt["extensions"][0]["dig_bank_levels"] == ["net3"]
    assert receipt["extensions"][0]["overlap_rows_verified"] == len(net3)
    assert receipt["extensions"][0]["overlap_array_files_verified"] == (
        len(net3) * len(extension.ARRAY_FOLDERS)
    )
    assert receipt["extensions"][0]["appended_scenarios"] == len(net3)
    assert receipt["output"]["manifest_sha256"] == extension._sha256(
        output / "manifest.csv"
    )
    for condition_id in net3:
        condition_index = next(
            index
            for index, condition in enumerate(generator.MAIN_CONDITIONS)
            if condition.id == condition_id
        )
        sample_index = generator.sample_index_of(condition_index, 1)
        assert (
            output / "dataset" / "images" / f"img_{sample_index}.npy"
        ).read_bytes() == (
            shard / "dataset" / "images" / f"img_{sample_index}.npy"
        ).read_bytes()
        output_metadata = json.loads(
            (
                output
                / "dataset"
                / extension.METADATA_FOLDER
                / f"trench_{sample_index}.json"
            ).read_text()
        )
        assert output_metadata["map_id"] == (f"base-candidates-{sample_index:04d}")


def test_rejects_partial_dig_bank_level(tmp_path):
    all_conditions = [condition.id for condition in generator.MAIN_CONDITIONS]
    base = _write_bank(tmp_path / "base", all_conditions, 1, "base")
    partial = _write_bank(
        tmp_path / "partial", [_canonical_level("net3")[0]], 2, "partial"
    )
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="whole dig-bank levels"):
        extension.materialize_candidate_extension(base, [partial], output)

    assert not output.exists()


def test_rejects_overlap_array_byte_mismatch(tmp_path, candidate_banks):
    base, shard, _ = candidate_banks
    first_condition = _canonical_level("net3")[0]
    condition_index = next(
        index
        for index, condition in enumerate(generator.MAIN_CONDITIONS)
        if condition.id == first_condition
    )
    sample_index = generator.sample_index_of(condition_index, 0)
    (shard / "dataset" / "actions" / f"img_{sample_index}.npy").write_bytes(b"corrupt")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="overlap array bytes differ"):
        extension.materialize_candidate_extension(base, [shard], output)

    assert not output.exists()


def test_rejects_provenance_mismatch(tmp_path, candidate_banks):
    base, shard, _ = candidate_banks
    summary_path = shard / "generation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_foundations_sha256"] = "b" * 64
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="source_foundations_sha256"):
        extension.materialize_candidate_extension(base, [shard], output)

    assert not output.exists()


def test_rejects_suffix_scenario_collision(tmp_path, candidate_banks):
    base, shard, net3 = candidate_banks
    base_rows = _rows(base / "manifest.csv")
    shard_rows = _rows(shard / "manifest.csv")
    for row in shard_rows:
        if row["condition_id"] == net3[0] and row["map_index"] == "1":
            row["scenario_sha256"] = base_rows[0]["scenario_sha256"]
    with (shard / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shard_rows[0]))
        writer.writeheader()
        writer.writerows(shard_rows)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="output scenario collision"):
        extension.materialize_candidate_extension(base, [shard], output)

    assert not output.exists()


def test_rejects_noncontiguous_extension_indices(tmp_path, candidate_banks):
    base, shard, net3 = candidate_banks
    rows = _rows(shard / "manifest.csv")
    rows = [
        row
        for row in rows
        if not (row["condition_id"] == net3[0] and row["map_index"] == "0")
    ]
    with (shard / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="not contiguous from zero"):
        extension.materialize_candidate_extension(base, [shard], output)

    assert not output.exists()
