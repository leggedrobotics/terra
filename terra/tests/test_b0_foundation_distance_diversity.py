import copy
from pathlib import Path

import numpy as np
import pytest

from tools import build_b0_foundation_distance_diversity as diversity


def dig_shape(group_index):
    dig = np.zeros((64, 64), dtype=np.bool_)
    if group_index == 0:
        dig[29:31, 29:32] = True
    elif group_index == 1:
        dig[29:32, 29] = True
        dig[31, 29:32] = True
    else:
        dig[28:33, 30] = True
        dig[30, 28:33] = True
    return dig


def fake_bank():
    records = []
    samples = {}
    group_index = 0
    reference_sources = set()
    for split, count in diversity.SPLIT_COUNTS.items():
        for identity_index in range(count):
            source_index = 100 + group_index
            source_id = f"osm-foundation:{source_index}"
            if identity_index < diversity.REFERENCE_GROUP_COUNT:
                reference_sources.add(source_id)
            dig = dig_shape(group_index)
            pair = f"{split}:foundation-distance:{identity_index:02d}"
            for distance, cell in zip((2, 4, 6, 8), diversity.CELLS):
                map_id = f"b0a-{split}-{cell}-{identity_index:02d}"
                target = np.zeros((64, 64), dtype=np.int8)
                target[dig] = -1
                target[10 + distance, 10:14] = 1
                records.append(
                    {
                        "map_id": map_id,
                        "source_id": source_id,
                        "split": split,
                        "family": "foundation",
                        "stratum": "B0D",
                        "primary_cell": cell,
                        "geometry": "foundation_osm",
                        "dump_layout": "broad_apron",
                        "distance_center_tiles": distance,
                        "side_access": "all",
                        "topology": None,
                        "generation_seed": 1000 + group_index,
                        "generation_attempt": 0,
                        "paired_source_group_id": pair,
                        "topology_match_group_id": None,
                        "dig_identity_sha256": f"dig-{group_index}",
                        "target_identity_sha256": f"target-{group_index}-{distance}",
                        "validation": {
                            "status": "passed",
                            "accepted_dump_contract": "exact_visible_dump_v1",
                            "dig_cells": int(dig.sum()),
                            "dump_cells": 4,
                            "capacity": {"single_layer_capacity_ratio": 3.25},
                            "distance": {"p50_tiles": float(distance)},
                        },
                    }
                )
                samples[map_id] = diversity.b0.Sample(
                    target=target,
                    occupancy=np.zeros((64, 64), dtype=np.bool_),
                    dumpability=np.ones((64, 64), dtype=np.bool_),
                    action=np.zeros((64, 64), dtype=np.int8),
                    distance=np.zeros((64, 64), dtype=np.float32),
                    metadata={},
                )
            group_index += 1
    return records, samples, reference_sources


def configure_small_bank(monkeypatch):
    monkeypatch.setattr(
        diversity,
        "SPLIT_COUNTS",
        {"train": 2, "development": 1},
    )
    monkeypatch.setattr(diversity, "REFERENCE_GROUP_COUNT", 1)


def test_validate_records_enforces_counts_pairing_and_source_disjointness(
    monkeypatch,
):
    configure_small_bank(monkeypatch)
    records, samples, reference_sources = fake_bank()
    result = diversity.validate_records(records, samples, reference_sources)
    assert result["identity_count"] == 12
    assert result["paired_geometry_groups"] == 3
    assert result["unique_sources_by_split"] == {"train": 2, "development": 1}

    incomplete = copy.deepcopy(records)
    incomplete.pop()
    with pytest.raises(RuntimeError, match="generated"):
        diversity.validate_records(incomplete, samples, reference_sources)

    broken_pair = copy.deepcopy(records)
    broken_pair[1]["generation_seed"] += 1
    with pytest.raises(RuntimeError, match="changed generation_seed"):
        diversity.validate_records(broken_pair, samples, reference_sources)

    leaked_samples = copy.deepcopy(samples)
    added_pair = "train:foundation-distance:01"
    retained_pair = "train:foundation-distance:00"
    for added in (
        record for record in records if record["paired_source_group_id"] == added_pair
    ):
        retained = next(
            record
            for record in records
            if record["paired_source_group_id"] == retained_pair
            and record["primary_cell"] == added["primary_cell"]
        )
        leaked_samples[added["map_id"]].target = samples[
            retained["map_id"]
        ].target.copy()
    with pytest.raises(RuntimeError, match="templated duplicate"):
        diversity.validate_records(records, leaked_samples, reference_sources)

    with monkeypatch.context() as threshold_patch:
        threshold_patch.setattr(
            diversity.b0,
            "maximum_dihedral_iou",
            lambda _left, _right: 0.96,
        )
        with pytest.raises(RuntimeError, match="templated duplicate"):
            diversity.validate_records(records, samples, reference_sources)


def test_reference_gate_preserves_all_fields_and_tensors_and_rejects_source_reuse(
    monkeypatch,
    tmp_path,
):
    configure_small_bank(monkeypatch)
    generator_root = tmp_path / "generator"
    generator_root.mkdir()
    generator_hashes = {}
    for name in diversity.GENERATOR_SHA256:
        path = generator_root / name
        path.write_text(name)
        generator_hashes[name] = diversity.b0.sha256_file(path)
    monkeypatch.setattr(diversity, "GENERATOR_SHA256", generator_hashes)
    assert len(diversity.verify_generator_files(generator_root)) == 5
    (generator_root / next(iter(generator_hashes))).write_text("changed")
    with pytest.raises(RuntimeError, match="unexpected map generator"):
        diversity.verify_generator_files(generator_root)
    observed_base_builder = diversity.b0.sha256_file(
        Path(diversity.b0.__file__).resolve()
    )
    monkeypatch.setattr(
        diversity,
        "BASE_BUILDER_SHA256",
        observed_base_builder,
    )
    assert diversity.verify_base_builder()["sha256"] == observed_base_builder
    monkeypatch.setattr(diversity, "BASE_BUILDER_SHA256", "wrong")
    with pytest.raises(RuntimeError, match="unexpected B0a base builder"):
        diversity.verify_base_builder()

    records, samples, reference_sources = fake_bank()
    retained = [
        copy.deepcopy(record)
        for record in records
        if diversity.reference_index(record) < diversity.REFERENCE_GROUP_COUNT
    ]
    reference_records = copy.deepcopy(retained)
    for record in reference_records:
        record["stratum"] = "B0a"
    reference_samples = {
        record["map_id"]: copy.deepcopy(samples[record["map_id"]])
        for record in retained
    }
    result = diversity.validate_reference_subset(
        records,
        samples,
        reference_records,
        reference_samples,
    )
    assert result["retained_records"] == 8
    assert set(result["exact_tensor_fields"]) == set(diversity.TENSOR_FIELDS)

    changed_samples = copy.deepcopy(samples)
    map_id = retained[0]["map_id"]
    changed_samples[map_id].target[0, 0] = 1
    with pytest.raises(RuntimeError, match="retained target tensor changed"):
        diversity.validate_reference_subset(
            records,
            changed_samples,
            reference_records,
            reference_samples,
        )

    reused = copy.deepcopy(records)
    added = next(
        record
        for record in reused
        if record["split"] == "train"
        and diversity.reference_index(record) >= diversity.REFERENCE_GROUP_COUNT
    )
    reused_source = next(iter(reference_sources))
    added_pair = added["paired_source_group_id"]
    for record in reused:
        if record["paired_source_group_id"] == added_pair:
            record["source_id"] = reused_source
    with pytest.raises(RuntimeError, match="sources cross splits|reuse B0a"):
        diversity.validate_records(reused, samples, reference_sources)
