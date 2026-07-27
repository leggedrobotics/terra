import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[2] / "tools" / "audit_pilot_foundation_source_support.py"
)
SPEC = importlib.util.spec_from_file_location(
    "pilot_foundation_source_support",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def metric_row(source_family, source_suffix, volume, perimeter):
    source_id = f"{source_family}:{source_suffix}"
    return {
        "source_family": source_family,
        "source_id": source_id,
        "source_group_id": source_id,
        "canonical_dig_sha256": source_suffix,
        "representative_relative_path": f"images/{source_suffix}.npy",
        "required_volume": volume,
        "required_volume_unit": "unit_depth_cell_volume",
        "perimeter_4_edges": perimeter,
        "compactness_4pi_area_over_perimeter_squared": (
            4.0 * math.pi * volume / perimeter**2
        ),
        "in_frozen_support": True,
    }


def test_square_perimeter_and_compactness_definition():
    mask = np.zeros((6, 6), dtype=np.bool_)
    mask[2:4, 2:4] = True
    metrics = support.foundation_metrics(mask)
    assert metrics["required_volume"] == 4
    assert metrics["perimeter_4_edges"] == 8
    assert metrics["compactness_4pi_area_over_perimeter_squared"] == pytest.approx(
        math.pi / 4.0
    )


def test_canonical_hash_groups_translation_rotation_and_reflection():
    mask = np.zeros((12, 12), dtype=np.bool_)
    mask[2:8, 3:5] = True
    mask[6:8, 5:9] = True
    translated = np.zeros_like(mask)
    rotated = np.rot90(support.crop_mask(mask))
    translated[3 : 3 + rotated.shape[0], 2 : 2 + rotated.shape[1]] = rotated
    reflected = np.fliplr(mask)
    assert support.canonical_dig_sha256(mask) == support.canonical_dig_sha256(
        translated
    )
    assert support.canonical_dig_sha256(mask) == support.canonical_dig_sha256(reflected)


def test_exact_matcher_keeps_sources_and_match_groups_distinct():
    osm_rows = [
        metric_row("osm", f"osm-{index}", 150 + index, 60 + 2 * index)
        for index in range(2)
    ]
    procedural_rows = []
    for index, osm in enumerate(osm_rows):
        for candidate in range(2):
            row = metric_row(
                "procedural",
                f"proc-{index}-{candidate}",
                osm["required_volume"],
                osm["perimeter_4_edges"],
            )
            row["sample_index"] = 10 * index + candidate
            procedural_rows.append(row)
    pairs, counts = support.match_exact_pairs(
        osm_rows,
        procedural_rows,
        pair_count=2,
        minimum_candidates=2,
    )
    assert len(pairs) == 2
    assert all(count == 2 for count in counts.values())
    assert len({row["procedural_source_id"] for row in pairs}) == 2
    for row in pairs:
        assert row["osm_source_id"] != row["procedural_source_id"]
        assert row["match_group_id"] not in {
            row["osm_source_id"],
            row["procedural_source_id"],
        }


def test_exact_matcher_fails_below_candidate_multiplicity():
    osm = metric_row("osm", "osm-only", 150, 60)
    procedural = metric_row("procedural", "proc-only", 150, 60)
    procedural["sample_index"] = 0
    with pytest.raises(RuntimeError, match="only 0 unique exact pairs"):
        support.match_exact_pairs(
            [osm],
            [procedural],
            pair_count=1,
            minimum_candidates=2,
        )


def test_exact_matcher_rejects_cross_family_canonical_raster_equality():
    osm = metric_row("osm", "shared-canonical", 150, 60)
    procedural = metric_row("procedural", "shared-canonical", 150, 60)
    procedural["sample_index"] = 0
    with pytest.raises(RuntimeError, match="share a canonical dig raster"):
        support.match_exact_pairs(
            [osm],
            [procedural],
            pair_count=1,
            minimum_candidates=1,
        )


def test_reserve_groups_never_reach_distribution_metrics():
    groups = [
        {
            "source_id": f"osm:{index}",
            "source_group_id": f"osm:{index}",
            "canonical_dig_sha256": f"{index:064x}",
            "representative_relative_path": f"images/{index}.npy",
            "member_files": [
                {
                    "relative_path": f"images/{index}.npy",
                    "file_sha256": f"{index:064x}",
                }
            ],
            "_dig": np.ones((2, 2), dtype=np.bool_),
        }
        for index in range(4)
    ]
    audit_train, reserve = support.partition_source_groups(
        groups,
        audit_train_count=2,
    )
    reserve_ids = {row["source_id"] for row in reserve}
    observed = []

    def metrics_fn(mask):
        group = next(row for row in audit_train if row["_dig"] is mask)
        observed.append(group["source_id"])
        return {
            "required_volume": 4,
            "required_volume_unit": "unit_depth_cell_volume",
            "perimeter_4_edges": 8,
            "compactness_4pi_area_over_perimeter_squared": math.pi / 4.0,
            "component_count_4": 1,
            "hole_cells": 0,
            "bbox_height_cells": 2,
            "bbox_width_cells": 2,
            "bbox_aspect_ratio": 1.0,
            "moment_aspect_ratio": 1.0,
            "moment_orientation_degrees": 0.0,
        }

    support.audit_osm_train_groups(audit_train, metrics_fn=metrics_fn)
    assert set(observed).isdisjoint(reserve_ids)
    assert len(observed) == 2
    registry = support.source_registry_rows(audit_train, reserve, pairs=[])
    reserve_registry = [
        row for row in registry if row["partition"] == support.RESERVE_PARTITION
    ]
    assert len(reserve_registry) == 2
    assert all("required_volume" not in row for row in reserve_registry)


def test_seed_stream_and_matching_are_deterministic():
    first = support.proposal_rng(7).integers(0, 2**31, size=8)
    second = support.proposal_rng(7).integers(0, 2**31, size=8)
    other = support.proposal_rng(8).integers(0, 2**31, size=8)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)
