import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import audit_pilot_foundation_source_support as support
from tools import build_b0_feasibility_panels as b0
from tools import build_pilot_apron_capacity_review as review


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _l_foundation(offset: int = 0, extension: int = 0) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[20 + offset : 40 + offset, 22:28] = True
    mask[34 + offset : 40 + offset, 22 : 37 + extension] = True
    return mask


def _source_row(mask: np.ndarray, index: int) -> dict:
    metrics = support.foundation_metrics(mask)
    canonical_hash = support.canonical_dig_sha256(mask)
    return {
        "audit_partition": "audit_train",
        "source_family": "osm",
        "source_id": f"osm:{canonical_hash}",
        "source_group_id": f"osm:{canonical_hash}",
        "canonical_dig_sha256": canonical_hash,
        "representative_relative_path": f"images/img_{index}.npy",
        "in_frozen_support": True,
        **metrics,
    }


def test_selection_excludes_all_burned_shapes_before_capacity_construction():
    rows = []
    for index in range(40):
        canonical_hash = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        rows.append(
            {
                "audit_partition": "audit_train",
                "source_family": "osm",
                "source_group_id": f"osm:{canonical_hash}",
                "canonical_dig_sha256": canonical_hash,
                "required_volume": 150,
                "compactness_4pi_area_over_perimeter_squared": 0.5,
                "in_frozen_support": True,
            }
        )
    burned = {rows[index]["canonical_dig_sha256"] for index in (0, 7, 31)}

    selected, receipt = review.select_fresh_sources(rows, burned, pair_count=32)
    selected_hashes = {row["canonical_dig_sha256"] for row in selected}

    assert not selected_hashes & burned
    assert len(selected) == 32
    assert receipt["frozen_support_count"] == 40
    assert receipt["b0a_burned_overlap_count"] == 3
    assert receipt["post_exclusion_eligible_count"] == 37
    assert receipt["rank_namespace"] == ("terra_pilot_apron_capacity_review_v1\0")
    assert receipt["selection_consulted_constructor_or_policy_outcomes"] is False
    assert selected == review.select_fresh_sources(rows, burned, pair_count=32)[0]


def test_b0a_exclusion_reads_every_osm_derived_cell_and_split(
    tmp_path,
    monkeypatch,
):
    identities = []
    expected = set()
    cases = (
        ("train", "f_osm_all", _l_foundation(0)),
        ("train", "f_apron_d02", _l_foundation(1, extension=1)),
        ("development", "f_apron_d08", _l_foundation(2, extension=2)),
    )
    for index, (split, cell, dig) in enumerate(cases):
        dataset = tmp_path / "cells" / split / cell
        (dataset / "images").mkdir(parents=True)
        target = np.zeros((64, 64), dtype=np.int8)
        target[dig] = -1
        target[target == 0] = 1
        np.save(dataset / "images" / "img_1.npy", target)
        map_id = f"b0a-{split}-{cell}-{index:02d}"
        _write_jsonl(
            dataset / "manifest.jsonl",
            [{"map_id": map_id, "slot_index": 1}],
        )
        identities.append(
            {
                "map_id": map_id,
                "split": split,
                "primary_cell": cell,
                "geometry": "foundation_osm",
                "target_identity_sha256": b0.sha256_array(target),
            }
        )
        expected.add(support.canonical_dig_sha256(dig))

    _write_jsonl(tmp_path / "identities.jsonl", identities)
    (tmp_path / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "test_b0a",
                "identity_manifest_sha256": review.sha256_file(
                    tmp_path / "identities.jsonl"
                ),
            }
        )
    )
    review.write_file_manifest(tmp_path)
    frozen_manifest_sha256 = review.sha256_file(tmp_path / "files.sha256")
    monkeypatch.setattr(
        review,
        "EXPECTED_B0A_FILES_MANIFEST_SHA256",
        frozen_manifest_sha256,
    )
    monkeypatch.setattr(review, "EXPECTED_B0A_SCHEMA", "test_b0a")
    monkeypatch.setattr(review, "EXPECTED_B0A_IDENTITY_COUNT", 3)
    monkeypatch.setattr(review, "EXPECTED_B0A_OSM_IDENTITY_COUNT", 3)
    monkeypatch.setattr(review, "EXPECTED_B0A_OSM_CANONICAL_DIG_COUNT", 3)

    observed, receipt = review._b0a_osm_canonical_hashes(tmp_path)

    assert observed == expected
    assert receipt["osm_identity_count"] == 3
    assert receipt["osm_canonical_dig_count"] == 3
    assert receipt["files_manifest_sha256"] == frozen_manifest_sha256

    (tmp_path / "identities.jsonl").write_text(
        (tmp_path / "identities.jsonl").read_text() + "\n"
    )
    provenance = json.loads((tmp_path / "provenance.json").read_text())
    provenance["identity_manifest_sha256"] = review.sha256_file(
        tmp_path / "identities.jsonl"
    )
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="Hash mismatch"):
        review._b0a_osm_canonical_hashes(tmp_path)

    review.write_file_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="B0a root file manifest changed"):
        review._b0a_osm_canonical_hashes(tmp_path)


def test_geodesic_distance_keeps_diagonal_frontier_after_float_rounding():
    target = np.zeros((64, 64), dtype=np.int8)
    target[32, 32] = 1
    observed = review._geodesic_distance(
        target,
        np.zeros((64, 64), dtype=np.bool_),
    )

    y, x = np.indices(target.shape)
    dy = np.abs(y - 32)
    dx = np.abs(x - 32)
    expected = np.minimum(dy, dx) * np.sqrt(2.0) + np.abs(dy - dx)
    expected /= expected.max()

    assert observed.dtype == np.float32
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-7)
    assert np.count_nonzero(observed == 1.0) < 8


def test_two_pair_review_is_exact_loader_compatible_and_reverifiable(tmp_path):
    source_foundations = tmp_path / "sources"
    (source_foundations / "images").mkdir(parents=True)
    selected = []
    for index, dig in enumerate(
        (_l_foundation(0), _l_foundation(2, extension=1)), start=1
    ):
        target = np.zeros((64, 64), dtype=np.int8)
        target[dig] = -1
        np.save(source_foundations / "images" / f"img_{index}.npy", target)
        selected.append(_source_row(dig, index))

    output = tmp_path / "review"
    summary = review._build_from_selected(
        selected=selected,
        source_foundations=source_foundations,
        output=output,
        input_receipts={"test": True},
        selection_receipt={
            "rank_namespace": review.RANK_NAMESPACE,
            "selected_count": 2,
        },
        source_state_receipt={
            "terra_revision": "test-revision",
            "terra_worktree_clean": True,
            "synthetic_test_receipt": True,
        },
    )
    with pytest.raises(RuntimeError, match="counts disagree"):
        review.verify_artifact(output)
    verification = review.verify_artifact(output, expected_pair_count=2)

    assert summary["status"] == review.EXACT_LOADER_STATUS
    assert summary["canonical_benchmark_format_admitted"] is False
    assert summary["s1_capacity_gate_complete"] is False
    assert summary["pair_count"] == 2
    assert summary["map_count"] == 4
    assert summary["all_pairs_share_exact_dig_and_volume"] is True
    assert verification["status"] == "passed"
    assert verification["static_status"] == review.STATIC_STATUS

    pairs = review.load_jsonl(output / "pairs.jsonl")
    identities = review.load_jsonl(output / "identities.jsonl")
    assert all(pair["exact_shared_dig_and_volume"] for pair in pairs)
    assert all(row["exact_loader_format_valid"] for row in identities)
    assert all(
        row["direct_service_status"] == "not_run_cost_gate" for row in identities
    )
    assert (output / "galleries" / "slcap03_04.png").is_file()
    assert (output / "galleries" / "slcap07_10.png").is_file()
    assert len(list((output / "pairs").glob("*.png"))) == 2
