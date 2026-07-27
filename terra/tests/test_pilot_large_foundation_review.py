import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import audit_pilot_foundation_source_support as support
from tools import build_pilot_large_foundation_review as review


def _mask(height: int, width: int, *, offset: int = 0) -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.bool_)
    mask[20 + offset : 20 + offset + height, 22 : 22 + width] = True
    return mask


def _row(mask: np.ndarray, sample_index: int) -> dict:
    metrics = support.foundation_metrics(mask)
    canonical_hash = support.canonical_dig_sha256(mask)
    return {
        "audit_partition": "audit_train_candidate",
        "source_family": "procedural",
        "source_id": f"procedural:{canonical_hash}",
        "source_group_id": f"procedural:{canonical_hash}",
        "canonical_dig_sha256": canonical_hash,
        "sample_index": sample_index,
        "generator_revision": support.PROCEDURAL_GENERATOR_REVISION,
        "generator_draw_count": 1,
        "generator_rejections_before_proposal": {
            "contains_holes": 0,
            "margin_violation": 0,
            "not_4_connected": 0,
            "required_volume_outside_generator_bounds": 0,
        },
        "main_center_y_cells": 32.0,
        "main_center_x_cells": 32.0,
        "main_angle_degrees": 0.0,
        "main_length_cells": float(metrics["bbox_width_cells"]),
        "main_width_cells": float(metrics["bbox_height_cells"]),
        "wing_count": 1,
        **metrics,
    }


def _selection_receipt(selected: list[dict], references: list[dict]) -> dict:
    return {
        "selected_count": len(selected),
        "selected_source_set_sha256": review.sha256_text_lines(
            row["source_group_id"] for row in selected
        ),
        "reference_count": len(references),
        "reference_source_set_sha256": review.sha256_text_lines(
            row["source_group_id"] for row in references
        ),
        "held_out_or_sealed_selection_performed": False,
        "policy_outcomes_consulted": False,
        "constructor_outcomes_consulted": False,
    }


def _input_receipts() -> dict:
    return {
        "b0a_bank": {
            "files_manifest_sha256": review.EXPECTED_B0A_FILES_MANIFEST_SHA256
        },
        "capacity_review": {
            "files_manifest_sha256": review.EXPECTED_CAPACITY_FILES_MANIFEST_SHA256
        },
        "source_support": {
            "files_manifest_sha256": (
                review.capacity_review.EXPECTED_SOURCE_SUPPORT_FILES_MANIFEST_SHA256
            ),
            "generator_files": {
                "generate_prototypes.py": support.EXPECTED_BASE_GENERATOR_SHA256
            },
        },
    }


def _source_state_receipt(revision: str = "test-revision") -> dict:
    return {
        "terra_revision": revision,
        "terra_worktree_clean": True,
        "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
    }


def _synthetic_review(tmp_path: Path, monkeypatch) -> dict:
    burned = _mask(10, 15, offset=4)
    burned[34, 22] = True
    masks = {
        1: _mask(10, 15),
        2: _mask(10, 16, offset=2),
        3: burned,
        11: _mask(10, 10),
        12: _mask(10, 11, offset=1),
    }
    selected = [_row(masks[index], index) for index in (1, 2)]
    references = [_row(masks[index], index) for index in (11, 12)]
    for name, value in (
        ("VOLUME_LOWER_INCLUSIVE", 140),
        ("VOLUME_UPPER_INCLUSIVE", 170),
        ("ANCHOR_VOLUME_LOWER_INCLUSIVE", 90),
        ("ANCHOR_VOLUME_UPPER_INCLUSIVE", 120),
        ("COMPACTNESS_LOWER_INCLUSIVE", 0.0),
        ("COMPACTNESS_UPPER_INCLUSIVE", 1.0),
        ("TARGET_AREA_FRACTION_LOWER", 140 / 4096),
        ("TARGET_AREA_FRACTION_UPPER", 170 / 4096),
        ("REFERENCE_COUNT", 2),
    ):
        monkeypatch.setattr(review, name, value)
    monkeypatch.setattr(
        review,
        "_regenerate_mask",
        lambda _base, row: masks[int(row["sample_index"])],
    )
    output = tmp_path / "large_review"
    inputs = _input_receipts()
    selection = _selection_receipt(selected, references)
    source_state = _source_state_receipt()
    summary = review._materialize_from_selected(
        selected=selected,
        references=references,
        base_generator=object(),
        output=output,
        input_receipts=inputs,
        selection_receipt=selection,
        source_state_receipt=source_state,
    )
    return {
        "output": output,
        "summary": summary,
        "masks": masks,
        "selected": selected,
        "references": references,
        "inputs": inputs,
        "selection": selection,
        "source_state": source_state,
    }


def _verify(data: dict) -> dict:
    return review._verify_artifact_from_fixtures(
        data["output"],
        base_generator=object(),
        selected=data["selected"],
        references=data["references"],
        input_receipts=data["inputs"],
        selection_receipt=data["selection"],
        expected_source_state_receipt=data["source_state"],
        expected_count=len(data["selected"]),
    )


def test_selection_is_train_only_fresh_and_outcome_independent(monkeypatch):
    monkeypatch.setattr(review, "VOLUME_LOWER_INCLUSIVE", 140)
    monkeypatch.setattr(review, "VOLUME_UPPER_INCLUSIVE", 170)
    monkeypatch.setattr(review, "ANCHOR_VOLUME_LOWER_INCLUSIVE", 90)
    monkeypatch.setattr(review, "ANCHOR_VOLUME_UPPER_INCLUSIVE", 120)
    monkeypatch.setattr(review, "COMPACTNESS_LOWER_INCLUSIVE", 0.0)
    monkeypatch.setattr(review, "COMPACTNESS_UPPER_INCLUSIVE", 1.0)
    monkeypatch.setattr(review, "REFERENCE_COUNT", 4)
    rows = []
    for index in range(24):
        mask = _mask(10, 14)
        mask[30, 22 : 23 + index] = True
        rows.append(_row(mask, index))
    for index in range(8):
        mask = _mask(9, 10)
        mask[29, 22 : 23 + index] = True
        rows.append(_row(mask, 100 + index))
    excluded = {
        rows[1]["canonical_dig_sha256"],
        rows[7]["canonical_dig_sha256"],
    }

    selected, references, receipt = review.select_large_tail_sources(
        rows,
        excluded,
        review_count=8,
    )

    assert len(selected) == 8
    assert len(references) == 4
    assert not {row["canonical_dig_sha256"] for row in selected + references} & excluded
    assert receipt["selected_count"] == 8
    assert receipt["held_out_or_sealed_selection_performed"] is False
    assert (
        selected == review.select_large_tail_sources(rows, excluded, review_count=8)[0]
    )
    bad = dict(rows[0], audit_partition="public_development")
    with pytest.raises(RuntimeError, match="train-only contract"):
        review.select_large_tail_sources([bad, *rows[1:]], excluded, review_count=8)


def test_materialized_review_loads_and_rebuild_verifies(tmp_path: Path, monkeypatch):
    data = _synthetic_review(tmp_path, monkeypatch)
    result = _verify(data)
    provenance = json.loads((data["output"] / "provenance.json").read_text())
    map_protocol = provenance["environment_protocol"]["map"]
    env_receipt = provenance["env_config_receipt"]
    tile_size_m = map_protocol["tile_size_m_derived_float64"]
    identities = review.load_jsonl(data["output"] / "identities.jsonl")

    assert data["summary"]["status"] == review.EXACT_LOADER_STATUS
    assert data["summary"]["canonical_benchmark_format_admitted"] is False
    assert data["summary"]["work_volume_condition_admitted"] is False
    assert data["summary"]["graphic_hash_portability"] == (
        review.GRAPHIC_HASH_PORTABILITY
    )
    assert result["status"] == "passed"
    assert result["map_count"] == 2
    assert len(list((data["output"] / "examples").glob("*.png"))) == 2
    assert map_protocol == {
        key: env_receipt[key]
        for key in (
            "edge_length_px",
            "edge_length_m",
            "tile_size_m_derived_float64",
            "tile_size_m_runtime_float32",
        )
    }
    assert tile_size_m == 36.5714285714 / 64
    assert tile_size_m != 0.6875
    for record in identities:
        for statistic in ("p50", "p95", "max"):
            assert record["separation"][f"{statistic}_metres"] == (
                record["separation"][f"{statistic}_tiles"] * tile_size_m
            )


def test_materialization_rejects_disagreeing_protocol_receipts(
    tmp_path: Path,
    monkeypatch,
):
    frozen_environment_protocol = review.frozen_environment_protocol

    def disagreeing_environment_protocol(revision: str) -> dict:
        receipt = frozen_environment_protocol(revision)
        receipt["map"]["tile_size_m_derived_float64"] = 0.6875
        return receipt

    monkeypatch.setattr(
        review,
        "frozen_environment_protocol",
        disagreeing_environment_protocol,
    )
    with pytest.raises(RuntimeError, match="map geometry receipts disagree"):
        _synthetic_review(tmp_path, monkeypatch)


def test_swapped_or_burned_selection_fails_frozen_rebuild(tmp_path: Path, monkeypatch):
    data = _synthetic_review(tmp_path, monkeypatch)
    for name, selected in (
        ("swapped", list(reversed(data["selected"]))),
        ("burned", [_row(data["masks"][3], 3), data["selected"][1]]),
    ):
        output = tmp_path / name
        review._materialize_from_selected(
            selected=selected,
            references=data["references"],
            base_generator=object(),
            output=output,
            input_receipts=data["inputs"],
            selection_receipt=_selection_receipt(selected, data["references"]),
            source_state_receipt=data["source_state"],
        )
        candidate = {**data, "output": output}
        with pytest.raises(RuntimeError, match="deterministic rebuild"):
            _verify(candidate)


def test_self_consistent_artifact_tamper_fails_exact_tree(tmp_path: Path, monkeypatch):
    data = _synthetic_review(tmp_path, monkeypatch)
    output = data["output"]

    registry_path = output / "source_registry.jsonl"
    registry = review.load_jsonl(registry_path)
    registry[0]["split"] = "public_development"
    review.write_jsonl(registry_path, registry)
    registry_hash = review.sha256_file(registry_path)

    dataset_path = output / "dataset" / "dataset.json"
    dataset = json.loads(dataset_path.read_text())
    dataset["source_registry_sha256"] = registry_hash
    dataset["distance_metric"] = "wrong_but_self_declared"
    review.write_json(dataset_path, dataset)

    metadata_path = output / "dataset" / "metadata" / "trench_1.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["family"] = "trench"
    review.write_json(metadata_path, metadata)

    identities_path = output / "identities.jsonl"
    identities = review.load_jsonl(identities_path)
    identities[0]["required_volume_unit"] = "wrong_unit"
    review.write_jsonl(identities_path, identities)

    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["source_registry_sha256"] = registry_hash
    provenance["identities_sha256"] = review.sha256_file(identities_path)
    fabricated_revision = "fabricated-clean-revision"
    provenance["source_state"] = _source_state_receipt(fabricated_revision)
    provenance["environment_protocol"] = review.frozen_environment_protocol(
        fabricated_revision
    )
    review.write_json(provenance_path, provenance)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["environment_protocol_sha256"] = provenance["environment_protocol"][
        "environment_protocol_sha256"
    ]
    review.write_json(summary_path, summary)
    (output / "examples" / "000.png").write_bytes(
        (output / "examples" / "000.png").read_bytes() + b"tampered"
    )
    review.write_file_manifest(output)

    with pytest.raises(RuntimeError, match="deterministic rebuild"):
        _verify(data)


def test_large_tail_rank_uses_canonical_source_identity():
    canonical_hash = hashlib.sha256(b"one-source").hexdigest()
    assert review._rank(review.RANK_NAMESPACE, canonical_hash) == review._rank(
        review.RANK_NAMESPACE,
        canonical_hash,
    )
    assert review._rank(review.RANK_NAMESPACE, canonical_hash) != review._rank(
        review.REFERENCE_RANK_NAMESPACE,
        canonical_hash,
    )
