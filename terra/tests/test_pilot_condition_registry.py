import copy
import json
from pathlib import Path

import pytest

from tools.verify_pilot_condition_registry import (
    DEFAULT_CONDITIONS,
    DEFAULT_PILOT,
    canonical_jsonl,
    load_json,
    load_jsonl,
    verify_registry,
)


def _write_registry(
    tmp_path: Path,
    pilot: dict,
    rows: list[dict],
) -> tuple[Path, Path]:
    pilot_path = tmp_path / "pilot_v03.json"
    conditions_path = tmp_path / "conditions.jsonl"
    pilot_path.write_text(json.dumps(pilot, indent=2, sort_keys=True) + "\n")
    conditions_path.write_text(canonical_jsonl(rows))
    return pilot_path, conditions_path


def test_frozen_pilot_registry_is_canonical() -> None:
    result = verify_registry(check_receipts=False)

    assert result["condition_count"] == 8
    assert result["anchor_count"] == 3
    assert result["one_axis_count"] == 5
    assert result["composed_count"] == 0
    assert result["family_condition_counts"] == {"foundation": 4, "trench": 4}
    assert result["scenario_count_when_materialized"] == 448
    assert result["similarity_threshold_status"] == "pending_s2_freeze"
    assert result["ppo_authorized"] is False


def test_conditions_jsonl_is_deterministic_projection() -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = load_jsonl(DEFAULT_CONDITIONS)

    assert DEFAULT_CONDITIONS.read_text() == canonical_jsonl(rows)
    assert canonical_jsonl(rows) == canonical_jsonl(pilot["conditions"])


def test_registry_rejects_unresolved_volume_alias(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    rows[2]["factors"]["volume_token"] = "vmatch"
    rows[2]["condition_id"] = rows[2]["condition_id"].replace("v140_189", "vmatch")
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="unresolved condition ID token"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)


def test_registry_rejects_nonliteral_dependency(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    rows[3]["requires"] = rows[0]["requires"]
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="wrong literal prerequisites"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)


def test_registry_rejects_pair_factor_drift(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    original_id = rows[5]["condition_id"]
    rows[5]["factors"]["site_class"] = "light"
    rows[5]["condition_id"] = rows[5]["condition_id"].replace(
        "__site.none", "__site.light"
    )
    for panel in pilot["pair_invariants"]:
        panel["members"] = [
            rows[5]["condition_id"] if member == original_id else member
            for member in panel["members"]
        ]
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="normalized factors changed"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)


def test_registry_rejects_pooled_source_gate(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    rows[1]["source_gate"] = rows[0]["source_gate"]
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="source gate changed"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)


def test_registry_rejects_unequal_condition_weight(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    rows[0]["evaluation_weight"] = 0.5
    rows[1]["evaluation_weight"] = 0.0
    rows[2]["evaluation_weight"] = 0.0
    rows[3]["evaluation_weight"] = 0.0
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="condition weight changed"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)


def test_registry_rejects_missing_panel_member(tmp_path: Path) -> None:
    pilot = load_json(DEFAULT_PILOT)
    rows = copy.deepcopy(pilot["conditions"])
    pilot["pair_invariants"][3]["members"].pop()
    pilot["conditions"] = rows
    pilot_path, conditions_path = _write_registry(tmp_path, pilot, rows)

    with pytest.raises(ValueError, match="wrong condition members"):
        verify_registry(pilot_path, conditions_path, check_receipts=False)
