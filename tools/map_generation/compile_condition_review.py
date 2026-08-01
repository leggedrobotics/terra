"""Validate one review bundle and emit the explicit accepted-condition set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.map_generation import curriculum_taxonomy

CONDITION_SCHEMA = "terra-condition-review-record-v1"
MAP_SCHEMA = "terra-map-review-record-v2"
OUTPUT_SCHEMA = "terra-accepted-condition-set-v1"
DECISIONS = {"accept", "reject", "quarantine"}
RELEASE_ID = "map-curriculum-diverse64-visual-review-20260730"
MANIFEST_SHA256 = "39f7cd2e8ce565bd384de214da5f2eee5e76764cb554e149c0ba675d815d6d51"
REVIEW_DATA_SHA256 = "8404fcaa9a6b66949ade2b0225d3e7800968951953d2b6363aabffe38100cc0b"
CONDITION_IDS = tuple(row[1] for row in curriculum_taxonomy.SPEC_TABLE_V6_MAIN)
ANCHOR_IDS = frozenset(
    row[1] for row in curriculum_taxonomy.SPEC_TABLE_V6_MAIN if row[2] == 0
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value, payload


def _read_jsonl(path: Path) -> tuple[list[tuple[int, dict]], bytes]:
    payload = path.read_bytes()
    records = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected one JSON object")
        records.append((line_number, value))
    if not records:
        raise ValueError(f"{path}: review bundle is empty")
    return records, payload


def _require_identity(record: dict, release: str, manifest: str, line: int) -> None:
    if record.get("datasetId") != release or record.get("release") != release:
        raise ValueError(f"line {line}: release identity does not match")
    if record.get("manifestHash") != manifest:
        raise ValueError(f"line {line}: manifest hash does not match")


def compile_condition_review(
    review_data_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> dict:
    review_data, review_payload = _read_json(review_data_path)
    records, decisions_payload = _read_jsonl(decisions_path)
    if _sha256(review_payload) != REVIEW_DATA_SHA256:
        raise ValueError("review data bytes do not match the pinned diverse-64 release")

    release_record = review_data.get("release")
    if not isinstance(release_record, dict):
        raise ValueError("review data has no release object")
    release = release_record.get("id")
    manifest = release_record.get("manifestHash")
    if release != RELEASE_ID:
        raise ValueError("review data has the wrong release id")
    if manifest != MANIFEST_SHA256:
        raise ValueError("review data has the wrong manifest hash")

    cells = review_data.get("cells")
    scenarios = review_data.get("scenarios")
    if not isinstance(cells, list) or not cells:
        raise ValueError("review data has no conditions")
    if not isinstance(scenarios, list):
        raise ValueError("review data has no scenarios")
    cell_by_id = {cell.get("id"): cell for cell in cells if isinstance(cell, dict)}
    if len(cell_by_id) != len(cells) or None in cell_by_id:
        raise ValueError("review data has missing or duplicate condition ids")
    if set(cell_by_id) != set(CONDITION_IDS):
        raise ValueError(
            "review data conditions do not match the canonical v6 registry"
        )
    scenario_by_id = {
        scenario.get("id"): scenario
        for scenario in scenarios
        if isinstance(scenario, dict)
    }
    if len(scenario_by_id) != len(scenarios) or None in scenario_by_id:
        raise ValueError("review data has missing or duplicate scenario ids")

    condition_records = {}
    map_records = set()
    for line, record in records:
        schema = record.get("schema")
        _require_identity(record, release, manifest, line)
        if schema == CONDITION_SCHEMA:
            condition_id = record.get("conditionId")
            if condition_id not in cell_by_id:
                raise ValueError(f"line {line}: unknown condition {condition_id!r}")
            if condition_id in condition_records:
                raise ValueError(f"line {line}: duplicate condition {condition_id}")
            if record.get("decision") not in DECISIONS:
                raise ValueError(f"line {line}: invalid condition decision")
            condition_records[condition_id] = record
            continue
        if schema == MAP_SCHEMA:
            scenario_id = record.get("scenarioId")
            if scenario_id not in scenario_by_id:
                raise ValueError(f"line {line}: unknown scenario {scenario_id!r}")
            if scenario_id in map_records:
                raise ValueError(f"line {line}: duplicate scenario {scenario_id}")
            expected_hash = (
                scenario_by_id[scenario_id].get("hashes", {}).get("scenario")
            )
            if not expected_hash or record.get("scenarioHash") != expected_hash:
                raise ValueError(f"line {line}: scenario hash does not match")
            map_records.add(scenario_id)
            continue
        raise ValueError(f"line {line}: unsupported review schema {schema!r}")

    missing = sorted(set(cell_by_id) - set(condition_records))
    if missing:
        raise ValueError(
            f"condition review is incomplete: {len(missing)} missing: {', '.join(missing)}"
        )

    accepted = sorted(
        condition_id
        for condition_id, record in condition_records.items()
        if record["decision"] == "accept"
    )
    if not accepted:
        raise ValueError("condition review accepts no conditions")
    accepted_anchor_families = {
        cell_by_id[condition_id].get("family")
        for condition_id in accepted
        if condition_id in ANCHOR_IDS
    }
    required_anchor_families = {"foundation", "trench"}
    if not required_anchor_families <= accepted_anchor_families:
        missing_families = sorted(required_anchor_families - accepted_anchor_families)
        raise ValueError(
            "accepted conditions require a 00-anchor for: "
            + ", ".join(missing_families)
        )

    dispositions = {
        decision: sorted(
            condition_id
            for condition_id, record in condition_records.items()
            if record["decision"] == decision
        )
        for decision in sorted(DECISIONS)
    }
    receipt = {
        "schema": OUTPUT_SCHEMA,
        "release": release,
        "manifest_sha256": manifest,
        "review_data_sha256": _sha256(review_payload),
        "review_bundle_sha256": _sha256(decisions_payload),
        "condition_count": len(cell_by_id),
        "map_record_count": len(map_records),
        "accepted_conditions": accepted,
        "rejected_conditions": dispositions["reject"],
        "quarantined_conditions": dispositions["quarantine"],
    }

    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "accepted_conditions.txt").write_text(
        ",".join(accepted) + "\n",
        encoding="utf-8",
    )
    (output_path / "review_admission.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile explicit condition decisions into a training-bank list."
    )
    parser.add_argument("--review-data", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = compile_condition_review(args.review_data, args.decisions, args.output)
    print(f"accepted_conditions={len(receipt['accepted_conditions'])}")
    print("--only=" + ",".join(receipt["accepted_conditions"]))


if __name__ == "__main__":
    main()
