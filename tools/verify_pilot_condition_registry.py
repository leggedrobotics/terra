#!/usr/bin/env python3
"""Verify the frozen eight-condition Terra pilot registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PILOT = REPO_ROOT / "benchmark" / "pilot_v03.json"
DEFAULT_CONDITIONS = REPO_ROOT / "benchmark" / "conditions.jsonl"
CONDITION_ID_GRAMMAR = (
    "{family}.{topology}.c{component_count}.j{junction_count}"
    "__source.{source_family}"
    "__{dump_layout}.{side_access}.{separation_token}.{capacity_token}.{volume_token}"
    "__site.{site_class}__reset.{reset_mode}"
)


def _expected_factors(
    family: str,
    geometry_class: str,
    topology: str,
    source_family: str,
    dump_layout: str,
    side_access: str,
    separation_token: str,
    capacity_token: str,
    volume_token: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "geometry_class": geometry_class,
        "topology": topology,
        "component_count": 1,
        "junction_count": 0,
        "source_family": source_family,
        "dump_layout": dump_layout,
        "side_access": side_access,
        "separation_token": separation_token,
        "capacity_token": capacity_token,
        "volume_token": volume_token,
        "site_class": "none",
        "reset_mode": "full",
    }


EXPECTED_SPLIT_COUNTS = {
    "public_train": 32,
    "promotion": 8,
    "public_dev": 8,
    "sealed_pilot": 8,
}
EXPECTED_PARENTS_BY_ALIAS = {
    "f.all.osm.sep00_02.slcap20_45.v140_189": [],
    "f.all.procedural.sep00_02.slcap20_45.v140_189": [],
    "f.apron.osm.sep02.slcap07_10.v140_189": ["f.all.osm.sep00_02.slcap20_45.v140_189"],
    "f.apron.osm.sep02.slcap03_04.v140_189": ["f.apron.osm.sep02.slcap07_10.v140_189"],
    "t.straight.both.sep02.slcap03_04.v68_77": [],
    "t.straight.one.sep02.slcap03_04.v68_77": [
        "t.straight.both.sep02.slcap03_04.v68_77"
    ],
    "t.segmented2.both.sep02.slcap03_04.v68_77": [
        "t.straight.both.sep02.slcap03_04.v68_77"
    ],
    "t.segmented3.both.sep02.slcap03_04.v68_77": [
        "t.straight.both.sep02.slcap03_04.v68_77"
    ],
}
EXPECTED_ORIGINAL_ALIAS_BY_RESOLVED = {
    alias: alias for alias in EXPECTED_PARENTS_BY_ALIAS
}
EXPECTED_ORIGINAL_ALIAS_BY_RESOLVED.update(
    {
        "f.apron.osm.sep02.slcap07_10.v140_189": (
            "f.apron.osm.sep02.slcap07_10.vmatch"
        ),
        "f.apron.osm.sep02.slcap03_04.v140_189": (
            "f.apron.osm.sep02.slcap03_04.vmatch"
        ),
    }
)
EXPECTED_ROW_CONTRACTS = {
    "f.all.osm.sep00_02.slcap20_45.v140_189": {
        "factors": _expected_factors(
            "foundation",
            "connected",
            "connected",
            "osm",
            "all_around",
            "all",
            "sep00_02",
            "slcap20_45",
            "v140_189",
        ),
        "source_gate": "foundation_osm_anchor",
        "suite_ids": ["pilot_v03", "foundation_source_panel"],
        "support_receipts": ["foundation_source_support_v3"],
    },
    "f.all.procedural.sep00_02.slcap20_45.v140_189": {
        "factors": _expected_factors(
            "foundation",
            "connected",
            "connected",
            "procedural",
            "all_around",
            "all",
            "sep00_02",
            "slcap20_45",
            "v140_189",
        ),
        "source_gate": "foundation_procedural_anchor",
        "suite_ids": ["pilot_v03", "foundation_source_panel"],
        "support_receipts": ["foundation_source_support_v3"],
    },
    "f.apron.osm.sep02.slcap07_10.v140_189": {
        "factors": _expected_factors(
            "foundation",
            "connected",
            "connected",
            "osm",
            "apron",
            "all",
            "sep02",
            "slcap07_10",
            "v140_189",
        ),
        "source_gate": "foundation_osm_apron_generous",
        "suite_ids": ["pilot_v03", "foundation_capacity_panel"],
        "support_receipts": [
            "foundation_source_support_v3",
            "foundation_capacity_review_v1",
        ],
    },
    "f.apron.osm.sep02.slcap03_04.v140_189": {
        "factors": _expected_factors(
            "foundation",
            "connected",
            "connected",
            "osm",
            "apron",
            "all",
            "sep02",
            "slcap03_04",
            "v140_189",
        ),
        "source_gate": "foundation_osm_apron_constrained",
        "suite_ids": ["pilot_v03", "foundation_capacity_panel"],
        "support_receipts": [
            "foundation_source_support_v3",
            "foundation_capacity_review_v1",
        ],
    },
    "t.straight.both.sep02.slcap03_04.v68_77": {
        "factors": _expected_factors(
            "trench",
            "trench",
            "straight",
            "procedural",
            "side_cast",
            "both",
            "sep02",
            "slcap03_04",
            "v68_77",
        ),
        "source_gate": "trench_straight_both_anchor",
        "suite_ids": [
            "pilot_v03",
            "trench_side_panel",
            "trench_topology_panel",
        ],
        "support_receipts": ["trench_volume_support_v2"],
    },
    "t.straight.one.sep02.slcap03_04.v68_77": {
        "factors": _expected_factors(
            "trench",
            "trench",
            "straight",
            "procedural",
            "side_cast",
            "one",
            "sep02",
            "slcap03_04",
            "v68_77",
        ),
        "source_gate": "trench_straight_one",
        "suite_ids": ["pilot_v03", "trench_side_panel"],
        "support_receipts": ["trench_volume_support_v2"],
    },
    "t.segmented2.both.sep02.slcap03_04.v68_77": {
        "factors": _expected_factors(
            "trench",
            "trench",
            "segmented_2",
            "procedural",
            "side_cast",
            "both",
            "sep02",
            "slcap03_04",
            "v68_77",
        ),
        "source_gate": "trench_segmented_2_both",
        "suite_ids": ["pilot_v03", "trench_topology_panel"],
        "support_receipts": ["trench_volume_support_v2"],
    },
    "t.segmented3.both.sep02.slcap03_04.v68_77": {
        "factors": _expected_factors(
            "trench",
            "trench",
            "segmented_3",
            "procedural",
            "side_cast",
            "both",
            "sep02",
            "slcap03_04",
            "v68_77",
        ),
        "source_gate": "trench_segmented_3_both",
        "suite_ids": ["pilot_v03", "trench_topology_panel"],
        "support_receipts": ["trench_volume_support_v2"],
    },
}
EXPECTED_FACTOR_BANDS = {
    "capacity": {
        "metric": "single_layer_area_ratio",
        "slcap03_04": {"lower_inclusive": 3.0, "upper_inclusive": 4.0},
        "slcap07_10": {"lower_inclusive": 7.0, "upper_inclusive": 10.0},
        "slcap20_45": {"lower_inclusive": 20.0, "upper_inclusive": 45.0},
    },
    "separation": {
        "metric": "p50_tiles",
        "sep00_02": {"lower_inclusive": 0.0, "upper_inclusive": 2.0},
        "sep02": {"lower_inclusive": 1.25, "upper_inclusive": 2.75},
    },
    "volume": {
        "metric": "required_volume",
        "unit": "unit_depth_cell_volume",
        "v140_189": {"lower_inclusive": 140, "upper_inclusive": 189},
        "v68_77": {"lower_inclusive": 68, "upper_inclusive": 77},
    },
}
EXPECTED_PANEL_CONTRACTS = {
    "foundation_source_panel": {
        "cross_family_canonical_dig_equality_rejected": True,
        "exact_pairwise_compactness": True,
        "exact_pairwise_perimeter_4_edges": True,
        "exact_pairwise_required_volume": True,
        "shared_source_group_id": False,
    },
    "foundation_capacity_panel": {
        "exact_pairwise_initial_agent_state": True,
        "exact_pairwise_required_volume": True,
        "exact_pairwise_separation_p50_p95": True,
        "exact_pairwise_target_dig_raster": True,
        "shared_source_group_id": True,
    },
    "trench_side_panel": {
        "exact_pairwise_initial_agent_state": True,
        "exact_pairwise_required_volume": True,
        "exact_pairwise_target_dig_raster": True,
        "shared_source_group_id": True,
    },
    "trench_topology_panel": {
        "exact_pairwise_target_dig_raster": False,
        "fresh_source_groups": True,
        "shared_closed_volume_band": "v68_77",
        "volume_conditioned_comparison": True,
    },
}
EXPECTED_PANEL_MEMBERS_BY_ALIAS = {
    "foundation_source_panel": [
        "f.all.osm.sep00_02.slcap20_45.v140_189",
        "f.all.procedural.sep00_02.slcap20_45.v140_189",
    ],
    "foundation_capacity_panel": [
        "f.apron.osm.sep02.slcap07_10.v140_189",
        "f.apron.osm.sep02.slcap03_04.v140_189",
    ],
    "trench_side_panel": [
        "t.straight.both.sep02.slcap03_04.v68_77",
        "t.straight.one.sep02.slcap03_04.v68_77",
    ],
    "trench_topology_panel": [
        "t.straight.both.sep02.slcap03_04.v68_77",
        "t.segmented2.both.sep02.slcap03_04.v68_77",
        "t.segmented3.both.sep02.slcap03_04.v68_77",
    ],
}
EXPECTED_RECEIPT_IDS = {
    "foundation_capacity_review_v1",
    "foundation_source_support_v3",
    "trench_volume_support_v2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files_manifest(root: Path) -> None:
    manifest = root / "files.sha256"
    listed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        try:
            expected, relative = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"{manifest}:{line_number}: malformed checksum line"
            ) from error
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in listed:
            raise ValueError(f"{manifest}:{line_number}: unsafe or duplicate path")
        listed.add(relative)
        actual = sha256_file(root / path)
        if actual != expected:
            raise ValueError(f"{root / path}: checksum mismatch")

    present = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if listed != present:
        raise ValueError(f"{root}: files.sha256 does not cover the exact file tree")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def canonical_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def condition_id(factors: dict[str, Any]) -> str:
    return (
        f"{factors['family']}.{factors['topology']}"
        f".c{factors['component_count']}.j{factors['junction_count']}"
        f"__source.{factors['source_family']}"
        f"__{factors['dump_layout']}.{factors['side_access']}"
        f".{factors['separation_token']}.{factors['capacity_token']}"
        f".{factors['volume_token']}"
        f"__site.{factors['site_class']}"
        f"__reset.{factors['reset_mode']}"
    )


def _assert_acyclic(rows_by_id: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(current: str) -> None:
        if current in visiting:
            raise ValueError(f"condition dependency cycle includes {current}")
        if current in visited:
            return
        visiting.add(current)
        for parent in rows_by_id[current]["requires"]:
            visit(parent)
        visiting.remove(current)
        visited.add(current)

    for current in rows_by_id:
        visit(current)


def _verify_pair_invariants(
    pilot: dict[str, Any], rows_by_id: dict[str, dict[str, Any]]
) -> None:
    expected_panels = {
        "foundation_source_panel": {"source_family"},
        "foundation_capacity_panel": {"capacity_token"},
        "trench_side_panel": {"side_access"},
        "trench_topology_panel": {"topology"},
    }
    panels = pilot["pair_invariants"]
    if {panel["id"] for panel in panels} != set(expected_panels):
        raise ValueError("pair invariant IDs do not match the frozen four panels")

    for panel in panels:
        members = panel["members"]
        if len(members) != len(set(members)):
            raise ValueError(f"{panel['id']}: duplicate member")
        if any(member not in rows_by_id for member in members):
            raise ValueError(f"{panel['id']}: unknown condition member")
        member_aliases = [rows_by_id[member]["resolved_alias"] for member in members]
        if member_aliases != EXPECTED_PANEL_MEMBERS_BY_ALIAS[panel["id"]]:
            raise ValueError(f"{panel['id']}: wrong condition members")
        factors = [rows_by_id[member]["factors"] for member in members]
        differing_keys = {
            key
            for key in factors[0]
            if any(row[key] != factors[0][key] for row in factors[1:])
        }
        expected_differences = expected_panels[panel["id"]]
        if differing_keys != expected_differences:
            raise ValueError(
                f"{panel['id']}: factor differences {sorted(differing_keys)} "
                f"!= {sorted(expected_differences)}"
            )
        if set(panel["allowed_factor_differences"]) != expected_differences:
            raise ValueError(f"{panel['id']}: stale allowed factor differences")
        if panel["scenario_contract"] != EXPECTED_PANEL_CONTRACTS[panel["id"]]:
            raise ValueError(f"{panel['id']}: stale scenario pairing contract")
        if panel["support_receipt"] not in EXPECTED_RECEIPT_IDS:
            raise ValueError(f"{panel['id']}: unknown support receipt")


def _verify_receipts(pilot: dict[str, Any]) -> None:
    receipts = pilot["support_receipts"]
    if set(receipts) != EXPECTED_RECEIPT_IDS:
        raise ValueError("support receipt set does not match the frozen pilot")

    for receipt_id, receipt in receipts.items():
        root = Path(receipt["artifact_path"])
        if not root.is_dir():
            raise FileNotFoundError(f"{receipt_id}: missing artifact {root}")
        for relative, expectation in receipt["files"].items():
            path = root / relative
            digest = sha256_file(path)
            if digest != expectation["sha256"]:
                raise ValueError(
                    f"{receipt_id}/{relative}: SHA-256 {digest} "
                    f"!= {expectation['sha256']}"
                )
            if "line_count" in expectation:
                line_count = len(path.read_text().splitlines())
                if line_count != expectation["line_count"]:
                    raise ValueError(
                        f"{receipt_id}/{relative}: {line_count} lines "
                        f"!= {expectation['line_count']}"
                    )
        verify_files_manifest(root)

    foundation = load_json(
        Path(receipts["foundation_source_support_v3"]["artifact_path"])
        / "support_summary.json"
    )
    frozen = foundation["frozen_contract"]
    if foundation["status"] != "passed" or foundation["held_out_selection_performed"]:
        raise ValueError("foundation receipt is not passed train-only support")
    if (
        frozen["required_volume_lower_inclusive"],
        frozen["required_volume_upper_inclusive"],
    ) != (140, 189):
        raise ValueError("foundation receipt does not freeze v140_189")
    if (
        frozen["compactness_lower_inclusive"],
        frozen["compactness_upper_inclusive"],
    ) != (0.3, 0.65):
        raise ValueError("foundation compactness support changed")
    if frozen["matched_pair_count"] != 32:
        raise ValueError("foundation receipt does not contain 32 matched pairs")
    if not frozen["cross_family_canonical_dig_equality_rejected"]:
        raise ValueError("foundation exact cross-family collision gate is disabled")

    trench = load_json(
        Path(receipts["trench_volume_support_v2"]["artifact_path"])
        / "support_summary.json"
    )
    selected = trench["selected_band"]
    if trench["status"] != "passed" or trench["seed_namespace"] != "train_only":
        raise ValueError("trench receipt is not passed train-only support")
    if (
        selected["lower_inclusive"],
        selected["upper_inclusive"],
    ) != (68, 77):
        raise ValueError("trench receipt does not freeze v68_77")
    if trench["proposal_count_per_topology"] != 20_000:
        raise ValueError("trench proposal count changed")
    if min(selected["support_rate"].values()) < 0.10:
        raise ValueError("trench support falls below the frozen 10% gate")

    capacity = load_json(
        Path(receipts["foundation_capacity_review_v1"]["artifact_path"])
        / "summary.json"
    )
    if not capacity["exact_loader_format_valid"]:
        raise ValueError("capacity review is not exact-loader valid")
    if capacity["pair_count"] != 32:
        raise ValueError("capacity review does not contain 32 pairs")
    if not capacity["all_pairs_share_exact_dig_and_volume"]:
        raise ValueError("capacity review lost exact dig/volume pairing")
    if capacity["s1_capacity_gate_complete"]:
        raise ValueError("registry must not claim the open Static capacity gate")
    deltas = capacity["pairwise_separation_delta_high_minus_low"]
    if any(value != 0.0 for stats in deltas.values() for value in stats.values()):
        raise ValueError("capacity counterfactuals changed achieved separation")


def verify_registry(
    pilot_path: Path = DEFAULT_PILOT,
    conditions_path: Path = DEFAULT_CONDITIONS,
    *,
    check_receipts: bool,
) -> dict[str, Any]:
    pilot = load_json(pilot_path)
    rows = load_jsonl(conditions_path)
    if pilot["schema"] != "terra_benchmark_pilot_registry_v1":
        raise ValueError("wrong pilot registry schema")
    if pilot["suite_id"] != "terramap-pilot-v0.3":
        raise ValueError("wrong pilot suite ID")
    if pilot["release_id"] != "terramap-bench-v1.0.0":
        raise ValueError("wrong benchmark release ID")
    if pilot["condition_registry_path"] != "benchmark/conditions.jsonl":
        raise ValueError("wrong canonical condition registry path")
    if pilot["condition_id_grammar"] != CONDITION_ID_GRAMMAR:
        raise ValueError("condition ID grammar changed")
    if pilot["split_counts_per_condition"] != EXPECTED_SPLIT_COUNTS:
        raise ValueError("top-level split counts changed")
    if pilot["factor_bands"] != EXPECTED_FACTOR_BANDS:
        raise ValueError("numeric factor bands changed")
    if pilot["protocol"] != {
        "apply_trench_rewards": False,
        "max_steps_in_episode": 450,
        "reset_mode": "full",
        "rewards_type": "DENSE",
    }:
        raise ValueError("pilot map-curriculum protocol changed")
    if pilot["source_spec"] != {
        "path": "MAP_BENCHMARK_SPEC.md",
        "terra_revision": "50ee15ddd8296f2293c85e0242d4004c02f63929",
    }:
        raise ValueError("pilot source authority changed")
    embedded = pilot["conditions"]
    if canonical_jsonl(rows) != canonical_jsonl(embedded):
        raise ValueError("conditions.jsonl is not the canonical pilot registry")
    if len(rows) != 8:
        raise ValueError(f"expected 8 pilot conditions, found {len(rows)}")

    ids = [row["condition_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("condition IDs are not unique")
    aliases = [row["resolved_alias"] for row in rows]
    if set(aliases) != set(EXPECTED_PARENTS_BY_ALIAS):
        raise ValueError("resolved aliases do not match the accepted pilot")

    rows_by_id = {row["condition_id"]: row for row in rows}
    id_by_alias = {row["resolved_alias"]: row["condition_id"] for row in rows}
    for row in rows:
        alias = row["resolved_alias"]
        expected_row = EXPECTED_ROW_CONTRACTS[alias]
        if row["schema"] != "terra_benchmark_condition_v1":
            raise ValueError(f"{alias}: wrong condition schema")
        expected_id = condition_id(row["factors"])
        if row["condition_id"] != expected_id:
            raise ValueError(f"{alias}: condition ID is not mechanical")
        if "vmatch" in row["condition_id"] or "null" in row["condition_id"]:
            raise ValueError(f"{alias}: unresolved condition ID token")
        if "vmatch" in row["resolved_alias"] or "null" in row["resolved_alias"]:
            raise ValueError(f"{alias}: unresolved alias token")
        if row["factors"] != expected_row["factors"]:
            raise ValueError(f"{alias}: normalized factors changed")
        if row["source_gate"] != expected_row["source_gate"]:
            raise ValueError(f"{alias}: source gate changed")
        if row["suite_ids"] != expected_row["suite_ids"]:
            raise ValueError(f"{alias}: suite membership changed")
        if row["evaluation_weight"] != 0.125:
            raise ValueError(f"{alias}: condition weight changed")
        provenance = row["provenance"]
        if provenance != {
            "original_candidate_alias": EXPECTED_ORIGINAL_ALIAS_BY_RESOLVED[alias],
            "original_candidate_alias_is_identity": False,
        }:
            raise ValueError(f"{alias}: wrong alias provenance")
        if row["expected_counts"] != EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"{alias}: wrong split counts")
        if row["expected_unique_source_group_counts"] != EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"{alias}: wrong unique-source counts")
        expected_parents = [
            id_by_alias[parent] for parent in EXPECTED_PARENTS_BY_ALIAS[alias]
        ]
        if row["requires"] != expected_parents:
            raise ValueError(f"{alias}: wrong literal prerequisites")
        expected_depth = "anchor" if not expected_parents else "one_axis"
        if row["display_depth"] != expected_depth:
            raise ValueError(f"{alias}: wrong display depth")

        factors = row["factors"]
        if factors["family"] == "foundation":
            if factors["volume_token"] != "v140_189":
                raise ValueError("foundation condition escaped v140_189 support")
            compactness = row["support"].get("compactness")
            if compactness is None or (
                compactness["lower_inclusive"],
                compactness["upper_inclusive"],
            ) != (0.3, 0.65):
                raise ValueError("foundation compactness support is not frozen")
        elif factors["family"] == "trench":
            if factors["volume_token"] != "v68_77":
                raise ValueError("trench condition escaped v68_77 support")
        else:
            raise ValueError(f"unsupported family: {factors['family']}")

        receipts = row["support"]["receipts"]
        if receipts != expected_row["support_receipts"]:
            raise ValueError(f"{alias}: bad support receipt reference")
        bands = pilot["factor_bands"]
        for band_type, token_key in (
            ("capacity", "capacity_token"),
            ("separation", "separation_token"),
            ("volume", "volume_token"),
        ):
            if factors[token_key] not in bands[band_type]:
                raise ValueError(f"{alias}: unknown {band_type} band")

    _assert_acyclic(rows_by_id)
    _verify_pair_invariants(pilot, rows_by_id)

    family_weights = {"foundation": 0.0, "trench": 0.0}
    for row in rows:
        family_weights[row["factors"]["family"]] += row["evaluation_weight"]
    if family_weights != {"foundation": 0.5, "trench": 0.5}:
        raise ValueError(f"unbalanced family weights: {family_weights}")

    similarity = pilot["similarity_policy"]
    if similarity["frozen_exact_rules"] != {
        "cross_family_canonical_dig_equality_rejected": True,
        "source_groups_must_be_disjoint_across_splits": True,
        "within_source_counterfactuals_must_share_split": True,
    }:
        raise ValueError("exact similarity rules changed")
    pending = similarity["thresholded_cross_split_similarity"]
    if (
        pending["status"] != "pending_s2_freeze"
        or not pending["blocks_s2_materialization"]
        or pending["metric"] != "maximum_centered_dihedral_iou"
        or pending["foundation_threshold"] != "UNSET"
        or pending["trench_threshold"] != "UNSET"
    ):
        raise ValueError("thresholded similarity must remain an explicit S2 blocker")

    status = pilot["admission_status"]
    if (
        not status["condition_registry_frozen"]
        or status["scenario_bank_materialized"]
        or status["static_validated"]
        or status["witnessed"]
        or status["ppo_authorized"]
    ):
        raise ValueError("registry overclaims scenario admission")

    if check_receipts:
        _verify_receipts(pilot)

    return {
        "schema": "terra_pilot_condition_registry_verification_v1",
        "status": "passed",
        "condition_count": len(rows),
        "anchor_count": sum(row["display_depth"] == "anchor" for row in rows),
        "one_axis_count": sum(row["display_depth"] == "one_axis" for row in rows),
        "composed_count": sum(row["display_depth"] == "composed" for row in rows),
        "family_condition_counts": {
            family: sum(row["factors"]["family"] == family for row in rows)
            for family in ("foundation", "trench")
        },
        "scenario_count_when_materialized": sum(
            sum(row["expected_counts"].values()) for row in rows
        ),
        "receipts_checked": check_receipts,
        "similarity_threshold_status": pending["status"],
        "ppo_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite benchmark/conditions.jsonl from benchmark/pilot_v03.json.",
    )
    parser.add_argument(
        "--check-receipts",
        action="store_true",
        help="Hash and inspect the frozen external train-only receipts.",
    )
    args = parser.parse_args()

    if args.write:
        pilot = load_json(DEFAULT_PILOT)
        DEFAULT_CONDITIONS.write_text(canonical_jsonl(pilot["conditions"]))
    result = verify_registry(check_receipts=args.check_receipts)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
