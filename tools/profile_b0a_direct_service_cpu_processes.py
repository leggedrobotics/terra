#!/usr/bin/env python3
"""Profile all frozen B0a identities with four exact CPU workers."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import tools.probe_b0a_direct_service_cpu_processes as process_probe

SCHEMA = "terra_direct_service_cpu_256_profile_v1"
OUTPUT_NAME = "validation_cost.json"
RESULTS_NAME = "direct_service_results.jsonl"
CANDIDATE_RESULTS_NAME = "direct_service_results.candidate.jsonl"
WORKER_MODULE = "tools.profile_b0a_direct_service_cpu_processes"

# Root pins this only after the single R-58 v2 cohort passes. UNSET makes the
# expensive profile impossible to launch from an unreviewed scaling result.
EXPECTED_CPU_PROCESS_PROBE_SHA256 = "UNSET"
EXPECTED_CPU_CONFIRMATION_SHA256 = process_probe.EXPECTED_CPU_CONFIRMATION_SHA256
EXPECTED_MIGRATION_VALIDATION_SHA256 = (
    "ce14b52e330cd734997f3c269b85d58a93ab24f67fc1e3f1499af0ecc5228b37"
)
EXPECTED_MIGRATION_SUMMARY_SHA256 = (
    "7ec113fa87705c239819162ce954b8ead014bc6c8d1d11dd2fb3ed8e8a57300f"
)
EXPECTED_B0A_IDENTITIES_SHA256 = (
    "911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e"
)
EXPECTED_B0A_FILES_MANIFEST_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)
EXPECTED_B0A_SOURCE_REGISTRY_SHA256 = (
    "1ffe22f8c3ed4cc608fc8fc9a5106f2ecd26d8e122d042d0d2630b025a293a8d"
)
EXPECTED_B0A_PROVENANCE_SHA256 = (
    "6d011ebe0787e555df7c5b53d990010d97f5fd886ea4280d33d5fea62cf31e37"
)
EXPECTED_MIGRATION_SCRIPT_SHA256 = (
    "6d963a82bfcf5b81ebe35072ef7527b41bf578948d0fe2d527dd5f1e3e3ea322"
)
EXPECTED_BENCHMARK_STATE_SHA256 = (
    "06b810b1088bd198e2fd30d661d38b1768b12a4e0a96e3575268861be6d4e355"
)
EXPECTED_ENV_CONFIG_SHA256 = (
    "02863f625923a6f1302a0fe8f09fc9840b0ef92bf84f68bf2ea645d50b460072"
)
EXPECTED_ENVIRONMENT_PROTOCOL_SHA256 = (
    "15e4d45f846dfa8f567c70611cc3af7a9248ccba6401567660823a7ba8fdb6fd"
)
EXPECTED_MIGRATION_TERRA_REVISION = "affc0d9216de6fa5748d5822c09f4c28feba8c43"
EXPECTED_SENTINEL_MAP_ID = "b0a-train-f_apron_d02-00"
EXPECTED_IDENTITY_COUNT = 256
EXPECTED_SOURCE_GROUP_COUNT = 144
EXPECTED_SOURCE_GROUP_SIZE_COUNTS = {1: 112, 4: 16, 5: 16}
EXPECTED_SPLIT_COUNTS = {"public_dev": 128, "public_train": 128}
LEGACY_TO_BENCHMARK_SPLIT = {
    "development": "public_dev",
    "train": "public_train",
}

WORKER_COUNT = process_probe.WORKER_COUNT
AFFINITY_SETS = process_probe.AFFINITY_SETS
SCENARIOS_PER_WORKER = EXPECTED_IDENTITY_COUNT // WORKER_COUNT
MAX_SCENARIO_SECONDS = 60 * 60
PROFILE_HARD_TIMEOUT_SECONDS = 24 * 60 * 60
PROJECTION_448_LIMIT_SECONDS = 48 * 60 * 60
MAX_MEMORY_FRACTION = process_probe.MAX_MEMORY_FRACTION
EXTRA_SCENARIOS_PER_WORKER_FOR_448 = (448 - EXPECTED_IDENTITY_COUNT) // WORKER_COUNT

EXTRA_CODE_PATHS = (
    "terra/benchmark_state.py",
    "tools/migrate_b0a_live_geometry.py",
    "tools/profile_b0a_direct_service_cpu_processes.py",
)

_WORKER_INDEX_ENV = "_TERRA_CPU_PROFILE_WORKER_INDEX"
_B0A_ROOT_ENV = "_TERRA_CPU_PROFILE_B0A_ROOT"
_PROCESS_PROBE_ENV = "_TERRA_CPU_PROFILE_PROCESS_PROBE"
_CONFIRMATION_ENV = "_TERRA_CPU_PROFILE_CONFIRMATION"
_SHARD_ENV = "_TERRA_CPU_PROFILE_SHARD"
_RESULT_ENV = "_TERRA_CPU_PROFILE_WORKER_RESULT"
_READY_ENV = "_TERRA_CPU_PROFILE_WORKER_READY"
_BARRIER_ENV = "_TERRA_CPU_PROFILE_START_BARRIER"
_SCENARIO_DIRECTORY_ENV = "_TERRA_CPU_PROFILE_SCENARIO_DIRECTORY"
_EXPECTED_CODE_BUNDLE_ENV = "_TERRA_CPU_PROFILE_CODE_BUNDLE"
_INTERNAL_ENV_KEYS = (
    _WORKER_INDEX_ENV,
    _B0A_ROOT_ENV,
    _PROCESS_PROBE_ENV,
    _CONFIRMATION_ENV,
    _SHARD_ENV,
    _RESULT_ENV,
    _READY_ENV,
    _BARRIER_ENV,
    _SCENARIO_DIRECTORY_ENV,
    _EXPECTED_CODE_BUNDLE_ENV,
)


class ProfileRejected(RuntimeError):
    """The one authorized 256-identity profile did not pass."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object.")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSON in {path} at line {line_number}: {error}."
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number} must be a JSON object.")
        rows.append(value)
    return rows


def _write_jsonl_once(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x") as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


_write_json_once_no_replace = process_probe._write_json_once


def _publish_existing_file_no_replace(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY)
    try:
        os.fsync(source_fd)
    finally:
        os.close(source_fd)
    os.link(source, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _persist_and_verify_candidate(
    path: Path,
    rows: list[dict[str, Any]],
) -> str:
    _write_jsonl_once(path, rows)
    persisted = _load_jsonl(path)
    if process_probe._typed_canonical(persisted) != process_probe._typed_canonical(
        rows
    ):
        raise RuntimeError("Persisted canonical candidate failed verification.")
    return process_probe._sha256_file(path)


def _existing_file_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if evidence["exists"]:
        try:
            evidence["sha256"] = process_probe._sha256_file(path)
        except Exception as error:
            evidence["sha256_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
    return evidence


def _require_sha256(path: Path, expected: str, label: str) -> str:
    actual = process_probe._sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 changed: {actual} != {expected}.")
    return actual


def _authorized_runtime_receipt(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    workers = authorization.get("workers")
    if not isinstance(workers, list) or len(workers) != WORKER_COUNT:
        raise RuntimeError("R-58 authorization omitted its four worker receipts.")
    receipts = []
    for worker_index, worker in enumerate(workers):
        device = worker.get("device", {}) if isinstance(worker, dict) else {}
        packages = device.get("packages")
        receipt = {
            "python_executable": device.get("python_executable"),
            "packages": packages,
        }
        if (
            not isinstance(receipt["python_executable"], str)
            or not receipt["python_executable"]
            or not isinstance(packages, dict)
            or set(packages) != {"jax", "jaxlib", "numpy", "scipy"}
            or any(
                value is not None and not isinstance(value, str)
                for value in packages.values()
            )
        ):
            raise RuntimeError(
                f"R-58 worker {worker_index} has an incomplete runtime receipt."
            )
        receipts.append(receipt)
    if any(receipt != receipts[0] for receipt in receipts[1:]):
        raise RuntimeError("R-58 workers used different Python/package runtimes.")
    return receipts[0]


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            digest, relative = line.split("  ", maxsplit=1)
        except ValueError as error:
            raise RuntimeError(
                f"{path}:{line_number} is not a sha256sum manifest row."
            ) from error
        relative_path = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in entries
        ):
            raise RuntimeError(f"{path}:{line_number} is not a canonical entry.")
        entries[relative] = digest
    if not entries:
        raise RuntimeError(f"{path} is empty.")
    return entries


def _verified_manifest_entries(root: Path) -> dict[str, str]:
    manifest_path = root / "files.sha256"
    entries = _parse_sha256_manifest(manifest_path)
    verified = {}
    for relative, expected in sorted(entries.items()):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"Manifest entry escapes the B0a root: {relative}."
            ) from error
        actual = process_probe._sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen B0a manifest entry changed for {relative}: "
                f"{actual} != {expected}."
            )
        verified[relative] = actual
    return verified


def _exact_file_receipt(
    named_paths: Mapping[str, tuple[Path, str]],
) -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": str(path),
            "sha256": _require_sha256(path, expected, name),
        }
        for name, (path, expected) in named_paths.items()
    }


def _selected_file_contract(
    b0a_root: Path,
    population: list[dict[str, Any]],
    manifest_entries: Mapping[str, str],
) -> dict[str, list[str]]:
    manifest_rows_by_directory: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, list[str]] = {}
    for profile_row in population:
        raw = profile_row["legacy_identity"]
        legacy_map_id = profile_row["migration"]["legacy_map_id"]
        relative_directory = f"cells/{raw['split']}/{raw['primary_cell']}"
        if relative_directory not in manifest_rows_by_directory:
            manifest_rows_by_directory[relative_directory] = _load_jsonl(
                b0a_root / relative_directory / "manifest.jsonl"
            )
        matches = [
            row
            for row in manifest_rows_by_directory[relative_directory]
            if row.get("map_id") == legacy_map_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"{legacy_map_id} has {len(matches)} rows in its cell manifest."
            )
        row = matches[0]
        slot_index = row.get("slot_index")
        if (
            isinstance(slot_index, bool)
            or not isinstance(slot_index, int)
            or slot_index <= 0
        ):
            raise RuntimeError(f"{legacy_map_id} has an invalid slot index.")
        for field in ("source_id", "split", "family", "primary_cell"):
            if row.get(field) != raw.get(field):
                raise RuntimeError(
                    f"{legacy_map_id} disagrees with its cell manifest for {field}."
                )
        paths = [
            f"{relative_directory}/dataset.json",
            f"{relative_directory}/manifest.jsonl",
            f"{relative_directory}/images/img_{slot_index}.npy",
            f"{relative_directory}/occupancy/img_{slot_index}.npy",
            f"{relative_directory}/dumpability/img_{slot_index}.npy",
            f"{relative_directory}/actions/img_{slot_index}.npy",
            f"{relative_directory}/distance/img_{slot_index}.npy",
            f"{relative_directory}/metadata/trench_{slot_index}.json",
        ]
        missing = [path for path in paths if path not in manifest_entries]
        if missing:
            raise RuntimeError(
                f"{legacy_map_id} selected files are absent from files.sha256: "
                f"{missing}."
            )
        selected[legacy_map_id] = paths
    if len(selected) != EXPECTED_IDENTITY_COUNT:
        raise RuntimeError("Selected-file contract is incomplete.")
    return selected


def _pinned_input_receipt(
    b0a_root: Path,
    migration_root: Path,
) -> dict[str, dict[str, str]]:
    pinned_paths = {
        "b0a_identities": (
            b0a_root / "identities.jsonl",
            EXPECTED_B0A_IDENTITIES_SHA256,
        ),
        "b0a_files_manifest": (
            b0a_root / "files.sha256",
            EXPECTED_B0A_FILES_MANIFEST_SHA256,
        ),
        "b0a_source_registry": (
            b0a_root / "source_registry.jsonl",
            EXPECTED_B0A_SOURCE_REGISTRY_SHA256,
        ),
        "b0a_provenance": (
            b0a_root / "provenance.json",
            EXPECTED_B0A_PROVENANCE_SHA256,
        ),
        "migration_validation": (
            migration_root / "migration_validation.jsonl",
            EXPECTED_MIGRATION_VALIDATION_SHA256,
        ),
        "migration_summary": (
            migration_root / "migration_summary.json",
            EXPECTED_MIGRATION_SUMMARY_SHA256,
        ),
    }
    return _exact_file_receipt(pinned_paths)


def _frozen_input_snapshot(
    b0a_root: Path,
    population: list[dict[str, Any]],
    pinned_files: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    manifest_entries = _verified_manifest_entries(b0a_root)
    selected_files = _selected_file_contract(
        b0a_root,
        population,
        manifest_entries,
    )
    snapshot = {
        "pinned_files": pinned_files,
        "manifest_entries": manifest_entries,
        "selected_files_by_legacy_map_id": selected_files,
    }
    snapshot["snapshot_sha256"] = process_probe._canonical_json_sha256(snapshot)
    return snapshot


def _compact_input_receipt(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    manifest_entries = snapshot["manifest_entries"]
    selected_files = snapshot["selected_files_by_legacy_map_id"]
    return {
        "pinned_files": snapshot["pinned_files"],
        "manifest_entry_count": len(manifest_entries),
        "manifest_entries_sha256": process_probe._canonical_json_sha256(
            manifest_entries
        ),
        "selected_identity_count": len(selected_files),
        "selected_file_contract_sha256": process_probe._canonical_json_sha256(
            selected_files
        ),
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }


def _code_receipt(repository: Path) -> dict[str, Any]:
    base = process_probe._code_receipt(repository)
    hashes = dict(base["code_file_sha256"])
    for relative in EXTRA_CODE_PATHS:
        hashes[relative] = process_probe._sha256_file(repository / relative)
    return {
        "git": base["git"],
        "code_file_sha256": dict(sorted(hashes.items())),
        "r58_code_bundle_sha256": base["code_bundle_sha256"],
        "code_bundle_sha256": process_probe._canonical_json_sha256(hashes),
    }


def _load_process_authorization(path: Path) -> tuple[dict[str, Any], str]:
    if EXPECTED_CPU_PROCESS_PROBE_SHA256 == "UNSET":
        raise RuntimeError(
            "EXPECTED_CPU_PROCESS_PROBE_SHA256 is UNSET; a reviewed passing "
            "R-58 v2 receipt must be pinned before this profile can execute."
        )
    digest = _require_sha256(
        path,
        EXPECTED_CPU_PROCESS_PROBE_SHA256,
        "CPU process-probe receipt",
    )
    receipt = _load_json(path)
    checks = {
        "schema": receipt.get("schema") == process_probe.SCHEMA,
        "release": receipt.get("release_id") == process_probe.EXPECTED_RELEASE_ID,
        "status": receipt.get("status") == "passed",
        "treatment": (
            receipt.get("experiment", {}).get("treatment")
            == "fixed_four_long_lived_cpu_workers_two_calls_each"
        ),
        "worker count": (
            receipt.get("experiment", {}).get("worker_count") == WORKER_COUNT
        ),
        "exactness": (
            receipt.get("gates", {}).get(
                "all_eight_outcomes_and_counters_match_confirmation"
            )
            is True
        ),
        "all gates": (
            receipt.get("gates", {}).get("all_process_treatment_gates_pass") is True
        ),
        "authorization": (
            receipt.get("decision", {}).get(
                "authorizes_one_deterministic_four_worker_256_profile"
            )
            is True
        ),
        "narrow authorization only": (
            receipt.get("decision", {}).get("authorizes_bank_profile") is False
            and receipt.get("decision", {}).get("authorizes_static_admission") is False
            and receipt.get("decision", {}).get("authorizes_ppo") is False
        ),
        "profile limits": (
            receipt.get("decision", {}).get("authorized_profile_must_finish_within_24h")
            is True
            and receipt.get("decision", {}).get(
                "authorized_profile_must_project_448_within_48h"
            )
            is True
        ),
        "no retry": (receipt.get("decision", {}).get("authorizes_retry") is False),
        "confirmation": (
            receipt.get("input", {}).get("cpu_confirmation_receipt_sha256")
            == EXPECTED_CPU_CONFIRMATION_SHA256
        ),
    }
    failed = [name for name, passes in checks.items() if not passes]
    if failed:
        raise RuntimeError(
            "CPU process-probe receipt does not authorize this profile: "
            + ", ".join(failed)
            + "."
        )
    return receipt, digest


def _validate_authorized_code(
    authorization: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, str]:
    authorized = authorization["validator"]["code_before"]["code_file_sha256"]
    current_hashes = current["code_file_sha256"]
    missing = sorted(set(authorized) - set(current_hashes))
    changed = sorted(
        path
        for path, digest in authorized.items()
        if path in current_hashes and current_hashes[path] != digest
    )
    if missing or changed:
        raise RuntimeError(
            "Exact execution dependencies changed after R-58: "
            f"missing={missing}, changed={changed}."
        )
    if current_hashes["terra/benchmark_state.py"] != EXPECTED_BENCHMARK_STATE_SHA256:
        raise RuntimeError("terra/benchmark_state.py changed after migration.")
    if (
        current_hashes["tools/migrate_b0a_live_geometry.py"]
        != EXPECTED_MIGRATION_SCRIPT_SHA256
    ):
        raise RuntimeError("The migration implementation changed.")
    return {path: current_hashes[path] for path in sorted(authorized)}


def _load_population(
    b0a_root: Path,
    migration_root: Path,
    pinned_files: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities_path = b0a_root / "identities.jsonl"
    migration_path = migration_root / "migration_validation.jsonl"
    summary_path = migration_root / "migration_summary.json"
    expected_pinned_names = {
        "b0a_identities",
        "b0a_files_manifest",
        "b0a_source_registry",
        "b0a_provenance",
        "migration_validation",
        "migration_summary",
    }
    if set(pinned_files) != expected_pinned_names:
        raise RuntimeError("Pinned population input receipt is incomplete.")
    receipts = {
        f"{name}_sha256": pinned_files[name]["sha256"]
        for name in sorted(expected_pinned_names)
    }
    summary = _load_json(summary_path)
    if (
        summary.get("schema") != "terra_b0a_live_geometry_migration_summary_v1"
        or summary.get("status") != "pending_exact_static"
        or summary.get("identity_count") != EXPECTED_IDENTITY_COUNT
        or summary.get("source_group_count") != EXPECTED_SOURCE_GROUP_COUNT
        or summary.get("output", {}).get("migration_validation_sha256")
        != EXPECTED_MIGRATION_VALIDATION_SHA256
        or summary.get("source", {}).get("identity_manifest_sha256")
        != EXPECTED_B0A_IDENTITIES_SHA256
        or summary.get("source", {}).get("files_sha256_sha256")
        != EXPECTED_B0A_FILES_MANIFEST_SHA256
        or summary.get("source", {}).get("source_registry_sha256")
        != EXPECTED_B0A_SOURCE_REGISTRY_SHA256
        or summary.get("source", {}).get("provenance_sha256")
        != EXPECTED_B0A_PROVENANCE_SHA256
        or summary.get("implementation", {}).get("migration_script_sha256")
        != EXPECTED_MIGRATION_SCRIPT_SHA256
        or summary.get("implementation", {}).get("env_config_sha256")
        != EXPECTED_ENV_CONFIG_SHA256
        or summary.get("implementation", {}).get("environment_protocol_sha256")
        != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
        or summary.get("implementation", {}).get("terra_revision")
        != EXPECTED_MIGRATION_TERRA_REVISION
    ):
        raise RuntimeError("Migration summary no longer matches the frozen profile.")

    raw_rows = _load_jsonl(identities_path)
    raw_by_map = {row.get("map_id"): row for row in raw_rows}
    if len(raw_rows) != EXPECTED_IDENTITY_COUNT or len(raw_by_map) != len(raw_rows):
        raise RuntimeError("Frozen B0a identities are missing or duplicated.")

    migration_rows = _load_jsonl(migration_path)
    legacy_ids = [row.get("legacy_map_id") for row in migration_rows]
    if (
        len(migration_rows) != EXPECTED_IDENTITY_COUNT
        or any(not isinstance(value, str) for value in legacy_ids)
        or legacy_ids != sorted(legacy_ids)
        or len(set(legacy_ids)) != EXPECTED_IDENTITY_COUNT
        or set(legacy_ids) != set(raw_by_map)
    ):
        raise RuntimeError(
            "Migration population must be the sorted one-to-one frozen B0a join."
        )

    scenario_ids = set()
    map_ids = set()
    state_hashes_by_group: dict[str, set[str]] = {}
    state_records_by_group: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    group_sizes: dict[str, int] = {}
    population = []
    for index, row in enumerate(migration_rows):
        legacy_map_id = row["legacy_map_id"]
        raw = raw_by_map[legacy_map_id]
        scenario = row.get("scenario")
        audit = row.get("audit")
        if (
            row.get("migration_status") != "pending_exact_static"
            or row.get("direct_service_status") != "direct_service_blocked_by_cost_gate"
            or row.get("errors") != []
            or not isinstance(scenario, dict)
            or not isinstance(audit, dict)
            or audit.get("validation", {}).get("initial_state_valid") is not True
            or audit.get("validation", {}).get("exact_capacity_valid") is not True
            or audit.get("validation", {}).get("static_valid") is not None
        ):
            raise RuntimeError(f"{legacy_map_id} is not a valid pending migration row.")
        source_group_id = row.get("source_group_id")
        initial = scenario.get("initial_condition", {})
        state_record = initial.get("initial_agent_state")
        state_hash = initial.get("initial_agent_state_sha256")
        benchmark_split = LEGACY_TO_BENCHMARK_SPLIT.get(raw.get("split"))
        if (
            scenario.get("legacy_map_id") != legacy_map_id
            or scenario.get("source_group_id") != raw.get("source_id")
            or source_group_id != raw.get("source_id")
            or scenario.get("family") != raw.get("family")
            or scenario.get("split") != benchmark_split
            or scenario.get("condition_id") is not None
            or scenario.get("environment_protocol_sha256")
            != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
            or not isinstance(state_record, dict)
            or not isinstance(state_hash, str)
            or initial.get("seed_receipt", {}).get("split") != benchmark_split
            or initial.get("seed_receipt", {}).get("source_group_id") != source_group_id
            or initial.get("seed_receipt", {}).get("initial_agent_state_sha256")
            != state_hash
            or initial.get("environment_reset_seed")
            != initial.get("seed_receipt", {}).get("seed_uint32")
            or initial.get("initial_soil_sha256")
            != scenario.get("layers", {}).get("initial_soil_sha256")
            or scenario.get("factor_vector", {}).get("required_volume")
            != audit.get("work", {}).get("required_volume")
        ):
            raise RuntimeError(f"{legacy_map_id} identity or state join changed.")
        scenario_id = scenario.get("scenario_id")
        map_id = scenario.get("map_id")
        if (
            not isinstance(scenario_id, str)
            or scenario_id in scenario_ids
            or not isinstance(map_id, str)
            or map_id in map_ids
        ):
            raise RuntimeError(f"{legacy_map_id} canonical IDs are duplicated.")
        scenario_ids.add(scenario_id)
        map_ids.add(map_id)
        state_hashes_by_group.setdefault(source_group_id, set()).add(state_hash)
        state_records_by_group.setdefault(source_group_id, set()).add(
            process_probe._canonical_json_sha256(state_record)
        )
        split_counts[benchmark_split] = split_counts.get(benchmark_split, 0) + 1
        group_sizes[source_group_id] = group_sizes.get(source_group_id, 0) + 1
        population.append(
            {
                "global_index": index,
                "legacy_identity": raw,
                "migration": row,
            }
        )

    size_counts: dict[int, int] = {}
    for size in group_sizes.values():
        size_counts[size] = size_counts.get(size, 0) + 1
    if (
        split_counts != EXPECTED_SPLIT_COUNTS
        or len(group_sizes) != EXPECTED_SOURCE_GROUP_COUNT
        or size_counts != EXPECTED_SOURCE_GROUP_SIZE_COUNTS
        or any(len(values) != 1 for values in state_hashes_by_group.values())
        or any(len(values) != 1 for values in state_records_by_group.values())
    ):
        raise RuntimeError(
            "Frozen split/source-group/state population invariants changed."
        )
    sentinel = [
        row
        for row in population
        if row["migration"]["legacy_map_id"] == EXPECTED_SENTINEL_MAP_ID
    ]
    if len(sentinel) != 1 or sentinel[0]["global_index"] % WORKER_COUNT != 0:
        raise RuntimeError("The confirmation sentinel no longer belongs to worker 0.")
    return population, {
        **receipts,
        "identity_count": len(population),
        "source_group_count": len(group_sizes),
        "source_group_size_counts": {
            str(size): count for size, count in sorted(size_counts.items())
        },
        "split_counts": dict(sorted(split_counts.items())),
        "canonical_order": "migration_validation.jsonl lexicographic legacy_map_id",
        "canonical_legacy_map_id_sha256": process_probe._canonical_json_sha256(
            legacy_ids
        ),
    }


def _build_shards(
    population: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    if len(population) != EXPECTED_IDENTITY_COUNT:
        raise ValueError(f"Expected {EXPECTED_IDENTITY_COUNT} profile rows.")
    shards = [
        [row for row in population if row["global_index"] % WORKER_COUNT == index]
        for index in range(WORKER_COUNT)
    ]
    if any(len(shard) != SCENARIOS_PER_WORKER for shard in shards):
        raise RuntimeError("Round-robin shards are not exactly 64 identities each.")
    assigned = sorted(row["global_index"] for shard in shards for row in shard)
    if assigned != list(range(EXPECTED_IDENTITY_COUNT)):
        raise RuntimeError("Round-robin shards are missing or duplicating identities.")
    for worker_index, shard in enumerate(shards):
        if any(row["global_index"] % WORKER_COUNT != worker_index for row in shard):
            raise RuntimeError(f"Worker {worker_index} shard assignment changed.")
    return shards


def _nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Nearest-rank input cannot be empty.")
    if not 0 < quantile <= 1:
        raise ValueError("Nearest-rank quantile must lie in (0, 1].")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError("Nearest-rank values must be finite and positive.")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _timing_summary(
    workers: list[dict[str, Any]],
    execution_makespan_seconds: float,
) -> dict[str, Any]:
    if len(workers) != WORKER_COUNT:
        raise ValueError("Timing summary requires four worker ledgers.")
    durations = []
    per_worker = {}
    for worker_index, worker in enumerate(
        sorted(workers, key=lambda row: row["worker_index"])
    ):
        ledger = sorted(
            worker.get("identity_timing_ledger", []),
            key=lambda row: row["processing_position"],
        )
        if (
            len(ledger) != SCENARIOS_PER_WORKER
            or [row["processing_position"] for row in ledger]
            != list(range(SCENARIOS_PER_WORKER))
            or any(row["worker_index"] != worker_index for row in ledger)
        ):
            raise RuntimeError(f"Worker {worker_index} timing ledger is incomplete.")
        worker_durations = [float(row["identity_duration_seconds"]) for row in ledger]
        durations.extend(worker_durations)
        warm = worker_durations[1:]
        per_worker[str(worker_index)] = {
            "identity_count": len(worker_durations),
            "post_first_identity_count": len(warm),
            "post_first_nearest_rank_p95_seconds": _nearest_rank(warm, 0.95),
        }
    if len(durations) != EXPECTED_IDENTITY_COUNT:
        raise RuntimeError("Expected exactly 256 identity durations.")
    maximum_worker_warm_p95 = max(
        row["post_first_nearest_rank_p95_seconds"] for row in per_worker.values()
    )
    projected_448 = (
        execution_makespan_seconds
        + EXTRA_SCENARIOS_PER_WORKER_FOR_448 * maximum_worker_warm_p95
    )
    return {
        "per_identity_observation_count": 1,
        "quantile_method": "nearest_rank",
        "identity_duration_boundary": (
            "exact map/state load through synchronized exact outcome and "
            "durable atomic no-replace scenario-receipt publication"
        ),
        "scenario_total_seconds": {
            "p50": _nearest_rank(durations, 0.50),
            "p95": _nearest_rank(durations, 0.95),
            "max": max(durations),
        },
        "per_worker": per_worker,
        "maximum_worker_post_first_p95_seconds": maximum_worker_warm_p95,
        "execution_makespan_256_seconds": execution_makespan_seconds,
        "execution_makespan_boundary": (
            "immediately before first worker spawn through successful exits, "
            "verification of all 256 receipts, and durable candidate merge "
            "plus exact reload verification"
        ),
        "projection_448": {
            "method": (
                "execution_makespan_256 + "
                "48 * max_i(nearest_rank_p95(worker_i_post_first_durations))"
            ),
            "additional_scenarios_per_worker": (EXTRA_SCENARIOS_PER_WORKER_FOR_448),
            "projected_makespan_seconds": projected_448,
            "limit_seconds": PROJECTION_448_LIMIT_SECONDS,
            "passes": projected_448 <= PROJECTION_448_LIMIT_SECONDS,
            "cost_extrapolation_only": True,
        },
    }


def _scenario_result_path(directory: Path, global_index: int) -> Path:
    return directory / f"scenario_{global_index:04d}.json"


def _scenario_started_path(directory: Path, global_index: int) -> Path:
    return directory / f"scenario_{global_index:04d}.started.json"


def _worker_processing_order(
    worker_index: int,
    shard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = sorted(shard, key=lambda row: row["global_index"])
    if worker_index != 0:
        return ordered
    sentinel = [
        row
        for row in ordered
        if row["migration"]["legacy_map_id"] == EXPECTED_SENTINEL_MAP_ID
    ]
    if len(sentinel) != 1:
        raise RuntimeError("Worker 0 shard is missing the exactness sentinel.")
    return sentinel + [row for row in ordered if row is not sentinel[0]]


def _worker_import_origins(
    repository: Path,
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "terra.benchmark_direct_service": "terra/benchmark_direct_service.py",
        "terra.benchmark_protocol": "terra/benchmark_protocol.py",
        "terra.benchmark_state": "terra/benchmark_state.py",
        "tools.migrate_b0a_live_geometry": "tools/migrate_b0a_live_geometry.py",
        "tools.profile_b0a_direct_service_cost": (
            "tools/profile_b0a_direct_service_cost.py"
        ),
        "tools.probe_b0a_direct_service_cpu_processes": (
            "tools/probe_b0a_direct_service_cpu_processes.py"
        ),
    }
    if set(modules) != set(expected):
        raise RuntimeError("Profile worker module-origin set changed.")
    origins = {}
    for name, relative in expected.items():
        observed = Path(modules[name].__file__).resolve()
        expected_path = (repository / relative).resolve()
        if observed != expected_path:
            raise RuntimeError(
                f"{name} resolved outside this worktree: {observed} != "
                f"{expected_path}."
            )
        origins[name] = {
            "observed_path": str(observed),
            "expected_path": str(expected_path),
            "matches": True,
        }
    return {
        "origins": origins,
        "all_origins_match_exact_paths": True,
        "pythonpath_override": os.environ.get("PYTHONPATH"),
    }


def _worker_device_receipt(jax: Any, devices: list[Any]) -> dict[str, Any]:
    return {
        "device_count": len(devices),
        "selected_device": str(devices[0]),
        "platform": devices[0].platform,
        "default_backend": jax.default_backend(),
        "device_kind": getattr(devices[0], "device_kind", None),
        "python_executable": sys.executable,
        "packages": {
            name: process_probe._package_version(name)
            for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }


def _worker_main() -> int:
    worker_index = int(os.environ[_WORKER_INDEX_ENV])
    result_path = Path(os.environ[_RESULT_ENV])
    ready_path = Path(os.environ[_READY_ENV])
    barrier_path = Path(os.environ[_BARRIER_ENV])
    shard_path = Path(os.environ[_SHARD_ENV])
    scenario_directory = Path(os.environ[_SCENARIO_DIRECTORY_ENV])
    affinity = set(AFFINITY_SETS[worker_index])
    receipt: dict[str, Any] = {
        "schema": f"{SCHEMA}_worker_v1",
        "status": "failed",
        "worker_index": worker_index,
        "pid": os.getpid(),
        "started_unix_seconds": time.time(),
    }
    monitor: Any = None
    monitor_receipt: dict[str, Any] | None = None
    persisted: list[Path] = []
    identity_timing_ledger: list[dict[str, Any]] = []
    try:
        os.sched_setaffinity(0, affinity)
        affinity_before = sorted(os.sched_getaffinity(0))
        if affinity_before != sorted(affinity):
            raise RuntimeError(f"Worker {worker_index} affinity changed.")
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            raise RuntimeError("Profile workers require JAX_PLATFORMS=cpu.")
        forbidden = {
            key: os.environ[key]
            for key in process_probe._DISALLOWED_RUNTIME_ENV
            if key in os.environ
        }
        if forbidden:
            raise RuntimeError(
                f"Profile worker inherited forbidden flags: {forbidden}."
            )

        monitor = process_probe._MemoryMonitor()
        monitor.start()
        repository = Path(__file__).resolve().parents[1]
        code_before = _code_receipt(repository)
        if code_before["code_bundle_sha256"] != os.environ[_EXPECTED_CODE_BUNDLE_ENV]:
            raise RuntimeError("Worker code differs from coordinator preflight.")

        import jax
        import jax.numpy as jnp
        import numpy as np

        import terra.benchmark_direct_service as direct_service
        import terra.benchmark_protocol as benchmark_protocol
        import terra.benchmark_state as benchmark_state
        import tools.migrate_b0a_live_geometry as migration_tool
        import tools.profile_b0a_direct_service_cost as profile_tool
        from terra.env import TerraEnv

        module_origins = _worker_import_origins(
            repository,
            {
                "terra.benchmark_direct_service": direct_service,
                "terra.benchmark_protocol": benchmark_protocol,
                "terra.benchmark_state": benchmark_state,
                "tools.migrate_b0a_live_geometry": migration_tool,
                "tools.profile_b0a_direct_service_cost": profile_tool,
                "tools.probe_b0a_direct_service_cpu_processes": process_probe,
            },
        )
        devices = jax.devices()
        if (
            len(devices) != 1
            or devices[0].platform != "cpu"
            or jax.default_backend() != "cpu"
        ):
            raise RuntimeError(f"Worker requires one CPU JAX device, got {devices}.")
        device_receipt = _worker_device_receipt(jax, devices)

        process_authorization, process_authorization_sha256 = (
            _load_process_authorization(Path(os.environ[_PROCESS_PROBE_ENV]))
        )
        authorized_runtime = _authorized_runtime_receipt(process_authorization)
        if {
            "python_executable": device_receipt["python_executable"],
            "packages": device_receipt["packages"],
        } != authorized_runtime:
            raise RuntimeError("Worker Python/package runtime differs from R-58.")
        confirmation, confirmation_sha256 = process_probe._load_reference(
            Path(os.environ[_CONFIRMATION_ENV])
        )
        _validate_authorized_code(process_authorization, code_before)
        shard = _load_jsonl(shard_path)
        if len(shard) != SCENARIOS_PER_WORKER or any(
            row.get("global_index", -1) % WORKER_COUNT != worker_index for row in shard
        ):
            raise RuntimeError(f"Worker {worker_index} shard changed.")
        processing_order = _worker_processing_order(worker_index, shard)
        b0a_root = Path(os.environ[_B0A_ROOT_ENV])
        env_config, env_config_receipt = profile_tool._frozen_env_config()
        if env_config_receipt.get("env_config_sha256") != EXPECTED_ENV_CONFIG_SHA256:
            raise RuntimeError("Current frozen environment config changed.")
        execution_environment_protocol = benchmark_protocol.frozen_environment_protocol(
            code_before["git"]["head"]
        )
        if (
            execution_environment_protocol.get("env_config_sha256")
            != EXPECTED_ENV_CONFIG_SHA256
            or execution_environment_protocol.get("terra_revision")
            != code_before["git"]["head"]
        ):
            raise RuntimeError("Current execution protocol is not code-bound.")
        migration_environment_protocol = benchmark_protocol.frozen_environment_protocol(
            EXPECTED_MIGRATION_TERRA_REVISION
        )
        if (
            migration_environment_protocol.get("environment_protocol_sha256")
            != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
        ):
            raise RuntimeError(
                "Current environment constants do not reproduce migration protocol."
            )
        migration_protocol_identity = {
            "terra_revision": EXPECTED_MIGRATION_TERRA_REVISION,
            "environment_protocol_sha256": EXPECTED_ENVIRONMENT_PROTOCOL_SHA256,
            "env_config_sha256": EXPECTED_ENV_CONFIG_SHA256,
            "historical_migration_evidence_only": True,
        }
        migration_protocol_receipt = {
            **migration_protocol_identity,
            "environment_protocol": migration_environment_protocol,
        }
        reference_outcome = confirmation["measurement"]["exact_outcome"]
        thread_affinity_before = process_probe._thread_affinity_receipt(affinity_before)
        process_probe._write_json_once(
            ready_path,
            {
                "schema": f"{SCHEMA}_worker_ready_v1",
                "worker_index": worker_index,
                "pid": os.getpid(),
                "affinity_cpus": affinity_before,
                "thread_affinity": thread_affinity_before,
                "device": device_receipt,
                "module_origins": module_origins,
                "code_bundle_sha256": code_before["code_bundle_sha256"],
                "shard_sha256": process_probe._sha256_file(shard_path),
                "migration_protocol": migration_protocol_receipt,
                "execution_environment_protocol": execution_environment_protocol,
                "ready_unix_seconds": time.time(),
            },
        )
        while not barrier_path.is_file():
            time.sleep(0.05)

        for processing_position, profile_row in enumerate(processing_order):
            global_index = profile_row["global_index"]
            migration = profile_row["migration"]
            legacy_map_id = migration["legacy_map_id"]
            started_path = _scenario_started_path(
                scenario_directory,
                global_index,
            )
            result_file = _scenario_result_path(scenario_directory, global_index)
            process_probe._write_json_once(
                started_path,
                {
                    "schema": f"{SCHEMA}_scenario_started_v1",
                    "global_index": global_index,
                    "worker_index": worker_index,
                    "processing_position": processing_position,
                    "legacy_map_id": legacy_map_id,
                    "pid": os.getpid(),
                    "started_unix_seconds": time.time(),
                },
            )
            scenario_started = time.perf_counter()
            try:
                raw = profile_row["legacy_identity"]
                scenario = profile_tool._load_scenario(b0a_root, raw)
                verified_files = profile_tool._verify_selected_inputs(
                    b0a_root,
                    set(scenario.selected_files),
                )
                scenario_record = migration["scenario"]
                initial = scenario_record["initial_condition"]
                agent = benchmark_state.agent_from_record(
                    initial["initial_agent_state"]
                )
                state_hash = benchmark_state.agent_state_sha256(agent)
                if state_hash != initial["initial_agent_state_sha256"]:
                    raise RuntimeError(
                        f"{legacy_map_id} serialized initial-agent hash changed."
                    )
                seed, seed_digest = benchmark_state.derive_initial_state_seed(
                    scenario_record["release_id"],
                    scenario_record["split"],
                    scenario_record["source_group_id"],
                    0,
                )
                seed_receipt = initial["seed_receipt"]
                if (
                    seed != initial["environment_reset_seed"]
                    or seed != seed_receipt["seed_uint32"]
                    or seed_digest != seed_receipt["seed_digest_sha256"]
                ):
                    raise RuntimeError(f"{legacy_map_id} seed namespace changed.")
                benchmark_state.validate_benchmark_initial_agent(
                    agent,
                    env_cfg=env_config,
                    padding_mask=scenario.padding_mask,
                    action_map=scenario.action_map,
                    dumpability_mask=scenario.dumpability_mask,
                )
                load_and_validate_finished = time.perf_counter()
                env = TerraEnv.new(maps_size_px=profile_tool.MAP_SIZE)
                timestep = env.reset(
                    jax.random.PRNGKey(seed),
                    jnp.asarray(scenario.target),
                    jnp.asarray(scenario.padding_mask),
                    jnp.asarray(scenario.trench_axes),
                    jnp.asarray(scenario.trench_type),
                    jnp.asarray(scenario.foundation_border_axes),
                    jnp.asarray(scenario.foundation_border_type),
                    jnp.asarray(scenario.dumpability_mask),
                    jnp.asarray(scenario.action_map),
                    jnp.asarray(scenario.distance_map),
                    env_config,
                    agent,
                )
                profile_tool._synchronize(timestep)
                if (
                    benchmark_state.agent_state_sha256(timestep.state.agent)
                    != state_hash
                    or int(np.asarray(jax.device_get(timestep.state.env_steps))) != 0
                ):
                    raise RuntimeError(f"{legacy_map_id} explicit reset changed state.")
                reset_finished = time.perf_counter()
                exact_outcome = direct_service.compute_initial_direct_service(
                    timestep.state
                )
                profile_tool._synchronize(exact_outcome)
                exact_finished = time.perf_counter()
                exact_outcome = profile_tool._jsonable(exact_outcome)
                if not isinstance(exact_outcome, dict) or set(exact_outcome) != set(
                    profile_tool.DIRECT_SERVICE_OUTCOME_KEYS
                ):
                    raise RuntimeError(f"{legacy_map_id} exact-outcome schema changed.")
                expected_volume = scenario_record["factor_vector"]["required_volume"]
                if exact_outcome["required_volume"] != expected_volume:
                    raise RuntimeError(
                        f"{legacy_map_id} required volume changed in exact replay."
                    )
                if legacy_map_id == EXPECTED_SENTINEL_MAP_ID and (
                    process_probe._typed_canonical(exact_outcome)
                    != process_probe._typed_canonical(reference_outcome)
                ):
                    raise RuntimeError(
                        "The serialized-state confirmation sentinel differs."
                    )
                through_outcome_seconds = exact_finished - scenario_started
                scenario_receipt = {
                    "schema": f"{SCHEMA}_scenario_v1",
                    "status": "passed",
                    "global_index": global_index,
                    "worker_index": worker_index,
                    "processing_position": processing_position,
                    "legacy_map_id": legacy_map_id,
                    "scenario_id": scenario_record["scenario_id"],
                    "map_id": scenario_record["map_id"],
                    "source_group_id": scenario_record["source_group_id"],
                    "split": scenario_record["split"],
                    "family": scenario_record["family"],
                    "initial_agent_state_sha256": state_hash,
                    "environment_reset_seed": seed,
                    "protocol_provenance": {
                        "migration": migration_protocol_identity,
                        "execution": {
                            "terra_revision": execution_environment_protocol[
                                "terra_revision"
                            ],
                            "environment_protocol_sha256": (
                                execution_environment_protocol[
                                    "environment_protocol_sha256"
                                ]
                            ),
                            "env_config_sha256": env_config_receipt[
                                "env_config_sha256"
                            ],
                            "code_bundle_sha256": code_before["code_bundle_sha256"],
                        },
                    },
                    "verified_input_file_sha256": verified_files,
                    "timing": {
                        "load_and_validate_seconds": (
                            load_and_validate_finished - scenario_started
                        ),
                        "explicit_reset_seconds": (
                            reset_finished - load_and_validate_finished
                        ),
                        "exact_entrypoint_seconds": (exact_finished - reset_finished),
                        "through_synchronized_outcome_seconds": (
                            through_outcome_seconds
                        ),
                    },
                    "exact_outcome": exact_outcome,
                    "typed_exact_outcome_sha256": (
                        process_probe._typed_canonical_sha256(exact_outcome)
                    ),
                    "confirmation_sentinel": (
                        legacy_map_id == EXPECTED_SENTINEL_MAP_ID
                    ),
                    "memory_after": process_probe._process_memory(),
                }
                _write_json_once_no_replace(result_file, scenario_receipt)
                identity_finished = time.perf_counter()
                persisted.append(result_file)
                identity_duration = identity_finished - scenario_started
                identity_timing_ledger.append(
                    {
                        "global_index": global_index,
                        "worker_index": worker_index,
                        "processing_position": processing_position,
                        "legacy_map_id": legacy_map_id,
                        "identity_duration_seconds": identity_duration,
                        "ends_after_atomic_scenario_receipt_publication": True,
                    }
                )
                if identity_duration > MAX_SCENARIO_SECONDS:
                    raise RuntimeError(
                        f"{legacy_map_id} exceeded the one-hour scenario gate."
                    )
            except Exception as error:
                if not result_file.exists():
                    _write_json_once_no_replace(
                        result_file,
                        {
                            "schema": f"{SCHEMA}_scenario_v1",
                            "status": "failed",
                            "global_index": global_index,
                            "worker_index": worker_index,
                            "processing_position": processing_position,
                            "legacy_map_id": legacy_map_id,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        },
                    )
                    persisted.append(result_file)
                raise

        affinity_after = sorted(os.sched_getaffinity(0))
        if affinity_after != affinity_before:
            raise RuntimeError(f"Worker {worker_index} affinity changed after work.")
        thread_affinity_after = process_probe._thread_affinity_receipt(affinity_after)
        code_after = _code_receipt(repository)
        if code_after != code_before:
            raise RuntimeError(f"Worker {worker_index} code changed during execution.")
        device_after = _worker_device_receipt(jax, devices)
        if device_after != device_receipt:
            raise RuntimeError(
                f"Worker {worker_index} runtime changed during execution."
            )
        monitor_receipt = monitor.stop()
        monitor = None
        if monitor_receipt["transient_swap_observed"]:
            raise RuntimeError(f"Worker {worker_index} used swap.")
        receipt.update(
            {
                "status": "passed",
                "finished_unix_seconds": time.time(),
                "affinity_before": affinity_before,
                "affinity_after": affinity_after,
                "thread_affinity_before": thread_affinity_before,
                "thread_affinity_after": thread_affinity_after,
                "device": device_after,
                "device_before": device_receipt,
                "runtime_unchanged": True,
                "module_origins": module_origins,
                "process_probe_receipt_sha256": process_authorization_sha256,
                "confirmation_receipt_sha256": confirmation_sha256,
                "protocol_provenance": {
                    "migration": migration_protocol_receipt,
                    "execution_environment": execution_environment_protocol,
                    "execution_env_config": env_config_receipt,
                    "migration_and_execution_hashes_are_not_assumed_equal": True,
                    "static_validity_claimed": False,
                },
                "validator": {
                    "code_before": code_before,
                    "code_after": code_after,
                    "exact_entrypoint": (
                        "terra.benchmark_direct_service."
                        "compute_initial_direct_service"
                    ),
                    "exact_entrypoint_call_count": len(persisted),
                },
                "shard": {
                    "path": str(shard_path),
                    "sha256": process_probe._sha256_file(shard_path),
                    "scenario_count": len(shard),
                },
                "scenario_receipt_sha256": {
                    str(path): process_probe._sha256_file(path) for path in persisted
                },
                "identity_timing_ledger": identity_timing_ledger,
                "memory_monitor": monitor_receipt,
                "process_peak_rss_kib": monitor_receipt["maximum_kib"]["ru_maxrss_kib"],
                "zero_swap": True,
            }
        )
    except Exception as error:
        if monitor is not None:
            try:
                monitor_receipt = monitor.stop()
            except Exception as monitor_error:
                monitor_receipt = {
                    "monitor_stop_error": {
                        "type": type(monitor_error).__name__,
                        "message": str(monitor_error),
                    }
                }
        receipt.update(
            {
                "finished_unix_seconds": time.time(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "memory_monitor": monitor_receipt,
                "persisted_scenario_receipt_sha256": {
                    str(path): process_probe._sha256_file(path)
                    for path in persisted
                    if path.is_file()
                },
                "identity_timing_ledger": identity_timing_ledger,
            }
        )
    process_probe._write_json_once(result_path, receipt)
    return 0 if receipt["status"] == "passed" else 1


def _worker_command() -> list[str]:
    return [sys.executable, "-m", WORKER_MODULE]


def _load_worker_receipt(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema") != f"{SCHEMA}_worker_v1":
        raise RuntimeError(f"Unexpected worker receipt schema in {path}.")
    return value


def _load_scenario_receipts(
    scenario_directory: Path,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    by_index = {}
    for path in sorted(scenario_directory.glob("scenario_????.json")):
        row = _load_json(path)
        index = row.get("global_index")
        if not isinstance(index, int) or not 0 <= index < EXPECTED_IDENTITY_COUNT:
            raise RuntimeError(f"{path} has an invalid global index.")
        if index in by_index:
            raise RuntimeError(f"Scenario index {index} is duplicated.")
        by_index[index] = row
    completed = sorted(
        index for index, row in by_index.items() if row.get("status") == "passed"
    )
    failed = sorted(
        index for index, row in by_index.items() if row.get("status") != "passed"
    )
    ordered = [by_index[index] for index in completed]
    return ordered, failed, sorted(set(range(EXPECTED_IDENTITY_COUNT)) - set(by_index))


def _validate_environment_protocol(
    protocol: Mapping[str, Any],
    expected_revision: str,
) -> None:
    digest = protocol.get("environment_protocol_sha256")
    payload = {
        key: value
        for key, value in protocol.items()
        if key != "environment_protocol_sha256"
    }
    if (
        protocol.get("schema") != "terra_environment_protocol_v1"
        or protocol.get("release_id") != process_probe.EXPECTED_RELEASE_ID
        or protocol.get("terra_revision") != expected_revision
        or protocol.get("env_config_sha256") != EXPECTED_ENV_CONFIG_SHA256
        or digest != process_probe._canonical_json_sha256(payload)
    ):
        raise RuntimeError("Environment protocol receipt is invalid.")


def _validated_exact_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    outcome = result.get("exact_outcome")
    if (
        not isinstance(outcome, dict)
        or set(outcome) != process_probe.DIRECT_SERVICE_OUTCOME_KEYS
        or result.get("typed_exact_outcome_sha256")
        != process_probe._typed_canonical_sha256(outcome)
    ):
        raise RuntimeError("Scenario exact-outcome receipt is invalid.")
    return outcome


def _validate_complete_results(
    workers: list[dict[str, Any]],
    scenario_results: list[dict[str, Any]],
    population: list[dict[str, Any]],
    *,
    expected_pids: list[int],
    expected_code_bundle_sha256: str,
    expected_process_probe_sha256: str,
    expected_runtime_receipt: Mapping[str, Any],
    expected_execution_revision: str,
    input_snapshot: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    scenario_directory: Path,
    shard_paths: list[Path],
) -> dict[str, Any]:
    if len(workers) != WORKER_COUNT or len(scenario_results) != EXPECTED_IDENTITY_COUNT:
        raise RuntimeError("Profile evidence is incomplete.")
    workers = sorted(workers, key=lambda row: row["worker_index"])
    if [row["worker_index"] for row in workers] != list(range(WORKER_COUNT)):
        raise RuntimeError("Worker indices are missing or duplicated.")
    if any(row.get("status") != "passed" for row in workers):
        raise RuntimeError("At least one profile worker failed.")
    pids = [row["pid"] for row in workers]
    if pids != expected_pids or len(set(pids)) != WORKER_COUNT:
        raise RuntimeError("Profile workers did not retain four distinct PIDs.")
    worker_protocols = [
        worker.get("protocol_provenance", {}).get("execution_environment")
        for worker in workers
    ]
    if any(not isinstance(protocol, dict) for protocol in worker_protocols):
        raise RuntimeError("A worker omitted its current execution protocol.")
    for protocol in worker_protocols:
        _validate_environment_protocol(
            protocol,
            expected_execution_revision,
        )
    if any(
        process_probe._typed_canonical(protocol)
        != process_probe._typed_canonical(worker_protocols[0])
        for protocol in worker_protocols[1:]
    ):
        raise RuntimeError("Workers derived different execution protocols.")
    execution_protocol = worker_protocols[0]
    migration_protocol_identity = {
        "terra_revision": EXPECTED_MIGRATION_TERRA_REVISION,
        "environment_protocol_sha256": EXPECTED_ENVIRONMENT_PROTOCOL_SHA256,
        "env_config_sha256": EXPECTED_ENV_CONFIG_SHA256,
        "historical_migration_evidence_only": True,
    }
    migration_protocols = [
        worker.get("protocol_provenance", {}).get("migration") for worker in workers
    ]
    if any(not isinstance(protocol, dict) for protocol in migration_protocols):
        raise RuntimeError("A worker omitted its historical migration protocol.")
    for protocol in migration_protocols:
        environment_protocol = protocol.get("environment_protocol", {})
        _validate_environment_protocol(
            environment_protocol,
            EXPECTED_MIGRATION_TERRA_REVISION,
        )
        if (
            environment_protocol.get("environment_protocol_sha256")
            != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
            or {
                key: value
                for key, value in protocol.items()
                if key != "environment_protocol"
            }
            != migration_protocol_identity
        ):
            raise RuntimeError("Historical migration protocol receipt changed.")
    if any(
        process_probe._typed_canonical(protocol)
        != process_probe._typed_canonical(migration_protocols[0])
        for protocol in migration_protocols[1:]
    ):
        raise RuntimeError("Workers derived different migration protocols.")
    migration_protocol = migration_protocols[0]
    expected_by_index = {row["global_index"]: row for row in population}
    if [row["global_index"] for row in scenario_results] != list(
        range(EXPECTED_IDENTITY_COUNT)
    ):
        raise RuntimeError("Scenario results are not in complete canonical order.")
    sentinel_count = 0
    for result in scenario_results:
        index = result["global_index"]
        expected = expected_by_index[index]
        worker_index = index % WORKER_COUNT
        scenario = expected["migration"]["scenario"]
        initial = scenario["initial_condition"]
        expected_sentinel = (
            expected["migration"]["legacy_map_id"] == EXPECTED_SENTINEL_MAP_ID
        )
        expected_selected_files = {
            relative: input_snapshot["manifest_entries"][relative]
            for relative in input_snapshot["selected_files_by_legacy_map_id"][
                expected["migration"]["legacy_map_id"]
            ]
        }
        exact_outcome = _validated_exact_outcome(result)
        protocol_provenance = result.get("protocol_provenance", {})
        expected_execution_receipt = {
            "terra_revision": expected_execution_revision,
            "environment_protocol_sha256": execution_protocol[
                "environment_protocol_sha256"
            ],
            "env_config_sha256": EXPECTED_ENV_CONFIG_SHA256,
            "code_bundle_sha256": expected_code_bundle_sha256,
        }
        if (
            result.get("schema") != f"{SCHEMA}_scenario_v1"
            or result.get("status") != "passed"
            or result.get("worker_index") != worker_index
            or result.get("legacy_map_id") != expected["migration"]["legacy_map_id"]
            or result.get("scenario_id") != scenario["scenario_id"]
            or result.get("map_id") != scenario["map_id"]
            or result.get("source_group_id") != scenario["source_group_id"]
            or result.get("split") != scenario["split"]
            or result.get("family") != scenario["family"]
            or result.get("initial_agent_state_sha256")
            != initial["initial_agent_state_sha256"]
            or result.get("environment_reset_seed") != initial["environment_reset_seed"]
            or protocol_provenance.get("migration") != migration_protocol_identity
            or protocol_provenance.get("execution") != expected_execution_receipt
            or exact_outcome.get("required_volume")
            != scenario["factor_vector"]["required_volume"]
            or result.get("verified_input_file_sha256") != expected_selected_files
            or result.get("confirmation_sentinel") is not expected_sentinel
            or result.get("memory_after", {}).get("VmSwap") != 0
        ):
            raise RuntimeError(f"Scenario result {index} violates its frozen contract.")
        if expected_sentinel:
            sentinel_count += 1
            if process_probe._typed_canonical(
                exact_outcome
            ) != process_probe._typed_canonical(
                confirmation["measurement"]["exact_outcome"]
            ):
                raise RuntimeError("Confirmation sentinel result changed.")
    if sentinel_count != 1:
        raise RuntimeError("Profile must contain exactly one confirmation sentinel.")
    result_by_index = {row["global_index"]: row for row in scenario_results}
    for index, worker in enumerate(workers):
        thread_before = worker.get("thread_affinity_before", {})
        thread_after = worker.get("thread_affinity_after", {})
        module_origins = worker.get("module_origins", {})
        shard_receipt = worker.get("shard", {})
        expected_shard_path = shard_paths[index]
        expected_scenario_paths = {
            str(_scenario_result_path(scenario_directory, global_index))
            for global_index in range(index, EXPECTED_IDENTITY_COUNT, WORKER_COUNT)
        }
        scenario_hashes = worker.get("scenario_receipt_sha256", {})
        timing_ledger = worker.get("identity_timing_ledger", [])
        expected_worker_indices = set(
            range(index, EXPECTED_IDENTITY_COUNT, WORKER_COUNT)
        )
        ledger_indices = {row.get("global_index") for row in timing_ledger}
        timing_ledger_valid = (
            len(timing_ledger) == SCENARIOS_PER_WORKER
            and ledger_indices == expected_worker_indices
            and sorted(row.get("processing_position") for row in timing_ledger)
            == list(range(SCENARIOS_PER_WORKER))
            and all(
                row.get("worker_index") == index
                and row.get("ends_after_atomic_scenario_receipt_publication") is True
                and isinstance(row.get("identity_duration_seconds"), float)
                and math.isfinite(row["identity_duration_seconds"])
                and row["identity_duration_seconds"] > 0
                and row.get("legacy_map_id")
                == result_by_index[row["global_index"]]["legacy_map_id"]
                and row.get("processing_position")
                == result_by_index[row["global_index"]]["processing_position"]
                for row in timing_ledger
            )
        )
        worker_protocol = worker.get("protocol_provenance", {})
        worker_runtime = {
            "python_executable": worker.get("device", {}).get("python_executable"),
            "packages": worker.get("device", {}).get("packages"),
        }
        if (
            worker.get("pid") != expected_pids[index]
            or worker.get("affinity_before") != list(AFFINITY_SETS[index])
            or worker.get("affinity_after") != list(AFFINITY_SETS[index])
            or worker.get("device", {}).get("platform") != "cpu"
            or worker.get("device", {}).get("default_backend") != "cpu"
            or worker.get("device", {}).get("device_count") != 1
            or worker_runtime != expected_runtime_receipt
            or worker.get("device_before") != worker.get("device")
            or worker.get("runtime_unchanged") is not True
            or worker.get("zero_swap") is not True
            or worker.get("memory_monitor", {}).get("transient_swap_observed")
            is not False
            or worker.get("process_probe_receipt_sha256")
            != expected_process_probe_sha256
            or worker.get("confirmation_receipt_sha256")
            != EXPECTED_CPU_CONFIRMATION_SHA256
            or thread_before.get("all_worker_threads_within_fixed_cpuset") is not True
            or thread_after.get("all_worker_threads_within_fixed_cpuset") is not True
            or thread_before.get("expected_cpus") != list(AFFINITY_SETS[index])
            or thread_after.get("expected_cpus") != list(AFFINITY_SETS[index])
            or module_origins.get("all_origins_match_exact_paths") is not True
            or module_origins.get("pythonpath_override") is not None
            or worker_protocol.get("migration") != migration_protocol
            or worker_protocol.get("execution_environment") != execution_protocol
            or worker_protocol.get("execution_env_config", {}).get("env_config_sha256")
            != EXPECTED_ENV_CONFIG_SHA256
            or worker_protocol.get(
                "migration_and_execution_hashes_are_not_assumed_equal"
            )
            is not True
            or worker_protocol.get("static_validity_claimed") is not False
            or shard_receipt.get("path") != str(expected_shard_path)
            or shard_receipt.get("sha256")
            != process_probe._sha256_file(expected_shard_path)
            or shard_receipt.get("scenario_count") != SCENARIOS_PER_WORKER
            or worker.get("validator", {})
            .get("code_before", {})
            .get("code_bundle_sha256")
            != expected_code_bundle_sha256
            or worker.get("validator", {}).get("code_after")
            != worker.get("validator", {}).get("code_before")
            or worker.get("validator", {}).get("exact_entrypoint_call_count")
            != SCENARIOS_PER_WORKER
            or len(worker.get("scenario_receipt_sha256", {})) != SCENARIOS_PER_WORKER
            or set(scenario_hashes) != expected_scenario_paths
            or any(
                process_probe._sha256_file(Path(path)) != digest
                for path, digest in scenario_hashes.items()
            )
            or not timing_ledger_valid
        ):
            raise RuntimeError(f"Worker {index} receipt violates the frozen profile.")
    return {
        "migration_environment": migration_protocol["environment_protocol"],
        "execution_environment": execution_protocol,
    }


def _run_workers(
    *,
    b0a_root: Path,
    process_probe_path: Path,
    confirmation_path: Path,
    output_directory: Path,
    shard_paths: list[Path],
    code_bundle_sha256: str,
    expected_runtime_receipt: Mapping[str, Any],
    expected_execution_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processes: list[subprocess.Popen[Any]] = []
    streams = []
    worker_paths = []
    ready_paths = []
    launches = []
    scenario_directory = output_directory / "scenario_receipts"
    scenario_directory.mkdir()
    barrier_path = output_directory / "start_barrier.json"
    coordinator_error = None
    barrier_released_perf: float | None = None
    spawn_started_perf_ns = time.perf_counter_ns()
    started = spawn_started_perf_ns / 1e9
    deadline = started + PROFILE_HARD_TIMEOUT_SECONDS
    try:
        for worker_index in range(WORKER_COUNT):
            result_path = output_directory / f"worker_{worker_index}.json"
            ready_path = output_directory / f"worker_{worker_index}.ready.json"
            log_path = output_directory / f"worker_{worker_index}.log"
            stream = log_path.open("x")
            environment = dict(os.environ)
            environment.update(
                {
                    _WORKER_INDEX_ENV: str(worker_index),
                    _B0A_ROOT_ENV: str(b0a_root),
                    _PROCESS_PROBE_ENV: str(process_probe_path),
                    _CONFIRMATION_ENV: str(confirmation_path),
                    _SHARD_ENV: str(shard_paths[worker_index]),
                    _RESULT_ENV: str(result_path),
                    _READY_ENV: str(ready_path),
                    _BARRIER_ENV: str(barrier_path),
                    _SCENARIO_DIRECTORY_ENV: str(scenario_directory),
                    _EXPECTED_CODE_BUNDLE_ENV: code_bundle_sha256,
                    "JAX_PLATFORMS": "cpu",
                }
            )
            process = subprocess.Popen(
                _worker_command(),
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            streams.append(stream)
            worker_paths.append(result_path)
            ready_paths.append(ready_path)
            launches.append(
                {
                    "worker_index": worker_index,
                    "pid": process.pid,
                    "affinity_cpus": list(AFFINITY_SETS[worker_index]),
                    "shard_path": str(shard_paths[worker_index]),
                    "result_path": str(result_path),
                    "ready_path": str(ready_path),
                    "log_path": str(log_path),
                }
            )
        while not all(path.is_file() for path in ready_paths):
            if any(process.poll() not in (None, 0) for process in processes):
                raise ProfileRejected("A worker failed before the ready barrier.")
            if time.perf_counter() > deadline:
                raise ProfileRejected("Profile exceeded 24 hours before readiness.")
            time.sleep(0.25)
        ready = [_load_json(path) for path in ready_paths]
        for index, row in enumerate(ready):
            device = row.get("device", {})
            thread_affinity = row.get("thread_affinity", {})
            runtime = {
                "python_executable": device.get("python_executable"),
                "packages": device.get("packages"),
            }
            execution_protocol = row.get("execution_environment_protocol", {})
            _validate_environment_protocol(
                execution_protocol,
                expected_execution_revision,
            )
            migration_protocol = row.get("migration_protocol", {})
            _validate_environment_protocol(
                migration_protocol.get("environment_protocol", {}),
                EXPECTED_MIGRATION_TERRA_REVISION,
            )
            if (
                row.get("schema") != f"{SCHEMA}_worker_ready_v1"
                or row.get("worker_index") != index
                or row.get("pid") != processes[index].pid
                or row.get("affinity_cpus") != list(AFFINITY_SETS[index])
                or device.get("platform") != "cpu"
                or device.get("default_backend") != "cpu"
                or device.get("device_count") != 1
                or runtime != expected_runtime_receipt
                or thread_affinity.get("expected_cpus") != list(AFFINITY_SETS[index])
                or thread_affinity.get("all_worker_threads_within_fixed_cpuset")
                is not True
                or row.get("module_origins", {}).get("all_origins_match_exact_paths")
                is not True
                or row.get("module_origins", {}).get("pythonpath_override") is not None
                or row.get("code_bundle_sha256") != code_bundle_sha256
                or row.get("shard_sha256")
                != process_probe._sha256_file(shard_paths[index])
                or migration_protocol.get("terra_revision")
                != EXPECTED_MIGRATION_TERRA_REVISION
                or migration_protocol.get("environment_protocol_sha256")
                != EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
                or migration_protocol.get("env_config_sha256")
                != EXPECTED_ENV_CONFIG_SHA256
                or migration_protocol.get("historical_migration_evidence_only")
                is not True
            ):
                raise ProfileRejected(f"Worker {index} ready receipt changed.")
        process_probe._write_json_once(
            barrier_path,
            {
                "schema": f"{SCHEMA}_start_barrier_v1",
                "released_unix_seconds": time.time(),
                "worker_ready_sha256": {
                    str(path): process_probe._sha256_file(path) for path in ready_paths
                },
            },
        )
        barrier_released_perf = time.perf_counter()
        while any(process.poll() is None for process in processes):
            failed_codes = [
                process.returncode
                for process in processes
                if process.poll() not in (None, 0)
            ]
            if failed_codes:
                raise ProfileRejected(
                    f"A worker failed after the barrier: {failed_codes}."
                )
            now = time.time()
            for started_path in scenario_directory.glob("scenario_????.started.json"):
                result_path = started_path.with_name(
                    started_path.name.replace(".started.json", ".json")
                )
                if result_path.exists():
                    continue
                started_row = _load_json(started_path)
                if (
                    now - float(started_row["started_unix_seconds"])
                    > MAX_SCENARIO_SECONDS
                ):
                    raise ProfileRejected(
                        f"{started_row['legacy_map_id']} exceeded one hour."
                    )
            if time.perf_counter() > deadline:
                raise ProfileRejected("The 256-identity profile exceeded 24 hours.")
            time.sleep(0.5)
    except Exception as error:
        coordinator_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        process_probe._terminate_workers(processes)
        return_codes = [
            process.wait() if process.poll() is None else int(process.returncode)
            for process in processes
        ]
        for stream in streams:
            stream.close()
    workers = [_load_worker_receipt(path) for path in worker_paths if path.is_file()]
    finished = time.perf_counter()
    return workers, {
        "coordinator_error": coordinator_error,
        "return_codes": return_codes,
        "launches": launches,
        "worker_result_count": len(workers),
        "cohort_wall_seconds": finished - started,
        "worker_cohort_seconds": (
            finished - barrier_released_perf
            if barrier_released_perf is not None
            else None
        ),
        "spawn_started_perf_ns": spawn_started_perf_ns,
        "hard_timeout_seconds": PROFILE_HARD_TIMEOUT_SECONDS,
        "start_barrier_sha256": (
            process_probe._sha256_file(barrier_path) if barrier_path.is_file() else None
        ),
        "worker_result_sha256": {
            str(path): process_probe._sha256_file(path)
            for path in worker_paths
            if path.is_file()
        },
        "worker_log_sha256": {
            str(output_directory / f"worker_{index}.log"): (
                process_probe._sha256_file(output_directory / f"worker_{index}.log")
            )
            for index in range(len(streams))
        },
    }


def run_profile(
    b0a_root: Path,
    migration_root: Path,
    process_probe_path: Path,
    confirmation_path: Path,
    output_directory: Path,
) -> Path:
    b0a_root = b0a_root.resolve()
    migration_root = migration_root.resolve()
    process_probe_path = process_probe_path.resolve()
    confirmation_path = confirmation_path.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_directory}.")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / OUTPUT_NAME
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "release_id": process_probe.EXPECTED_RELEASE_ID,
        "result_scope": "exact_256_identity_direct_service_cost_profile",
        "status": "failed",
        "static_valid_claimed": False,
        "bank_admission_result_emitted": False,
        "witness_authorized": False,
        "ppo_authorized": False,
        "retry_or_resume_authorized": False,
        "worker_count": WORKER_COUNT,
        "scenario_count": EXPECTED_IDENTITY_COUNT,
    }
    workers: list[dict[str, Any]] = []
    execution: dict[str, Any] | None = None
    processes_started = False
    input_before: dict[str, Any] | None = None
    input_after: dict[str, Any] | None = None
    dependency_before: dict[str, Any] | None = None
    dependency_after: dict[str, Any] | None = None
    candidate_results_path = output_directory / CANDIDATE_RESULTS_NAME
    results_path = output_directory / RESULTS_NAME
    try:
        process_probe._coordinator_import_boundary()
        environment = process_probe._require_runtime_environment(os.environ)
        topology = process_probe._require_cpu_topology()
        authorization, authorization_sha256 = _load_process_authorization(
            process_probe_path
        )
        authorized_runtime = _authorized_runtime_receipt(authorization)
        confirmation, confirmation_sha256 = process_probe._load_reference(
            confirmation_path
        )
        dependency_paths = {
            "r58_process_authorization": (
                process_probe_path,
                authorization_sha256,
            ),
            "cpu_confirmation": (
                confirmation_path,
                confirmation_sha256,
            ),
        }
        dependency_before = _exact_file_receipt(dependency_paths)
        if b0a_root != Path(confirmation["input"]["b0a_root"]).resolve():
            raise RuntimeError("B0a root differs from the exact CPU confirmation.")
        if (
            migration_root
            != Path(
                "/home/lorenzo/moleworks/.artifacts/"
                "terra_b0a_live_migration_20260727_v1"
            ).resolve()
        ):
            raise RuntimeError("Profile requires the exact frozen migration root.")
        pinned_inputs_before = _pinned_input_receipt(
            b0a_root,
            migration_root,
        )
        population, population_receipt = _load_population(
            b0a_root,
            migration_root,
            pinned_inputs_before,
        )
        input_before = _frozen_input_snapshot(
            b0a_root,
            population,
            pinned_inputs_before,
        )
        shards = _build_shards(population)
        repository = Path(__file__).resolve().parents[1]
        code_before = _code_receipt(repository)
        process_probe._require_clean_code(code_before)
        authorized_code = _validate_authorized_code(authorization, code_before)
        cgroup_before = process_probe._cgroup_memory_events()

        shard_directory = output_directory / "shards"
        shard_directory.mkdir()
        shard_paths = []
        for worker_index, shard in enumerate(shards):
            path = shard_directory / f"worker_{worker_index}.jsonl"
            _write_jsonl_once(path, shard)
            shard_paths.append(path)
        process_probe._write_json_once(
            output_directory / "population.json",
            {
                "schema": f"{SCHEMA}_population_v1",
                "population": population_receipt,
                "worker_count": WORKER_COUNT,
                "scenarios_per_worker": SCENARIOS_PER_WORKER,
                "assignment": "global_index modulo 4",
                "shard_sha256": {
                    str(path): process_probe._sha256_file(path) for path in shard_paths
                },
            },
        )
        processes_started = True
        workers, execution = _run_workers(
            b0a_root=b0a_root,
            process_probe_path=process_probe_path,
            confirmation_path=confirmation_path,
            output_directory=output_directory,
            shard_paths=shard_paths,
            code_bundle_sha256=code_before["code_bundle_sha256"],
            expected_runtime_receipt=authorized_runtime,
            expected_execution_revision=code_before["git"]["head"],
        )
        scenario_results, failed_indices, missing_indices = _load_scenario_receipts(
            output_directory / "scenario_receipts"
        )
        if execution["coordinator_error"] is not None:
            raise ProfileRejected(execution["coordinator_error"]["message"])
        if execution["return_codes"] != [0] * WORKER_COUNT:
            raise ProfileRejected(
                f"Worker return codes were {execution['return_codes']}."
            )
        if failed_indices or missing_indices:
            raise ProfileRejected(
                f"Scenario evidence failed={failed_indices}, missing={missing_indices}."
            )
        protocol_receipts = _validate_complete_results(
            workers,
            scenario_results,
            population,
            expected_pids=[launch["pid"] for launch in execution["launches"]],
            expected_code_bundle_sha256=code_before["code_bundle_sha256"],
            expected_process_probe_sha256=authorization_sha256,
            expected_runtime_receipt=authorized_runtime,
            expected_execution_revision=code_before["git"]["head"],
            input_snapshot=input_before,
            confirmation=confirmation,
            scenario_directory=output_directory / "scenario_receipts",
            shard_paths=shard_paths,
        )
        pinned_inputs_after = _pinned_input_receipt(
            b0a_root,
            migration_root,
        )
        input_after = _frozen_input_snapshot(
            b0a_root,
            population,
            pinned_inputs_after,
        )
        if input_after != input_before:
            raise RuntimeError("Frozen profile inputs changed during execution.")
        dependency_after = _exact_file_receipt(dependency_paths)
        if dependency_after != dependency_before:
            raise RuntimeError(
                "Authorization or confirmation changed during execution."
            )
        candidate_sha256 = _persist_and_verify_candidate(
            candidate_results_path,
            scenario_results,
        )
        profile_makespan_seconds = (
            time.perf_counter_ns() - execution["spawn_started_perf_ns"]
        ) / 1e9
        timing = _timing_summary(
            workers,
            profile_makespan_seconds,
        )
        code_after = _code_receipt(repository)
        if code_after != code_before:
            raise RuntimeError("Profile code changed during execution.")
        cgroup_after = process_probe._cgroup_memory_events()
        oom_delta = {
            key: cgroup_after[key] - cgroup_before.get(key, 0)
            for key in ("oom", "oom_kill")
        }
        if oom_delta != {"oom": 0, "oom_kill": 0}:
            raise ProfileRejected(f"Cgroup OOM counters increased: {oom_delta}.")
        coordinator_memory = process_probe._process_memory()
        summed_worker_peak = sum(
            int(worker["process_peak_rss_kib"]) for worker in workers
        )
        conservative_peak = coordinator_memory["ru_maxrss_kib"] + summed_worker_peak
        memory_capacity = process_probe._host_memory_capacity_kib()
        memory_fraction = conservative_peak / memory_capacity
        gates = {
            "all_256_exact_results_complete": True,
            "confirmation_sentinel_matches": True,
            "four_fixed_cpu_workers_pass": True,
            "all_scenarios_at_most_one_hour": all(
                timing_row["identity_duration_seconds"] <= MAX_SCENARIO_SECONDS
                for worker in workers
                for timing_row in worker["identity_timing_ledger"]
            ),
            "profile_makespan_at_most_24h": (
                profile_makespan_seconds <= PROFILE_HARD_TIMEOUT_SECONDS
            ),
            "projection_448_at_most_48h": timing["projection_448"]["passes"],
            "zero_worker_swap": all(worker["zero_swap"] for worker in workers),
            "zero_cgroup_oom": oom_delta == {"oom": 0, "oom_kill": 0},
            "conservative_memory_at_most_80_percent": (
                memory_fraction <= MAX_MEMORY_FRACTION
            ),
            "code_unchanged": True,
            "frozen_inputs_unchanged": input_after == input_before,
            "all_manifest_entries_reverified_pre_and_post": True,
            "all_consumed_paths_manifested": True,
            "authorization_and_confirmation_unchanged": (
                dependency_after == dependency_before
            ),
            "canonical_candidate_reverified_inside_makespan": True,
        }
        gates["all_profile_gates_pass"] = all(gates.values())
        if not gates["all_profile_gates_pass"]:
            raise ProfileRejected("At least one 256-profile gate failed.")

        scenario_receipt_sha256 = {
            str(
                _scenario_result_path(
                    output_directory / "scenario_receipts",
                    row["global_index"],
                )
            ): process_probe._sha256_file(
                _scenario_result_path(
                    output_directory / "scenario_receipts",
                    row["global_index"],
                )
            )
            for row in scenario_results
        }
        receipt.update(
            {
                "status": "passed",
                "input": {
                    "b0a_root": str(b0a_root),
                    "migration_root": str(migration_root),
                    "process_probe_receipt": str(process_probe_path),
                    "process_probe_receipt_sha256": authorization_sha256,
                    "confirmation_receipt": str(confirmation_path),
                    "confirmation_receipt_sha256": confirmation_sha256,
                    "population": population_receipt,
                    "pre_execution": _compact_input_receipt(input_before),
                    "post_execution": _compact_input_receipt(input_after),
                    "dependency_pre_execution": dependency_before,
                    "dependency_post_execution": dependency_after,
                },
                "validator": {
                    "code_before": code_before,
                    "code_after": code_after,
                    "authorized_r58_dependency_sha256": authorized_code,
                    "exact_entrypoint": (
                        "terra.benchmark_direct_service."
                        "compute_initial_direct_service"
                    ),
                    "exact_entrypoint_call_count": EXPECTED_IDENTITY_COUNT,
                },
                "protocol_provenance": {
                    "migration": {
                        "terra_revision": EXPECTED_MIGRATION_TERRA_REVISION,
                        "environment_protocol_sha256": (
                            EXPECTED_ENVIRONMENT_PROTOCOL_SHA256
                        ),
                        "env_config_sha256": EXPECTED_ENV_CONFIG_SHA256,
                        "historical_migration_evidence_only": True,
                        "environment_protocol": protocol_receipts[
                            "migration_environment"
                        ],
                    },
                    "execution_environment": protocol_receipts["execution_environment"],
                    "execution_code_bundle_sha256": code_before["code_bundle_sha256"],
                    "migration_and_execution_hashes_are_not_assumed_equal": True,
                    "profile_does_not_make_migration_static_valid": True,
                },
                "machine": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "topology": topology,
                    "environment": environment,
                    "authorized_worker_runtime": authorized_runtime,
                    "cgroup_memory_events_before": cgroup_before,
                    "cgroup_memory_events_after": cgroup_after,
                    "cgroup_oom_delta": oom_delta,
                },
                "execution": execution,
                "timing": timing,
                "resources": {
                    "coordinator_peak_rss_kib": coordinator_memory["ru_maxrss_kib"],
                    "summed_worker_peak_rss_kib": summed_worker_peak,
                    "conservative_aggregate_peak_rss_kib": conservative_peak,
                    "host_physical_memory_kib": memory_capacity,
                    "conservative_peak_fraction_of_host": memory_fraction,
                    "maximum_allowed_fraction": MAX_MEMORY_FRACTION,
                },
                "gates": gates,
                "output": {
                    "canonical_candidate": str(candidate_results_path),
                    "canonical_candidate_sha256": candidate_sha256,
                    "canonical_candidate_reloaded_and_exactly_compared": True,
                    "direct_service_results": str(results_path),
                    "direct_service_results_sha256": candidate_sha256,
                    "direct_service_results_requires_passing_validation_cost": True,
                    "scenario_receipt_count": len(scenario_results),
                    "scenario_receipt_sha256": scenario_receipt_sha256,
                },
                "decision": {
                    "exact_256_identity_cost_profile_passes": True,
                    "direct_service_sidecar_available_for_later_static_merge": True,
                    "complete_static_fields_emitted": False,
                    "reachable_capacity_ratio_emitted": False,
                    "authorizes_448_execution": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_static_admission": False,
                    "authorizes_bank_admission": False,
                    "authorizes_witness": False,
                    "authorizes_ppo": False,
                },
            }
        )
        _publish_existing_file_no_replace(candidate_results_path, results_path)
    except Exception as error:
        receipt["status"] = "failed"
        completed_indices: list[int] = []
        failed_indices: list[int] = []
        missing_indices = list(range(EXPECTED_IDENTITY_COUNT))
        partial_load_error: dict[str, str] | None = None
        scenario_directory = output_directory / "scenario_receipts"
        if scenario_directory.is_dir():
            try:
                completed, failed_indices, missing_indices = _load_scenario_receipts(
                    scenario_directory
                )
                completed_indices = [row["global_index"] for row in completed]
            except Exception as evidence_error:
                partial_load_error = {
                    "type": type(evidence_error).__name__,
                    "message": str(evidence_error),
                }
        receipt.update(
            {
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                    "processes_started": processes_started,
                },
                "execution": execution,
                "partial_evidence": {
                    "completed_indices": completed_indices,
                    "failed_indices": failed_indices,
                    "missing_indices": missing_indices,
                    "completed_count": len(completed_indices),
                    "scenario_receipt_load_error": partial_load_error,
                    "canonical_candidate": _existing_file_evidence(
                        candidate_results_path
                    ),
                    "success_filename": {
                        **_existing_file_evidence(results_path),
                        "authoritative_without_passing_validation_cost": False,
                    },
                    "pre_execution_input": (
                        _compact_input_receipt(input_before)
                        if input_before is not None
                        else None
                    ),
                    "post_execution_input": (
                        _compact_input_receipt(input_after)
                        if input_after is not None
                        else None
                    ),
                    "dependency_pre_execution": dependency_before,
                    "dependency_post_execution": dependency_after,
                },
                "decision": {
                    "exact_256_identity_cost_profile_passes": False,
                    "direct_service_sidecar_available_for_later_static_merge": False,
                    "complete_static_fields_emitted": False,
                    "reachable_capacity_ratio_emitted": False,
                    "authorizes_448_execution": False,
                    "authorizes_retry_or_resume": False,
                    "authorizes_static_admission": False,
                    "authorizes_bank_admission": False,
                    "authorizes_witness": False,
                    "authorizes_ppo": False,
                },
            }
        )
    receipt["end_to_end_wall_seconds"] = time.perf_counter() - started
    process_probe._write_json_once(output_path, receipt)
    print(output_path)
    if receipt["status"] != "passed":
        raise ProfileRejected(f"Exact 256-identity profile failed; see {output_path}.")
    return output_path


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one authorized exact 256-identity four-worker CPU profile."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--migration-root", type=Path, required=True)
    parser.add_argument("--cpu-process-probe-receipt", type=Path, required=True)
    parser.add_argument("--cpu-confirmation-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> None:
    if _WORKER_INDEX_ENV in os.environ:
        raise SystemExit(_worker_main())
    if any(key in os.environ for key in _INTERNAL_ENV_KEYS):
        raise RuntimeError("Incomplete internal profile-worker environment.")
    args = _public_parser().parse_args()
    run_profile(
        args.b0a_root,
        args.migration_root,
        args.cpu_process_probe_receipt,
        args.cpu_confirmation_receipt,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
