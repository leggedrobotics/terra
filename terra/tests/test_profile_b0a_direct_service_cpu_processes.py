import ast
import inspect
import json
from pathlib import Path

import pytest

import tools.profile_b0a_direct_service_cpu_processes as profile


def _minimal_population():
    rows = []
    for index in range(profile.EXPECTED_IDENTITY_COUNT):
        legacy_map_id = f"map-{index:03d}"
        rows.append(
            {
                "global_index": index,
                "legacy_identity": {"map_id": legacy_map_id},
                "migration": {
                    "legacy_map_id": (
                        profile.EXPECTED_SENTINEL_MAP_ID
                        if index == 128
                        else legacy_map_id
                    )
                },
            }
        )
    return rows


def _outcome():
    outcome = {
        key: index
        for index, key in enumerate(
            sorted(profile.process_probe.DIRECT_SERVICE_OUTCOME_KEYS)
        )
    }
    outcome["any_direct_transfer_pose_exists_initial"] = True
    outcome["initial_workspace_coverage"] = 1.0
    outcome["direct_service_coverage_initial"] = 1.0
    return outcome


def test_top_level_imports_remain_coordinator_safe():
    tree = ast.parse(inspect.getsource(profile))
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert not any(
        name == "jax"
        or name.startswith("jax.")
        or name == "numpy"
        or name.startswith("terra.")
        or name == "tools.profile_b0a_direct_service_cost"
        for name in imports
    )


def test_public_cli_exposes_no_backend_worker_retry_or_resume_knobs():
    destinations = {action.dest for action in profile._public_parser()._actions}

    assert destinations == {
        "help",
        "b0a_root",
        "migration_root",
        "cpu_process_probe_receipt",
        "cpu_confirmation_receipt",
        "output_directory",
    }


def test_profile_is_fail_closed_until_passing_r58_hash_is_pinned(tmp_path, monkeypatch):
    receipt = tmp_path / "cpu_process_probe.json"
    receipt.write_text("{}")

    monkeypatch.setattr(profile, "EXPECTED_CPU_PROCESS_PROBE_SHA256", "UNSET")
    with pytest.raises(RuntimeError, match="is UNSET"):
        profile._load_process_authorization(receipt)


def test_r58_authorization_is_narrow_not_generic(tmp_path, monkeypatch):
    receipt = {
        "schema": profile.process_probe.SCHEMA,
        "release_id": profile.process_probe.EXPECTED_RELEASE_ID,
        "status": "passed",
        "experiment": {
            "treatment": "fixed_four_long_lived_cpu_workers_two_calls_each",
            "worker_count": 4,
        },
        "gates": {
            "all_eight_outcomes_and_counters_match_confirmation": True,
            "all_process_treatment_gates_pass": True,
        },
        "input": {
            "cpu_confirmation_receipt_sha256": (
                profile.EXPECTED_CPU_CONFIRMATION_SHA256
            )
        },
        "decision": {
            "authorizes_one_deterministic_four_worker_256_profile": True,
            "authorized_profile_must_finish_within_24h": True,
            "authorized_profile_must_project_448_within_48h": True,
            "authorizes_bank_profile": False,
            "authorizes_static_admission": False,
            "authorizes_ppo": False,
            "authorizes_retry": False,
        },
    }
    path = tmp_path / "cpu_process_probe.json"
    path.write_text(json.dumps(receipt))
    digest = profile.process_probe._sha256_file(path)
    monkeypatch.setattr(profile, "EXPECTED_CPU_PROCESS_PROBE_SHA256", digest)

    assert profile._load_process_authorization(path) == (receipt, digest)

    receipt["decision"]["authorizes_bank_profile"] = True
    path.write_text(json.dumps(receipt))
    digest = profile.process_probe._sha256_file(path)
    monkeypatch.setattr(profile, "EXPECTED_CPU_PROCESS_PROBE_SHA256", digest)
    with pytest.raises(RuntimeError, match="narrow authorization only"):
        profile._load_process_authorization(path)


def test_r58_runtime_receipt_requires_four_identical_workers():
    runtime = {
        "python_executable": "/opt/venv/bin/python",
        "packages": {
            "jax": "1",
            "jaxlib": "1",
            "numpy": "2",
            "scipy": "3",
        },
    }
    authorization = {
        "workers": [{"device": runtime} for _ in range(profile.WORKER_COUNT)]
    }

    assert profile._authorized_runtime_receipt(authorization) == runtime
    authorization["workers"][3] = {
        "device": {
            **runtime,
            "packages": {**runtime["packages"], "jax": "changed"},
        }
    }
    with pytest.raises(RuntimeError, match="different Python/package"):
        profile._authorized_runtime_receipt(authorization)


def test_round_robin_shards_are_fixed_complete_and_balanced():
    population = _minimal_population()
    shards = profile._build_shards(population)

    assert len(shards) == 4
    assert [len(shard) for shard in shards] == [64, 64, 64, 64]
    for worker_index, shard in enumerate(shards):
        assert [row["global_index"] for row in shard] == list(
            range(worker_index, 256, 4)
        )


def test_worker_zero_runs_the_existing_sentinel_first_without_duplication():
    shards = profile._build_shards(_minimal_population())
    ordered = profile._worker_processing_order(0, shards[0])

    assert ordered[0]["global_index"] == 128
    assert ordered[0]["migration"]["legacy_map_id"] == (
        profile.EXPECTED_SENTINEL_MAP_ID
    )
    assert sorted(row["global_index"] for row in ordered) == list(range(0, 256, 4))
    assert len({row["global_index"] for row in ordered}) == 64


def test_nearest_rank_and_448_projection_are_literal():
    workers = []
    for worker_index in range(profile.WORKER_COUNT):
        ledger = []
        for position, global_index in enumerate(range(worker_index, 256, 4)):
            ledger.append(
                {
                    "global_index": global_index,
                    "worker_index": worker_index,
                    "processing_position": position,
                    "identity_duration_seconds": float(global_index + 1),
                }
            )
        workers.append(
            {
                "worker_index": worker_index,
                "identity_timing_ledger": ledger,
            }
        )
    summary = profile._timing_summary(workers, execution_makespan_seconds=1000.0)
    expected_per_worker_p95 = {}
    for worker in workers:
        warm = [
            row["identity_duration_seconds"]
            for row in worker["identity_timing_ledger"][1:]
        ]
        expected_per_worker_p95[str(worker["worker_index"])] = sorted(warm)[
            -1 + __import__("math").ceil(0.95 * len(warm))
        ]
    expected_maximum = max(expected_per_worker_p95.values())

    assert summary["quantile_method"] == "nearest_rank"
    assert summary["scenario_total_seconds"] == {
        "p50": 128.0,
        "p95": 244.0,
        "max": 256.0,
    }
    assert {
        key: row["post_first_nearest_rank_p95_seconds"]
        for key, row in summary["per_worker"].items()
    } == expected_per_worker_p95
    assert summary["maximum_worker_post_first_p95_seconds"] == expected_maximum
    assert summary["projection_448"]["projected_makespan_seconds"] == (
        1000.0 + 48 * expected_maximum
    )


def test_scenario_receipt_loader_preserves_evidence_but_reports_gaps(tmp_path):
    directory = tmp_path / "scenario_receipts"
    directory.mkdir()
    passed = {
        "schema": f"{profile.SCHEMA}_scenario_v1",
        "status": "passed",
        "global_index": 0,
    }
    failed = {
        "schema": f"{profile.SCHEMA}_scenario_v1",
        "status": "failed",
        "global_index": 2,
    }
    (directory / "scenario_0000.json").write_text(json.dumps(passed))
    (directory / "scenario_0002.json").write_text(json.dumps(failed))

    completed, failed_indices, missing = profile._load_scenario_receipts(directory)

    assert completed == [passed]
    assert failed_indices == [2]
    assert 0 not in missing and 2 not in missing
    assert len(missing) == 254


def test_scenario_receipt_is_atomically_published_and_never_replaced(
    tmp_path, monkeypatch
):
    path = tmp_path / "scenario_0000.json"
    links = []
    real_link = profile.os.link

    def record_link(source, destination):
        links.append((Path(source), Path(destination)))
        real_link(source, destination)

    monkeypatch.setattr(profile.os, "link", record_link)
    profile._write_json_once_no_replace(path, {"status": "passed"})

    assert json.loads(path.read_text()) == {"status": "passed"}
    assert len(links) == 1 and links[0][1] == path
    with pytest.raises(FileExistsError):
        profile._write_json_once_no_replace(path, {"status": "changed"})
    assert json.loads(path.read_text()) == {"status": "passed"}


def test_existing_candidate_is_fsynced_before_success_link(tmp_path, monkeypatch):
    source = tmp_path / "candidate.jsonl"
    destination = tmp_path / "results.jsonl"
    source.write_text('{"value":1}\n')
    events = []
    real_fsync = profile.os.fsync
    real_link = profile.os.link

    def record_fsync(file_descriptor):
        events.append("fsync")
        real_fsync(file_descriptor)

    def record_link(source_path, destination_path):
        events.append("link")
        real_link(source_path, destination_path)

    monkeypatch.setattr(profile.os, "fsync", record_fsync)
    monkeypatch.setattr(profile.os, "link", record_link)
    profile._publish_existing_file_no_replace(source, destination)

    assert events == ["fsync", "link", "fsync"]
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        profile._publish_existing_file_no_replace(source, destination)


def test_manifest_snapshot_rejects_post_preflight_mutation(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"before")
    digest = profile.process_probe._sha256_file(payload)
    (tmp_path / "files.sha256").write_text(f"{digest}  payload.bin\n")

    assert profile._verified_manifest_entries(tmp_path) == {"payload.bin": digest}
    payload.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="manifest entry changed"):
        profile._verified_manifest_entries(tmp_path)


def test_candidate_is_reloaded_and_typed_compared(tmp_path, monkeypatch):
    path = tmp_path / "direct_service_results.candidate.jsonl"
    rows = [{"value": True}]

    assert profile._persist_and_verify_candidate(path, rows) == (
        profile.process_probe._sha256_file(path)
    )

    tampered_path = tmp_path / "tampered.jsonl"
    real_writer = profile._write_jsonl_once

    def write_integer_instead_of_boolean(destination, expected_rows):
        real_writer(destination, [{"value": 1}])

    monkeypatch.setattr(
        profile,
        "_write_jsonl_once",
        write_integer_instead_of_boolean,
    )
    with pytest.raises(RuntimeError, match="failed verification"):
        profile._persist_and_verify_candidate(tampered_path, rows)


def test_typed_outcome_hash_rejects_tampered_scenario_receipt():
    outcome = _outcome()
    result = {
        "exact_outcome": outcome,
        "typed_exact_outcome_sha256": (
            profile.process_probe._typed_canonical_sha256(outcome)
        ),
    }

    assert profile._validated_exact_outcome(result) == outcome
    result["exact_outcome"] = {**outcome, "required_volume": 999}
    with pytest.raises(RuntimeError, match="exact-outcome receipt is invalid"):
        profile._validated_exact_outcome(result)


def test_environment_protocol_hash_binds_full_payload():
    payload = {
        "schema": "terra_environment_protocol_v1",
        "release_id": profile.process_probe.EXPECTED_RELEASE_ID,
        "terra_revision": "revision",
        "env_config_sha256": profile.EXPECTED_ENV_CONFIG_SHA256,
        "environment_constant": 7,
    }
    protocol = {
        **payload,
        "environment_protocol_sha256": (
            profile.process_probe._canonical_json_sha256(payload)
        ),
    }

    profile._validate_environment_protocol(protocol, "revision")
    protocol["environment_constant"] = 8
    with pytest.raises(RuntimeError, match="protocol receipt is invalid"):
        profile._validate_environment_protocol(protocol, "revision")


def test_success_publication_follows_all_gates_and_candidate_verification():
    source = inspect.getsource(profile.run_profile)

    process_flag = source.index("processes_started = True")
    worker_call = source.index("workers, execution = _run_workers(")
    candidate = source.index("candidate_sha256 = _persist_and_verify_candidate(")
    resource_snapshot = source.index(
        "cgroup_after = process_probe._cgroup_memory_events()",
        candidate,
    )
    gate_rejection = source.index('if not gates["all_profile_gates_pass"]')
    success_receipt = source.index('"status": "passed"', gate_rejection)
    publication = source.index("_publish_existing_file_no_replace(", success_receipt)

    assert process_flag < worker_call < candidate < resource_snapshot < gate_rejection
    assert gate_rejection < success_receipt < publication
    assert source[publication:].count("_publish_existing_file_no_replace(") == 1


def test_profile_uses_serialized_migration_state_not_resampling():
    source = inspect.getsource(profile._worker_main)

    assert "agent_from_record" in source
    assert "environment_reset_seed" in source
    assert "_materialize_initial_state" not in source
    assert "sample_benchmark_initial_agent" not in source


def test_exact_outcome_contract_remains_23_typed_fields():
    outcome = _outcome()

    assert len(outcome) == 23
    assert set(outcome) == profile.process_probe.DIRECT_SERVICE_OUTCOME_KEYS
    changed = dict(outcome)
    changed["any_direct_transfer_pose_exists_initial"] = 1
    assert profile.process_probe._typed_canonical(outcome) != (
        profile.process_probe._typed_canonical(changed)
    )
