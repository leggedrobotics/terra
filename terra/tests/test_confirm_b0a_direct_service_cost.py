import json
from types import SimpleNamespace

import pytest

import tools.confirm_b0a_direct_service_cost as confirmation
from tools.profile_b0a_direct_service_cost import SELECTED_MAP_ID


def _projections():
    return {
        "requires_one_complete_scenario_confirmation": True,
        "first_scenario_cold_seconds_p50": 10.0,
        "first_scenario_cold_seconds_p95": 11.0,
        "scenario_counts": {
            "256": {"seconds_p95": 100_000.0},
            "448": {"seconds_p95": 400_000.0},
        },
    }


def _initial_state():
    return {
        "schema": "terra_initial_state_seed_v1",
        "release_id": "terramap-bench-v1.0.0",
        "split": "public_train",
        "source_group_id": "osm-foundation:108",
        "state_index": 0,
        "seed_uint32": 7,
        "seed_byte_order": "big",
        "seed_digest_sha256": "a" * 64,
        "initial_agent_state_sha256": "b" * 64,
        "initial_agent_state": {"schema": "terra_agent_state_v1"},
        "group_variant_count": 4,
        "initial_agent_sampling_seconds": 1.0,
        "explicit_reset_compile_execute_seconds": 2.0,
    }


def _probe():
    selected_identity = {
        "map_id": SELECTED_MAP_ID,
        "source_group_id": "osm-foundation:108",
    }
    return {
        "schema": "terra_direct_service_validation_cost_probe_v1",
        "admission_result_emitted": False,
        "selected_identity": selected_identity,
        "input": {
            "files_sha256_manifest_sha256": "c" * 64,
            "source_grouping": {"source_group_count": 144},
            "verified_selected_files": {"identities.jsonl": "d" * 64},
        },
        "protocol": {"env_config_sha256": "e" * 64},
        "initial_state": _initial_state(),
        "validator": {
            "code_file_sha256": {"terra/benchmark_direct_service.py": "f" * 64}
        },
        "machine": {
            "hostname": confirmation.socket.gethostname(),
            "jax": {"platform": "cpu"},
        },
        "measurement": {
            "memory": {"process_max_rss_kib_after": 10},
            "projections": _projections(),
        },
    }


def test_calibration_scales_bank_p95_and_receipts_all_gates():
    result = confirmation._calibrate_cost(5.0, _projections())

    assert result["observed_to_probe_p50_ratio"] == pytest.approx(0.5)
    assert result["scenario_counts"]["256"]["calibrated_p95_seconds"] == pytest.approx(
        50_000.0
    )
    assert result["scenario_counts"]["448"]["calibrated_p95_seconds"] == pytest.approx(
        200_000.0
    )
    assert result["gates"] == {
        "observed_within_factor_two_of_probe_p50": True,
        "calibrated_256_p95_at_most_24h": True,
        "calibrated_448_p95_at_most_48h": False,
        "all_runtime_gates_pass": False,
    }


@pytest.mark.parametrize(
    ("observed", "passes"),
    ((4.99, False), (5.0, True), (20.0, True), (20.01, False)),
)
def test_factor_two_gate_has_frozen_inclusive_bounds(observed, passes):
    result = confirmation._calibrate_cost(observed, _projections())

    assert result["gates"]["observed_within_factor_two_of_probe_p50"] is passes


def test_pinned_probe_loader_requires_exact_bytes_and_cost_only_contract(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "validation_cost_probe.json"
    path.write_text(json.dumps(_probe()))
    digest = confirmation.probe_tool._sha256_file(path)
    monkeypatch.setattr(confirmation, "EXPECTED_PROBE_SHA256", digest)
    monkeypatch.setattr(confirmation, "_host_memory_capacity_kib", lambda: 100)

    assert confirmation._load_pinned_probe(path) == _probe()

    changed = _probe()
    changed["admission_result_emitted"] = True
    path.write_text(json.dumps(changed))
    changed_digest = confirmation.probe_tool._sha256_file(path)
    monkeypatch.setattr(confirmation, "EXPECTED_PROBE_SHA256", changed_digest)
    with pytest.raises(RuntimeError, match="non-admission"):
        confirmation._load_pinned_probe(path)

    no_headroom = _probe()
    no_headroom["measurement"]["memory"]["process_max_rss_kib_after"] = 81
    path.write_text(json.dumps(no_headroom))
    no_headroom_digest = confirmation.probe_tool._sha256_file(path)
    monkeypatch.setattr(confirmation, "EXPECTED_PROBE_SHA256", no_headroom_digest)
    with pytest.raises(RuntimeError, match="memory-headroom"):
        confirmation._load_pinned_probe(path)


def test_memory_headroom_uses_inclusive_eighty_percent_cpu_gate():
    at_limit = confirmation._memory_headroom(80, 100, "cpu")
    above_limit = confirmation._memory_headroom(80.0001, 100, "cpu")

    assert at_limit["all_memory_gates_pass"] is True
    assert above_limit["all_memory_gates_pass"] is False
    assert at_limit["device_memory_gate"] == "shared_host_memory"


def test_memory_headroom_fails_closed_for_accelerators():
    with pytest.raises(RuntimeError, match="Accelerator memory"):
        confirmation._memory_headroom(10, 100, "gpu")


def test_accelerator_rejection_precedes_exact_validator_call(monkeypatch):
    calls = []
    monkeypatch.setattr(
        confirmation.jax,
        "devices",
        lambda: [SimpleNamespace(platform="gpu")],
    )
    monkeypatch.setattr(
        confirmation.direct_service,
        "compute_initial_direct_service",
        lambda state: calls.append(state),
    )

    with pytest.raises(RuntimeError, match="single CPU"):
        confirmation._compute_exact_once(object())
    assert calls == []


def test_rebuilt_contract_requires_probe_identity_state_protocol_and_code():
    probe = _probe()
    arguments = {
        "probe": probe,
        "selected_identity": probe["selected_identity"],
        "files_sha256_manifest_sha256": probe["input"]["files_sha256_manifest_sha256"],
        "source_grouping": probe["input"]["source_grouping"],
        "verified_selected_files": probe["input"]["verified_selected_files"],
        "protocol": probe["protocol"],
        "initial_state": {
            **probe["initial_state"],
            "initial_agent_sampling_seconds": 999.0,
            "explicit_reset_compile_execute_seconds": 888.0,
        },
        "dependency_file_sha256": probe["validator"]["code_file_sha256"],
    }

    confirmation._validate_rebuilt_contract(**arguments)

    arguments["protocol"] = {"env_config_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="protocol"):
        confirmation._validate_rebuilt_contract(**arguments)

    arguments["protocol"] = probe["protocol"]
    arguments["initial_state"] = {
        **probe["initial_state"],
        "initial_agent_state_sha256": "0" * 64,
    }
    arguments["dependency_file_sha256"] = {"terra/state.py": "1" * 64}
    with pytest.raises(
        RuntimeError,
        match="explicit initial state, direct-service dependency files",
    ):
        confirmation._validate_rebuilt_contract(**arguments)
