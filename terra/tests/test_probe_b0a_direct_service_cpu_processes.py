import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tools.probe_b0a_direct_service_cpu_processes as probe

REFERENCE_SHA256 = "a" * 64
CODE_BUNDLE_SHA256 = "b" * 64


def _outcome():
    outcome = {
        key: index
        for index, key in enumerate(sorted(probe.DIRECT_SERVICE_OUTCOME_KEYS))
    }
    outcome["any_direct_transfer_pose_exists_initial"] = True
    outcome["initial_workspace_coverage"] = 1.0
    outcome["direct_service_coverage_initial"] = 1.0
    return outcome


def _worker(index, *, cold=100.0, warm=10.0, peak=100, pid=None):
    outcome = _outcome()
    code = {"code_bundle_sha256": CODE_BUNDLE_SHA256}
    thread_affinity = {
        "expected_cpus": list(probe.AFFINITY_SETS[index]),
        "tasks": {"100": {"parsed_cpus": list(probe.AFFINITY_SETS[index])}},
        "all_worker_threads_within_fixed_cpuset": True,
    }
    calls = [
        {
            "call_index": 0,
            "launch_through_result_seconds": cold,
            "rematerialization_through_result_seconds": 90.0,
            "exact_outcome": outcome,
            "memory_after": {"VmSwap": 0},
        },
        {
            "call_index": 1,
            "launch_through_result_seconds": None,
            "rematerialization_through_result_seconds": warm,
            "exact_outcome": outcome,
            "memory_after": {"VmSwap": 0},
        },
    ]
    return {
        "schema": f"{probe.SCHEMA}_worker_v1",
        "worker_index": index,
        "pid": index + 100 if pid is None else pid,
        "status": "passed",
        "affinity_cpus": list(probe.AFFINITY_SETS[index]),
        "affinity_before_calls": list(probe.AFFINITY_SETS[index]),
        "affinity_after_calls": list(probe.AFFINITY_SETS[index]),
        "thread_affinity_before_calls": thread_affinity,
        "thread_affinity_after_calls": thread_affinity,
        "device": {"device_count": 1, "platform": "cpu", "default_backend": "cpu"},
        "reference": {"sha256": REFERENCE_SHA256},
        "input": {"selected_input_contract_sha256": "c" * 64},
        "protocol": {"protocol_receipt_sha256": "d" * 64},
        "initial_state": {"stable_receipt_sha256": "e" * 64},
        "validator": {
            "code_before": code,
            "code_after": code,
            "exact_entrypoint_call_count": 2,
        },
        "calls": calls,
        "persisted_call_receipts": {"first": "f" * 64, "second": "0" * 64},
        "memory_monitor": {"transient_swap_observed": False},
        "process_peak_rss_kib": peak,
    }


def test_module_import_is_stdlib_only_until_worker_runtime():
    tree = ast.parse(inspect.getsource(probe))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.append(node.module)

    assert not any(
        name == "jax"
        or name.startswith("jax.")
        or name == "numpy"
        or name.startswith("terra.")
        or name.startswith("tools.")
        for name in top_level_imports
    )


def test_worker_module_resolves_in_fresh_interpreter_from_pinned_cwd():
    repository = Path(probe.__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPROFILEIMPORTTIME"] = "1"
    result = subprocess.run(
        [*probe._worker_command(), "--help"],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe._worker_command() == [
        sys.executable,
        "-m",
        "tools.probe_b0a_direct_service_cpu_processes",
    ]
    assert result.returncode == 0, result.stderr
    assert "four-worker CPU scaling gate" in result.stdout
    imported_modules = {
        line.rsplit("|", maxsplit=1)[-1].strip()
        for line in result.stderr.splitlines()
        if line.startswith("import time:")
    }
    assert imported_modules
    assert not any(
        name == "jax"
        or name.startswith("jax.")
        or name == "terra"
        or name.startswith("terra.")
        for name in imported_modules
    )


def test_fixed_affinities_partition_the_pinned_topology():
    flattened = [cpu for cpus in probe.AFFINITY_SETS for cpu in cpus]

    assert probe.WORKER_COUNT == 4
    assert probe.CALLS_PER_WORKER == 2
    assert len(flattened) == len(set(flattened)) == 32
    assert sorted(flattened) == list(range(32))
    assert probe.AFFINITY_SETS == (
        (0, 1, 2, 3, 16, 17, 18, 19),
        (4, 5, 6, 7, 20, 21, 22, 23),
        (8, 9, 10, 11, 24, 25, 26, 27),
        (12, 13, 14, 15, 28, 29, 30, 31),
    )


def test_public_cli_has_no_worker_or_wave_count_knobs():
    parser = probe._public_parser()
    destinations = {action.dest for action in parser._actions}

    assert destinations == {
        "help",
        "b0a_root",
        "cpu_confirmation_receipt",
        "output_directory",
    }


def test_receipts_publish_atomically_without_replacing_existing_bytes(tmp_path):
    path = tmp_path / "receipt.json"
    probe._write_json_once(path, {"complete": True})

    assert json.loads(path.read_text()) == {"complete": True}
    assert list(tmp_path.glob(".*.tmp")) == []
    with pytest.raises(FileExistsError):
        probe._write_json_once(path, {"complete": False})
    assert json.loads(path.read_text()) == {"complete": True}


def test_linux_cpu_list_parser_is_exact():
    assert probe._parse_cpu_list("0-3,16-19") == [0, 1, 2, 3, 16, 17, 18, 19]
    with pytest.raises(RuntimeError, match="duplicates"):
        probe._parse_cpu_list("0-2,2")


def test_projection_uses_preregistered_max_worker_formula():
    result = probe._projection(
        cold_seconds=[100.0, 200.0, 300.0, 400.0],
        warm_seconds=[10.0, 20.0, 30.0, 40.0],
    )

    assert result["method"] == "max_i(C_i + (ceil(N/4)-1)*W_i)"
    assert result["scenario_counts"]["256"]["per_worker_scenario_count"] == 64
    assert result["scenario_counts"]["256"][
        "projected_makespan_seconds"
    ] == pytest.approx(400.0 + 63 * 40.0)
    assert result["scenario_counts"]["448"]["per_worker_scenario_count"] == 112
    assert result["scenario_counts"]["448"][
        "projected_makespan_seconds"
    ] == pytest.approx(400.0 + 111 * 40.0)


def test_two_call_runner_rematerializes_and_checks_every_exact_outcome(monkeypatch):
    materialized = []
    computed = []
    persisted = []
    outcome = _outcome()
    states = [object(), object()]
    memories = iter(
        [
            {"VmRSS": 1, "VmHWM": 2, "VmSwap": 0, "ru_maxrss_kib": 2},
            {"VmRSS": 1, "VmHWM": 2, "VmSwap": 0, "ru_maxrss_kib": 2},
        ]
    )
    monkeypatch.setattr(probe, "_process_memory", lambda: next(memories))

    def materialize():
        state = states[len(materialized)]
        materialized.append(state)
        return state, {
            "initial_agent_state_sha256": "a" * 64,
            "stable": "same",
        }

    def compute(state):
        computed.append(state)
        return dict(outcome)

    calls = probe._run_two_exact_calls(
        launch_perf_ns=probe.time.perf_counter_ns() - 1_000_000,
        materialize=materialize,
        compute=compute,
        synchronize=lambda value: value,
        stable_initial_state=lambda value: {"stable": value["stable"]},
        jsonable=lambda value: value,
        persist_call=persisted.append,
        reference_initial_state={"stable": "same"},
        reference_outcome=outcome,
        outcome_keys=set(outcome),
    )

    assert materialized == computed == states
    assert persisted == calls
    assert len(calls) == 2
    assert calls[0]["launch_through_result_seconds"] > 0
    assert calls[1]["launch_through_result_seconds"] is None
    assert all(call["exact_outcome"] == outcome for call in calls)


def test_worker_gate_receipts_projection_memory_and_exactness():
    workers = [_worker(index) for index in range(4)]

    projection, resources = probe._validate_worker_results(
        workers,
        reference_outcome=_outcome(),
        reference_sha256=REFERENCE_SHA256,
        expected_code_bundle_sha256=CODE_BUNDLE_SHA256,
        expected_pids=[worker["pid"] for worker in workers],
        host_memory_capacity_kib=1000,
        coordinator_peak_rss_kib=100,
    )

    assert projection["all_projection_gates_pass"] is True
    assert resources["measurement"]["conservative_aggregate_peak_rss_kib"] == 500
    assert resources["gates"]["all_process_treatment_gates_pass"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda workers: workers[1].update(pid=workers[0]["pid"]), "distinct PIDs"),
        (
            lambda workers: workers[2].update(affinity_cpus=[0]),
            "affinity changed",
        ),
        (
            lambda workers: workers[3]["calls"][1]["memory_after"].update(VmSwap=1),
            "used swap",
        ),
        (
            lambda workers: workers[0]["calls"][0].update(exact_outcome={"wrong": 1}),
            "exact outcome differs",
        ),
        (
            lambda workers: workers[0]["calls"][0].update(
                rematerialization_through_result_seconds=3600.01
            ),
            "one-hour",
        ),
        (
            lambda workers: workers[0]["calls"][0].update(
                launch_through_result_seconds=3600.01
            ),
            "cold launch",
        ),
    ],
)
def test_worker_gate_fails_closed_on_contract_violation(mutation, message):
    workers = [_worker(index) for index in range(4)]
    mutation(workers)

    with pytest.raises(RuntimeError, match=message):
        probe._validate_worker_results(
            workers,
            reference_outcome=_outcome(),
            reference_sha256=REFERENCE_SHA256,
            expected_code_bundle_sha256=CODE_BUNDLE_SHA256,
            expected_pids=[index + 100 for index in range(probe.WORKER_COUNT)],
            host_memory_capacity_kib=1000,
            coordinator_peak_rss_kib=100,
        )


def test_reference_loader_pins_exact_bytes(tmp_path, monkeypatch):
    reference = {
        "schema": probe.EXPECTED_CPU_CONFIRMATION_SCHEMA,
        "release_id": probe.EXPECTED_RELEASE_ID,
        "result_scope": "one_complete_exact_scenario_cost_confirmation",
        "bank_admission_result_emitted": False,
        "single_scenario_exact_outcome_emitted": True,
        "validator": {"exact_entrypoint_call_count": 1},
        "measurement": {"exact_outcome": _outcome()},
    }
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(reference))
    monkeypatch.setattr(
        probe, "EXPECTED_CPU_CONFIRMATION_SHA256", probe._sha256_file(path)
    )

    assert probe._load_reference(path) == (
        reference,
        probe._sha256_file(path),
    )

    path.write_text(json.dumps({**reference, "bank_admission_result_emitted": True}))
    monkeypatch.setattr(
        probe, "EXPECTED_CPU_CONFIRMATION_SHA256", probe._sha256_file(path)
    )
    with pytest.raises(RuntimeError, match="non-admission"):
        probe._load_reference(path)


def test_typed_outcome_comparison_does_not_treat_bool_as_int(monkeypatch):
    monkeypatch.setattr(
        probe,
        "_process_memory",
        lambda: {"VmRSS": 1, "VmHWM": 1, "VmSwap": 0, "ru_maxrss_kib": 1},
    )
    reference = _outcome()
    changed = dict(reference)
    changed["any_direct_transfer_pose_exists_initial"] = 1

    with pytest.raises(RuntimeError, match="exact outcome differs"):
        probe._run_two_exact_calls(
            launch_perf_ns=probe.time.perf_counter_ns() - 1_000_000,
            materialize=lambda: (
                object(),
                {"initial_agent_state_sha256": "a" * 64, "stable": "same"},
            ),
            compute=lambda state: changed,
            synchronize=lambda value: value,
            stable_initial_state=lambda value: {"stable": value["stable"]},
            jsonable=lambda value: value,
            persist_call=lambda value: None,
            reference_initial_state={"stable": "same"},
            reference_outcome=reference,
            outcome_keys=set(reference),
        )


def test_failure_path_writes_non_authorizing_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        probe,
        "_coordinator_import_boundary",
        lambda: {
            "jax_terra_profile_confirmation_modules_loaded": [],
            "passes": True,
        },
    )
    monkeypatch.setattr(probe, "_require_runtime_environment", lambda env: {})
    monkeypatch.setattr(
        probe,
        "_require_cpu_topology",
        lambda: (_ for _ in ()).throw(RuntimeError("topology failed")),
    )

    output = tmp_path / "output"
    with pytest.raises(probe.ProbeRejected):
        probe.run_probe(
            Path("/input"),
            Path("/confirmation"),
            output,
        )

    receipt = json.loads((output / probe.OUTPUT_NAME).read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure"]["message"] == "topology failed"
    assert receipt["decision"] == {
        "fixed_four_process_cpu_treatment_passes": False,
        "authorizes_one_deterministic_four_worker_256_profile": False,
        "authorizes_448_execution": False,
        "authorizes_bank_profile": False,
        "authorizes_witness": False,
        "authorizes_retry": False,
        "authorizes_worker_width_sweep": False,
        "authorizes_alternate_kernel": False,
        "authorizes_in_process_vectorization": False,
        "authorizes_static_admission": False,
        "authorizes_bank_admission": False,
        "authorizes_ppo": False,
    }
