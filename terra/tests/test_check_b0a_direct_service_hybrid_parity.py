import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.check_b0a_direct_service_hybrid_parity as hybrid


def _device(platform, kind):
    return SimpleNamespace(platform=platform, device_kind=kind)


def _reference_outcome():
    return {
        key: index for index, key in enumerate(hybrid.POPULATION_COUNTER_KEYS, start=1)
    }


def _capture(candidate_hash="a" * 64, output_hash="b" * 64):
    population = _reference_outcome()
    return {
        "candidate": {
            "ordered_unpadded_sha256": candidate_hash,
            "dispatch_roundtrip_sha256": candidate_hash,
        },
        "population_counters": population,
        "service": {
            "ordered_unpadded_outputs": {
                "ordered_unpadded_output_sha256": output_hash,
            }
        },
    }


def test_jax_platforms_must_be_absent_and_devices_are_exact():
    assert hybrid._require_jax_platforms_unset({})["passes"] is True
    with pytest.raises(RuntimeError, match="must be absent"):
        hybrid._require_jax_platforms_unset({"JAX_PLATFORMS": "gpu"})

    cpu = _device("cpu", "cpu")
    gpu = _device("gpu", hybrid.EXPECTED_DEVICE_KIND)
    selected_cpu, selected_gpu, receipt = hybrid._validate_devices(
        [cpu],
        [gpu],
        "gpu",
    )
    assert selected_cpu is cpu
    assert selected_gpu is gpu
    assert receipt["passes"] is True

    with pytest.raises(RuntimeError, match="exactly one JAX CPU"):
        hybrid._validate_devices([], [gpu], "gpu")
    with pytest.raises(RuntimeError, match="exactly one JAX GPU"):
        hybrid._validate_devices([cpu], [], "gpu")
    with pytest.raises(RuntimeError, match="RTX 4090"):
        hybrid._validate_devices([cpu], [_device("gpu", "A100")], "gpu")
    with pytest.raises(RuntimeError, match="GPU default backend"):
        hybrid._validate_devices([cpu], [gpu], "cpu")


def test_ordered_leaf_hash_matches_materialized_unpadded_arrays():
    first = (
        np.arange(4 * 2, dtype=np.int32).reshape(4, 2),
        np.arange(4 * 3, dtype=np.int32).reshape(4, 3),
        np.arange(4, dtype=np.int32).reshape(4, 1),
    )
    second = tuple(np.concatenate((leaf[:1] + 100, leaf[:3]), axis=0) for leaf in first)
    hasher = hybrid._OrderedLeafHasher(
        total_rows=5,
        leaf_count=3,
        required_dtype=np.dtype(np.int32),
    )
    hasher.update(first, 4)
    hasher.update(second, 1)
    receipt = hasher.finish()

    expected = tuple(
        np.concatenate((first[index], second[index][:1]), axis=0) for index in range(3)
    )
    expected_leaf_hashes = [hybrid._sha256_array(leaf) for leaf in expected]
    assert [leaf["sha256"] for leaf in receipt["leaves"]] == expected_leaf_hashes
    assert receipt["ordered_unpadded_output_sha256"] == (
        hybrid._canonical_json_sha256(expected_leaf_hashes)
    )
    assert receipt["leaves"][0]["shape"] == [5, 2]


def test_ordered_leaf_hash_fails_on_dtype_shape_or_row_loss():
    hasher = hybrid._OrderedLeafHasher(
        total_rows=2,
        leaf_count=1,
        required_dtype=np.dtype(np.int32),
    )
    with pytest.raises(RuntimeError, match="must be int32"):
        hasher.update((np.zeros((2, 1), dtype=np.float32),), 2)

    hasher.update((np.zeros((1, 1), dtype=np.int32),), 1)
    with pytest.raises(RuntimeError, match="expected 2"):
        hasher.finish()

    with pytest.raises(RuntimeError, match="shape changed"):
        hasher.update((np.zeros((1, 2), dtype=np.int32),), 1)


def test_population_counter_mapping_and_reference_gate():
    poses = np.zeros((3, 3), dtype=np.int32)
    candidates = np.zeros((36, 4), dtype=np.int32)
    accepted = np.zeros((5, 4), dtype=np.int32)
    movement = {
        "source_rows_logical": 3,
        "transition_attempts_logical": 12,
        "source_rows_padded_executed": 128,
        "transition_attempts_padded_executed": 512,
    }
    prefilter = {
        "candidate_rows_logical": 36,
        "candidate_rows_padded_executed": 128,
    }
    counters = hybrid._population_counters(
        poses=poses,
        movement_stats=movement,
        candidates=candidates,
        accepted=accepted,
        prefilter_stats=prefilter,
        cabin_headings=12,
    )
    assert counters["service_dig_candidate_attempts_logical"] == 5
    assert counters["service_candidate_rows_padded_executed"] == 8
    assert counters["dump_do_attempts_logical"] == 60
    assert counters["dump_do_transitions_padded_executed"] == 96

    hybrid._validate_population_against_reference(counters, counters)
    changed = {**counters, "admissible_pose_count_initial": 4}
    with pytest.raises(RuntimeError, match="admissible_pose_count_initial"):
        hybrid._validate_population_against_reference(changed, counters)


def test_full_population_gate_requires_hash_counter_and_final_result_equality():
    reference = _reference_outcome()
    cpu_capture = _capture()
    gpu_capture = _capture()
    result = hybrid._validate_full_population_parity(
        reference_outcome=reference,
        cpu_outcome=reference,
        cpu_capture=cpu_capture,
        gpu_outcome=reference,
        gpu_capture=gpu_capture,
    )
    assert result["all_full_population_parity_gates_pass"] is True

    gpu_capture["service"]["ordered_unpadded_outputs"] = {
        "ordered_unpadded_output_sha256": "0" * 64
    }
    with pytest.raises(RuntimeError, match="ordered_output_hashes_match"):
        hybrid._validate_full_population_parity(
            reference_outcome=reference,
            cpu_outcome=reference,
            cpu_capture=cpu_capture,
            gpu_outcome=reference,
            gpu_capture=gpu_capture,
        )

    changed_outcome = {**reference, "admissible_pose_count_initial": -1}
    with pytest.raises(RuntimeError, match="hybrid_final_outcome"):
        hybrid._validate_full_population_parity(
            reference_outcome=reference,
            cpu_outcome=reference,
            cpu_capture=cpu_capture,
            gpu_outcome=changed_outcome,
            gpu_capture=_capture(),
        )


def test_confirmation_loader_pins_bytes_and_complete_nonadmission_contract(
    tmp_path,
    monkeypatch,
):
    outcome_keys = {"counter"}
    receipt = {
        "schema": hybrid.EXPECTED_CPU_CONFIRMATION_SCHEMA,
        "release_id": "terramap-bench-v1.0.0",
        "result_scope": "one_complete_exact_scenario_cost_confirmation",
        "bank_admission_result_emitted": False,
        "single_scenario_exact_outcome_emitted": True,
        "selected_identity": {"map_id": "map"},
        "measurement": {"exact_outcome": {"counter": 1}},
        "validator": {"exact_entrypoint_call_count": 1},
    }
    path = tmp_path / "confirmation.json"
    path.write_text(json.dumps(receipt))
    digest = hybrid._sha256_file(path)
    monkeypatch.setattr(hybrid, "EXPECTED_CPU_CONFIRMATION_SHA256", digest)

    loaded, actual = hybrid._load_cpu_confirmation(
        path,
        outcome_keys=outcome_keys,
        selected_map_id="map",
    )
    assert loaded == receipt
    assert actual == digest

    receipt["bank_admission_result_emitted"] = True
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        hybrid,
        "EXPECTED_CPU_CONFIRMATION_SHA256",
        hybrid._sha256_file(path),
    )
    with pytest.raises(RuntimeError, match="non-admission"):
        hybrid._load_cpu_confirmation(
            path,
            outcome_keys=outcome_keys,
            selected_map_id="map",
        )


def test_patch_restores_all_canonical_functions_after_failure():
    direct_service = SimpleNamespace(
        _reachable_base_poses=object(),
        _prefilter_candidates=object(),
        _service_batch=object(),
    )
    originals = (
        direct_service._reachable_base_poses,
        direct_service._prefilter_candidates,
        direct_service._service_batch,
    )
    capture = SimpleNamespace(
        reachable=object(),
        prefilter=object(),
        service=object(),
    )
    with pytest.raises(RuntimeError, match="stop"):
        with hybrid._patched_direct_service(direct_service, capture):
            assert direct_service._reachable_base_poses is capture.reachable
            assert direct_service._prefilter_candidates is capture.prefilter
            assert direct_service._service_batch is capture.service
            raise RuntimeError("stop")
    assert (
        direct_service._reachable_base_poses,
        direct_service._prefilter_candidates,
        direct_service._service_batch,
    ) == originals


def test_nonadmission_receipt_allows_profile_authorization_but_no_results():
    receipt = {
        "admission_result_emitted": False,
        "bank_admission_result_emitted": False,
        "timing_result_emitted": False,
        "cost_profile_called": False,
        "bank_profile_called": False,
        "static_admission_authorized": False,
        "ppo_authorized": False,
        "decision": {"authorizes_one_hybrid_cost_profile": True},
    }
    hybrid._assert_non_admission_receipt(receipt)

    receipt["ppo_authorized"] = True
    with pytest.raises(RuntimeError, match="exceeds its scope"):
        hybrid._assert_non_admission_receipt(receipt)


def test_actual_confirmation_and_shared_code_are_still_pinned():
    receipt_path = Path(
        "/home/lorenzo/moleworks/.artifacts/"
        "terra_b0a_direct_service_cost_confirmation_20260727_v1/"
        "validation_cost_confirmation.json"
    )
    if not receipt_path.is_file():
        pytest.skip("Pinned local CPU confirmation is unavailable.")
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        hybrid.EXPECTED_CPU_CONFIRMATION_SHA256
    )
    reference = json.loads(receipt_path.read_text())
    repository = Path(__file__).resolve().parents[2]
    current = {
        "code_file_sha256": {
            relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for relative in (
                *hybrid.CPU_SHARED_EXECUTION_CODE_PATHS,
                *hybrid.CURRENT_ONLY_CODE_PATHS,
            )
        }
    }
    result = hybrid._validate_execution_code_reference(reference, current)
    assert result["all_shared_execution_dependency_hashes_match_cpu"] is True
    assert (
        result["benchmark_protocol_sha256"]
        == current["code_file_sha256"]["terra/benchmark_protocol.py"]
    )


def test_gpu_runner_source_has_only_the_exact_service_kernel_boundary():
    source = inspect.getsource(hybrid._GpuServiceRunner)
    assert "_service_batch.lower" in source
    for forbidden in (
        "_reachable_base_poses",
        "_movement_successor_batch",
        "_prefilter_candidates",
        "_dig_prefilter_batch",
        "compute_initial_direct_service",
    ):
        assert forbidden not in source
