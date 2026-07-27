import ast
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.check_b0a_direct_service_gpu_parity as parity


def _gpu(
    *,
    platform="gpu",
    kind="NVIDIA GeForce RTX 4090",
    process_index=0,
    local_hardware_id=0,
):
    return SimpleNamespace(
        platform=platform,
        device_kind=kind,
        process_index=process_index,
        local_hardware_id=local_hardware_id,
    )


def _reference():
    return {
        "schema": "terra_direct_service_batch_size_sweep_v1",
        "bank_admission_result_emitted": False,
        "full_exact_validator_called": False,
        "input": {
            "files_sha256_manifest_sha256": parity.EXPECTED_FILES_MANIFEST_SHA256
        },
        "experiment": {
            "parity_subset_logical_rows": parity.PARITY_ROWS,
            "parity_subset_sha256": parity.EXPECTED_CANDIDATE_SHA256,
            "service_kernel": "terra.benchmark_direct_service._service_batch",
        },
        "decision": {"selected_for_reprobe_batch_size": 4},
        "arms": {
            "4_open": {
                "parity": {
                    "concatenated_output_sha256": parity.EXPECTED_OUTPUT_SHA256,
                    "exact_concatenated_output_parity": True,
                }
            }
        },
    }


def test_jax_platforms_must_be_absent():
    assert parity._require_jax_platforms_unset({})["passes"] is True
    for value in ("", "gpu", "cuda"):
        with pytest.raises(RuntimeError, match="must be absent"):
            parity._require_jax_platforms_unset({"JAX_PLATFORMS": value})


def test_device_gate_requires_one_exact_rtx4090_gpu():
    receipt = parity._validate_gpu_device([_gpu()], "gpu")
    assert receipt["device_count"] == 1
    assert receipt["platform"] == "gpu"
    assert receipt["device_kind"] == "NVIDIA GeForce RTX 4090"

    with pytest.raises(RuntimeError, match="exactly one"):
        parity._validate_gpu_device([], "gpu")
    with pytest.raises(RuntimeError, match="exactly one"):
        parity._validate_gpu_device([_gpu(), _gpu()], "gpu")
    with pytest.raises(RuntimeError, match="device platform"):
        parity._validate_gpu_device([_gpu(platform="cpu")], "cpu")
    with pytest.raises(RuntimeError, match="default backend"):
        parity._validate_gpu_device([_gpu()], "cpu")
    with pytest.raises(RuntimeError, match="RTX 4090"):
        parity._validate_gpu_device([_gpu(kind="NVIDIA A100-SXM4-80GB")], "gpu")


def test_parity_hash_gate_receipts_every_output_leaf():
    outputs = (
        np.zeros((parity.PARITY_ROWS, 64, 64), dtype=np.int8),
        np.ones((parity.PARITY_ROWS, 64, 64), dtype=np.float32),
        np.zeros((parity.PARITY_ROWS, 5), dtype=np.int32),
    )
    receipt = parity._validate_parity_hashes(
        candidate_sha256=parity.EXPECTED_CANDIDATE_SHA256,
        output_sha256=parity.EXPECTED_OUTPUT_SHA256,
        outputs=outputs,
    )
    assert receipt["exact_dtype_shape_leaf_count_and_content_parity"] is True
    assert receipt["leaf_count"] == 3
    assert receipt["leaves"] == [
        {"leaf_index": 0, "dtype": "int8", "shape": [18, 64, 64]},
        {"leaf_index": 1, "dtype": "float32", "shape": [18, 64, 64]},
        {"leaf_index": 2, "dtype": "int32", "shape": [18, 5]},
    ]

    with pytest.raises(RuntimeError, match="candidate rows changed"):
        parity._validate_parity_hashes(
            candidate_sha256="0" * 64,
            output_sha256=parity.EXPECTED_OUTPUT_SHA256,
            outputs=outputs,
        )
    with pytest.raises(RuntimeError, match="differs from the pinned CPU"):
        parity._validate_parity_hashes(
            candidate_sha256=parity.EXPECTED_CANDIDATE_SHA256,
            output_sha256="0" * 64,
            outputs=outputs,
        )
    with pytest.raises(RuntimeError, match="leading dimension"):
        parity._validate_parity_hashes(
            candidate_sha256=parity.EXPECTED_CANDIDATE_SHA256,
            output_sha256=parity.EXPECTED_OUTPUT_SHA256,
            outputs=(np.zeros((17, 5), dtype=np.int32),),
        )
    with pytest.raises(RuntimeError, match="leaf count changed"):
        parity._validate_parity_hashes(
            candidate_sha256=parity.EXPECTED_CANDIDATE_SHA256,
            output_sha256=parity.EXPECTED_OUTPUT_SHA256,
            outputs=outputs[:2],
        )


def test_memory_receipts_normalize_process_peaks_against_capacity():
    gpu = parity._normalize_accelerator_memory(
        {
            "bytes_limit": 100,
            "bytes_in_use": 40,
            "peak_bytes_in_use": 80,
        }
    )
    host = parity._normalize_host_memory(80, 100)
    assert gpu["peak_fraction_of_capacity"] == pytest.approx(0.8)
    assert gpu["peak_at_most_eighty_percent"] is True
    assert host["peak_fraction_of_capacity"] == pytest.approx(0.8)
    assert host["peak_at_most_eighty_percent"] is True

    with pytest.raises(RuntimeError, match="unavailable"):
        parity._normalize_accelerator_memory(None)
    with pytest.raises(RuntimeError, match="cannot exceed bytes_limit"):
        parity._normalize_accelerator_memory(
            {
                "bytes_limit": 100,
                "bytes_in_use": 90,
                "peak_bytes_in_use": 101,
            }
        )
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        parity._normalize_accelerator_memory(
            {
                "bytes_limit": 100.0,
                "bytes_in_use": 40,
                "peak_bytes_in_use": 80,
            }
        )


def test_cpu_reference_gate_is_fail_closed():
    parity._validate_cpu_reference(_reference())

    changed = _reference()
    changed["experiment"]["parity_subset_sha256"] = "0" * 64
    changed["bank_admission_result_emitted"] = True
    with pytest.raises(
        RuntimeError,
        match="non-admission result scope, candidate hash",
    ):
        parity._validate_cpu_reference(changed)


def test_rebuilt_contract_requires_exact_input_protocol_and_state():
    stable = lambda value: value["stable"]
    reference = {
        "selected_identity": {"map_id": "map"},
        "input": {
            "files_sha256_manifest_sha256": "a" * 64,
            "source_grouping": {"count": 1},
            "verified_selected_files": {"file": "b" * 64},
        },
        "protocol": {"hash": "c" * 64},
        "initial_state": {"stable": {"hash": "d" * 64}},
    }
    arguments = {
        "reference": reference,
        "selected_identity": reference["selected_identity"],
        "manifest_sha256": reference["input"]["files_sha256_manifest_sha256"],
        "source_grouping": reference["input"]["source_grouping"],
        "verified_selected_files": reference["input"]["verified_selected_files"],
        "protocol": reference["protocol"],
        "initial_state": reference["initial_state"],
        "stable_initial_state": stable,
    }
    parity._validate_rebuilt_contract(**arguments)

    arguments["protocol"] = {"hash": "0" * 64}
    arguments["initial_state"] = {"stable": {"hash": "1" * 64}}
    with pytest.raises(
        RuntimeError,
        match="protocol, explicit initial state",
    ):
        parity._validate_rebuilt_contract(**arguments)


def test_execution_code_reference_requires_every_pinned_dependency_hash():
    hashes = {
        path: f"{index:064x}"
        for index, path in enumerate(parity.CPU_SHARED_EXECUTION_CODE_PATHS)
    }
    reference = {
        "validator": {"code_before": {"code_file_sha256": hashes}},
    }
    current = {"code_file_sha256": dict(hashes)}

    receipt = parity._validate_execution_code_reference(reference, current)
    assert receipt["all_execution_dependency_hashes_match_cpu"] is True

    changed = {"code_file_sha256": {**hashes, "terra/state.py": "f" * 64}}
    with pytest.raises(RuntimeError, match="terra/state.py"):
        parity._validate_execution_code_reference(reference, changed)

    missing = {"code_file_sha256": dict(hashes)}
    del missing["code_file_sha256"]["terra/env.py"]
    with pytest.raises(RuntimeError, match="missing dependencies"):
        parity._validate_execution_code_reference(reference, missing)


def test_actual_cpu_receipt_matches_current_shared_execution_code():
    receipt_path = Path(
        "/home/lorenzo/moleworks/.artifacts/"
        "terra_b0a_direct_service_batch_sweep_20260727_v1/"
        "direct_service_batch_size_sweep.json"
    )
    if not receipt_path.is_file():
        pytest.skip("Pinned local CPU sweep receipt is unavailable.")
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        == parity.EXPECTED_CPU_SWEEP_SHA256
    )
    reference = json.loads(receipt_path.read_text())
    repository = Path(__file__).resolve().parents[2]
    current_paths = (
        *parity.CPU_SHARED_EXECUTION_CODE_PATHS,
        parity.CURRENT_ONLY_PROTOCOL_CODE_PATH,
    )
    current = {
        "code_file_sha256": {
            relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for relative in current_paths
        }
    }

    assert (
        parity.CURRENT_ONLY_PROTOCOL_CODE_PATH
        not in reference["validator"]["code_before"]["code_file_sha256"]
    )
    result = parity._validate_execution_code_reference(reference, current)
    assert result["all_execution_dependency_hashes_match_cpu"] is True
    assert set(result["compared_file_sha256"]) == set(
        parity.CPU_SHARED_EXECUTION_CODE_PATHS
    )


def test_runtime_reference_allows_only_gpu_and_unset_jax_platforms():
    reference = {
        "hostname": "starship",
        "platform": "Linux",
        "machine": "x86_64",
        "python": "3.12.0",
        "packages": {"jax": "1", "jaxlib": "1", "numpy": "2", "scipy": "1"},
        "compilation_cache_environment": {
            "JAX_COMPILATION_CACHE_DIR": None,
            "JAX_ENABLE_COMPILATION_CACHE": None,
            "JAX_PLATFORMS": "cpu",
        },
    }
    current = {
        **reference,
        "compilation_cache_environment": {
            **reference["compilation_cache_environment"],
            "JAX_PLATFORMS": None,
        },
    }
    receipt = parity._validate_runtime_reference(reference, current)
    assert receipt["same_host_platform_python_and_packages"] is True
    assert receipt["intentional_differences"]["JAX_PLATFORMS"] == {
        "cpu_reference": "cpu",
        "gpu_treatment": None,
    }

    changed = {**current, "packages": {**current["packages"], "jax": "2"}}
    with pytest.raises(RuntimeError, match="packages"):
        parity._validate_runtime_reference(reference, changed)


def test_parity_tool_cannot_call_cost_confirmation_or_full_validator():
    tree = ast.parse(inspect.getsource(parity))
    forbidden_names = {
        "compute_initial_direct_service",
        "run_probe",
        "run_confirmation",
        "run_sweep",
    }
    forbidden_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_names
    ]
    assert forbidden_calls == []
    assert "probe_tool._assert_cost_only(receipt)" in inspect.getsource(
        parity.run_parity
    )
