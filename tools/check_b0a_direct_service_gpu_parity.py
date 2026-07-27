#!/usr/bin/env python3
"""Check exact GPU parity for the pinned B0a direct-service subset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "terra_direct_service_gpu_parity_v1"
OUTPUT_NAME = "direct_service_gpu_parity.json"
EXPECTED_CPU_SWEEP_SHA256 = (
    "727c358357a026ff75b9e8340d6cd117efff6c782d455eba82630fe8e9312db9"
)
EXPECTED_CANDIDATE_SHA256 = (
    "b19744abe759e0d229cc6a6d6095dd39beef3495311f10bb4ef7c2476438dd56"
)
EXPECTED_OUTPUT_SHA256 = (
    "fa8dd4f8d579cd1c08ffd64876117f6be3ab41b6164a9220eacb37f4f64e1106"
)
EXPECTED_FILES_MANIFEST_SHA256 = (
    "89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347"
)
EXPECTED_DEVICE_PLATFORM = "gpu"
EXPECTED_DEVICE_KIND = "NVIDIA GeForce RTX 4090"
PARITY_ROWS = 18
SERVICE_BATCH_SIZE = 4
EXPECTED_OUTPUT_LEAF_COUNT = 3
MAX_MEMORY_FRACTION = 0.8
CPU_SHARED_EXECUTION_CODE_PATHS = (
    "terra/actions.py",
    "terra/agent.py",
    "terra/benchmark_direct_service.py",
    "terra/benchmark_state.py",
    "terra/config.py",
    "terra/env.py",
    "terra/map.py",
    "terra/maps_buffer.py",
    "terra/settings.py",
    "terra/state.py",
    "terra/utils.py",
    "terra/wrappers.py",
    "tools/confirm_b0a_direct_service_cost.py",
    "tools/sweep_b0a_direct_service_batch_size.py",
)
CURRENT_ONLY_PROTOCOL_CODE_PATH = "terra/benchmark_protocol.py"


def _require_jax_platforms_unset(environment: Mapping[str, str]) -> dict[str, Any]:
    if "JAX_PLATFORMS" in environment:
        raise RuntimeError(
            "JAX_PLATFORMS must be absent for the GPU parity run; explicitly "
            "selecting gpu enters the host's broken ROCm registration path."
        )
    return {
        "policy": "environment_variable_must_be_absent",
        "observed_value": None,
        "passes": True,
    }


def _validate_gpu_device(
    devices: Sequence[Any],
    default_backend: str,
) -> dict[str, Any]:
    if len(devices) != 1:
        raise RuntimeError(
            f"GPU parity requires exactly one visible JAX device, got {len(devices)}."
        )
    device = devices[0]
    device_platform = getattr(device, "platform", None)
    device_kind = getattr(device, "device_kind", None)
    if device_platform != EXPECTED_DEVICE_PLATFORM:
        raise RuntimeError(
            "GPU parity requires device platform "
            f"{EXPECTED_DEVICE_PLATFORM!r}, got {device_platform!r}."
        )
    if default_backend != EXPECTED_DEVICE_PLATFORM:
        raise RuntimeError(
            "GPU parity requires default backend "
            f"{EXPECTED_DEVICE_PLATFORM!r}, got {default_backend!r}."
        )
    if device_kind != EXPECTED_DEVICE_KIND:
        raise RuntimeError(
            f"GPU parity requires {EXPECTED_DEVICE_KIND!r}, got {device_kind!r}."
        )
    return {
        "device_count": 1,
        "selected_device": str(device),
        "platform": device_platform,
        "default_backend": default_backend,
        "device_kind": device_kind,
        "process_index": int(getattr(device, "process_index")),
        "local_hardware_id": int(getattr(device, "local_hardware_id")),
        "passes": True,
    }


def _finite_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0:
        raise RuntimeError(f"{name} must be a nonnegative integer.")
    return result


def _normalize_accelerator_memory(
    stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if stats is None:
        raise RuntimeError("JAX GPU memory statistics are unavailable.")
    capacity = _finite_nonnegative_integer(stats.get("bytes_limit"), "bytes_limit")
    current = _finite_nonnegative_integer(
        stats.get("bytes_in_use"),
        "bytes_in_use",
    )
    peak = _finite_nonnegative_integer(
        stats.get("peak_bytes_in_use"),
        "peak_bytes_in_use",
    )
    if capacity == 0:
        raise RuntimeError("bytes_limit must be positive.")
    if current > peak:
        raise RuntimeError("bytes_in_use cannot exceed peak_bytes_in_use.")
    if peak > capacity:
        raise RuntimeError("peak_bytes_in_use cannot exceed bytes_limit.")
    return {
        "capacity_bytes": capacity,
        "current_bytes": current,
        "peak_bytes": peak,
        "current_fraction_of_capacity": current / capacity,
        "peak_fraction_of_capacity": peak / capacity,
        "maximum_allowed_fraction_for_later_cost_probe": MAX_MEMORY_FRACTION,
        "peak_at_most_eighty_percent": peak <= MAX_MEMORY_FRACTION * capacity,
        "measurement_kind": "process_lifetime_allocator_peak",
    }


def _normalize_host_memory(
    peak_rss_kib: Any,
    capacity_kib: Any,
) -> dict[str, Any]:
    peak = _finite_nonnegative_integer(peak_rss_kib, "process_peak_rss_kib")
    capacity = _finite_nonnegative_integer(
        capacity_kib,
        "host_physical_memory_kib",
    )
    if capacity == 0:
        raise RuntimeError("host_physical_memory_kib must be positive.")
    if peak > capacity:
        raise RuntimeError(
            "process_peak_rss_kib cannot exceed host_physical_memory_kib."
        )
    return {
        "process_peak_rss_kib": peak,
        "host_physical_memory_kib": capacity,
        "peak_fraction_of_capacity": peak / capacity,
        "maximum_allowed_fraction_for_later_cost_probe": MAX_MEMORY_FRACTION,
        "peak_at_most_eighty_percent": peak <= MAX_MEMORY_FRACTION * capacity,
        "measurement_kind": "process_lifetime_cumulative_peak",
    }


def _output_structure(outputs: tuple[np.ndarray, ...]) -> list[dict[str, Any]]:
    if not outputs:
        raise RuntimeError("The parity execution returned no output leaves.")
    structure = []
    for index, leaf in enumerate(outputs):
        value = np.asarray(leaf)
        if value.ndim == 0 or value.shape[0] != PARITY_ROWS:
            raise RuntimeError(
                f"Output leaf {index} has shape {value.shape}; its leading "
                f"dimension must be {PARITY_ROWS}."
            )
        structure.append(
            {
                "leaf_index": index,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        )
    return structure


def _validate_parity_hashes(
    *,
    candidate_sha256: str,
    output_sha256: str,
    outputs: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    if candidate_sha256 != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            "The first 18 accepted candidate rows changed: "
            f"{EXPECTED_CANDIDATE_SHA256} != {candidate_sha256}."
        )
    structure = _output_structure(outputs)
    if len(structure) != EXPECTED_OUTPUT_LEAF_COUNT:
        raise RuntimeError(
            "GPU direct-service output leaf count changed: "
            f"{EXPECTED_OUTPUT_LEAF_COUNT} != {len(structure)}."
        )
    if output_sha256 != EXPECTED_OUTPUT_SHA256:
        raise RuntimeError(
            "GPU direct-service output differs from the pinned CPU output: "
            f"{EXPECTED_OUTPUT_SHA256} != {output_sha256}."
        )
    return {
        "exact_dtype_shape_leaf_count_and_content_parity": True,
        "candidate_subset_sha256": candidate_sha256,
        "concatenated_output_sha256": output_sha256,
        "leaf_count": len(structure),
        "leaves": structure,
        "hash_contract": (
            "each leaf hash prefixes dtype and shape to contiguous content; "
            "the final hash covers the ordered leaf-hash list"
        ),
    }


def _validate_cpu_reference(receipt: dict[str, Any]) -> None:
    checks = {
        "schema": receipt.get("schema") == "terra_direct_service_batch_size_sweep_v1",
        "non-admission result scope": (
            receipt.get("bank_admission_result_emitted") is False
            and receipt.get("full_exact_validator_called") is False
        ),
        "manifest": (
            receipt.get("input", {}).get("files_sha256_manifest_sha256")
            == EXPECTED_FILES_MANIFEST_SHA256
        ),
        "parity row count": (
            receipt.get("experiment", {}).get("parity_subset_logical_rows")
            == PARITY_ROWS
        ),
        "candidate hash": (
            receipt.get("experiment", {}).get("parity_subset_sha256")
            == EXPECTED_CANDIDATE_SHA256
        ),
        "service kernel": (
            receipt.get("experiment", {}).get("service_kernel")
            == "terra.benchmark_direct_service._service_batch"
        ),
        "batch 4 retained": (
            receipt.get("decision", {}).get("selected_for_reprobe_batch_size")
            == SERVICE_BATCH_SIZE
        ),
        "CPU output hash": (
            receipt.get("arms", {})
            .get("4_open", {})
            .get("parity", {})
            .get("concatenated_output_sha256")
            == EXPECTED_OUTPUT_SHA256
        ),
        "CPU reference parity": (
            receipt.get("arms", {})
            .get("4_open", {})
            .get("parity", {})
            .get("exact_concatenated_output_parity")
            is True
        ),
    }
    failures = [name for name, passes in checks.items() if not passes]
    if failures:
        raise RuntimeError(
            "Pinned CPU sweep violates the parity reference contract: "
            + ", ".join(failures)
            + "."
        )


def _validate_rebuilt_contract(
    *,
    reference: dict[str, Any],
    selected_identity: dict[str, Any],
    manifest_sha256: str,
    source_grouping: dict[str, Any],
    verified_selected_files: dict[str, str],
    protocol: dict[str, Any],
    initial_state: dict[str, Any],
    stable_initial_state: Any,
) -> None:
    comparisons = {
        "selected identity": (selected_identity, reference["selected_identity"]),
        "files.sha256 manifest": (
            manifest_sha256,
            reference["input"]["files_sha256_manifest_sha256"],
        ),
        "source grouping": (
            source_grouping,
            reference["input"]["source_grouping"],
        ),
        "selected input files": (
            verified_selected_files,
            reference["input"]["verified_selected_files"],
        ),
        "protocol": (protocol, reference["protocol"]),
        "explicit initial state": (
            stable_initial_state(initial_state),
            stable_initial_state(reference["initial_state"]),
        ),
    }
    mismatches = [
        name for name, (current, expected) in comparisons.items() if current != expected
    ]
    if mismatches:
        raise RuntimeError(
            "GPU parity run no longer matches the pinned CPU contract: "
            + ", ".join(mismatches)
            + "."
        )


def _validate_execution_code_reference(
    reference: dict[str, Any],
    current_code: dict[str, Any],
) -> dict[str, Any]:
    reference_hashes = reference["validator"]["code_before"]["code_file_sha256"]
    current_hashes = current_code["code_file_sha256"]
    missing = [
        path
        for path in CPU_SHARED_EXECUTION_CODE_PATHS
        if path not in reference_hashes or path not in current_hashes
    ]
    if missing:
        raise RuntimeError(
            f"CPU/GPU code comparison is missing dependencies: {missing}."
        )
    changed = [
        path
        for path in CPU_SHARED_EXECUTION_CODE_PATHS
        if current_hashes[path] != reference_hashes[path]
    ]
    if changed:
        raise RuntimeError(
            "Exact service execution dependencies changed since the CPU "
            f"reference: {changed}."
        )
    return {
        "compared_file_sha256": {
            path: current_hashes[path] for path in CPU_SHARED_EXECUTION_CODE_PATHS
        },
        "all_execution_dependency_hashes_match_cpu": True,
        "profile_helper_comparison": (
            "tools/profile_b0a_direct_service_cost.py is not an execution "
            "dependency comparison: its frozen-protocol logic was moved "
            "without changing the exact receipted protocol. Rebuilt selected "
            "inputs, protocol, and initial state are compared separately."
        ),
    }


def _validate_runtime_reference(
    reference_machine: dict[str, Any],
    current_machine: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        "hostname": (
            current_machine["hostname"],
            reference_machine["hostname"],
        ),
        "platform": (
            current_machine["platform"],
            reference_machine["platform"],
        ),
        "machine": (
            current_machine["machine"],
            reference_machine["machine"],
        ),
        "python": (
            current_machine["python"],
            reference_machine["python"],
        ),
        "packages": (
            current_machine["packages"],
            reference_machine["packages"],
        ),
    }
    reference_cache = reference_machine["compilation_cache_environment"]
    current_cache = current_machine["compilation_cache_environment"]
    for key in ("JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE"):
        comparisons[key] = (current_cache[key], reference_cache[key])
    mismatches = [
        name for name, (current, expected) in comparisons.items() if current != expected
    ]
    if mismatches:
        raise RuntimeError(
            "GPU runtime differs from the CPU reference beyond the device "
            f"treatment: {mismatches}."
        )
    if reference_cache.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("CPU reference did not explicitly select JAX CPU.")
    if current_cache.get("JAX_PLATFORMS") is not None:
        raise RuntimeError("GPU treatment must leave JAX_PLATFORMS unset.")
    return {
        "same_host_platform_python_and_packages": True,
        "same_compilation_cache_settings": True,
        "intentional_differences": {
            "jax_device": "CPU reference to single RTX 4090 GPU",
            "JAX_PLATFORMS": {"cpu_reference": "cpu", "gpu_treatment": None},
        },
    }


def _code_receipt(repository: Path, sweep_tool: Any, probe_tool: Any) -> dict[str, Any]:
    base, _ = sweep_tool._code_receipt(repository)
    code_hashes = dict(base["code_file_sha256"])
    for script_path in (
        Path(__file__).resolve(),
        repository / CURRENT_ONLY_PROTOCOL_CODE_PATH,
    ):
        code_hashes[str(script_path.relative_to(repository))] = probe_tool._sha256_file(
            script_path
        )
    return {
        "git": base["git"],
        "code_file_sha256": code_hashes,
        "code_bundle_sha256": probe_tool._canonical_json_sha256(code_hashes),
    }


def _require_clean_worktree(code_receipt: dict[str, Any]) -> None:
    git_receipt = code_receipt["git"]
    if git_receipt.get("dirty") or git_receipt.get("porcelain_v1"):
        raise RuntimeError("GPU parity must run from a clean committed Terra worktree.")


def run_parity(
    b0a_root: Path,
    cpu_sweep_path: Path,
    output_directory: Path,
) -> Path:
    jax_platforms = _require_jax_platforms_unset(os.environ)

    import jax
    import jax.numpy as jnp

    import terra.benchmark_direct_service as direct_service
    import tools.confirm_b0a_direct_service_cost as confirmation
    import tools.profile_b0a_direct_service_cost as probe_tool
    import tools.sweep_b0a_direct_service_batch_size as sweep_tool

    b0a_root = b0a_root.resolve()
    cpu_sweep_path = cpu_sweep_path.resolve()
    output_directory = output_directory.resolve()
    output_path = output_directory / OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(output_path)

    repository = Path(__file__).resolve().parents[1]
    code_before = _code_receipt(repository, sweep_tool, probe_tool)
    _require_clean_worktree(code_before)

    cpu_sweep_sha256 = probe_tool._sha256_file(cpu_sweep_path)
    if cpu_sweep_sha256 != EXPECTED_CPU_SWEEP_SHA256:
        raise RuntimeError(
            "CPU sweep receipt SHA-256 changed: "
            f"{EXPECTED_CPU_SWEEP_SHA256} != {cpu_sweep_sha256}."
        )
    cpu_reference = json.loads(cpu_sweep_path.read_text())
    _validate_cpu_reference(cpu_reference)
    execution_code_reference = _validate_execution_code_reference(
        cpu_reference,
        code_before,
    )
    reference_root = Path(cpu_reference["input"]["b0a_root"]).resolve()
    if b0a_root != reference_root:
        raise RuntimeError(
            f"GPU parity requires B0a root {reference_root}, got {b0a_root}."
        )

    probe_path = Path(cpu_reference["input"]["probe_receipt"]).resolve()
    probe_receipt = confirmation._load_pinned_probe(probe_path)
    devices = jax.devices()
    device = devices[0] if len(devices) == 1 else None
    gpu = _validate_gpu_device(devices, jax.default_backend())
    packages = {
        name: probe_tool._package_version(name)
        for name in ("jax", "jaxlib", "numpy", "scipy")
    }
    runtime = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "compilation_cache_environment": {
            key: os.environ.get(key)
            for key in (
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_PLATFORMS",
            )
        },
    }
    runtime_reference = _validate_runtime_reference(
        cpu_reference["machine"],
        runtime,
    )
    memory_before_raw = probe_tool._device_memory_stats(device)
    memory_before = _normalize_accelerator_memory(memory_before_raw)
    host_capacity_kib = confirmation._host_memory_capacity_kib()
    host_rss_before = probe_tool._max_rss_kib()

    identities_path = b0a_root / "identities.jsonl"
    selected_record, group_records, source_group_id = probe_tool._select_group_records(
        identities_path
    )
    source_grouping = probe_tool._frozen_source_group_receipt(identities_path)
    group = [probe_tool._load_scenario(b0a_root, record) for record in group_records]
    selected = next(
        scenario
        for scenario in group
        if scenario.record["map_id"] == probe_tool.SELECTED_MAP_ID
    )
    selected_identity = confirmation._selected_identity(
        selected_record,
        group,
        source_group_id,
    )
    selected_paths = {
        identities_path,
        b0a_root / "provenance.json",
        b0a_root / "source_registry.jsonl",
        *(path for scenario in group for path in scenario.selected_files),
    }
    verified_selected_files = probe_tool._verify_selected_inputs(
        b0a_root,
        selected_paths,
    )
    manifest_sha256 = probe_tool._sha256_file(b0a_root / "files.sha256")
    env_config, protocol = probe_tool._frozen_env_config()
    state, initial_state = probe_tool._materialize_initial_state(
        selected,
        group,
        source_group_id,
        env_config,
    )
    _validate_rebuilt_contract(
        reference=cpu_reference,
        selected_identity=selected_identity,
        manifest_sha256=manifest_sha256,
        source_grouping=source_grouping,
        verified_selected_files=verified_selected_files,
        protocol=protocol,
        initial_state=initial_state,
        stable_initial_state=confirmation._stable_initial_state,
    )
    if confirmation._stable_initial_state(
        initial_state
    ) != confirmation._stable_initial_state(probe_receipt["initial_state"]):
        raise RuntimeError("Pinned probe and CPU sweep initial states disagree.")
    if int(direct_service._SERVICE_BATCH_SIZE) != SERVICE_BATCH_SIZE:
        raise RuntimeError(
            "The exact direct-service batch size changed: "
            f"{SERVICE_BATCH_SIZE} != {direct_service._SERVICE_BATCH_SIZE}."
        )

    graph = probe_tool._run_graph_prefilter(state)
    replay_candidates = graph["accepted"]
    if len(replay_candidates) < PARITY_ROWS:
        raise RuntimeError(
            f"Need {PARITY_ROWS} accepted rows, got {len(replay_candidates)}."
        )
    parity_rows = replay_candidates[:PARITY_ROWS]
    candidate_sha256 = probe_tool._sha256_array(parity_rows)
    if candidate_sha256 != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(
            "The first 18 accepted candidate rows changed: "
            f"{EXPECTED_CANDIDATE_SHA256} != {candidate_sha256}."
        )

    first_rows, _ = direct_service._pad_rows(
        parity_rows[:SERVICE_BATCH_SIZE],
        SERVICE_BATCH_SIZE,
    )
    device_rows = jnp.asarray(first_rows, dtype=jnp.int32)
    probe_tool._synchronize(device_rows)
    compiled = direct_service._service_batch.lower(state, device_rows).compile()
    execution, outputs = sweep_tool._execute_subset(
        compiled,
        state,
        parity_rows,
        SERVICE_BATCH_SIZE,
        capture_outputs=True,
    )
    if outputs is None:
        raise RuntimeError("GPU parity execution returned no captured outputs.")
    parity = _validate_parity_hashes(
        candidate_sha256=candidate_sha256,
        output_sha256=sweep_tool._output_sha256(outputs),
        outputs=outputs,
    )
    expected_execution = {
        "logical_rows": PARITY_ROWS,
        "padded_rows": 20,
        "batch_count": 5,
    }
    actual_execution = {
        key: execution[key] for key in ("logical_rows", "padded_rows", "batch_count")
    }
    if actual_execution != expected_execution:
        raise RuntimeError(
            f"GPU parity execution counters changed: {actual_execution}."
        )

    memory_after_raw = probe_tool._device_memory_stats(device)
    memory_after = _normalize_accelerator_memory(memory_after_raw)
    host_rss_after = probe_tool._max_rss_kib()
    host_memory = _normalize_host_memory(host_rss_after, host_capacity_kib)
    code_after = _code_receipt(repository, sweep_tool, probe_tool)
    if code_after != code_before:
        raise RuntimeError("Terra or GPU parity code changed during execution.")

    receipt = {
        "schema": SCHEMA,
        "release_id": probe_tool.RELEASE_ID,
        "result_scope": "non_admission_exact_cpu_gpu_parity",
        "admission_result_emitted": False,
        "bank_admission_result_emitted": False,
        "full_exact_validator_called": False,
        "cost_probe_called": False,
        "timing_result_emitted": False,
        "selected_identity": selected_identity,
        "input": {
            "b0a_root": str(b0a_root),
            "cpu_sweep_receipt": str(cpu_sweep_path),
            "cpu_sweep_receipt_sha256": cpu_sweep_sha256,
            "probe_receipt": str(probe_path),
            "probe_receipt_sha256": confirmation.EXPECTED_PROBE_SHA256,
            "files_sha256_manifest_sha256": manifest_sha256,
            "source_grouping": source_grouping,
            "verified_selected_files": verified_selected_files,
        },
        "protocol": protocol,
        "initial_state": initial_state,
        "experiment": {
            "treatment": "execution_device_only",
            "cpu_reference_platform": "cpu",
            "gpu_treatment_platform": EXPECTED_DEVICE_PLATFORM,
            "service_kernel": "terra.benchmark_direct_service._service_batch",
            "service_batch_size": SERVICE_BATCH_SIZE,
            "parity_subset_logical_rows": PARITY_ROWS,
            "parity_subset_padded_rows": actual_execution["padded_rows"],
            "parity_batch_count": actual_execution["batch_count"],
            "candidate_subset_sha256": candidate_sha256,
            "cpu_reference_output_sha256": EXPECTED_OUTPUT_SHA256,
        },
        "parity": parity,
        "memory": {
            "gpu_before": memory_before,
            "gpu_after": memory_after,
            "host_before_process_peak_rss_kib": host_rss_before,
            "host_after": host_memory,
            "scope": (
                "diagnostic process-lifetime peaks; the later cost probe owns "
                "the preregistered 80% memory gate"
            ),
        },
        "validator": {
            "code_before": code_before,
            "code_after": code_after,
            "pre_and_post_execution_receipts_identical": True,
            "cpu_execution_code_reference": execution_code_reference,
            "full_validator_entrypoint_call_count": 0,
        },
        "machine": {
            **runtime,
            "jax": gpu,
            "jax_platforms_environment": jax_platforms,
            "cpu_runtime_reference": runtime_reference,
        },
        "decision": {
            "exact_cpu_gpu_parity_passes": True,
            "authorizes_one_gpu_batch4_cost_probe": True,
            "authorizes_complete_scenario_confirmation": False,
            "authorizes_bank_profile": False,
            "authorizes_static_admission": False,
            "authorizes_ppo": False,
        },
        "command": [str(argument) for argument in sys.argv],
    }
    probe_tool._assert_cost_only(receipt)
    output_directory.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the pinned 18-row exact direct-service subset on one RTX "
            "4090 and emit a non-admission CPU/GPU parity receipt."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--cpu-sweep-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_parity(args.b0a_root, args.cpu_sweep_receipt, args.output_directory)


if __name__ == "__main__":
    main()
