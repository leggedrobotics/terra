#!/usr/bin/env python3
"""Gate one fixed four-process CPU direct-service treatment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "terra_direct_service_cpu_process_probe_v1"
OUTPUT_NAME = "cpu_process_probe.json"
WORKER_MODULE = "tools.probe_b0a_direct_service_cpu_processes"
EXPECTED_CPU_CONFIRMATION_SHA256 = (
    "f4bc393a7eabcdc058eb5f4de69281c5e1bed9feef275f9f75833f3f3c4aaae7"
)
EXPECTED_CPU_CONFIRMATION_SCHEMA = (
    "terra_direct_service_validation_cost_confirmation_v1"
)
EXPECTED_RELEASE_ID = "terramap-bench-v1.0.0"
EXPECTED_HOSTNAME = "starship"
EXPECTED_CPU_COUNT = 32
CALLS_PER_WORKER = 2
MAX_CALL_SECONDS = 60 * 60
MAX_MEMORY_FRACTION = 0.8
COHORT_HARD_TIMEOUT_SECONDS = 3 * 60 * 60
PROJECTION_LIMITS_SECONDS = {256: 24 * 60 * 60, 448: 48 * 60 * 60}
AFFINITY_SETS = (
    (0, 1, 2, 3, 16, 17, 18, 19),
    (4, 5, 6, 7, 20, 21, 22, 23),
    (8, 9, 10, 11, 24, 25, 26, 27),
    (12, 13, 14, 15, 28, 29, 30, 31),
)
WORKER_COUNT = len(AFFINITY_SETS)

SHARED_EXECUTION_CODE_PATHS = (
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
)
CURRENT_CODE_PATHS = (
    "terra/benchmark_protocol.py",
    "tools/profile_b0a_direct_service_cost.py",
    "tools/probe_b0a_direct_service_cpu_processes.py",
)
POPULATION_COUNTER_KEYS = (
    "admissible_pose_count_initial",
    "base_pose_cabin_heading_candidates_initial",
    "movement_source_rows_logical",
    "movement_transition_attempts_logical",
    "movement_source_rows_padded_executed",
    "movement_transition_attempts_padded_executed",
    "dig_prefilter_candidate_rows_logical",
    "dig_prefilter_candidate_rows_padded_executed",
    "service_dig_candidate_attempts_logical",
    "service_candidate_rows_padded_executed",
    "service_dig_do_transitions_padded_executed",
    "dump_do_attempts_logical",
    "dump_do_transitions_padded_executed",
)
DIRECT_SERVICE_OUTCOME_KEYS = {
    "required_volume",
    "admissible_pose_count_initial",
    "base_pose_cabin_heading_candidates_initial",
    "successful_target_dig_replays",
    "movement_source_rows_logical",
    "movement_transition_attempts_logical",
    "movement_source_rows_padded_executed",
    "movement_transition_attempts_padded_executed",
    "dig_prefilter_candidate_rows_logical",
    "dig_prefilter_candidate_rows_padded_executed",
    "service_dig_candidate_attempts_logical",
    "service_candidate_rows_padded_executed",
    "service_dig_do_transitions_padded_executed",
    "dump_do_attempts_logical",
    "dump_do_transitions_padded_executed",
    "legal_complete_dump_attempts",
    "wrong_complete_dump_attempts",
    "rejected_dump_attempts",
    "workspace_serviceable_volume_initial",
    "direct_serviceable_volume_initial",
    "initial_workspace_coverage",
    "direct_service_coverage_initial",
    "any_direct_transfer_pose_exists_initial",
}

_WORKER_INDEX_ENV = "_TERRA_CPU_PROBE_WORKER_INDEX"
_B0A_ROOT_ENV = "_TERRA_CPU_PROBE_B0A_ROOT"
_CONFIRMATION_ENV = "_TERRA_CPU_PROBE_CONFIRMATION"
_WORKER_RESULT_ENV = "_TERRA_CPU_PROBE_WORKER_RESULT"
_WORKER_READY_ENV = "_TERRA_CPU_PROBE_WORKER_READY"
_START_BARRIER_ENV = "_TERRA_CPU_PROBE_START_BARRIER"
_CALL_RECEIPT_PREFIX_ENV = "_TERRA_CPU_PROBE_CALL_RECEIPT_PREFIX"
_LAUNCHED_PERF_NS_ENV = "_TERRA_CPU_PROBE_LAUNCHED_PERF_NS"
_EXPECTED_CODE_BUNDLE_ENV = "_TERRA_CPU_PROBE_CODE_BUNDLE"
_INTERNAL_ENV_KEYS = (
    _WORKER_INDEX_ENV,
    _B0A_ROOT_ENV,
    _CONFIRMATION_ENV,
    _WORKER_RESULT_ENV,
    _WORKER_READY_ENV,
    _START_BARRIER_ENV,
    _CALL_RECEIPT_PREFIX_ENV,
    _LAUNCHED_PERF_NS_ENV,
    _EXPECTED_CODE_BUNDLE_ENV,
)
_DISALLOWED_RUNTIME_ENV = (
    "PYTHONPATH",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)
EXPECTED_WORKER_MODULE_PATHS = {
    "terra.benchmark_direct_service": "terra/benchmark_direct_service.py",
    "tools.confirm_b0a_direct_service_cost": (
        "tools/confirm_b0a_direct_service_cost.py"
    ),
    "tools.profile_b0a_direct_service_cost": (
        "tools/profile_b0a_direct_service_cost.py"
    ),
}


class ProbeRejected(RuntimeError):
    """The single preregistered process treatment did not pass."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _typed_canonical(value: Any) -> Any:
    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Typed canonical values must be finite.")
        return {"type": "float", "value_hex": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, list):
        return {"type": "list", "items": [_typed_canonical(item) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Typed canonical dictionary keys must be strings.")
        return {
            "type": "dict",
            "items": [[key, _typed_canonical(value[key])] for key in sorted(value)],
        }
    raise TypeError(f"Unsupported typed canonical value: {type(value).__name__}.")


def _typed_canonical_sha256(value: Any) -> str:
    return _canonical_json_sha256(_typed_canonical(value))


def _write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
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


def _coordinator_import_boundary() -> dict[str, Any]:
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "terra"
        or name.startswith("terra.")
        or name == "tools.profile_b0a_direct_service_cost"
        or name == "tools.confirm_b0a_direct_service_cost"
    )
    if forbidden:
        raise RuntimeError(
            "Coordinator imported worker-only scientific modules: " f"{forbidden[:10]}."
        )
    return {
        "jax_terra_profile_confirmation_modules_loaded": [],
        "passes": True,
    }


class _MemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._maximum = {"VmRSS": 0, "VmHWM": 0, "VmSwap": 0, "ru_maxrss_kib": 0}
        self._sample_count = 0
        self._error: Exception | None = None

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._sample()
        if self._error is not None:
            raise RuntimeError("Worker memory monitor failed.") from self._error
        return {
            "sample_interval_seconds": 0.5,
            "sample_count": self._sample_count,
            "maximum_kib": self._maximum,
            "transient_swap_observed": self._maximum["VmSwap"] > 0,
        }

    def _sample(self) -> None:
        memory = _process_memory()
        self._sample_count += 1
        for key, value in memory.items():
            self._maximum[key] = max(self._maximum[key], value)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._sample()
            except Exception as error:
                self._error = error
                return


def _git_receipt(repository: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "head": head,
        "dirty": bool(porcelain),
        "porcelain_v1": porcelain,
        "code_file_sha256_is_authoritative_for_worktree_state": True,
    }


def _code_receipt(repository: Path) -> dict[str, Any]:
    paths = (*SHARED_EXECUTION_CODE_PATHS, *CURRENT_CODE_PATHS)
    hashes = {relative: _sha256_file(repository / relative) for relative in paths}
    return {
        "git": _git_receipt(repository),
        "code_file_sha256": hashes,
        "code_bundle_sha256": _canonical_json_sha256(hashes),
    }


def _load_reference(path: Path) -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    if digest != EXPECTED_CPU_CONFIRMATION_SHA256:
        raise RuntimeError(
            "CPU confirmation receipt SHA-256 changed: "
            f"{EXPECTED_CPU_CONFIRMATION_SHA256} != {digest}."
        )
    reference = json.loads(path.read_text())
    outcome = reference.get("measurement", {}).get("exact_outcome")
    checks = {
        "schema": reference.get("schema") == EXPECTED_CPU_CONFIRMATION_SCHEMA,
        "release": reference.get("release_id") == EXPECTED_RELEASE_ID,
        "scope": (
            reference.get("result_scope")
            == "one_complete_exact_scenario_cost_confirmation"
        ),
        "non-admission": reference.get("bank_admission_result_emitted") is False,
        "complete outcome": (
            reference.get("single_scenario_exact_outcome_emitted") is True
        ),
        "one exact call": (
            reference.get("validator", {}).get("exact_entrypoint_call_count") == 1
        ),
        "complete typed outcome": (
            isinstance(outcome, dict)
            and set(outcome) == DIRECT_SERVICE_OUTCOME_KEYS
            and len(outcome) == 23
        ),
    }
    failed = [name for name, passes in checks.items() if not passes]
    if failed:
        raise RuntimeError(
            "Pinned CPU confirmation violates the reference contract: "
            + ", ".join(failed)
            + "."
        )
    return reference, digest


def _validate_code_reference(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    reference_hashes = reference["validator"]["code_before"]["code_file_sha256"]
    current_hashes = current["code_file_sha256"]
    missing = [
        path
        for path in SHARED_EXECUTION_CODE_PATHS
        if path not in reference_hashes or path not in current_hashes
    ]
    changed = [
        path
        for path in SHARED_EXECUTION_CODE_PATHS
        if path in reference_hashes
        and path in current_hashes
        and reference_hashes[path] != current_hashes[path]
    ]
    if missing or changed:
        raise RuntimeError(
            "Confirmed exact CPU dependencies changed: "
            f"missing={missing}, changed={changed}."
        )
    return {
        "compared_file_sha256": {
            path: current_hashes[path] for path in SHARED_EXECUTION_CODE_PATHS
        },
        "all_shared_execution_dependency_hashes_match_confirmation": True,
        "benchmark_protocol_sha256": current_hashes["terra/benchmark_protocol.py"],
        "current_profile_helper_sha256": current_hashes[
            "tools/profile_b0a_direct_service_cost.py"
        ],
        "profile_helper_note": (
            "The confirmation predates benchmark_protocol.py extraction. "
            "Current input, protocol, and state are rebuilt and compared exactly."
        ),
    }


def _require_clean_code(code: dict[str, Any]) -> None:
    git = code["git"]
    if git["dirty"] or git["porcelain_v1"]:
        raise RuntimeError(
            "The CPU process probe must run from a clean committed worktree."
        )


def _require_runtime_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    present = {
        key: environment[key]
        for key in ("JAX_PLATFORMS", *_DISALLOWED_RUNTIME_ENV)
        if key in environment
    }
    if present:
        raise RuntimeError(
            "CPU process probe requires an unmodified import/JAX/XLA environment; "
            f"unset {sorted(present)}."
        )
    return {
        "parent_required_absent": [
            "JAX_PLATFORMS",
            *_DISALLOWED_RUNTIME_ENV,
        ],
        "observed": {},
        "child_sets_only": {"JAX_PLATFORMS": "cpu"},
        "persistent_compilation_cache": False,
        "new_xla_or_thread_flags": False,
        "passes": True,
    }


def _module_origin_receipt(
    repository: Path,
    modules: Mapping[str, Any],
) -> dict[str, Any]:
    if set(modules) != set(EXPECTED_WORKER_MODULE_PATHS):
        raise RuntimeError("Worker origin audit module set changed.")
    origins = {}
    for name, relative in EXPECTED_WORKER_MODULE_PATHS.items():
        expected = (repository / relative).resolve()
        module_file = getattr(modules[name], "__file__", None)
        if not isinstance(module_file, str):
            raise RuntimeError(f"{name} has no filesystem origin.")
        observed = Path(module_file).resolve()
        if observed != expected:
            raise RuntimeError(
                f"{name} resolved outside the executing repository: "
                f"{observed} != {expected}."
            )
        origins[name] = {
            "expected_path": str(expected),
            "observed_path": str(observed),
            "matches": True,
        }
    return {
        "executing_repository": str(repository.resolve()),
        "origins": origins,
        "all_origins_match_exact_paths": True,
        "pythonpath_override": os.environ.get("PYTHONPATH"),
    }


def _require_cpu_topology() -> dict[str, Any]:
    online = os.cpu_count()
    available = sorted(os.sched_getaffinity(0))
    expected = list(range(EXPECTED_CPU_COUNT))
    if online != EXPECTED_CPU_COUNT or available != expected:
        raise RuntimeError(
            "CPU process probe requires the pinned 32-CPU starship topology: "
            f"online={online}, affinity={available}."
        )
    flattened = [cpu for affinity in AFFINITY_SETS for cpu in affinity]
    if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
        raise RuntimeError("Frozen worker affinities no longer partition CPUs 0-31.")
    if socket.gethostname() != EXPECTED_HOSTNAME:
        raise RuntimeError(
            f"CPU process probe requires {EXPECTED_HOSTNAME}, got "
            f"{socket.gethostname()}."
        )
    return {
        "hostname": socket.gethostname(),
        "online_logical_cpu_count": online,
        "coordinator_available_cpus": available,
        "worker_affinity_sets": [list(value) for value in AFFINITY_SETS],
        "partition_is_exact_and_disjoint": True,
    }


def _host_memory_capacity_kib() -> int:
    capacity = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 1024
    if capacity <= 0:
        raise RuntimeError("Host physical-memory capacity must be positive.")
    return int(capacity)


def _process_memory() -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, separator, remainder = line.partition(":")
        if separator and name in {"VmRSS", "VmHWM", "VmSwap"}:
            fields[name] = int(remainder.strip().split()[0])
    if set(fields) != {"VmRSS", "VmHWM", "VmSwap"}:
        raise RuntimeError(f"Incomplete /proc/self/status memory receipt: {fields}.")
    fields["ru_maxrss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return fields


def _parse_cpu_list(value: str) -> list[int]:
    cpus = []
    for part in value.split(","):
        bounds = part.split("-", maxsplit=1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        if stop < start:
            raise RuntimeError(f"Invalid CPU-list range: {part!r}.")
        cpus.extend(range(start, stop + 1))
    if len(cpus) != len(set(cpus)):
        raise RuntimeError(f"CPU list contains duplicates: {value!r}.")
    return sorted(cpus)


def _thread_affinity_receipt(expected_cpus: list[int]) -> dict[str, Any]:
    tasks = {}
    for status_path in sorted(
        Path("/proc/self/task").glob("*/status"),
        key=lambda path: int(path.parent.name),
    ):
        allowed = None
        for line in status_path.read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                allowed = line.partition(":")[2].strip()
                break
        if allowed is None:
            raise RuntimeError(f"{status_path} omitted Cpus_allowed_list.")
        parsed = _parse_cpu_list(allowed)
        if parsed != expected_cpus:
            raise RuntimeError(
                f"Thread {status_path.parent.name} affinity changed: {parsed}."
            )
        tasks[status_path.parent.name] = {
            "cpus_allowed_list": allowed,
            "parsed_cpus": parsed,
        }
    if not tasks:
        raise RuntimeError("Worker has no visible Linux task records.")
    return {
        "expected_cpus": expected_cpus,
        "task_count": len(tasks),
        "tasks": tasks,
        "all_worker_threads_within_fixed_cpuset": True,
    }


def _cgroup_memory_events() -> dict[str, int]:
    cgroup_path = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", maxsplit=2)
        if hierarchy == "0" and controllers == "":
            cgroup_path = relative
            break
    if cgroup_path is None:
        raise RuntimeError("A cgroup-v2 memory receipt is required.")
    path = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "memory.events"
    values = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        values[key] = int(value)
    for required in ("oom", "oom_kill"):
        if required not in values:
            raise RuntimeError(f"{path} omitted {required}.")
    return values


def _projection(cold_seconds: list[float], warm_seconds: list[float]) -> dict[str, Any]:
    if len(cold_seconds) != WORKER_COUNT or len(warm_seconds) != WORKER_COUNT:
        raise ValueError("Projection requires one cold and warm time per worker.")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for value in (*cold_seconds, *warm_seconds)
    ):
        raise ValueError("Projection times must be finite and positive.")
    scenarios = {}
    for count, limit in PROJECTION_LIMITS_SECONDS.items():
        per_worker_scenarios = math.ceil(count / WORKER_COUNT)
        per_worker = [
            cold + (per_worker_scenarios - 1) * warm
            for cold, warm in zip(cold_seconds, warm_seconds, strict=True)
        ]
        projected = max(per_worker)
        scenarios[str(count)] = {
            "per_worker_scenario_count": per_worker_scenarios,
            "per_worker_seconds": per_worker,
            "projected_makespan_seconds": projected,
            "limit_seconds": limit,
            "passes": projected <= limit,
        }
    return {
        "method": "max_i(C_i + (ceil(N/4)-1)*W_i)",
        "cold_launch_through_first_result_seconds": cold_seconds,
        "warm_rematerialization_through_second_result_seconds": warm_seconds,
        "scenario_counts": scenarios,
        "all_projection_gates_pass": all(
            value["passes"] for value in scenarios.values()
        ),
    }


def _run_two_exact_calls(
    *,
    launch_perf_ns: int,
    materialize: Callable[[], tuple[Any, dict[str, Any]]],
    compute: Callable[[Any], dict[str, Any]],
    synchronize: Callable[[Any], Any],
    stable_initial_state: Callable[[dict[str, Any]], dict[str, Any]],
    jsonable: Callable[[Any], Any],
    persist_call: Callable[[dict[str, Any]], None],
    reference_initial_state: dict[str, Any],
    reference_outcome: dict[str, Any],
    outcome_keys: set[str],
    counter_keys: tuple[str, ...] = POPULATION_COUNTER_KEYS,
) -> list[dict[str, Any]]:
    calls = []
    for call_index in range(CALLS_PER_WORKER):
        call_started = time.perf_counter()
        state, initial_state = materialize()
        stable_state = stable_initial_state(initial_state)
        if _typed_canonical(stable_state) != _typed_canonical(reference_initial_state):
            raise RuntimeError(
                f"Call {call_index} initial state differs from confirmation."
            )
        exact_started = time.perf_counter()
        exact_outcome = compute(state)
        synchronize(exact_outcome)
        finished_ns = time.perf_counter_ns()
        finished = finished_ns / 1e9
        exact_outcome = jsonable(exact_outcome)
        if not isinstance(exact_outcome, dict) or set(exact_outcome) != outcome_keys:
            raise RuntimeError(f"Call {call_index} outcome schema changed.")
        if _typed_canonical(exact_outcome) != _typed_canonical(reference_outcome):
            raise RuntimeError(f"Call {call_index} exact outcome differs.")
        counters = {key: exact_outcome[key] for key in counter_keys}
        reference_counters = {key: reference_outcome[key] for key in counter_keys}
        if counters != reference_counters:
            raise RuntimeError(f"Call {call_index} population counters differ.")
        call = {
            "call_index": call_index,
            "rematerialization_through_result_seconds": finished - call_started,
            "exact_entrypoint_seconds": finished - exact_started,
            "launch_through_result_seconds": (
                (finished_ns - launch_perf_ns) / 1e9 if call_index == 0 else None
            ),
            "initial_agent_state_sha256": initial_state["initial_agent_state_sha256"],
            "stable_initial_state_receipt_sha256": _typed_canonical_sha256(
                stable_state
            ),
            "exact_outcome": exact_outcome,
            "typed_exact_outcome_sha256": _typed_canonical_sha256(exact_outcome),
            "population_counters": counters,
            "memory_after": _process_memory(),
            "passes": True,
        }
        persist_call(call)
        calls.append(call)
    return calls


def _worker_contract(
    *,
    b0a_root: Path,
    reference: dict[str, Any],
    code_before: dict[str, Any],
) -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    import jax

    import terra.benchmark_direct_service as direct_service
    import tools.confirm_b0a_direct_service_cost as confirmation
    import tools.profile_b0a_direct_service_cost as probe_tool

    repository = Path(__file__).resolve().parents[1]
    module_origins = _module_origin_receipt(
        repository,
        {
            "terra.benchmark_direct_service": direct_service,
            "tools.confirm_b0a_direct_service_cost": confirmation,
            "tools.profile_b0a_direct_service_cost": probe_tool,
        },
    )
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "cpu":
        raise RuntimeError(f"Worker requires one CPU JAX device, got {devices}.")
    default_backend = jax.default_backend()
    if default_backend != "cpu":
        raise RuntimeError(
            f"Worker requires the CPU default backend, got {default_backend}."
        )
    if set(probe_tool.DIRECT_SERVICE_OUTCOME_KEYS) != set(
        reference["measurement"]["exact_outcome"]
    ):
        raise RuntimeError("Current exact-outcome schema differs from confirmation.")

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
    }
    mismatches = [
        name
        for name, (current, expected) in comparisons.items()
        if _typed_canonical(current) != _typed_canonical(expected)
    ]
    if mismatches:
        raise RuntimeError(
            "Worker no longer matches pinned confirmation: "
            + ", ".join(mismatches)
            + "."
        )

    def materialize() -> tuple[Any, dict[str, Any]]:
        return probe_tool._materialize_initial_state(
            selected,
            group,
            source_group_id,
            env_config,
        )

    input_receipt = {
        "b0a_root": str(b0a_root),
        "files_sha256_manifest_sha256": manifest_sha256,
        "source_grouping": source_grouping,
        "verified_selected_files": verified_selected_files,
        "selected_identity": selected_identity,
        "module_origins": module_origins,
        "selected_input_contract_sha256": _canonical_json_sha256(
            {
                "manifest": manifest_sha256,
                "source_grouping": source_grouping,
                "files": verified_selected_files,
                "identity": selected_identity,
            }
        ),
    }
    protocol_receipt = {
        "protocol": protocol,
        "protocol_receipt_sha256": _canonical_json_sha256(protocol),
        "benchmark_protocol_file_sha256": code_before["code_file_sha256"][
            "terra/benchmark_protocol.py"
        ],
    }
    device_receipt = {
        "device_count": len(devices),
        "selected_device": str(devices[0]),
        "platform": devices[0].platform,
        "device_kind": getattr(devices[0], "device_kind", None),
        "default_backend": default_backend,
        "python_executable": sys.executable,
        "packages": {
            name: _package_version(name) for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }
    return (
        jax,
        direct_service,
        confirmation,
        probe_tool,
        materialize,
        input_receipt,
        protocol_receipt,
        device_receipt,
    )


def _worker_main() -> int:
    result_path = Path(os.environ[_WORKER_RESULT_ENV])
    ready_path = Path(os.environ[_WORKER_READY_ENV])
    start_barrier_path = Path(os.environ[_START_BARRIER_ENV])
    call_receipt_prefix = Path(os.environ[_CALL_RECEIPT_PREFIX_ENV])
    worker_index = int(os.environ[_WORKER_INDEX_ENV])
    launch_perf_ns = int(os.environ[_LAUNCHED_PERF_NS_ENV])
    affinity = set(AFFINITY_SETS[worker_index])
    started_at = time.time()
    receipt: dict[str, Any] = {
        "schema": f"{SCHEMA}_worker_v1",
        "worker_index": worker_index,
        "pid": os.getpid(),
        "status": "failed",
        "started_unix_seconds": started_at,
    }
    monitor: _MemoryMonitor | None = None
    monitor_receipt: dict[str, Any] | None = None
    persisted_call_paths: list[Path] = []
    try:
        os.sched_setaffinity(0, affinity)
        affinity_before = sorted(os.sched_getaffinity(0))
        if affinity_before != sorted(affinity):
            raise RuntimeError(
                f"Worker {worker_index} affinity mismatch: {affinity_before}."
            )
        monitor = _MemoryMonitor()
        monitor.start()
        if os.environ.get("JAX_PLATFORMS") != "cpu":
            raise RuntimeError("Worker must set JAX_PLATFORMS=cpu before import.")
        unexpected = {
            key: os.environ[key] for key in _DISALLOWED_RUNTIME_ENV if key in os.environ
        }
        if unexpected:
            raise RuntimeError(f"Worker inherited forbidden flags: {unexpected}.")

        repository = Path(__file__).resolve().parents[1]
        code_before = _code_receipt(repository)
        if code_before["code_bundle_sha256"] != os.environ[_EXPECTED_CODE_BUNDLE_ENV]:
            raise RuntimeError("Worker code differs from coordinator preflight.")
        reference_path = Path(os.environ[_CONFIRMATION_ENV])
        reference, reference_sha256 = _load_reference(reference_path)
        b0a_root = Path(os.environ[_B0A_ROOT_ENV])
        (
            jax,
            direct_service,
            confirmation,
            probe_tool,
            materialize,
            input_receipt,
            protocol_receipt,
            device_receipt,
        ) = _worker_contract(
            b0a_root=b0a_root,
            reference=reference,
            code_before=code_before,
        )
        reference_initial_state = confirmation._stable_initial_state(
            reference["initial_state"]
        )
        memory_before = _process_memory()
        thread_affinity_before = _thread_affinity_receipt(affinity_before)

        _write_json_once(
            ready_path,
            {
                "schema": f"{SCHEMA}_worker_ready_v1",
                "worker_index": worker_index,
                "pid": os.getpid(),
                "affinity_cpus": affinity_before,
                "thread_affinity": thread_affinity_before,
                "module_origins": input_receipt["module_origins"],
                "device": device_receipt,
                "ready_unix_seconds": time.time(),
                "code_bundle_sha256": code_before["code_bundle_sha256"],
            },
        )
        while not start_barrier_path.is_file():
            time.sleep(0.05)

        def persist_call(call: dict[str, Any]) -> None:
            path = Path(f"{call_receipt_prefix}_{call['call_index']}.json")
            _write_json_once(
                path,
                {
                    "schema": f"{SCHEMA}_call_v1",
                    "worker_index": worker_index,
                    "pid": os.getpid(),
                    "affinity_cpus": sorted(os.sched_getaffinity(0)),
                    "call": call,
                },
            )
            persisted_call_paths.append(path)

        calls = _run_two_exact_calls(
            launch_perf_ns=launch_perf_ns,
            materialize=materialize,
            compute=direct_service.compute_initial_direct_service,
            synchronize=probe_tool._synchronize,
            stable_initial_state=confirmation._stable_initial_state,
            jsonable=probe_tool._jsonable,
            persist_call=persist_call,
            reference_initial_state=reference_initial_state,
            reference_outcome=reference["measurement"]["exact_outcome"],
            outcome_keys=set(probe_tool.DIRECT_SERVICE_OUTCOME_KEYS),
        )
        affinity_after = sorted(os.sched_getaffinity(0))
        if affinity_after != affinity_before:
            raise RuntimeError(
                f"Worker {worker_index} affinity changed after both calls."
            )
        thread_affinity_after = _thread_affinity_receipt(affinity_after)
        code_after = _code_receipt(repository)
        if code_after != code_before:
            raise RuntimeError("Worker code changed during execution.")
        final_memory = _process_memory()
        monitor_receipt = monitor.stop()
        monitor = None
        if monitor_receipt["transient_swap_observed"] or any(
            call["memory_after"]["VmSwap"] != 0 for call in calls
        ):
            raise RuntimeError(f"Worker {worker_index} used swap.")
        if len(calls) != CALLS_PER_WORKER:
            raise RuntimeError(f"Worker {worker_index} did not complete two calls.")

        receipt.update(
            {
                "status": "passed",
                "finished_unix_seconds": time.time(),
                "affinity_before_calls": affinity_before,
                "affinity_after_calls": affinity_after,
                "affinity_cpus": affinity_after,
                "thread_affinity_before_calls": thread_affinity_before,
                "thread_affinity_after_calls": thread_affinity_after,
                "ready_receipt": {
                    "path": str(ready_path),
                    "sha256": _sha256_file(ready_path),
                },
                "start_barrier": {
                    "path": str(start_barrier_path),
                    "sha256": _sha256_file(start_barrier_path),
                },
                "jax_environment": {
                    "JAX_PLATFORMS": "cpu",
                    "persistent_compilation_cache": False,
                    "new_xla_or_thread_flags": False,
                },
                "device": device_receipt,
                "reference": {
                    "path": str(reference_path),
                    "sha256": reference_sha256,
                },
                "input": input_receipt,
                "protocol": protocol_receipt,
                "initial_state": {
                    "expected_initial_agent_state_sha256": reference["initial_state"][
                        "initial_agent_state_sha256"
                    ],
                    "stable_receipt_sha256": _typed_canonical_sha256(
                        reference_initial_state
                    ),
                },
                "validator": {
                    "code_before": code_before,
                    "code_after": code_after,
                    "pre_and_post_execution_receipts_identical": True,
                    "exact_entrypoint": (
                        "terra.benchmark_direct_service."
                        "compute_initial_direct_service"
                    ),
                    "exact_entrypoint_call_count": len(calls),
                },
                "calls": calls,
                "persisted_call_receipts": {
                    str(path): _sha256_file(path) for path in persisted_call_paths
                },
                "memory_before": memory_before,
                "memory_after": final_memory,
                "memory_monitor": monitor_receipt,
                "process_peak_rss_kib": max(
                    final_memory["ru_maxrss_kib"],
                    monitor_receipt["maximum_kib"]["ru_maxrss_kib"],
                ),
                "process_cpu_seconds": time.process_time(),
                "all_outcomes_and_counters_match_confirmation": True,
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
                "memory_at_failure": _process_memory(),
                "memory_monitor": monitor_receipt,
                "persisted_call_receipts": {
                    str(path): _sha256_file(path) for path in persisted_call_paths
                },
            }
        )
    _write_json_once(result_path, receipt)
    return 0 if receipt["status"] == "passed" else 1


def _load_worker_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("schema") != f"{SCHEMA}_worker_v1":
        raise RuntimeError(f"Unexpected worker result schema in {path}.")
    return result


def _load_worker_result_ready(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    if result.get("schema") != f"{SCHEMA}_worker_ready_v1":
        raise RuntimeError(f"Unexpected worker ready schema in {path}.")
    return result


def _validate_worker_results(
    workers: list[dict[str, Any]],
    *,
    reference_outcome: dict[str, Any],
    reference_sha256: str,
    expected_code_bundle_sha256: str,
    expected_pids: list[int],
    host_memory_capacity_kib: int,
    coordinator_peak_rss_kib: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(workers) != WORKER_COUNT:
        raise RuntimeError(f"Expected {WORKER_COUNT} worker results.")
    workers = sorted(workers, key=lambda value: value["worker_index"])
    if [value["worker_index"] for value in workers] != list(range(WORKER_COUNT)):
        raise RuntimeError("Worker indices are missing or duplicated.")
    if any(value.get("status") != "passed" for value in workers):
        raise RuntimeError("At least one worker did not pass.")
    pids = [value["pid"] for value in workers]
    if len(set(pids)) != WORKER_COUNT:
        raise RuntimeError("Workers did not use four distinct PIDs.")
    if pids != expected_pids:
        raise RuntimeError(f"Worker result PIDs differ from launches: {pids}.")

    cold_seconds = []
    warm_seconds = []
    input_contract_hashes = set()
    protocol_hashes = set()
    initial_state_hashes = set()
    for index, worker in enumerate(workers):
        if worker.get("affinity_cpus") != list(AFFINITY_SETS[index]):
            raise RuntimeError(f"Worker {index} affinity changed.")
        if worker.get("device", {}).get("platform") != "cpu":
            raise RuntimeError(f"Worker {index} did not use CPU JAX.")
        if worker.get("device", {}).get("default_backend") != "cpu":
            raise RuntimeError(f"Worker {index} default backend was not CPU.")
        if worker.get("device", {}).get("device_count") != 1:
            raise RuntimeError(f"Worker {index} exposed multiple JAX devices.")
        if worker.get("affinity_before_calls") != list(AFFINITY_SETS[index]):
            raise RuntimeError(f"Worker {index} pre-call affinity changed.")
        if worker.get("affinity_after_calls") != list(AFFINITY_SETS[index]):
            raise RuntimeError(f"Worker {index} post-call affinity changed.")
        for phase in ("before", "after"):
            thread_affinity = worker.get(
                f"thread_affinity_{phase}_calls",
                {},
            )
            if (
                thread_affinity.get("expected_cpus") != list(AFFINITY_SETS[index])
                or thread_affinity.get("all_worker_threads_within_fixed_cpuset")
                is not True
                or not thread_affinity.get("tasks")
            ):
                raise RuntimeError(
                    f"Worker {index} {phase}-call thread affinity changed."
                )
        if worker.get("memory_monitor", {}).get("transient_swap_observed") is not False:
            raise RuntimeError(f"Worker {index} transiently used swap.")
        if worker.get("reference", {}).get("sha256") != reference_sha256:
            raise RuntimeError(f"Worker {index} reference receipt changed.")
        module_origins = worker.get("input", {}).get("module_origins", {})
        origin_rows = module_origins.get("origins", {})
        if (
            module_origins.get("all_origins_match_exact_paths") is not True
            or module_origins.get("pythonpath_override") is not None
            or set(origin_rows) != set(EXPECTED_WORKER_MODULE_PATHS)
            or any(
                row.get("matches") is not True
                or row.get("observed_path") != row.get("expected_path")
                for row in origin_rows.values()
            )
        ):
            raise RuntimeError(f"Worker {index} module origins changed.")
        validator = worker.get("validator", {})
        if (
            validator.get("code_before", {}).get("code_bundle_sha256")
            != expected_code_bundle_sha256
            or validator.get("code_after") != validator.get("code_before")
            or validator.get("exact_entrypoint_call_count") != CALLS_PER_WORKER
        ):
            raise RuntimeError(f"Worker {index} code or call receipt changed.")
        if len(worker.get("persisted_call_receipts", {})) != CALLS_PER_WORKER:
            raise RuntimeError(f"Worker {index} call receipts are incomplete.")
        input_contract_hashes.add(
            worker.get("input", {}).get("selected_input_contract_sha256")
        )
        protocol_hashes.add(worker.get("protocol", {}).get("protocol_receipt_sha256"))
        initial_state_hashes.add(
            worker.get("initial_state", {}).get("stable_receipt_sha256")
        )
        calls = worker.get("calls", [])
        if len(calls) != CALLS_PER_WORKER:
            raise RuntimeError(f"Worker {index} did not make exactly two calls.")
        for call in calls:
            if _typed_canonical(call.get("exact_outcome")) != _typed_canonical(
                reference_outcome
            ):
                raise RuntimeError(f"Worker {index} exact outcome differs.")
            if (
                call.get("rematerialization_through_result_seconds", math.inf)
                > MAX_CALL_SECONDS
            ):
                raise RuntimeError(f"Worker {index} exceeded the one-hour call gate.")
            if call.get("memory_after", {}).get("VmSwap") != 0:
                raise RuntimeError(f"Worker {index} used swap.")
        cold_seconds.append(calls[0]["launch_through_result_seconds"])
        warm_seconds.append(calls[1]["rematerialization_through_result_seconds"])
        if cold_seconds[-1] > MAX_CALL_SECONDS:
            raise RuntimeError(
                f"Worker {index} cold launch exceeded the one-hour gate."
            )
    if (
        len(input_contract_hashes) != 1
        or None in input_contract_hashes
        or len(protocol_hashes) != 1
        or None in protocol_hashes
        or len(initial_state_hashes) != 1
        or None in initial_state_hashes
    ):
        raise RuntimeError("Worker input, protocol, or state receipts differ.")

    projection = _projection(cold_seconds, warm_seconds)
    summed_worker_peak = sum(worker["process_peak_rss_kib"] for worker in workers)
    conservative_peak = coordinator_peak_rss_kib + summed_worker_peak
    memory_fraction = conservative_peak / host_memory_capacity_kib
    memory = {
        "coordinator_peak_rss_kib": coordinator_peak_rss_kib,
        "summed_worker_peak_rss_kib": summed_worker_peak,
        "conservative_aggregate_peak_rss_kib": conservative_peak,
        "host_physical_memory_kib": host_memory_capacity_kib,
        "conservative_peak_fraction_of_host": memory_fraction,
        "maximum_allowed_fraction": MAX_MEMORY_FRACTION,
        "passes": memory_fraction <= MAX_MEMORY_FRACTION,
    }
    gates = {
        "four_distinct_successful_worker_pids": len(set(pids)) == WORKER_COUNT,
        "all_eight_outcomes_and_counters_match_confirmation": True,
        "all_workers_cpu_only": True,
        "all_affinities_match": True,
        "all_calls_at_most_one_hour": True,
        "zero_worker_swap": True,
        "conservative_aggregate_memory_at_most_eighty_percent": memory["passes"],
        "projection_256_at_most_24h": projection["scenario_counts"]["256"]["passes"],
        "projection_448_at_most_48h": projection["scenario_counts"]["448"]["passes"],
    }
    gates["all_process_treatment_gates_pass"] = all(gates.values())
    return projection, {"measurement": memory, "gates": gates}


def _terminate_workers(processes: list[subprocess.Popen[Any]]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def _worker_command() -> list[str]:
    return [sys.executable, "-m", WORKER_MODULE]


def _run_cohort(
    *,
    b0a_root: Path,
    confirmation_path: Path,
    output_directory: Path,
    code_bundle_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    processes: list[subprocess.Popen[Any]] = []
    log_streams = []
    worker_paths: list[Path] = []
    ready_paths: list[Path] = []
    launches = []
    return_codes: list[int] = []
    coordinator_error: dict[str, str] | None = None
    barrier_path = output_directory / "start_barrier.json"
    cohort_started = time.perf_counter()
    deadline = cohort_started + COHORT_HARD_TIMEOUT_SECONDS
    try:
        if barrier_path.exists():
            raise FileExistsError(barrier_path)
        for worker_index in range(WORKER_COUNT):
            result_path = output_directory / f"worker_{worker_index}.json"
            ready_path = output_directory / f"worker_{worker_index}.ready.json"
            call_prefix = output_directory / f"worker_{worker_index}_call"
            log_path = output_directory / f"worker_{worker_index}.log"
            occupied = [
                path
                for path in (
                    result_path,
                    ready_path,
                    Path(f"{call_prefix}_0.json"),
                    Path(f"{call_prefix}_1.json"),
                    log_path,
                )
                if path.exists()
            ]
            if occupied:
                raise FileExistsError(occupied[0])
            log_stream = log_path.open("x")
            launch_perf_ns = time.perf_counter_ns()
            environment = dict(os.environ)
            environment.update(
                {
                    _WORKER_INDEX_ENV: str(worker_index),
                    _B0A_ROOT_ENV: str(b0a_root),
                    _CONFIRMATION_ENV: str(confirmation_path),
                    _WORKER_RESULT_ENV: str(result_path),
                    _WORKER_READY_ENV: str(ready_path),
                    _START_BARRIER_ENV: str(barrier_path),
                    _CALL_RECEIPT_PREFIX_ENV: str(call_prefix),
                    _LAUNCHED_PERF_NS_ENV: str(launch_perf_ns),
                    _EXPECTED_CODE_BUNDLE_ENV: code_bundle_sha256,
                    "JAX_PLATFORMS": "cpu",
                }
            )
            process = subprocess.Popen(
                _worker_command(),
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            log_streams.append(log_stream)
            worker_paths.append(result_path)
            ready_paths.append(ready_path)
            launches.append(
                {
                    "worker_index": worker_index,
                    "pid": process.pid,
                    "affinity_cpus": list(AFFINITY_SETS[worker_index]),
                    "launch_perf_ns": launch_perf_ns,
                    "result_path": str(result_path),
                    "ready_path": str(ready_path),
                    "call_receipt_prefix": str(call_prefix),
                    "log_path": str(log_path),
                    "process_group_id": process.pid,
                }
            )

        while not all(path.is_file() for path in ready_paths):
            exited = [
                process.returncode
                for process in processes
                if process.poll() is not None
            ]
            if exited:
                raise ProbeRejected(
                    f"Worker exited before the cold-cohort barrier: {exited}."
                )
            if time.perf_counter() > deadline:
                raise ProbeRejected("Cold-cohort ready barrier exceeded three hours.")
            time.sleep(0.1)
        ready_receipts = [_load_worker_result_ready(path) for path in ready_paths]
        for index, ready in enumerate(ready_receipts):
            if (
                ready["worker_index"] != index
                or ready["pid"] != processes[index].pid
                or ready["affinity_cpus"] != list(AFFINITY_SETS[index])
                or ready["device"].get("platform") != "cpu"
                or ready["device"].get("default_backend") != "cpu"
                or ready.get("thread_affinity", {}).get(
                    "all_worker_threads_within_fixed_cpuset"
                )
                is not True
                or ready.get("module_origins", {}).get("all_origins_match_exact_paths")
                is not True
            ):
                raise ProbeRejected(f"Worker {index} ready receipt changed.")
        _write_json_once(
            barrier_path,
            {
                "schema": f"{SCHEMA}_start_barrier_v1",
                "released_unix_seconds": time.time(),
                "worker_ready_sha256": {
                    str(path): _sha256_file(path) for path in ready_paths
                },
                "all_four_workers_ready_before_release": True,
            },
        )

        while any(process.poll() is None for process in processes):
            failed = [
                process.returncode
                for process in processes
                if process.poll() not in (None, 0)
            ]
            if failed:
                raise ProbeRejected(f"Worker failed after barrier release: {failed}.")
            if time.perf_counter() > deadline:
                raise ProbeRejected("Fixed CPU process cohort exceeded three hours.")
            time.sleep(0.1)
    except Exception as error:
        coordinator_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        _terminate_workers(processes)
        return_codes = [
            process.wait() if process.poll() is None else int(process.returncode)
            for process in processes
        ]
        for stream in log_streams:
            stream.close()

    workers = []
    worker_load_errors = {}
    for path in worker_paths:
        if not path.is_file():
            continue
        try:
            workers.append(_load_worker_result(path))
        except Exception as error:
            worker_load_errors[str(path)] = {
                "type": type(error).__name__,
                "message": str(error),
            }
    call_paths = sorted(output_directory.glob("worker_*_call_[01].json"))
    execution = {
        "cohort_wall_seconds": time.perf_counter() - cohort_started,
        "hard_timeout_seconds": COHORT_HARD_TIMEOUT_SECONDS,
        "direct_fresh_interpreter_execs": True,
        "worker_process_groups_are_distinct": True,
        "launches": launches,
        "return_codes": return_codes,
        "worker_result_count": len(workers),
        "worker_load_errors": worker_load_errors,
        "coordinator_error": coordinator_error,
        "start_barrier_path": str(barrier_path),
        "start_barrier_sha256": (
            _sha256_file(barrier_path) if barrier_path.is_file() else None
        ),
        "ready_receipt_sha256": {
            str(path): _sha256_file(path) for path in ready_paths if path.is_file()
        },
        "persisted_call_receipt_sha256": {
            str(path): _sha256_file(path) for path in call_paths
        },
        "worker_result_sha256": {
            str(path): _sha256_file(path) for path in worker_paths if path.is_file()
        },
        "worker_log_sha256": {
            str(output_directory / f"worker_{index}.log"): _sha256_file(
                output_directory / f"worker_{index}.log"
            )
            for index in range(len(log_streams))
        },
    }
    return workers, execution


def run_probe(
    b0a_root: Path,
    confirmation_path: Path,
    output_directory: Path,
) -> Path:
    b0a_root = b0a_root.resolve()
    confirmation_path = confirmation_path.resolve()
    output_directory = output_directory.resolve()
    output_path = output_directory / OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(output_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    repository = Path(__file__).resolve().parents[1]
    coordinator_started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "release_id": EXPECTED_RELEASE_ID,
        "result_scope": "non_admission_fixed_four_process_cpu_scaling_gate",
        "status": "failed",
        "admission_result_emitted": False,
        "bank_admission_result_emitted": False,
        "static_admission_authorized": False,
        "ppo_authorized": False,
        "bank_profile_called": False,
        "experiment": {
            "treatment": "fixed_four_long_lived_cpu_workers_two_calls_each",
            "unique_scenario_count": 1,
            "worker_count": WORKER_COUNT,
            "calls_per_worker": CALLS_PER_WORKER,
            "worker_count_or_wave_count_configurable": False,
            "retry_or_width_sweep": False,
            "state_rematerialized_before_every_call": True,
            "affinity_sets": [list(value) for value in AFFINITY_SETS],
            "one_cold_ready_barrier": True,
            "cohort_hard_timeout_seconds": COHORT_HARD_TIMEOUT_SECONDS,
        },
        "command": [str(argument) for argument in sys.argv],
    }
    processes_finished = False
    try:
        import_boundary_before = _coordinator_import_boundary()
        environment = _require_runtime_environment(os.environ)
        topology = _require_cpu_topology()
        reference, reference_sha256 = _load_reference(confirmation_path)
        reference_root = Path(reference["input"]["b0a_root"]).resolve()
        if b0a_root != reference_root:
            raise RuntimeError(
                f"Probe requires B0a root {reference_root}, got {b0a_root}."
            )
        if reference.get("machine", {}).get("hostname") != socket.gethostname():
            raise RuntimeError("Confirmation was produced on another host.")
        code_before = _code_receipt(repository)
        _require_clean_code(code_before)
        code_reference = _validate_code_reference(reference, code_before)
        cgroup_before = _cgroup_memory_events()

        workers, execution = _run_cohort(
            b0a_root=b0a_root,
            confirmation_path=confirmation_path,
            output_directory=output_directory,
            code_bundle_sha256=code_before["code_bundle_sha256"],
        )
        processes_finished = True
        import_boundary_after = _coordinator_import_boundary()
        code_after = _code_receipt(repository)
        cgroup_after = _cgroup_memory_events()
        oom_delta = {
            key: cgroup_after[key] - cgroup_before.get(key, 0)
            for key in ("oom", "oom_kill")
        }
        coordinator_memory = _process_memory()
        receipt.update(
            {
                "input": {
                    "b0a_root": str(b0a_root),
                    "cpu_confirmation_receipt": str(confirmation_path),
                    "cpu_confirmation_receipt_sha256": reference_sha256,
                    "files_sha256_manifest_sha256": reference["input"][
                        "files_sha256_manifest_sha256"
                    ],
                    "reference_typed_exact_outcome_sha256": (
                        _typed_canonical_sha256(
                            reference["measurement"]["exact_outcome"]
                        )
                    ),
                },
                "validator": {
                    "code_before": code_before,
                    "code_after": code_after,
                    "confirmation_code_reference": code_reference,
                    "coordinator_import_boundary_before": import_boundary_before,
                    "coordinator_import_boundary_after": import_boundary_after,
                    "exact_entrypoint": (
                        "terra.benchmark_direct_service."
                        "compute_initial_direct_service"
                    ),
                },
                "machine": {
                    "hostname": socket.gethostname(),
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "python_executable": sys.executable,
                    "topology": topology,
                    "environment": environment,
                    "cgroup_memory_events_before": cgroup_before,
                    "cgroup_memory_events_after": cgroup_after,
                    "cgroup_oom_delta": oom_delta,
                    "coordinator_memory": coordinator_memory,
                },
                "execution": execution,
                "workers": workers,
            }
        )
        if code_after != code_before:
            raise RuntimeError("Coordinator code changed during execution.")
        if execution["coordinator_error"] is not None:
            raise ProbeRejected(
                "Cohort orchestration failed: "
                f"{execution['coordinator_error']['message']}."
            )
        if execution["return_codes"] != [0] * WORKER_COUNT:
            raise ProbeRejected(
                f"Worker return codes were {execution['return_codes']}."
            )
        if (
            execution["worker_result_count"] != WORKER_COUNT
            or len(execution["ready_receipt_sha256"]) != WORKER_COUNT
            or len(execution["persisted_call_receipt_sha256"])
            != WORKER_COUNT * CALLS_PER_WORKER
            or execution["start_barrier_sha256"] is None
        ):
            raise ProbeRejected("Cohort evidence files are incomplete.")
        if oom_delta != {"oom": 0, "oom_kill": 0}:
            raise ProbeRejected(f"Cgroup OOM counters increased: {oom_delta}.")

        projection, resource_gates = _validate_worker_results(
            workers,
            reference_outcome=reference["measurement"]["exact_outcome"],
            reference_sha256=reference_sha256,
            expected_code_bundle_sha256=code_before["code_bundle_sha256"],
            expected_pids=[launch["pid"] for launch in execution["launches"]],
            host_memory_capacity_kib=_host_memory_capacity_kib(),
            coordinator_peak_rss_kib=coordinator_memory["ru_maxrss_kib"],
        )
        resource_gates["gates"]["zero_worker_oom"] = True
        resource_gates["gates"]["all_process_treatment_gates_pass"] = all(
            resource_gates["gates"].values()
        )
        if not resource_gates["gates"]["all_process_treatment_gates_pass"]:
            raise ProbeRejected("At least one fixed process-treatment gate failed.")

        receipt.update(
            {
                "status": "passed",
                "input": {
                    **receipt["input"],
                    "selected_input_contract_sha256": workers[0]["input"][
                        "selected_input_contract_sha256"
                    ],
                },
                "protocol": workers[0]["protocol"],
                "initial_state": workers[0]["initial_state"],
                "validator": {
                    **receipt["validator"],
                    "pre_and_post_execution_receipts_identical": True,
                    "exact_entrypoint_call_count": WORKER_COUNT * CALLS_PER_WORKER,
                },
                "projection": projection,
                "resources": resource_gates["measurement"],
                "gates": resource_gates["gates"],
                "decision": {
                    "fixed_four_process_cpu_treatment_passes": True,
                    "authorizes_one_deterministic_four_worker_256_profile": True,
                    "authorized_profile_must_finish_within_24h": True,
                    "authorized_profile_must_project_448_within_48h": True,
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
                },
            }
        )
    except Exception as error:
        receipt["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "processes_finished": processes_finished,
        }
        receipt["decision"] = {
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
    receipt["coordinator_wall_seconds"] = time.perf_counter() - coordinator_started
    _write_json_once(output_path, receipt)
    print(output_path)
    if receipt["status"] != "passed":
        raise ProbeRejected(f"Fixed CPU process treatment failed; see {output_path}.")
    return output_path


def _public_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered four-worker CPU scaling gate for the pinned "
            "B0a direct-service scenario."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--cpu-confirmation-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> None:
    if _WORKER_INDEX_ENV in os.environ:
        raise SystemExit(_worker_main())
    if any(key in os.environ for key in _INTERNAL_ENV_KEYS):
        raise RuntimeError("Incomplete internal worker environment.")
    args = _public_parser().parse_args()
    run_probe(
        args.b0a_root,
        args.cpu_confirmation_receipt,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
