#!/usr/bin/env python3
"""Check full-population parity for the bounded CPU-graph/GPU-service path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA = "terra_direct_service_hybrid_parity_v1"
OUTPUT_NAME = "direct_service_hybrid_parity.json"
CANDIDATE_OUTPUT_NAME = "ordered_service_candidates.npy"
EXPECTED_CPU_CONFIRMATION_SHA256 = (
    "f4bc393a7eabcdc058eb5f4de69281c5e1bed9feef275f9f75833f3f3c4aaae7"
)
EXPECTED_CPU_CONFIRMATION_SCHEMA = (
    "terra_direct_service_validation_cost_confirmation_v1"
)
EXPECTED_DEVICE_KIND = "NVIDIA GeForce RTX 4090"
SERVICE_BATCH_SIZE = 4
EXPECTED_OUTPUT_LEAF_COUNT = 3
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
)
CURRENT_ONLY_CODE_PATHS = (
    "terra/benchmark_protocol.py",
    "tools/profile_b0a_direct_service_cost.py",
    "tools/check_b0a_direct_service_hybrid_parity.py",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_jax_platforms_unset(environment: Mapping[str, str]) -> dict[str, Any]:
    if "JAX_PLATFORMS" in environment:
        raise RuntimeError(
            "JAX_PLATFORMS must be absent so one process can use canonical CPU "
            "graph traversal and the single GPU service kernel."
        )
    return {
        "policy": "environment_variable_must_be_absent",
        "observed_value": None,
        "passes": True,
    }


def _validate_devices(
    cpu_devices: Sequence[Any],
    gpu_devices: Sequence[Any],
    default_backend: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if len(cpu_devices) != 1:
        raise RuntimeError(
            f"Hybrid parity requires exactly one JAX CPU device, got "
            f"{len(cpu_devices)}."
        )
    if len(gpu_devices) != 1:
        raise RuntimeError(
            f"Hybrid parity requires exactly one JAX GPU device, got "
            f"{len(gpu_devices)}."
        )
    cpu = cpu_devices[0]
    gpu = gpu_devices[0]
    if getattr(cpu, "platform", None) != "cpu":
        raise RuntimeError("The canonical graph device is not a CPU.")
    if getattr(gpu, "platform", None) != "gpu":
        raise RuntimeError("The service device is not a GPU.")
    if getattr(gpu, "device_kind", None) != EXPECTED_DEVICE_KIND:
        raise RuntimeError(
            f"Hybrid parity requires {EXPECTED_DEVICE_KIND!r}, got "
            f"{getattr(gpu, 'device_kind', None)!r}."
        )
    if default_backend != "gpu":
        raise RuntimeError(
            f"Hybrid parity requires the GPU default backend, got "
            f"{default_backend!r}."
        )
    receipt = {
        "cpu": {
            "device_count": 1,
            "selected_device": str(cpu),
            "platform": getattr(cpu, "platform", None),
            "device_kind": getattr(cpu, "device_kind", None),
        },
        "gpu": {
            "device_count": 1,
            "selected_device": str(gpu),
            "platform": getattr(gpu, "platform", None),
            "device_kind": getattr(gpu, "device_kind", None),
        },
        "default_backend": default_backend,
        "passes": True,
    }
    return cpu, gpu, receipt


def _load_cpu_confirmation(
    path: Path,
    *,
    outcome_keys: set[str],
    selected_map_id: str,
) -> tuple[dict[str, Any], str]:
    digest = _sha256_file(path)
    if digest != EXPECTED_CPU_CONFIRMATION_SHA256:
        raise RuntimeError(
            "CPU confirmation receipt SHA-256 changed: "
            f"{EXPECTED_CPU_CONFIRMATION_SHA256} != {digest}."
        )
    receipt = json.loads(path.read_text())
    exact_outcome = receipt.get("measurement", {}).get("exact_outcome")
    checks = {
        "schema": receipt.get("schema") == EXPECTED_CPU_CONFIRMATION_SCHEMA,
        "release": receipt.get("release_id") == "terramap-bench-v1.0.0",
        "scope": (
            receipt.get("result_scope")
            == "one_complete_exact_scenario_cost_confirmation"
        ),
        "non-admission": receipt.get("bank_admission_result_emitted") is False,
        "complete outcome": receipt.get("single_scenario_exact_outcome_emitted")
        is True,
        "selected identity": (
            receipt.get("selected_identity", {}).get("map_id") == selected_map_id
        ),
        "outcome schema": (
            isinstance(exact_outcome, dict) and set(exact_outcome) == outcome_keys
        ),
        "one exact entrypoint call": (
            receipt.get("validator", {}).get("exact_entrypoint_call_count") == 1
        ),
    }
    failures = [name for name, passes in checks.items() if not passes]
    if failures:
        raise RuntimeError(
            "Pinned CPU confirmation violates the reference contract: "
            + ", ".join(failures)
            + "."
        )
    return receipt, digest


def _code_receipt(repository: Path, probe_tool: Any) -> dict[str, Any]:
    relative_paths = (*CPU_SHARED_EXECUTION_CODE_PATHS, *CURRENT_ONLY_CODE_PATHS)
    hashes = {
        relative: probe_tool._sha256_file(repository / relative)
        for relative in relative_paths
    }
    return {
        "git": probe_tool._git_receipt(repository),
        "code_file_sha256": hashes,
        "code_bundle_sha256": probe_tool._canonical_json_sha256(hashes),
    }


def _require_clean_worktree(code_receipt: dict[str, Any]) -> None:
    git_receipt = code_receipt["git"]
    if git_receipt.get("dirty") or git_receipt.get("porcelain_v1"):
        raise RuntimeError(
            "Hybrid parity must run from a clean committed Terra worktree."
        )


def _validate_execution_code_reference(
    reference: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    reference_hashes = reference["validator"]["code_before"]["code_file_sha256"]
    current_hashes = current["code_file_sha256"]
    missing = [
        path
        for path in CPU_SHARED_EXECUTION_CODE_PATHS
        if path not in reference_hashes or path not in current_hashes
    ]
    if missing:
        raise RuntimeError(
            f"CPU/hybrid code comparison is missing dependencies: {missing}."
        )
    changed = [
        path
        for path in CPU_SHARED_EXECUTION_CODE_PATHS
        if reference_hashes[path] != current_hashes[path]
    ]
    if changed:
        raise RuntimeError(
            "Canonical CPU graph, reducer, or service dependencies changed "
            f"since confirmation: {changed}."
        )
    protocol_path = "terra/benchmark_protocol.py"
    if protocol_path not in current_hashes:
        raise RuntimeError("The hybrid receipt omitted terra/benchmark_protocol.py.")
    return {
        "compared_file_sha256": {
            path: current_hashes[path] for path in CPU_SHARED_EXECUTION_CODE_PATHS
        },
        "all_shared_execution_dependency_hashes_match_cpu": True,
        "benchmark_protocol_sha256": current_hashes[protocol_path],
        "profile_helper_note": (
            "The CPU confirmation predates benchmark_protocol.py extraction. "
            "The current profile helper is hashed but compared through rebuilt "
            "input, protocol, and stable-state equality rather than stale bytes."
        ),
    }


def _validate_rebuilt_contract(
    *,
    reference: dict[str, Any],
    selected_identity: dict[str, Any],
    manifest_sha256: str,
    source_grouping: dict[str, Any],
    verified_selected_files: dict[str, str],
    protocol: dict[str, Any],
    initial_state: dict[str, Any],
    stable_initial_state: Callable[[dict[str, Any]], dict[str, Any]],
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
            "Hybrid parity no longer matches the pinned CPU contract: "
            + ", ".join(mismatches)
            + "."
        )


def _tree_device_platforms(jax: Any, tree: Any) -> set[str]:
    platforms: set[str] = set()
    for leaf in jax.tree_util.tree_leaves(tree):
        devices_method = getattr(leaf, "devices", None)
        if devices_method is None:
            continue
        platforms.update(device.platform for device in devices_method())
    return platforms


def _require_tree_on_platform(jax: Any, tree: Any, platform_name: str) -> None:
    platforms = _tree_device_platforms(jax, tree)
    if platforms != {platform_name}:
        raise RuntimeError(
            f"Expected arrays only on {platform_name!r}, got {sorted(platforms)}."
        )


def _require_int32_rows(value: Any, name: str) -> np.ndarray:
    rows = np.asarray(value)
    if rows.dtype != np.int32:
        raise RuntimeError(f"{name} must remain int32, got {rows.dtype}.")
    return rows


class _OrderedLeafHasher:
    """Stream ordered unpadded rows using the repository's array-hash contract."""

    def __init__(
        self,
        *,
        total_rows: int,
        leaf_count: int,
        required_dtype: np.dtype[Any],
    ) -> None:
        if total_rows <= 0:
            raise ValueError("Ordered output hashing requires positive row count.")
        self.total_rows = total_rows
        self.leaf_count = leaf_count
        self.required_dtype = np.dtype(required_dtype)
        self.processed_rows = 0
        self._digests: list[Any] | None = None
        self._structures: list[dict[str, Any]] | None = None

    def update(self, leaves: Sequence[np.ndarray], valid_count: int) -> None:
        if len(leaves) != self.leaf_count:
            raise RuntimeError(
                f"Expected {self.leaf_count} output leaves, got {len(leaves)}."
            )
        if valid_count <= 0:
            raise RuntimeError("Every service batch must contain a logical row.")
        values = tuple(np.asarray(leaf) for leaf in leaves)
        if any(value.ndim == 0 or value.shape[0] < valid_count for value in values):
            raise RuntimeError("An output leaf cannot supply its logical rows.")
        if any(value.dtype != self.required_dtype for value in values):
            raise RuntimeError(
                f"Every output leaf must be {self.required_dtype}, got "
                f"{[str(value.dtype) for value in values]}."
            )

        if self._digests is None:
            self._digests = []
            self._structures = []
            for index, value in enumerate(values):
                full_shape = (self.total_rows, *value.shape[1:])
                digest = hashlib.sha256()
                digest.update(str(value.dtype).encode())
                digest.update(np.asarray(full_shape, dtype=np.int64).tobytes())
                self._digests.append(digest)
                self._structures.append(
                    {
                        "leaf_index": index,
                        "dtype": str(value.dtype),
                        "shape": list(full_shape),
                    }
                )
        else:
            assert self._structures is not None
            for structure, value in zip(self._structures, values):
                if list(value.shape[1:]) != structure["shape"][1:]:
                    raise RuntimeError("Output leaf shape changed between batches.")

        assert self._digests is not None
        for digest, value in zip(self._digests, values):
            digest.update(np.ascontiguousarray(value[:valid_count]).tobytes())
        self.processed_rows += valid_count
        if self.processed_rows > self.total_rows:
            raise RuntimeError("Service execution exceeded the candidate population.")

    def finish(self) -> dict[str, Any]:
        if self.processed_rows != self.total_rows:
            raise RuntimeError(
                f"Hashed {self.processed_rows} rows, expected {self.total_rows}."
            )
        if self._digests is None or self._structures is None:
            raise RuntimeError("No ordered output rows were hashed.")
        leaf_hashes = [digest.hexdigest() for digest in self._digests]
        return {
            "logical_rows": self.total_rows,
            "leaf_count": self.leaf_count,
            "leaves": [
                {**structure, "sha256": digest}
                for structure, digest in zip(self._structures, leaf_hashes)
            ],
            "ordered_unpadded_output_sha256": _canonical_json_sha256(leaf_hashes),
            "hash_contract": (
                "each leaf prefixes dtype and full unpadded shape to ordered "
                "contiguous row bytes; the combined hash covers the ordered "
                "leaf-hash list"
            ),
        }


def _population_counters(
    *,
    poses: np.ndarray,
    movement_stats: dict[str, int],
    candidates: np.ndarray,
    accepted: np.ndarray,
    prefilter_stats: dict[str, int],
    cabin_headings: int,
) -> dict[str, int]:
    service_padded_rows = (
        (len(accepted) + SERVICE_BATCH_SIZE - 1) // SERVICE_BATCH_SIZE
    ) * SERVICE_BATCH_SIZE
    return {
        "admissible_pose_count_initial": int(len(poses)),
        "base_pose_cabin_heading_candidates_initial": int(len(candidates)),
        "movement_source_rows_logical": int(movement_stats["source_rows_logical"]),
        "movement_transition_attempts_logical": int(
            movement_stats["transition_attempts_logical"]
        ),
        "movement_source_rows_padded_executed": int(
            movement_stats["source_rows_padded_executed"]
        ),
        "movement_transition_attempts_padded_executed": int(
            movement_stats["transition_attempts_padded_executed"]
        ),
        "dig_prefilter_candidate_rows_logical": int(
            prefilter_stats["candidate_rows_logical"]
        ),
        "dig_prefilter_candidate_rows_padded_executed": int(
            prefilter_stats["candidate_rows_padded_executed"]
        ),
        "service_dig_candidate_attempts_logical": int(len(accepted)),
        "service_candidate_rows_padded_executed": service_padded_rows,
        "service_dig_do_transitions_padded_executed": service_padded_rows,
        "dump_do_attempts_logical": int(len(accepted) * cabin_headings),
        "dump_do_transitions_padded_executed": int(
            service_padded_rows * cabin_headings
        ),
    }


def _validate_population_against_reference(
    current: dict[str, int],
    reference_outcome: dict[str, Any],
) -> None:
    expected = {key: reference_outcome[key] for key in POPULATION_COUNTER_KEYS}
    if current != expected:
        mismatches = [
            key for key in POPULATION_COUNTER_KEYS if current[key] != expected[key]
        ]
        raise RuntimeError(
            "Canonical CPU candidate population differs from confirmation: "
            + ", ".join(mismatches)
            + "."
        )


class _CpuServiceRunner:
    def __init__(self, service_batch: Any, jax: Any) -> None:
        self._service_batch = service_batch
        self._jax = jax

    def __call__(self, state: Any, rows: Any) -> tuple[Any, np.ndarray]:
        _require_tree_on_platform(self._jax, state, "cpu")
        _require_tree_on_platform(self._jax, rows, "cpu")
        result = self._service_batch(state, rows)
        _require_tree_on_platform(self._jax, result, "cpu")
        host_rows = _require_int32_rows(
            self._jax.device_get(rows),
            "Canonical CPU service rows",
        )
        return result, host_rows


class _GpuServiceRunner:
    def __init__(
        self,
        *,
        service_batch: Any,
        jax: Any,
        gpu_device: Any,
    ) -> None:
        self._service_batch = service_batch
        self._jax = jax
        self._gpu_device = gpu_device
        self._gpu_state: Any | None = None
        self._compiled: Any | None = None

    def __call__(self, state: Any, rows: Any) -> tuple[Any, np.ndarray]:
        _require_tree_on_platform(self._jax, state, "cpu")
        _require_tree_on_platform(self._jax, rows, "cpu")
        host_rows = _require_int32_rows(
            self._jax.device_get(rows),
            "Pre-transfer CPU service rows",
        )
        with self._jax.default_device(self._gpu_device):
            if self._gpu_state is None:
                self._gpu_state = self._jax.device_put(state, self._gpu_device)
                _require_tree_on_platform(self._jax, self._gpu_state, "gpu")
            gpu_rows = self._jax.device_put(host_rows, self._gpu_device)
            _require_tree_on_platform(self._jax, gpu_rows, "gpu")
            if self._compiled is None:
                self._compiled = self._service_batch.lower(
                    self._gpu_state,
                    gpu_rows,
                ).compile()
            result = self._compiled(self._gpu_state, gpu_rows)
            _require_tree_on_platform(self._jax, result, "gpu")
            roundtrip_rows = _require_int32_rows(
                self._jax.device_get(gpu_rows),
                "Post-transfer GPU service rows",
            )
            host_result = tuple(
                np.asarray(self._jax.device_get(leaf)) for leaf in result
            )
        if not np.array_equal(roundtrip_rows, host_rows):
            raise RuntimeError("Candidate rows changed during CPU-to-GPU transfer.")
        return host_result, roundtrip_rows


class _ReplayCapture:
    def __init__(
        self,
        *,
        direct_service: Any,
        jax: Any,
        service_runner: Callable[[Any, Any], tuple[Any, np.ndarray]],
        reference_outcome: dict[str, Any],
        cached_cpu_capture: _ReplayCapture | None = None,
    ) -> None:
        self.direct_service = direct_service
        self.jax = jax
        self.service_runner = service_runner
        self.reference_outcome = reference_outcome
        self.cached_cpu_capture = cached_cpu_capture
        self.poses: np.ndarray | None = None
        self.movement_stats: dict[str, int] | None = None
        self.candidates: np.ndarray | None = None
        self.accepted: np.ndarray | None = None
        self.prefilter_stats: dict[str, int] | None = None
        self.population: dict[str, int] | None = None
        self.candidate_sha256: str | None = None
        self._candidate_transfer_hasher: _OrderedLeafHasher | None = None
        self._output_hasher: _OrderedLeafHasher | None = None
        self._service_rows_processed = 0
        self._service_batches = 0

    def reachable(self, state: Any) -> tuple[np.ndarray, dict[str, int]]:
        _require_tree_on_platform(self.jax, state, "cpu")
        if self.poses is not None:
            raise RuntimeError("Reachable-pose traversal ran more than once.")
        if self.cached_cpu_capture is None:
            poses, stats = self._original_reachable(state)
        else:
            assert self.cached_cpu_capture.poses is not None
            assert self.cached_cpu_capture.movement_stats is not None
            poses = self.cached_cpu_capture.poses.copy()
            stats = dict(self.cached_cpu_capture.movement_stats)
        self.poses = np.asarray(poses, dtype=np.int32)
        self.movement_stats = dict(stats)
        return self.poses.copy(), dict(self.movement_stats)

    def prefilter(
        self,
        state: Any,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, int]]:
        _require_tree_on_platform(self.jax, state, "cpu")
        if self.accepted is not None:
            raise RuntimeError("Dig prefilter ran more than once.")
        candidate_rows = np.asarray(candidates)
        if candidate_rows.dtype != np.int32 or candidate_rows.ndim != 2:
            raise RuntimeError("Canonical candidate rows must be a 2D int32 array.")
        if self.cached_cpu_capture is None:
            accepted, stats = self._original_prefilter(state, candidate_rows)
        else:
            cached = self.cached_cpu_capture
            assert cached.candidates is not None
            assert cached.accepted is not None
            assert cached.prefilter_stats is not None
            if not np.array_equal(candidate_rows, cached.candidates):
                raise RuntimeError(
                    "Candidate construction changed before the GPU replay."
                )
            accepted = cached.accepted.copy()
            stats = dict(cached.prefilter_stats)
        accepted_rows = np.asarray(accepted)
        if (
            accepted_rows.dtype != np.int32
            or accepted_rows.ndim != 2
            or accepted_rows.shape[1:] != (4,)
            or len(accepted_rows) == 0
        ):
            raise RuntimeError(
                "Ordered service candidates must be a nonempty (N, 4) int32 array."
            )
        assert self.poses is not None
        assert self.movement_stats is not None
        self.candidates = candidate_rows.copy()
        self.accepted = accepted_rows.copy()
        self.prefilter_stats = dict(stats)
        self.population = _population_counters(
            poses=self.poses,
            movement_stats=self.movement_stats,
            candidates=self.candidates,
            accepted=self.accepted,
            prefilter_stats=self.prefilter_stats,
            cabin_headings=int(state.env_cfg.agent.angles_cabin),
        )
        _validate_population_against_reference(
            self.population,
            self.reference_outcome,
        )
        self.candidate_sha256 = _sha256_array(self.accepted)
        self._candidate_transfer_hasher = _OrderedLeafHasher(
            total_rows=len(self.accepted),
            leaf_count=1,
            required_dtype=np.dtype(np.int32),
        )
        self._output_hasher = _OrderedLeafHasher(
            total_rows=len(self.accepted),
            leaf_count=EXPECTED_OUTPUT_LEAF_COUNT,
            required_dtype=np.dtype(np.int32),
        )
        return self.accepted.copy(), dict(self.prefilter_stats)

    def service(self, state: Any, rows: Any) -> Any:
        if self.accepted is None:
            raise RuntimeError("Service execution started before CPU prefiltering.")
        remaining = len(self.accepted) - self._service_rows_processed
        valid_count = min(SERVICE_BATCH_SIZE, remaining)
        if valid_count <= 0:
            raise RuntimeError("Service executed beyond the candidate population.")
        expected_chunk = self.accepted[
            self._service_rows_processed : self._service_rows_processed + valid_count
        ]
        expected_padded, _ = self.direct_service._pad_rows(
            expected_chunk,
            SERVICE_BATCH_SIZE,
        )
        host_rows = _require_int32_rows(
            self.jax.device_get(rows),
            "Canonical replay service rows",
        )
        if not np.array_equal(host_rows, expected_padded):
            raise RuntimeError(
                "Canonical service order or padding changed before dispatch."
            )
        result, transferred_rows = self.service_runner(state, rows)
        transferred = np.asarray(transferred_rows)
        if transferred.dtype != np.int32 or not np.array_equal(
            transferred,
            expected_padded,
        ):
            raise RuntimeError("Ordered int32 candidates changed during dispatch.")
        host_result = tuple(np.asarray(self.jax.device_get(leaf)) for leaf in result)
        assert self._candidate_transfer_hasher is not None
        assert self._output_hasher is not None
        self._candidate_transfer_hasher.update((transferred,), valid_count)
        self._output_hasher.update(host_result, valid_count)
        self._service_rows_processed += valid_count
        self._service_batches += 1
        return result

    def finish(self) -> dict[str, Any]:
        if (
            self.accepted is None
            or self.population is None
            or self.candidate_sha256 is None
            or self._candidate_transfer_hasher is None
            or self._output_hasher is None
        ):
            raise RuntimeError("Replay capture is incomplete.")
        candidate_transfer = self._candidate_transfer_hasher.finish()
        output = self._output_hasher.finish()
        transfer_hash = candidate_transfer["leaves"][0]["sha256"]
        if transfer_hash != self.candidate_sha256:
            raise RuntimeError(
                "Ordered candidate hash changed across the service boundary."
            )
        expected_batches = (
            len(self.accepted) + SERVICE_BATCH_SIZE - 1
        ) // SERVICE_BATCH_SIZE
        if self._service_batches != expected_batches:
            raise RuntimeError(
                f"Executed {self._service_batches} service batches, expected "
                f"{expected_batches}."
            )
        return {
            "candidate": {
                "dtype": "int32",
                "shape": list(self.accepted.shape),
                "ordered_unpadded_sha256": self.candidate_sha256,
                "dispatch_roundtrip_sha256": transfer_hash,
                "exact_dispatch_roundtrip": True,
            },
            "population_counters": self.population,
            "service": {
                "batch_size": SERVICE_BATCH_SIZE,
                "batch_count": self._service_batches,
                "padded_rows_executed": expected_batches * SERVICE_BATCH_SIZE,
                "ordered_unpadded_outputs": output,
            },
        }

    _original_reachable: Any
    _original_prefilter: Any


@contextmanager
def _patched_direct_service(
    direct_service: Any,
    capture: _ReplayCapture,
) -> Iterator[None]:
    originals = {
        "reachable": direct_service._reachable_base_poses,
        "prefilter": direct_service._prefilter_candidates,
        "service": direct_service._service_batch,
    }
    capture._original_reachable = originals["reachable"]
    capture._original_prefilter = originals["prefilter"]
    direct_service._reachable_base_poses = capture.reachable
    direct_service._prefilter_candidates = capture.prefilter
    direct_service._service_batch = capture.service
    try:
        yield
    finally:
        direct_service._reachable_base_poses = originals["reachable"]
        direct_service._prefilter_candidates = originals["prefilter"]
        direct_service._service_batch = originals["service"]


def _run_replay(
    *,
    direct_service: Any,
    jax: Any,
    cpu_device: Any,
    state: Any,
    capture: _ReplayCapture,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with jax.default_device(cpu_device):
        with _patched_direct_service(direct_service, capture):
            outcome = direct_service.compute_initial_direct_service(state)
    return outcome, capture.finish()


def _validate_full_population_parity(
    *,
    reference_outcome: dict[str, Any],
    cpu_outcome: dict[str, Any],
    cpu_capture: dict[str, Any],
    gpu_outcome: dict[str, Any],
    gpu_capture: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "cpu_final_outcome_matches_confirmation": cpu_outcome == reference_outcome,
        "hybrid_final_outcome_matches_confirmation": gpu_outcome == reference_outcome,
        "hybrid_final_outcome_matches_cpu_replay": gpu_outcome == cpu_outcome,
        "population_counters_match": (
            cpu_capture["population_counters"]
            == gpu_capture["population_counters"]
            == {key: reference_outcome[key] for key in POPULATION_COUNTER_KEYS}
        ),
        "ordered_candidate_hash_matches": (
            cpu_capture["candidate"]["ordered_unpadded_sha256"]
            == gpu_capture["candidate"]["ordered_unpadded_sha256"]
            == gpu_capture["candidate"]["dispatch_roundtrip_sha256"]
        ),
        "ordered_output_hashes_match": (
            cpu_capture["service"]["ordered_unpadded_outputs"]
            == gpu_capture["service"]["ordered_unpadded_outputs"]
        ),
    }
    failures = [name for name, passes in checks.items() if not passes]
    if failures:
        raise RuntimeError(
            "Full-population CPU/hybrid parity failed: " + ", ".join(failures) + "."
        )
    return {
        **checks,
        "all_full_population_parity_gates_pass": True,
        "reference_limitation": (
            "The pinned CPU confirmation predates full candidate/output hashes. "
            "This run establishes them with a fresh canonical CPU replay whose "
            "entire outcome, every population counter, shared execution code, "
            "input, protocol, and state match that pinned confirmation, then "
            "requires exact GPU service equality."
        ),
    }


def _assert_non_admission_receipt(receipt: dict[str, Any]) -> None:
    required_false = (
        "admission_result_emitted",
        "bank_admission_result_emitted",
        "timing_result_emitted",
        "cost_profile_called",
        "bank_profile_called",
        "static_admission_authorized",
        "ppo_authorized",
    )
    failures = [key for key in required_false if receipt.get(key) is not False]
    if failures:
        raise RuntimeError(f"Hybrid parity receipt exceeds its scope: {failures}.")
    decision = receipt.get("decision")
    if not isinstance(decision, dict):
        raise RuntimeError("Hybrid parity receipt has no decision object.")
    expected_decision = {
        "authorizes_one_hybrid_cost_profile": True,
        "authorizes_bank_profile": False,
        "authorizes_static_admission": False,
        "authorizes_ppo": False,
    }
    decision_failures = [
        key
        for key, expected in expected_decision.items()
        if decision.get(key) is not expected
    ]
    if decision_failures:
        raise RuntimeError(
            "Hybrid parity decision exceeds or contradicts its scope: "
            f"{decision_failures}."
        )
    forbidden_result_keys = {
        "wall_seconds",
        "timings_seconds",
        "projections",
        "calibration",
        "memory_headroom",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            present = forbidden_result_keys.intersection(value)
            if present:
                raise RuntimeError(
                    "Hybrid parity receipt must not emit cost/timing fields: "
                    f"{sorted(present)}."
                )
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(receipt)


def run_parity(
    b0a_root: Path,
    cpu_confirmation_path: Path,
    output_directory: Path,
) -> Path:
    jax_platforms = _require_jax_platforms_unset(os.environ)

    import jax

    import terra.benchmark_direct_service as direct_service
    import tools.confirm_b0a_direct_service_cost as confirmation
    import tools.profile_b0a_direct_service_cost as probe_tool

    b0a_root = b0a_root.resolve()
    cpu_confirmation_path = cpu_confirmation_path.resolve()
    output_directory = output_directory.resolve()
    output_path = output_directory / OUTPUT_NAME
    candidate_output_path = output_directory / CANDIDATE_OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(output_path)
    if candidate_output_path.exists():
        raise FileExistsError(candidate_output_path)

    reference, reference_sha256 = _load_cpu_confirmation(
        cpu_confirmation_path,
        outcome_keys=probe_tool.DIRECT_SERVICE_OUTCOME_KEYS,
        selected_map_id=probe_tool.SELECTED_MAP_ID,
    )
    reference_root = Path(reference["input"]["b0a_root"]).resolve()
    if b0a_root != reference_root:
        raise RuntimeError(
            f"Hybrid parity requires B0a root {reference_root}, got {b0a_root}."
        )

    cpu_devices = jax.devices("cpu")
    gpu_devices = jax.devices("gpu")
    cpu_device, gpu_device, devices = _validate_devices(
        cpu_devices,
        gpu_devices,
        jax.default_backend(),
    )
    repository = Path(__file__).resolve().parents[1]
    code_before = _code_receipt(repository, probe_tool)
    _require_clean_worktree(code_before)
    execution_code_reference = _validate_execution_code_reference(
        reference,
        code_before,
    )

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
    with jax.default_device(cpu_device):
        env_config, protocol = probe_tool._frozen_env_config()
        state, initial_state = probe_tool._materialize_initial_state(
            selected,
            group,
            source_group_id,
            env_config,
        )
        state = jax.device_put(state, cpu_device)
        probe_tool._synchronize(state)
    _require_tree_on_platform(jax, state, "cpu")
    _validate_rebuilt_contract(
        reference=reference,
        selected_identity=selected_identity,
        manifest_sha256=manifest_sha256,
        source_grouping=source_grouping,
        verified_selected_files=verified_selected_files,
        protocol=protocol,
        initial_state=initial_state,
        stable_initial_state=confirmation._stable_initial_state,
    )
    if int(direct_service._SERVICE_BATCH_SIZE) != SERVICE_BATCH_SIZE:
        raise RuntimeError(
            "The exact direct-service batch size changed: "
            f"{SERVICE_BATCH_SIZE} != {direct_service._SERVICE_BATCH_SIZE}."
        )

    original_service_batch = direct_service._service_batch
    reference_outcome = reference["measurement"]["exact_outcome"]
    cpu_capture_state = _ReplayCapture(
        direct_service=direct_service,
        jax=jax,
        service_runner=_CpuServiceRunner(original_service_batch, jax),
        reference_outcome=reference_outcome,
    )
    cpu_outcome, cpu_capture = _run_replay(
        direct_service=direct_service,
        jax=jax,
        cpu_device=cpu_device,
        state=state,
        capture=cpu_capture_state,
    )
    gpu_capture_state = _ReplayCapture(
        direct_service=direct_service,
        jax=jax,
        service_runner=_GpuServiceRunner(
            service_batch=original_service_batch,
            jax=jax,
            gpu_device=gpu_device,
        ),
        reference_outcome=reference_outcome,
        cached_cpu_capture=cpu_capture_state,
    )
    gpu_outcome, gpu_capture = _run_replay(
        direct_service=direct_service,
        jax=jax,
        cpu_device=cpu_device,
        state=state,
        capture=gpu_capture_state,
    )
    parity = _validate_full_population_parity(
        reference_outcome=reference_outcome,
        cpu_outcome=cpu_outcome,
        cpu_capture=cpu_capture,
        gpu_outcome=gpu_outcome,
        gpu_capture=gpu_capture,
    )

    code_after = _code_receipt(repository, probe_tool)
    if code_after != code_before:
        raise RuntimeError("Terra or hybrid parity code changed during execution.")
    assert cpu_capture_state.accepted is not None
    output_directory.mkdir(parents=True, exist_ok=True)
    with candidate_output_path.open("xb") as stream:
        np.save(stream, cpu_capture_state.accepted, allow_pickle=False)
    candidate_file_sha256 = _sha256_file(candidate_output_path)

    receipt = {
        "schema": SCHEMA,
        "release_id": probe_tool.RELEASE_ID,
        "result_scope": "non_admission_full_population_hybrid_exact_parity",
        "admission_result_emitted": False,
        "bank_admission_result_emitted": False,
        "timing_result_emitted": False,
        "cost_profile_called": False,
        "bank_profile_called": False,
        "static_admission_authorized": False,
        "ppo_authorized": False,
        "selected_identity": selected_identity,
        "input": {
            "b0a_root": str(b0a_root),
            "cpu_confirmation_receipt": str(cpu_confirmation_path),
            "cpu_confirmation_receipt_sha256": reference_sha256,
            "files_sha256_manifest_sha256": manifest_sha256,
            "source_grouping": source_grouping,
            "verified_selected_files": verified_selected_files,
            "ordered_service_candidates": str(candidate_output_path),
            "ordered_service_candidates_file_sha256": candidate_file_sha256,
            "ordered_service_candidates_array_sha256": cpu_capture["candidate"][
                "ordered_unpadded_sha256"
            ],
        },
        "protocol": protocol,
        "initial_state": initial_state,
        "experiment": {
            "treatment": "canonical_cpu_graph_prefilter_gpu_service_only",
            "cpu_graph_prefilter_execution_count": 1,
            "cpu_service_reference_execution_count": 1,
            "gpu_service_replay_execution_count": 1,
            "gpu_kernel": "terra.benchmark_direct_service._service_batch",
            "other_gpu_kernels": [],
            "service_batch_size": SERVICE_BATCH_SIZE,
            "candidate_transfer": (
                "ordered unpadded int32 CPU rows; each padded batch is checked "
                "before transfer and after GPU roundtrip"
            ),
        },
        "cpu_reference_replay": {
            "outcome": cpu_outcome,
            **cpu_capture,
        },
        "hybrid_gpu_replay": {
            "outcome": gpu_outcome,
            **gpu_capture,
        },
        "parity": parity,
        "validator": {
            "code_before": code_before,
            "code_after": code_after,
            "pre_and_post_execution_receipts_identical": True,
            "cpu_execution_code_reference": execution_code_reference,
            "benchmark_protocol_explicitly_hashed": True,
            "canonical_entrypoint": (
                "terra.benchmark_direct_service.compute_initial_direct_service"
            ),
            "canonical_reducer_call_count": 2,
        },
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": {
                name: probe_tool._package_version(name)
                for name in ("jax", "jaxlib", "numpy", "scipy")
            },
            "jax": devices,
            "jax_platforms_environment": jax_platforms,
        },
        "decision": {
            "exact_full_population_hybrid_parity_passes": True,
            "authorizes_one_hybrid_cost_profile": True,
            "authorizes_bank_profile": False,
            "authorizes_static_admission": False,
            "authorizes_ppo": False,
        },
        "command": [str(argument) for argument in sys.argv],
    }
    _assert_non_admission_receipt(receipt)
    with output_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the confirmed full B0a candidate population on canonical "
            "CPU and the bounded CPU-graph/GPU-service path without timing or "
            "admission."
        )
    )
    parser.add_argument("--b0a-root", type=Path, required=True)
    parser.add_argument("--cpu-confirmation-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    run_parity(
        args.b0a_root,
        args.cpu_confirmation_receipt,
        args.output_directory,
    )


if __name__ == "__main__":
    main()
