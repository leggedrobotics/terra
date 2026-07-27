import ast
import inspect

import numpy as np
import pytest

import tools.sweep_b0a_direct_service_batch_size as sweep


class _IdentityServiceBatch:
    def __call__(self, state, rows):
        del state
        values = np.asarray(rows)[:, :1]
        return values, 2 * values, 3 * values


class _LowerableIdentityServiceBatch:
    @staticmethod
    def lower(state, rows):
        del state, rows

        class _Lowered:
            @staticmethod
            def compile():
                return _IdentityServiceBatch()

        return _Lowered()


def test_equal_subsets_and_padded_tail_have_exact_output_parity(monkeypatch):
    monkeypatch.setattr(sweep.probe_tool, "_synchronize", lambda value: value)
    rows = np.column_stack(
        (
            np.arange(sweep.PARITY_SUBSET_ROWS, dtype=np.int32),
            np.zeros((sweep.PARITY_SUBSET_ROWS, 3), dtype=np.int32),
        )
    )
    reference = None
    expected_batches = {4: 5, 8: 3, 16: 2}
    expected_padded_rows = {4: 20, 8: 24, 16: 32}

    for batch_size in sweep.BATCH_SIZES:
        execution, outputs = sweep._execute_subset(
            _IdentityServiceBatch(),
            None,
            rows,
            batch_size,
            capture_outputs=True,
        )
        assert execution["logical_rows"] == sweep.PARITY_SUBSET_ROWS
        assert execution["batch_count"] == expected_batches[batch_size]
        assert execution["padded_rows"] == expected_padded_rows[batch_size]
        if reference is None:
            reference = outputs
        else:
            assert sweep._output_sha256(reference) == sweep._output_sha256(outputs)


def test_output_hash_covers_dtype_shape_and_leaf_count():
    value = np.asarray((1, 2), dtype=np.int32)
    reference = sweep._output_sha256((value,))
    assert reference != sweep._output_sha256((value.astype(np.int64),))
    assert reference != sweep._output_sha256((value.reshape(1, 2),))
    assert reference != sweep._output_sha256((value, value))


def test_measurement_receipts_equal_warmup_repeats_and_padded_tail(monkeypatch):
    clock = {"value": 0.0}

    def perf_counter():
        clock["value"] += 0.01
        return clock["value"]

    monkeypatch.setattr(sweep.time, "perf_counter", perf_counter)
    monkeypatch.setattr(sweep.jax, "clear_caches", lambda: None)
    monkeypatch.setattr(sweep.probe_tool, "_synchronize", lambda value: value)
    monkeypatch.setattr(sweep.probe_tool, "_max_rss_kib", lambda: 10)
    monkeypatch.setattr(
        sweep.direct_service,
        "_service_batch",
        _LowerableIdentityServiceBatch(),
    )
    rows = np.column_stack(
        (
            np.arange(20, dtype=np.int32),
            np.zeros((20, 3), dtype=np.int32),
        )
    )

    arm, _ = sweep._measure_batch_size(
        state=None,
        replay_candidates=rows,
        batch_size=8,
        cold_graph_seconds=1.0,
        warm_graph_seconds=0.5,
        host_capacity_kib=100,
        cabin_headings=12,
    )

    counters = arm["execution_counters"]
    assert counters["full_service_batch_count_projected"] == 3
    assert counters["full_service_padded_rows_projected"] == 24
    assert counters["warmup_logical_rows_executed"] == 16
    assert counters["warmup_padded_rows_executed"] == 16
    assert counters["timed_repeat_count"] == 12
    assert counters["timed_logical_rows_executed"] == 192
    assert counters["timed_padded_rows_executed"] == 192
    assert counters["parity_logical_rows_executed"] == 18
    assert counters["parity_padded_rows_executed"] == 24
    assert counters["actual_service_dig_transitions_logical"] == 234
    assert counters["actual_service_dig_transitions_padded"] == 240
    assert counters["actual_dump_do_transitions_logical"] == 2808
    assert counters["actual_dump_do_transitions_padded"] == 2880
    assert arm["memory"]["all_memory_gates_pass"] is True


def _arm(*, p50, p95, repeat_p50=1.0, parity=True, timing=True, memory=True):
    return {
        "timing": {"complete_16_row_repeat_p50": repeat_p50},
        "projections": {
            "scenario_counts": {
                "256": {
                    "seconds_p50": p50,
                    "seconds_p95": p95,
                }
            }
        },
        "gates": {
            "exact_concatenated_output_parity": parity,
            "timing_finite_positive": timing,
            "cumulative_memory_headroom_passes": memory,
            "eligible_for_reprobe": parity and timing and memory,
        },
    }


def test_selection_is_strict_excludes_failed_arms_and_breaks_ties_smallest():
    arms = {
        "4_open": _arm(p50=100.0, p95=110.0),
        "4_close": _arm(p50=100.0, p95=110.0),
        "8": _arm(p50=70.0, p95=80.0),
        "16": _arm(p50=60.0, p95=70.0, parity=False),
    }
    assert sweep._select_batch_size(arms)["selected_for_reprobe_batch_size"] == 8

    arms["16"] = _arm(p50=70.0, p95=80.0)
    assert sweep._select_batch_size(arms)["selected_for_reprobe_batch_size"] == 8

    arms["8"] = _arm(p50=100.0, p95=100.0)
    arms["16"] = _arm(p50=90.0, p95=100.0)
    decision = sweep._select_batch_size(arms)
    assert decision["selected_for_reprobe_batch_size"] == 4
    assert decision["strict_p95_below_baseline_p50"] is False

    arms["8"] = _arm(p50=60.0, p95=70.0)
    arms["4_close"] = _arm(p50=100.0, p95=110.0, repeat_p50=1.1)
    decision = sweep._select_batch_size(arms)
    assert decision["selected_for_reprobe_batch_size"] == 4
    assert decision["baseline_drift"]["passes"] is False


def test_batch_count_uses_padded_tail_and_full_validator_is_never_called():
    assert [sweep._service_batch_count(18, size) for size in sweep.BATCH_SIZES] == [
        5,
        3,
        2,
    ]

    tree = ast.parse(inspect.getsource(sweep))
    forbidden_names = {
        "compute_initial_direct_service",
        "_compute_exact_once",
        "run_confirmation",
    }
    forbidden_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forbidden_names
    ]
    assert forbidden_calls == []
    assert "_assert_cost_only(receipt)" in inspect.getsource(sweep.run_sweep)


def test_runtime_must_match_pinned_packages_and_cache_environment():
    current = {
        "hostname": "host",
        "machine": "x86_64",
        "python": "3.12",
        "packages": {"jax": "1"},
        "jax": {"device_count": 1, "platform": "cpu", "device_kind": "cpu"},
        "compilation_cache_environment": {
            "JAX_COMPILATION_CACHE_DIR": None,
            "JAX_ENABLE_COMPILATION_CACHE": None,
            "JAX_PLATFORMS": "cpu",
        },
    }
    probe = {"machine": current}
    sweep._validate_pinned_runtime(probe, current)

    changed = {
        **current,
        "compilation_cache_environment": {
            **current["compilation_cache_environment"],
            "JAX_PLATFORMS": None,
        },
    }
    with pytest.raises(RuntimeError, match="compilation-cache environment"):
        sweep._validate_pinned_runtime(probe, changed)

    enabled = {
        **current,
        "compilation_cache_environment": {
            **current["compilation_cache_environment"],
            "JAX_COMPILATION_CACHE_DIR": "/tmp/cache",
        },
    }
    with pytest.raises(RuntimeError, match="must be disabled"):
        sweep._validate_pinned_runtime(probe, enabled)
