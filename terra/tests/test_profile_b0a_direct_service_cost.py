import ast
import inspect
import json
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

import tools.profile_b0a_direct_service_cost as profiler
from tools.profile_b0a_direct_service_cost import MAP_SIZE
from tools.profile_b0a_direct_service_cost import SELECTED_MAP_ID
from tools.profile_b0a_direct_service_cost import ScenarioInput
from tools.profile_b0a_direct_service_cost import _assert_cost_only
from tools.profile_b0a_direct_service_cost import _intersect_spawn_contract
from tools.profile_b0a_direct_service_cost import _project_cost
from tools.profile_b0a_direct_service_cost import _select_group_records


def _scenario(map_id: str) -> ScenarioInput:
    shape = (MAP_SIZE, MAP_SIZE)
    return ScenarioInput(
        record={"map_id": map_id},
        dataset_directory=None,
        slot_index=1,
        target=np.zeros(shape, dtype=np.int8),
        padding_mask=np.zeros(shape, dtype=np.int8),
        trench_axes=np.zeros((3, 3), dtype=np.float32),
        trench_type=np.asarray(-1, dtype=np.int32),
        foundation_border_axes=np.zeros((64, 3), dtype=np.float32),
        foundation_border_type=np.asarray(-1, dtype=np.int32),
        dumpability_mask=np.ones(shape, dtype=np.bool_),
        action_map=np.zeros(shape, dtype=np.int8),
        distance_map=np.zeros(shape, dtype=np.float32),
        selected_files=(),
    )


def test_named_identity_selects_the_complete_counterfactual_group(tmp_path):
    rows = []
    for distance in ("d02", "d04", "d06", "d08"):
        rows.append(
            {
                "map_id": f"b0a-train-f_apron_{distance}-00",
                "split": "train",
                "stratum": "B0a",
                "source_id": "osm-foundation:108",
                "paired_source_group_id": "train:foundation-distance:00",
                "dig_identity_sha256": "a" * 64,
            }
        )
    rows.append(
        {
            "map_id": "b0a-development-f_apron_d02-00",
            "split": "development",
            "stratum": "B0a",
            "source_id": "osm-foundation:999",
            "paired_source_group_id": "development:foundation-distance:00",
            "dig_identity_sha256": "b" * 64,
        }
    )
    path = tmp_path / "identities.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected, group, source_group_id = _select_group_records(path)

    assert selected["map_id"] == SELECTED_MAP_ID
    assert source_group_id == "osm-foundation:108"
    assert [row["map_id"] for row in group] == sorted(row["map_id"] for row in rows[:4])


def test_repeated_source_identity_cannot_cross_splits(tmp_path):
    rows = [
        {
            "map_id": SELECTED_MAP_ID,
            "source_id": "osm-foundation:108",
            "split": "train",
            "stratum": "B0a",
            "paired_source_group_id": None,
            "dig_identity_sha256": "a" * 64,
        },
        {
            "map_id": "b0a-development-f_apron_d02-00",
            "source_id": "osm-foundation:108",
            "split": "development",
            "stratum": "B0a",
            "paired_source_group_id": None,
            "dig_identity_sha256": "a" * 64,
        },
    ]
    path = tmp_path / "identities.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(RuntimeError, match="crosses split or dig identity"):
        _select_group_records(path)


def test_spawn_contract_intersects_every_counterfactual_variant():
    first = _scenario("first")
    second = _scenario("second")
    first.padding_mask[3, 4] = 1
    second.padding_mask[5, 6] = 1
    first.action_map[7, 8] = -1
    second.action_map[9, 10] = 4
    first.dumpability_mask[11, 12] = False
    second.dumpability_mask[13, 14] = False

    padding, actions, dumpability = _intersect_spawn_contract([first, second])

    assert padding[3, 4] == padding[5, 6] == 1
    assert actions[7, 8] == actions[9, 10] == 1
    assert not dumpability[11, 12]
    assert not dumpability[13, 14]


def test_projection_uses_one_cold_scenario_then_warm_padded_batches():
    projected = _project_cost(
        cold_graph_prefilter_seconds=2.0,
        warm_graph_prefilter_seconds=1.0,
        lowering_seconds=0.5,
        cold_compile_seconds=3.0,
        first_execute_seconds=0.25,
        service_padded_batch_count=10,
        steady_seconds_per_padded_batch_p50=0.1,
        steady_seconds_per_padded_batch_p95=0.2,
    )

    assert projected["first_scenario_cold_seconds_p50"] == pytest.approx(6.65)
    assert projected["first_scenario_cold_seconds_p95"] == pytest.approx(7.55)
    assert projected["later_scenario_warm_seconds_p50"] == pytest.approx(2.0)
    assert projected["later_scenario_warm_seconds_p95"] == pytest.approx(3.0)
    assert projected["scenario_counts"]["1"]["seconds_p50"] == pytest.approx(6.65)
    assert projected["scenario_counts"]["256"]["seconds_p50"] == pytest.approx(516.65)
    assert projected["scenario_counts"]["448"]["seconds_p95"] == pytest.approx(1348.55)
    assert all(
        row["cold_compile_executions"] == 1
        for row in projected["scenario_counts"].values()
    )


def test_cost_only_guard_rejects_accidental_subset_feasibility():
    _assert_cost_only(
        {
            "admission_result_emitted": False,
            "timings_seconds": {"service_cold_compile": 1.0},
        }
    )
    for key in sorted(profiler.DIRECT_SERVICE_OUTCOME_KEYS):
        with pytest.raises(RuntimeError, match="Cost-only probe"):
            _assert_cost_only({key: 0})
    with pytest.raises(RuntimeError, match="Cost-only probe"):
        _assert_cost_only({"subset_validity": True})


def test_cost_only_guard_covers_every_direct_service_outcome_key():
    source = textwrap.dedent(
        inspect.getsource(profiler.direct_service.compute_initial_direct_service)
    )
    tree = ast.parse(source)
    returned_dicts = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert len(returned_dicts) == 1
    returned_keys = {
        key.value
        for key in returned_dicts[0].keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert returned_keys == profiler.DIRECT_SERVICE_OUTCOME_KEYS


def test_profile_uses_validator_counters_and_discards_mocked_outputs(monkeypatch):
    poses = np.zeros((2, 3), dtype=np.int32)
    candidates = np.zeros((24, 4), dtype=np.int32)
    accepted = np.zeros((6, 4), dtype=np.int32)
    reach_calls = []
    prefilter_calls = []

    def reachable(state):
        reach_calls.append(state)
        return (
            poses,
            {
                "source_rows_logical": 2,
                "transition_attempts_logical": 8,
                "source_rows_padded_executed": 128,
                "transition_attempts_padded_executed": 512,
            },
        )

    def prefilter(state, rows):
        prefilter_calls.append((state, rows))
        return (
            accepted,
            {
                "candidate_rows_logical": 24,
                "candidate_rows_padded_executed": 128,
            },
        )

    monkeypatch.setattr(
        profiler.direct_service,
        "_reachable_base_poses",
        reachable,
    )
    monkeypatch.setattr(
        profiler.direct_service,
        "_candidate_rows",
        lambda poses, angles: candidates,
    )
    monkeypatch.setattr(
        profiler.direct_service,
        "_prefilter_candidates",
        prefilter,
    )

    class FakeLowered:
        @staticmethod
        def compile():
            return lambda state, rows: np.zeros((len(rows), 1), dtype=np.int8)

    class FakeServiceBatch:
        @staticmethod
        def lower(state, rows):
            return FakeLowered()

    monkeypatch.setattr(
        profiler.direct_service,
        "_service_batch",
        FakeServiceBatch(),
    )
    monkeypatch.setattr(profiler.jax, "clear_caches", lambda: None)
    state = SimpleNamespace(
        env_cfg=SimpleNamespace(agent=SimpleNamespace(angles_cabin=12))
    )

    result = profiler._profile_exact_service_cost(state)

    counts = result["logical_and_padded_counters"]
    assert counts["movement_source_rows_logical_per_full_execution"] == 2
    assert counts["movement_source_rows_padded_per_full_execution"] == 128
    assert counts["prefilter_rows_logical_per_full_execution"] == 24
    assert counts["prefilter_rows_padded_per_full_execution"] == 128
    assert counts["movement_source_rows_logical_executed_total"] == 4
    assert counts["movement_source_rows_padded_executed_total"] == 256
    assert counts["prefilter_rows_logical_executed_total"] == 48
    assert counts["prefilter_rows_padded_executed_total"] == 256
    assert counts["exact_service_logical_rows_available"] == 6
    assert counts["full_service_padded_batch_count_projected"] == 2
    assert counts["cold_service_dig_transitions_logical_executed"] == 4
    assert counts["cold_service_dig_transitions_padded_executed"] == 4
    assert counts["cold_service_dump_transitions_logical_executed"] == 48
    assert counts["cold_service_dump_transitions_padded_executed"] == 48
    assert counts["warm_service_dig_transitions_logical_executed"] == 18
    assert counts["warm_service_dig_transitions_padded_executed"] == 24
    assert counts["warm_service_dump_transitions_logical_executed"] == 216
    assert counts["warm_service_dump_transitions_padded_executed"] == 288
    assert counts["graph_prefilter_full_executions"] == 2
    assert len(reach_calls) == len(prefilter_calls) == 2
    assert "direct_service_coverage_initial" not in json.dumps(result)
