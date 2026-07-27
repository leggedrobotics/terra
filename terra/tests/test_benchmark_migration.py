from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tools.migrate_b0a_live_geometry import CONDITION_STATUS
from tools.migrate_b0a_live_geometry import DIRECT_SERVICE_STATUS
from tools.migrate_b0a_live_geometry import MIGRATION_STATUS
from tools.migrate_b0a_live_geometry import LoadedLegacyScenario
from tools.migrate_b0a_live_geometry import MaterializedState
from tools.migrate_b0a_live_geometry import derive_content_ids
from tools.migrate_b0a_live_geometry import migrate_loaded_scenarios
from tools.migrate_b0a_live_geometry import normalize_factor_vector
from tools.migrate_b0a_live_geometry import recompute_affordable_audit

EDGE_LENGTH_M = 36.5714285714
TILE_SIZE_M = EDGE_LENGTH_M / 64


def _identity(
    *,
    legacy_map_id: str = "legacy-a",
    source_id: str = "source-a",
    dump_layout: str = "broad_apron",
    geometry: str = "foundation_osm",
    family: str = "foundation",
    topology: str | None = None,
    side_access: str = "all",
) -> dict:
    return {
        "map_id": legacy_map_id,
        "source_id": source_id,
        "split": "train",
        "stratum": "B0a",
        "primary_cell": "synthetic",
        "family": family,
        "geometry": geometry,
        "topology": topology,
        "dump_layout": dump_layout,
        "side_access": side_access,
        "side_sign": None,
        "distance_center_tiles": 2,
        "paired_source_group_id": "legacy-pair",
        "topology_match_group_id": None,
        "validation": {"status": "passed"},
    }


def _layers(dump_shift: int = 0) -> tuple[np.ndarray, ...]:
    target = np.zeros((64, 64), dtype=np.int8)
    target[30:32, 30:32] = -1
    target[27 + dump_shift : 30 + dump_shift, 28:32] = 1
    occupancy = np.zeros_like(target, dtype=np.int8)
    dumpability = np.ones_like(target, dtype=np.bool_)
    initial_soil = np.zeros_like(target, dtype=np.int8)
    reward_distance = np.zeros_like(target, dtype=np.float32)
    return target, occupancy, dumpability, initial_soil, reward_distance


def _scenario(
    *,
    legacy_map_id: str = "legacy-a",
    source_id: str = "source-a",
    dump_shift: int = 0,
) -> LoadedLegacyScenario:
    target, occupancy, dumpability, initial_soil, reward_distance = _layers(dump_shift)
    return LoadedLegacyScenario(
        identity=_identity(
            legacy_map_id=legacy_map_id,
            source_id=source_id,
        ),
        metadata={
            "family": "foundation",
            "geometry": "foundation_osm",
            "map_id": legacy_map_id,
            "primary_cell": "synthetic",
            "axes_ABC": [],
            "foundation_border_axes_ABC": [],
        },
        target=target,
        occupancy=occupancy,
        dumpability=dumpability,
        initial_soil=initial_soil,
        reward_distance=reward_distance,
    )


def _factor(scenario: LoadedLegacyScenario) -> dict:
    audit = recompute_affordable_audit(
        scenario,
        tile_size_m=TILE_SIZE_M,
        initial_base_position=[10, 10],
    )
    return normalize_factor_vector(
        scenario.identity,
        separation_p50_tiles=audit["dump"]["dig_dump_separation_tiles"]["p50"],
        single_layer_area_ratio=audit["dump"]["single_layer_area_ratio"],
        required_volume=audit["work"]["required_volume"],
    )


def _state_record() -> dict:
    zeros = [0, 0, 0, 0]
    return {
        "schema": "terra_agent_state_v1",
        "width": 7,
        "height": 11,
        "max_agents": 4,
        "num_agents": 1,
        "current_agent": 0,
        "moving_dumped_dirt": False,
        "agent_active": [True, False, False, False],
        "agent_states": {
            "pos_base": [[10, 10], [0, 0], [0, 0], [0, 0]],
            "angle_base": zeros,
            "angle_cabin": zeros,
            "wheel_angle": zeros,
            "loaded": zeros,
            "agent_type": zeros,
            "action_type": zeros,
            "shovel_lifted": zeros,
            "carry_baseline_potential": [0.0, 0.0, 0.0, 0.0],
            "carry_potential_after_lift": [0.0, 0.0, 0.0, 0.0],
        },
    }


def _protocol() -> dict:
    return {
        "environment_protocol_sha256": "protocol-hash",
        "map": {
            "edge_length_px": 64,
            "edge_length_m": EDGE_LENGTH_M,
            "tile_size_m_derived_float64": TILE_SIZE_M,
        },
    }


@pytest.mark.parametrize(
    (
        "geometry",
        "topology",
        "expected_source",
        "expected_geometry",
        "expected_topology",
    ),
    [
        ("foundation_osm", None, "osm", "connected", None),
        ("foundation_procedural", None, "procedural", "connected", None),
        ("trench_straight", "straight", "procedural", "trench", "straight"),
        (
            "trench_segmented2",
            "segmented_end_to_end_2",
            "procedural",
            "trench",
            "segmented_2",
        ),
        (
            "trench_segmented3",
            "segmented_end_to_end_3",
            "procedural",
            "trench",
            "segmented_3",
        ),
        ("trench_T", "T", "procedural", "trench", "T"),
        ("trench_X", "X", "procedural", "trench", "X"),
        (
            "trench_disconnected",
            "disconnected_2",
            "procedural",
            "trench",
            "disconnected",
        ),
    ],
)
def test_legacy_geometry_and_topology_tokens_are_normalized(
    geometry,
    topology,
    expected_source,
    expected_geometry,
    expected_topology,
):
    family = "foundation" if geometry.startswith("foundation") else "trench"
    identity = _identity(
        geometry=geometry,
        family=family,
        topology=topology,
        dump_layout="broad_side_cast" if family == "trench" else "broad_apron",
        side_access="both" if family == "trench" else "all",
    )
    factors = normalize_factor_vector(
        identity,
        separation_p50_tiles=2.0,
        single_layer_area_ratio=3.25,
        required_volume=72,
    )
    assert factors["source_family"] == expected_source
    assert factors["geometry_class"] == expected_geometry
    assert factors["topology"] == expected_topology
    assert factors["dump_layout"] == ("side_cast" if family == "trench" else "apron")


def test_unknown_legacy_taxonomy_fails_loudly():
    identity = _identity(geometry="foundation_mystery")
    with pytest.raises(ValueError, match="Unsupported legacy geometry"):
        normalize_factor_vector(
            identity,
            separation_p50_tiles=2.0,
            single_layer_area_ratio=3.25,
            required_volume=72,
        )


def test_content_ids_respect_geometry_map_and_scenario_boundaries():
    scenario = _scenario()
    factors = _factor(scenario)
    baseline = derive_content_ids(
        scenario,
        factors,
        state_sha256="a" * 64,
        reset_seed_uint32=7,
    )

    changed_reward = replace(
        scenario,
        reward_distance=np.ones((64, 64), dtype=np.float32),
    )
    reward_ids = derive_content_ids(
        changed_reward,
        factors,
        state_sha256="a" * 64,
        reset_seed_uint32=7,
    )
    assert reward_ids["reward_distance_sha256"] != baseline["reward_distance_sha256"]
    assert reward_ids["geometry_id"] == baseline["geometry_id"]
    assert reward_ids["map_id"] == baseline["map_id"]
    assert reward_ids["scenario_id"] == baseline["scenario_id"]

    changed_dumpability = scenario.dumpability.copy()
    changed_dumpability[0, 0] = False
    map_ids = derive_content_ids(
        replace(scenario, dumpability=changed_dumpability),
        factors,
        state_sha256="a" * 64,
        reset_seed_uint32=7,
    )
    assert map_ids["geometry_id"] == baseline["geometry_id"]
    assert map_ids["map_id"] != baseline["map_id"]
    assert map_ids["scenario_id"] != baseline["scenario_id"]

    changed_soil = scenario.initial_soil.copy()
    changed_soil[30, 30] = -1
    soil_ids = derive_content_ids(
        replace(scenario, initial_soil=changed_soil),
        factors,
        state_sha256="a" * 64,
        reset_seed_uint32=7,
    )
    assert soil_ids["map_id"] == baseline["map_id"]
    assert soil_ids["scenario_id"] != baseline["scenario_id"]

    state_ids = derive_content_ids(
        scenario,
        factors,
        state_sha256="b" * 64,
        reset_seed_uint32=7,
    )
    seed_ids = derive_content_ids(
        scenario,
        factors,
        state_sha256="a" * 64,
        reset_seed_uint32=8,
    )
    assert state_ids["scenario_id"] != baseline["scenario_id"]
    assert seed_ids["scenario_id"] != baseline["scenario_id"]


def test_affordable_audit_uses_live_scale_exact_capacity_and_mass():
    audit = recompute_affordable_audit(
        _scenario(),
        tile_size_m=TILE_SIZE_M,
        initial_base_position=[10, 10],
    )
    tiles = audit["dump"]["dig_dump_separation_tiles"]
    metres = audit["dump"]["dig_dump_separation_m"]
    assert metres["p50"] == pytest.approx(tiles["p50"] * TILE_SIZE_M)
    assert metres["p95"] == pytest.approx(tiles["p95"] * TILE_SIZE_M)
    assert audit["dump"]["accepted_cells"] == 12
    assert audit["dump"]["single_layer_area_ratio"] == pytest.approx(3.0)
    assert audit["capacity_validation"]["representable_remaining_volume"] == (12 * 127)
    assert audit["reset"]["completion_fraction"] == 0.0
    assert audit["reset"]["initial_negative_volume"] == 0
    assert audit["reset"]["initial_positive_volume"] == 0
    assert audit["reset"]["mass_balance"] == {
        "residual": 0,
        "conserved": True,
    }


def test_source_group_gets_one_shared_state_and_static_claim_stays_deferred():
    scenarios = [
        _scenario(legacy_map_id="legacy-a", dump_shift=0),
        _scenario(legacy_map_id="legacy-b", dump_shift=-4),
    ]
    calls = []

    def materialize(source_group_id, split, variants, env_config):
        calls.append((source_group_id, split, len(variants), env_config))
        return MaterializedState(
            state_record=_state_record(),
            state_sha256="c" * 64,
            seed_receipt={
                "schema": "terra_initial_state_seed_v1",
                "release_id": "terramap-bench-v1.0.0",
                "split": split,
                "source_group_id": source_group_id,
                "state_index": 0,
                "seed_uint32": 123,
                "seed_byte_order": "big",
                "seed_digest_sha256": "d" * 64,
                "initial_agent_state_sha256": "c" * 64,
            },
        )

    outcomes = migrate_loaded_scenarios(
        scenarios,
        environment_protocol=_protocol(),
        env_config="synthetic-env",
        state_materializer=materialize,
        expected_identity_count=2,
        expected_source_group_count=1,
        expected_source_group_size_counts={2: 1},
    )
    assert calls == [("source-a", "public_train", 2, "synthetic-env")]
    assert len(outcomes) == 2
    assert {row["source_group_id"] for row in outcomes} == {"source-a"}
    assert {row["migration_status"] for row in outcomes} == {MIGRATION_STATUS}
    assert {row["condition_id"] for row in outcomes} == {None}
    assert {row["condition_status"] for row in outcomes} == {CONDITION_STATUS}
    assert {
        row["scenario"]["initial_condition"]["initial_agent_state_sha256"]
        for row in outcomes
    } == {"c" * 64}
    assert len({row["scenario"]["geometry_id"] for row in outcomes}) == 1
    assert len({row["scenario"]["map_id"] for row in outcomes}) == 2
    for row in outcomes:
        validation = row["audit"]["validation"]
        assert validation["format_valid"] is True
        assert validation["static_valid"] is None
        assert validation["static_status"] == MIGRATION_STATUS
        assert validation["direct_service_status"] == DIRECT_SERVICE_STATUS


def test_group_state_failure_is_listed_for_every_affected_identity():
    scenarios = [
        _scenario(legacy_map_id="legacy-a", dump_shift=0),
        _scenario(legacy_map_id="legacy-b", dump_shift=-4),
    ]

    def fail_materialization(*_args):
        raise ValueError("no shared spawn")

    outcomes = migrate_loaded_scenarios(
        scenarios,
        environment_protocol=_protocol(),
        env_config="synthetic-env",
        state_materializer=fail_materialization,
        expected_identity_count=2,
        expected_source_group_count=1,
        expected_source_group_size_counts={2: 1},
    )
    assert len(outcomes) == 2
    assert {row["migration_status"] for row in outcomes} == {"failed"}
    assert all("no shared spawn" in row["errors"][0] for row in outcomes)
