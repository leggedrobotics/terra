from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import tools.migrate_b0a_live_geometry as migration
from tools.migrate_b0a_live_geometry import SHORTEST_PATH_MOVES
from tools.migrate_b0a_live_geometry import CONDITION_STATUS
from tools.migrate_b0a_live_geometry import DIRECT_SERVICE_STATUS
from tools.migrate_b0a_live_geometry import MIGRATION_STATUS
from tools.migrate_b0a_live_geometry import LoadedLegacyScenario
from tools.migrate_b0a_live_geometry import MaterializedState
from tools.migrate_b0a_live_geometry import VerifiedInputIntegrity
from tools.migrate_b0a_live_geometry import derive_content_ids
from tools.migrate_b0a_live_geometry import derive_reward_treatment
from tools.migrate_b0a_live_geometry import migrate_loaded_scenarios
from tools.migrate_b0a_live_geometry import normalize_factor_vector
from tools.migrate_b0a_live_geometry import recompute_affordable_audit
from tools.migrate_b0a_live_geometry import reward_contract_sha256
from tools.migrate_b0a_live_geometry import sha256_file
from tools.migrate_b0a_live_geometry import validate_checkout_state
from tools.migrate_b0a_live_geometry import verify_input_integrity

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
        "schema": "terra_agent_state_v2",
        "width": 7,
        "height": 11,
        "max_agents": 4,
        "num_agents": 1,
        "current_agent": 0,
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
            "carry_relocation_credit": [0.0, 0.0, 0.0, 0.0],
        },
    }


def _protocol() -> dict:
    return {
        "environment_protocol_sha256": "protocol-hash",
        "accepted_dump_contract": "exact_visible_dump_v1",
        "map": {
            "edge_length_px": 64,
            "edge_length_m": EDGE_LENGTH_M,
            "tile_size_m_derived_float64": TILE_SIZE_M,
        },
        "episode": {
            "rewards_type": "DENSE",
            "rewards_sha256": "reward-hash",
            "apply_trench_rewards": False,
            "trench_shaping": {
                "alignment_coefficient": 0.0,
                "distance_coefficient": 0.0,
                "cabin_alignment_coefficient": 0.0,
            },
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
    contract_sha256 = reward_contract_sha256(_protocol())
    baseline_treatment = derive_reward_treatment(
        scenario_id=baseline["scenario_id"],
        reward_distance_sha256=baseline["reward_distance_sha256"],
        frozen_reward_contract_sha256=contract_sha256,
    )
    changed_treatment = derive_reward_treatment(
        scenario_id=reward_ids["scenario_id"],
        reward_distance_sha256=reward_ids["reward_distance_sha256"],
        frozen_reward_contract_sha256=contract_sha256,
    )
    assert changed_treatment["treatment_id"] != baseline_treatment["treatment_id"]

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
        assert row["scenario"]["schema"] == "terra_b0a_design_input_scenario_v1"
        treatment = row["scenario"]["reward_treatment"]
        assert treatment["treatment_id"].startswith("treatment:sha256:")
        validation = row["audit"]["validation"]
        assert validation["migration_record_valid"] is True
        assert validation["benchmark_format_valid"] is False
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


def _write_integrity_tree(root, *, wrong_provenance_identity=False):
    identity_path = root / "identities.jsonl"
    source_registry_path = root / "source_registry.jsonl"
    identity_path.write_text('{"map_id":"one"}\n')
    source_registry_path.write_text('{"source_id":"one"}\n')
    identity_sha256 = sha256_file(identity_path)
    provenance = {
        "identity_manifest_sha256": (
            "0" * 64 if wrong_provenance_identity else identity_sha256
        ),
        "source_registry_sha256": sha256_file(source_registry_path),
    }
    (root / "provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
    (root / "validation.json").write_text('{"status":"passed"}\n')
    paths = [
        identity_path,
        root / "provenance.json",
        source_registry_path,
        root / "validation.json",
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths)
    ]
    manifest = root / "files.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return sha256_file(manifest)


def test_input_integrity_verifies_root_manifest_and_provenance(tmp_path):
    manifest_sha256 = _write_integrity_tree(tmp_path)
    verified = verify_input_integrity(
        tmp_path,
        expected_files_sha256=manifest_sha256,
    )
    receipt = verified.receipt
    assert receipt["files_sha256_sha256"] == manifest_sha256
    assert receipt["verified_file_count"] == 4
    assert verified.verified_relative_paths == frozenset(
        {
            "identities.jsonl",
            "provenance.json",
            "source_registry.jsonl",
            "validation.json",
        }
    )
    assert receipt["identity_manifest_sha256"] == sha256_file(
        tmp_path / "identities.jsonl"
    )


def test_input_integrity_rejects_a_file_changed_after_manifest(tmp_path):
    manifest_sha256 = _write_integrity_tree(tmp_path)
    (tmp_path / "identities.jsonl").write_text('{"map_id":"tampered"}\n')
    with pytest.raises(ValueError, match="checksum mismatch for identities.jsonl"):
        verify_input_integrity(
            tmp_path,
            expected_files_sha256=manifest_sha256,
        )


def test_legacy_target_identity_uses_raw_dtype_and_checks_loader_values():
    raw_target, *_ = _layers()
    loaded_target = raw_target.astype(np.int16)
    identity = {
        "target_identity_sha256": migration.sha256_array(raw_target),
        "dig_identity_sha256": migration.sha256_array(
            (raw_target < 0).astype(np.uint8)
        ),
    }

    migration._validate_legacy_target_identity(
        legacy_map_id="legacy-a",
        identity=identity,
        raw_target=raw_target,
        loaded_target=loaded_target,
    )

    changed_target = loaded_target.copy()
    changed_target[0, 0] = 1
    with pytest.raises(ValueError, match="exact loader changed target values"):
        migration._validate_legacy_target_identity(
            legacy_map_id="legacy-a",
            identity=identity,
            raw_target=raw_target,
            loaded_target=changed_target,
        )


def test_input_integrity_rejects_provenance_cross_hash_mismatch(tmp_path):
    manifest_sha256 = _write_integrity_tree(
        tmp_path,
        wrong_provenance_identity=True,
    )
    with pytest.raises(ValueError, match="identity_manifest_sha256"):
        verify_input_integrity(
            tmp_path,
            expected_files_sha256=manifest_sha256,
        )


def test_shortest_path_neighbour_set_has_eight_unique_edges():
    offsets = [(row, column) for row, column, _cost in SHORTEST_PATH_MOVES]
    assert len(offsets) == len(set(offsets)) == 8
    assert offsets.count((1, -1)) == 1


def test_real_migration_requires_matching_clean_checkout():
    validate_checkout_state(
        requested_revision="abc",
        head_revision="abc",
        porcelain_status="",
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_checkout_state(
            requested_revision="requested",
            head_revision="actual",
            porcelain_status="",
        )
    with pytest.raises(ValueError, match="dirty"):
        validate_checkout_state(
            requested_revision="abc",
            head_revision="abc",
            porcelain_status="?? untracked\n",
        )


def test_summary_receipts_already_written_migration_validation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(migration, "verify_clean_checkout", lambda *_args: None)
    monkeypatch.setattr(
        migration,
        "verify_input_integrity",
        lambda _root: VerifiedInputIntegrity(
            receipt={
                "files_sha256_sha256": "f" * 64,
                "verified_file_count": 4,
            },
            verified_relative_paths=frozenset(),
        ),
    )
    monkeypatch.setattr(
        migration,
        "frozen_benchmark_protocol",
        lambda: ("env", {"env_config_sha256": "e" * 64}),
    )
    monkeypatch.setattr(
        migration,
        "frozen_environment_protocol",
        lambda revision: {
            "terra_revision": revision,
            "environment_protocol_sha256": "p" * 64,
        },
    )
    monkeypatch.setattr(
        migration,
        "load_legacy_scenarios",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        migration,
        "migrate_loaded_scenarios",
        lambda *_args, **_kwargs: [],
    )

    output = tmp_path / "output"
    summary = migration.run_migration(
        input_root=tmp_path,
        output=output,
        terra_revision="abc",
    )
    validation_path = output / "migration_validation.jsonl"
    assert sorted(path.name for path in output.iterdir()) == [
        "migration_summary.json",
        "migration_validation.jsonl",
    ]
    assert summary["output"]["migration_validation_sha256"] == sha256_file(
        validation_path
    )
    assert summary["output"]["migration_validation_record_count"] == 0
