from __future__ import annotations

import numpy as np
import pytest

import terra.benchmark_protocol as protocol
from terra.actions import TrackedAction
from terra.actions import TrackedActionType
from terra.benchmark_state import SCHEMA as AGENT_STATE_SCHEMA
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import Rewards
from terra.state import CORRECTED_DENSE_CONTRACT


def test_protocol_is_derived_from_live_tracked_geometry():
    config, receipt = protocol.frozen_benchmark_protocol()
    live = BatchConfig()

    assert receipt["edge_length_px"] == protocol.BENCHMARK_MAP_SIZE
    assert receipt["tile_size_m_derived_float64"] == (
        live.maps.edge_length_m / protocol.BENCHMARK_MAP_SIZE
    )
    assert receipt["tile_size_m_runtime_float32"] == float(config.tile_size)
    assert config.maps.edge_length_m / config.maps.edge_length_px == pytest.approx(
        receipt["tile_size_m_derived_float64"]
    )
    assert (config.agent.width, config.agent.height) == (7, 11)
    assert (receipt["agent_width_tiles"], receipt["agent_height_tiles"]) == (7, 11)
    assert config.agent.move_tiles == EnvConfig().agent.move_tiles == 5
    assert config.agent.dig_radius_tiles == EnvConfig().agent.dig_radius_tiles == 5
    assert config.agent.dig_depth == EnvConfig().agent.dig_depth == 1
    assert config.agent.angles_base == live.agent.angles_base == 12
    assert config.agent.angles_cabin == live.agent.angles_cabin == 12
    tracked_type = int(np.asarray(TrackedAction().type).reshape(-1)[0])
    assert config.agent_types == (tracked_type,)
    assert config.action_types == (tracked_type,)


def test_protocol_freezes_horizon_and_dense_reward_without_trench_absolute_shaping():
    config, receipt = protocol.frozen_benchmark_protocol()

    assert config.max_steps_in_episode == protocol.BENCHMARK_MAX_STEPS == 450
    assert receipt["max_steps_in_episode"] == 450
    assert config.rewards == Rewards.dense()
    assert config.apply_trench_rewards is False


def test_protocol_hash_is_deterministic_and_matches_the_existing_probe_contract():
    first_config, first_receipt = protocol.frozen_benchmark_protocol()
    second_config, second_receipt = protocol.frozen_benchmark_protocol()

    assert first_config == second_config
    assert first_receipt == second_receipt
    assert (
        first_receipt["env_config_sha256"]
        == protocol.FROZEN_ENV_CONFIG_SHA256
        == "f720809566bf73d8a740124ae46c487af497ecda7de6d3f722a06b62c79dd0a3"
    )
    assert protocol.canonical_json_sha256(first_receipt["env_config"]) == (
        first_receipt["env_config_sha256"]
    )
    defaults = {
        "lateral_dig_cost": 0.0,
        "base_travel_cost": 0.0,
        "base_turn_cost": 0.0,
        "executable_dig_observation": False,
    }
    assert {name: getattr(first_config, name) for name in defaults} == defaults
    assert not defaults.keys() & first_receipt["env_config"].keys()

    # Compatibility is local to the frozen base receipt. Serializing an
    # actual treatment must retain every non-default field.
    treatment = {
        "lateral_dig_cost": 0.6,
        "base_travel_cost": 0.07,
        "base_turn_cost": 0.11,
        "executable_dig_observation": True,
    }
    serialized = protocol._jsonable(first_config._replace(**treatment))
    assert {name: serialized[name] for name in treatment} == treatment


@pytest.mark.parametrize("name,value", [
    ("lateral_dig_cost", 0.6),
    ("base_travel_cost", 0.07),
    ("base_turn_cost", 0.11),
    ("executable_dig_observation", True),
])
def test_frozen_protocol_does_not_hide_nondefault_foundation_behavior(
    monkeypatch, name, value,
):
    changed = EnvConfig()._replace(**{name: value})
    monkeypatch.setattr(protocol, "EnvConfig", lambda: changed)

    with pytest.raises(RuntimeError, match=f"inert foundation behavior defaults: {name}"):
        protocol.frozen_environment_protocol("terra-commit-a")


def test_protocol_fails_loudly_if_live_footprint_changes(monkeypatch):
    live = BatchConfig()
    changed_dimensions = live.agent.dimensions._replace(WIDTH=4.0, HEIGHT=2.0)
    changed = live._replace(agent=live.agent._replace(dimensions=changed_dimensions))
    monkeypatch.setattr(protocol, "BatchConfig", lambda: changed)

    with pytest.raises(RuntimeError, match="footprint changed"):
        protocol.frozen_benchmark_protocol()


def test_protocol_fails_loudly_if_live_config_changes_without_release_bump(
    monkeypatch,
):
    live = EnvConfig()
    changed = live._replace(agent=live.agent._replace(move_tiles=6))
    monkeypatch.setattr(protocol, "EnvConfig", lambda: changed)

    with pytest.raises(RuntimeError, match="without a release bump"):
        protocol.frozen_benchmark_protocol()


def test_environment_protocol_hash_is_deterministic_and_revision_sensitive():
    first = protocol.frozen_environment_protocol("terra-commit-a")
    repeated = protocol.frozen_environment_protocol("terra-commit-a")
    changed_revision = protocol.frozen_environment_protocol("terra-commit-b")

    assert first == repeated
    assert (
        first["environment_protocol_sha256"]
        != changed_revision["environment_protocol_sha256"]
    )
    unhashed_payload = {
        key: value
        for key, value in first.items()
        if key != "environment_protocol_sha256"
    }
    assert first["environment_protocol_sha256"] == protocol.canonical_json_sha256(
        unhashed_payload
    )


def test_environment_protocol_receipts_the_executable_action_and_config_contract():
    config, env_receipt = protocol.frozen_benchmark_protocol()
    receipt = protocol.frozen_environment_protocol("terra-commit-a")

    assert receipt["schema"] == "terra_environment_protocol_v1"
    assert receipt["schema_version"] == 1
    assert receipt["release_id"] == protocol.BENCHMARK_RELEASE_ID
    assert receipt["env_config_sha256"] == env_receipt["env_config_sha256"]
    assert receipt["reset_prng"] == {
        "jax_default_prng_impl": "threefry2x32",
        "jax_threefry_partitionable": True,
    }
    assert receipt["map"] == {
        "edge_length_px": 64,
        "edge_length_m": config.maps.edge_length_m,
        "tile_size_m_derived_float64": (
            config.maps.edge_length_m / config.maps.edge_length_px
        ),
        "tile_size_m_runtime_float32": float(config.tile_size),
    }

    tracked = receipt["tracked_excavator"]
    assert tracked["footprint_tiles"] == {"width": 7, "height": 11}
    assert tracked["move_tiles"] == config.agent.move_tiles
    assert tracked["dig_radius_tiles"] == config.agent.dig_radius_tiles
    assert tracked["dig_depth"] == config.agent.dig_depth
    assert tracked["base_orientations"] == config.agent.angles_base
    assert tracked["cabin_orientations"] == config.agent.angles_cabin
    assert tracked["action_enum"] == {
        action.name: int(action.value) for action in TrackedActionType
    }
    assert tracked["agent_state_schema"] == AGENT_STATE_SCHEMA
    assert tracked["physical_dimensions_m"] == {
        "long_side": BatchConfig().agent.dimensions.WIDTH,
        "short_side": BatchConfig().agent.dimensions.HEIGHT,
    }
    assert tracked["workspace"] == {
        "radial_min_m": pytest.approx(3.6428571428556877),
        "radial_max_m": pytest.approx(6.499999999996312),
        "radial_min_tiles": pytest.approx(6.375),
        "radial_max_tiles": pytest.approx(11.375),
        "cabin_half_angle_degrees": pytest.approx(30.0),
        "mask_implementation": "terra_state_exact_dig_dump_cone_v1",
    }

    assert receipt["accepted_dump_contract"] == CORRECTED_DENSE_CONTRACT
    assert receipt["episode"]["max_steps_in_episode"] == 450
    assert receipt["episode"]["rewards_type"] == "DENSE"
    assert receipt["episode"]["apply_trench_rewards"] is False
    assert receipt["episode"]["rewards_sha256"] == protocol.canonical_json_sha256(
        protocol._jsonable(config.rewards)
    )
    assert receipt["episode"]["trench_shaping"] == {
        "alignment_coefficient": config.alignment_coefficient,
        "distance_coefficient": config.distance_coefficient,
        "cabin_alignment_coefficient": config.cabin_alignment_coefficient,
    }


@pytest.mark.parametrize("revision", ["", " ", "\n"])
def test_environment_protocol_rejects_empty_or_ambiguous_revision(revision):
    with pytest.raises(ValueError, match="terra_revision"):
        protocol.frozen_environment_protocol(revision)
