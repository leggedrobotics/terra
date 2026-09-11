"""Frozen TerraMap-Bench tracked-excavator protocol derived from live Terra."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from terra.actions import TrackedAction
from terra.actions import TrackedActionType
from terra.benchmark_state import SCHEMA as AGENT_STATE_SCHEMA
from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import RewardStage
from terra.config import Rewards
from terra.config import RewardsType
from terra.state import CORRECTED_DENSE_CONTRACT

BENCHMARK_RELEASE_ID = "terramap-bench-v1.0.2"
BENCHMARK_MAP_SIZE = 64
BENCHMARK_MAX_STEPS = 450
ENVIRONMENT_PROTOCOL_SCHEMA = "terra_environment_protocol_v1"
ENVIRONMENT_PROTOCOL_SCHEMA_VERSION = 1
BENCHMARK_JAX_DEFAULT_PRNG_IMPL = "threefry2x32"
BENCHMARK_JAX_THREEFRY_PARTITIONABLE = True

# This is the derived discrete footprint frozen by the v1 benchmark, not a
# second copy of the excavator's physical dimensions.
_FROZEN_TRACKED_FOOTPRINT = (7, 11)
FROZEN_ENV_CONFIG_SHA256 = (
    "f720809566bf73d8a740124ae46c487af497ecda7de6d3f722a06b62c79dd0a3"
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: _jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} into the protocol.")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _odd_tile_extent(length_m: float, tile_size_m: float) -> int:
    extent = round(length_m / tile_size_m)
    return extent if extent % 2 else extent + 1


def frozen_benchmark_protocol() -> tuple[EnvConfig, dict[str, Any]]:
    """Return the sole v1 benchmark EnvConfig and its canonical receipt."""

    base = EnvConfig()
    live = BatchConfig()
    if live.action_type is not TrackedAction:
        raise RuntimeError("The live Terra action type is no longer tracked.")
    if base.maps.edge_length_m != live.maps.edge_length_m:
        raise RuntimeError("EnvConfig and BatchConfig map scales disagree.")
    if (
        base.agent.angles_base,
        base.agent.angles_cabin,
    ) != (
        live.agent.angles_base,
        live.agent.angles_cabin,
    ):
        raise RuntimeError("EnvConfig and BatchConfig orientation counts disagree.")

    edge_length_m = float(live.maps.edge_length_m)
    tile_size_m = edge_length_m / BENCHMARK_MAP_SIZE
    agent_width = _odd_tile_extent(live.agent.dimensions.HEIGHT, tile_size_m)
    agent_height = _odd_tile_extent(live.agent.dimensions.WIDTH, tile_size_m)
    if (agent_width, agent_height) != _FROZEN_TRACKED_FOOTPRINT:
        raise RuntimeError(
            "Frozen tracked-excavator footprint changed: "
            f"{agent_width} x {agent_height}; expected "
            f"{_FROZEN_TRACKED_FOOTPRINT[0]} x {_FROZEN_TRACKED_FOOTPRINT[1]}."
        )

    tracked_type = int(np.asarray(TrackedAction().type).reshape(-1)[0])
    if base.agent_types != (tracked_type,):
        raise RuntimeError(
            "The live EnvConfig no longer describes one tracked excavator."
        )

    config = base._replace(
        tile_size=np.float32(tile_size_m),
        agent=base.agent._replace(
            angles_base=live.agent.angles_base,
            angles_cabin=live.agent.angles_cabin,
            width=agent_width,
            height=agent_height,
        ),
        maps=base.maps._replace(
            edge_length_m=live.maps.edge_length_m,
            edge_length_px=BENCHMARK_MAP_SIZE,
        ),
        rewards=Rewards.dense(),
        apply_trench_rewards=False,
        max_steps_in_episode=BENCHMARK_MAX_STEPS,
        agent_types=(tracked_type,),
        action_types=(tracked_type,),
    )
    payload = _jsonable(config)
    if config.reward_stage != RewardStage.DENSE_SKILL:
        raise RuntimeError("The v1 benchmark must use dense-skill rewards.")
    # RewardStage was appended after the v1 protocol was frozen. Its default
    # preserves the original dense behavior, so keep the existing v1 payload
    # and hashes byte-compatible. The reward-v2.1 timing selector is appended
    # for the same reason: it is inert under dense-skill rewards, and its
    # default is the frozen reward_v2 anyway. reset_tier=0 likewise selects the
    # canonical full reset and is inert for the v1 benchmark.
    payload.pop("reward_stage")
    payload.pop("terminal_reward_mix")
    payload.pop("reward_v2_timing_variant")
    payload.pop("reset_tier")
    # The opt-in trench-alignment treatment was appended later. Its disabled
    # default and inert tolerance fields must not mutate the frozen v1 receipt.
    payload.pop("enforce_trench_dig_alignment")
    payload.pop("trench_dig_yaw_tolerance_rad")
    payload.pop("trench_dig_standoff_min_m")
    payload.pop("trench_dig_standoff_max_m")
    # The v2 gate-semantics selector was appended later still. It is inert
    # whenever the gate itself is disabled, which the v1 benchmark requires.
    payload.pop("trench_dig_standoff_enforced")
    payload.pop("trench_dig_max_offset_m")
    # Later foundation behavior fields preserve v1 only at their inert
    # defaults. Exclude them from this base receipt, not from EnvConfig or
    # checkpoint/evaluation treatment metadata. A changed default must fail.
    for name, default in {
        "lateral_dig_cost": 0.0,
        "base_travel_cost": 0.0,
        "base_turn_cost": 0.0,
        "executable_dig_observation": False,
    }.items():
        if payload[name] != default:
            raise RuntimeError(
                "The v1 benchmark requires inert foundation behavior defaults: "
                f"{name}={payload[name]!r}; expected {default!r}."
            )
        payload.pop(name)
    config_sha256 = canonical_json_sha256(payload)
    if config_sha256 != FROZEN_ENV_CONFIG_SHA256:
        raise RuntimeError(
            "Frozen benchmark EnvConfig changed without a release bump: "
            f"{config_sha256} != {FROZEN_ENV_CONFIG_SHA256}."
        )

    return config, {
        "env_config": payload,
        "env_config_sha256": config_sha256,
        "edge_length_m": edge_length_m,
        "edge_length_px": BENCHMARK_MAP_SIZE,
        "tile_size_m_derived_float64": tile_size_m,
        "tile_size_m_runtime_float32": float(np.float32(tile_size_m)),
        "agent_width_tiles": agent_width,
        "agent_height_tiles": agent_height,
        "max_steps_in_episode": BENCHMARK_MAX_STEPS,
    }


def frozen_environment_protocol(terra_revision: str) -> dict[str, Any]:
    """Return the broader executable environment contract and its hash."""

    if not isinstance(terra_revision, str) or not terra_revision:
        raise ValueError("terra_revision must be a non-empty string.")
    if terra_revision != terra_revision.strip():
        raise ValueError("terra_revision must not contain surrounding whitespace.")

    config, env_receipt = frozen_benchmark_protocol()
    live = BatchConfig()
    max_half_extent_tiles = max(config.agent.width, config.agent.height) / 2.0
    workspace_min_m = 0.5 + config.tile_size * max_half_extent_tiles
    workspace_max_m = workspace_min_m + config.agent.dig_radius_tiles * config.tile_size
    payload = {
        "schema": ENVIRONMENT_PROTOCOL_SCHEMA,
        "schema_version": ENVIRONMENT_PROTOCOL_SCHEMA_VERSION,
        "release_id": BENCHMARK_RELEASE_ID,
        "terra_revision": terra_revision,
        "env_config_sha256": env_receipt["env_config_sha256"],
        "reset_prng": {
            "jax_default_prng_impl": BENCHMARK_JAX_DEFAULT_PRNG_IMPL,
            "jax_threefry_partitionable": (
                BENCHMARK_JAX_THREEFRY_PARTITIONABLE
            ),
        },
        "map": {
            "edge_length_px": env_receipt["edge_length_px"],
            "edge_length_m": env_receipt["edge_length_m"],
            "tile_size_m_derived_float64": env_receipt["tile_size_m_derived_float64"],
            "tile_size_m_runtime_float32": env_receipt["tile_size_m_runtime_float32"],
        },
        "tracked_excavator": {
            "physical_dimensions_m": {
                "long_side": live.agent.dimensions.WIDTH,
                "short_side": live.agent.dimensions.HEIGHT,
            },
            "footprint_tiles": {
                "width": env_receipt["agent_width_tiles"],
                "height": env_receipt["agent_height_tiles"],
            },
            "move_tiles": config.agent.move_tiles,
            "dig_radius_tiles": config.agent.dig_radius_tiles,
            "dig_depth": config.agent.dig_depth,
            "base_orientations": config.agent.angles_base,
            "cabin_orientations": config.agent.angles_cabin,
            "action_enum": {
                action.name: int(action.value) for action in TrackedActionType
            },
            "agent_state_schema": AGENT_STATE_SCHEMA,
            "workspace": {
                "radial_min_m": float(workspace_min_m),
                "radial_max_m": float(workspace_max_m),
                "radial_min_tiles": float(workspace_min_m / config.tile_size),
                "radial_max_tiles": float(workspace_max_m / config.tile_size),
                "cabin_half_angle_degrees": float(360.0 / config.agent.angles_cabin),
                "mask_implementation": "terra_state_exact_dig_dump_cone_v1",
            },
        },
        "accepted_dump_contract": CORRECTED_DENSE_CONTRACT,
        "episode": {
            "max_steps_in_episode": config.max_steps_in_episode,
            "rewards_type": RewardsType.DENSE.name,
            "rewards_sha256": canonical_json_sha256(_jsonable(config.rewards)),
            "apply_trench_rewards": config.apply_trench_rewards,
            "trench_shaping": {
                "alignment_coefficient": config.alignment_coefficient,
                "distance_coefficient": config.distance_coefficient,
                "cabin_alignment_coefficient": config.cabin_alignment_coefficient,
            },
        },
    }
    return {
        **payload,
        "environment_protocol_sha256": canonical_json_sha256(payload),
    }
