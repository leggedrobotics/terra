"""Canonical host-side codec for benchmark initial Agent trees."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from terra.agent import Agent
from terra.agent import AgentState
from terra.agent import TRACKED_MIN_BORDER_DISTANCE_TILES
from terra.config import EnvConfig
from terra.utils import compute_polygon_mask
from terra.utils import get_agent_corners

SCHEMA = "terra_agent_state_v2"
MAX_AGENTS = 4
INITIAL_STATE_SEED_SCHEMA = "terra_initial_state_seed_v1"

_AGENT_FIELDS = (
    "width",
    "height",
    "agent_states",
    "agent_active",
    "num_agents",
    "current_agent",
)
_AGENT_STATE_FIELDS = (
    "pos_base",
    "angle_base",
    "angle_cabin",
    "wheel_angle",
    "loaded",
    "agent_type",
    "action_type",
    "shovel_lifted",
    "carry_relocation_credit",
)

if Agent._fields != _AGENT_FIELDS or AgentState._fields != _AGENT_STATE_FIELDS:
    raise RuntimeError(
        "terra_agent_state_v2 no longer covers the complete Agent tree; "
        "define a new state schema before changing the codec."
    )

_ARRAY_SCHEMA = (
    ("width", np.dtype("<i4"), ()),
    ("height", np.dtype("<i4"), ()),
    ("max_agents", np.dtype("<i4"), ()),
    ("num_agents", np.dtype("<i4"), ()),
    ("current_agent", np.dtype("<i4"), ()),
    ("agent_active", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.pos_base", np.dtype("<i2"), (MAX_AGENTS, 2)),
    ("agent_states.angle_base", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.angle_cabin", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.wheel_angle", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.loaded", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.agent_type", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.action_type", np.dtype("i1"), (MAX_AGENTS,)),
    ("agent_states.shovel_lifted", np.dtype("i1"), (MAX_AGENTS,)),
    (
        "agent_states.carry_relocation_credit",
        np.dtype("<f4"),
        (MAX_AGENTS,),
    ),
)

_RECORD_KEYS = (
    "schema",
    "width",
    "height",
    "max_agents",
    "num_agents",
    "current_agent",
    "agent_active",
    "agent_states",
)


def _canonical_array(
    path: str,
    value: Any,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    try:
        raw = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path} must have shape {shape} and contain concrete numeric values."
        ) from error
    if raw.shape != shape:
        raise ValueError(f"{path} must have shape {shape}, got {raw.shape}.")

    if dtype.kind == "b":
        if raw.dtype.kind == "b":
            pass
        elif raw.dtype.kind in "iu" and np.all((raw == 0) | (raw == 1)):
            pass
        else:
            raise ValueError(f"{path} must contain booleans or integer 0/1 values.")
    elif dtype.kind in "iu":
        if raw.dtype.kind not in "iub":
            raise ValueError(f"{path} must contain integers.")
        info = np.iinfo(dtype)
        if np.any(raw < info.min) or np.any(raw > info.max):
            raise ValueError(f"{path} contains a value outside {dtype.name}.")
    elif dtype.kind == "f":
        if raw.dtype.kind not in "iuf":
            raise ValueError(f"{path} must contain finite numbers.")
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"{path} must contain finite numbers.")
        limit = np.finfo(dtype).max
        if np.any(raw < -limit) or np.any(raw > limit):
            raise ValueError(f"{path} contains a value outside {dtype.name}.")
    else:
        raise RuntimeError(f"Unsupported canonical dtype for {path}: {dtype}.")

    canonical = np.asarray(raw, dtype=dtype, order="C")
    if not canonical.flags.c_contiguous:
        canonical = np.array(canonical, dtype=dtype, order="C", copy=True)
    return canonical


def _agent_arrays(agent: Agent) -> dict[str, np.ndarray]:
    if not isinstance(agent, Agent):
        raise TypeError(f"Expected Agent, got {type(agent).__name__}.")
    if agent.agent_states is None or len(agent.agent_states) != MAX_AGENTS:
        raise ValueError(f"agent_states must contain exactly {MAX_AGENTS} slots.")
    if agent.agent_active is None:
        raise ValueError("agent_active is required.")
    if any(not isinstance(state, AgentState) for state in agent.agent_states):
        raise TypeError("Every agent_states slot must be an AgentState.")

    raw: dict[str, Any] = {
        "width": agent.width,
        "height": agent.height,
        "max_agents": MAX_AGENTS,
        "num_agents": agent.num_agents,
        "current_agent": agent.current_agent,
        "agent_active": agent.agent_active,
        "agent_states.pos_base": np.stack(
            [np.asarray(jax.device_get(state.pos_base)) for state in agent.agent_states]
        ),
    }
    for field in _AGENT_STATE_FIELDS[1:8]:
        raw[f"agent_states.{field}"] = np.stack(
            [
                np.asarray(jax.device_get(getattr(state, field))).reshape(-1)[0]
                for state in agent.agent_states
            ]
        )
    for field in _AGENT_STATE_FIELDS[8:]:
        raw[f"agent_states.{field}"] = np.stack(
            [
                np.asarray(jax.device_get(getattr(state, field))).reshape(())
                for state in agent.agent_states
            ]
        )

    return {
        path: _canonical_array(path, raw[path], dtype, shape)
        for path, dtype, shape in _ARRAY_SCHEMA
    }


def agent_to_record(agent: Agent) -> dict[str, Any]:
    """Convert a complete four-slot Agent tree to the v2 JSON record."""
    arrays = _agent_arrays(agent)
    state_fields = {
        field: arrays[f"agent_states.{field}"].tolist() for field in _AGENT_STATE_FIELDS
    }
    return {
        "schema": SCHEMA,
        "width": int(arrays["width"]),
        "height": int(arrays["height"]),
        "max_agents": int(arrays["max_agents"]),
        "num_agents": int(arrays["num_agents"]),
        "current_agent": int(arrays["current_agent"]),
        "agent_active": arrays["agent_active"].astype(bool).tolist(),
        "agent_states": state_fields,
    }


def _require_exact_keys(
    record: Mapping[str, Any],
    expected: tuple[str, ...],
    path: str,
) -> None:
    missing = sorted(set(expected) - set(record))
    extra = sorted(set(record) - set(expected))
    if missing or extra:
        raise ValueError(f"{path} keys differ: missing={missing}, extra={extra}.")


def agent_from_record(record: Mapping[str, Any]) -> Agent:
    """Decode a v2 JSON record into the exact dtypes consumed by reset."""
    if not isinstance(record, Mapping):
        raise TypeError("Agent state record must be a mapping.")
    _require_exact_keys(record, _RECORD_KEYS, "initial_agent_state")
    if record["schema"] != SCHEMA:
        raise ValueError(f"Unsupported agent-state schema: {record['schema']!r}.")
    if record["max_agents"] != MAX_AGENTS:
        raise ValueError(f"max_agents must be {MAX_AGENTS}.")

    state_record = record["agent_states"]
    if not isinstance(state_record, Mapping):
        raise TypeError("initial_agent_state.agent_states must be a mapping.")
    _require_exact_keys(
        state_record,
        _AGENT_STATE_FIELDS,
        "initial_agent_state.agent_states",
    )

    raw = {
        "width": record["width"],
        "height": record["height"],
        "max_agents": record["max_agents"],
        "num_agents": record["num_agents"],
        "current_agent": record["current_agent"],
        "agent_active": record["agent_active"],
        **{
            f"agent_states.{field}": state_record[field]
            for field in _AGENT_STATE_FIELDS
        },
    }
    arrays = {
        path: _canonical_array(path, raw[path], dtype, shape)
        for path, dtype, shape in _ARRAY_SCHEMA
    }

    states = []
    for index in range(MAX_AGENTS):
        states.append(
            AgentState(
                pos_base=jnp.asarray(
                    arrays["agent_states.pos_base"][index], dtype=jnp.int16
                ),
                angle_base=jnp.asarray(
                    arrays["agent_states.angle_base"][index : index + 1],
                    dtype=jnp.int8,
                ),
                angle_cabin=jnp.asarray(
                    arrays["agent_states.angle_cabin"][index : index + 1],
                    dtype=jnp.int8,
                ),
                wheel_angle=jnp.asarray(
                    arrays["agent_states.wheel_angle"][index : index + 1],
                    dtype=jnp.int8,
                ),
                loaded=jnp.asarray(
                    arrays["agent_states.loaded"][index : index + 1],
                    dtype=jnp.int8,
                ),
                agent_type=jnp.asarray(
                    arrays["agent_states.agent_type"][index : index + 1],
                    dtype=jnp.int8,
                ),
                action_type=jnp.asarray(
                    arrays["agent_states.action_type"][index : index + 1],
                    dtype=jnp.int8,
                ),
                shovel_lifted=jnp.asarray(
                    arrays["agent_states.shovel_lifted"][index : index + 1],
                    dtype=jnp.int8,
                ),
                carry_relocation_credit=jnp.asarray(
                    arrays["agent_states.carry_relocation_credit"][index],
                    dtype=jnp.float32,
                ),
            )
        )

    return Agent(
        width=jnp.asarray(arrays["width"], dtype=jnp.int32),
        height=jnp.asarray(arrays["height"], dtype=jnp.int32),
        agent_states=tuple(states),
        agent_active=jnp.asarray(arrays["agent_active"], dtype=jnp.int8),
        num_agents=jnp.asarray(arrays["num_agents"], dtype=jnp.int32),
        current_agent=jnp.asarray(arrays["current_agent"], dtype=jnp.int32),
    )


def canonical_agent_bytes(agent: Agent) -> bytes:
    """Encode every Agent leaf with explicit path, dtype, rank, and shape."""
    arrays = _agent_arrays(agent)
    encoded = bytearray()
    schema_bytes = SCHEMA.encode("ascii")
    encoded.extend(struct.pack("<H", len(schema_bytes)))
    encoded.extend(schema_bytes)
    for path, dtype, _ in _ARRAY_SCHEMA:
        array = arrays[path]
        path_bytes = path.encode("ascii")
        dtype_bytes = dtype.str.encode("ascii")
        encoded.extend(struct.pack("<H", len(path_bytes)))
        encoded.extend(path_bytes)
        encoded.extend(struct.pack("<B", len(dtype_bytes)))
        encoded.extend(dtype_bytes)
        encoded.extend(struct.pack("<B", array.ndim))
        for dimension in array.shape:
            encoded.extend(struct.pack("<I", dimension))
        payload = array.tobytes(order="C")
        encoded.extend(struct.pack("<Q", len(payload)))
        encoded.extend(payload)
    return bytes(encoded)


def agent_state_sha256(agent: Agent) -> str:
    """Return the portable terra_agent_state_v2 digest."""
    return hashlib.sha256(canonical_agent_bytes(agent)).hexdigest()


def derive_initial_state_seed(
    release_id: str,
    split: str,
    source_group_id: str,
    state_index: int,
) -> tuple[int, str]:
    """Derive R-25's big-endian uint32 seed and full namespace digest."""
    namespace_fields = {
        "release_id": release_id,
        "split": split,
        "source_group_id": source_group_id,
    }
    for name, value in namespace_fields.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string.")
        if "\0" in value:
            raise ValueError(f"{name} cannot contain a NUL delimiter.")
    if isinstance(state_index, bool) or not isinstance(state_index, int):
        raise TypeError("state_index must be an integer.")
    if not 0 <= state_index <= np.iinfo(np.uint32).max:
        raise ValueError("state_index must fit uint32.")

    payload = (
        f"{INITIAL_STATE_SEED_SCHEMA}\0"
        f"{release_id}\0{split}\0{source_group_id}\0{state_index}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return seed, digest.hex()


def sample_benchmark_initial_agent(
    *,
    release_id: str,
    split: str,
    source_group_id: str,
    state_index: int,
    env_cfg: EnvConfig,
    padding_mask: Any,
    action_map: Any,
    dumpability_mask: Any,
) -> tuple[Agent, dict[str, Any]]:
    """Sample one shared tracked-agent state from an intersected spawn contract."""
    seed, seed_digest = derive_initial_state_seed(
        release_id,
        split,
        source_group_id,
        state_index,
    )
    padding = jnp.asarray(padding_mask)
    actions = jnp.asarray(action_map)
    dumpability = jnp.asarray(dumpability_mask)
    if (
        padding.ndim != 2
        or actions.shape != padding.shape
        or dumpability.shape != padding.shape
    ):
        raise ValueError(
            "padding_mask, action_map, and dumpability_mask must share one 2D shape."
        )

    max_traversable_x = jnp.sum(padding[:, 0] == 0, dtype=jnp.int32)
    max_traversable_y = jnp.sum(padding[0] == 0, dtype=jnp.int32)
    agent, _ = Agent.new(
        jax.random.PRNGKey(seed),
        env_cfg,
        max_traversable_x,
        max_traversable_y,
        padding,
        actions,
        dumpability_map=dumpability,
        agent_types=(0,),
        action_types=(0,),
    )
    agent = jax.tree_util.tree_map(jnp.asarray, agent)
    validate_benchmark_initial_agent(
        agent,
        env_cfg=env_cfg,
        padding_mask=padding,
        action_map=actions,
        dumpability_mask=dumpability,
    )
    receipt = {
        "schema": INITIAL_STATE_SEED_SCHEMA,
        "release_id": release_id,
        "split": split,
        "source_group_id": source_group_id,
        "state_index": state_index,
        "seed_byte_order": "big",
        "seed_uint32": seed,
        "seed_digest_sha256": seed_digest,
        "initial_agent_state_sha256": agent_state_sha256(agent),
    }
    return agent, receipt


def validate_benchmark_initial_agent(
    agent: Agent,
    *,
    env_cfg: EnvConfig,
    padding_mask: Any,
    action_map: Any,
    dumpability_mask: Any,
) -> None:
    """Validate the one-active-tracked-excavator v2 benchmark reset."""
    arrays = _agent_arrays(agent)
    expected_width = int(np.asarray(jax.device_get(env_cfg.agent.width)))
    expected_height = int(np.asarray(jax.device_get(env_cfg.agent.height)))
    if (
        int(arrays["width"]) != expected_width
        or int(arrays["height"]) != expected_height
    ):
        raise ValueError(
            "Agent footprint does not match EnvConfig: "
            f"got {int(arrays['width'])}x{int(arrays['height'])}, "
            f"expected {expected_width}x{expected_height}."
        )
    if int(arrays["num_agents"]) != 1:
        raise ValueError("terra_agent_state_v2 supports exactly one active agent.")
    if not np.array_equal(
        arrays["agent_active"], np.array([True, False, False, False])
    ):
        raise ValueError("agent_active must be [true, false, false, false].")
    if int(arrays["current_agent"]) != 0:
        raise ValueError("current_agent must select the sole active slot 0.")
    active_values = {
        field: arrays[f"agent_states.{field}"][0] for field in _AGENT_STATE_FIELDS[1:]
    }
    if int(active_values["agent_type"]) != 0 or int(active_values["action_type"]) != 0:
        raise ValueError("The v2 benchmark agent must be a tracked excavator.")
    if int(active_values["loaded"]) != 0:
        raise ValueError("A full benchmark reset must start unloaded.")
    if (
        int(active_values["wheel_angle"]) != 0
        or int(active_values["shovel_lifted"]) != 0
    ):
        raise ValueError(
            "Tracked benchmark resets require zero wheel and shovel state."
        )
    if np.any(arrays["agent_states.carry_relocation_credit"] != 0.0):
        raise ValueError("The v2 full reset records zero relocation credit.")

    inactive_fields = ("pos_base",) + _AGENT_STATE_FIELDS[1:]
    for field in inactive_fields:
        if np.any(arrays[f"agent_states.{field}"][1:] != 0):
            raise ValueError(f"Inactive slots require canonical zero {field} bytes.")

    angle_base = int(active_values["angle_base"])
    angle_cabin = int(active_values["angle_cabin"])
    if not 0 <= angle_base < int(env_cfg.agent.angles_base):
        raise ValueError("angle_base is outside the configured orientation range.")
    if not 0 <= angle_cabin < int(env_cfg.agent.angles_cabin):
        raise ValueError("angle_cabin is outside the configured orientation range.")

    padding = np.asarray(jax.device_get(padding_mask))
    actions = np.asarray(jax.device_get(action_map))
    dumpability = np.asarray(jax.device_get(dumpability_mask))
    if (
        padding.ndim != 2
        or actions.shape != padding.shape
        or dumpability.shape != padding.shape
    ):
        raise ValueError(
            "padding_mask, action_map, and dumpability_mask must share one 2D shape."
        )
    edge_length = int(np.asarray(jax.device_get(env_cfg.maps.edge_length_px)))
    if padding.shape != (edge_length, edge_length):
        raise ValueError(
            "Benchmark maps must match the configured square edge length: "
            f"got {padding.shape}, expected {(edge_length, edge_length)}."
        )

    pos_base = arrays["agent_states.pos_base"][0]
    corners = np.asarray(
        get_agent_corners(
            jnp.asarray(pos_base, dtype=jnp.int16),
            jnp.asarray([angle_base], dtype=jnp.int8),
            expected_width,
            expected_height,
            env_cfg.agent.angles_base,
        )
    )
    map_bounds = np.asarray(padding.shape, dtype=np.int32)
    if np.any(corners < 0) or np.any(corners >= map_bounds):
        raise ValueError("Initial excavator footprint extends outside the map.")

    occupancy = np.asarray(
        compute_polygon_mask(
            jnp.asarray(corners),
            padding.shape[0],
            padding.shape[1],
        )
    )
    if np.any(occupancy & (padding != 0)):
        raise ValueError("Initial excavator footprint overlaps an obstacle.")
    if np.any(occupancy & (actions != 0)):
        raise ValueError("Initial excavator footprint overlaps existing soil work.")
    if np.any(occupancy & ~dumpability.astype(bool)):
        raise ValueError("Initial excavator footprint overlaps a non-dumpable tile.")

    border_distance = min(
        int(pos_base[0]),
        int(pos_base[1]),
        padding.shape[0] - 1 - int(pos_base[0]),
        padding.shape[1] - 1 - int(pos_base[1]),
    )
    if border_distance < TRACKED_MIN_BORDER_DISTANCE_TILES:
        raise ValueError(
            "Initial tracked excavator base must be at least "
            f"{TRACKED_MIN_BORDER_DISTANCE_TILES} tiles from the border."
        )
