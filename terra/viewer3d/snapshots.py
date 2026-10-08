"""Detach one Terra timestep into the portable viewer schema, without importing JAX."""

from collections.abc import Mapping
import math

import numpy as np

SCHEMA = "terra.viewer3d.v1"
MAX_GRID_SIZE = 128
MAX_AGENTS = 4
_MISSING = object()
_MAP_FIELDS = {
    "action": "action_map",
    "target": "target_map",
    "padding": "padding_mask",
    "dumpability": "dumpability_mask",
    "dumpability_static": "dumpability_mask_init",
    "interaction": "interaction_mask",
    "traversability": "traversability_mask",
}
_OPTIONAL_MAPS = {"dumpability_static", "interaction", "traversability"}


def _field(obj, name, default=_MISSING):
    value = (
        obj.get(name, default)
        if isinstance(obj, Mapping)
        else getattr(obj, name, default)
    )
    if value is _MISSING:
        raise ValueError(f"Missing timestep field: {name}")
    return value


class _Selection:
    def __init__(self, action_map, env_index):
        shape = np.shape(action_map)
        if len(shape) < 2:
            raise ValueError("action_map must have trailing row and column dimensions")
        self.batch_shape = shape[:-2]
        self.grid_shape = shape[-2:]
        if self.batch_shape and env_index is None:
            raise ValueError("Batched timesteps require an explicit env_index")
        if env_index is None:
            self.index = ()
        elif isinstance(env_index, (int, np.integer)) and not isinstance(
            env_index, (bool, np.bool_)
        ):
            self.index = (int(env_index),)
        elif isinstance(env_index, tuple):
            self.index = env_index
        else:
            raise ValueError("env_index must be an integer or tuple of integers")
        if len(self.index) != len(self.batch_shape):
            raise ValueError(
                f"env_index must select all {len(self.batch_shape)} batch dimensions"
            )
        for index, size in zip(self.index, self.batch_shape):
            if isinstance(index, (bool, np.bool_)) or not isinstance(
                index, (int, np.integer)
            ):
                raise ValueError("env_index must contain integers")
            if not 0 <= index < size:
                raise ValueError(
                    f"env_index {self.index} is outside batch shape {self.batch_shape}"
                )

    def array(self, value, name, shapes):
        shape = np.shape(value)
        batch_rank = len(self.batch_shape)
        if (
            self.batch_shape
            and shape[:batch_rank] == self.batch_shape
            and shape[batch_rank:] in shapes
        ):
            # Slice device arrays before transferring to the host, so recording
            # one world does not copy an entire distributed training batch.
            if isinstance(value, (tuple, list)):
                value = np.asarray(value)
            value = value[self.index]
        elif shape not in shapes:
            raise ValueError(
                f"{name} has unexpected shape {shape}; expected {shapes} after env_index"
            )
        # tolist/item below detach values, including device arrays, from the input.
        return np.asarray(value)

    def scalar(self, value, name):
        return self.array(value, name, ((), (1,))).reshape(()).item()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return value


def _integer(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if (minimum is not None and value < minimum) or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} is outside the supported range")
    return value


def validate_frame(frame):
    """Validate a JSON-compatible frame without changing it; return the frame."""
    if not isinstance(frame, dict):
        raise ValueError("Each replay frame must be an object")
    required = {
        "step",
        "action",
        "actor_id",
        "current_agent",
        "reward",
        "done",
        "task_done",
        "grid",
        "maps",
        "agents",
    }
    missing = required - frame.keys()
    if missing:
        raise ValueError(
            f"Frame is missing required fields: {', '.join(sorted(missing))}"
        )
    _integer(frame["step"], "step", 0)
    _number(frame["reward"], "reward")
    if frame["action"] is not None:
        _integer(frame["action"], "action", 0, 7)
    for name in ("done", "task_done"):
        if not isinstance(frame[name], bool):
            raise ValueError(f"{name} must be a boolean")
    if frame["task_done"] and not frame["done"]:
        raise ValueError("task_done requires done")
    grid = frame["grid"]
    if not isinstance(grid, dict) or not {"rows", "cols", "tile_size_m"} <= grid.keys():
        raise ValueError("grid requires rows, cols, and tile_size_m")
    rows = _integer(grid["rows"], "grid.rows", 1, MAX_GRID_SIZE)
    cols = _integer(grid["cols"], "grid.cols", 1, MAX_GRID_SIZE)
    if _number(grid["tile_size_m"], "grid.tile_size_m") <= 0:
        raise ValueError("grid.tile_size_m must be positive")
    maps = frame["maps"]
    if not isinstance(maps, dict):
        raise ValueError("maps must be an object")
    for name in _MAP_FIELDS:
        layer = maps.get(name)
        if layer is None:
            if name in _OPTIONAL_MAPS:
                continue
            raise ValueError(f"maps.{name} is required")
        if (
            not isinstance(layer, list)
            or len(layer) != rows
            or any(not isinstance(row, list) or len(row) != cols for row in layer)
        ):
            raise ValueError(f"maps.{name} must have shape {rows} x {cols}")
        try:
            array = np.asarray(layer)
            numeric = array.dtype.kind in "biuf" and np.isfinite(array).all()
        except (TypeError, ValueError, OverflowError):
            numeric = False
        if not numeric:
            raise ValueError(f"maps.{name} must contain finite numbers")
        if name in ("action", "target"):
            if array.dtype.kind == "b" or not np.equal(array, np.floor(array)).all():
                raise ValueError(f"maps.{name} must contain integer soil units")
        else:
            allowed = (-1, 0, 1) if name == "traversability" else (0, 1)
            if not np.isin(array, allowed).all():
                raise ValueError(f"maps.{name} contains invalid mask values")
    agents = frame["agents"]
    if not isinstance(agents, list) or not 1 <= len(agents) <= MAX_AGENTS:
        raise ValueError("agents must contain between one and four active agents")
    ids = set()
    agent_fields = {
        "id",
        "type",
        "action_type",
        "position",
        "base_yaw",
        "cabin_yaw",
        "width",
        "height",
        "loaded",
        "wheel_angle",
        "shovel_lifted",
        "reach",
    }
    for agent in agents:
        if not isinstance(agent, dict) or not agent_fields <= agent.keys():
            raise ValueError("Agent is missing required fields")
        agent_id = _integer(agent["id"], "agent.id", 0, MAX_AGENTS - 1)
        if agent_id in ids:
            raise ValueError("Agent IDs must be unique original state slots")
        ids.add(agent_id)
        _integer(agent["type"], "agent.type", 0, 2)
        _integer(agent["action_type"], "agent.action_type", 0, 1)
        _integer(agent["loaded"], "agent.loaded", 0)
        _integer(agent["wheel_angle"], "agent.wheel_angle")
        _integer(agent["shovel_lifted"], "agent.shovel_lifted", 0, 1)
        for name in ("width", "height"):
            if _number(agent[name], f"agent.{name}") <= 0:
                raise ValueError(f"agent.{name} must be positive")
        for name in ("base_yaw", "cabin_yaw"):
            _number(agent[name], f"agent.{name}")
        position = agent["position"]
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError("agent.position must contain row and column")
        for value, bound in zip(position, (rows, cols)):
            if not 0 <= _number(value, "agent.position") < bound:
                raise ValueError("agent.position is outside the grid")
        reach = agent["reach"]
        if not isinstance(reach, list) or len(reach) != 2:
            raise ValueError("agent.reach must contain inner and outer radius")
        if (
            not 0
            <= _number(reach[0], "agent.reach")
            <= _number(reach[1], "agent.reach")
        ):
            raise ValueError("agent.reach must satisfy 0 <= inner <= outer")
    current = _integer(frame["current_agent"], "current_agent", 0, MAX_AGENTS - 1)
    if current not in ids:
        raise ValueError("current_agent must identify an active agent")
    if frame["actor_id"] is not None:
        actor = _integer(frame["actor_id"], "actor_id", 0, MAX_AGENTS - 1)
        if actor not in ids:
            raise ValueError("actor_id must identify an active agent")
    if "joint_actions" in frame:
        if frame["action"] is not None or frame["actor_id"] is not None:
            raise ValueError("A joint round cannot name one action or actor")

        def slots(values, name, *, flags=False):
            if not isinstance(values, list) or not max(ids) < len(values) <= MAX_AGENTS:
                raise ValueError(f"{name} must use stable machine slots")
            for value in values:
                if flags:
                    if type(value) is not bool:
                        raise ValueError(f"{name} must contain boolean rejection flags")
                else:
                    _integer(value, name, 0, 7)

        requested = frame["joint_actions"]
        if requested is not None:
            slots(requested, "joint_actions")
        for name in ("effective_joint_actions", "workspace_blocked"):
            values = frame.get(name)
            if values is not None:
                slots(values, name, flags=name == "workspace_blocked")
                if requested is None or len(values) != len(requested):
                    raise ValueError(f"{name} must match requested slots")
    polygons = frame.get("workspace_polygons")
    if polygons is not None:
        if not isinstance(polygons, list):
            raise ValueError("workspace_polygons must be an array")
        components = set()
        for polygon in polygons:
            if not isinstance(polygon, dict):
                raise ValueError("A workspace polygon must be an object")
            identity = _integer(
                polygon.get("id"), "workspace polygon id", 0, MAX_AGENTS - 1
            )
            component = polygon.get("component")
            if identity not in ids or component not in ("body", "work"):
                raise ValueError("Unknown workspace component or machine")
            key = (identity, component)
            if key in components:
                raise ValueError("Repeated workspace component")
            components.add(key)
            vertices = polygon.get("vertices")
            if not isinstance(vertices, list) or len(vertices) < 3:
                raise ValueError("Workspace polygons require at least three vertices")
            for vertex in vertices:
                if not isinstance(vertex, list) or len(vertex) != 2:
                    raise ValueError("Workspace vertices must contain row and column")
                for value in vertex:
                    _number(value, "workspace vertex")
    return frame


def snapshot_from_timestep(timestep, action=None, actor_id=None, env_index=None):
    """Return a detached full-state frame from one unbatched or explicitly indexed timestep.

    ``env_index=(device, environment)`` selects both leading axes of a distributed
    batch. Shared scalar configuration values are supported alongside batched
    scalar leaves. Observation agent ordering is deliberately never consulted.
    ``wheel_angle`` retains Terra's discrete steering index.
    """
    state = _field(timestep, "state")
    world = _field(state, "world")
    raw_action = _field(_field(world, "action_map"), "map")
    selection = _Selection(raw_action, env_index)
    scalar = selection.scalar
    cfg = _field(timestep, "env_cfg")
    agent_cfg = _field(cfg, "agent")
    tile_size = scalar(_field(cfg, "tile_size"), "tile_size")
    if _number(tile_size, "tile_size") <= 0:
        raise ValueError("tile_size must be positive")
    maps = {}
    for name, field in _MAP_FIELDS.items():
        layer = _field(world, field, None)
        raw = _field(layer, "map", None) if layer is not None else None
        if raw is None:
            if name not in _OPTIONAL_MAPS:
                raise ValueError(f"Missing required map: {field}")
            maps[name] = None
            continue
        shapes = (
            (selection.grid_shape, (1, 1))
            if name in _OPTIONAL_MAPS
            else (selection.grid_shape,)
        )
        array = selection.array(raw, field, shapes)
        # Terra uses a 1 x 1 dummy GridMap for diagnostic layers not yet wrapped.
        maps[name] = None if array.shape != selection.grid_shape else array.tolist()
    agent_group = _field(state, "agent")
    states = _field(agent_group, "agent_states")
    if not isinstance(states, (tuple, list)) or not 1 <= len(states) <= MAX_AGENTS:
        raise ValueError("agent_states must contain one to four original state slots")
    active = selection.array(
        _field(agent_group, "agent_active"), "agent_active", ((len(states),),)
    )
    if not np.isin(active, (0, 1)).all():
        raise ValueError("agent_active must contain boolean or 0/1 values")
    width = scalar(_field(agent_group, "width"), "agent.width")
    height = scalar(_field(agent_group, "height"), "agent.height")
    angles_base = _integer(
        scalar(_field(agent_cfg, "angles_base"), "angles_base"), "angles_base", 1
    )
    angles_cabin = _integer(
        scalar(_field(agent_cfg, "angles_cabin"), "angles_cabin"), "angles_cabin", 1
    )
    dig_radius = _number(
        scalar(_field(agent_cfg, "dig_radius_tiles"), "dig_radius_tiles"),
        "dig_radius_tiles",
    )
    half_extent = max(width, height) / 2
    agents = []
    for index, (agent_state, enabled) in enumerate(zip(states, active)):
        if not enabled:
            continue
        agent_type = scalar(_field(agent_state, "agent_type"), "agent_type")
        if agent_type == 0:
            inner = half_extent + 0.5 / tile_size
            outer = inner + dig_radius
        else:
            # Terra routes trucks and skid steers through the same closer,
            # wider interaction workspace (State._get_dig_dump_mask_cyl_skidsteer).
            inner = half_extent - 2 + 0.1 * dig_radius
            outer = half_extent - 2 + 1.5 * dig_radius
        agents.append(
            {
                "id": index,
                "type": agent_type,
                "action_type": scalar(
                    _field(agent_state, "action_type"), "action_type"
                ),
                "position": selection.array(
                    _field(agent_state, "pos_base"), "pos_base", ((2,),)
                ).tolist(),
                "base_yaw": scalar(_field(agent_state, "angle_base"), "angle_base")
                * 2
                * math.pi
                / angles_base,
                "cabin_yaw": scalar(_field(agent_state, "angle_cabin"), "angle_cabin")
                * 2
                * math.pi
                / angles_cabin,
                "width": width,
                "height": height,
                "loaded": scalar(_field(agent_state, "loaded"), "loaded"),
                "wheel_angle": scalar(
                    _field(agent_state, "wheel_angle"), "wheel_angle"
                ),
                "shovel_lifted": scalar(
                    _field(agent_state, "shovel_lifted"), "shovel_lifted"
                ),
                "reach": [inner, outer],
            }
        )
    info = _field(timestep, "info")
    frame = {
        "step": scalar(_field(state, "env_steps"), "env_steps"),
        "action": (
            None
            if action is None
            else scalar(_field(action, "action", action), "action")
        ),
        "actor_id": None if actor_id is None else scalar(actor_id, "actor_id"),
        "current_agent": scalar(_field(agent_group, "current_agent"), "current_agent"),
        "reward": scalar(_field(timestep, "reward"), "reward"),
        "done": scalar(_field(timestep, "done"), "done"),
        "task_done": scalar(_field(info, "task_done"), "task_done"),
        "grid": {
            "rows": selection.grid_shape[0],
            "cols": selection.grid_shape[1],
            "tile_size_m": tile_size,
        },
        "maps": maps,
        "agents": agents,
    }
    return validate_frame(frame)
