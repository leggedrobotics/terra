"""Offline postprocessing of two excavators or an excavator and a skid.

The source is an ordered action trace, not alternating boolean dig/dump pairs.
Quantities, initial terrain, material order and machine setup are authoritative.
Cleanup and retiming produce geometric proposals; changed commands still need
native replay. This module does not issue ROS commands or evaluate a policy.
"""

from collections import defaultdict, deque
import copy
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .geometry import (
    CONTINUOUS_RESIDUAL_TOLERANCE_M2,
    source_mask_geometry,
    continuous_radial_runs,
    pull_band_polygon,
)
from . import fleet_geometry as geometry

SCHEMA = "terra_fleet_source"
RESULT_SCHEMA = "terra_fleet_postprocessed"
STATE_FIELDS = (
    "position_xy_m",
    "base_yaw_rad",
    "cabin_yaw_rad",
    "load_units",
    "wheel_angle_rad",
    "shovel_lifted",
)
EPS = 1e-7


def _finite(value, name, *, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.number))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its positive/nonnegative range")
    return float(value)


def _integer(value, name, *, minimum=0):
    number = _finite(value, name)
    if number != int(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _state(value):
    if not isinstance(value, dict) or any(key not in value for key in STATE_FIELDS):
        raise ValueError(f"Each machine state requires {STATE_FIELDS}")
    out = copy.deepcopy(value)
    position = np.asarray(out["position_xy_m"], dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("position_xy_m must be a finite XY position")
    out["position_xy_m"] = position.tolist()
    for field in ("base_yaw_rad", "cabin_yaw_rad", "wheel_angle_rad", "load_units"):
        out[field] = _finite(out[field], field, nonnegative=field == "load_units")
    if out["shovel_lifted"] not in (False, True, 0, 1):
        raise ValueError("shovel_lifted must be a boolean or 0/1")
    out["shovel_lifted"] = bool(out["shovel_lifted"])
    for field in ("body_polygon_xy_m", "work_origin_xy_m"):
        if field not in out:
            continue
        points = np.asarray(out[field], dtype=float)
        valid_shape = (
            points.shape == (2,)
            if field == "work_origin_xy_m"
            else (points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3)
        )
        if not valid_shape or not np.isfinite(points).all():
            raise ValueError(f"{field} has invalid coordinates")
        out[field] = points.tolist()
    return out


def _same_state(left, right, *, atol=EPS):
    fields = STATE_FIELDS + ("body_polygon_xy_m", "work_origin_xy_m")
    for key in fields:
        if (key in left) != (key in right):
            return False
        if key not in left:
            continue
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=atol):
            return False
    return True


def _alignment(grid):
    return {
        "meters_per_tile": grid["resolution_m"],
        "origin_map_xy_m": grid["origin_xy_m"],
        "yaw_map_from_plan_rad": grid["yaw_rad"],
    }


def _mask_geometry(mask, grid):
    return source_mask_geometry(np.asarray(mask, bool), _alignment(grid))


def _world(points, grid):
    c, s = math.cos(grid["yaw_rad"]), math.sin(grid["yaw_rad"])
    return (
        np.asarray(points) * grid["resolution_m"] @ np.array([[c, s], [-s, c]])
        + grid["origin_xy_m"]
    )


def validate_source(source):
    """Normalize and validate the complete recorded material/state sequence."""
    out = copy.deepcopy(source)
    if out.get("kind") != SCHEMA or out.get("schema_version") != 1:
        raise ValueError("Expected terra_fleet_source schema_version 1")
    grid = out["grid"]
    grid["resolution_m"] = _finite(
        grid.get("resolution_m"), "grid.resolution_m", positive=True
    )
    grid["cell_height_m"] = _finite(
        grid.get("cell_height_m"), "grid.cell_height_m", positive=True
    )
    grid["yaw_rad"] = _finite(grid.get("yaw_rad", 0.0), "grid.yaw_rad")
    origin = np.asarray(grid.get("origin_xy_m"), dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError(
            "grid.origin_xy_m must be the finite world position of cell (0,0)'s lower corner"
        )
    grid["origin_xy_m"] = origin.tolist()
    if grid.get("array_axes", "xy") != "xy":
        raise ValueError(
            "Fleet source arrays require array_axes='xy': axis 0 is plan X, axis 1 is plan Y"
        )
    grid["array_axes"] = "xy"
    terrain = np.asarray(out["initial_terrain"], dtype=float)
    if terrain.ndim != 2 or not terrain.size or not np.isfinite(terrain).all():
        raise ValueError(
            "initial_terrain must be a nonempty finite 2-D array of native soil units"
        )
    for key in ("target", "obstacles", "accepted_dump_mask"):
        array = np.asarray(out[key])
        if array.shape != terrain.shape or not np.isfinite(array).all():
            raise ValueError(f"{key} must share initial_terrain's finite 2-D shape")
        if key != "target" and not np.isin(array, [0, 1]).all():
            raise ValueError(f"{key} must be boolean")
        out[key] = (
            array.astype(float) if key == "target" else array.astype(bool)
        ).tolist()
    out["initial_terrain"] = terrain.tolist()
    agents = out["agents"]
    if len(agents) != 2:
        raise ValueError(
            "This fleet postprocessor currently supports exactly two agents"
        )
    identities = [agent.get("id") for agent in agents]
    if (
        any(not isinstance(identity, str) or not identity for identity in identities)
        or len(set(identities)) != 2
    ):
        raise ValueError("Agents need distinct, nonempty stable string IDs")
    kinds = sorted(agent.get("type") for agent in agents)
    if kinds not in (["excavator", "excavator"], ["excavator", "skid"]):
        raise ValueError(
            "Supported fleets are two excavators or one excavator and one skid"
        )
    states = {}
    for agent in agents:
        if agent.get("action_type", 0) != 0:
            raise ValueError(
                "Fleet native commands currently require tracked action_type=0"
            )
        agent["action_type"] = 0
        agent["geometry"] = geometry.validate_geometry(agent["geometry"])
        agent["initial_state"] = _state(agent["initial_state"])
        geometry.body_polygon(agent["initial_state"], agent["geometry"])
        states[agent["id"]] = agent["initial_state"]
    keys, action_ids, round_agents = [], set(), set()
    initial_mass = float(terrain.sum()) + sum(
        state["load_units"] for state in states.values()
    )
    for index, action in enumerate(out["actions"]):
        identity = action.get("agent_id")
        if identity not in states:
            raise ValueError(f"Action {index} names an unknown agent")
        action_id = action.get("id")
        if not isinstance(action_id, str) or not action_id or action_id in action_ids:
            raise ValueError("Actions need distinct, nonempty string IDs")
        action_ids.add(action_id)
        action["round"] = _integer(action["round"], "round")
        action["order"] = _integer(action["order"], "within-round order")
        key = action["round"], action["order"]
        if keys and key <= keys[-1]:
            raise ValueError("Actions must retain strictly increasing (round, order)")
        keys.append(key)
        round_agent = action["round"], identity
        if round_agent in round_agents:
            raise ValueError("Each agent may act at most once per source round")
        round_agents.add(round_agent)
        action["action"] = _integer(action["action"], "native action")
        if action["action"] > 7:
            raise ValueError("Native tracked action must be in 0..7")
        if any(
            field in action
            for field in ("requested_action", "effective_action", "workspace_blocked")
        ):
            if not all(
                field in action
                for field in (
                    "requested_action",
                    "effective_action",
                    "workspace_blocked",
                )
            ):
                raise ValueError(
                    "Native guard evidence requires requested/effective actions and rejection flag"
                )
            requested = _integer(action["requested_action"], "requested action")
            effective = _integer(action["effective_action"], "effective action")
            blocked = action["workspace_blocked"]
            if requested > 7 or effective > 7 or type(blocked) is not bool:
                raise ValueError(
                    "Native guard evidence has invalid commands or rejection flag"
                )
            if effective != (7 if blocked else requested):
                raise ValueError(
                    "Effective action disagrees with the native rejection flag"
                )
            if action.get("source_action", action["action"]) != effective:
                raise ValueError(
                    "Recorded source command disagrees with its effective action"
                )
        before, after = _state(action["before"]), _state(action["after"])
        if not _same_state(states[identity], before):
            raise ValueError(f"{action_id}: machine state is discontinuous")
        action["before"], action["after"] = before, after
        cells, seen, delta = [], set(), 0.0
        for cell in action["changed_cells"]:
            if len(cell) != 4:
                raise ValueError(
                    "changed_cells rows must be [axis0, axis1, before, after]"
                )
            x, y = _integer(cell[0], "cell axis0"), _integer(cell[1], "cell axis1")
            old, new = _finite(cell[2], "old terrain"), _finite(cell[3], "new terrain")
            if x >= terrain.shape[0] or y >= terrain.shape[1] or (x, y) in seen:
                raise ValueError(
                    f"{action_id}: duplicate or out-of-bounds terrain cell"
                )
            seen.add((x, y))
            if abs(terrain[x, y] - old) > EPS or old == new:
                raise ValueError(
                    f"{action_id}: changed cell does not match sequential terrain"
                )
            terrain[x, y] = new
            delta += new - old
            cells.append([x, y, old, new])
        if abs(delta + after["load_units"] - before["load_units"]) > EPS:
            raise ValueError(f"{action_id}: material is not conserved")
        if action["action"] == 7 and (cells or not _same_state(before, after)):
            raise ValueError(f"{action_id}: WAIT changes machine or material state")
        action["changed_cells"] = cells
        states[identity] = after
    final_mass = float(terrain.sum()) + sum(
        state["load_units"] for state in states.values()
    )
    if abs(final_mass - initial_mass) > EPS:
        raise ValueError("Final fleet material inventory is inconsistent")
    return out


def import_native_npz(
    path,
    *,
    agents,
    tile_size_m,
    cell_height_m,
    case_id=None,
    origin_xy_m=(0.0, 0.0),
    yaw_rad=0.0,
    angles_base=12,
    angles_cabin=12,
    material_events_path=None,
):
    """Import captured NPZ substeps without policy inference or campaign paths.

    ``agents`` supplies stable IDs, types and explicit per-machine geometry;
    its order is the native slot order. ``cell_height_m`` is a declared display
    conversion, not a physical volume model. Arrays must contain exact substeps.
    """
    if len(agents) != 2:
        raise ValueError("Two explicit native-slot agent descriptions are required")
    angles_base = _integer(angles_base, "angles_base", minimum=1)
    angles_cabin = _integer(angles_cabin, "angles_cabin", minimum=1)
    grid = {
        "resolution_m": _finite(tile_size_m, "tile_size_m", positive=True),
        "cell_height_m": _finite(cell_height_m, "cell_height_m", positive=True),
        "origin_xy_m": list(origin_xy_m),
        "yaw_rad": float(yaw_rad),
        "array_axes": "xy",
    }
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    fields = (
        "positions",
        "base_angles",
        "cabin_angles",
        "loads",
        "wheel_angles",
        "shovel_lifted",
    )
    for key in (
        "terrain",
        "actions",
        "orders",
        "case_ids",
        "rounds",
        "target",
        "accepted_mask",
        "obstacles",
    ):
        if key not in data:
            raise ValueError(f"Native trace is missing {key}")
    for prefix in ("", "before_", "after_"):
        for field in fields + ("terrain",):
            if prefix + field not in data:
                raise ValueError(f"Native trace lacks exact substeps: {prefix + field}")
    if data["actions"].ndim != 3 or data["actions"].shape[-1] != 2:
        raise ValueError(
            "Native actions must have shape [round, case, two agent slots]"
        )
    if data["orders"].shape != data["actions"].shape:
        raise ValueError("Native orders and actions must have the same shape")
    for key in (
        "obstacles",
        "accepted_mask",
        "shovel_lifted",
        "before_shovel_lifted",
        "after_shovel_lifted",
    ):
        if not np.isfinite(data[key]).all() or not np.isin(data[key], [0, 1]).all():
            raise ValueError(f"Native {key} must contain finite binary values")
    if (
        not np.isfinite(data["actions"]).all()
        or not np.isin(data["actions"], range(8)).all()
    ):
        raise ValueError("Native actions must be integral tracked commands in 0..7")
    guard_evidence = "effective_actions" in data or "workspace_blocked" in data
    if guard_evidence:
        if not all(key in data for key in ("effective_actions", "workspace_blocked")):
            raise ValueError(
                "Native guard evidence requires effective_actions and workspace_blocked"
            )
        effective, blocked = data["effective_actions"], data["workspace_blocked"]
        if (
            effective.shape != data["actions"].shape
            or not np.isin(effective, range(8)).all()
        ):
            raise ValueError(
                "Native effective_actions must match the integral commands array"
            )
        if blocked.shape != data["actions"].shape or not np.isin(blocked, [0, 1]).all():
            raise ValueError(
                "Native workspace_blocked must match the binary commands array"
            )
        if not np.array_equal(effective, np.where(blocked, 7, data["actions"])):
            raise ValueError("Native effective_actions disagree with workspace_blocked")
    selected = [
        i
        for i, identity in enumerate(data["case_ids"])
        if case_id is None or int(identity) == int(case_id)
    ]
    if not selected:
        raise ValueError(f"Native trace has no case {case_id}")
    recorded = (
        json.loads(Path(material_events_path).read_text())
        if material_events_path is not None
        else None
    )

    def state(prefix, index, slot):
        position = data[prefix + "positions"][index][slot]
        result = {
            "position_xy_m": _world(position, grid).tolist(),
            "base_yaw_rad": float(data[prefix + "base_angles"][index][slot])
            * 2
            * math.pi
            / angles_base
            + math.pi / 2
            + yaw_rad,
            "cabin_yaw_rad": float(data[prefix + "cabin_angles"][index][slot])
            * 2
            * math.pi
            / angles_cabin,
            "wheel_angle_rad": float(data[prefix + "wheel_angles"][index][slot])
            * math.radians(20),
            "shovel_lifted": bool(data[prefix + "shovel_lifted"][index][slot]),
            "load_units": float(data[prefix + "loads"][index][slot]),
            "work_origin_xy_m": _world(np.asarray(position) + 0.5, grid).tolist(),
        }
        if prefix + "native_corners" in data:
            result["body_polygon_xy_m"] = _world(
                data[prefix + "native_corners"][index][slot], grid
            ).tolist()
        return result

    results = []
    for case in selected:
        identity = int(data["case_ids"][case])
        rounds = _integer(data["rounds"][case], "native rounds")
        if rounds > data["actions"].shape[0] or rounds >= data["terrain"].shape[0]:
            raise ValueError("Native rounds exceed captured trace length")
        source_agents = [
            {**copy.deepcopy(agent), "initial_state": state("", (0, case), slot)}
            for slot, agent in enumerate(agents)
        ]
        current = np.array(data["terrain"][0, case], dtype=float)
        states = [agent["initial_state"] for agent in source_agents]
        actions = []
        for t in range(rounds):
            order = data["orders"][t, case]
            if sorted(order.tolist()) != [0, 1]:
                raise ValueError(
                    "Native order must be a permutation of fixed slots 0,1"
                )
            for within, slot_value in enumerate(order):
                slot = int(slot_value)
                index = t, case, within
                old, new = data["before_terrain"][index], data["after_terrain"][index]
                if not np.array_equal(old, current):
                    raise ValueError("Native substep terrain is discontinuous")
                before_all = [state("before_", index, peer) for peer in range(2)]
                after_all = [state("after_", index, peer) for peer in range(2)]
                if any(
                    not _same_state(states[peer], before_all[peer], atol=0)
                    for peer in range(2)
                ):
                    raise ValueError("Native substep machine state is discontinuous")
                if not _same_state(before_all[1 - slot], after_all[1 - slot], atol=0):
                    raise ValueError(
                        "A native action changes its peer; direct carrier transfers need a separate schema"
                    )
                changed = [
                    [int(x), int(y), float(old[x, y]), float(new[x, y])]
                    for x, y in np.argwhere(old != new)
                ]
                actions.append(
                    {
                        "id": f"{identity}:{t + 1}:{within}",
                        "round": t + 1,
                        "order": within,
                        "agent_id": source_agents[slot]["id"],
                        "action": int(
                            data["effective_actions" if guard_evidence else "actions"][
                                t, case, slot
                            ]
                        ),
                        "before": before_all[slot],
                        "after": after_all[slot],
                        "changed_cells": changed,
                    }
                )
                if guard_evidence:
                    actions[-1].update(
                        requested_action=int(data["actions"][t, case, slot]),
                        effective_action=int(data["effective_actions"][t, case, slot]),
                        workspace_blocked=bool(
                            data["workspace_blocked"][t, case, slot]
                        ),
                    )
                current, states = np.array(new), after_all
            if not np.array_equal(current, data["terrain"][t + 1, case]):
                raise ValueError(
                    "Native substeps do not match the recorded round terrain"
                )
            if any(
                not _same_state(states[slot], state("", (t + 1, case), slot), atol=0)
                for slot in range(2)
            ):
                raise ValueError(
                    "Native substeps do not match the recorded round machine states"
                )
        source = {
            "kind": SCHEMA,
            "schema_version": 1,
            "case_id": identity,
            "grid": grid,
            "initial_terrain": data["terrain"][0, case].tolist(),
            "target": data["target"][case].tolist(),
            "obstacles": np.asarray(data["obstacles"][case], bool).tolist(),
            "accepted_dump_mask": data["accepted_mask"][case].tolist(),
            "agents": source_agents,
            "actions": actions,
            "load_unit": "soil_units",
            "cabin_yaw_convention": "relative_to_base",
            "provenance": {
                "format": "captured_native_substeps",
                "path": str(Path(path).resolve()),
                "native_reexecuted": False,
                "height_conversion": "declared display conversion, not calibrated physical volume",
            },
        }
        source = validate_source(source)
        if recorded is not None:
            expected = {
                (int(event["round"]), int(event["within_round_order"])): event
                for event in recorded
                if int(event["case_id"]) == identity
            }
            actual = {
                (a["round"], a["order"]): a for a in source["actions"] if _material(a)
            }
            if expected.keys() != actual.keys():
                raise ValueError("Material JSON and native substep events differ")
            for key, action in actual.items():
                event = expected[key]
                if (
                    event["action"] != action["action"]
                    or source_agents[int(event["agent"])]["id"] != action["agent_id"]
                    or not np.array_equal(
                        event["changed_cells"], action["changed_cells"]
                    )
                    or abs(event["before"]["load"] - action["before"]["load_units"])
                    > EPS
                    or abs(event["after"]["load"] - action["after"]["load_units"]) > EPS
                ):
                    raise ValueError(
                        "Material JSON does not agree with captured substeps"
                    )
        results.append(source)
    return results


class NativeFleetRecorder:
    """Capture fresh selected joint rollouts through the maintained fleet schema.

    Requires Terra's joint-step runtime: ``State._apply_action`` and
    ``State._with_traversability_mask``, with both agents executing once in an
    explicit order per round. Older round-robin/viewer-only runtimes are rejected;
    use the SAME joint-step runtime that generated the selected rollout. This
    adapter takes *one unbatched native State*, explicit machine profiles and
    accepted-dump mask. The policy/environment still owns inference and
    termination. With joint-runtime ``TerraEnv.step_no_reset(state, actions,
    config, order)``, integrate in the selected rollout as follows::

        recorder = NativeFleetRecorder(
            initial_state, agents=agent_profiles, tile_size_m=tile_size,
            cell_height_m=display_height, accepted_dump_mask=accepted_mask)
        for round_index in range(horizon):
            actions, order = policy_actions_and_order(...)
            next_timestep = env.step_no_reset(state, actions, config, order)
            recorder.record_round(
                state, next_timestep.state, actions, order,
                effective_actions=next_timestep.info["effective_actions"],
                workspace_blocked=next_timestep.info["workspace_blocked"],
                round_index=round_index)
            state = next_timestep.state
            if bool(next_timestep.done):
                break
        source = recorder.source()

    ``record_round`` replays the native *effective* commands, with no policy
    inference. Guarded runtimes must supply their effective commands and
    rejection flags; requested actions remain evidence beside them. This does
    not reimplement the native workspace guard. It requires exact terrain and
    captured machine-state agreement with the canonical post-round State before
    accepting any substeps. Use the same
    native backend/runtime as the original step. A generator that already
    captures substeps can call ``append_substep`` directly and ``verify_round``
    at each native round boundary. Once-per-round frames cannot substitute for
    these captures. Native clocks/rewards are not reconstructed by this adapter.
    """

    def __init__(
        self,
        initial_state,
        *,
        agents,
        tile_size_m,
        cell_height_m,
        accepted_dump_mask,
        origin_xy_m=(0.0, 0.0),
        yaw_rad=0.0,
        angles_base=12,
        angles_cabin=12,
        wheel_step_deg=20.0,
        case_id=None,
    ):
        if len(agents) != 2:
            raise ValueError(
                "NativeFleetRecorder requires two stable native-slot agents"
            )
        required_methods = ("_apply_action", "_with_traversability_mask", "_replace")
        if any(
            not callable(getattr(initial_state, name, None))
            for name in required_methods
        ):
            raise ValueError(
                "NativeFleetRecorder requires Terra's joint-step State._apply_action and "
                "State._with_traversability_mask APIs. Use the same joint-step runtime as the "
                "rollout, not an older round-robin or viewer-only Terra checkout."
            )
        types = {"excavator": 0, "skid": 2}
        if any(agent.get("type") not in types for agent in agents):
            raise ValueError("Native recorder profiles require excavator or skid types")
        self._native_types = [types[agent["type"]] for agent in agents]
        self._guarded = self._guard_enabled(initial_state)
        self.grid = {
            "resolution_m": _finite(tile_size_m, "tile_size_m", positive=True),
            "cell_height_m": _finite(cell_height_m, "cell_height_m", positive=True),
            "origin_xy_m": list(origin_xy_m),
            "yaw_rad": _finite(yaw_rad, "yaw_rad"),
            "array_axes": "xy",
        }
        self.angles_base = _integer(angles_base, "angles_base", minimum=1)
        self.angles_cabin = _integer(angles_cabin, "angles_cabin", minimum=1)
        self.wheel_step_deg = _finite(wheel_step_deg, "wheel_step_deg", positive=True)
        terrain, states = self._snapshot(initial_state)
        prepared = [
            {**copy.deepcopy(agent), "initial_state": states[slot]}
            for slot, agent in enumerate(agents)
        ]
        self._source = validate_source(
            {
                "kind": SCHEMA,
                "schema_version": 1,
                "case_id": case_id,
                "grid": self.grid,
                "initial_terrain": terrain.tolist(),
                "target": np.asarray(initial_state.world.target_map.map).tolist(),
                "obstacles": np.asarray(initial_state.world.padding_mask.map).tolist(),
                "accepted_dump_mask": np.asarray(accepted_dump_mask).tolist(),
                "agents": prepared,
                "actions": [],
                "load_unit": "soil_units",
                "cabin_yaw_convention": "relative_to_base",
                "provenance": {
                    "format": "recorded_native_substeps",
                    "verified_rounds": 0,
                    "native_workspace_guard_enabled": self._guarded,
                    "height_conversion": "declared display conversion, not calibrated physical volume",
                },
            }
        )
        self._terrain, self._states = terrain.copy(), states
        self._unverified_round = False
        self._verified_actions = 0

    @staticmethod
    def _guard_enabled(native):
        enabled = np.asarray(
            getattr(getattr(native, "env_cfg", None), "workspace_guard_enabled", False)
        )
        if enabled.size != 1 or not np.isin(enabled, [0, 1]).all():
            raise ValueError(
                "Recorder requires one finite binary native workspace guard setting"
            )
        return bool(enabled.reshape(-1)[0])

    def _snapshot(self, native):
        terrain = np.asarray(native.world.action_map.map)
        if terrain.ndim != 2 or not np.isfinite(terrain).all():
            raise ValueError(
                "Recorder requires one unbatched native State with finite terrain"
            )
        count = np.asarray(native.agent.num_agents)
        if count.size != 1 or int(count.reshape(-1)[0]) != 2:
            raise ValueError("Recorder requires exactly two active native slots")
        states = []
        for slot in range(2):
            agent = native.agent.agent_states[slot]

            def scalar(name):
                value = np.asarray(getattr(agent, name))
                if value.size != 1:
                    raise ValueError("Recorder requires scalar native agent fields")
                if name == "shovel_lifted" and value.reshape(-1)[0] in (0, 1):
                    return float(value.reshape(-1)[0])
                return _finite(value.reshape(-1)[0], name)

            position = np.asarray(agent.pos_base)
            if position.shape != (2,):
                raise ValueError("Recorder requires unbatched native XY positions")
            if (
                scalar("agent_type") != self._native_types[slot]
                or scalar("action_type") != 0
            ):
                raise ValueError(
                    "Native slot types/action types disagree with the declared agent profiles"
                )
            shovel = scalar("shovel_lifted")
            if shovel not in (0, 1):
                raise ValueError("Native shovel_lifted must be binary")
            machine = {
                "position_xy_m": _world(position, self.grid).tolist(),
                "base_yaw_rad": scalar("angle_base") * 2 * math.pi / self.angles_base
                + math.pi / 2
                + self.grid["yaw_rad"],
                "cabin_yaw_rad": scalar("angle_cabin")
                * 2
                * math.pi
                / self.angles_cabin,
                "wheel_angle_rad": scalar("wheel_angle")
                * math.radians(self.wheel_step_deg),
                "load_units": scalar("loaded"),
                "shovel_lifted": bool(shovel),
                "work_origin_xy_m": _world(position + 0.5, self.grid).tolist(),
            }
            if hasattr(native, "_get_agent_corners"):
                corners = native._get_agent_corners(
                    agent.pos_base,
                    agent.angle_base,
                    native.env_cfg.agent.width,
                    native.env_cfg.agent.height,
                )
                machine["body_polygon_xy_m"] = _world(
                    np.asarray(corners), self.grid
                ).tolist()
            states.append(_state(machine))
        return terrain.copy(), states

    def append_substep(
        self,
        before,
        after,
        *,
        slot,
        action,
        round_index,
        within_round_order,
        requested_action=None,
        workspace_blocked=None,
    ):
        if (
            self._guard_enabled(before) != self._guarded
            or self._guard_enabled(after) != self._guarded
        ):
            raise ValueError("Native workspace guard setting changed during capture")
        slot = _integer(slot, "native slot")
        if slot > 1:
            raise ValueError("Native slot must be 0 or 1")
        old, entry = self._snapshot(before)
        new, exit_states = self._snapshot(after)
        if not np.array_equal(old, self._terrain) or any(
            not _same_state(a, b, atol=0) for a, b in zip(entry, self._states)
        ):
            raise ValueError("Fresh native substep is discontinuous")
        if not _same_state(entry[1 - slot], exit_states[1 - slot], atol=0):
            raise ValueError(
                "Native substep changes its peer; direct transfers are not supported"
            )
        round_index = _integer(round_index, "round_index")
        within_round_order = _integer(within_round_order, "within_round_order")
        command = {
            "id": f"{round_index}:{within_round_order}",
            "round": round_index,
            "order": within_round_order,
            "agent_id": self._source["agents"][slot]["id"],
            "action": _integer(action, "native action"),
            "before": entry[slot],
            "after": exit_states[slot],
            "changed_cells": [
                [int(x), int(y), float(old[x, y]), float(new[x, y])]
                for x, y in np.argwhere(old != new)
            ],
        }
        if self._guarded and (requested_action is None or workspace_blocked is None):
            raise ValueError(
                "Guarded native substeps require requested/effective actions and rejection flags"
            )
        if requested_action is not None or workspace_blocked is not None:
            requested = _integer(requested_action, "requested action")
            if requested > 7 or not isinstance(workspace_blocked, (bool, np.bool_)):
                raise ValueError(
                    "Native guard evidence requires valid commands and a boolean rejection flag"
                )
            if command["action"] != (7 if workspace_blocked else requested):
                raise ValueError(
                    "Effective action disagrees with the native rejection flag"
                )
            command.update(
                requested_action=requested,
                effective_action=command["action"],
                workspace_blocked=bool(workspace_blocked),
            )
        if (
            abs(
                float(new.sum() - old.sum())
                + exit_states[slot]["load_units"]
                - entry[slot]["load_units"]
            )
            > EPS
        ):
            raise ValueError("Fresh native substep does not conserve material")
        self._source["actions"].append(command)
        self._terrain, self._states = new, exit_states
        self._unverified_round = True

    def verify_round(self, canonical_after):
        pending = self._source["actions"][self._verified_actions :]
        if (
            len(pending) != 2
            or pending[0]["round"] != pending[1]["round"]
            or [action["order"] for action in pending] != [0, 1]
            or pending[0]["agent_id"] == pending[1]["agent_id"]
        ):
            raise ValueError(
                "A verified joint round requires exactly two ordered substeps, one per native slot"
            )
        terrain, states = self._snapshot(canonical_after)
        if not np.array_equal(terrain, self._terrain) or any(
            not _same_state(a, b, atol=0) for a, b in zip(states, self._states)
        ):
            raise ValueError(
                "Captured native substeps disagree with canonical post-round state"
            )
        self._source["provenance"]["verified_rounds"] += 1
        self._unverified_round = False
        self._verified_actions = len(self._source["actions"])

    def record_round(
        self,
        before,
        canonical_after,
        joint_actions,
        order,
        *,
        round_index,
        effective_actions=None,
        workspace_blocked=None,
    ):
        if self._unverified_round:
            raise ValueError(
                "Verify the pending round before recording another joint round"
            )
        values = (
            joint_actions.action if hasattr(joint_actions, "action") else joint_actions
        )
        if isinstance(values, (list, tuple)):
            values = np.asarray(values)
        if (
            np.asarray(values).shape != (2,)
            or not np.isin(np.asarray(values), range(8)).all()
        ):
            raise ValueError("Expected one tracked action for each of two native slots")
        if (
            self._guard_enabled(before) != self._guarded
            or self._guard_enabled(canonical_after) != self._guarded
        ):
            raise ValueError("Native workspace guard setting changed during capture")
        evidence = effective_actions is not None or workspace_blocked is not None
        if self._guarded and not evidence:
            raise ValueError(
                "Guarded native rounds require effective_actions and workspace_blocked evidence"
            )
        if evidence:
            effective = np.asarray(effective_actions)
            blocked = np.asarray(workspace_blocked)
            if effective.shape != (2,) or not np.isin(effective, range(8)).all():
                raise ValueError(
                    "Effective commands must contain one tracked action per native slot"
                )
            if blocked.shape != (2,) or not np.isin(blocked, [0, 1]).all():
                raise ValueError(
                    "workspace_blocked must contain one binary flag per native slot"
                )
            if not np.array_equal(effective, np.where(blocked, 7, np.asarray(values))):
                raise ValueError(
                    "Effective actions disagree with native rejection flags"
                )
            # Retain the native array type: JAX states use the effective slice in lax.switch.
            commands = values * 0 + effective
        else:
            commands = values
            blocked = np.zeros(2, dtype=bool)
        ordered = np.asarray(order)
        if ordered.shape != (2,) or sorted(ordered.tolist()) != [0, 1]:
            raise ValueError("Expected a permutation of the two native slots")
        current, captured = before, []
        for within, slot_value in enumerate(ordered):
            slot = int(slot_value)
            acting = current._replace(agent=current.agent._replace(current_agent=slot))
            next_state = acting._apply_action(commands[slot : slot + 1])
            captured.append(
                (acting, next_state, slot, int(np.asarray(commands[slot])), within)
            )
            current = (
                next_state._with_traversability_mask() if within == 0 else next_state
            )
        actual_terrain, actual_states = self._snapshot(current)
        expected_terrain, expected_states = self._snapshot(canonical_after)
        if not np.array_equal(actual_terrain, expected_terrain) or any(
            not _same_state(a, b, atol=0)
            for a, b in zip(actual_states, expected_states)
        ):
            raise ValueError(
                "Native replay does not exactly match canonical post-round state"
            )
        for entry, exit_state, slot, command, within in captured:
            self.append_substep(
                entry,
                exit_state,
                slot=slot,
                action=command,
                round_index=round_index,
                within_round_order=within,
                requested_action=int(np.asarray(values[slot])),
                workspace_blocked=bool(blocked[slot]),
            )
        self.verify_round(canonical_after)

    def source(self):
        if self._unverified_round:
            raise ValueError(
                "Verify the canonical post-round state before exporting fresh substeps"
            )
        return validate_source(self._source)


def _material(action):
    return (
        bool(action["changed_cells"])
        or abs(action["after"]["load_units"] - action["before"]["load_units"]) > EPS
    )


def clean_motion(actions, *, max_loop_actions=64):
    """Replace only material-free closed motion loops with explicit WAIT.

    Wheel/shovel/load setup is fixed throughout a removable loop. Source round
    order and productive entry/exit states are unchanged. Native replay remains
    required because removing a loop can change another machine's access.
    """
    cleaned, edits = copy.deepcopy(actions), []
    streams = defaultdict(list)
    for index, action in enumerate(cleaned):
        streams[action["agent_id"]].append(index)
    for stream in streams.values():
        cursor = 0
        while cursor < len(stream):
            start = cleaned[stream[cursor]]["before"]
            last_return = None
            for end in range(cursor, min(len(stream), cursor + max_loop_actions)):
                action = cleaned[stream[end]]
                if action["action"] not in range(6) or _material(action):
                    break
                if any(
                    abs(float(action[when][key]) - float(start[key])) > EPS
                    for when in ("before", "after")
                    for key in ("load_units", "wheel_angle_rad", "shovel_lifted")
                ):
                    break
                if _same_state(action["after"], start):
                    last_return = end
            if last_return is None:
                cursor += 1
                continue
            members = []
            for offset in range(cursor, last_return + 1):
                action = cleaned[stream[offset]]
                members.append(action["id"])
                action["source_action"] = action["action"]
                action["action"] = 7
                action["before"] = copy.deepcopy(start)
                action["after"] = copy.deepcopy(start)
            edits.append(
                {
                    "kind": "closed_motion_loop_to_wait",
                    "agent_id": cleaned[stream[cursor]]["agent_id"],
                    "action_ids": members,
                    "native_replay_required": True,
                }
            )
            cursor = last_return + 1
    return cleaned, edits


def _workspace(action, kind, agent, source):
    shape_xy = np.asarray(source["target"]).shape
    masks = {
        key: np.zeros(shape_xy, bool) for key in ("dig", "finish", "dump", "deposit")
    }
    target = np.asarray(source["target"])
    for x, y, old, new in action["changed_cells"]:
        masks["dig"][x, y] = new < old
        masks["finish"][x, y] = new < old and target[x, y] < 0 and new <= target[x, y]
        masks["dump"][x, y] = new > old
        masks["deposit"][x, y] = new > old and new > 0
    affected = masks["dig"] | masks["dump"]
    permission = _mask_geometry(masks["dig"], source["grid"])
    envelope = geometry.work_polygon(action["before"], agent["geometry"])
    refinement = {
        "status": "RECORDED_MATERIAL_SUPPORT",
        "lanes": [],
        "uncovered_cells": [],
    }
    refined = _mask_geometry(affected, source["grid"])
    if (
        agent["type"] == "excavator"
        and masks["dig"].any()
        and "tool_width_m" in agent["geometry"]
    ):
        profile = agent["geometry"]
        pivot = geometry.work_origin(action["before"], profile)
        centres = _world(np.argwhere(masks["dig"]) + 0.5, source["grid"])
        directions = np.arctan2(centres[:, 1] - pivot[1], centres[:, 0] - pivot[0])
        heading = action["before"]["base_yaw_rad"] + action["before"]["cabin_yaw_rad"]
        # Aim at each affected cell plus a bounded regular sweep. Geometry is
        # admitted by the offline continuous full-width permission model.
        bearings = sorted(
            set(
                np.round(
                    np.r_[
                        directions,
                        np.linspace(
                            heading - profile["work_half_angle_rad"],
                            heading + profile["work_half_angle_rad"],
                            49,
                        ),
                    ],
                    10,
                )
            )
        )
        polygons, lanes = [], []
        for bearing in bearings:
            if (
                abs(geometry.angle_delta(heading, bearing))
                > profile["work_half_angle_rad"] + EPS
            ):
                continue
            unit = np.array([math.cos(bearing), math.sin(bearing)])
            for near, far in continuous_radial_runs(
                permission,
                pivot,
                unit,
                profile["work_min_radius_m"],
                profile["work_reach_m"],
                profile["tool_width_m"] / 2,
            ):
                if far - near <= EPS:
                    continue
                band = pull_band_polygon(
                    pivot, unit, near, far, profile["tool_width_m"] / 2
                )
                polygons.append(band)
                lanes.append(
                    {
                        "heading_rad": float(bearing),
                        "near_m": float(near),
                        "far_m": float(far),
                        "width_m": profile["tool_width_m"],
                        "geometry": geometry.polygon_record(band),
                    }
                )
        refined = unary_union(polygons) if polygons else Polygon()
        import shapely

        covered = np.asarray(
            shapely.covers(refined.buffer(EPS), shapely.points(centres)), bool
        )
        residual = float(permission.difference(refined).area)
        complete = bool(covered.all()) and residual <= CONTINUOUS_RESIDUAL_TOLERANCE_M2
        refinement = {
            "status": "RADIAL_BANDS" if complete else "RADIAL_BANDS_INCOMPLETE",
            "lanes": lanes,
            "covered_cells": int(covered.sum()),
            "uncovered_cells": np.argwhere(masks["dig"])[~covered].tolist(),
            "residual_area_m2": residual,
            "residual_tolerance_m2": CONTINUOUS_RESIDUAL_TOLERANCE_M2,
            "effect": "geometric support refinement; recorded material quantities remain unchanged",
        }
    elif agent["type"] == "skid" and masks["dig"].any():
        refinement["status"] = "RECORDED_PICKUP_SWEEP"
        refinement["motion_required"] = True
    elif agent["type"] == "excavator" and masks["dig"].any():
        refinement["status"] = "MISSING_TOOL_WIDTH"
        refinement["uncovered_cells"] = np.argwhere(masks["dig"]).tolist()
    return {
        "id": action["id"],
        "agent_id": action["agent_id"],
        "kind": kind,
        "pose": copy.deepcopy(action["before"]),
        "masks": {key: np.argwhere(mask).tolist() for key, mask in masks.items()},
        "geometry": geometry.polygon_record(_mask_geometry(affected, source["grid"])),
        "refined_geometry": geometry.polygon_record(refined),
        "reservation_geometry": geometry.polygon_record(envelope),
        "refinement": refinement,
    }


def compile_events(source, actions):
    agents = {agent["id"]: agent for agent in source["agents"]}
    stock = defaultdict(deque)
    initial = np.asarray(source["initial_terrain"])
    for x, y in np.argwhere(initial > 0):
        stock[int(x), int(y)].append((f"initial:{x}:{y}", float(initial[x, y])))
    events, cycles, open_cycles = [], [], defaultdict(list)
    previous = {identity: None for identity in agents}
    accepted, target = np.asarray(source["accepted_dump_mask"]), np.asarray(
        source["target"]
    )
    for action in actions:
        if not _material(action):
            continue
        consumed = defaultdict(float)
        fresh, delivered = 0.0, 0.0
        for x, y, old, new in action["changed_cells"]:
            remaining = max(old, 0.0) - max(new, 0.0)
            while remaining > EPS:
                if not stock[x, y]:
                    raise ValueError("Pickup precedes available shared soil")
                producer, quantity = stock[x, y].popleft()
                used = min(quantity, remaining)
                consumed[producer] += used
                remaining -= used
                if quantity - used > EPS:
                    stock[x, y].appendleft((producer, quantity - used))
            if remaining < -EPS:
                stock[x, y].append((action["id"], -remaining))
            if target[x, y] < 0:
                fresh += min(max(-new, 0), -target[x, y]) - min(
                    max(-old, 0), -target[x, y]
                )
            if accepted[x, y]:
                delivered += max(new, 0) - max(old, 0)
        delta = action["after"]["load_units"] - action["before"]["load_units"]
        kind = (
            ("excavate" if fresh > EPS else "collect")
            if delta > EPS
            else (
                ("deliver" if delivered > EPS else "stage")
                if delta < -EPS
                else "redistribute"
            )
        )
        identity = action["agent_id"]
        event = {
            "id": action["id"],
            "action_id": action["id"],
            "agent_id": identity,
            "round": action["round"],
            "order": action["order"],
            "kind": kind,
            "load_delta": delta,
            "quantity": abs(delta),
            "fresh_units": fresh,
            "accepted_delta": delivered,
            "before": action["before"],
            "after": action["after"],
            "changed_cells": action["changed_cells"],
            "previous_agent_event": previous[identity],
            "previous_material_event": events[-1]["id"] if events else None,
            "material_dependencies": [
                {"event_id": key, "units": value}
                for key, value in sorted(consumed.items())
            ],
            "preserve_motion_during_event": action["action"] != 6,
            "workspace": _workspace(action, kind, agents[identity], source),
        }
        action["material_event_id"] = event["id"]
        action["workspace"] = event["workspace"]
        events.append(event)
        previous[identity] = event["id"]
        open_cycles[identity].append(event["id"])
        if action["after"]["load_units"] <= EPS:
            cycles.append(
                {
                    "id": f"{identity}:cycle:{len(cycles)}",
                    "agent_id": identity,
                    "event_ids": open_cycles[identity],
                    "complete": True,
                }
            )
            open_cycles[identity] = []
    for identity, members in open_cycles.items():
        if members:
            cycles.append(
                {
                    "id": f"{identity}:cycle:{len(cycles)}",
                    "agent_id": identity,
                    "event_ids": members,
                    "complete": False,
                }
            )
    return events, cycles


class _Scene:
    def __init__(self, source):
        self.source = source
        self.agents = {agent["id"]: agent for agent in source["agents"]}
        self.shape = np.asarray(source["initial_terrain"]).shape
        self.domain = _mask_geometry(np.ones(self.shape, bool), source["grid"])
        self.cache = {}

    def hazards(self, terrain, identity, key=None):
        cache_key = key, identity
        if key is not None and cache_key in self.cache:
            return self.cache[cache_key]
        profile = self.agents[identity]["geometry"]
        height = terrain * self.source["grid"]["cell_height_m"]
        blocked = (
            np.asarray(self.source["obstacles"], bool)
            | (height < -profile["max_drivable_cut_m"] - EPS)
            | (height > profile["max_drivable_height_m"] + EPS)
        )
        result = _mask_geometry(blocked, self.source["grid"])
        if key is not None:
            self.cache[cache_key] = result
        return result

    def body_check(self, body, terrain, identity, key=None):
        hazards = self.hazards(terrain, identity, key)
        overlap = float(body.intersection(hazards).area)
        outside = float(body.difference(self.domain).area)
        return {
            "passed": overlap <= EPS and outside <= EPS,
            "hazard_overlap_m2": overlap,
            "outside_domain_m2": outside,
        }


def audit(source, actions):
    scene = _Scene(source)
    terrain = np.asarray(source["initial_terrain"], dtype=float).copy()
    states = {agent["id"]: agent["initial_state"] for agent in source["agents"]}
    intermachine, terrain_issues = [], []
    by_round = defaultdict(list)
    for action in actions:
        by_round[action["round"]].append(action)
    ids = list(states)
    for source_round, members in by_round.items():
        reservations = {
            identity: geometry.reservation(
                state, state, scene.agents[identity]["geometry"]
            )
            for identity, state in states.items()
        }
        for action in members:
            identity = action["agent_id"]
            reservations[identity] = geometry.reservation(
                action["before"], action["after"], scene.agents[identity]["geometry"]
            )
            check = scene.body_check(reservations[identity]["body"], terrain, identity)
            if not check["passed"]:
                terrain_issues.append(
                    {
                        "action_id": action["id"],
                        "agent_id": identity,
                        "phase": "sweep",
                        **check,
                    }
                )
            for x, y, _old, new in action["changed_cells"]:
                terrain[x, y] = new
            states[identity] = action["after"]
            # A deposit/cut can also invalidate the waiting teammate's stance.
            for peer in ids:
                body = geometry.body_polygon(
                    states[peer], scene.agents[peer]["geometry"]
                )
                check = scene.body_check(body, terrain, peer)
                if not check["passed"]:
                    terrain_issues.append(
                        {
                            "action_id": action["id"],
                            "agent_id": peer,
                            "phase": "after",
                            **check,
                        }
                    )
        clearance = max(
            scene.agents[identity]["geometry"]["clearance_m"] for identity in ids
        )
        check = geometry.conflict(reservations[ids[0]], reservations[ids[1]], clearance)
        if check["conflict"]:
            intermachine.append(
                {
                    "round": source_round,
                    "action_ids": [a["id"] for a in members],
                    **check,
                }
            )
    target, accepted = np.asarray(source["target"]), np.asarray(
        source["accepted_dump_mask"]
    )
    required = target < 0
    remaining = float(np.maximum(terrain[required] - target[required], 0).sum())
    loose = float(np.maximum(terrain[~accepted], 0).sum())
    loaded = sum(state["load_units"] for state in states.values())
    return {
        "intermachine_conflicts": intermachine,
        "terrain_conflicts": terrain_issues,
        "checked_rounds": len(by_round),
        "clearance_passed": not intermachine,
        "terrain_passed": not terrain_issues,
        "goal": {
            "complete": remaining <= EPS and loose <= EPS and loaded <= EPS,
            "remaining_excavation_units": remaining,
            "loose_outside_final_units": loose,
            "carrier_units": loaded,
        },
        "final_terrain": terrain.tolist(),
        "final_states": states,
    }


def retime_fixed_paths(
    source, actions, *, max_search_states=20000, max_schedule_steps=2000
):
    """Bounded shortest schedule on fixed paths with original material order.

    Advancing paths and held peers reserve full bodies/tools. Shared terrain at
    each search node follows its exact completed material prefix. The output is
    a geometric candidate, not proof that reordered native commands still work.
    """
    max_search_states = _integer(max_search_states, "max_search_states", minimum=1)
    max_schedule_steps = _integer(max_schedule_steps, "max_schedule_steps", minimum=1)
    scene = _Scene(source)
    ids = [agent["id"] for agent in source["agents"]]
    streams = [
        [a for a in actions if a["agent_id"] == identity and a["action"] != 7]
        for identity in ids
    ]
    states = [
        [scene.agents[identity]["initial_state"]] + [a["after"] for a in stream]
        for identity, stream in zip(ids, streams)
    ]
    reservations = [
        [
            geometry.reservation(
                a["before"], a["after"], scene.agents[identity]["geometry"]
            )
            for a in stream
        ]
        for identity, stream in zip(ids, streams)
    ]
    holds = [
        [
            geometry.reservation(state, state, scene.agents[identity]["geometry"])
            for state in row
        ]
        for identity, row in zip(ids, states)
    ]
    material = [action for action in actions if _material(action)]
    event_index = {action["id"]: index for index, action in enumerate(material)}
    markers = [[event_index.get(a["id"], -1) for a in stream] for stream in streams]
    prefix = [np.r_[0, np.cumsum(np.asarray(row) >= 0)] for row in markers]
    terrains = [np.asarray(source["initial_terrain"], dtype=float).copy()]
    for action in material:
        terrain = terrains[-1].copy()
        for x, y, _old, new in action["changed_cells"]:
            terrain[x, y] = new
        terrains.append(terrain)
    clearance = max(agent["geometry"]["clearance_m"] for agent in source["agents"])

    def check(progress, advance):
        if any(progress[a] + advance[a] > len(streams[a]) for a in range(2)):
            return {"passed": False, "reason": "end_of_path"}
        completed = int(sum(prefix[a][progress[a]] for a in range(2)))
        next_ids = sorted(
            markers[a][progress[a]]
            for a in range(2)
            if advance[a] and markers[a][progress[a]] >= 0
        )
        if next_ids != list(range(completed, completed + len(next_ids))):
            return {"passed": False, "reason": "global_material_order"}
        shapes = [
            reservations[a][progress[a]] if advance[a] else holds[a][progress[a]]
            for a in range(2)
        ]
        collision = geometry.conflict(shapes[0], shapes[1], clearance)
        if collision["conflict"]:
            return {"passed": False, "reason": "full_workspace_conflict", **collision}
        for a, identity in enumerate(ids):
            after_count = completed + len(next_ids)
            # A scheduled step has no duration/order proof for a moving peer.
            # Check its entire sweep against every material prefix in the step,
            # including transient soil and cuts before the next material action.
            for terrain_count in range(completed, after_count + 1):
                during = scene.body_check(
                    shapes[a]["body"], terrains[terrain_count], identity, terrain_count
                )
                if not during["passed"]:
                    return {
                        "passed": False,
                        "reason": "route_or_hold_terrain",
                        "agent_id": identity,
                        "material_prefix": terrain_count,
                        **during,
                    }
            after = scene.body_check(
                holds[a][progress[a] + advance[a]]["body"],
                terrains[after_count],
                identity,
                after_count,
            )
            if not after["passed"]:
                return {
                    "passed": False,
                    "reason": "post_work_terrain",
                    "agent_id": identity,
                    **after,
                }
        return {"passed": True}

    initial = check((0, 0), (0, 0))
    total = len(material)
    if not initial["passed"]:
        return {
            "status": "INITIAL_RESERVATION_CONFLICT",
            "steps": [],
            "complete": False,
            "explored_states": 1,
            "completed_material_events": 0,
            "total_material_events": total,
            "frontier": {"progress": [0, 0], "checks": [initial]},
            "native_replay_required": True,
        }
    queue, predecessor, distance = deque([(0, 0)]), {(0, 0): None}, {(0, 0): 0}
    goal, best, explored, bounded = None, (0, 0), 0, False
    edges = ((1, 1), (1, 0), (0, 1))
    lengths = tuple(map(len, streams))
    while queue and explored < max_search_states:
        progress = queue.popleft()
        explored += 1
        completed = int(sum(prefix[a][progress[a]] for a in range(2)))
        if (completed, sum(progress)) > (
            int(sum(prefix[a][best[a]] for a in range(2))),
            sum(best),
        ):
            best = progress
        if progress == lengths:
            goal = progress
            break
        if distance[progress] >= max_schedule_steps:
            bounded = True
            continue
        for advance in edges:
            next_progress = tuple(progress[a] + advance[a] for a in range(2))
            if next_progress in predecessor or not check(progress, advance)["passed"]:
                continue
            predecessor[next_progress] = progress, advance
            distance[next_progress] = distance[progress] + 1
            queue.append(next_progress)
    if goal is None:
        return {
            "status": "SEARCH_LIMIT" if queue or bounded else "FIXED_PATH_CONFLICT",
            "complete": False,
            "steps": [],
            "explored_states": explored,
            "total_material_events": total,
            "completed_material_events": int(sum(prefix[a][best[a]] for a in range(2))),
            "frontier": {
                "progress": list(best),
                "checks": [
                    dict(advance=list(edge), **check(best, edge)) for edge in edges
                ],
            },
            "native_replay_required": True,
            "unresolved": "requires replacement approaches, holding poses or work poses beyond fixed-path retiming",
        }
    reverse, progress = [], goal
    source_order = {a["id"]: i for i, a in enumerate(actions)}
    while predecessor[progress] is not None:
        before, advance = predecessor[progress]
        active = [streams[a][before[a]]["id"] for a in range(2) if advance[a]]
        reverse.append(
            {
                "action_ids": sorted(active, key=source_order.__getitem__),
                "held_agents": [ids[a] for a in range(2) if not advance[a]],
                "before_progress": list(before),
            }
        )
        progress = before
    steps = [{"step": i, **step} for i, step in enumerate(reversed(reverse))]
    return {
        "status": "COMPLETE_GEOMETRIC_CANDIDATE",
        "complete": True,
        "steps": steps,
        "explored_states": explored,
        "total_material_events": total,
        "completed_material_events": total,
        "scheduled_steps": len(steps),
        "native_replay_required": True,
        "checks": [
            "global_material_order",
            "per_agent_action_and_setup_order",
            "shared_sequential_terrain",
            "full_tool_and_body_sweeps",
            "held_peer_reservations",
            "complete_final_machine_states",
        ],
    }


def _travel(actions):
    result = defaultdict(float)
    for action in actions:
        result[action["agent_id"]] += float(
            np.linalg.norm(
                np.asarray(action["after"]["position_xy_m"])
                - action["before"]["position_xy_m"]
            )
        )
    return dict(result)


def postprocess(
    source,
    *,
    cleanup=True,
    retime=True,
    max_search_states=20000,
    max_schedule_steps=2000,
):
    source = validate_source(source)
    actions, edits = (
        clean_motion(source["actions"])
        if cleanup
        else (copy.deepcopy(source["actions"]), [])
    )
    # Revalidate every retained event against shared terrain and all setup state.
    validate_source({**source, "actions": actions})
    events, cycles = compile_events(source, actions)
    original_audit = audit(source, source["actions"])
    cleaned_audit = audit(source, actions)
    if not np.array_equal(
        original_audit["final_terrain"], cleaned_audit["final_terrain"]
    ) or any(
        not _same_state(
            original_audit["final_states"][identity],
            cleaned_audit["final_states"][identity],
        )
        for identity in original_audit["final_states"]
    ):
        raise ValueError("Cleanup changed a required final state")
    schedule = (
        retime_fixed_paths(
            source,
            actions,
            max_search_states=max_search_states,
            max_schedule_steps=max_schedule_steps,
        )
        if retime
        else {"status": "NOT_REQUESTED", "complete": False, "steps": []}
    )
    incomplete = [
        e["id"]
        for e in events
        if e["workspace"]["refinement"]["status"]
        in ("RADIAL_BANDS_INCOMPLETE", "MISSING_TOOL_WIDTH")
    ]
    status = (
        "GEOMETRIC_CANDIDATE" if schedule.get("complete") else "UNRESOLVED_FLEET_PLAN"
    )
    if incomplete:
        status = "INCOMPLETE_WORKSPACE_REFINEMENT"
    if not cleaned_audit["goal"]["complete"]:
        status = "INCOMPLETE_MATERIAL_GOAL"
    report = {
        "status": status,
        "material_contract_passed": True,
        "final_state_preserved": True,
        "material_events": len(events),
        "carrier_cycles": len(cycles),
        "load_unit": "soil_units",
        "native_replay_performed": False,
        "physical_execution_validated": False,
        "source_goal": original_audit["goal"],
        "goal": cleaned_audit["goal"],
        "cleanup": {
            "edited_actions": sum(len(e["action_ids"]) for e in edits),
            "edits": edits,
            "travel_before_m": _travel(source["actions"]),
            "travel_after_m": _travel(actions),
            "rounds_unchanged": True,
            "native_replay_required": bool(edits),
        },
        "original": {
            k: v
            for k, v in original_audit.items()
            if k not in ("final_terrain", "final_states")
        },
        "cleaned": {
            k: v
            for k, v in cleaned_audit.items()
            if k not in ("final_terrain", "final_states")
        },
        "workspace_refinement_incomplete": incomplete,
        "limits": [
            "native replay is required after cleanup or retiming",
            "no calibrated physical time or soil volume",
            "geometric interpolation does not model steering, arm trajectories or stability",
            "all workspaces remain reserved during waiting and travel",
            "radial bands refine recorded support without changing native material effects",
            "FIFO dependencies allocate available quantities, not unique physical soil provenance",
        ],
    }
    return {
        "kind": RESULT_SCHEMA,
        "schema_version": 1,
        "case_id": source.get("case_id"),
        "source": source,
        "grid": source["grid"],
        "agents": source["agents"],
        "actions": actions,
        "events": events,
        "cycles": cycles,
        "schedule": schedule,
        "report": report,
        "final_terrain": cleaned_audit["final_terrain"],
        "final_states": cleaned_audit["final_states"],
    }


def write_result(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "source": output / "fleet_source.json",
        "plan": output / "fleet_plan.json",
        "report": output / "fleet_report.json",
    }
    for key, value in (
        ("source", result["source"]),
        ("plan", result),
        ("report", result["report"]),
    ):
        files[key].write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {key: str(path) for key, path in files.items()}
