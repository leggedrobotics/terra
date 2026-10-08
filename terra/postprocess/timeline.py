"""Metric playback shared by converted solo plans and fleet postprocessing.

This is display data, not a Terra action replay or a robot command format. Grid
rows advance map Y, columns map X; origin_xy_m is the centre of cell [0, 0].
Headings, including cabin_yaw, are absolute map headings. Terrain changes are
sparse, flat YX indices with *new* native and loose heights in metres. Retaining
each returned route pose avoids straightening corners or deleting reversals.
"""

from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = "terra.postprocessed.v1"
ROUTE_STATES = {"checked", "failed", "missing", "unverified"}


def runs(mask):
    """Lossless [row, start_column, length] encoding; no grid downsampling."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("A workspace mask must be two dimensional")
    out = []
    for row, values in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, values, False]))
        out.extend([row, int(a), int(b - a)] for a, b in zip(edges[::2], edges[1::2]))
    return out


def _heading(mask, pose, origin, resolution):
    row, col = np.nonzero(mask)
    if not row.size:
        return float(pose[2])
    x, y = origin + np.array([col.mean(), row.mean()]) * resolution
    return math.atan2(y - pose[1], x - pose[0])


def _state(
    agent_id, pose, *, cabin_yaw=None, load=0.0, wheel_angle=0.0, shovel_lifted=0.0
):
    return dict(
        id=agent_id,
        pose=[float(v) for v in pose],
        cabin_yaw=float(pose[2] if cabin_yaw is None else cabin_yaw),
        load=float(load),
        wheel_angle=float(wheel_angle),
        shovel_lifted=float(shovel_lifted),
    )


def validate(data):
    """Reject malformed playback rather than silently truncating IDs or cells."""
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Playback requires {SCHEMA}")
    grid = data["grid"]
    rows, cols = grid["rows"], grid["cols"]
    if any(type(n) is not int or n <= 0 for n in (rows, cols)):
        raise ValueError("Grid dimensions must be positive integers")
    size = rows * cols
    if size > 1_048_576:
        raise ValueError("Playback grid exceeds the viewer's 1048576-cell limit")
    if not math.isfinite(grid["resolution_m"]) or grid["resolution_m"] <= 0:
        raise ValueError("Grid resolution must be positive and finite")
    if len(grid["origin_xy_m"]) != 2 or not np.isfinite(grid["origin_xy_m"]).all():
        raise ValueError("Grid origin must contain two finite coordinates")
    if not math.isfinite(grid.get("yaw_rad", 0.0)):
        raise ValueError("Grid yaw must be finite")
    ids = [a["id"] for a in data["agents"]]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(type(i) is not int or i < 0 for i in ids)
    ):
        raise ValueError("Machine IDs must be distinct nonnegative integers")
    for agent in data["agents"]:
        if agent["type"] not in (0, 1, 2):
            raise ValueError("Unknown machine type")
        if agent["action_type"] not in (0, 1) or len(agent["reach_m"]) != 2:
            raise ValueError("Invalid action type or reach bounds")
        lengths = [agent["width_m"], agent["length_m"], *agent["reach_m"]]
        if (
            not np.isfinite(lengths).all()
            or min(lengths[:2]) <= 0
            or min(lengths[2:]) < 0
            or lengths[2] > lengths[3]
        ):
            raise ValueError("Invalid machine dimensions or reach")

    def mask(value):
        for row, col, width in value:
            if any(type(n) is not int for n in (row, col, width)):
                raise ValueError("Mask runs must use integer indices")
            if not (0 <= row < rows and 0 <= col < cols and 0 < width <= cols - col):
                raise ValueError("Mask run extends outside the grid")

    for name in ("known", "obstacle", "final", "target"):
        if name in grid:
            mask(grid[name])
    for name in ("native_m", "loose_m"):
        values = np.asarray(data["initial"][name])
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Initial {name} must contain one finite value per cell")
        if name == "loose_m" and (values < 0).any():
            raise ValueError("Initial loose soil thickness cannot be negative")
    if "design_m" in grid and (
        len(grid["design_m"]) != size or not np.isfinite(grid["design_m"]).all()
    ):
        raise ValueError("Design heights must match the grid")
    workspaces = {}
    for workspace in data["workspaces"]:
        key = workspace["id"]
        if key in workspaces or workspace["agent_id"] not in ids:
            raise ValueError(
                "Workspace IDs must be unique and reference a known machine"
            )
        workspaces[key] = workspace
        for value in workspace["masks"].values():
            mask(value)

    def states(value):
        if len(value) != len(ids) or {a["id"] for a in value} != set(ids):
            raise ValueError("Every frame must preserve all machine IDs")
        for state in value:
            if (
                len(state["pose"]) != 3
                or not np.isfinite(
                    [
                        *state["pose"],
                        state["cabin_yaw"],
                        state["load"],
                        state["wheel_angle"],
                        state["shovel_lifted"],
                    ]
                ).all()
                or state["load"] < 0
                or state["shovel_lifted"] not in (0, 1)
            ):
                raise ValueError("Invalid machine state")

    states(data["initial"]["agents"])
    for frame in data["frames"]:
        states(frame["agents"])
        if frame.get("agent_id") is not None and frame["agent_id"] not in ids:
            raise ValueError("Frame refers to an unknown machine")
        if (
            frame.get("workspace_id") is not None
            and frame["workspace_id"] not in workspaces
        ):
            raise ValueError("Frame refers to an unknown workspace")
        if frame.get("route_status", "unverified") not in ROUTE_STATES:
            raise ValueError("Unknown route status")
        changed = set()
        for index, native, loose in frame["terrain_changes"]:
            if type(index) is not int or not 0 <= index < size or index in changed:
                raise ValueError("Changed cell index is invalid or duplicated")
            if not np.isfinite([native, loose]).all() or loose < 0:
                raise ValueError("Invalid terrain height")
            changed.add(index)
        owned = set()
        for work in frame.get("work", []):
            if work["agent_id"] not in ids:
                raise ValueError("Work refers to an unknown machine")
            for index in work["changed_indices"]:
                if index not in changed or index in owned:
                    raise ValueError("Work cells must attribute each changed cell once")
                owned.add(index)
        if changed and owned != changed:
            raise ValueError("Changed terrain requires explicit machine attribution")
    return data


def from_replay(result, *, station_validator=None):
    """Adapt the existing metric solo replay without recomputing soil or routes."""
    from . import replay as tmm_replay

    case = result["case"]
    native, loose = result["native_frames"], result["loose_frames"]
    if len(native) != len(result["timeline"]) + 1 or native.shape != loose.shape:
        raise ValueError("Replay terrain and timeline lengths differ")
    agent_id, agent_type = 0, 0
    source = Path(case.report["inputs"]["input_dir"]) / "terra_plan.json"
    if source.is_file() or source.with_suffix(".json.gz").is_file():
        from . import replay as tmm_replay

        identities = {
            (w["agent_index"], w["agent_type"])
            for w in tmm_replay.read_json(source)["waypoints"]
        }
        if len(identities) != 1:
            raise ValueError(
                "Solo replay has lost fleet identity; use fleet postprocessing"
            )
        agent_id, agent_type = next(iter(identities))
    first_pose = case.poses[0] if len(case.poses) else [*case.origin, 0.0]
    body = np.asarray(case.footprint)
    state = _state(agent_id, first_pose)
    converted_waypoints = None
    converted_path = case.conversion / "terra_plan.json"
    if converted_path.is_file() or converted_path.with_suffix(".json.gz").is_file():
        from . import replay as tmm_replay

        converted_waypoints = tmm_replay.read_json(converted_path)["waypoints"]
    data = dict(
        schema=SCHEMA,
        metadata=dict(
            title=case.tag,
            source="converted_plan",
            conversion=str(case.conversion),
            terrain_model="workspace soil forecast",
            motion_model="saved route poses; illustrative workspace arm motion",
            load_unit="m3",
            timing="illustrative, not physical duration",
            route_scope="saved offline Nav2 checks and current terrain replay; no controller execution",
            limits=list(tmm_replay.NOT_MODELED),
            validation=tmm_replay.validity(result) if "issues" in result else None,
        ),
        grid=dict(
            rows=int(case.shape[0]),
            cols=int(case.shape[1]),
            resolution_m=float(case.res),
            origin_xy_m=case.origin.tolist(),
            known=runs(case.known),
            obstacle=runs(case.obstacle),
            final=runs(case.final_ground),
            target=runs(case.target),
            design_m=result["design"].ravel().tolist(),
        ),
        agents=[
            dict(
                id=agent_id,
                type=agent_type,
                action_type=0,
                width_m=float(np.ptp(body[:, 1])),
                length_m=float(np.ptp(body[:, 0])),
                reach_m=[
                    float(result["rules"].band_min_m),
                    float(result["rules"].entry_max_m),
                ],
                body_xy_m=body.tolist(),
            )
        ],
        initial=dict(
            native_m=native[0].ravel().tolist(),
            loose_m=loose[0].ravel().tolist(),
            agents=[deepcopy(state)],
        ),
        workspaces=[],
        frames=[],
    )

    def append(
        phase,
        workspace_id,
        pose,
        *,
        terrain_index=None,
        work_kind=None,
        load=0.0,
        cabin_yaw=None,
        route_status="unverified",
        replay_step=None,
    ):
        nonlocal state
        state = _state(agent_id, pose, cabin_yaw=cabin_yaw, load=load)
        changed = []
        if terrain_index is not None:
            a, b = native[terrain_index].ravel(), loose[terrain_index].ravel()
            old_a, old_b = (
                native[terrain_index - 1].ravel(),
                loose[terrain_index - 1].ravel(),
            )
            indices = np.flatnonzero((a != old_a) | (b != old_b))
            changed = [[int(i), float(a[i]), float(b[i])] for i in indices]
        frame = dict(
            phase=phase,
            workspace_id=workspace_id,
            agent_id=agent_id,
            agents=[deepcopy(state)],
            terrain_changes=changed,
            route_status=route_status,
            replay_step=replay_step,
            work=(
                [
                    dict(
                        agent_id=agent_id,
                        kind=work_kind,
                        changed_indices=[c[0] for c in changed],
                    )
                ]
                if changed
                else []
            ),
        )
        data["frames"].append(frame)

    for k, event in enumerate(result["events"]):
        workspace_id = event["workspace"]
        route = case.routes.get((k, k + 1)) if k else None
        evidence = event.get("route") or {}
        route_status = (
            "unverified" if not k else "missing" if route is None else "failed"
        )
        if route is not None and route.get("path"):
            route_status = (
                "checked"
                if (
                    route.get("passed") is True
                    and not evidence.get("blocked", True)
                    and evidence.get("checker_passed") is not False
                )
                else "failed"
            )
            if route_status != "failed" and not np.allclose(
                state["pose"], route["path"][0], atol=1e-4, rtol=0
            ):
                route_status = "unverified"
            last = route["path"][-1]
            nominal = case.poses[k]
            tolerance = float(
                case.report.get("dump_rule", {}).get("station_tolerance_m", 0.0)
            )
            endpoint_valid = (
                math.dist(last[:2], nominal[:2]) <= tolerance + 1e-5
                and abs(math.remainder(last[2] - nominal[2], 2 * math.pi)) <= 0.1
            )
            if converted_waypoints is not None:
                workspace = dict(
                    pose=nominal,
                    radial_workspace=converted_waypoints[2 * k]["radial_workspace"],
                )
                recorded_arrival = route.get("station_arrival")
                if station_validator is not None:
                    arrival_check = station_validator(last, workspace, tolerance)
                else:
                    arrival_check = recorded_arrival
                # A coordinate bound alone does not establish radial-workspace admission.
                # Missing explicit evidence keeps the endpoint at its nominal workspace.
                endpoint_valid = (
                    endpoint_valid
                    and isinstance(arrival_check, dict)
                    and arrival_check.get("passed") is True
                )
            if route.get("goal") is not None and not np.allclose(
                route["goal"], nominal, atol=1e-4, rtol=0
            ):
                endpoint_valid = False
            if not endpoint_valid and route_status != "failed":
                route_status = "unverified"
            for pose in route["path"]:
                append(
                    "drive",
                    workspace_id,
                    pose,
                    route_status=route_status,
                    replay_step=event["steps"]["cut"] - 1,
                )
            arrival = state["pose"]
            if not endpoint_valid:
                arrival = nominal.tolist()
                append(
                    "relocate",
                    workspace_id,
                    arrival,
                    route_status=route_status,
                    replay_step=event["steps"]["cut"] - 1,
                )
        else:
            arrival = case.poses[k].tolist()
            # This discontinuity is a station-order connector, not a navigation path.
            if k:
                append(
                    "relocate",
                    workspace_id,
                    arrival,
                    route_status=route_status,
                    replay_step=event["steps"]["cut"] - 1,
                )
        workspace = dict(
            id=workspace_id,
            agent_id=agent_id,
            kind=event["kind"],
            pose=case.poses[k].tolist(),
            source_pair=event["source_pair"],
            route_status=route_status,
            masks=dict(
                dig=runs(case.support[k]),
                finish=runs(case.completion[k]),
                dump=runs(case.dump_centres[k]),
                deposit=runs(case.converter_deposit[k]),
            ),
        )
        data["workspaces"].append(workspace)
        cut_heading = _heading(case.support[k], arrival, case.origin, case.res)
        append(
            "arrive",
            workspace_id,
            arrival,
            cabin_yaw=cut_heading,
            route_status=route_status,
            replay_step=event["steps"]["cut"] - 1,
        )
        cut = event["steps"]["cut"]
        append(
            "cut",
            workspace_id,
            arrival,
            terrain_index=cut,
            work_kind="dig" if event["kind"] == "excavate" else "collect",
            load=event["cut"]["payload_m3"],
            cabin_yaw=cut_heading,
            route_status=route_status,
            replay_step=cut,
        )
        for load in event["dump"]["loads"]:
            heading = math.atan2(load["y_m"] - arrival[1], load["x_m"] - arrival[0])
            append(
                "dump",
                workspace_id,
                arrival,
                terrain_index=load["step"],
                work_kind="dump",
                cabin_yaw=heading,
                route_status=route_status,
                replay_step=load["step"],
            )
    return validate(data)


def from_fleet(result, *, variant="auto"):
    """Display a complete fleet proposal, including rejected motions and held peers.

    ``scheduled`` retains the chosen schedule's explicit substep order. Display
    durations are illustrative; they are not concurrent controller commands.
    Soil units use the source's declared height conversion without inventing a
    physical volume model. String source identities are retained beside slots.
    """
    from . import fleet as tmm_fleet
    from . import fleet_geometry as geometry

    if (
        result.get("kind") != tmm_fleet.RESULT_SCHEMA
        or result.get("schema_version") != 1
    ):
        raise ValueError("Expected a fleet postprocessing result")
    if variant not in ("auto", "original", "cleaned", "scheduled"):
        raise ValueError("Unknown fleet playback variant")
    source = tmm_fleet.validate_source(result["source"])
    scheduled = result["schedule"].get("complete", False)
    if variant == "auto":
        variant = "scheduled" if scheduled else "cleaned"
    if variant == "scheduled" and not scheduled:
        raise ValueError(
            "Cannot display an incomplete schedule as a complete fleet plan"
        )
    actions = source["actions"] if variant == "original" else result["actions"]
    tmm_fleet.validate_source({**source, "actions": actions})
    by_id = {action["id"]: action for action in actions}
    steps = (
        result["schedule"]["steps"]
        if variant == "scheduled"
        else [
            {"step": action["round"], "action_ids": [action["id"]]}
            for action in actions
        ]
    )
    selected = [key for step in steps for key in step["action_ids"]]
    required = {
        action["id"]
        for action in actions
        if variant != "scheduled" or action["action"] != 7
    }
    if len(selected) != len(required) or set(selected) != required:
        raise ValueError(
            "Fleet timeline must retain every required action exactly once"
        )
    profiles = {agent["id"]: agent for agent in source["agents"]}
    ids = {identity: index for index, identity in enumerate(profiles)}
    states = {
        identity: deepcopy(agent["initial_state"])
        for identity, agent in profiles.items()
    }
    grid = source["grid"]
    resolution, height, yaw = (
        grid["resolution_m"],
        grid["cell_height_m"],
        grid["yaw_rad"],
    )
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    origin = np.asarray(grid["origin_xy_m"]) + rotation @ np.full(2, resolution / 2)
    terrain = np.asarray(source["initial_terrain"], dtype=float).T.copy()
    rows, cols = terrain.shape

    def display_state(identity, state):
        return _state(
            ids[identity],
            [*state["position_xy_m"], state["base_yaw_rad"]],
            cabin_yaw=state["base_yaw_rad"] + state["cabin_yaw_rad"],
            load=state["load_units"],
            wheel_angle=state["wheel_angle_rad"],
            shovel_lifted=float(state["shovel_lifted"]),
        )

    def cell_runs(cells):
        mask = np.zeros((rows, cols), dtype=bool)
        for x, y in cells:
            mask[y, x] = True
        return runs(mask)

    report = result["report"]
    audit = report["original" if variant == "original" else "cleaned"]
    plan_status = (
        report["status"]
        if variant != "original"
        else (
            "ORIGINAL_RECORDED_TRACE"
            if audit["clearance_passed"] and audit["terrain_passed"]
            else "ORIGINAL_TRACE_WITH_CONFLICTS"
        )
    )
    data = dict(
        schema=SCHEMA,
        metadata=dict(
            title=f"Fleet {source.get('case_id', '')} · {variant}",
            source="fleet_postprocess",
            variant=variant,
            plan_status=plan_status,
            schedule_status=(
                result["schedule"]["status"]
                if variant != "original"
                else "SOURCE_ORDER"
            ),
            omitted_wait_ids=[
                action["id"] for action in actions if action["id"] not in required
            ],
            source_ids=ids,
            terrain_model="recorded native soil units times declared cell_height_m",
            motion_model="ordered recorded substeps; full body and tool reservations",
            load_unit="soil_units",
            timing="illustrative substeps, not physical duration",
            route_scope="declared geometric model; native replay and physical execution not validated",
            native_replay_performed=False,
            goal=report["goal"],
            incomplete_workspaces=report["workspace_refinement_incomplete"],
            limits=report["limits"],
            provenance=source.get("provenance", {}),
        ),
        grid=dict(
            rows=rows,
            cols=cols,
            resolution_m=resolution,
            origin_xy_m=origin.tolist(),
            yaw_rad=yaw,
            known=runs(np.ones_like(terrain, dtype=bool)),
            obstacle=runs(np.asarray(source["obstacles"]).T),
            final=runs(np.asarray(source["accepted_dump_mask"]).T),
            target=runs(np.asarray(source["target"]).T < 0),
            design_m=(np.minimum(np.asarray(source["target"]).T, 0) * height)
            .ravel()
            .tolist(),
        ),
        agents=[
            dict(
                id=ids[identity],
                source_id=identity,
                name=identity,
                type=0 if agent["type"] == "excavator" else 2,
                action_type=0,
                width_m=agent["geometry"]["body_width_m"],
                length_m=agent["geometry"]["body_length_m"],
                reach_m=[
                    agent["geometry"]["work_min_radius_m"],
                    agent["geometry"]["work_reach_m"],
                ],
            )
            for identity, agent in profiles.items()
        ],
        initial=dict(
            native_m=(np.minimum(terrain, 0) * height).ravel().tolist(),
            loose_m=(np.maximum(terrain, 0) * height).ravel().tolist(),
            agents=[
                display_state(identity, state) for identity, state in states.items()
            ],
        ),
        workspaces=[],
        frames=[],
    )
    events = {event["id"]: event for event in result["events"]}
    next_workspace, route_ids = {}, {}
    for key in reversed(selected):
        action = by_id[key]
        identity = action["agent_id"]
        if key in events:
            next_workspace[identity] = key
        route_ids[key] = f"{identity}:{next_workspace.get(identity, 'final_transit')}"
    for event in events.values():
        workspace = event["workspace"]
        pose = workspace["pose"]
        data["workspaces"].append(
            dict(
                id=workspace["id"],
                agent_id=ids[workspace["agent_id"]],
                kind=workspace["kind"],
                pose=[*pose["position_xy_m"], pose["base_yaw_rad"]],
                source_pair=None,
                masks={
                    name: cell_runs(cells) for name, cells in workspace["masks"].items()
                },
                refined_geometry=workspace["refined_geometry"],
                reservation_geometry=workspace["reservation_geometry"],
                refinement=workspace["refinement"],
                route_status="unverified",
            )
        )
    conflicts = audit["intermachine_conflicts"] + audit["terrain_conflicts"]
    work_kinds = {
        "excavate": "dig",
        "collect": "collect",
        "deliver": "dump",
        "stage": "dump",
        "redistribute": "transfer",
    }
    for step in steps:
        step_actions = [by_id[key] for key in step["action_ids"]]
        active = {action["agent_id"]: action for action in step_actions}
        reservations = [
            dict(
                agent_id=ids[identity],
                geometry=geometry.polygon_record(
                    geometry.reservation(
                        active[identity]["before"] if identity in active else state,
                        active[identity]["after"] if identity in active else state,
                        profiles[identity]["geometry"],
                    )["occupied"]
                ),
            )
            for identity, state in states.items()
        ]
        for action in step_actions:
            identity = action["agent_id"]
            if not tmm_fleet._same_state(states[identity], action["before"]):
                raise ValueError("Selected schedule changes an agent's action order")
            changed = []
            for x, y, old, new in action["changed_cells"]:
                if abs(terrain[y, x] - old) > tmm_fleet.EPS:
                    raise ValueError("Selected schedule changes shared material order")
                terrain[y, x] = new
                changed.append(
                    [y * cols + x, min(new, 0) * height, max(new, 0) * height]
                )
            event = events.get(action["id"])
            state = action["after"]
            states[identity] = deepcopy(state)
            phase = (
                event["kind"]
                if event
                else (
                    "wait"
                    if action["action"] == 7
                    else (
                        "drive"
                        if not np.allclose(
                            action["before"]["position_xy_m"], state["position_xy_m"]
                        )
                        else (
                            "turn"
                            if any(
                                abs(action["before"][key] - state[key]) > 1e-9
                                for key in ("base_yaw_rad", "cabin_yaw_rad")
                            )
                            else "setup"
                        )
                    )
                )
            )
            issues = (
                []
                if variant == "scheduled"
                else [
                    issue
                    for issue in conflicts
                    if issue.get("round") == action["round"]
                    or issue.get("action_id") == action["id"]
                ]
            )
            data["frames"].append(
                dict(
                    phase=phase,
                    workspace_id=event["id"] if event else None,
                    agent_id=ids[identity],
                    source_action_id=action["id"],
                    schedule_step=step["step"],
                    source_round=action["round"],
                    action=action["action"],
                    requested_action=action.get(
                        "requested_action",
                        action.get("source_action", action["action"]),
                    ),
                    effective_action=action.get(
                        "effective_action",
                        action.get("source_action", action["action"]),
                    ),
                    workspace_blocked=action.get("workspace_blocked", False),
                    route_id=route_ids[action["id"]],
                    agents=[display_state(key, value) for key, value in states.items()],
                    terrain_changes=changed,
                    work=(
                        [
                            dict(
                                agent_id=ids[identity],
                                kind=work_kinds[event["kind"]],
                                changed_indices=[cell[0] for cell in changed],
                            )
                        ]
                        if changed
                        else []
                    ),
                    route_status=(
                        "failed"
                        if issues
                        else "checked" if variant == "scheduled" else "unverified"
                    ),
                    reservations=reservations,
                    conflicts=issues,
                )
            )
    if not np.allclose(terrain.T, result["final_terrain"], rtol=0, atol=tmm_fleet.EPS):
        raise ValueError(
            "Fleet timeline final terrain differs from the postprocessed plan"
        )
    return validate(data)


def write(data, path):
    validate(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
    return path


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return validate(json.load(stream))
