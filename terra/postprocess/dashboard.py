"""Build one offline dashboard comparing converted Terra plans and native replays.

A manifest groups named cases and conversion versions, with optional saved
routes and original terra.viewer3d.v1 recordings. The page includes modeled
terrain history, verdicts, checks, and the packaged studio player. Runtime
validation is supplied explicitly by the execution adapter.
"""

import base64
import gzip
import html
import json
import math
import re
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage

from . import replay as plan_replay
from . import seams
from .render import player_bundle

TEMPLATE = Path(__file__).resolve().parent / "assets" / "dashboard.html"
TERRA3D_SCHEMA = "terra.viewer3d.v1"
# Cell state bits of the page raster (dig state in the low 3 bits).
REQUIRED_LEFT, OPTIONAL_LEFT, PARTIAL, FINISHED = 1, 2, 3, 4
FELL_BACK_BIT = 16
TEMPORARY_BIT = 32  # soil of loads released off the final dump zones, outside them and the required ground
PILE_STEP_M = 0.02  # pile thickness quantum of the page raster
FIX_RADIUS_M = 0.45  # dump permission radius of the suggested set-dump edit
CHECK_NAMES = {
    "converter_plan": "Converter plan",
    "coverage": "Coverage",
    "completion_band": "Completion band",
    "cut_clear_of_body": "Cut clear of body",
    "arrival": "Arrival",
    "routes": "Routes",
    "dump_admission": "Dump admission",
    "pile_clearance": "Pile clearance",
    "spoil_at_end": "Spoil at end",
}


def _pack(array, dtype):
    return base64.b64encode(
        gzip.compress(np.ascontiguousarray(array, dtype=dtype).tobytes(), 6)
    ).decode("ascii")


def _pack_bits(mask):
    """A boolean raster as gzip-compressed bits (numpy packbits order), base64."""
    return base64.b64encode(
        gzip.compress(
            np.packbits(np.ascontiguousarray(mask, dtype=bool).ravel()).tobytes(), 6
        )
    ).decode("ascii")


def _round(xy, digits=3):
    return [[round(float(x), digits), round(float(y), digits)] for x, y in xy]


def headline(entry):
    """A short title, key-number chips and an optional note for one checklist entry."""
    kind, v = entry.get("kind", ""), entry
    m = lambda value, digits=2: f"{float(value):.{digits}f}"  # noqa: E731
    if kind in ("dump_into_pit", "dump_touches_pit"):
        verb = "put soil in" if kind == "dump_into_pit" else "touch"
        chips = [f"{m(v['pit_m3'], 3)} m³ in the pit"]
        note = ""
        fix = v.get("fix")
        if fix:
            note = (
                f"One pile of {m(fix['pile_m3'])} m³ spreads {m(fix['pile_toe_radius_m'])} m: its centre needs "
                f"{m(fix['needed_centre_distance_m'])} m from the pit, now {m(fix['current_centre_distance_m'])} m."
            )
            if "converter_used_fallback_patch_at" in fix:
                x, y = fix["converter_used_fallback_patch_at"]
                note += f" The converter ignored this step's dump permission and used its own patch near ({x}, {y})."
            if fix.get("terra_step") is None:
                note += " Added by the converter: edit a neighbouring Terra step."
            elif "suggested_centre_xy_m" not in fix:
                note += " No final-zone centre within reach fits one pile: move the station or split its soil."
        return f"{v['loads']} of {v['of']} loads {verb} the excavation", chips, note
    if kind == "converter_step_rejected":
        chips = [
            f"{reason.replace('_', ' ')} ×{count}"
            for reason, count in list(v.get("reasons", {}).items())[:2]
        ]
        note = ""
        fix = v.get("fix")
        if fix:
            note = (
                f"Its {m(fix['pile_m3'])} m³ pile spreads {m(fix['pile_toe_radius_m'])} m: a dump centre needs "
                f"{m(fix['needed_centre_distance_m'])} m from the pit."
            )
            if "suggested_centre_xy_m" not in fix:
                note += " No final-zone centre within reach fits: move the station or split its soil."
        return f"Terra step {v['terra_step']} rejected by the converter", chips, note
    if kind == "runtime_refused":
        return "The runtime refuses the plan", [], v["message"]
    if kind == "no_plan":
        chips = [f"coverage {m(v['coverage_pct'])} %"]
        if v.get("residual_m2") is not None:
            chips.append(f"residual {m(v['residual_m2'], 3)} m²")
        return "The converter wrote no executable plan", chips, ""
    if kind == "required_left":
        return (
            "Required soil left",
            [f"{m(v['area_m2'])} m²", f"{m(v['volume_m3'], 3)} m³"],
            "",
        )
    if kind == "required_left_within_tolerance":
        return (
            "Required soil left within tolerance",
            [f"{m(v['area_m2'])} m²", f"{m(v['volume_m3'], 3)} m³"],
            "",
        )
    if kind == "temporary_spoil_left":
        return (
            "Soil of loads released off the final zones left outside them",
            [f"{m(v['volume_m3'], 3)} m³ at the end"],
            "",
        )
    if kind == "loose_in_finished_excavation":
        return "Loose soil in finished excavation", [f"{m(v['volume_m3'], 3)} m³"], ""
    if kind == "completion_out_of_band":
        return (
            f"{v['cells']} completion cells outside the cutting band",
            [f"nearest {m(v['nearest_m'])} m", f"farthest {m(v['farthest_m'])} m"],
            "",
        )
    if kind == "arrival_blocked":
        return (
            "Machine body on blocked ground",
            [str(name) for name in v.get("hazards", {})],
            "",
        )
    if kind == "arrival_unverified":
        return (
            "Machine body on spoil of 0.5 m or more",
            [],
            "No robot limit identified for this height.",
        )
    leg = f" {v['workspace'] - 1} → {v['workspace']}" if v.get("workspace") else ""
    if kind == "route_blocked":
        return (
            f"Saved Nav2 route{leg} blocked",
            [f"pose {v['pose_index']}"] + [str(h) for h in v.get("hazards", [])],
            "",
        )
    if kind == "route_not_found":
        return (
            f"Nav2 found no route{leg}",
            [
                (
                    "start clear on the replayed terrain"
                    if v.get("start_clear")
                    else "start blocked"
                )
            ],
            str(v.get("error", ""))[:160],
        )
    if kind == "route_check_failed":
        chips = (
            [f"{v['unchecked_legs']} later drives not checked"]
            if v.get("unchecked_legs")
            else []
        )
        reason = v.get("reason", "")
        return (
            f"Route{leg} fails the offline Nav2 check",
            chips,
            reason[:1].upper() + reason[1:] + ".",
        )
    if kind == "route_over_high_spoil":
        return (
            f"Route{leg} crosses spoil of 0.5-1.0 m",
            [f"{v['poses']} poses"],
            "Drivable with chassis balancing (Lorenzo, 29 September).",
        )
    if kind == "cut_under_body":
        return "Completion cells under the machine body", [f"{v['cells']} cells"], ""
    if kind == "tiny_workspace":
        return (
            "Tiny workspace",
            [f"finishes {m(v['finished_m2'])} m²", f"cuts {m(v['cut_m3'], 3)} m³"],
            "",
        )
    if kind == "dump_ros_refuses":
        return f"ROS admits no dump centre for {v['loads']} of {v['of']} loads", [], ""
    if kind == "dump_out_of_reach":
        return f"{v['loads']} of {v['of']} loads out of dump reach", [], ""
    if kind == "no_dump_region":
        return "Soil to dump but no dump region", [f"{m(v['payload_m3'], 3)} m³"], ""
    return entry["text"].split(": ", 1)[-1][:120], [], ""


def dump_rule(case):
    """The dump ground the converter recorded (None for runs before it was recorded) and its forecast spoil."""
    report = case.report
    alternative = report.get("dump_ground_alternative")
    forecast = report.get("temporary_spoil_to_rearrange_m3")
    return dict(
        rule=report.get("dump_ground"),
        alternative=(
            None
            if alternative is None
            else dict(
                rule=alternative.get("dump_ground"),
                complete=bool(alternative.get("complete_geometric_plan")),
                residual_m2=alternative.get("continuous_required_residual_m2"),
            )
        ),
        height_model=bool(report.get("spoil_model")),
        converter_m3=None if forecast is None else round(float(forecast), 3),
    )


def edit_command(case, fix):
    """The set-dump edit a fix suggests, with this version's own evaluation settings when its request.json exists."""
    if not fix or fix.get("terra_step") is None or "suggested_centre_xy_m" not in fix:
        return None
    step = fix["terra_step"]
    x, y = fix["suggested_centre_xy_m"]
    edit = f"--step {step} --center {x} {y} --radius {FIX_RADIUS_M}"
    version = case.conversion.parent
    if not (version / "request.json").is_file():
        return f"plan set-dump PLAN {edit} --out NEW"
    request = json.loads((version / "request.json").read_text())
    out = f"{version}_step{step}_dump"
    flags = []
    if request.get("repair"):
        flags.append("--repair")
    if request.get("compact_dump_regions"):
        flags.append("--compact-dump-regions")
    if request.get("drivable_spoil_height_m") is not None:
        flags.append(
            f"--drivable-spoil-height-m {request['drivable_spoil_height_m']:g}"
        )
    if request.get("required_residual_tolerance_m2") is not None:
        flags.append(
            f"--required-residual-tolerance-m2 {request['required_residual_tolerance_m2']:g}"
        )
    if request.get("target_depth_m", 0.5) != 0.5:
        flags.append(f"--target-depth-m {request['target_depth_m']:g}")
    return " ".join(
        [
            f"python3 scripts/TerraMapMaker/tmm.py plan set-dump {version}/source/terra_plan.json {edit}",
            f"--out {out}.json --source {request['source']} --profile {request['base_profile']} --review-out {out}",
            *flags,
        ]
    )


def terra_plan(case, window):
    """The Terra plan a conversion came from, for the page's Terra view.

    ``action`` is Terra's end state on the page grid: per cell the loads of soil Terra left there (n > 0), or a
    negative count where it dug (each waypoint's ``dump_mask`` minus ``dug_mask``, summed for the native-plan overview
    it). ``stations`` are the BASE poses of its dig/dump pairs in the map frame and whether the pair collects dumped
    soil, indexed like the workspaces' ``source``. ``steps`` are the rollout steps of each pair's dig and dump (the
    waypoints' ``step``; None when a waypoint has none), which link the pairs to a Terra 3D replay of that rollout.
    None when the Terra plan is not on disk (synthetic cases).
    """
    path = Path(case.report["inputs"]["input_dir"]) / "terra_plan.json"
    if not plan_replay.json_exists(path):
        return None
    plan = plan_replay.read_json(path)
    waypoints = plan["waypoints"]
    action = np.zeros(
        np.asarray(plan["waypoints"][0]["dug_mask"]).shape if waypoints else (1, 1),
        dtype=np.int16,
    )
    for waypoint in waypoints:
        action += np.asarray(waypoint["dump_mask"], dtype=np.int16) - np.asarray(
            waypoint["dug_mask"], dtype=np.int16
        )
    on_grid, _ = plan_replay._native_on_grid(
        action, plan["alignment"], case.origin, case.res, case.target.shape, 0
    )
    stations = [
        [
            round(x, 3),
            round(y, 3),
            round(yaw, 3),
            waypoints[2 * pair]["workspace_type"] != "excavate",
        ]
        for pair, (x, y, yaw) in enumerate(case.native_poses)
    ]
    steps = (
        [
            [int(waypoints[2 * pair]["step"]), int(waypoints[2 * pair + 1]["step"])]
            for pair in range(len(waypoints) // 2)
        ]
        if all("step" in waypoint for waypoint in waypoints)
        else None
    )
    return dict(
        action=_pack(np.clip(on_grid[window], -127, 127).ravel(), np.int8),
        stations=stations,
        steps=steps,
    )


def _read_replay(path):
    from terra.viewer3d import load_replay

    return load_replay(path)


def _deltas(stack, dtype):
    """A layer of every frame as its first frame plus, per later frame, the cells that change and their new values.

    ``off[f]`` counts the changes of frames 1..f, so frame f changes entries ``off[f - 1]`` to ``off[f]`` of ``idx``
    (flat cell index) and ``val``; the page applies them in order like the machine plan's history.
    """
    flat = stack.reshape(len(stack), -1)
    offsets, index, value = [0], [], []
    for frame in range(1, len(flat)):
        changed = np.flatnonzero(flat[frame] != flat[frame - 1])
        index.append(changed)
        value.append(flat[frame, changed])
        offsets.append(offsets[-1] + changed.size)
    return dict(
        init=_pack(flat[0], dtype),
        off=_pack(np.asarray(offsets), np.uint32),
        idx=_pack(np.concatenate(index) if index else np.zeros(0), np.uint16),
        val=_pack(np.concatenate(value) if value else np.zeros(0), dtype),
    )


def _terra_events(action, loaded, target, steps):
    """The digs and dumps of a Terra episode and their dig/dump pairs (Terra steps, as in the native plan).

    Each frame whose action map changes is one event: a dig when the load grows, a dump when it shrinks. Its place
    is the centroid of the cells that lost soil (dig) or gained it (dump), weighted by the change; ``units`` is the
    load change. A dig that lifts soil Terra dumped earlier is a collection (``collect``) and lists the steps whose
    soil it lifts (``collects``: per lifted cell, the last step that dumped there). A dump records the share of its
    soil on Terra's dump target (``zone``, target > 0) and the first later step that lifts any of it
    (``collected_by``). A dig and the next dump form a pair.
    """
    events, cells = [], []
    for frame in range(1, len(action)):
        delta = action[frame].astype(np.int32) - action[frame - 1]
        if not delta.any():
            continue
        load = int(loaded[frame] - loaded[frame - 1])
        kind = (
            "dig"
            if load > 0 and (delta < 0).any()
            else "dump" if load < 0 and (delta > 0).any() else "terrain"
        )
        moved = (
            np.maximum(-delta, 0)
            if kind == "dig"
            else np.maximum(delta, 0) if kind == "dump" else np.abs(delta)
        )
        rows, cols = np.nonzero(moved)
        weights = moved[rows, cols]
        event = dict(
            frame=frame,
            step=int(steps[frame]),
            kind=kind,
            units=abs(load) if kind != "terrain" else int(weights.sum()),
            cells=int(rows.size),
            row=round(float(np.average(rows, weights=weights)), 2),
            col=round(float(np.average(cols, weights=weights)), 2),
        )
        if kind == "dig":
            event["collect"] = bool(((action[frame - 1] > 0) & (delta < 0)).any())
        if kind == "dump":
            event["zone"] = round(
                float(moved[target > 0].sum() / max(int(moved.sum()), 1)), 3
            )
        events.append(event)
        cells.append(set(zip(rows.tolist(), cols.tolist())))
    pairs, pending = [], None
    for k, event in enumerate(events):
        if event["kind"] == "dig":
            pending = k
        elif event["kind"] == "dump" and pending is not None:
            pairs.append(dict(dig=pending, dump=k))
            pending = None
    pair_of = {pair["dig"]: p for p, pair in enumerate(pairs)}
    for pair in pairs:
        dumped = cells[pair["dump"]]
        later = (
            pair_of[k]
            for k in range(pair["dump"] + 1, len(events))
            if k in pair_of and cells[k] & dumped
        )
        events[pair["dump"]]["collected_by"] = next(later, None)
    last_dump = {}  # cell -> the last Terra step that dumped there
    step_of = {pair["dump"]: p for p, pair in enumerate(pairs)}
    for k, event in enumerate(events):
        if k in step_of:
            last_dump.update((cell, step_of[k]) for cell in cells[k])
        elif k in pair_of and event["collect"]:
            event["collects"] = sorted(
                {last_dump[cell] for cell in cells[k] if cell in last_dump}
            )
    return events, pairs


def terra3d_data(path):
    """A Terra 3D replay as compact page data: the Terra episode a job's versions were converted from.

    The replay is a ``terra.viewer3d.v1`` recording (JSON or JSON.gz, ``terra.viewer3d.ReplayRecorder``). The page
    rebuilds every frame in the viewer's snapshot format: the fixed layers once (target, obstacles, static
    dumpability), the changing ones (soil heights, current dumpability, the machine's interaction workspace) as
    their first frame plus the cells each frame changes, and per frame the machines' poses and loads. Traversability
    is left out (the view does not draw it). ``events`` and ``pairs`` are the digs and dumps (``_terra_events``).
    """
    replay = _read_replay(path)
    frames = replay["frames"]
    grid = frames[0]["grid"]

    def stack(name, dtype):
        if any(frame["maps"].get(name) is None for frame in frames):
            return None
        return np.stack(
            [np.asarray(frame["maps"][name], dtype=dtype) for frame in frames]
        )

    action = stack("action", np.int16)
    if action.min() < -128 or action.max() > 127:
        raise ValueError(f"{path}: soil heights outside the page's int8 range")
    fixed_layers = dict(
        target=stack("target", np.int16),
        padding=stack("padding", np.uint8),
        dump_static=stack("dumpability_static", np.uint8),
    )
    for name, layer in fixed_layers.items():
        if layer is not None and (layer != layer[0]).any():
            raise ValueError(f"{path}: {name} changes during the episode")
    fixed_keys = ("id", "type", "action_type", "width", "height", "reach")
    fixed = [{key: agent[key] for key in fixed_keys} for agent in frames[0]["agents"]]
    if any(
        [{key: agent[key] for key in fixed_keys} for agent in frame["agents"]] != fixed
        for frame in frames
    ):
        raise ValueError(f"{path}: the machines change during the episode")
    if any(frame["done"] for frame in frames[:-1]):
        raise ValueError(f"{path}: frames after the end of the episode")
    loaded = np.array(
        [sum(agent["loaded"] for agent in frame["agents"]) for frame in frames]
    )
    steps = [frame["step"] for frame in frames]
    joint = "joint_actions" in frames[0]
    if any(("joint_actions" in frame) != joint for frame in frames):
        raise ValueError(
            "Cannot mix joint and sequential frames in one dashboard replay"
        )
    events, pairs = (
        ([], [])
        if joint
        else _terra_events(action, loaded, fixed_layers["target"][0], steps)
    )
    interaction, dumpability = stack("interaction", np.uint8), stack(
        "dumpability", np.uint8
    )
    metadata = replay["metadata"]
    return dict(
        title=metadata["title"],
        source=metadata["source"],
        file=str(path),
        grid=dict(rows=grid["rows"], cols=grid["cols"], tile=grid["tile_size_m"]),
        count=len(frames),
        step=steps,
        action=[-1 if frame["action"] is None else frame["action"] for frame in frames],
        actor=[
            -1 if frame["actor_id"] is None else frame["actor_id"] for frame in frames
        ],
        current=[frame["current_agent"] for frame in frames],
        done=bool(frames[-1]["done"]),
        task_done=bool(frames[-1]["task_done"]),
        target=_pack(fixed_layers["target"][0].ravel(), np.int8),
        padding=_pack(fixed_layers["padding"][0].ravel(), np.uint8),
        dump_static=(
            None
            if fixed_layers["dump_static"] is None
            else _pack(fixed_layers["dump_static"][0].ravel(), np.uint8)
        ),
        layers=dict(
            action=_deltas(action, np.int8),
            dumpability=_deltas(dumpability, np.uint8),
            interaction=None if interaction is None else _deltas(interaction, np.uint8),
        ),
        agents=dict(
            fixed=fixed,
            track=[
                [
                    [
                        agent["position"][0],
                        agent["position"][1],
                        agent["base_yaw"],
                        agent["cabin_yaw"],
                        agent["loaded"],
                        agent["wheel_angle"],
                        agent["shovel_lifted"],
                    ]
                    for agent in frame["agents"]
                ]
                for frame in frames
            ],
        ),
        events=events,
        pairs=pairs,
        joint=joint,
        frame_extras=[
            {
                key: frame[key]
                for key in (
                    "joint_actions",
                    "effective_joint_actions",
                    "workspace_blocked",
                    "workspace_polygons",
                )
                if key in frame
            }
            for frame in frames
        ],
    )


def terra_linked(terra, episode):
    """Whether a version's Terra plan is the episode's rollout: pair p digs and dumps at its waypoints' steps.

    A waypoint's ``step`` is the rollout step that acted; the frame after it carries ``step + 1``.
    """
    if terra is None or episode is None or terra.get("steps") is None:
        return False
    events, pairs = episode["events"], episode["pairs"]
    return len(pairs) == len(terra["steps"]) and all(
        events[pair["dig"]]["step"] == dig + 1
        and events[pair["dump"]]["step"] == dump + 1
        for pair, (dig, dump) in zip(pairs, terra["steps"])
    )


def terra3d_bundle(bundle=None):
    """Return the installed player or an explicit prebuilt JavaScript override."""
    return player_bundle(bundle)


def _seam_fields(entry):
    """A workspace's seams for the page: [x, y, ok] every 10 cm, their length and the share with a 0.3 m lip."""
    if entry is None:
        return dict(seams=[], seam_m=0.0, seam_share=None)
    samples = entry["seams"]
    length, share = seams.share(samples)
    return dict(
        seams=[
            [round(x, 2), round(y, 2), int(lip + 1e-9 >= need)]
            for x, y, lip, need, _ in samples[::5]
        ],
        seam_m=round(length, 2),
        seam_share=None if share is None else round(share, 4),
    )


def version_data(result, label, note=""):
    """The page data of one replayed conversion."""
    case, rules = result["case"], result["rules"]
    checklist = plan_replay.validity(result)
    r0, r1, c0, c1 = plan_replay._crop(result)
    window = (slice(r0, r1), slice(c0, c1))
    h, w = r1 - r0, c1 - c0
    native = result["native_frames"][:, r0:r1, c0:c1].astype(np.float64)
    loose = result["loose_frames"][:, r0:r1, c0:c1].astype(np.float64)
    excavated = result["excavated_frames"][:, r0:r1, c0:c1]
    design = result["design"][window]
    target, required = case.target[window], case.required[window]
    # Soil of loads released off the final zones (temporary soil) fails the end-state rule here at the end.
    temporary = result["temporary_frames"][:, r0:r1, c0:c1]
    outside = ~case.final_ground[window] & ~required

    base = np.zeros((h, w), dtype=np.uint8)
    base[case.final_zone[window]] = 1
    if case.final_spread is not None:
        base[case.final_spread[window]] = (
            4  # final ground the converter grew for pile spread
        )
    base[case.obstacle[window]] = 2
    base[~case.known[window]] = 3

    def state(frame):
        remaining = native[frame] > design + rules.finished_m
        cells = np.zeros((h, w), dtype=np.uint8)
        cells[required & remaining] = REQUIRED_LEFT
        cells[target & ~required & remaining] = OPTIONAL_LEFT
        cells[excavated[frame] & remaining] = PARTIAL
        cells[target & ~remaining] = FINISHED
        cells[target & ~remaining & (loose[frame] >= rules.toe_m)] |= FELL_BACK_BIT
        cells[temporary[frame] & outside] |= TEMPORARY_BIT
        pile = np.clip(
            np.rint(
                np.where(loose[frame] >= rules.toe_m, loose[frame], 0.0) / PILE_STEP_M
            ),
            0,
            255,
        )
        return cells.ravel(), pile.astype(np.uint8).ravel()

    frames = native.shape[0]
    first_state, first_pile = state(0)
    previous_state, previous_pile = first_state, first_pile
    offsets, indices, states, piles = [0], [], [], []
    for frame in range(1, frames):
        current_state, current_pile = state(frame)
        changed = np.flatnonzero(
            (current_state != previous_state) | (current_pile != previous_pile)
        )
        indices.append(changed)
        states.append(current_state[changed])
        piles.append(current_pile[changed])
        offsets.append(offsets[-1] + changed.size)
        previous_state, previous_pile = current_state, current_pile
    indices = np.concatenate(indices) if indices else np.zeros(0, dtype=np.uint32)
    states = np.concatenate(states) if states else np.zeros(0, dtype=np.uint8)
    piles = np.concatenate(piles) if piles else np.zeros(0, dtype=np.uint8)

    # Seam lips (seams): each excavation workspace's lip beyond its completed ground, as bits on the page grid,
    # and the real seams into it, every 10 cm, with whether the lip there is at least 0.3 m.
    seam_by_pose = {
        (round(entry["pose"][0], 3), round(entry["pose"][1], 3)): entry
        for entry in seams.seam_lips(case.report)
        if entry is not None
    }
    window_x = case.origin[0] + np.arange(c0, c1) * case.res
    window_y = case.origin[1] + np.arange(r0, r1) * case.res
    grid_x, grid_y = np.meshgrid(window_x, window_y)

    def seam_entry(pose):
        return seam_by_pose.get((round(float(pose[0]), 3), round(float(pose[1]), 3)))

    def lip_bits(entry):
        if entry is None or entry["lip"].is_empty:
            return np.zeros((h, w), dtype=bool)
        return shapely.contains_xy(entry["lip"], grid_x, grid_y)

    area = case.res**2
    events = []
    for step, entry in enumerate(result["timeline"], start=1):
        remaining = np.maximum(native[step] - design, 0.0)
        left = required & (remaining > rules.finished_m)
        events.append(
            dict(
                ws=entry["workspace"],
                phase=entry["phase"],
                load=entry.get("load"),
                loads=entry.get("loads"),
                left=round(float(left.sum()) * area, 2),
                pile=round(float(loose[step].max()), 2),
                soil=round(float(loose[step].sum()) * area, 2),
            )
        )
    # Which loads put the soil that ends outside the final zones: a load released off them, whose soil stays unless a
    # later cut or collection lifts its cell (only those lower the loose soil of a cell).
    end_outside = temporary[-1] & outside
    lifted = np.full((h, w), -1)
    for frame in range(1, frames):
        lifted[loose[frame] < loose[frame - 1] - 1e-9] = frame
    workspaces = []
    for k, event in enumerate(result["events"]):
        pose = case.poses[k]
        route = case.routes.get((k, k + 1))
        loads = []
        for load in event["dump"]["loads"]:
            step = load["step"]
            added = np.maximum(loose[step] - loose[step - 1], 0.0)
            released_off = not load["final_release"]
            loads.append(
                dict(
                    step=step,
                    x=load["x_m"],
                    y=load["y_m"],
                    m3=load["volume_m3"],
                    top=load["pile_top_m"],
                    toe=load["toe_radius_m"],
                    gap=load["gap_to_excavation_m"],
                    band=load["spill"],
                    choice=load["choice"],
                    spill=round(load["pit_spill_m3"], 3),
                    zone=bool(load["final_release"]),
                    out=(
                        round(float(added[outside].sum()) * area, 3)
                        if released_off
                        else 0.0
                    ),
                    kept=(
                        round(
                            float(added[end_outside & (lifted < step)].sum()) * area, 3
                        )
                        if released_off
                        else 0.0
                    ),
                )
            )
        leg = event["route"] or {}
        workspaces.append(
            dict(
                n=k + 1,
                source=event["source_pair"],
                kind=event["kind"],
                pose=[round(float(v), 3) for v in pose],
                body=_round(plan_replay.world_polygon(case.footprint, pose)),
                cut=event["steps"]["cut"],
                arrival=(
                    "blocked"
                    if event["arrival"]["blocked"]
                    else "unverified" if event["arrival"]["unverified"] else "clear"
                ),
                finished_m2=event["cut"]["new_required_m2"],
                cut_m3=event["cut"]["native_cut_m3"],
                rehandled_m3=event["cut"]["loose_lifted_m3"],
                radius=event["cut"]["completion_radius_m"],
                loads=loads,
                route=(
                    [
                        [round(float(x), 2), round(float(y), 2), round(float(yaw), 3)]
                        for x, y, yaw in route["path"]
                    ]
                    if route is not None and route.get("path")
                    else None
                ),
                route_state=(
                    None
                    if event["route"] is None
                    else (
                        "check failed"
                        if event["route"].get("checker_passed") is False
                        else (
                            "not found"
                            if not event["route"].get("found", True)
                            else "blocked" if event["route"]["blocked"] else "clear"
                        )
                    )
                ),
                route_reason=leg.get("checker_reason"),
                route_m=leg.get("length_m"),
                straight_m=(
                    round(math.hypot(*(pose[:2] - case.poses[k - 1][:2])), 2)
                    if k
                    else None
                ),
                route_high_spoil=leg.get("high_spoil_pose_indices", []),
                route_blocked_at=leg.get("first_blocked_pose"),
                # What the plan gives this station: the ground it may dig, the ground it must finish, the centres
                # its bucket loads may be dumped on (the plan's dump mask; the workspace planner picks each load's
                # centre in it) and the ground that soil may cover (the 1.1 m deposit support around those centres).
                regions=dict(
                    dig=_pack_bits(case.support[k][window]),
                    finish=_pack_bits(case.completion[k][window]),
                    dump=_pack_bits(case.dump_centres[k][window]),
                    deposit=_pack_bits(case.converter_deposit[k][window]),
                    lip=_pack_bits(lip_bits(seam_entry(pose))),
                ),
                **_seam_fields(seam_entry(pose)),
            )
        )
    totals = result["totals"]
    kept = [(load["zone"], load["kept"]) for s in workspaces for load in s["loads"]]
    # The extent of each piece of spoil left outside the final zones, matched to the replay's pieces by centroid.
    labels, count = ndimage.label(end_outside, structure=np.ones((3, 3)))
    extents = []
    if count:
        centres = ndimage.center_of_mass(end_outside, labels, range(1, count + 1))
        for (row, col), (rows, cols) in zip(centres, ndimage.find_objects(labels)):
            x0, y0 = case.xy(r0 + rows.start, c0 + cols.start)
            x1, y1 = case.xy(r0 + rows.stop - 1, c0 + cols.stop - 1)
            extents.append(
                (
                    *case.xy(r0 + row, c0 + col),
                    [round(float(v), 2) for v in (x0, y0, x1, y1)],
                )
            )

    def extent(x, y):
        return (
            min(extents, key=lambda e: math.hypot(e[0] - x, e[1] - y))[2]
            if extents
            else None
        )

    dump = dict(
        dump_rule(case),
        left_m3=totals["temporary_spoil_left_m3"],
        released_m3=round(sum(m3 for zone, m3 in kept if not zone), 3),
        # Final loads count as final wherever they spread; this is how much of them lies past the zones.
        spread_m3=totals["final_soil_past_final_zones_m3"],
        pieces=[
            dict(
                m3=round(i["volume_m3"], 3),
                x=i["x_m"],
                y=i["y_m"],
                box=extent(i["x_m"], i["y_m"]),
            )
            for i in result["issues"]
            if i["kind"] == "temporary_spoil_left"
        ],
    )
    # Required ground left within the converter's tolerance is a note on coverage, not a failure; so is a route over
    # spoil below the /map height (chassis balancing).
    notes = dict(
        coverage=[
            i for i in result["issues"] if i["kind"] == "required_left_within_tolerance"
        ],
        routes=[i for i in result["issues"] if i["kind"] == "route_over_high_spoil"],
    )
    checks = []
    for check in checklist["checks"]:
        items = []
        for kind, entries in (
            ("fail", check["failures"]),
            ("unverified", check["unverified"]),
            (
                "note",
                [plan_replay._entry(i) for i in notes.get(check["id"], [])]
                + check.get("notes", []),
            ),
        ):
            for entry in entries:
                title, chips, note_text = headline(entry)
                x, y = entry.get("x_m"), entry.get("y_m")
                items.append(
                    dict(
                        kind=kind,
                        issue=entry.get("kind"),
                        title=title,
                        chips=chips,
                        note=note_text,
                        text=entry["text"],
                        step=entry["step"],
                        ws=entry["workspace"],
                        x=x,
                        y=y,
                        fix=entry.get("fix"),
                        command=edit_command(case, entry.get("fix")),
                        leg=(
                            entry["workspace"]
                            if str(entry.get("kind", "")).startswith("route_")
                            else None
                        ),
                    )
                )
        checks.append(
            dict(
                id=check["id"],
                name=CHECK_NAMES.get(check["id"], check["id"]),
                rule=check["rule"],
                status=check["status"],
                detail=check["detail"],
                items=items,
            )
        )
    return dict(
        label=label,
        note=note,
        tag=case.tag,
        verdict=checklist["verdict"],
        checks=checks,
        totals=dict(
            stations=totals["workspaces"],
            added=totals["added_workspaces"],
            required_m2=totals["required_m2"],
            left_m2=totals["required_left_m2"],
            left_m3=totals["required_left_m3"],
            within_m2=round(sum(i["area_m2"] for i in notes["coverage"]), 3),
            tolerance_m2=case.report.get("required_residual_tolerance_m2"),
            cut_m3=totals["native_cut_m3"],
            rehandled_m3=totals["rehandled_loose_m3"],
            loads=totals["loads"],
            top_pile_m=totals["highest_pile_m"],
            spoil_left_m3=totals["temporary_spoil_left_m3"],
            route_m=totals["saved_routes_m"],
            routes=totals["saved_routes"],
            straight_m=totals["straight_connectors_m"],
            seam_m=round(sum(s["seam_m"] for s in workspaces), 2),
            seam_share=(
                round(
                    sum(
                        s["seam_m"] * s["seam_share"]
                        for s in workspaces
                        if s["seam_share"] is not None
                    )
                    / sum(s["seam_m"] for s in workspaces),
                    4,
                )
                if sum(s["seam_m"] for s in workspaces) > 0
                else None
            ),
        ),
        dump=dump,
        route_check=result.get("route_check"),
        rules=dict(
            repose_deg=rules.repose_deg,
            pass_height_m=rules.pass_height_m,
            obstacle_height_m=rules.ros_obstacle_height_m,
            completion=[rules.band_min_m, rules.band_max_m],
            entry=rules.entry_max_m,
        ),
        grid=dict(
            x0=round(float(case.origin[0] + c0 * case.res), 4),
            y0=round(float(case.origin[1] + r0 * case.res), 4),
            res=case.res,
            w=w,
            h=h,
        ),
        base=_pack(base.ravel(), np.uint8),
        state0=_pack(first_state, np.uint8),
        pile0=_pack(first_pile, np.uint8),
        delta=dict(
            offsets=_pack(np.asarray(offsets), np.uint32),
            index=_pack(indices, np.uint32),
            state=_pack(states, np.uint8),
            pile=_pack(piles, np.uint8),
        ),
        events=events,
        workspaces=workspaces,
        terra=terra_plan(case, window),
        footprint=_round(case.footprint),
        source=dict(
            conversion=str(case.conversion), native=case.report["inputs"]["input_dir"]
        ),
        not_modeled=list(plan_replay.NOT_MODELED),
    )


def build(
    manifest,
    out,
    rules=None,
    *,
    bundle=None,
    runtime_validator=None,
    station_validator=None,
):
    """Replay every version of the manifest and write the page; returns the page path and per-version verdicts."""
    rules = plan_replay.Rules() if rules is None else rules
    manifest = (
        json.loads(Path(manifest).read_text())
        if not isinstance(manifest, dict)
        else manifest
    )
    postprocessed = bool(manifest.get("postprocessed3d", True))
    cases, rows = [], []
    for case_entry in manifest["cases"]:
        episode = (
            terra3d_data(case_entry["terra3d"]) if case_entry.get("terra3d") else None
        )
        versions = []
        for version in case_entry["versions"]:
            loaded = plan_replay.load_case(
                version["conversion"], routes_dir=version.get("routes")
            )
            result = plan_replay.replay(
                loaded, rules, runtime_validator=runtime_validator
            )
            data = version_data(result, version["label"], version.get("note", ""))
            if postprocessed:
                from . import timeline

                data["postprocessed3d"] = timeline.from_replay(
                    result, station_validator=station_validator
                )
            if data["terra"] is not None:
                data["terra"]["linked"] = terra_linked(data["terra"], episode)
            versions.append(data)
            rows.append(
                dict(
                    case=case_entry["name"],
                    version=version["label"],
                    verdict=data["verdict"],
                )
            )
        cases.append(
            dict(
                name=case_entry["name"],
                kind=case_entry.get("kind", ""),
                group=case_entry.get("group", ""),
                versions=versions,
                terra3d=episode,
            )
        )
    data = dict(title=manifest.get("title", "Terra plans"), cases=cases)
    data.update(
        {key: manifest[key] for key in ("labels", "headline") if key in manifest}
    )
    script = ""
    if postprocessed or any(case["terra3d"] for case in cases):
        javascript, data["terra3d_viewer"] = terra3d_bundle(bundle)
        # HTML raw-text elements end at "</script" even inside JavaScript strings.
        script = (
            "<script>"
            + re.sub(r"</script", r"<\\/script", javascript, flags=re.IGNORECASE)
            + "</script>"
        )
    payload = json.dumps(data, separators=(",", ":"))
    page = TEMPLATE.read_text().replace(
        "__TITLE__", html.escape(manifest.get("title", "Terra plans"))
    )
    page = page.replace("__DATA__", payload.replace("</", "<\\/"))
    page = page.replace("<!--TERRA3D-->", script, 1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out, rows
