"""Chronological replay of a converted Terra plan under the Terra-to-machine ground rules.

``tmm.py plan replay`` runs this on the converter's output directory (``conversion/report.json`` and
``conversion/coverage.npz``; complete and incomplete conversions alike) and writes a JSON report for
agents plus a webviz run history for the offline viewer (moleworks_newton ``scripts/benchmark/webviz``).

Every retained workspace, in plan order, on the converter's 0.1 m grid:

* arrive: the machine body (the profile footprint) at the station against the terrain the previous
  workspace left. Excavated ground, static obstacles, ground off the map and terrain ROS ``/map``
  marks occupied block it; loose soil under ``pass_height_m`` does not; loose soil between the two is
  unverified. A saved Nav2 route to the station gets the same test at every pose, except that loose soil under the
  ``/map`` height is drivable there: with chassis balancing the machine drives over it (Lorenzo, 29 September);
  such poses are noted per leg.
* cut: loose soil in the swept cells (the exported support) inside the design is lifted, then the design material
  there is cut to the design surface, as the runtime planner pulls through the whole support (``Rules.cut``
  "corridor"; "completion" cuts the completion cells only). Required ground counts as finished only where a
  completion cut reached it. A collection lifts the loose soil in its support and completion cells.
* dump: the plan works at workspace level, never at single scoops (Lorenzo, 2 October 2026): the lifted soil goes
  out as one deposit over the workspace's dump region, filling it to one level with the repose slope around it on the
  current surface (``spoil_heights.plateau_deposit``, the converter's forecast too). No soil of it, its spread
  included, may land in the current excavation (trench or foundation); there is no band beyond it (Lorenzo,
  1 October 2026). ROS must admit a centre of the region at the deposit's surface from the station and from every
  stop within the station tolerance.
* end: a final zone grows where its own piles spread. Soil released on final ground (the region's share on it)
  counts as final wherever it spreads; soil released elsewhere (temporary soil) must not be left outside the final
  ground. A lift takes all loose soil of a cell, temporary and final in proportion.

Ground rules and their sources: .artifacts/terra_export_review_20260927/PLANNER_GROUND_RULES_20260928.md.
What the replay does not model is listed in ``NOT_MODELED`` and in every report.
"""

import contextlib
import gzip
import json
import math
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
import yaml

from .soil import plateau_deposit, release_region

# ROS dump selector and runtime admission (workspace_planner/dump_targets.py).
ROS_SUPPORT_RADIUS_M = 1.1  # DUMP_DEPOSIT_SUPPORT_RADIUS_M, the support the runtime keeps clear of excavation
ROS_APPROACH_M = 1.0  # DUMP_APPROACH_CLEARANCE_M, dump point height above the ground for the reach test
ROS_PILE_EDGE_M = 0.45  # DUMP_FOOTPRINT_RADIUS_M, pile edge for the BASE keep-away
NOT_MODELED = (
    "original ground is flat (the converter output has no survey elevation) and the design depth is uniform",
    "all cuts of a workspace happen before its dumps (conservative for soil spilling into the excavation)",
    "dump points follow a simplified copy of the ROS ranking on this forecast, not the live selector on the "
    "measured map; the release point is taken as the landing point",
    "piles are cones at the repose angle stacked on the current surface; later cuts do not relax the "
    "steep faces they leave",
    "an excavation cuts all design material in its exported support (the runtime cuts part of it; 60-75 % in the "
    "offline runtime replay of 2 October 2026), so its payload is an upper bound and later stations get what is left",
    "CABIN is at BASE in plan view for the dump reach test",
    "routes are straight connectors unless a saved Nav2 route is given; connectors are not validated paths",
    "no wheel contact, sinkage, traction, stability or arm/controller execution",
)

# Hazard classes of a cell for the machine body.
FREE, SPOIL, UNVERIFIED, ROS_OBSTACLE, HOLE, OBSTACLE, UNKNOWN = range(7)
BLOCKING = (ROS_OBSTACLE, HOLE, OBSTACLE, UNKNOWN)
HAZARD_NAMES = {
    SPOIL: "loose soil under the drivable height",
    UNVERIFIED: "loose soil above the drivable height (unverified under a station, drivable on a route)",
    ROS_OBSTACLE: "terrain ROS /map marks occupied",
    HOLE: "excavated ground",
    OBSTACLE: "static obstacle",
    UNKNOWN: "off the known map",
}
SEVERITIES = ("physical", "runtime", "unverified", "efficiency", "converter")
# Validity checks: (id, rule, issue kinds that fail it, issue kinds that leave it unverified).
CHECKS = (
    (
        "converter_plan",
        "The converter wrote an executable plan: complete geometry, residual under tolerance; the runtime accepts it "
        "(Terra plan schema and launch preflight)",
        (),
        (),
    ),
    (
        "coverage",
        "All required ground ends at the design surface (within 5 cm)",
        ("required_left",),
        (),
    ),
    (
        "completion_band",
        "Every excavation's completion cell lies 4.0-6.5 m from its station's BASE",
        ("completion_out_of_band",),
        (),
    ),
    (
        "cut_clear_of_body",
        "No completion cell lies under the machine body",
        ("cut_under_body",),
        (),
    ),
    (
        "arrival",
        "At every station the body stands on drivable ground: no excavation, obstacle, unknown ground or "
        "terrain 1.0 m over original ground; loose soil under 0.5 m is drivable",
        ("arrival_blocked",),
        ("arrival_unverified",),
    ),
    (
        "routes",
        "Every drive has a saved Nav2 route that stays clear of the terrain at its time and passes the offline route "
        "check; loose soil under the /map height is drivable on the way (chassis balancing)",
        ("route_blocked", "route_not_found", "route_check_failed"),
        (),
    ),
    (
        "dump_admission",
        # Reach and keep-away are the converter's recorded dump rule (report dump_rule), filled in by validity().
        "ROS admits a centre of every workspace's dump region at the nominal station: {reach:g} m CABIN reach at "
        "the deposit's surface, pile edge {keep_away:g} m from BASE, 1.1 m support clear of excavation; stops within "
        "the station tolerance that lose every centre at that surface leave it unverified",
        ("dump_ros_refuses", "dump_out_of_reach", "no_dump_region"),
        ("dump_unreachable_from_stop",),
    ),
    (
        "pile_clearance",
        "No workspace's deposit (its soil over its dump region, 27 deg slopes) puts soil, its spread included, in the "
        "current excavation (trench or foundation)",
        ("dump_into_pit",),
        ("dump_touches_pit", "pile_off_map"),
    ),
    (
        "spoil_at_end",
        "At the end all loose soil lies on the final dump zones: the soil of loads released on a final zone (or within "
        "one Terra tile of it) counts as final wherever it spreads; none in finished excavation, no soil of loads "
        "released elsewhere left outside the final zones",
        ("loose_in_finished_excavation", "temporary_spoil_left"),
        (),
    ),
)


@dataclass(frozen=True)
class Rules:
    """Replay parameters. Defaults and their sources: PLANNER_GROUND_RULES_20260928.md."""

    repose_deg: float = (
        27.0  # Newton workspace-planner soil model, SOIL_MOTION_V2_REPOSE_ANGLE_RAD
    )
    bulking: float = 1.0  # loose / bank volume; Terra, Newton and ROS conserve volume
    bucket_m3: float = (
        0.255  # stock 1.3 m shovel, description/mole_description/config/bucket_specs.yaml
    )
    pass_height_m: float = 0.5  # Lorenzo: loose soil under 0.5 m is drivable
    ros_obstacle_height_m: float = (
        1.0  # excavation_mapping.yaml height_obstacle_z_thresh over original ground
    )
    lowering_m: float = (
        0.05  # DUMP_EXCAVATED_LOWERING_TOLERANCE_M: ground this far down is excavated
    )
    finished_m: float = (
        0.05  # a required cell is finished within this of the design surface
    )
    toe_m: float = 0.01  # a pile reaches a cell where it adds at least this much
    band_min_m: float = (
        4.0  # confirmed BASE completion band, terra_planner docs/dig_reach_requirements.md
    )
    band_max_m: float = 6.5
    entry_max_m: float = 7.0
    tiny_m2: float = (
        0.1  # a workspace that finishes less required area than this is tiny
    )
    # Leftover soil: a connected piece smaller than this (8 % of a bucket) is below the cone model's resolution and no
    # issue, at the end (outside the final zones, in finished excavation) and per dump in the excavation.
    min_leftover_m3: float = 0.02
    drive_frames: int = 0  # poses drawn along each drive between stations (0: none)
    # What a workspace cuts: "corridor", all design material its exported support still holds, as the runtime
    # workspace planner pulls full-width bands through the whole support (offline runtime replay, 2 October 2026);
    # "completion", only its completion cells (the model before). end_state.forecast uses the same rule.
    cut: str = "corridor"


@dataclass
class Case:
    """One converted plan on the converter's grid: rows advance +y, columns +x, origin is cell [0, 0]'s centre."""

    tag: str
    conversion: Path
    report: dict
    origin: np.ndarray
    res: float
    target: np.ndarray
    required: np.ndarray
    depth_m: float
    tile_m: float
    final_zone: np.ndarray
    obstacle: np.ndarray
    known: np.ndarray
    kinds: list
    sources: list
    poses: np.ndarray
    support: np.ndarray
    completion: np.ndarray
    dump_centres: np.ndarray
    converter_blocked: np.ndarray
    converter_deposit: np.ndarray
    native_poses: list
    native_dump: list  # per native pair: its dump permission tiles on this grid
    native_dig: list  # per native pair: its dig tiles on this grid
    footprint: np.ndarray
    dump_rule: dict
    routes: dict
    complete: bool
    profile: Path | None = (
        None  # the conversion's profile.yaml, for the runtime's plan checks
    )
    navigation: dict | None = (
        None  # the saved route check's report.json (it stops at the first failing leg)
    )
    # Cells beside the painted final zones that the converter counts as final ground for pile spread (height model,
    # report key final_zone_spread_m); None for a conversion that records no spread allowance.
    final_spread: np.ndarray | None = None

    @property
    def shape(self):
        return self.target.shape

    @property
    def final_ground(self):
        """The final dump zones: the painted zones and the converter's allowance around them (1.0 m). A load released here adds
        final soil, wherever it spreads."""
        return (
            self.final_zone
            if self.final_spread is None
            else self.final_zone | self.final_spread
        )

    def xy(self, rows, cols):
        return (
            self.origin[0] + np.asarray(cols) * self.res,
            self.origin[1] + np.asarray(rows) * self.res,
        )


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _native_on_grid(values, alignment, origin, res, shape, fill):
    """Sample a native Terra [x_tile, y_tile] layer at every grid cell centre."""
    rows, cols = np.indices(shape)
    world = (
        np.stack([origin[0] + cols * res, origin[1] + rows * res], axis=-1)
        - alignment["origin_map_xy_m"]
    )
    plan = (
        world
        @ _rotation(float(alignment["yaw_map_from_plan_rad"]))
        / float(alignment["meters_per_tile"])
    )
    ix, iy = np.floor(plan[..., 0]).astype(int), np.floor(plan[..., 1]).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < values.shape[0]) & (iy < values.shape[1])
    out = np.full(shape, fill, dtype=values.dtype)
    out[inside] = values[ix[inside], iy[inside]]
    return out, inside


def case_tag(directory):
    """Plan name of a converter output: DIR for DIR/conversion, else the conversion directory's parent."""
    directory = Path(directory)
    return (
        directory.name
        if (directory / "conversion" / "report.json").is_file()
        else directory.resolve().parent.name
    )


def read_json(path):
    """A JSON file, or its gzip copy PATH.gz: the bank runner gzips the plans once a job is evaluated."""
    path = Path(path)
    packed = path.with_name(path.name + ".gz")
    if path.is_file() or not packed.is_file():
        return json.loads(path.read_text())
    with gzip.open(packed, "rt") as stream:
        return json.load(stream)


def json_exists(path):
    path = Path(path)
    return path.is_file() or path.with_name(path.name + ".gz").is_file()


@contextlib.contextmanager
def _plain_copy(path):
    """PATH itself, or a temporary plain copy of PATH.gz for readers that need a file."""
    path = Path(path)
    packed = path.with_name(path.name + ".gz")
    if path.is_file() or not packed.is_file():
        yield path
        return
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / path.name
        with gzip.open(packed, "rb") as source, open(copy, "wb") as target:
            shutil.copyfileobj(source, target)
        yield copy


def load_case(directory, routes_dir=None, tag=None):
    """Read one converter output: DIR holding conversion/ (or the conversion directory) and profile.yaml."""
    directory = Path(directory)
    conversion = (
        directory / "conversion"
        if (directory / "conversion" / "report.json").is_file()
        else directory
    )
    report = json.loads((conversion / "report.json").read_text())
    with np.load(conversion / "coverage.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    source = Path(report["inputs"]["input_dir"])
    native_plan = read_json(source / "terra_plan.json")
    with np.load(source / "arrays.npz") as archive:
        native = {key: archive[key] for key in archive.files}
    profile_path = conversion.parent / "profile.yaml"
    if not profile_path.is_file():
        profile_path = Path(report["inputs"]["profile"])
    workspace = yaml.safe_load(profile_path.read_text())["workspace"]
    footprint = np.asarray(json.loads(workspace["base_footprint_xy_json"]), dtype=float)

    grid = report["grid_geometry"]
    origin = np.asarray(grid["origin_xy"], dtype=float)
    res = float(grid["resolution_m"])
    if not np.allclose(origin, arrays["origin_xy"]):
        raise ValueError(
            f"{conversion}: report grid origin {origin} differs from coverage.npz {arrays['origin_xy']}"
        )
    shape = tuple(arrays["target"].shape)
    if tuple(grid["shape_yx"]) != shape:
        raise ValueError(
            f"{conversion}: report grid shape {grid['shape_yx']} differs from coverage.npz {shape}"
        )
    alignment = native_plan["alignment"]
    images, known = _native_on_grid(native["images"], alignment, origin, res, shape, 0)
    obstacle, _ = _native_on_grid(
        native["occupancy"].astype(bool), alignment, origin, res, shape, False
    )

    waypoints = native_plan["waypoints"]
    sources = [int(s) for s in arrays["retained_source_pair_indices"]]
    if "retained_collection" in arrays:
        # The converter's collections, Terra's and those it added for the end state (source -1, as added stations).
        kinds = [
            "collect_dumped_soil" if collect else "excavate"
            for collect in arrays["retained_collection"]
        ]
    else:
        kinds = [
            waypoints[2 * s]["workspace_type"] if s >= 0 else "excavate"
            for s in sources
        ]
    native_poses = []
    for pair in range(len(waypoints) // 2):
        state = waypoints[2 * pair]["agent_state"]
        yaw = float(alignment["yaw_map_from_plan_rad"])
        xy = np.asarray(alignment["origin_map_xy_m"]) + _rotation(yaw) @ (
            np.asarray(state["pos_base"], dtype=float)
            * float(alignment["meters_per_tile"])
        )
        native_poses.append(
            (float(xy[0]), float(xy[1]), float(state["angle_base_rad"]) + yaw)
        )

    native_dump = [
        _native_on_grid(
            np.asarray(
                waypoints[2 * pair + 1]["terrain_modification_mask"], dtype=bool
            ),
            alignment,
            origin,
            res,
            shape,
            False,
        )[0]
        for pair in range(len(waypoints) // 2)
    ]
    native_dig = [
        _native_on_grid(
            np.asarray(waypoints[2 * pair]["terrain_modification_mask"], dtype=bool),
            alignment,
            origin,
            res,
            shape,
            False,
        )[0]
        for pair in range(len(waypoints) // 2)
    ]
    n = len(sources)
    deposit = arrays.get("retained_deposit_support")
    routes, navigation = {}, None
    if routes_dir is not None:
        for path in sorted(Path(routes_dir).glob("route_*_*.json")):
            route = json.loads(path.read_text())
            routes[(int(route["from_workspace"]), int(route["to_workspace"]))] = route
        if (Path(routes_dir) / "report.json").is_file():
            navigation = json.loads((Path(routes_dir) / "report.json").read_text())
    return Case(
        tag=tag or case_tag(directory),
        conversion=conversion,
        report=report,
        origin=origin,
        res=res,
        target=arrays["target"].astype(bool),
        required=arrays["required_core"].astype(bool),
        depth_m=float(report["target_depth_m"]),
        tile_m=float(alignment["meters_per_tile"]),
        final_zone=images == 1,
        obstacle=obstacle,
        known=known,
        kinds=kinds,
        sources=sources,
        poses=np.asarray(arrays["retained_base_pose"], dtype=float).reshape(n, 3),
        support=arrays["retained_dig_support"].astype(bool),
        completion=arrays["retained_dig_completion"].astype(bool),
        dump_centres=arrays["retained_dump_support"].astype(bool),
        converter_blocked=arrays["retained_traversability"].astype(bool),
        converter_deposit=(
            np.zeros((n, *shape), dtype=bool)
            if deposit is None
            else deposit.astype(bool)
        ),
        native_poses=native_poses,
        native_dump=native_dump,
        native_dig=native_dig,
        footprint=footprint,
        dump_rule=report["dump_rule"],
        routes=routes,
        complete=bool(report.get("complete_geometric_plan"))
        and json_exists(conversion / "terra_plan.json"),
        profile=profile_path,
        navigation=navigation,
        # The coverage mask the converter wrote with the allowance it records in the report.
        final_spread=(
            arrays["final_zone_spread"].astype(bool)
            if "final_zone_spread_m" in report
            else None
        ),
    )


def route_check_failure(case):
    """The saved route check's own failure, or None when it passed or its report is absent.

    The checker (``check_converted_terra_routes.py``) plans every leg with the installed Smac and sweeps the padded
    parked body along it; it stops at the first failing leg. Returns the failing leg's destination workspace (one-based,
    None for a body check at the first or last station), the checker's reason and the legs it never checked.
    """
    report = case.navigation
    if not report or report.get("connected_internal_routes", True):
        return None
    legs = report.get("legs") or []
    failed = next((leg for leg in legs if not leg.get("passed", True)), None)
    checked = {int(leg["to_workspace"]) for leg in legs}
    unchecked = [n for n in range(2, len(case.sources) + 1) if n not in checked]
    if failed is None:
        reason = str(report.get("failure", "the route check failed")).replace("_", " ")
    elif not failed.get("route_found"):
        message = str(failed.get("error_message", ""))
        quoted = re.search(r'"([^"]+)"\s*$', message)
        reason = "Nav2 found no path" + (
            f" ({quoted.group(1) if quoted else message[:80]})" if message else ""
        )
    elif not (failed.get("swept_body") or {}).get("passed", True):
        body = failed["swept_body"]
        gap = body.get("minimum_body_to_hazard_gap_m")
        reason = (
            f"the padded body touches hazards at {body.get('hazard_intersection_count', 0)} poses"
            + ("" if gap is None else f" (closest gap {gap:.2f} m)")
        )
        if body.get("outside_known_domain_count"):
            reason += (
                f", {body['outside_known_domain_count']} poses leave the known map"
            )
    elif not (failed.get("station_arrival") or {}).get("passed", True):
        reason = "the route ends outside the station arrival bound"
    else:
        reason = "the leg misses the station tolerance or the path frame"
    return dict(
        failure=report.get("failure"),
        to_workspace=None if failed is None else int(failed["to_workspace"]),
        reason=reason,
        unchecked=unchecked,
    )


def runtime_checks(case, validator=None):
    """Consume an explicit runtime adapter; portable replay does not certify ROS admission.

    ``validator(case)`` returns a mapping with status ``pass``, ``fail`` or
    ``not checked`` and any diagnostic evidence. No environment discovery or
    implicit profile loading occurs here.
    """
    if validator is None:
        return dict(status="not checked", message="No runtime validator was supplied")
    result = validator(case)
    if not isinstance(result, dict) or result.get("status") not in (
        "pass",
        "fail",
        "not checked",
    ):
        raise ValueError(
            "Runtime validator must return an explicit pass/fail/not checked status"
        )
    result = dict(result)
    result.setdefault("message", f"Runtime validator returned {result['status']}")
    if not isinstance(result["message"], str):
        raise ValueError("Runtime validator diagnostic message must be a string")
    return result


def _inside_polygon(points, polygon):
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_cross)
    return inside


def body_points(footprint, step):
    """Sample points of the body polygon (BASE frame): its interior lattice plus its outline."""
    lo, hi = footprint.min(axis=0), footprint.max(axis=0)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    lattice = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    outline = []
    for a, b in zip(footprint, np.roll(footprint, -1, axis=0)):
        count = max(2, int(math.ceil(np.linalg.norm(b - a) / step)) + 1)
        outline.append(a + np.linspace(0.0, 1.0, count)[:, None] * (b - a))
    return np.vstack([lattice[_inside_polygon(lattice, footprint)], *outline])


def world_polygon(footprint, pose):
    return footprint @ _rotation(pose[2]).T + np.asarray(pose[:2])


def _cells(case, xy):
    """Grid cells of world points; points off the grid come back as -1."""
    cols = np.rint((xy[:, 0] - case.origin[0]) / case.res).astype(int)
    rows = np.rint((xy[:, 1] - case.origin[1]) / case.res).astype(int)
    inside = (rows >= 0) & (cols >= 0) & (rows < case.shape[0]) & (cols < case.shape[1])
    return np.where(inside, rows, -1), np.where(inside, cols, -1)


def hazard_classes(case, rules, native, loose, excavated):
    cls = np.full(case.shape, FREE, dtype=np.int8)
    cls[loose >= rules.toe_m] = SPOIL
    cls[loose >= rules.pass_height_m] = UNVERIFIED
    cls[native + loose >= rules.ros_obstacle_height_m] = ROS_OBSTACLE
    cls[excavated] = HOLE
    cls[case.obstacle] = OBSTACLE
    cls[~case.known] = UNKNOWN
    return cls


def body_check(case, points, pose, classes, hole_distance):
    """Hazards under the body at one pose: blocked and unverified cells, the loosest cell, the gap to excavation."""
    rows, cols = _cells(case, points @ _rotation(pose[2]).T + np.asarray(pose[:2]))
    off_grid = rows < 0
    rows, cols = rows[~off_grid], cols[~off_grid]
    under = classes[rows, cols]
    counts = {
        HAZARD_NAMES[c]: int(np.count_nonzero(under == c))
        for c in HAZARD_NAMES
        if np.any(under == c)
    }
    if off_grid.any():
        counts[HAZARD_NAMES[UNKNOWN]] = counts.get(HAZARD_NAMES[UNKNOWN], 0) + int(
            off_grid.sum()
        )
    blocked = np.isin(under, BLOCKING)
    unique = np.unique(rows * case.shape[1] + cols)
    return dict(
        blocked=bool(blocked.any() or off_grid.any()),
        unverified=bool(np.any(under == UNVERIFIED)),
        hazards=counts,
        blocked_cells=np.unique((rows * case.shape[1] + cols)[blocked]),
        unverified_cells=np.unique((rows * case.shape[1] + cols)[under == UNVERIFIED]),
        body_cells=unique,
        gap_to_excavation_m=(
            float(np.min(hole_distance[rows, cols])) if rows.size else None
        ),
    )


def leg_poses(case, k, count):
    """``count`` poses of the drive from station k - 1 to k: (fraction, pose, saved-path index or None).

    A saved Nav2 route is sampled evenly by length; otherwise the straight connector is interpolated, which is a
    drawing aid, not a validated path.
    """
    route = case.routes.get((k, k + 1))
    fractions = [(i + 1) / (count + 1) for i in range(count)]
    if route is not None and route.get("path"):
        path = np.asarray(route["path"], dtype=float)
        length = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))]
        )
        out = []
        for fraction in fractions:
            index = int(np.searchsorted(length, fraction * length[-1]))
            out.append(
                (fraction, path[min(index, len(path) - 1)], min(index, len(path) - 1))
            )
        return out
    a, b = case.poses[k - 1], case.poses[k]
    turn = math.remainder(float(b[2] - a[2]), 2.0 * math.pi)
    return [
        (f, np.array([*(a[:2] + f * (b[:2] - a[:2])), a[2] + f * turn]), None)
        for f in fractions
    ]


def _from_centre(added, row, col):
    """``added`` on the cells connected (8-neighbour) to cell (row, col) through cells that get soil; zero elsewhere."""
    labels, count = ndimage.label(added > 0.0, structure=np.ones((3, 3)))
    if count <= 1 or not labels[row, col]:
        return added if count <= 1 else np.zeros_like(added)
    return np.where(labels == labels[row, col], added, 0.0)


def cone_deposit(case, surface, x, y, volume, tan_repose, known):
    """Add ``volume`` as a cone of slope ``tan_repose`` around (x, y) on ``surface``; nothing is clipped.

    Soil spreads from the centre over the surface: a cell the cone surface lies above only beyond bare ground (a pit
    past the toe) gets none; the cone is then solved on the cells connected to the centre. Returns the window
    slices, the added thickness in the window, the apex height and whether soil would leave the known map (then the
    load keeps its volume on the map and the case is flagged).
    """
    res = case.res
    flat_radius = (3.0 * volume / (math.pi * tan_repose)) ** (1.0 / 3.0)
    radius = 2.0 * flat_radius + 3.0 * res
    height, width = case.shape
    for _ in range(8):
        c0 = int(math.floor((x - radius - case.origin[0]) / res))
        c1 = int(math.ceil((x + radius - case.origin[0]) / res)) + 1
        r0 = int(math.floor((y - radius - case.origin[1]) / res))
        r1 = int(math.ceil((y + radius - case.origin[1]) / res)) + 1
        clipped = c0 < 0 or r0 < 0 or c1 > width or r1 > height
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, width), min(r1, height)
        window = (slice(r0, r1), slice(c0, c1))
        s = surface[window]
        xs = case.origin[0] + np.arange(c0, c1) * res
        ys = case.origin[1] + np.arange(r0, r1) * res
        slope = tan_repose * np.hypot(xs[None, :] - x, ys[:, None] - y)
        top = float(s.max()) + 2.0 * flat_radius * tan_repose + 0.1
        lo, hi = float(s.min()), top
        for _ in range(60):
            apex = 0.5 * (lo + hi)
            if np.maximum(apex - slope - s, 0.0).sum() * res * res > volume:
                hi = apex
            else:
                lo = apex
        added = np.maximum(hi - slope - s, 0.0)
        row, col = (
            int(round((y - case.origin[1]) / res)) - r0,
            int(round((x - case.origin[0]) / res)) - c0,
        )
        if ndimage.label(added > 0.0, structure=np.ones((3, 3)))[1] > 1:
            lo, hi = float(s.min()), top
            for _ in range(60):
                apex = 0.5 * (lo + hi)
                if (
                    _from_centre(np.maximum(apex - slope - s, 0.0), row, col).sum()
                    * res
                    * res
                    > volume
                ):
                    hi = apex
                else:
                    lo = apex
            added = _from_centre(np.maximum(hi - slope - s, 0.0), row, col)
        rim = max(
            added[0].max(), added[-1].max(), added[:, 0].max(), added[:, -1].max()
        )
        if rim <= 0.0 or clipped:
            break
        radius *= 1.6
    total = added.sum() * res * res
    if total > 0.0:
        added *= volume / total
    off_map = bool((clipped and rim > 0.0) or np.any((added > 0.0) & ~known[window]))
    return window, added, float((s + added).max()), off_map


def station_stops(pose, tolerance_m):
    """BASE poses navigation may stop at for a station, as the converter samples them
    (radial_conversion.station_stops): the pose, 8 at half and 16 at the full tolerance.
    """
    stops = [tuple(pose)]
    for radius, count in ((tolerance_m / 2.0, 8), (tolerance_m, 16)):
        if radius > 0.0:
            for angle in np.arange(count) * 2.0 * math.pi / count:
                stops.append(
                    (
                        pose[0] + radius * math.cos(angle),
                        pose[1] + radius * math.sin(angle),
                        pose[2],
                    )
                )
    return stops


def dump_fix(
    case, rules, k, pose, payload_m3, excavated_distance, ros_distance, current_xy
):
    """What a dump edit needs: the clearance one pile of ``payload_m3`` needs, the converter's dump source and the
    nearest final-zone centre that ROS admits and keeps that pile clear, with the edit command for it.
    """
    fix = _centre_fix(
        case, rules, pose, payload_m3, excavated_distance, ros_distance, current_xy
    )
    fix["terra_step"] = case.sources[k] if case.sources[k] >= 0 else None
    used = case.dump_centres[k]
    if (
        case.sources[k] >= 0
        and used.any()
        and not (used & case.native_dump[case.sources[k]]).any()
    ):
        rows_used, cols_used = np.nonzero(used)
        ux, uy = case.xy(rows_used, cols_used)
        fix["converter_used_fallback_patch_at"] = [
            round(float(ux.mean()), 2),
            round(float(uy.mean()), 2),
        ]
    return fix


def _centre_fix(
    case, rules, pose, payload_m3, excavated_distance, ros_distance, current_xy
):
    """Pile clearance for ``payload_m3`` from ``pose`` and the nearest final-zone centre that keeps it."""
    tan_repose = math.tan(math.radians(rules.repose_deg))
    toe_m = (3.0 * payload_m3 / (math.pi * tan_repose)) ** (1.0 / 3.0)
    ros_keepout_m = (
        ROS_SUPPORT_RADIUS_M
        + float(case.dump_rule.get("min_excavated_clearance_m", 2.0 * case.tile_m))
        + case.res / math.sqrt(2.0)
    )
    # No soil in the excavation: the toe stays a cell clear of it.
    needed_m = max(toe_m + case.res, ros_keepout_m)
    rows, cols = np.nonzero(case.final_zone & case.known)
    x, y = case.xy(rows, cols)
    base = np.hypot(x - pose[0], y - pose[1])
    reach_z = float(case.dump_rule["reach_origin_xyz"][2])
    # The converter admits a centre only from every stop within the station tolerance.
    tolerance = float(case.dump_rule.get("station_tolerance_m", 0.0))
    ok = (
        (
            base - tolerance - ROS_PILE_EDGE_M
            >= float(case.dump_rule["min_base_radius_m"])
        )
        & (
            (base + tolerance) ** 2 + (ROS_APPROACH_M - reach_z) ** 2
            <= float(case.dump_rule["reach_radius_m"]) ** 2
        )
        & (excavated_distance[rows, cols] >= needed_m)
        & (ros_distance[rows, cols] >= ros_keepout_m)
    )
    fix = dict(
        pile_m3=round(payload_m3, 3),
        pile_toe_radius_m=round(toe_m, 2),
        needed_centre_distance_m=round(needed_m, 2),
        current_centre_distance_m=round(
            float(excavated_distance[_cells(case, np.array([current_xy]))][0]), 2
        ),
    )
    if ok.any():
        best = int(np.argmin(np.hypot(x[ok] - current_xy[0], y[ok] - current_xy[1])))
        fix["suggested_centre_xy_m"] = [
            round(float(x[ok][best]), 2),
            round(float(y[ok][best]), 2),
        ]
    return fix


def _fix_text(fix):
    text = (
        f" One pile of its {fix['pile_m3']:.2f} m3 reaches {fix['pile_toe_radius_m']:.2f} m, so its centre needs "
        f"{fix['needed_centre_distance_m']:.2f} m from the excavation (now {fix['current_centre_distance_m']:.2f} m)."
    )
    if "converter_used_fallback_patch_at" in fix:
        x, y = fix["converter_used_fallback_patch_at"]
        text += (
            f" The converter did not use this step's native dump permission (no admissible compact patch in it) and "
            f"dumped at its own patch near ({x}, {y}); an edit there needs room for a 0.35 m patch reachable from "
            "every stop, so widen --radius or move inward."
        )
    if fix["terra_step"] is None:
        text += " The station was added by the converter: edit a neighbouring Terra step instead."
    elif "suggested_centre_xy_m" in fix:
        x, y = fix["suggested_centre_xy_m"]
        text += f" Nearest final-zone centre that fits: plan set-dump --step {fix['terra_step']} --center {x} {y} --radius 0.45."
    else:
        text += " No final-zone centre within reach fits one pile: split the soil or move the station."
    return text


def _connected_pieces(case, mask, amount):
    """Connected pieces (8-neighbour) of ``mask`` with area, centroid and summed ``amount`` [m3]."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    pieces = []
    for index in range(1, count + 1):
        rows, cols = np.nonzero(labels == index)
        x, y = case.xy(rows, cols)
        pieces.append(
            dict(
                area_m2=round(rows.size * case.res**2, 4),
                volume_m3=round(float(amount[rows, cols].sum()) * case.res**2, 4),
                x_m=round(float(x.mean()), 3),
                y_m=round(float(y.mean()), 3),
                cells=rows * case.shape[1] + cols,
            )
        )
    return sorted(pieces, key=lambda piece: -piece["volume_m3"])


def _distance(mask, res):
    """Distance [m] from every cell centre to the nearest ``mask`` cell centre (inf without one)."""
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask) * res


def replay(case, rules=Rules(), *, runtime_validator=None):
    """Replay every retained workspace in order; returns the per-frame terrain, events, issues and totals."""
    res, area = case.res, case.res**2
    n = len(case.sources)
    design = np.where(case.target, -case.depth_m, 0.0)
    native = np.zeros(case.shape)
    # Native ground as the completion cuts alone leave it: what counts as finished. With the corridor cut, ``native``
    # (the soil forecast) is lower: a support cell outside every completion mask is cut in the forecast, but no
    # station must finish it.
    native_done = np.zeros(case.shape)
    loose = np.zeros(case.shape)
    # The part of the loose soil from loads released off the final ground (end-state rule).
    temporary = np.zeros(case.shape)
    tan_repose = math.tan(math.radians(rules.repose_deg))
    ros_clearance_m = float(
        case.dump_rule.get("min_excavated_clearance_m", 2.0 * case.tile_m)
    )
    ros_keepout_m = ROS_SUPPORT_RADIUS_M + ros_clearance_m + res / math.sqrt(2.0)
    reach_m = float(case.dump_rule["reach_radius_m"])
    reach_z = float(case.dump_rule["reach_origin_xyz"][2])
    min_base_m = float(case.dump_rule["min_base_radius_m"])
    points = body_points(case.footprint, res / 2.0)
    checker = route_check_failure(case)
    native_frames, loose_frames = [native.astype(np.float32)], [
        loose.astype(np.float32)
    ]
    done_frames = [native_done.astype(np.float32)]
    temporary_frames = [temporary >= rules.toe_m]
    events, issues = [], []
    # One entry per frame after the initial one: drive poses to a station, its cut, then each load.
    timeline = []

    def issue(kind, severity, k, step, text, x=None, y=None, **values):
        issues.append(
            dict(
                kind=kind,
                severity=severity,
                workspace=None if k is None else k + 1,
                step=step,
                x_m=None if x is None else round(float(x), 3),
                y_m=None if y is None else round(float(y), 3),
                text=text,
                **values,
            )
        )

    finished = case.required & (native_done <= design + rules.finished_m)
    # Swept design cells count as excavated from their workspace on, as ROS dug_zone history and the
    # converter treat them: the bucket cuts there even where it does not reach the design surface.
    dug = np.zeros(case.shape, dtype=bool)
    excavated_frames = [dug.copy()]

    def snapshot(k, phase, pose, **extra):
        native_frames.append(native.astype(np.float32))
        done_frames.append(native_done.astype(np.float32))
        loose_frames.append(loose.astype(np.float32))
        temporary_frames.append(temporary >= rules.toe_m)
        excavated_frames.append(dug | (native < -rules.lowering_m))
        timeline.append(
            dict(
                workspace=k + 1,
                phase=phase,
                pose=[round(float(v), 4) for v in pose],
                **extra,
            )
        )
        return len(timeline)

    for k in range(n):
        pose = case.poses[k]
        kind = case.kinds[k]
        label = f"workspace {k + 1}"
        excavated = dug | (native < -rules.lowering_m)
        classes = hazard_classes(case, rules, native, loose, excavated)
        hole_distance = _distance(excavated, res)

        # Drive: poses along the saved route, or the straight connector, on the terrain the last workspace left.
        route = case.routes.get((k, k + 1)) if k > 0 else None
        drive_steps = []
        if k > 0:
            for fraction, drive_pose, path_index in leg_poses(
                case, k, rules.drive_frames
            ):
                drive_steps.append(
                    snapshot(
                        k,
                        "drive",
                        drive_pose,
                        fraction=round(fraction, 3),
                        path_pose=path_index,
                    )
                )
        cut_step = len(timeline) + 1

        def drive_step(path_index=None):
            if not drive_steps:
                return cut_step
            if path_index is None:
                return drive_steps[0]
            poses = [timeline[step - 1]["path_pose"] for step in drive_steps]
            return drive_steps[int(np.argmin([abs(p - path_index) for p in poses]))]

        # Arrive: the route from the previous station, then the body at this station.
        route_result = None
        checker_fails_here = checker is not None and checker["to_workspace"] == k + 1
        if route is not None and not route.get("path") and checker_fails_here:
            start = body_check(case, points, case.poses[k - 1], classes, hole_distance)
            route_result = dict(
                found=False,
                nav2_error=str(route.get("error_message", ""))[:200],
                start_blocked_here=start["blocked"],
            )
        elif route is not None and not route.get("path"):
            # Nav2 found no route; the check ran on the converter's binary spoil costmap, so test the start here.
            start = body_check(case, points, case.poses[k - 1], classes, hole_distance)
            route_result = dict(
                found=False,
                nav2_error=str(route.get("error_message", ""))[:200],
                start_blocked_here=start["blocked"],
            )
            issue(
                "route_not_found",
                "runtime",
                k,
                drive_step(),
                f"{label}: the saved Nav2 check found no route ({route_result['nav2_error'][:80]}); on the replayed "
                + (
                    "terrain the body at the start is blocked by "
                    + ", ".join(start["hazards"])
                    if start["blocked"]
                    else "terrain the body at the start is clear"
                ),
                case.poses[k - 1][0],
                case.poses[k - 1][1],
                error=route_result["nav2_error"],
                start_clear=not start["blocked"],
            )
        elif route is not None:
            path = np.asarray(route["path"], dtype=float)
            first = None
            # Poses over loose soil from pass_height_m up to the /map height: drivable on a route, noted.
            high_spoil_at = []
            for index, path_pose in enumerate(path):
                check = body_check(case, points, path_pose, classes, hole_distance)
                if check["unverified"]:
                    high_spoil_at.append(index)
                if check["blocked"] and first is None:
                    first = (index, path_pose, check)
            route_result = dict(
                length_m=round(float(route["path_length_m"]), 3),
                poses=len(path),
                blocked=first is not None,
                high_spoil_poses=len(high_spoil_at),
                high_spoil_pose_indices=high_spoil_at,
                nav2_passed=bool(route.get("passed")),
            )
            if first is not None:
                index, path_pose, check = first
                route_result["first_blocked_pose"] = index
                issue(
                    "route_blocked",
                    "physical",
                    k,
                    drive_step(index),
                    f"{label}: the saved Nav2 route hits {', '.join(check['hazards'])} at pose {index}",
                    path_pose[0],
                    path_pose[1],
                    pose_index=int(index),
                    hazards=list(check["hazards"]),
                )
            elif high_spoil_at:
                x, y = path[high_spoil_at[0]][:2]
                issue(
                    "route_over_high_spoil",
                    "efficiency",
                    k,
                    drive_step(high_spoil_at[0]),
                    f"{label}: the saved Nav2 route crosses loose soil of {rules.pass_height_m:g}-"
                    f"{rules.ros_obstacle_height_m:g} m at {len(high_spoil_at)} poses (drivable with chassis balancing)",
                    x,
                    y,
                    poses=len(high_spoil_at),
                )
        if checker_fails_here:
            route_result = dict(
                route_result or {},
                checker_passed=False,
                checker_reason=checker["reason"],
            )
            unchecked = checker["unchecked"]
            where = (
                case.poses[k - 1]
                if route is None or not route.get("path")
                else route["path"][len(route["path"]) // 2]
            )
            text = f"{label}: the offline Nav2 route check fails the drive from workspace {k}: {checker['reason']}"
            if route_result.get("found") is False:
                text += "; on the replayed terrain the body at the start is " + (
                    "blocked" if route_result["start_blocked_here"] else "clear"
                )
            if len(unchecked) == 1:
                text += f"; it never checked the drive to workspace {unchecked[0]}"
            elif unchecked:
                text += (
                    f"; it never checked the {len(unchecked)} later drives "
                    f"(to workspaces {unchecked[0]}-{unchecked[-1]})"
                )
            issue(
                "route_check_failed",
                "runtime",
                k,
                drive_step(),
                text,
                where[0],
                where[1],
                reason=checker["reason"],
                unchecked_legs=len(unchecked),
            )
        arrival = body_check(case, points, pose, classes, hole_distance)
        if arrival["blocked"]:
            hazards = {
                name: count
                for name, count in arrival["hazards"].items()
                if "loose soil under" not in name
            }
            issue(
                "arrival_blocked",
                "physical",
                k,
                cut_step,
                f"{label}: the machine body stands on {', '.join(hazards)} at arrival",
                pose[0],
                pose[1],
                hazards=hazards,
            )
        elif arrival["unverified"]:
            issue(
                "arrival_unverified",
                "unverified",
                k,
                cut_step,
                f"{label}: the body stands on loose soil of {rules.pass_height_m:g} m or more "
                "(no robot limit identified)",
                pose[0],
                pose[1],
            )

        # Cut.
        completion = case.completion[k]
        swept = case.support[k] | completion
        body_rows, body_cols = np.divmod(arrival["body_cells"], case.shape[1])
        under_body = completion[body_rows, body_cols]
        if under_body.any():
            issue(
                "cut_under_body",
                "physical",
                k,
                cut_step,
                f"{label}: {int(under_body.sum())} completion cells lie under the machine body",
                pose[0],
                pose[1],
                cells=int(under_body.sum()),
            )
        rows, cols = np.nonzero(completion)
        x, y = case.xy(rows, cols)
        radius = np.hypot(x - pose[0], y - pose[1])
        # Completion rasters include every cell the band touches: allow one cell. The band is the fresh-cut
        # contract; a collection lifts loose soil anywhere in reach (cutting_band.py).
        slack = res
        outside_band = (radius < rules.band_min_m - slack) | (
            radius > rules.band_max_m + slack
        )
        if kind == "excavate" and outside_band.any():
            issue(
                "completion_out_of_band",
                "physical",
                k,
                cut_step,
                f"{label}: {int(outside_band.sum())} completion cells lie outside "
                f"{rules.band_min_m:g}-{rules.band_max_m:g} m from BASE "
                f"(nearest {radius.min():.2f} m, farthest {radius.max():.2f} m)",
                float(x[outside_band].mean()),
                float(y[outside_band].mean()),
                cells=int(outside_band.sum()),
                nearest_m=round(float(radius.min()), 2),
                farthest_m=round(float(radius.max()), 2),
            )
        if kind == "excavate":
            lifted = swept & case.target
            loose_lifted = float(loose[lifted].sum()) * area
            loose[lifted] = 0.0
            temporary[lifted] = 0.0
            depth = np.where(
                lifted if rules.cut == "corridor" else completion,
                np.maximum(native - design, 0.0),
                0.0,
            )
            native_cut = float(depth.sum()) * area
            native -= depth
            native_done -= np.where(
                completion, np.maximum(native_done - design, 0.0), 0.0
            )
            dug |= lifted
        else:
            lifted = swept if rules.cut == "corridor" else completion
            loose_lifted = float(loose[lifted].sum()) * area
            loose[lifted] = 0.0
            temporary[lifted] = 0.0
            native_cut = 0.0
        payload = loose_lifted + rules.bulking * native_cut
        newly_finished = (
            case.required & (native_done <= design + rules.finished_m) & ~finished
        )
        finished |= newly_finished
        new_required_m2 = float(newly_finished.sum()) * area
        assert snapshot(k, "cut", pose) == cut_step
        if kind == "excavate" and new_required_m2 < rules.tiny_m2:
            issue(
                "tiny_workspace",
                "efficiency",
                k,
                cut_step,
                f"{label}: finishes {new_required_m2:.2f} m2 of required ground and cuts {native_cut:.3f} m3",
                pose[0],
                pose[1],
                finished_m2=round(new_required_m2, 3),
                cut_m3=round(native_cut, 3),
            )

        # Dump.
        holes = dug | (native < -rules.lowering_m)
        hole_distance = _distance(holes, res)
        ros_blocked_distance = _distance(holes | ~case.known, res)
        c_rows, c_cols = np.nonzero(case.dump_centres[k])
        cx, cy = case.xy(c_rows, c_cols)
        base_distance = np.hypot(cx - pose[0], cy - pose[1])
        support_inside = (
            (cx - ROS_SUPPORT_RADIUS_M >= case.origin[0] - res / 2)
            & (cy - ROS_SUPPORT_RADIUS_M >= case.origin[1] - res / 2)
            & (
                cx + ROS_SUPPORT_RADIUS_M
                <= case.origin[0] + (case.shape[1] - 0.5) * res
            )
            & (
                cy + ROS_SUPPORT_RADIUS_M
                <= case.origin[1] + (case.shape[0] - 0.5) * res
            )
        )
        ros_admitted = support_inside & (
            ros_blocked_distance[c_rows, c_cols] >= ros_keepout_m - 1e-9
        )
        keep_away = base_distance - ROS_PILE_EDGE_M >= min_base_m - 1e-6
        # The runtime applies the rule from where BASE stopped; the converter admits a patch from every stop within
        # the station tolerance, but only on original ground (radial_conversion._dump_admissible, ground_z=0).
        stops = station_stops(
            pose, float(case.dump_rule.get("station_tolerance_m", 0.0))
        )
        stop_distance = np.stack(
            [np.hypot(cx - stop[0], cy - stop[1]) for stop in stops]
        )
        stop_keep_away = stop_distance - ROS_PILE_EDGE_M >= min_base_m - 1e-6
        # The workspace's dump is one deposit of its whole soil over its dump region (Lorenzo, 2 October 2026: work
        # at workspace level, never single scoops; the runtime planner picks the scoops inside the region).
        records = []
        if payload > 1e-9 and not c_rows.size:
            issue(
                "no_dump_region",
                "physical",
                k,
                cut_step,
                f"{label}: {payload:.3f} m3 to dump but the workspace has no dump centre",
                pose[0],
                pose[1],
                payload_m3=round(payload, 3),
            )
        elif payload > 1e-9:
            region = case.dump_centres[k]
            surface = native + loose
            # ROS drops the soil on the region's centres with a whole 0.45 m footprint in it, when it has some.
            window, added, off_map = plateau_deposit(
                surface,
                release_region(region, res),
                payload,
                res,
                tan_repose,
                case.known,
            )
            toe = added >= rules.toe_m
            gap = hole_distance[window] - res
            gap_m = float(gap[toe].min()) if toe.any() else math.inf
            clear = not (toe & holes[window]).any() and not off_map
            pit_spill_m3 = float(added[holes[window]].sum()) * area
            after = surface.copy()
            after[window] += added
            ground = after[c_rows, c_cols]  # the deposit's surface at each centre
            reach = (
                base_distance**2 + (ground + ROS_APPROACH_M - reach_z) ** 2
                <= reach_m**2 + 1e-6
            ) & keep_away
            # Stops within the tolerance from which ROS admits no centre at the deposit's surface: reach, keep-away
            # from that stop, support clear of excavation.
            stop_admitted = (
                (
                    stop_distance**2 + (ground + ROS_APPROACH_M - reach_z) ** 2
                    <= reach_m**2 + 1e-6
                )
                & stop_keep_away
                & ros_admitted
            )
            stranded_stops = int(np.count_nonzero(~stop_admitted.any(axis=1)))
            admitted = reach & ros_admitted
            choice = (
                "ros"
                if admitted.any()
                else "modeled_only" if clear else "no_clear_point"
            )
            loose[window] += added
            # The share of the region on final ground releases final soil wherever it spreads, the rest temporary.
            final_share = float(case.final_ground[c_rows, c_cols].mean())
            temporary[window] += added * (1.0 - final_share)
            toe_rows, toe_cols = np.nonzero(toe)
            toe_x, toe_y = case.xy(
                toe_rows + window[0].start, toe_cols + window[1].start
            )
            x_m, y_m = float(cx.mean()), float(cy.mean())
            record = dict(
                load=1,
                volume_m3=round(payload, 4),
                bucket_loads=int(math.ceil(payload / rules.bucket_m3 - 1e-9)),
                x_m=round(x_m, 3),
                y_m=round(y_m, 3),
                choice=choice,
                ros_admitted=bool(ros_admitted.any()),
                reachable=bool(reach.any()),
                pile_clear=bool(clear),
                gap_to_excavation_m=None if math.isinf(gap_m) else round(gap_m, 3),
                pile_top_m=round(
                    (
                        float(after[window][toe].max())
                        if toe.any()
                        else float(ground.max())
                    ),
                    3,
                ),
                soil_top_m=round(float(added.max()), 3),
                toe_radius_m=(
                    round(float(np.hypot(toe_x - x_m, toe_y - y_m).max()), 3)
                    if toe_x.size
                    else 0.0
                ),
                pit_spill_m3=round(pit_spill_m3, 4),
                off_map=off_map,
                final_release=final_share >= 0.5,
                final_share=round(final_share, 3),
                centres=int(c_rows.size),
                centres_ros_admitted=int(np.count_nonzero(admitted)),
                stops=len(stops),
                stops_without_admitted_centre=stranded_stops,
            )
            # Under min_leftover_m3 of soil in the excavation is below the pile model's resolution: the toe touches
            # the pit (marginal), it does not spill into it.
            record["spill"] = (
                "clear"
                if record["pile_clear"]
                else (
                    "spills"
                    if record["pit_spill_m3"] >= rules.min_leftover_m3
                    else "marginal"
                )
            )
            record["step"] = snapshot(k, "load", pose, load=1, loads=1)
            records.append(record)
        dump = dict(loads=records)
        for record in records:
            # One deposit per workspace (workspace level): its issues name the deposit, not single loads.
            where = (record["x_m"], record["y_m"])
            if record["choice"] == "modeled_only":
                issue(
                    "dump_ros_refuses",
                    "runtime",
                    k,
                    record["step"],
                    f"{label}: ROS admits no centre of the dump region at the deposit's surface (its 1.1 m support "
                    f"must stay {ros_clearance_m:.2f} m from excavation, the reach is tested at the pile top); the "
                    "modeled deposit itself stays clear",
                    *where,
                    loads=1,
                    of=1,
                )
            if not record["pile_clear"]:
                deep = record["spill"] == "spills"
                reason = (
                    "no centre keeps the deposit out of the excavation and ROS admits none"
                    if record["choice"] == "no_clear_point"
                    else "the deposit of the workspace's soil outgrows its region's clearance"
                )
                fix = None
                if deep:
                    fix = dump_fix(
                        case,
                        rules,
                        k,
                        pose,
                        payload,
                        hole_distance,
                        ros_blocked_distance,
                        where,
                    )
                source = (
                    "added by the converter"
                    if case.sources[k] < 0
                    else f"Terra step {case.sources[k]}"
                )
                issue(
                    "dump_into_pit" if deep else "dump_touches_pit",
                    "physical" if deep else "unverified",
                    k,
                    record["step"],
                    f"{label} ({source}): its {payload:.2f} m3 deposit puts soil in the excavation ({reason}); "
                    f"{record['pit_spill_m3']:.3f} m3 in the pit."
                    + ("" if fix is None else _fix_text(fix)),
                    *where,
                    loads=1,
                    fix=fix,
                    of=1,
                    pit_m3=round(record["pit_spill_m3"], 4),
                )
            if not record["reachable"]:
                issue(
                    "dump_out_of_reach",
                    "runtime",
                    k,
                    record["step"],
                    f"{label}: no dump centre within {reach_m:g} m CABIN reach of the deposit's surface "
                    f"(top {record['pile_top_m']:.2f} m)",
                    pose[0],
                    pose[1],
                    loads=1,
                    of=1,
                )
            if record["stops_without_admitted_centre"]:
                issue(
                    "dump_unreachable_from_stop",
                    "runtime",
                    k,
                    record["step"],
                    f"{label}: {record['stops_without_admitted_centre']} of {record['stops']} stops within the "
                    f"{float(case.dump_rule.get('station_tolerance_m', 0.0)):g} m station tolerance keep no dump centre "
                    f"ROS admits at the deposit's surface (top {record['pile_top_m']:.2f} m)",
                    *where,
                    loads=1,
                    of=1,
                    stops=record["stops_without_admitted_centre"],
                )
            if record["off_map"]:
                issue(
                    "pile_off_map",
                    "unverified",
                    k,
                    record["step"],
                    f"{label}: its deposit reaches ground off the known map",
                    pose[0],
                    pose[1],
                )
        events.append(
            dict(
                workspace=k + 1,
                kind=kind,
                source_pair=case.sources[k] if case.sources[k] >= 0 else None,
                pose=[round(float(v), 4) for v in pose],
                arrival=dict(
                    blocked=arrival["blocked"],
                    unverified=arrival["unverified"],
                    hazards=arrival["hazards"],
                    gap_to_excavation_m=(
                        None
                        if arrival["gap_to_excavation_m"] is None
                        or math.isinf(arrival["gap_to_excavation_m"])
                        else round(arrival["gap_to_excavation_m"], 3)
                    ),
                    blocked_cells=arrival["blocked_cells"],
                    unverified_cells=arrival["unverified_cells"],
                ),
                route=route_result,
                steps=dict(
                    drive=drive_steps, cut=cut_step, loads=[r["step"] for r in records]
                ),
                cut=dict(
                    completion_cells=int(completion.sum()),
                    new_required_m2=round(new_required_m2, 4),
                    native_cut_m3=round(native_cut, 4),
                    loose_lifted_m3=round(loose_lifted, 4),
                    payload_m3=round(payload, 4),
                    completion_radius_m=(
                        [round(float(radius.min()), 3), round(float(radius.max()), 3)]
                        if radius.size
                        else None
                    ),
                ),
                dump=dump,
            )
        )

    # End state.
    last_step = len(timeline)
    if checker is not None and checker["to_workspace"] is None and n:
        k = 0 if checker["failure"] == "initial_station_body_blocked" else n - 1
        issue(
            "route_check_failed",
            "runtime",
            k,
            events[k]["steps"]["cut"],
            f"workspace {k + 1}: the offline Nav2 route check fails: {checker['reason']}"
            + (
                f"; it never checked the {len(checker['unchecked'])} drives"
                if checker["unchecked"]
                else ""
            ),
            case.poses[k][0],
            case.poses[k][1],
            reason=checker["reason"],
            unchecked_legs=len(checker["unchecked"]),
        )
    final_native, final_loose, final_temporary = native, loose, temporary
    remaining = np.maximum(native_done - design, 0.0)
    left = case.required & (remaining > rules.finished_m)
    # The converter records the required area a complete plan may leave (none by default).
    tolerance_m2 = float(case.report.get("required_residual_tolerance_m2") or 0.0)
    within = bool(left.any()) and float(left.sum()) * area <= tolerance_m2 + 1e-9
    for piece in _connected_pieces(case, left, remaining):
        issue(
            "required_left_within_tolerance" if within else "required_left",
            "efficiency" if within else "physical",
            None,
            last_step,
            f"required soil left: {piece['area_m2']:.2f} m2, {piece['volume_m3']:.3f} m3 near "
            f"({piece['x_m']:.1f}, {piece['y_m']:.1f})"
            + (f", within the plan's {tolerance_m2:g} m2 tolerance" if within else ""),
            piece["x_m"],
            piece["y_m"],
            area_m2=piece["area_m2"],
            volume_m3=piece["volume_m3"],
        )
    fell_back = case.required & ~left & (final_loose >= rules.toe_m)
    for piece in _connected_pieces(case, fell_back, final_loose):
        if piece["volume_m3"] < rules.min_leftover_m3:
            continue
        issue(
            "loose_in_finished_excavation",
            "physical",
            None,
            last_step,
            f"loose soil in finished excavation: {piece['volume_m3']:.3f} m3 near ({piece['x_m']:.1f}, {piece['y_m']:.1f})",
            piece["x_m"],
            piece["y_m"],
            volume_m3=piece["volume_m3"],
        )
    # Terra's plans end with all soil on the final zones; a converted plan must move its soil there itself (Lorenzo,
    # 1 October 2026). A final zone grows where its own piles spread: only soil of loads released off it counts.
    left_outside = (
        (final_temporary >= rules.toe_m) & ~case.final_ground & ~case.required
    )
    for piece in _connected_pieces(case, left_outside, final_temporary):
        if piece["volume_m3"] < rules.min_leftover_m3:
            continue
        issue(
            "temporary_spoil_left",
            "physical",
            None,
            last_step,
            f"soil of loads released off the final zones left outside them at the end: {piece['volume_m3']:.3f} m3 "
            f"near ({piece['x_m']:.1f}, {piece['y_m']:.1f})",
            piece["x_m"],
            piece["y_m"],
            volume_m3=piece["volume_m3"],
        )
    for error in case.report.get("dump_sequence_errors", []):
        k = int(error["workspace"]) - 1
        note = ""
        if error["reason"] == "excavation_before_collection":
            note = " (superseded rule: a cut may remove spoil)"
        elif error["reason"] == "no_dump_support_for_every_station_stop":
            loads = events[k]["dump"]["loads"] if k < len(events) else []
            outside = int(((case.support[k] | case.completion[k]) & ~case.target).sum())
            note = (
                f" (1.1 m support + {ros_clearance_m:.2f} m gap; the converter also counts {outside} swept cells "
                "outside the design as excavated"
                + (
                    "; with swept design cells only, ROS admits the replay's centres)"
                    if loads and all(r["choice"] == "ros" for r in loads)
                    else ")"
                )
            )
        steps = events[k]["steps"] if k < len(events) else dict(cut=last_step, loads=[])
        fix = None
        if error["reason"] == "no_dump_support_for_every_station_stop" and k < len(
            events
        ):
            # Where this station's soil could go instead, measured against the converter's holes: every cell the
            # exported supports of this and earlier excavations touch (grown by the station tolerance).
            holes = np.logical_or.reduce(
                [np.zeros(case.shape, bool)]
                + [case.support[j] for j in range(k + 1) if case.kinds[j] == "excavate"]
            )
            centre = [case.poses[k][0], case.poses[k][1]]
            if case.dump_centres[k].any():
                rows, cols = np.nonzero(case.dump_centres[k])
                xs, ys = case.xy(rows, cols)
                centre = [float(xs.mean()), float(ys.mean())]
            fix = dump_fix(
                case,
                rules,
                k,
                case.poses[k],
                events[k]["cut"]["payload_m3"],
                _distance(holes, res),
                _distance(holes | ~case.known, res),
                centre,
            )
        issue(
            "converter_" + error["reason"],
            "converter",
            k,
            (steps["loads"] or [steps["cut"]])[-1],
            f"converter: workspace {k + 1} {error['reason'].replace('_', ' ')}{note}"
            + ("" if fix is None else _fix_text(fix)),
            case.poses[k][0],
            case.poses[k][1],
            **({} if fix is None else {"fix": fix}),
        )
    if not case.complete:
        # Native steps the converter rejected: their ground went to other stations or stays owed. A rejection for want
        # of a dump gets the same suggestion as a pile spill, against the digs queued before it and its own.
        accepted = {s for s in case.sources if s >= 0}
        rejected = {}
        for pair in case.report.get("per_pair", []):
            source = pair.get("pair_index", -1)
            if source >= 0 and not pair.get("accepted") and source not in accepted:
                counts = rejected.setdefault(
                    source, dict(pose=pair.get("original_pose"), reasons={})
                )
                for reason, count in (pair.get("placement_rejections") or {}).items():
                    counts["reasons"][reason] = counts["reasons"].get(reason, 0) + count
        for source, entry in sorted(rejected.items()):
            if entry["pose"] is None or source >= len(case.native_dig):
                continue
            reasons = sorted(entry["reasons"].items(), key=lambda item: -item[1])
            position = next((k for k in range(n) if case.sources[k] > source), n)
            holes = case.native_dig[source] | np.logical_or.reduce(
                [np.zeros(case.shape, bool)]
                + [
                    case.support[j]
                    for j in range(position)
                    if case.kinds[j] == "excavate"
                ]
            )
            fix = None
            if any("dump" in reason for reason, _ in reasons):
                payload = (
                    float((case.native_dig[source] & case.target).sum())
                    * area
                    * case.depth_m
                )
                intended = case.native_dump[source]
                if intended.any():
                    rows, cols = np.nonzero(intended)
                    xs, ys = case.xy(rows, cols)
                    current = [float(xs.mean()), float(ys.mean())]
                else:
                    current = list(entry["pose"][:2])
                fix = _centre_fix(
                    case,
                    rules,
                    entry["pose"],
                    payload,
                    _distance(holes, res),
                    _distance(holes | ~case.known, res),
                    current,
                )
                fix["terra_step"] = source
            text = f"converter rejected Terra step {source}: " + ", ".join(
                f"{reason.replace('_', ' ')} ({count})" for reason, count in reasons[:3]
            )
            issue(
                "converter_step_rejected",
                "converter",
                None,
                last_step,
                text + ("" if fix is None else _fix_text(fix)),
                entry["pose"][0],
                entry["pose"][1],
                terra_step=source,
                reasons=dict(reasons),
                **({} if fix is None else {"fix": fix}),
            )
    for error in case.report.get("completion_band_errors", []):
        k = int(error.get("workspace", 0)) - 1 if isinstance(error, dict) else -1
        step = events[k]["steps"]["cut"] if 0 <= k < len(events) else 1
        issue(
            "converter_completion_band",
            "converter",
            k if k >= 0 else None,
            step,
            f"converter: {error}",
        )

    loads = [r for event in events for r in event["dump"]["loads"]]
    connectors = [
        float(np.hypot(*(case.poses[k][:2] - case.poses[k - 1][:2])))
        for k in range(1, n)
    ]
    routed = [
        event["route"]["length_m"]
        for event in events
        if event["route"] and "length_m" in event["route"]
    ]
    totals = dict(
        workspaces=n,
        added_workspaces=sum(s < 0 for s in case.sources),
        collections=sum(kind != "excavate" for kind in case.kinds),
        required_m2=round(float(case.required.sum()) * area, 3),
        required_finished_m2=round(float((case.required & ~left).sum()) * area, 3),
        required_left_m2=round(float(left.sum()) * area, 3),
        required_left_m3=round(float(remaining[left].sum()) * area, 3),
        native_cut_m3=round(sum(e["cut"]["native_cut_m3"] for e in events), 3),
        rehandled_loose_m3=round(sum(e["cut"]["loose_lifted_m3"] for e in events), 3),
        # One deposit per workspace with soil (workspace level); bucket_loads only counts the bucket fills it holds.
        loads=len(loads),
        bucket_loads=sum(r["bucket_loads"] for r in loads),
        loads_ros_refused=sum(r["choice"] != "ros" for r in loads),
        loads_unreachable_from_some_stop=sum(
            r["stops_without_admitted_centre"] > 0 for r in loads
        ),
        loads_into_pit=sum(r["spill"] == "spills" for r in loads),
        loads_touching_pit=sum(r["spill"] == "marginal" for r in loads),
        loads_released_off_final=sum(not r["final_release"] for r in loads),
        loose_on_ground_m3=round(float(final_loose.sum()) * area, 3),
        temporary_spoil_left_m3=round(
            float(final_temporary[left_outside].sum()) * area, 3
        ),
        final_soil_past_final_zones_m3=round(
            float(
                (final_loose - final_temporary)[
                    ~case.final_ground & ~case.required & (final_loose >= rules.toe_m)
                ].sum()
            )
            * area,
            3,
        ),
        loose_in_finished_m3=round(float(final_loose[fell_back].sum()) * area, 3),
        highest_pile_m=round(float(final_loose.max()), 3),
        arrivals_blocked=sum(e["arrival"]["blocked"] for e in events),
        arrivals_unverified=sum(
            e["arrival"]["unverified"] and not e["arrival"]["blocked"] for e in events
        ),
        straight_connectors_m=round(sum(connectors), 2),
        saved_routes=len(routed),
        saved_routes_m=round(sum(routed), 2),
    )
    return dict(
        case=case,
        rules=rules,
        native_frames=np.stack(native_frames),
        done_frames=np.stack(done_frames),
        loose_frames=np.stack(loose_frames),
        temporary_frames=np.stack(temporary_frames),
        excavated_frames=np.stack(excavated_frames),
        design=design,
        events=events,
        timeline=timeline,
        issues=issues,
        totals=totals,
        runtime=runtime_checks(case, runtime_validator) if case.complete else None,
        route_check=checker,
    )


def _entry(issue):
    """A checklist entry: where and what, plus the issue's measured values (and a suggested fix, if any)."""
    bookkeeping = {"severity", "workspace", "step", "text", "x_m", "y_m"}
    return dict(
        workspace=issue.get("workspace"),
        step=issue.get("step"),
        text=issue["text"],
        x_m=issue.get("x_m"),
        y_m=issue.get("y_m"),
        **{
            key: value
            for key, value in issue.items()
            if key not in bookkeeping and value is not None
        },
    )


def coverage_area(report):
    """The conversion report's coverage areas (required, covered, residual m2 and covered fraction), from its
    required geometry and residual when it predates them; None without either."""
    if report.get("coverage_area") is not None:
        return report["coverage_area"]
    required, residual = report.get("required_geometry_area_m2"), report.get(
        "continuous_required_residual_m2"
    )
    if required is None or residual is None:
        return None
    covered = max(float(required) - float(residual), 0.0)
    return dict(
        required_m2=float(required),
        covered_m2=covered,
        residual_m2=float(residual),
        covered_fraction=covered / float(required) if float(required) > 0.0 else 1.0,
    )


def validity(result):
    """The checklist and verdict of one replay: each rule passes, fails, is unverified or was not checked; notes are
    facts a reader should know that fail nothing (rejected Terra stations whose ground other workspaces dig).
    """
    case, events, issues = result["case"], result["events"], result["issues"]
    rules = result["rules"]
    checks = []
    for check_id, rule, failing, uncertain in CHECKS:
        fails = [i for i in issues if i["kind"] in failing]
        unverified = [i for i in issues if i["kind"] in uncertain]
        notes = []
        detail = ""
        not_checked = False
        if check_id == "converter_plan":
            report = case.report
            residual = report.get("continuous_required_residual_m2")
            area = coverage_area(report)
            detail = (
                f"complete_geometric_plan={bool(report.get('complete_geometric_plan'))}, "
                f"plan written={case.complete}, coverage {100.0 * float(report.get('coverage_fraction', 0.0)):.2f} %, "
                f"continuous residual {residual if residual is None else round(float(residual), 4)} m2"
            )
            if area is not None:
                detail += (
                    f", covers {area['covered_m2']:.2f} of {area['required_m2']:.2f} m2 required "
                    f"({100.0 * area['covered_fraction']:.2f} %)"
                )
            # Terra stations the converter rejected whose ground other workspaces dig, and loads at the end of Terra's
            # plan it never unloads, are notes (Lorenzo, 2 October 2026), not failures.
            rejected = report.get("rejected_terra_stations")
            excused = {
                note["pair_index"]
                for note in rejected or []
                if not note["blocks_completion"]
            }
            notes = [
                dict(
                    kind="converter_station_rejected",
                    text=(
                        f"converter rejected Terra station {note['pair_index']} (steps {note['source_steps']}): "
                        + (
                            ", ".join(
                                f"{reason.replace('_', ' ')} ({count})"
                                for reason, count in note["reasons"].items()
                            )
                            # Legal stances that dig none of what was left of its assignment record no reason.
                            or "no stance dug any of its remaining ground"
                        )
                        + f"; {note['note']}"
                    ),
                    terra_step=note["pair_index"],
                )
                for note in rejected or []
                if not note["blocks_completion"]
            ] + [
                dict(
                    kind="trailing_lift_dropped",
                    step=None,
                    text=f"Terra step {entry['step']}: {entry['note']}",
                )
                for entry in report.get("dropped_trailing_lifts", [])
            ]
            if notes:
                detail += f"; {len(notes)} notes"
            if not case.complete:
                fails = [
                    dict(
                        kind="no_plan",
                        workspace=None,
                        step=len(result["timeline"]),
                        text="no executable plan: " + detail,
                        coverage_pct=round(
                            100.0 * float(report.get("coverage_fraction", 0.0)), 2
                        ),
                        residual_m2=(
                            None if residual is None else round(float(residual), 4)
                        ),
                    )
                ] + [
                    i
                    for i in issues
                    if i["kind"] == "converter_step_rejected"
                    and (rejected is None or i["terra_step"] not in excused)
                ]
            runtime = result.get("runtime")
            if runtime is not None:
                detail += f", runtime checks: {runtime['status']}"
                if runtime["status"] == "fail":
                    # The runtime names the converted waypoint it refuses; waypoints are dig/dump pairs per workspace.
                    waypoint = re.search(r"waypoint\[(\d+)\]", runtime["message"])
                    k = int(waypoint.group(1)) // 2 if waypoint else None
                    k = k if k is not None and k < len(events) else None
                    fails = [
                        dict(
                            kind="runtime_refused",
                            workspace=None if k is None else k + 1,
                            step=(
                                len(result["timeline"])
                                if k is None
                                else events[k]["steps"]["cut"]
                            ),
                            text=f"the runtime refuses the plan: {runtime['message']}",
                            x_m=(
                                None if k is None else round(float(case.poses[k][0]), 3)
                            ),
                            y_m=(
                                None if k is None else round(float(case.poses[k][1]), 3)
                            ),
                            message=runtime["message"],
                        )
                    ]
                elif runtime["status"] == "not checked":
                    not_checked = True
                    detail += f" ({runtime['message']})"
        elif check_id == "routes":
            legs = max(len(events) - 1, 0)
            saved = sum(1 for e in events[1:] if e["route"])
            detail = f"{saved} of {legs} drives have a saved Nav2 route"
            not_checked = saved < legs
            checker = result.get("route_check")
            if checker is not None and checker["unchecked"]:
                detail += f"; the offline route check stopped early, {len(checker['unchecked'])} drives never checked"
            high = sum(i["kind"] == "route_over_high_spoil" for i in issues)
            if high:
                detail += (
                    f"; {high} drives cross loose soil of {rules.pass_height_m:g}-{rules.ros_obstacle_height_m:g} m, "
                    "drivable with chassis balancing"
                )
        elif check_id == "dump_admission":
            rule = rule.format(
                reach=float(case.dump_rule["reach_radius_m"]),
                keep_away=float(case.dump_rule["min_base_radius_m"]),
            )
            detail = "checked at the nominal station pose only; the converter samples 25 stops within 0.35 m"
        elif check_id == "coverage":
            within = [
                i for i in issues if i["kind"] == "required_left_within_tolerance"
            ]
            if within:
                detail = f"{sum(i['area_m2'] for i in within):.2f} m2 left, within the plan's tolerance"
        elif check_id == "spoil_at_end":
            totals = result["totals"]
            outside = totals["temporary_spoil_left_m3"]
            if outside > 0:
                detail = (
                    f"{outside:.2f} m3 of loads released off the final zones left outside them, pieces under "
                    f"{rules.min_leftover_m3:g} m3 not counted"
                )
            if totals["final_soil_past_final_zones_m3"] > 0:
                detail += ("; " if detail else "") + (
                    f"{totals['final_soil_past_final_zones_m3']:.2f} m3 of final loads spread past the zones (counts as "
                    "final)"
                )
        elif check_id == "pile_clearance":
            loads = [r for e in events for r in e["dump"]["loads"]]
            gaps = [
                r["gap_to_excavation_m"]
                for r in loads
                if r["gap_to_excavation_m"] is not None
            ]
            detail = (
                f"{len(loads)} workspace deposits; closest toe {min(gaps):.2f} m from the excavation"
                if gaps
                else f"{len(loads)} workspace deposits"
            )
        if fails:
            status = "fail"
        elif unverified:
            status = "unverified"
        elif not_checked:
            status = "not checked"
        else:
            status = "pass"
        checks.append(
            dict(
                id=check_id,
                rule=rule,
                status=status,
                detail=detail,
                failures=[_entry(i) for i in fails],
                unverified=[_entry(i) for i in unverified],
                notes=[_entry(i) for i in notes],
            )
        )
    statuses = {c["status"] for c in checks}
    verdict = (
        "invalid"
        if "fail" in statuses
        else (
            "valid with unverified items"
            if statuses & {"unverified", "not checked"}
            else "valid under the modeled rules"
        )
    )
    return dict(verdict=verdict, checks=checks)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def report(result):
    """The JSON report: inputs, rules, totals, per-workspace events and every issue."""
    case = result["case"]
    converter = case.report
    return _jsonable(
        dict(
            case=case.tag,
            validity=validity(result),
            conversion=str(case.conversion),
            native_source=converter["inputs"]["input_dir"],
            converter=dict(
                complete_geometric_plan=bool(converter.get("complete_geometric_plan")),
                # Required area, covered area, residual [m2] and covered fraction (conversion report).
                coverage_area=coverage_area(converter),
                executable_plan_written=case.complete,
                coverage_fraction=converter.get("coverage_fraction"),
                continuous_required_residual_m2=converter.get(
                    "continuous_required_residual_m2"
                ),
                dump_sequence_errors=len(converter.get("dump_sequence_errors", [])),
                fresh_cutting_rule=converter.get("fresh_cutting_rule"),
                dump_rule=case.dump_rule,
                runtime_checks=result.get("runtime"),
            ),
            rules=asdict(result["rules"]) | dict(tile_m=case.tile_m),
            not_modeled=list(NOT_MODELED),
            totals=result["totals"],
            issue_counts={
                severity: sum(i["severity"] == severity for i in result["issues"])
                for severity in SEVERITIES
            },
            issues=result["issues"],
            events=result["events"],
        )
    )


# ---------------------------------------------------------------------------------------------------------
# webviz run history (moleworks_newton scripts/benchmark/webviz/README.md): one run per plan, two steps per
# workspace (arrive and cut, then dump), frame 0 the initial ground.

CROP_MARGIN_M = 1.5
PAGE_ELEVATION_BYTES = 6_000_000  # per run: the viewer inlines every frame, so coarsen the page grid above this
CIRCLE_POINTS = 48
LAYERS = (
    (
        "design",
        "main",
        True,
        [
            ("required, not finished", "#f59e0b"),
            ("required, finished", "#14b8a6"),
            ("optional design edge, not dug", "#fde68a"),
        ],
    ),
    (
        "this workspace",
        "main",
        True,
        [
            ("completion cells", "#16a34a"),
            ("other swept cells", "#86efac"),
            ("dump centres ROS may choose", "#c4b5fd"),
        ],
    ),
    (
        "loose soil",
        "main",
        True,
        [
            ("under 0.5 m: drivable", "#d6a86b"),
            ("0.5 m and more: unverified", "#f97316"),
            ("ROS /map obstacle (1.0 m over original ground)", "#7c2d12"),
            ("in finished excavation", "#db2777"),
        ],
    ),
    (
        "excavation",
        "main",
        True,
        [("excavated or swept ground", "#0ea5e9")],
    ),
    (
        "station",
        "main",
        True,
        [
            ("machine body", "--text"),
            ("body on blocking ground", "#dc2626"),
            ("body on unverified spoil", "#f59e0b"),
            ("completion band 4.0 / 6.5 m", "#f97316"),
            ("entry 7.0 m", "--muted"),
            ("heading", "--text"),
        ],
    ),
    (
        "dump loads",
        "main",
        True,
        [
            ("chosen point", "#7c3aed"),
            ("modeled pile toe", "#a78bfa"),
            ("earlier points", "#c4b5fd"),
            ("dump reach and keep-away", "--muted"),
        ],
    ),
    (
        "driving",
        "main",
        True,
        [
            ("straight connector (not a route)", "--muted"),
            ("saved Nav2 route", "#2563eb"),
            ("route blocked here", "#dc2626"),
        ],
    ),
    (
        "issues",
        "main",
        True,
        [
            (severity, color)
            for severity, color in zip(
                SEVERITIES, ("#dc2626", "#ea580c", "#ca8a04", "#2563eb", "#6b7280")
            )
        ],
    ),
    (
        "Terra final dump zones and obstacles",
        "more",
        False,
        [
            ("final dump zone", "#e9d5ff"),
            ("static obstacle", "#111827"),
            ("final zone grown for pile spread", "#f5f0ff"),
        ],
    ),
    (
        "converter's own view",
        "more",
        False,
        [
            ("blocked at arrival (binary spoil rule)", "#9ca3af"),
            ("1.1 m deposit support", "#d1d5db"),
        ],
    ),
    (
        "Terra native stations",
        "more",
        False,
        [("Terra station", "#a3a3a3"), ("moved to", "#737373")],
    ),
)
LAYER = {name: index for index, (name, _, _, _) in enumerate(LAYERS)}
KIND_LABELS = {
    "arrival_blocked": "body on blocking ground at arrival",
    "arrival_unverified": "body on spoil of 0.5 m or more at arrival",
    "route_blocked": "saved route blocked",
    "route_over_high_spoil": "saved route over spoil of 0.5-1.0 m (drivable with chassis balancing)",
    "route_not_found": "saved Nav2 check found no route",
    "route_check_failed": "offline Nav2 route check failed",
    "cut_under_body": "completion cells under the body",
    "completion_out_of_band": "completion cells outside 4.0-6.5 m",
    "no_dump_region": "soil to dump but no dump region",
    "dump_ros_refuses": "ROS refuses every dump centre (pile model would be clear)",
    "dump_into_pit": "pile puts soil in the excavation",
    "dump_touches_pit": "pile toe touches the excavation (under 0.02 m3)",
    "dump_out_of_reach": "dump out of reach",
    "pile_off_map": "pile off the known map",
    "tiny_workspace": "tiny workspace",
    "required_left": "required soil left",
    "loose_in_finished_excavation": "loose soil in finished excavation",
    "temporary_spoil_left": "soil of loads released off the final zones left outside them",
}


def _circle(x, y, radius):
    angles = np.linspace(0.0, 2.0 * math.pi, CIRCLE_POINTS + 1)
    return [
        [round(x + radius * math.cos(a), 3), round(y + radius * math.sin(a), 3)]
        for a in angles
    ]


def _crop(result):
    case = result["case"]
    keep = case.target | case.dump_centres.any(axis=0) | case.completion.any(axis=0)
    keep |= result["loose_frames"].max(axis=0) >= result["rules"].toe_m
    rows, cols = np.nonzero(keep)
    x, y = case.xy(rows, cols)
    xs, ys = [x.min(), x.max()], [y.min(), y.max()]
    for pose in case.poses:
        polygon = world_polygon(case.footprint, pose)
        xs += [polygon[:, 0].min(), polygon[:, 0].max()]
        ys += [polygon[:, 1].min(), polygon[:, 1].max()]
    for route in case.routes.values():
        if route.get("path"):
            path = np.asarray(route["path"])
            xs += [path[:, 0].min(), path[:, 0].max()]
            ys += [path[:, 1].min(), path[:, 1].max()]
    c0 = max(int(math.floor((min(xs) - CROP_MARGIN_M - case.origin[0]) / case.res)), 0)
    c1 = min(
        int(math.ceil((max(xs) + CROP_MARGIN_M - case.origin[0]) / case.res)) + 1,
        case.shape[1],
    )
    r0 = max(int(math.floor((min(ys) - CROP_MARGIN_M - case.origin[1]) / case.res)), 0)
    r1 = min(
        int(math.ceil((max(ys) + CROP_MARGIN_M - case.origin[1]) / case.res)) + 1,
        case.shape[0],
    )
    return r0, r1, c0, c1


def _runs(stack):
    """(start, end, cell) of every run of True along axis 0 of a (frames, cells) array; end -1 = last frame."""
    frames = stack.shape[0]
    padded = np.zeros((frames + 2, stack.shape[1]), dtype=np.int8)
    padded[1:-1] = stack
    change = np.diff(padded, axis=0)
    start_frame, start_cell = np.nonzero(change == 1)
    end_frame, end_cell = np.nonzero(change == -1)
    start = np.lexsort((start_frame, start_cell))
    end = np.lexsort((end_frame, end_cell))
    last = end_frame[end] - 1
    last[last == frames - 1] = -1
    return start_frame[start], last, start_cell[start]


def _block(array, factor, reduce):
    """Reduce the last two axes by ``factor`` x ``factor`` blocks (padded at the far edge by the edge values)."""
    if factor == 1:
        return array
    *lead, h, w = array.shape
    pad = [(0, 0)] * len(lead) + [(0, -h % factor), (0, -w % factor)]
    array = np.pad(array, pad, mode="edge")
    h, w = array.shape[-2:]
    blocks = array.reshape(*lead, h // factor, factor, w // factor, factor)
    return reduce(blocks, axis=(-3, -1))


def history_arrays(result, display_res_m=None):
    """The run's arrays under the webviz contract, keyed by sub-key (without the ``tag__`` prefix).

    ``display_res_m`` coarsens the page grid (block means of height, any-cell masks) so long plans fit the
    viewer's page size; the replay itself and its JSON report stay on the converter's grid.
    """
    case, rules = result["case"], result["rules"]
    r0, r1, c0, c1 = _crop(result)
    frames_total = len(result["timeline"]) + 1
    if display_res_m is None:
        factor = 1
        while (
            frames_total * -(-(r1 - r0) // factor) * -(-(c1 - c0) // factor) * 2
            > PAGE_ELEVATION_BYTES
        ):
            factor += 1
    else:
        factor = max(1, int(round(display_res_m / case.res)))
    window = (slice(r0, r1), slice(c0, c1))
    h, w = r1 - r0, c1 - c0
    native = result["native_frames"][:, r0:r1, c0:c1].astype(np.float64)
    done = result["done_frames"][:, r0:r1, c0:c1].astype(np.float64)
    loose = result["loose_frames"][:, r0:r1, c0:c1].astype(np.float64)
    excavated = result["excavated_frames"][:, r0:r1, c0:c1]
    design = result["design"][window]
    frames = native.shape[0]
    n = len(case.sources)
    origin = case.origin + np.array([c0, r0]) * case.res
    hc, wc = -(-h // factor), -(-w // factor)
    cells = []

    def add_stack(layer, category, stack):
        start, end, cell = _runs(_block(stack, factor, np.any).reshape(frames, -1))
        if start.size:
            cells.append(
                np.column_stack(
                    [
                        np.full(start.size, LAYER[layer]),
                        np.full(start.size, category),
                        start,
                        end,
                        cell,
                    ]
                )
            )

    def add_step(layer, category, step, mask_or_cells, until=None):
        """``mask_or_cells`` from ``step`` to ``until`` (default ``step``): a crop mask, or flat crop-cell indices."""
        if np.asarray(mask_or_cells).dtype == bool:
            flat = np.flatnonzero(_block(mask_or_cells, factor, np.any))
        else:
            rows, cols = np.divmod(np.asarray(mask_or_cells, dtype=int), w)
            flat = np.unique((rows // factor) * wc + cols // factor)
        if len(flat):
            cells.append(
                np.column_stack(
                    [
                        np.full(len(flat), LAYER[layer]),
                        np.full(len(flat), category),
                        np.full(len(flat), step),
                        np.full(len(flat), step if until is None else until),
                        flat,
                    ]
                )
            )

    required = case.required[window]
    target = case.target[window]
    remaining = native > design[None] + rules.finished_m
    # Required ground is finished by the completion cuts (done), the rest of the design shows the soil forecast.
    unfinished = done > design[None] + rules.finished_m
    add_stack("design", 0, required[None] & unfinished)
    add_stack("design", 1, required[None] & ~unfinished)
    add_stack("design", 2, (target & ~required)[None] & remaining)
    spoil = loose >= rules.toe_m
    in_finished = spoil & required[None] & ~unfinished
    surface = native + loose
    obstacle_height = surface >= rules.ros_obstacle_height_m
    add_stack("loose soil", 3, in_finished)
    add_stack("loose soil", 2, spoil & obstacle_height & ~in_finished)
    add_stack(
        "loose soil",
        1,
        spoil & (loose >= rules.pass_height_m) & ~obstacle_height & ~in_finished,
    )
    add_stack(
        "loose soil",
        0,
        spoil & (loose < rules.pass_height_m) & ~obstacle_height & ~in_finished,
    )
    add_stack("excavation", 0, excavated)
    static = np.zeros((frames, h, w), dtype=bool)
    static[:] = case.final_zone[window]
    add_stack("Terra final dump zones and obstacles", 0, static)
    static[:] = case.obstacle[window]
    add_stack("Terra final dump zones and obstacles", 1, static)
    if case.final_spread is not None:
        static[:] = case.final_spread[window]
        add_stack("Terra final dump zones and obstacles", 2, static)

    # The converter's arrival masks hold over each workspace's two events, so unchanged cells merge into runs.
    held = np.zeros((frames, h, w), dtype=bool)
    spans = [
        (e["steps"]["cut"], (e["steps"]["loads"] or [e["steps"]["cut"]])[-1])
        for e in result["events"]
    ]
    for k, (first, last) in enumerate(spans):
        held[first : last + 1] = case.converter_blocked[k][window]
    add_stack("converter's own view", 0, held)
    held[:] = False
    for k, (_, last) in enumerate(spans):
        held[last] = case.converter_deposit[k][window]
    add_stack("converter's own view", 1, held)

    def to_crop(full_cells):
        rows, cols = np.divmod(np.asarray(full_cells, dtype=int), case.shape[1])
        inside = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
        return (rows[inside] - r0) * w + (cols[inside] - c0)

    shapes = []

    def shape(layer, category, kind, xy, first, last, **extra):
        shapes.append(
            dict(
                layer=LAYER[layer],
                category=category,
                kind=kind,
                xy=xy,
                **{"from": first, "to": last},
                **extra,
            )
        )

    reach = math.sqrt(
        max(
            float(case.dump_rule["reach_radius_m"]) ** 2
            - (ROS_APPROACH_M - float(case.dump_rule["reach_origin_xyz"][2])) ** 2,
            0.0,
        )
    )
    keep_away = float(case.dump_rule["min_base_radius_m"]) + ROS_PILE_EDGE_M
    for k, event in enumerate(result["events"]):
        cut_step, load_steps, drive_steps = (
            event["steps"]["cut"],
            event["steps"]["loads"],
            event["steps"]["drive"],
        )
        last_step = load_steps[-1] if load_steps else cut_step
        pose = case.poses[k]
        add_step("this workspace", 0, cut_step, case.completion[k][window])
        add_step(
            "this workspace",
            1,
            cut_step,
            (case.support[k] & ~case.completion[k])[window],
        )
        add_step(
            "this workspace",
            2,
            load_steps[0] if load_steps else cut_step,
            case.dump_centres[k][window],
            last_step,
        )
        add_step("station", 1, cut_step, to_crop(event["arrival"]["blocked_cells"]))
        add_step("station", 2, cut_step, to_crop(event["arrival"]["unverified_cells"]))
        polygon = world_polygon(case.footprint, pose).round(3).tolist()
        shape(
            "station",
            0,
            "polygon",
            polygon + polygon[:1],
            cut_step,
            last_step,
            label=f"workspace {k + 1}",
        )
        tip = pose[:2] + 2.0 * np.array([math.cos(pose[2]), math.sin(pose[2])])
        shape(
            "station",
            5,
            "line",
            [pose[:2].round(3).tolist(), tip.round(3).tolist()],
            cut_step,
            last_step,
        )
        for radius in (rules.band_min_m, rules.band_max_m):
            shape(
                "station",
                3,
                "line",
                _circle(pose[0], pose[1], radius),
                cut_step,
                cut_step,
            )
        shape(
            "station",
            4,
            "line",
            _circle(pose[0], pose[1], rules.entry_max_m),
            cut_step,
            cut_step,
            dash=True,
        )
        if load_steps:
            for radius in (reach, keep_away):
                shape(
                    "dump loads",
                    3,
                    "line",
                    _circle(pose[0], pose[1], radius),
                    load_steps[0],
                    last_step,
                    dash=True,
                )
        for load in event["dump"]["loads"]:
            step = load["step"]
            shape("dump loads", 0, "points", [[load["x_m"], load["y_m"]]], step, step)
            if load["toe_radius_m"] > 0:
                circle = _circle(load["x_m"], load["y_m"], load["toe_radius_m"])
                shape("dump loads", 1, "line", circle, step, step, dash=True)
            if step < frames - 1:
                shape(
                    "dump loads",
                    2,
                    "points",
                    [[load["x_m"], load["y_m"]]],
                    step + 1,
                    -1,
                )
        for step in drive_steps:
            drive_pose = np.asarray(result["timeline"][step - 1]["pose"])
            polygon = world_polygon(case.footprint, drive_pose).round(3).tolist()
            shape("station", 0, "polygon", polygon + polygon[:1], step, step)
        drawn_from = drive_steps[0] if drive_steps else cut_step
        if k > 0:
            previous = case.poses[k - 1]
            connector = [previous[:2].round(3).tolist(), pose[:2].round(3).tolist()]
            shape("driving", 0, "line", connector, drawn_from, -1, dash=True)
        route = case.routes.get((k, k + 1))
        if route is not None and route.get("path"):
            path = np.asarray(route["path"], dtype=float)
            shape("driving", 1, "line", path[:, :2].round(3).tolist(), drawn_from, -1)
            if event["route"] and event["route"].get("first_blocked_pose") is not None:
                blocked = path[event["route"]["first_blocked_pose"]]
                shape(
                    "driving",
                    2,
                    "points",
                    [blocked[:2].round(3).tolist()],
                    drawn_from,
                    cut_step,
                )
                polygon = world_polygon(case.footprint, blocked).round(3).tolist()
                shape(
                    "driving", 2, "polygon", polygon + polygon[:1], drawn_from, cut_step
                )
    for index, pose in enumerate(case.native_poses):
        shapes.append(
            dict(
                layer=LAYER["Terra native stations"],
                category=0,
                kind="points",
                xy=[[round(pose[0], 3), round(pose[1], 3)]],
                **{"from": 0, "to": -1},
            )
        )
    for k, source in enumerate(case.sources):
        if source >= 0:
            native_xy = case.native_poses[source][:2]
            if (
                math.hypot(
                    native_xy[0] - case.poses[k][0], native_xy[1] - case.poses[k][1]
                )
                > 0.01
            ):
                shapes.append(
                    dict(
                        layer=LAYER["Terra native stations"],
                        category=1,
                        kind="line",
                        xy=[
                            [round(native_xy[0], 3), round(native_xy[1], 3)],
                            case.poses[k][:2].round(3).tolist(),
                        ],
                        **{"from": 0, "to": -1},
                    )
                )
    for item in result["issues"]:
        if item["x_m"] is not None:
            step = min(max(int(item["step"]), 1), frames - 1)
            shapes.append(
                dict(
                    layer=LAYER["issues"],
                    category=SEVERITIES.index(item["severity"]),
                    kind="points",
                    xy=[[item["x_m"], item["y_m"]]],
                    **{"from": step, "to": step},
                )
            )

    steps = _step_facts(result, done, loose, design, required)
    overlay = dict(
        step_label="event",
        note=(
            "Replay of the converter output under the 1 Oct ground rules: no soil in the excavation, final zones "
            f"grow where their own piles spread, one deposit per workspace (its soil over its dump region, "
            f"{rules.repose_deg:g} deg slopes), loose soil under {rules.pass_height_m:g} m drivable. Map frame, metres. "
            "Events: each workspace's drive, arrival and cut, then its deposit."
        ),
        layers=[
            dict(
                name=name,
                group=group,
                on=on,
                categories=[dict(name=c, color=color) for c, color in categories],
            )
            for name, group, on, categories in LAYERS
        ],
        shapes=shapes,
        warnings=_warnings(result),
        summary=_summary(result),
    )
    layer_cells = (
        np.vstack(cells).astype(np.int32) if cells else np.zeros((0, 5), dtype=np.int32)
    )
    return dict(
        elev_steps=np.rint(
            np.clip(_block(native + loose, factor, np.mean) * 1000.0, -32000, 32000)
        ).astype(np.int16),
        desired=_block(design, factor, np.mean).astype(np.float32),
        footprint=_block(target, factor, np.any),
        bbox=np.array(
            [r0 // factor, r0 // factor + hc, c0 // factor, c0 // factor + wc],
            dtype=np.int32,
        ),
        origin_xy=origin + 0.5 * (factor - 1) * case.res,
        res=np.asarray(factor * case.res, dtype=np.float64),
        steps=np.asarray(json.dumps(_jsonable(steps))),
        overlay=np.asarray(json.dumps(_jsonable(overlay))),
        layer_cells=layer_cells,
    )


def _step_facts(result, native, loose, design, required):
    """One facts dict per timeline event: drive pose, arrival and cut, or the workspace's deposit."""
    case, rules = result["case"], result["rules"]
    area = case.res**2
    by_step = {}
    for item in result["issues"]:
        by_step.setdefault(int(item["step"]), []).append(
            f"[{item['severity']}] {item['text']}"
        )
    steps = []
    for step, entry in enumerate(result["timeline"], start=1):
        k = entry["workspace"] - 1
        event = result["events"][k]
        facts = dict(
            step=step,
            workspace=k + 1,
            event="deposit" if entry["phase"] == "load" else entry["phase"],
            operation="excavate" if event["kind"] == "excavate" else "collect spoil",
            source=(
                "added by the converter"
                if event["source_pair"] is None
                else f"Terra step {event['source_pair']}"
            ),
        )
        if entry["phase"] == "drive":
            route = event["route"]
            facts["drive"] = (
                f"saved Nav2 route, {entry['fraction']:.0%} along"
                if route and "length_m" in route
                else f"straight connector (not a validated route), {entry['fraction']:.0%} along"
            )
        elif entry["phase"] == "cut":
            x, y, yaw = event["pose"]
            facts["station"] = dict(x_m=x, y_m=y, yaw_deg=round(math.degrees(yaw), 2))
            arrival = event["arrival"]
            facts["arrival"] = (
                "blocked: " + ", ".join(arrival["hazards"])
                if arrival["blocked"]
                else (
                    "unverified spoil under the body"
                    if arrival["unverified"]
                    else "clear"
                )
            )
            facts["gap_body_to_excavation_m"] = arrival["gap_to_excavation_m"]
            if event["route"] and not event["route"].get("found", True):
                facts["route"] = (
                    "saved Nav2 check found no route: "
                    + event["route"]["nav2_error"][:80]
                )
            elif event["route"]:
                facts["route"] = (
                    f"saved Nav2 route {event['route']['length_m']:.2f} m, "
                    + (
                        "blocked"
                        if event["route"]["blocked"]
                        else "clear on this terrain"
                    )
                )
            elif k > 0:
                gap = float(np.hypot(*(case.poses[k][:2] - case.poses[k - 1][:2])))
                facts["route"] = f"not checked; straight connector {gap:.2f} m"
            facts.update(
                {
                    key: event["cut"][key]
                    for key in (
                        "completion_cells",
                        "new_required_m2",
                        "native_cut_m3",
                        "loose_lifted_m3",
                        "payload_m3",
                    )
                }
            )
            if event["cut"]["completion_radius_m"]:
                facts["completion_from_base_m"] = (
                    "{:.2f}-{:.2f} (allowed {:g}-{:g})".format(
                        *event["cut"]["completion_radius_m"],
                        rules.band_min_m,
                        rules.band_max_m,
                    )
                )
            facts["loads_to_dump"] = len(event["dump"]["loads"])
        else:
            load = event["dump"]["loads"][entry["load"] - 1]
            facts.update(
                load_m3=load["volume_m3"],
                point=f"({load['x_m']:.2f}, {load['y_m']:.2f})",
                choice={
                    "ros": "ROS admits this centre",
                    "modeled_only": "ROS admits no centre; replay uses one whose modeled pile stays clear",
                    "no_clear_point": "ROS admits no centre and no pile stays clear; worst case shown",
                }[load["choice"]],
                reachable=load["reachable"],
                pile_top_m=load["pile_top_m"],
                pile_toe_radius_m=load["toe_radius_m"],
                pile_gap_to_excavation_m=load["gap_to_excavation_m"],
                pit_spill_m3=load["pit_spill_m3"],
                spill=load["spill"],
                released_on_final_ground=load["final_release"],
            )
        remaining = np.maximum(native[step] - design, 0.0)
        left = required & (remaining > rules.finished_m)
        facts["required_left_m2"] = round(float(left.sum()) * area, 3)
        facts["required_left_m3"] = round(float(remaining[left].sum()) * area, 3)
        facts["loose_on_ground_m3"] = round(float(loose[step].sum()) * area, 3)
        facts["highest_pile_m"] = round(float(loose[step].max()), 3)
        if by_step.get(step):
            facts["issues"] = {f"{i + 1}": text for i, text in enumerate(by_step[step])}
        steps.append(facts)
    return steps


def _warnings(result):
    """Failures first, grouped by check, then unverified items, unchecked rules, efficiency and converter notes."""
    checks = validity(result)["checks"]
    warnings = []

    def items(tag, entries):
        if len(entries) <= 8:
            warnings.extend(
                dict(text=f"{tag} {e['text']}", steps=[max(int(e["step"]), 1)])
                for e in entries
            )
        else:
            steps = sorted({max(int(e["step"]), 1) for e in entries})
            warnings.append(
                dict(
                    text=f"{tag} {len(entries)} items, first: {entries[0]['text']}",
                    steps=steps[:60],
                )
            )

    for check in checks:
        if check["failures"]:
            items(f"[FAIL · {check['id']}]", check["failures"])
    for check in checks:
        if check["unverified"]:
            items(f"[UNVERIFIED · {check['id']}]", check["unverified"])
    for check in checks:
        if check["status"] == "not checked":
            warnings.append(
                dict(text=f"[NOT CHECKED · {check['id']}] {check['detail']}", steps=[])
            )
    for severity in ("efficiency", "converter"):
        entries = [i for i in result["issues"] if i["severity"] == severity]
        if entries:
            items(f"[{severity}]", entries)
    return warnings


def _summary(result):
    case, rules = result["case"], result["rules"]
    totals = result["totals"]
    converter = case.report
    counts = {
        severity: sum(i["severity"] == severity for i in result["issues"])
        for severity in SEVERITIES
    }
    verdict = (
        "no physical or runtime issue found under these rules"
        if not counts["physical"] and not counts["runtime"]
        else f"{counts['physical']} physical and {counts['runtime']} runtime issues"
    )
    checklist = validity(result)
    return dict(
        verdict=checklist["verdict"],
        checks={
            c["id"]: c["status"].upper()
            + (f", {len(c['failures'])} failing" if c["failures"] else "")
            + (f", {len(c['unverified'])} unverified" if c["unverified"] else "")
            + (f"; {c['detail']}" if c["detail"] else "")
            for c in checklist["checks"]
        },
        rule_text={c["id"]: c["rule"] for c in checklist["checks"]},
        result=verdict,
        converter=dict(
            complete_geometric_plan=bool(converter.get("complete_geometric_plan")),
            executable_plan_written=case.complete,
            coverage_pct=round(
                100.0 * float(converter.get("coverage_fraction", 0.0)), 2
            ),
            continuous_required_residual_m2=converter.get(
                "continuous_required_residual_m2"
            ),
            dump_sequence_errors=len(converter.get("dump_sequence_errors", [])),
        ),
        replay=totals,
        issues=counts,
        rules=dict(
            no_soil_in_excavation=True,
            repose_deg=rules.repose_deg,
            bulking=rules.bulking,
            bucket_m3=rules.bucket_m3,
            drivable_spoil_under_m=rules.pass_height_m,
            ros_obstacle_height_m=rules.ros_obstacle_height_m,
            completion_band_m=f"{rules.band_min_m:g}-{rules.band_max_m:g} from BASE",
            final_zone_spread_m=case.report.get("final_zone_spread_m"),
            dump_reach=f"{case.dump_rule['reach_radius_m']} m from CABIN, pile edge {case.dump_rule['min_base_radius_m']} m "
            "from BASE",
        ),
        not_modeled={f"{i + 1}": text for i, text in enumerate(NOT_MODELED)},
        inputs=dict(
            conversion=str(case.conversion), native=converter["inputs"]["input_dir"]
        ),
    )


def write_history(path, results, display_res_m=None):
    """One webviz history file, one run per replayed plan (tag = case name)."""
    arrays = {}
    for result in results:
        tag = result["case"].tag
        if "__" in tag:
            raise ValueError(f"run tag {tag!r} must not contain '__'")
        for key, value in history_arrays(result, display_res_m).items():
            arrays[f"{tag}__{key}"] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
