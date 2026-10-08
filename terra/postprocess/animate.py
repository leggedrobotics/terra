"""Terra-style animations of a Moleworks excavation plan: the native Terra plan, its converted machine plan, or both.

``terra-postprocess animate EVALUATION --out FILE`` draws them the way Terra's own renderer (terra/viz/game) does: a top-down
tile map in Terra's palette and the Terra excavator, a body with a cabin coloured by whether it is loaded. Every panel
is in the plan frame as that renderer and ``extract_map.py --render_plan_gif`` draw it: tile rows (plan x) run down,
tile columns (plan y) run right. A compass in each panel shows the map axes.

* native: the Terra plan on its tile grid. The agent drives between the stations of the dig/dump pairs; at each
  waypoint the tiles it works turn red (Terra's dig/dump cone colour), then dug or dumped.
* machine: the converted plan on the replay's 0.1 m grid (tmm_replay): the design, dug ground, loose soil shaded by
  height, the machine footprint driving along the saved Nav2 routes, the arm to the dig area or dump point.
* side-by-side: both, in step with the Terra plan. A station the converter added, or the next station of a Terra pair
  the converter split, holds the Terra panel; a Terra pair the converter did not keep holds the machine panel.

The plan keeps where each waypoint changed Terra's action map, not by how much: the tile state here lowers a
waypoint's ``dug_mask`` tiles and raises its ``dump_mask`` tiles by one, so dumped tiles shade by dump count.
"""

import dataclasses
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import replay as tmm_replay

# Terra's palette (terra/viz/game/settings.py COLORS) and agent colours (terra/viz/game/agent.py).
NEUTRAL = (207, 207, 207)  # "#cfcfcf"
TO_DIG = (136, 0, 255)  # "#8800ff"
DUG = (38, 189, 108)  # "#26bd6c"
DUG_EDGE = (0, 107, 46)  # "#006b2e": dug to the design within two tiles of its edge
DUMPED = (0, 43, 91)  # "#002B5B", the dark end of Terra's dumped-soil ramp
DUMPED_LIGHT = (173, 216, 230)  # its light end (terra/viz/game/world.py)
OBSTACLE = (0, 0, 0)
NON_DUMPABLE = (171, 159, 149)  # "#ab9f95"
FINAL_DUMP = (243, 230, 200)  # "#F3E6C8"
WORK = (255, 107, 107)  # "#ff6b6b", the dig/dump cone
WORK_PALE = (255, 179, 179)  # "#ffb3b3"
BACKGROUND = (240, 240, 240)  # "#F0F0F0", Terra's frame fill
BODY = (0, 43, 91)
CABIN = {True: (165, 115, 75), False: (234, 84, 85)}  # loaded, not loaded
SHOVEL = {True: (139, 69, 19), False: (192, 192, 192)}  # lowered, lifted
INK = (28, 28, 28)
MUTED = (90, 90, 90)
ROUTE = (45, 45, 45)

TERRA_BODY_M = (6.08, 3.5)  # terra.config.ExcavatorDims: long and short side
CABIN_TILES = np.array(
    [[3.0, 0.0], [-1.5, -1.5], [-1.5, 1.5]]
)  # Terra's cabin triangle at its 3 px tiles
SHOVEL_M = (0.7, 1.3)  # drawn bucket: along the arm, across it (stock 1.3 m shovel)
EDGE_TILES = 2  # Terra draws dug design tiles this close to the design's edge darker
PILE_SCALE_M = 1.0  # loose soil shades from light at 0 m to dark at this height (the ROS /map obstacle height)
REACH_M = 5.0  # arm length drawn when a waypoint changed no tile

DRIVE_M_PER_FRAME = 0.8
DRIVE_FRAMES = (4, 20)  # fewest and most frames of one drive
WORK_FRAMES = 5  # the work area shows red while the cabin turns to it
TURN_FRAMES = 3  # of those, the frames the cabin turns in
DONE_FRAMES = 3  # the worked tiles changed
START_FRAMES, END_FRAMES = 12, 40
MODES = ("native", "machine", "side-by-side")
PANEL_PX = {
    "native": 720,
    "machine": 720,
    "side-by-side": 700,
}  # longer side of a panel
MARGIN, GAP, TITLE_H, CAPTION_H, LEGEND_ROW_H = 14, 14, 30, 44, 22
MODE_TITLES = {
    "native": "Terra plan",
    "machine": "converted machine plan",
    "side-by-side": "Terra plan (left) and converted machine plan (right)",
}


def wrap(angle):
    return math.remainder(angle, 2.0 * math.pi)


def lerp_angle(a, b, t):
    return a + t * wrap(b - a)


def terra_body_tiles(tile_m):
    """Terra's excavator in whole tiles, long side first (terra/viz/game/game.py get_agent_dims)."""

    def odd(n):
        return n if n % 2 else n + 1

    return odd(round(TERRA_BODY_M[0] / tile_m)), odd(round(TERRA_BODY_M[1] / tile_m))


def edge_mask(design, cells):
    """Design cells within ``cells`` of its edge, as Terra's _get_foundation_edge_mask (3x3 erosions)."""
    if not design.any():
        return design.copy()
    return design & ~ndimage.binary_erosion(
        design, np.ones((3, 3), bool), iterations=max(1, cells), border_value=0
    )


def ramp(t):
    """Terra's dumped-soil colours for fractions ``t`` in [0, 1]: light blue to #002B5B."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)[..., None]
    return np.rint(
        np.asarray(DUMPED_LIGHT) + t * (np.asarray(DUMPED) - np.asarray(DUMPED_LIGHT))
    ).astype(np.uint8)


def drive_frames(distance_m, turn_rad=0.0):
    if distance_m < 1e-6 and abs(turn_rad) < 1e-6:
        return 0
    return int(np.clip(math.ceil(distance_m / DRIVE_M_PER_FRAME), *DRIVE_FRAMES))


@dataclasses.dataclass(frozen=True)
class Alignment:
    """The plan-to-map transform of a schema-v2 plan (terra_planner docs/terra_frames.md)."""

    tile_m: float
    origin: tuple
    yaw: float

    @classmethod
    def of(cls, plan):
        a = plan["alignment"]
        return cls(
            float(a["meters_per_tile"]),
            tuple(map(float, a["origin_map_xy_m"])),
            float(a["yaw_map_from_plan_rad"]),
        )

    def to_map(self, xy):
        xy = np.asarray(xy, dtype=float)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack(
            [
                self.origin[0] + c * xy[..., 0] - s * xy[..., 1],
                self.origin[1] + s * xy[..., 0] + c * xy[..., 1],
            ],
            -1,
        )

    def to_plan(self, xy):
        xy = np.asarray(xy, dtype=float) - np.asarray(self.origin)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack(
            [c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]], -1
        )

    def pose_to_plan(self, pose):
        x, y = self.to_plan(np.asarray(pose[:2], dtype=float))
        return (float(x), float(y), wrap(float(pose[2]) - self.yaw))


@dataclasses.dataclass
class NativePlan:
    """A schema-v2 Terra plan and its tile arrays. Tile [row, col] spans plan x in [row, row + 1) tiles and plan y in
    [col, col + 1) tiles; ``pos_base`` is [row, col] in the same tile-corner units (terra_planner docs/terra_frames.md).
    """

    plan: dict
    images: np.ndarray  # -1 ground to dig, 1 final dump area, 0 neutral
    occupancy: np.ndarray
    dumpability: np.ndarray

    def __post_init__(self):
        waypoints = self.plan["waypoints"]
        if len(waypoints) % 2:
            raise ValueError("a schema-v2 plan holds dig/dump waypoint pairs")
        self.alignment = Alignment.of(self.plan)
        self.pairs = [
            (waypoints[i], waypoints[i + 1]) for i in range(0, len(waypoints), 2)
        ]

    @classmethod
    def load(cls, source):
        source = Path(source)
        plan = json.loads((source / "terra_plan.json").read_text())
        with np.load(source / "arrays.npz") as arrays:
            return cls(
                plan,
                np.asarray(arrays["images"]),
                np.asarray(arrays["occupancy"], dtype=bool),
                np.asarray(arrays["dumpability"], dtype=bool),
            )

    @property
    def tile_m(self):
        return self.alignment.tile_m

    def station(self, p):
        """BASE of pair ``p`` in the plan frame: x, y (m) and heading."""
        state = self.pairs[p][0]["agent_state"]
        return (
            state["pos_base"][0] * self.tile_m,
            state["pos_base"][1] * self.tile_m,
            float(state["angle_base_rad"]),
        )

    def steps(self, p):
        return f"{self.pairs[p][0]['step']}-{self.pairs[p][1]['step']}"

    def work(self, waypoint):
        """The tiles a waypoint changed and their centroid in the plan frame (None when it changed none)."""
        mask = np.asarray(waypoint["terrain_modification_mask"], dtype=bool)
        if not mask.any():
            return mask, None
        rows, cols = np.nonzero(mask)
        return mask, (
            (rows.mean() + 0.5) * self.tile_m,
            (cols.mean() + 0.5) * self.tile_m,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Shot:
    """What one panel shows in one frame. Poses and points are in the plan frame (m)."""

    terrain: tuple  # the panel's terrain key, compared by identity
    pose: tuple  # BASE x, y, heading
    cabin: float  # cabin yaw
    loaded: bool
    tool: tuple | None = None  # ((x, y), lowered)
    route: object = None  # polyline of the current drive
    caption: tuple = ("", "")


def _tool(pose, cabin, reach, lowered):
    return (
        (pose[0] + reach * math.cos(cabin), pose[1] + reach * math.sin(cabin)),
        lowered,
    )


class NativeTimeline:
    """Shots of the Terra plan: drives between stations, then each waypoint's work."""

    def __init__(self, native):
        self.native = native
        self.action = np.zeros(native.images.shape, dtype=np.int16)
        count = len(native.pairs)
        if count:
            pose = native.station(0)
        else:
            pose = (
                native.images.shape[0] * native.tile_m / 2,
                native.images.shape[1] * native.tile_m / 2,
                0.0,
            )
        steps = (
            f"Terra steps {native.pairs[0][0]['step']}-{native.pairs[-1][1]['step']}"
            if count
            else "no waypoints"
        )
        self.summary = f"{count} dig/dump pairs, {steps}"
        self.last = Shot(
            (self.action, None),
            pose,
            pose[2],
            False,
            caption=("Terra plan", self.summary),
        )

    def _shot(self, **changes):
        self.last = dataclasses.replace(self.last, **changes)
        return self.last

    def _title(self, p, what):
        return f"Terra plan · pair {p + 1} of {len(self.native.pairs)} · {what}"

    def drive_frames(self, p):
        start, end = self.last.pose, self.native.station(p)
        return drive_frames(
            math.hypot(end[0] - start[0], end[1] - start[1]), wrap(end[2] - start[2])
        )

    def drive(self, p, count):
        start, end = self.last.pose, self.native.station(p)
        relative = wrap(self.last.cabin - start[2])
        caption = (
            self._title(p, "drive"),
            f"to the station of Terra steps {self.native.steps(p)}",
        )
        shots = []
        for i in range(1, count + 1):
            t = i / count
            heading = lerp_angle(start[2], end[2], t)
            pose = (
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
                heading,
            )
            shots.append(
                self._shot(
                    terrain=(self.action, None),
                    pose=pose,
                    cabin=heading + relative,
                    tool=None,
                    caption=caption,
                )
            )
        self.last = dataclasses.replace(self.last, pose=end, cabin=end[2] + relative)
        return shots

    def waypoint(self, p, index):
        """The dig (``index`` 0) or dump (1) waypoint of pair ``p``: its tiles show red, then change."""
        waypoint = self.native.pairs[p][index]
        mask, target = self.native.work(waypoint)
        pose, start = self.last.pose, self.last.cabin
        if target is None:
            aim, reach = (
                pose[2] + float(waypoint["agent_state"]["angle_cabin_rad"]),
                REACH_M,
            )
        else:
            aim = math.atan2(target[1] - pose[1], target[0] - pose[0])
            reach = math.hypot(target[0] - pose[0], target[1] - pose[1])
        verb = (
            "dump"
            if index
            else (
                "dig" if waypoint["workspace_type"] == "excavate" else "collect spoil"
            )
        )
        caption = (
            self._title(p, verb),
            f"Terra step {waypoint['step']} · {int(mask.sum())} tiles",
        )
        digging = index == 0
        shots = []
        for i in range(WORK_FRAMES):
            cabin = lerp_angle(start, aim, min(1.0, (i + 1) / TURN_FRAMES))
            tool = _tool(pose, cabin, reach, digging)
            shots.append(
                self._shot(
                    terrain=(self.action, mask), cabin=cabin, tool=tool, caption=caption
                )
            )
        self.action = (
            self.action
            - np.asarray(waypoint["dug_mask"], dtype=np.int16)
            + np.asarray(waypoint["dump_mask"], dtype=np.int16)
        )
        loaded = bool(waypoint["agent_state"]["loaded"])
        done = self._shot(
            terrain=(self.action, None),
            cabin=aim,
            loaded=loaded,
            tool=_tool(pose, aim, reach, False),
        )
        return shots + [done] * DONE_FRAMES

    def pair(self, p, drive=None):
        count = self.drive_frames(p) if drive is None else drive
        return self.drive(p, count) + self.waypoint(p, 0) + self.waypoint(p, 1)

    def hold(self, count, caption=None):
        shot = self.last if caption is None else self._shot(caption=caption)
        return [shot] * count


class MachineTimeline:
    """Shots of the replayed machine plan: drives along the saved routes, each workspace's cut, then its loads."""

    def __init__(self, result, native):
        self.result, self.native = result, native
        self.case = case = result["case"]
        self.align = native.alignment
        self.frame = 0
        self.count = len(case.sources)
        loads = sum(len(event["dump"]["loads"]) for event in result["events"])
        pose = self.align.pose_to_plan(case.poses[0]) if self.count else (0.0, 0.0, 0.0)
        self.summary = f"{self.count} workspaces, {loads} bucket loads"
        self.last = Shot(
            (0, None, None),
            pose,
            pose[2],
            False,
            caption=("Machine plan", self.summary),
        )

    def _shot(self, **changes):
        self.last = dataclasses.replace(self.last, **changes)
        return self.last

    def _title(self, k, what):
        return f"Machine plan · workspace {k + 1} of {self.count} · {what}"

    def source(self, k):
        s = self.case.sources[k]
        return (
            "added by the converter" if s < 0 else f"Terra steps {self.native.steps(s)}"
        )

    def route(self, k):
        """The drive to workspace ``k`` (k >= 1) in the plan frame as (x, y, heading) rows, and its description."""
        route = self.case.routes.get((k, k + 1))
        if route is not None and route.get("path"):
            path = np.asarray(route["path"], dtype=float)[:, :3]
            text = f"Nav2 route {float(route.get('path_length_m', 0.0)):.1f} m"
        else:
            path = np.asarray([self.case.poses[k - 1], self.case.poses[k]], dtype=float)
            gap = float(np.hypot(*(path[1, :2] - path[0, :2])))
            text = f"straight connector {gap:.1f} m, no saved route"
        xy = self.align.to_plan(path[:, :2])
        headings = np.unwrap(path[:, 2]) - self.align.yaw
        return np.column_stack([xy, headings]), text

    def drive_frames(self, k):
        if k == 0:
            return 0
        path, _ = self.route(k)
        return drive_frames(
            float(np.hypot(*np.diff(path[:, :2], axis=0).T).sum()),
            path[-1, 2] - path[0, 2],
        )

    def drive(self, k, count):
        if k == 0 or count == 0:
            return []
        path, text = self.route(k)
        length = np.concatenate(
            [[0.0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))]
        )
        relative = wrap(self.last.cabin - self.last.pose[2])
        caption = (self._title(k, "drive"), f"{self.source(k)} · {text}")
        shots = []
        for i in range(1, count + 1):
            s = i / count * length[-1]
            j = int(np.clip(np.searchsorted(length, s, side="right"), 1, len(path) - 1))
            span = length[j] - length[j - 1]
            t = 0.0 if span <= 0 else (s - length[j - 1]) / span
            x, y, heading = path[j - 1] + t * (path[j] - path[j - 1])
            if i == count:
                x, y, heading = self.align.pose_to_plan(self.case.poses[k])
            pose = (float(x), float(y), float(heading))
            shots.append(
                self._shot(
                    terrain=(self.frame, None, None),
                    pose=pose,
                    cabin=pose[2] + relative,
                    tool=None,
                    route=path[:, :2],
                    caption=caption,
                )
            )
        return shots

    def cut(self, k):
        case, event = self.case, self.result["events"][k]
        pose = self.align.pose_to_plan(case.poses[k])
        rows, cols = np.nonzero(case.completion[k])
        if rows.size:
            target = self.align.to_plan(
                np.mean(np.column_stack(case.xy(rows, cols)), axis=0)
            )
            aim = math.atan2(target[1] - pose[1], target[0] - pose[0])
            reach = math.hypot(target[0] - pose[0], target[1] - pose[1])
        else:
            aim, reach = pose[2], REACH_M
        collect = case.kinds[k] != "excavate"
        cut = event["cut"]
        what = (
            f"lifts {cut['loose_lifted_m3']:.2f} m³ of spoil"
            if collect
            else f"cuts {cut['native_cut_m3']:.2f} m³"
        )
        caption = (
            self._title(k, "collect spoil" if collect else "cut"),
            f"{self.source(k)} · {what}",
        )
        step, start = event["steps"]["cut"], self.last.cabin
        shots = []
        for i in range(WORK_FRAMES):
            cabin = lerp_angle(start, aim, min(1.0, (i + 1) / TURN_FRAMES))
            shots.append(
                self._shot(
                    terrain=(step - 1, case.completion[k], None),
                    pose=pose,
                    cabin=cabin,
                    tool=_tool(pose, cabin, reach, True),
                    route=None,
                    caption=caption,
                )
            )
        self.frame = step
        loaded = cut["payload_m3"] > 1e-9
        done = self._shot(
            terrain=(step, None, None),
            cabin=aim,
            loaded=loaded,
            tool=_tool(pose, aim, reach, False),
        )
        return shots + [done] * DONE_FRAMES

    def loads(self, k):
        case, event = self.case, self.result["events"][k]
        records, steps = event["dump"]["loads"], event["steps"]["loads"]
        if not records:
            return []
        pose, pale = self.last.pose, case.dump_centres[k]
        points = [self.align.to_plan(np.array([r["x_m"], r["y_m"]])) for r in records]
        aims = [math.atan2(p[1] - pose[1], p[0] - pose[0]) for p in points]
        reaches = [math.hypot(p[0] - pose[0], p[1] - pose[1]) for p in points]
        shots, start = [], self.last.cabin
        for i in range(TURN_FRAMES):
            cabin = lerp_angle(start, aims[0], (i + 1) / TURN_FRAMES)
            caption = (self._title(k, "dump the workspace's soil"), f"{self.source(k)}")
            shots.append(
                self._shot(
                    terrain=(steps[0] - 1, None, pale),
                    cabin=cabin,
                    tool=_tool(pose, cabin, reaches[0], False),
                    caption=caption,
                )
            )
        for j, (record, step) in enumerate(zip(records, steps)):
            caption = (
                self._title(k, f"dump load {j + 1} of {len(records)}"),
                f"{self.source(k)} · {record['volume_m3']:.2f} m³, pile top {record['pile_top_m']:.2f} m",
            )
            tool = _tool(pose, aims[j], reaches[j], False)
            shots.append(
                self._shot(
                    terrain=(step - 1, None, pale),
                    cabin=aims[j],
                    tool=tool,
                    loaded=True,
                    caption=caption,
                )
            )
            shots.append(
                self._shot(terrain=(step, None, pale), loaded=j < len(records) - 1)
            )
            self.frame = step
        return shots

    def workspace(self, k):
        return self.drive(k, self.drive_frames(k)) + self.cut(k) + self.loads(k)

    def hold(self, count, caption=None):
        shot = self.last if caption is None else self._shot(caption=caption)
        return [shot] * count

    def not_kept(self, p):
        """Why the converter kept no workspace of Terra pair ``p``."""
        rows = [
            row
            for row in self.case.report.get("per_pair", [])
            if row.get("pair_index") == p
        ]
        omitted = next((row for row in rows if row.get("omitted")), None)
        if omitted is not None:
            return str(omitted.get("reason") or "omitted by the converter").replace(
                "_", " "
            )
        return "rejected by the converter"


def workspace_sources(result):
    """Terra pairs of every machine workspace: the accepted, not omitted per_pair rows in order, or the retained pair
    indices when those rows do not line up with the workspaces."""
    case = result["case"]
    rows = [
        row
        for row in case.report.get("per_pair", [])
        if row.get("accepted") and not row.get("omitted")
    ]
    if len(rows) == len(case.sources) and all(
        row.get("pair_index") == source for row, source in zip(rows, case.sources)
    ):
        return [
            [int(p) for p in (row.get("source_pair_indices") or []) if int(p) >= 0]
            for row in rows
        ]
    return [[int(s)] if s >= 0 else [] for s in case.sources]


def _pad(a, b):
    n = max(len(a), len(b))
    return a + a[-1:] * (n - len(a)), b + b[-1:] * (n - len(b))


def native_shots(native):
    timeline = NativeTimeline(native)
    shots = timeline.hold(START_FRAMES)
    for p in range(len(native.pairs)):
        shots += timeline.pair(p)
    return [
        (shot,)
        for shot in shots
        + timeline.hold(END_FRAMES, ("Terra plan · done", timeline.summary))
    ]


def machine_shots(result, native):
    timeline = MachineTimeline(result, native)
    shots = timeline.hold(START_FRAMES)
    for k in range(timeline.count):
        shots += timeline.workspace(k)
    verdict = tmm_replay.validity(result)["verdict"]
    ending = ("Machine plan · done", f"{timeline.summary} · replay: {verdict}")
    return [(shot,) for shot in shots + timeline.hold(END_FRAMES, ending)]


def side_by_side_shots(native, result):
    """Frames of both panels in step with the Terra plan: a workspace plays with the first of its Terra pairs."""
    left, right = NativeTimeline(native), MachineTimeline(result, native)
    sources = workspace_sources(result)
    kept = {p for pairs in sources for p in pairs}
    shown = set()
    a, b = left.hold(START_FRAMES), right.hold(START_FRAMES)

    def terra_only(p):
        shots = left.pair(p)
        why = right.not_kept(p)
        return shots, right.hold(
            len(shots),
            ("Machine plan · holds", f"Terra steps {native.steps(p)} not kept: {why}"),
        )

    for k, pairs in enumerate(sources):
        new = [p for p in pairs if p not in shown]
        for p in range(len(native.pairs)):
            if new and p < new[0] and p not in kept and p not in shown:
                shots, held = terra_only(p)
                a, b = a + shots, b + held
                shown.add(p)
        if new:
            count = max(left.drive_frames(new[0]), right.drive_frames(k))
            shots = (
                left.drive(new[0], count)
                + left.waypoint(new[0], 0)
                + left.waypoint(new[0], 1)
            )
            for p in new[1:]:
                shots += left.pair(p)
            machine = right.drive(k, count) + right.cut(k) + right.loads(k)
            shots, machine = _pad(shots, machine)
            shown.update(new)
        else:
            machine = right.workspace(k)
            why = (
                "the converter added this station"
                if not pairs
                else f"Terra steps {native.steps(pairs[0])}: another station"
            )
            shots = left.hold(len(machine), ("Terra plan · holds", why))
        a, b = a + shots, b + machine
    for p in range(len(native.pairs)):
        if p not in shown:
            shots, held = terra_only(p)
            a, b = a + shots, b + held
    verdict = tmm_replay.validity(result)["verdict"]
    a += left.hold(END_FRAMES, ("Terra plan · done", left.summary))
    b += right.hold(
        END_FRAMES, ("Machine plan · done", f"{right.summary} · replay: {verdict}")
    )
    return list(zip(a, b))


@dataclasses.dataclass(frozen=True)
class View:
    """The drawn window: whole tiles [r0, r1) x [c0, c1) of the plan frame, ``tile_px`` pixels each."""

    r0: int
    r1: int
    c0: int
    c1: int
    tile_px: int
    tile_m: float

    @property
    def width(self):
        return (self.c1 - self.c0) * self.tile_px

    @property
    def height(self):
        return (self.r1 - self.r0) * self.tile_px

    @property
    def px_per_m(self):
        return self.tile_px / self.tile_m

    def px(self, xy):
        """Pixel (x right, y down) of plan-frame points."""
        xy = np.asarray(xy, dtype=float)
        return np.stack(
            [
                xy[..., 1] * self.px_per_m - self.c0 * self.tile_px,
                xy[..., 0] * self.px_per_m - self.r0 * self.tile_px,
            ],
            -1,
        )

    def pixel_plan(self):
        """Plan-frame point of every pixel centre, shape (height, width, 2)."""
        ys, xs = np.mgrid[0 : self.height, 0 : self.width] + 0.5
        return np.stack(
            [
                ys / self.px_per_m + self.r0 * self.tile_m,
                xs / self.px_per_m + self.c0 * self.tile_m,
            ],
            -1,
        )

    def crop(self, tiles):
        """A [row, col] tile layer cut to the window and scaled to pixels."""
        return np.repeat(
            np.repeat(tiles[self.r0 : self.r1, self.c0 : self.c1], self.tile_px, 0),
            self.tile_px,
            1,
        )


def _box(rows, cols):
    return (
        (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)
        if rows.size
        else None
    )


def native_box(native):
    mask = (native.images != 0) | native.occupancy | ~native.dumpability
    for dig, dump in native.pairs:
        for waypoint in (dig, dump):
            mask |= np.asarray(waypoint["terrain_modification_mask"], dtype=bool)
            mask |= np.asarray(waypoint["traversability_mask"]) == -1
    box = _box(*np.nonzero(mask))
    return None if box is None else (box[0] - 2, box[1] + 2, box[2] - 2, box[3] + 2)


def machine_box(result, alignment):
    case = result["case"]
    keep = (
        case.target
        | case.final_zone
        | case.dump_centres.any(axis=0)
        | case.completion.any(axis=0)
    )
    keep |= result["loose_frames"][-1] >= result["rules"].toe_m
    rows, cols = np.nonzero(keep)
    points = [np.column_stack(case.xy(rows, cols))] if rows.size else []
    reach = float(np.hypot(*case.footprint.T).max())
    paths = [np.asarray(case.poses, dtype=float)[:, :2]]
    paths += [
        np.asarray(route["path"], dtype=float)[:, :2]
        for route in case.routes.values()
        if route.get("path")
    ]
    for path in paths:
        points += [
            path + offset
            for offset in (
                (reach, reach),
                (-reach, reach),
                (reach, -reach),
                (-reach, -reach),
            )
        ]
    if not points:
        return None
    tiles = alignment.to_plan(np.vstack(points)) / alignment.tile_m
    return (
        int(math.floor(tiles[:, 0].min())) - 1,
        int(math.ceil(tiles[:, 0].max())) + 1,
        int(math.floor(tiles[:, 1].min())) - 1,
        int(math.ceil(tiles[:, 1].max())) + 1,
    )


def choose_view(mode, native, result):
    boxes = []
    if mode != "machine":
        boxes.append(native_box(native))
    if mode != "native":
        boxes.append(machine_box(result, native.alignment))
    boxes = [box for box in boxes if box is not None]
    rows, cols = native.images.shape
    if not boxes:
        boxes = [(0, rows, 0, cols)]
    r0, r1 = max(min(b[0] for b in boxes), 0), min(max(b[1] for b in boxes), rows)
    c0, c1 = max(min(b[2] for b in boxes), 0), min(max(b[3] for b in boxes), cols)
    tile_px = max(3, PANEL_PX[mode] // max(r1 - r0, c1 - c0))
    return View(r0, r1, c0, c1, tile_px, native.tile_m)


def base_tiles(native, design):
    """Terra's static tile colours: neutral, then the target (with or without ground to dig), obstacles, and
    non-dumpable ground over them, in the order terra/viz/game/world.py applies them."""
    rgb = np.empty((*native.images.shape, 3), dtype=np.uint8)
    rgb[:] = NEUTRAL
    if design:
        rgb[native.images == -1] = TO_DIG
    rgb[native.images == 1] = FINAL_DUMP
    rgb[native.occupancy] = OBSTACLE
    rgb[~native.dumpability] = NON_DUMPABLE
    return rgb


def _polygon(view, pose, points):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    points = np.asarray(points, dtype=float)
    plan = np.column_stack(
        [
            pose[0] + c * points[:, 0] - s * points[:, 1],
            pose[1] + s * points[:, 0] + c * points[:, 1],
        ]
    )
    return [tuple(p) for p in view.px(plan)]


class Panel:
    """Draws shots on terrain images; subclasses provide the terrain and the body outline (BASE frame, m)."""

    body = None

    def __init__(self, view):
        self.view = view
        self._key, self._terrain = None, None
        self.cabin = CABIN_TILES * view.tile_m

    def terrain(self, key):
        raise NotImplementedError

    def render(self, shot):
        key = tuple(
            id(item) if isinstance(item, np.ndarray) else item for item in shot.terrain
        )
        if key != self._key:
            self._key, self._terrain = key, self.terrain(shot.terrain)
        image = Image.fromarray(self._terrain)
        draw = ImageDraw.Draw(image)
        view = self.view
        if shot.route is not None:
            draw.line(
                [tuple(p) for p in view.px(shot.route)],
                fill=ROUTE,
                width=3,
                joint="curve",
            )
        pose = shot.pose
        draw.polygon(_polygon(view, pose, self.body), fill=BODY)
        if shot.tool is not None:
            (x, y), lowered = shot.tool
            draw.line(
                [tuple(view.px((pose[0], pose[1]))), tuple(view.px((x, y)))],
                fill=BODY,
                width=max(2, round(0.3 * view.px_per_m)),
            )
            bucket = (
                np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
                * np.asarray(SHOVEL_M)
                / 2
            )
            draw.polygon(
                _polygon(view, (x, y, shot.cabin), bucket),
                fill=SHOVEL[lowered],
                outline=BODY,
            )
        draw.polygon(
            _polygon(view, (pose[0], pose[1], shot.cabin), self.cabin),
            fill=CABIN[shot.loaded],
        )
        return image


class NativePanel(Panel):
    def __init__(self, native, view):
        super().__init__(view)
        self.native = native
        self.base = base_tiles(native, design=True)
        self.edge = edge_mask(native.images == -1, EDGE_TILES)
        long, short = (n * native.tile_m for n in terra_body_tiles(native.tile_m))
        self.body = [
            (-long / 2, -short / 2),
            (long / 2, -short / 2),
            (long / 2, short / 2),
            (-long / 2, short / 2),
        ]

    def terrain(self, key):
        action, highlight = key
        rgb = self.base.copy()
        dumped = action > 0
        if dumped.any():
            rgb[dumped] = ramp(action[dumped] / action.max())
        dug = action < 0
        rgb[dug] = DUG
        rgb[dug & self.edge & (action <= self.native.images)] = DUG_EDGE
        if highlight is not None:
            rgb[highlight] = WORK
        return self.view.crop(rgb)


class MachinePanel(Panel):
    def __init__(self, result, native, view):
        super().__init__(view)
        self.result = result
        case = self.case = result["case"]
        self.rules = result["rules"]
        self.body = case.footprint
        grid = native.alignment.to_map(view.pixel_plan())
        cols = np.rint((grid[..., 0] - case.origin[0]) / case.res).astype(int)
        rows = np.rint((grid[..., 1] - case.origin[1]) / case.res).astype(int)
        inside = (
            (rows >= 0) & (cols >= 0) & (rows < case.shape[0]) & (cols < case.shape[1])
        )
        self.rows, self.cols = np.clip(rows, 0, case.shape[0] - 1), np.clip(
            cols, 0, case.shape[1] - 1
        )
        self.inside = inside & case.known[self.rows, self.cols]
        self.base = view.crop(base_tiles(native, design=False))
        self.target = case.target[self.rows, self.cols] & self.inside
        self.base[self.target] = TO_DIG
        self.base[~self.inside] = BACKGROUND
        cells = max(1, round(EDGE_TILES * native.tile_m / case.res))
        self.edge = edge_mask(case.target, cells)[self.rows, self.cols] & self.inside
        self.design = result["design"][self.rows, self.cols]

    def terrain(self, key):
        index, highlight, pale = key
        at = (self.rows, self.cols)
        native = self.result["native_frames"][index][at]
        loose = self.result["loose_frames"][index][at]
        excavated = self.result["excavated_frames"][index][at] & self.inside
        rgb = self.base.copy()
        rgb[excavated] = DUG
        rgb[excavated & self.edge & (native <= self.design + self.rules.finished_m)] = (
            DUG_EDGE
        )
        soil = (loose >= self.rules.toe_m) & self.inside
        if pale is not None:
            rgb[pale[at] & ~soil & self.inside] = WORK_PALE
        rgb[soil] = ramp(loose[soil] / PILE_SCALE_M)
        if highlight is not None:
            rgb[highlight[at] & self.inside] = WORK
        return rgb


_FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        try:
            _FONTS[key] = ImageFont.truetype(
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
            )
        except OSError:
            _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


def legend_items(mode, native, view):
    items = [
        ("to dig" if mode == "native" else "design, to dig", "fill", TO_DIG),
        ("dug", "fill", DUG),
    ]
    items.append(("edge dug to design", "fill", DUG_EDGE))
    if mode != "machine":
        items.append(("dumped soil (more: darker)", "ramp", None))
    if mode != "native":
        items.append((f"loose soil 0-{PILE_SCALE_M:g} m", "ramp", None))
    items += [("final dump area", "fill", FINAL_DUMP), ("work area", "fill", WORK)]
    if mode != "native":
        items += [("dump centres", "fill", WORK_PALE), ("Nav2 route", "line", ROUTE)]
    window = (slice(view.r0, view.r1), slice(view.c0, view.c1))
    if (~native.dumpability[window]).any():
        items.append(("non-dumpable", "fill", NON_DUMPABLE))
    if native.occupancy[window].any():
        items.append(("obstacle", "fill", OBSTACLE))
    return items


class Animation:
    """The frames of one animation: a title bar, one or two captioned panels and a legend."""

    def __init__(self, mode, native, result=None, title=""):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "native" and result is None:
            raise ValueError(f"mode {mode} needs the replayed conversion")
        self.mode = mode
        self.view = view = choose_view(mode, native, result)
        if mode == "native":
            self.panels, self.shots = [NativePanel(native, view)], native_shots(native)
        elif mode == "machine":
            self.panels, self.shots = [
                MachinePanel(result, native, view)
            ], machine_shots(result, native)
        else:
            self.panels = [
                NativePanel(native, view),
                MachinePanel(result, native, view),
            ]
            self.shots = side_by_side_shots(native, result)
        # Each panel's slot fits the panel and its widest caption; the frame fits the slots and the title.
        title = f"{title} · {MODE_TITLES[mode]}" if title else MODE_TITLES[mode]
        slots = []
        for side in range(len(self.panels)):
            captions = {shots[side].caption for shots in self.shots}
            widest = max(
                max(font(15, True).getlength(top), font(13).getlength(bottom))
                for top, bottom in captions
            )
            slots.append(max(view.width, math.ceil(widest)))
        x = [MARGIN + sum(slots[:side]) + side * GAP for side in range(len(slots))]
        width = max(
            x[-1] + slots[-1] + MARGIN,
            math.ceil(font(17, True).getlength(title)) + 2 * MARGIN,
        )
        legend, rows = self._legend(legend_items(mode, native, view), width)
        height = (
            MARGIN
            + TITLE_H
            + CAPTION_H
            + view.height
            + 8
            + rows * LEGEND_ROW_H
            + MARGIN
        )
        self.size = (width + width % 2, height + height % 2)
        self.template = Image.new("RGB", self.size, BACKGROUND)
        ImageDraw.Draw(self.template).text(
            (MARGIN, MARGIN), title, fill=INK, font=font(17, True)
        )
        self.template.paste(legend, (0, MARGIN + TITLE_H + CAPTION_H + view.height + 8))
        self.panel_origins = [(left, MARGIN + TITLE_H + CAPTION_H) for left in x]
        self.compass = self._compass(native.alignment)
        self._last = (None, None)

    def __len__(self):
        return len(self.shots)

    def _legend(self, items, width):
        """The legend strip and its row count: swatches and labels, wrapped to ``width``."""
        rows, x, placed = 0, MARGIN, []
        for label, kind, color in items:
            w = 16 + 6 + int(font(13).getlength(label)) + 18
            if x + w > width - MARGIN and x > MARGIN:
                rows, x = rows + 1, MARGIN
            placed.append((x, rows, label, kind, color))
            x += w
        strip = Image.new("RGB", (width, (rows + 1) * LEGEND_ROW_H), BACKGROUND)
        draw = ImageDraw.Draw(strip)
        for x, row, label, kind, color in placed:
            y = row * LEGEND_ROW_H + 3
            if kind == "fill":
                draw.rectangle(
                    (x, y, x + 15, y + 15), fill=color, outline=(120, 120, 120)
                )
            elif kind == "ramp":
                strip.paste(
                    Image.fromarray(
                        np.repeat(ramp(np.linspace(0.15, 1.0, 16))[None], 16, 0)
                    ),
                    (x, y),
                )
            else:
                draw.line((x, y + 8, x + 15, y + 8), fill=color, width=3)
            draw.text((x + 22, y), label, fill=INK, font=font(13))
        return strip, rows + 1

    def _compass(self, alignment):
        """Map-frame axes and a 5 m scale bar, drawn in each panel's lower corners."""
        view = self.view

        # Pixel directions of the map axes: plan x runs down, plan y right, and map = R(yaw) plan.
        axes = (
            ("x", (-math.sin(alignment.yaw), math.cos(alignment.yaw))),
            ("y", (math.cos(alignment.yaw), math.sin(alignment.yaw))),
        )

        def draw(image):
            d = ImageDraw.Draw(image)
            ox, oy = 44, view.height - 44
            for name, (dx, dy) in axes:
                d.line((ox, oy, ox + 24 * dx, oy + 24 * dy), fill=INK, width=2)
                d.text(
                    (ox + 34 * dx, oy + 34 * dy),
                    name,
                    fill=INK,
                    font=font(13, True),
                    anchor="mm",
                )
            d.ellipse((ox - 3, oy - 3, ox + 3, oy + 3), fill=INK)
            length = 5.0 * view.px_per_m
            x1, y = view.width - 14, view.height - 12
            d.line((x1 - length, y, x1, y), fill=INK, width=3)
            d.text((x1 - length, y - 17), "5 m", fill=INK, font=font(12))

        return draw

    def frame(self, index):
        """Frame ``index`` as an RGB array of shape (height, width, 3)."""
        shots = self.shots[index]
        if self._last[0] is not None and all(
            a is b for a, b in zip(self._last[0], shots)
        ):
            return self._last[1]
        image = self.template.copy()
        draw = ImageDraw.Draw(image)
        for panel, shot, (x, y) in zip(self.panels, shots, self.panel_origins):
            picture = panel.render(shot)
            self.compass(picture)
            image.paste(picture, (x, y))
            draw.text(
                (x, y - CAPTION_H + 4), shot.caption[0], fill=INK, font=font(15, True)
            )
            draw.text(
                (x, y - CAPTION_H + 24), shot.caption[1], fill=MUTED, font=font(13)
            )
        array = np.asarray(image)
        self._last = (shots, array)
        return array


def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def gif_palette():
    """One fixed palette for every GIF frame: the drawing colours, the soil ramp and greys for text edges."""
    colours = [
        NEUTRAL,
        TO_DIG,
        DUG,
        DUG_EDGE,
        OBSTACLE,
        NON_DUMPABLE,
        FINAL_DUMP,
        WORK,
        WORK_PALE,
        BACKGROUND,
        BODY,
    ]
    colours += [
        *CABIN.values(),
        *SHOVEL.values(),
        INK,
        MUTED,
        ROUTE,
        (255, 255, 255),
        (120, 120, 120),
    ]
    colours += [tuple(int(v) for v in c) for c in ramp(np.linspace(0.0, 1.0, 96))]
    colours += [(g, g, g) for g in range(0, 256, 8)]
    colours = list(dict.fromkeys(colours))[:256]
    palette = Image.new("P", (1, 1))
    palette.putpalette([v for c in colours for v in c] + [0] * 3 * (256 - len(colours)))
    return palette


def write(animation, out, fps, stills=0):
    """Encode ``animation`` to ``out`` (.mp4 through ffmpeg, else .gif) and save ``stills`` evenly spaced PNGs next to
    it. Returns the file, its format, frame count, duration, size and the still paths.
    """
    out = Path(out)
    if out.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError(f"--out must end in .mp4 or .gif, got {out.name}")
    if fps <= 0:
        raise ValueError("--fps must be positive")
    ffmpeg = find_ffmpeg() if out.suffix.lower() == ".mp4" else None
    notes = []
    if out.suffix.lower() == ".mp4" and ffmpeg is None:
        out = out.with_suffix(".gif")
        notes.append("no ffmpeg or imageio-ffmpeg here: wrote a GIF instead")
    out.parent.mkdir(parents=True, exist_ok=True)
    count = len(animation)
    at = (
        sorted({int(i) for i in np.linspace(0, count - 1, stills).round()})
        if stills > 0 and count
        else []
    )
    paths = []

    def frames():
        for index in range(count):
            frame = animation.frame(index)
            if index in at:
                path = out.with_name(f"{out.stem}_still_{index:04d}.png")
                Image.fromarray(frame).save(path)
                paths.append(str(path))
            yield frame

    width, height = animation.size
    if ffmpeg is not None:
        command = [
            ffmpeg,
            *"-y -loglevel error -f rawvideo -pix_fmt rgb24".split(),
            "-s",
            f"{width}x{height}",
        ]
        command += [
            "-r",
            f"{fps:g}",
            *"-i - -an -c:v libx264 -preset medium -tune animation -crf 20".split(),
        ]
        command += [*"-pix_fmt yuv420p -movflags +faststart".split(), str(out)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in frames():
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"ffmpeg failed to write {out}")
    else:
        palette = gif_palette()
        images = [
            Image.fromarray(frame).quantize(palette=palette, dither=Image.Dither.NONE)
            for frame in frames()
        ]
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=round(1000.0 / fps),
            loop=0,
        )
    return dict(
        path=str(out.resolve()),
        format=out.suffix.lower()[1:],
        mode=animation.mode,
        frames=count,
        fps=fps,
        duration_s=round(count / fps, 2),
        size_mb=round(out.stat().st_size / 1e6, 2),
        width=width,
        height=height,
        stills=paths,
        notes=notes,
    )


def load(evaluation, mode=None, *, runtime_validator=None):
    """The native plan and, for the machine modes, the replayed conversion of an evaluation directory (source/,
    conversion/, navigation/ and profile.yaml) or of a native source directory (terra_plan.json, arrays.npz).
    """
    evaluation = Path(evaluation).resolve()
    if (evaluation / "terra_plan.json").is_file() and (
        evaluation / "arrays.npz"
    ).is_file():
        mode = mode or "native"
        if mode != "native":
            raise ValueError(
                f"{evaluation} is a native source: it animates only --mode native"
            )
        return NativePlan.load(evaluation), None, mode
    report_path = evaluation / "conversion" / "report.json"
    if not report_path.is_file():
        raise ValueError(
            f"{evaluation}: expected an evaluation directory (conversion/report.json) or a native source"
        )
    mode = mode or "side-by-side"
    source = Path(json.loads(report_path.read_text())["inputs"]["input_dir"])
    if not (source / "terra_plan.json").is_file():
        source = evaluation / "source"
    native = NativePlan.load(source)
    if mode == "native":
        return native, None, mode
    routes = evaluation / "navigation"
    case = tmm_replay.load_case(
        evaluation, routes_dir=routes if routes.is_dir() else None
    )
    return (
        native,
        tmm_replay.replay(
            case, tmm_replay.Rules(), runtime_validator=runtime_validator
        ),
        mode,
    )


def animate(
    evaluation,
    out,
    mode=None,
    fps=20.0,
    stills=0,
    title=None,
    *,
    runtime_validator=None,
):
    """Write the animation of an evaluation (see ``load``); returns ``write``'s summary."""
    native, result, mode = load(evaluation, mode, runtime_validator=runtime_validator)
    evaluation = Path(evaluation).resolve()
    if title is None:
        title = (
            evaluation.name
            if (evaluation / "terra_plan.json").is_file()
            else f"{evaluation.parent.name}/{evaluation.name}"
        )
    return write(Animation(mode, native, result, title), out, fps, stills)
