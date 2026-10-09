"""Portable JSON and offline HTML recording for the Terra 3D viewer."""

from collections.abc import Mapping
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re

import numpy as np

from .snapshots import (
    MAX_AGENTS,
    SCHEMA,
    _field,
    _Selection,
    snapshot_from_timestep,
    validate_frame,
)


def _json_text(value):
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Replay must contain finite JSON-compatible values: {error}"
        ) from error


def validate_replay(replay):
    """Reject unsupported or malformed recordings and return the unchanged input."""
    if not isinstance(replay, dict):
        raise ValueError("Replay must be a JSON object")
    if replay.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported replay schema; expected {SCHEMA}")
    metadata = replay.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Replay metadata must be an object")
    for name in ("title", "source"):
        if not isinstance(metadata.get(name), str):
            raise ValueError(f"Replay metadata.{name} must be a string")
    frames = replay.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay must contain at least one frame")
    for index, frame in enumerate(frames):
        try:
            validate_frame(frame)
        except ValueError as error:
            raise ValueError(f"Frame {index}: {error}") from error
    _json_text(replay)
    return replay


def load_replay(path_or_dict):
    """Load and validate plain JSON, gzip JSON, or an in-memory replay mapping."""
    if isinstance(path_or_dict, Mapping):
        return deepcopy(validate_replay(dict(path_or_dict)))
    path = Path(path_or_dict)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as file:
            replay = json.load(file)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        gzip.BadGzipFile,
        EOFError,
    ) as error:
        raise ValueError(f"Cannot read Terra replay {path}: {error}") from error
    return validate_replay(replay)


class ReplayRecorder:
    """Collect detached full-state frames outside a compiled training loop."""

    def __init__(self, metadata=None):
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        self.metadata = {
            "title": "Terra replay",
            "source": "terra",
            **deepcopy(dict(metadata or {})),
        }
        _json_text(self.metadata)
        self.frames = []

    def append(self, timestep, action=None, actor_id=None, env_index=None):
        frame = snapshot_from_timestep(
            timestep, action=action, actor_id=actor_id, env_index=env_index
        )
        # Only consecutive steps establish which agent executed this transition.
        if action is not None and actor_id is None and self.frames:
            previous = self.frames[-1]
            if frame["step"] == previous["step"] + 1:
                frame["actor_id"] = previous["current_agent"]
                validate_frame(frame)
        self.frames.append(frame)
        return frame

    def append_joint(
        self,
        timestep,
        *,
        requested_actions,
        effective_actions,
        order,
        workspace_blocked,
        workspace_polygons,
        env_index=None,
    ):
        """Record an actual joint endpoint with explicit native transition evidence.

        Action and rejection vectors use original machine slots, including any
        inactive slots. ``order`` contains each active slot once. These vectors
        may carry the timestep's batch axes or already select one environment.
        Polygons always select one environment: one ``body`` and one ``work``
        polygon per active slot, with vertices in cell-edge grid coordinates.

        Pass all four transition arguments as ``None`` for the reset frame at
        step zero. Subsequent frames require every argument; neither execution
        order, rejected commands nor reservation geometry is inferred here.
        """
        frame = snapshot_from_timestep(timestep, env_index=env_index)
        transition = {
            "joint_actions": requested_actions,
            "effective_joint_actions": effective_actions,
            "joint_order": order,
            "workspace_blocked": workspace_blocked,
        }
        absent = [name for name, value in transition.items() if value is None]
        if absent and (len(absent) != len(transition) or frame["step"] != 0):
            raise ValueError(
                "Joint transitions require requested_actions, effective_actions, "
                "order and workspace_blocked; only reset may supply all as None"
            )
        state = _field(timestep, "state")
        action_map = _field(_field(_field(state, "world"), "action_map"), "map")
        selection = _Selection(action_map, env_index)
        vector_shapes = tuple((count,) for count in range(1, MAX_AGENTS + 1))
        for name, value in transition.items():
            # Preserve validation of mixed Python lists such as [False, 2];
            # NumPy would otherwise silently convert the boolean to slot zero.
            if name != "workspace_blocked" and isinstance(value, (tuple, list)):
                if any(
                    isinstance(item, (bool, np.bool_))
                    for item in np.asarray(value, dtype=object).flat
                ):
                    raise ValueError(f"{name} must contain integers, not booleans")
            frame[name] = (
                None
                if value is None
                else selection.array(value, name, vector_shapes).tolist()
            )
        if not isinstance(workspace_polygons, (list, tuple)):
            raise ValueError(
                "workspace_polygons must contain native body/work polygons"
            )
        polygons = []
        for polygon in workspace_polygons:
            if not isinstance(polygon, Mapping):
                raise ValueError("A workspace polygon must be an object")
            detached = deepcopy(dict(polygon))
            if "id" in detached:
                identity = np.asarray(detached["id"])
                if identity.shape != ():
                    raise ValueError("Workspace polygon id must be a scalar")
                detached["id"] = identity.item()
            if "vertices" in detached:
                detached["vertices"] = np.asarray(detached["vertices"]).tolist()
            polygons.append(detached)
        frame["workspace_polygons"] = polygons
        validate_frame(frame)
        expected = {
            (agent["id"], component)
            for agent in frame["agents"]
            for component in ("body", "work")
        }
        if {(p["id"], p["component"]) for p in polygons} != expected:
            raise ValueError(
                "Joint capture requires native body and work polygons for every active slot"
            )
        _json_text(frame)
        self.frames.append(frame)
        return frame

    def to_dict(self):
        return deepcopy(
            {"schema": SCHEMA, "metadata": self.metadata, "frames": self.frames}
        )

    def save(self, path):
        """Save .json or .json.gz; return the destination Path."""
        path = Path(path)
        if not (path.name.endswith(".json") or path.name.endswith(".json.gz")):
            raise ValueError("Replay output must end in .json or .json.gz")
        replay = validate_replay(self.to_dict())
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as file:
            file.write(_json_text(replay))
        return path

    def save_html(self, path):
        """Export a self-contained, network-independent browser replay."""
        path = Path(path)
        replay = validate_replay(self.to_dict())
        static = Path(__file__).with_name("static")
        html = (static / "index.html").read_text(encoding="utf-8")
        css = (static / "style.css").read_text(encoding="utf-8")
        javascript = (static / "viewer.js").read_text(encoding="utf-8")
        # HTML raw-text elements close on </script even inside quoted JSON/JS.
        payload = (
            _json_text(replay)
            .replace("<", "\\u003c")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
        javascript = re.sub(r"</script", r"<\\/script", javascript, flags=re.IGNORECASE)
        embedded = (
            f'<script type="application/json" id="terra-replay">{payload}</script>'
        )
        html, styles_count = re.subn(
            r"<link\b[^>]*href=[\"\'](?:/static/|static/|\./)?style\.css[\"\'][^>]*>",
            lambda _: f"<style>{css}</style>",
            html,
            flags=re.IGNORECASE,
        )
        html, scripts_count = re.subn(
            r"<script\b[^>]*src=[\"\'](?:/static/|static/|\./)?viewer\.js[\"\'][^>]*>\s*</script\s*>",
            lambda _: embedded + f"<script>{javascript}</script>",
            html,
            flags=re.IGNORECASE,
        )
        if styles_count != 1 or scripts_count != 1:
            raise ValueError(
                "Viewer template must reference one style.css and one viewer.js bundle"
            )
        path.write_text(html, encoding="utf-8")
        return path
