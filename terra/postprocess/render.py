"""Export native Terra replays and metric plans using the shared 3D viewer."""

from collections.abc import Mapping
import gzip
import html
from importlib import resources
import json
from pathlib import Path
import re


def player_bundle(bundle=None):
    """Read the installed plan player, or an explicitly supplied prebuilt JS file."""
    if bundle is not None:
        path = Path(bundle)
        if path.suffix.lower() != ".js" or not path.is_file():
            raise ValueError("A player override must be a prebuilt .js file")
        return path.read_text(encoding="utf-8"), dict(
            viewer=str(path), revision="prebuilt bundle"
        )
    asset = resources.files("terra.viewer3d").joinpath("static", "postprocessed.js")
    return asset.read_text(encoding="utf-8"), dict(
        viewer="terra.viewer3d/static/postprocessed.js", revision="packaged"
    )


def load_recording(data_or_path):
    """Read and validate either supported recording schema without a simulator."""
    if isinstance(data_or_path, Mapping):
        data = dict(data_or_path)
    else:
        path = Path(data_or_path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("A recording must be a JSON object")
    if data.get("schema") == "terra.viewer3d.v1":
        from terra.viewer3d import validate_replay

        validate_replay(data)
    elif data.get("schema") == "terra.postprocessed.v1":
        from . import timeline

        timeline.validate(data)
    else:
        raise ValueError("Expected terra.viewer3d.v1 or terra.postprocessed.v1")
    return data


def write_html(data_or_path, out, *, bundle=None):
    """Write one native or metric recording using installed assets; return its Path.

    Reading and rendering recordings does not initialize JAX or a ROS runtime.
    Node is only needed when developing the checked-in browser bundles.
    """
    data = load_recording(data_or_path)
    out = Path(out)
    if data["schema"] == "terra.viewer3d.v1":
        from terra.viewer3d import ReplayRecorder

        recorder = ReplayRecorder(metadata=data["metadata"])
        recorder.frames.extend(data["frames"])
        out.parent.mkdir(parents=True, exist_ok=True)
        return recorder.save_html(out)
    javascript, _ = player_bundle(bundle)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    title = html.escape(
        data.get("metadata", {}).get("title", "Postprocessed Terra plan")
    )
    javascript = (
        "<script>"
        + re.sub(r"</script", r"<\\/script", javascript, flags=re.IGNORECASE)
        + "</script>"
    )
    template = (
        resources.files(__package__)
        .joinpath("assets", "player.html")
        .read_text(encoding="utf-8")
    )
    page = (
        template.replace("__TITLE__", title)
        .replace("__DATA__", payload)
        .replace("__BUNDLE__", javascript)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def write_video(data_or_path, out, **options):
    """Export MP4/GIF with the same viewer; browser dependencies are optional."""
    from .video import write_video as export

    return export(data_or_path, out, **options)
