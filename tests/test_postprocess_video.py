"""Endpoint timing and optional real browser/encoder checks for both schemas."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from terra.postprocess.cli import main
from terra.postprocess.video import _clip, _encode, write_video
from terra.tests.test_postprocess_render import metric_plan, native_joint


def test_clip_keeps_both_endpoints_and_rounds_the_hold():
    indices, repeats = _clip(12, 3, 11, 20, 0.51, 1280, 720)
    assert list(indices) == list(range(3, 12))
    assert repeats == 10
    assert list(_clip(1, 0, None, 20, 0.5, 1280, 720)[0]) == [0]


@pytest.mark.parametrize(
    "changes",
    [
        {"start": -1},
        {"stop": 12},
        {"start": 6, "stop": 5},
        {"fps": 0},
        {"fps": 1.5},
        {"step_seconds": math.nan},
        {"step_seconds": math.inf},
        {"step_seconds": -1},
        {"width": 1279},
        {"height": 0},
    ],
)
def test_invalid_clip_options_fail_before_starting_a_browser(changes):
    options = dict(start=0, stop=None, fps=20, step_seconds=0.5, width=1280, height=720)
    options.update(changes)
    with pytest.raises(ValueError):
        _clip(12, **options)


def test_unknown_extension_does_not_overwrite_destination(tmp_path, capsys):
    out = tmp_path / "saved.txt"
    out.write_text("keep me")
    assert main(["render", "not-read.json", "--out", str(out)]) == 1
    assert ".html, .mp4 or .gif" in capsys.readouterr().err
    assert out.read_text() == "keep me"


def test_missing_encoder_is_actionable_and_preserves_output(tmp_path):
    out = tmp_path / "existing.mp4"
    out.write_bytes(b"keep me")
    with pytest.raises(RuntimeError, match="requires ffmpeg"):
        write_video(metric_plan(), out, ffmpeg=str(tmp_path / "missing-ffmpeg"))
    assert out.read_bytes() == b"keep me"
    assert not list(tmp_path.glob(".terra-video-*"))


def _probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="optional ffmpeg installation",
)
@pytest.mark.parametrize("suffix", ["mp4", "gif"])
def test_encoder_keeps_initial_and_terminal_holds(tmp_path, suffix):
    from PIL import Image

    for i, color in enumerate(("red", "blue", "green")):
        Image.new("RGB", (64, 64), color).save(tmp_path / f"{i:06d}.png")
    out = tmp_path / f"clip.{suffix}"
    _encode("ffmpeg", tmp_path, out, fps=20, repeats=10, count=3)
    probe = _probe(out)
    assert float(probe["format"]["duration"]) == pytest.approx(1.5, abs=0.02)
    # Decode actual pixels: first and final endpoint must survive the encoder.
    pixels = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(out),
            "-vf",
            "scale=1:1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "-",
        ]
    )
    assert pixels[0] > 200 and pixels[1] < 30 and pixels[2] < 30
    assert pixels[-2] > 100 and pixels[-3] < 30 and pixels[-1] < 30


@pytest.mark.skipif(
    not os.environ.get("TERRA_VIDEO_BROWSER"),
    reason="set TERRA_VIDEO_BROWSER to run real browser integration",
)
@pytest.mark.parametrize("schema,suffix", [("native", "mp4"), ("metric", "gif")])
def test_real_browser_uses_saved_recording_without_runtime(tmp_path, schema, suffix):
    data = native_joint() if schema == "native" else metric_plan()
    out = tmp_path / f"{schema}.{suffix}"
    result = write_video(
        data,
        out,
        fps=10,
        step_seconds=0.2,
        width=640,
        height=360,
        quality="fast",
        browser=os.environ["TERRA_VIDEO_BROWSER"],
    )
    count = len(data["frames"]) + (schema == "metric")
    assert result["recorded_frames"] == count
    assert result["duration_s"] == pytest.approx(count * 0.2)
    assert result["playback"] == "recorded endpoints"
    probe = _probe(out)
    assert float(probe["format"]["duration"]) == pytest.approx(count * 0.2, abs=0.02)
    assert (probe["streams"][0]["width"], probe["streams"][0]["height"]) == (640, 360)
    assert not list(tmp_path.glob(".terra-video-*"))
