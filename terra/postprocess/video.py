"""Offline endpoint videos from the same Three.js scene as the HTML renderer."""

from importlib import resources
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from .render import load_recording, write_html


def _clip(count, start, stop, fps, step_seconds, width, height):
    stop = count - 1 if stop is None else stop
    if not all(type(value) is int for value in (start, stop, fps, width, height)):
        raise ValueError("Frame indices, fps, width and height must be integers")
    if not 0 <= start <= stop < count:
        raise ValueError(f"Expected 0 <= start <= stop < {count} recorded frames")
    if fps <= 0 or not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("fps and step-seconds must be positive and finite")
    if min(width, height) < 64 or width % 2 or height % 2:
        raise ValueError("Video dimensions must be even and at least 64 pixels")
    repeats = max(1, math.floor(fps * step_seconds + 0.5))
    return range(start, stop + 1), repeats


def _encode(ffmpeg, images, destination, *, fps, repeats, count):
    # One PNG per exact endpoint; ffmpeg repeats it without browser wall-clock
    # sampling. Every endpoint, including the terminal one, receives a full hold.
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "2",
        "-filter_complex_threads",
        "1",
        "-filter_threads",
        "1",
        "-framerate",
        f"{fps}/{repeats}",
        "-i",
        str(images / "%06d.png"),
    ]
    rate = f"fps={fps}:round=near"
    if destination.suffix.lower() == ".mp4":
        command += [
            "-vf",
            rate,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
        ]
    else:
        command += [
            "-filter_complex",
            rate + ",split[a][b];[a]palettegen[p];" "[b][p]paletteuse=dither=bayer",
            "-loop",
            "0",
        ]
    command += ["-frames:v", str(count * repeats), str(destination)]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr[-4000:].strip()}")
    if not destination.is_file() or not destination.stat().st_size:
        raise RuntimeError("ffmpeg did not produce a video")


def write_video(
    data_or_path,
    out,
    *,
    fps=20,
    step_seconds=0.5,
    start=0,
    stop=None,
    width=1280,
    height=720,
    camera="home",
    quality="high",
    presentation="studio",
    browser=None,
    ffmpeg="ffmpeg",
):
    """Write an MP4/GIF and return its path and timing as a JSON-compatible dict.

    No policy/environment is run. Both schemas keep their exact recorded states,
    native reservations or refined metric workspaces, and visible result labels.
    Playback timing is illustrative; no transition motion is inferred. A failed
    export leaves an existing destination intact.
    """
    out = Path(out).resolve()
    if out.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("Video output must end in .mp4 or .gif")
    data = load_recording(data_or_path)
    count = len(data["frames"]) + int(data["schema"] == "terra.postprocessed.v1")
    indices, repeats = _clip(count, start, stop, fps, step_seconds, width, height)
    if camera not in ("home", "top") or quality not in ("fast", "high"):
        raise ValueError("Expected camera home/top and quality fast/high")
    if presentation not in ("studio", "paper", "diorama"):
        raise ValueError("Expected presentation studio, paper or diorama")
    encoder = shutil.which(str(ffmpeg))
    if encoder is None:
        raise RuntimeError("Video export requires ffmpeg on PATH or --ffmpeg PATH")
    if browser is not None and not Path(browser).is_file():
        raise ValueError(f"Chromium/Chrome executable does not exist: {browser}")
    try:
        from playwright.sync_api import Error as BrowserError, sync_playwright
    except ImportError as error:
        raise RuntimeError(
            "Video export requires playwright (Terra's [video] extra). Install it "
            "and run `python -m playwright install chromium`, or supply --browser."
        ) from error

    driver = resources.files(__package__).joinpath("assets", "capture.js").read_text()
    errors = []
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final replace is atomic, even when
    # /tmp and the chosen output directory live on different filesystems.
    with tempfile.TemporaryDirectory(prefix=".terra-video-", dir=out.parent) as tmp:
        directory = Path(tmp)
        page_path = write_html(data, directory / "recording.html")
        images = directory / "frames"
        images.mkdir()
        try:
            with sync_playwright() as playwright:
                chromium = playwright.chromium.launch(
                    executable_path=str(Path(browser).resolve()) if browser else None,
                    headless=True,
                    args=[
                        "--disable-dev-shm-usage",
                        "--use-angle=swiftshader",
                        "--enable-unsafe-swiftshader",
                    ],
                )
                try:
                    page = chromium.new_page(
                        viewport={"width": width, "height": height},
                        device_scale_factor=1,
                        reduced_motion="reduce",
                    )
                    page.on("pageerror", lambda error: errors.append(str(error)))

                    def route(request):
                        if request.request.url.startswith(("http:", "https:")):
                            errors.append(
                                "Unexpected network request: " + request.request.url
                            )
                            request.abort()
                        else:
                            request.continue_()

                    page.route("**/*", route)
                    # Disable live render/playback loops before the page starts.
                    # The capture driver renders explicitly after each seek.
                    page.add_init_script(
                        "window.terraVideoRAF = window.requestAnimationFrame;"
                        "window.requestAnimationFrame = () => 0;"
                    )
                    page.goto(page_path.as_uri(), wait_until="load", timeout=60000)
                    page.wait_for_function(
                        "() => (window.timelineView || window.terraViewer)?.scene?.frame",
                        polling=100,
                        timeout=60000,
                    )
                    page.evaluate("() => { window.terraCapture = " + driver + "; }")
                    page.evaluate(
                        "options => window.terraCapture.prepare(options)",
                        {
                            "camera": camera,
                            "quality": quality,
                            "presentation": presentation,
                        },
                    )
                    for number, index in enumerate(indices):
                        page.evaluate("index => window.terraCapture.show(index)", index)
                        if errors:
                            raise RuntimeError(
                                "3D capture failed: " + "; ".join(errors)
                            )
                        page.screenshot(
                            path=str(images / f"{number:06d}.png"),
                            animations="disabled",
                            timeout=60000,
                        )
                    if errors:
                        raise RuntimeError("3D capture failed: " + "; ".join(errors))
                finally:
                    chromium.close()
        except BrowserError as error:
            raise RuntimeError(
                "3D browser capture failed. Install Chromium with "
                "`python -m playwright install chromium` or use --browser PATH. "
                + str(error)
            ) from error
        staged = directory / ("video" + out.suffix.lower())
        _encode(encoder, images, staged, fps=fps, repeats=repeats, count=len(indices))
        staged.replace(out)

    return dict(
        video=str(out),
        schema=data["schema"],
        recorded_frames=len(indices),
        first_frame=indices.start,
        last_frame=indices.stop - 1,
        width=width,
        height=height,
        fps=fps,
        step_seconds=repeats / fps,
        duration_s=len(indices) * repeats / fps,
        playback="recorded endpoints",
    )
