"""One command for offline Terra plan processing and visual review."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _json(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def _ros_adapter(repo):
    """Load an explicitly selected ROS checkout's authoring adapter."""
    if repo is None:
        return {}
    path = Path(repo).resolve() / "scripts/TerraMapMaker/tmm_ros_adapter.py"
    if not path.is_file():
        raise ValueError(f"ROS checkout has no postprocessing adapter: {path}")
    spec = importlib.util.spec_from_file_location("terra_ros_postprocess_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure_checkout(Path(repo).resolve())
    return {
        "runtime_validator": module.runtime_checks,
        "station_validator": module.station_arrival,
    }


def fleet_command(args):
    from . import fleet, render, timeline

    source = json.loads(args.source.read_text())
    result = fleet.postprocess(
        source,
        cleanup=not args.no_cleanup,
        retime=not args.no_retime,
        max_search_states=args.max_search_states,
        max_schedule_steps=args.max_schedule_steps,
    )
    files = fleet.write_result(result, args.out)
    for variant, name in (("original", "original"), ("auto", "postprocessed")):
        data = timeline.from_fleet(result, variant=variant)
        files[f"{name}_timeline"] = str(
            timeline.write(data, args.out / f"{name}.json.gz").resolve()
        )
        files[f"{name}_page"] = str(
            render.write_html(data, args.out / f"{name}.html").resolve()
        )
    _json(
        {
            "status": result["report"]["status"],
            "material_events": result["report"]["material_events"],
            "schedule": result["schedule"]["status"],
            "files": files,
        }
    )
    return 0 if result["report"]["status"] == "GEOMETRIC_CANDIDATE" else 2


def import_fleet_command(args):
    from .fleet import import_native_npz

    sources = import_native_npz(
        args.trace,
        **json.loads(args.profile.read_text()),
        case_id=args.case,
        material_events_path=args.material_events,
    )
    paths = []
    for source in sources:
        path = args.out / f"case_{source['case_id']}" / "fleet_source.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
        paths.append(str(path.resolve()))
    _json({"sources": paths})
    return 0


def _solo_result(args):
    from . import render, replay, timeline

    adapter = _ros_adapter(args.ros_repo)
    result = replay.replay(
        replay.load_case(args.conversion, routes_dir=args.routes),
        runtime_validator=adapter.get("runtime_validator"),
    )
    report = replay.report(result)
    data = timeline.from_replay(
        result, station_validator=adapter.get("station_validator")
    )
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    recorded = timeline.write(data, args.out / "postprocessed.json.gz")
    page = render.write_html(data, args.out / "postprocessed.html")
    verdict = report["validity"]["verdict"]
    return (
        {
            "verdict": verdict,
            "report": str(report_path.resolve()),
            "timeline": str(recorded.resolve()),
            "page": str(page.resolve()),
        },
        2 if verdict == "invalid" else 0,
    )


def solo_command(args):
    result, status = _solo_result(args)
    _json(result)
    return status


def render_command(args):
    from .render import write_html

    _json({"page": str(write_html(args.recording, args.out).resolve())})
    return 0


def dashboard_command(args):
    from .dashboard import build

    page, rows = build(args.manifest, args.out, **_ros_adapter(args.ros_repo))
    _json({"page": str(page.resolve()), "versions": rows})
    return 0


def gallery_command(args):
    from .gallery import build

    page = build(args.manifest, args.out)
    _json({"page": str(page.resolve())})
    return 0


def animate_command(args):
    from .animate import animate

    _json(
        animate(
            args.evaluation,
            args.out,
            mode=args.mode,
            fps=args.fps,
            stills=args.stills,
            title=args.title,
            runtime_validator=_ros_adapter(args.ros_repo).get("runtime_validator"),
        )
    )
    return 0


def evaluate_command(args):
    """Robot conversion and Nav2 remain in the selected ROS checkout."""
    script = args.ros_repo.resolve() / "scripts/TerraMapMaker/tmm.py"
    if not script.is_file():
        raise ValueError(f"ROS checkout has no conversion command: {script}")
    command = [
        sys.executable,
        str(script),
        "plan",
        "evaluate",
        str(args.source),
        "--profile",
        str(args.profile),
        "--out",
        str(args.out),
        "--target-depth-m",
        str(args.target_depth_m),
    ]
    for name in ("plan", "tool", "margin_m", "drivable_spoil_height_m"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    for name in ("repair", "routes", "compact_dump_regions"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, text=True)
    if completed.returncode not in (0, 2):
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        return completed.returncode
    evaluation = json.loads((args.out / "summary.json").read_text())
    # A failed conversion still gets a visual review when its evidence is readable.
    review, solo_status = _solo_result(
        argparse.Namespace(
            conversion=args.out,
            routes=args.out / "navigation" if args.routes else None,
            ros_repo=args.ros_repo,
            out=args.out,
        )
    )
    _json({"evaluation": evaluation, "review": review})
    return completed.returncode or solo_status


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)

    command = commands.add_parser("fleet", help="clean and refine exact fleet substeps")
    command.add_argument("source", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--no-cleanup", action="store_true")
    command.add_argument("--no-retime", action="store_true")
    command.add_argument("--max-search-states", type=int, default=20000)
    command.add_argument("--max-schedule-steps", type=int, default=2000)
    command.set_defaults(run=fleet_command)

    command = commands.add_parser("import-fleet", help="import ordered native substeps")
    command.add_argument("trace", type=Path)
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--case", type=int)
    command.add_argument("--material-events", type=Path)
    command.set_defaults(run=import_fleet_command)

    command = commands.add_parser("solo", help="replay and render a saved conversion")
    command.add_argument("conversion", type=Path)
    command.add_argument("--routes", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--ros-repo", type=Path, help="also run ROS admission checks")
    command.set_defaults(run=solo_command)

    command = commands.add_parser("render", help="render a native or metric recording")
    command.add_argument("recording", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=render_command)

    command = commands.add_parser("dashboard", help="compare converted plan versions")
    command.add_argument("manifest", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--ros-repo", type=Path, help="also run ROS admission checks")
    command.set_defaults(run=dashboard_command)

    command = commands.add_parser("gallery", help="index successful and failed cases")
    command.add_argument("manifest", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=gallery_command)

    command = commands.add_parser("animate", help="export a plan as MP4 or GIF")
    command.add_argument("evaluation", type=Path)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--mode", choices=("native", "machine", "side-by-side"))
    command.add_argument("--fps", type=float, default=20.0)
    command.add_argument("--stills", type=int, default=0)
    command.add_argument("--title")
    command.add_argument("--ros-repo", type=Path, help="also run ROS admission checks")
    command.set_defaults(run=animate_command)

    command = commands.add_parser("evaluate", help="convert through the ROS adapter")
    command.add_argument("source", type=Path)
    command.add_argument("--ros-repo", type=Path, required=True)
    command.add_argument("--profile", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--plan", type=Path)
    command.add_argument("--tool", help="bucket name (default: selected ROS profile)")
    command.add_argument("--target-depth-m", type=float, default=0.5)
    command.add_argument("--margin-m", type=float)
    command.add_argument("--drivable-spoil-height-m", type=float)
    command.add_argument("--repair", action="store_true")
    command.add_argument("--routes", action="store_true")
    command.add_argument("--compact-dump-regions", action="store_true")
    command.set_defaults(run=evaluate_command)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}), file=sys.stderr)
        return 1
