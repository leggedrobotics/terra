"""Launch manual play or a portable recording: python -m terra.viewer3d."""

import argparse
import os
from pathlib import Path
import webbrowser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--map",
        type=Path,
        help="Terra single-map directory (image.npy or images/img_1.npy and companion layers).",
    )
    inputs.add_argument(
        "--replay", type=Path, help="Replay JSON or JSON.gz; viewing requires no JAX."
    )
    inputs.add_argument(
        "--demo-replay",
        action="store_true",
        help="Record and open a real Terra dig/dump demonstration.",
    )
    inputs.add_argument("--pull-inspector", action="store_true",
                        help="Inspect the exact saved October 8 pull-rule starts with native manual actions.")
    parser.add_argument("--initial-states", type=Path, help="Saved diagnostic initial_states.pkl for --pull-inspector.")
    parser.add_argument("--manual-output", type=Path, help="Directory for manual actions and complete native state recordings.")
    parser.add_argument("--case", default="17411", help="Saved source slot for the pull inspector.")
    parser.add_argument("--precision-mode", choices=("bulk", "precision"), default="precision")
    parser.add_argument("--start", type=int, default=0, choices=(0, 1))
    parser.add_argument(
        "--export",
        type=Path,
        help="Save the selected recording as .json, .json.gz, or standalone .html, then exit.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Loopback HTTP port (default: 8765; 0 picks a free port).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the URL without opening a browser.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--agents",
        type=int,
        nargs="+",
        default=[0],
        metavar="TYPE",
        help="One to four types: 0 excavator, 1 truck, 2 skid steer.",
    )
    parser.add_argument(
        "--action-types",
        type=int,
        nargs="+",
        metavar="TYPE",
        help="One per agent: 0 tracked, 1 wheeled. Defaults to wheeled for trucks.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535.")
    if args.pull_inspector and (args.initial_states is None or not args.initial_states.is_file()):
        parser.error("--pull-inspector requires --initial-states /path/to/initial_states.pkl (an existing saved native state bank).")
    from .replay import ReplayRecorder, load_replay
    from .server import STATIC_DIR, make_server

    if not (STATIC_DIR / "viewer.js").is_file() and (
        not args.export or args.export.suffix == ".html"
    ):
        parser.error(
            "Build the viewer once: cd terra/viewer3d/web && npm ci && npm run build"
        )
    session = None
    replay = None
    if args.replay:
        replay = load_replay(args.replay)
        recorder = ReplayRecorder(metadata=replay["metadata"])
        recorder.frames.extend(replay["frames"])
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        print(
            "Preparing Terra for manual play (the first JAX compilation takes a moment)...",
            flush=True,
        )
        if args.pull_inspector:
            from .pull_session import PullSession
            options = dict(case_id=args.case, precision=args.precision_mode == "precision", start=args.start)
            if args.initial_states:
                options["initial_states"] = args.initial_states
            if args.manual_output:
                options["output"] = args.manual_output
            session = PullSession(**options)
        else:
            from .session import ManualSession
            session = ManualSession(
                map_path=args.map,
                seed=args.seed,
                max_steps=args.max_steps,
                agent_types=tuple(args.agents),
                action_types=tuple(args.action_types) if args.action_types else None,
            )
        recorder = session.play_demo() if args.demo_replay else session.recorder
        if args.demo_replay:
            replay = recorder.to_dict()
            session = None
    if args.export:
        if args.export.suffix.lower() == ".html":
            recorder.save_html(args.export)
        else:
            recorder.save(args.export)
        print(f"Saved {len(recorder.frames)} frames to {args.export.resolve()}")
        return
    try:
        server = make_server(session=session, replay=replay, port=args.port)
    except OSError as exc:
        parser.error(
            f"Cannot bind viewer port {args.port}: {exc}. Choose --port 0 for a free port."
        )
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Terra 3D: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
