# Terra recordings, postprocessing and media

`terra-postprocess` is the shared entry point for saved solo plans, exact fleet
traces, native recordings, comparison dashboards and galleries. The Python API
is `terra.postprocess`; `python -m terra.postprocess` runs the same commands.
The renderer and its browser assets are shipped with Terra. HTML export does
not need Node.js, ROS, a browser installation or a running web server. Video
export uses the same scene through headless Chromium and an encoder.

For the agent-driven repair loop, see [LLM-assisted plan refinement](REFINEMENT.md):
diagnose and edit, run deterministic checks, export a continuous-space artifact,
then feed its metric timeline directly to the Terra 3D viewer. Native execution
of a revised plan is a separate check.

## Choose a pipeline

1. **Record the native episode** in its matching policy and environment runtime.
   Save selected states with `terra.viewer3d.ReplayRecorder`; use `append_joint`
   for joint rounds. The [recording guide](../viewer3d/README.md#record-from-a-rollout)
   covers sequential and joint captures. Policy loading, recurrent state,
   resets, action selection and RNG stay in the evaluation adapter.
2. **Optionally postprocess** exact fleet substeps with `fleet`, or a saved solo
   conversion with `solo`. Postprocessing produces a separate metric timeline
   and validation report. It is not required to make a native video.
3. **Render either recording** with `render --out FILE.html`, `.mp4` or `.gif`.
   The same command supports solo, two excavators and mixed fleets.
4. **Index results** with `gallery`, or compare converted solo versions with
   `dashboard`. Keep successes, failures and missing recordings visible.

| Use | Entry point | Output |
| --- | --- | --- |
| Saved native or processed recording | `terra-postprocess render` | 3D HTML, MP4 or GIF |
| Fleet cleanup and workspace refinement | `terra-postprocess fleet` | Report, timelines and original/refined HTML |
| Solo converted-plan review | `terra-postprocess solo` or `evaluate` | Report, timeline and HTML |
| Result index or version comparison | `terra-postprocess gallery` or `dashboard` | HTML |
| Manual environment inspection | `python -m terra.viewer3d` | Live 3D viewer; export JSON or PNG |
| Solo waypoint/conversion comparison | `terra-postprocess animate` | 2D MP4 or GIF |

The older Pygame viewer in `terra.viz` and the compatible policy GIF tools in
`terra-baselines` remain useful for 2D diagnostics. They do not export the 3D
scene. There is no generic joint-checkpoint loader: new policy captures still
need the model and native joint-step adapter paired with that checkpoint.

## Install

In the Terra checkout and your existing Python environment:

```bash
python -m pip install -e '.[postprocess]'
terra-postprocess --help
```

For an offline-tools-only environment, install the package without its training
dependencies, then the portable dependencies:

```bash
python -m pip install --no-deps -e .
python -m pip install numpy scipy 'shapely>=2.0' PyYAML Pillow
```

The offline commands do not import JAX. Native capture uses the environment and
configuration that produced the plan; it does not change the training runtime.

For 3D video, also install the optional browser dependency and Chromium:

```bash
python -m pip install -e '.[postprocess,video]'
python -m playwright install chromium
```

The video exporter also needs `ffmpeg` on `PATH`, or an executable selected with
`--ffmpeg`. An installed Chrome or Chromium can be selected with `--browser`
instead of installing Playwright's Chromium. In an offline-tools-only environment,
install `playwright` alongside the portable dependencies above; keep the Terra
installation on `--no-deps` to avoid changing training dependencies.

## Native recordings and processed plans

These are different inputs with different evidence:

| Input | Meaning | Command |
| --- | --- | --- |
| `terra.viewer3d.v1` JSON or JSON.gz | Native states, including joint-round endpoints | `render` |
| `terra_fleet_source`, version 1 | Exact ordered per-agent native substeps with explicit geometry and soil units | `fleet` |
| Saved ROS conversion directory | Refined workspaces, terrain forecast and optional route evidence | `solo` |
| `terra.postprocessed.v1` JSON or JSON.gz | Metric playback of a processed plan | `render` |
| Gallery manifest | Index of successes, failures and missing recordings | `gallery` |
| Comparison manifest | Several converted versions of each solo plan | `dashboard` |

Native task success does not establish postprocessed geometry, route feasibility
or physical execution. Joint-round endpoints retain both machines' requests,
effective actions, rejection flags and stationary reservations when recorded.
The viewer does not interpolate these endpoints or infer per-agent soil
ownership from them. Reservations use native cell-edge coordinates; their drawn
boundaries exclude any extra guard margin and do not show action sweeps.

Metric playback keeps original terrain resolution, refined workspace masks and
saved route poses. Missing routes are explicit jumps. Failed, checked and
unverified routes retain different statuses. Arm animation and playback speed
are illustrative, not controller trajectories or physical durations.

## Render a saved recording

```bash
terra-postprocess render episode.json.gz --out episode.html
terra-postprocess render postprocessed.json.gz --out refined.html
terra-postprocess render episode.json.gz --out episode.mp4
terra-postprocess render postprocessed.json.gz --out refined.gif
```

The output extension selects the format. HTML is a self-contained interactive
page that opens offline. Its Python equivalent:

```python
from terra.postprocess.render import write_html

write_html("episode.json.gz", "episode.html")
```

Video shows each selected recorded endpoint for a fixed duration. It does not
interpolate machine motion, infer joint substeps or rerun the environment.
Native joint reservations and processed workspace/route overlays use the same
geometry as their HTML players. Display time is independent of physical time.

```bash
# Inclusive recording indices 0 through 30; half a second per endpoint.
terra-postprocess render episode.json.gz --out clip.mp4 --start 0 --stop 30 --step-seconds 0.5 --fps 20

# A top view for inspection.
terra-postprocess render postprocessed.json.gz --out top.mp4 --camera top --width 1280 --height 720
```

Video defaults are 1280 × 720, 20 fps and 0.5 seconds per recorded endpoint.
`--camera` selects `home` or `top`; `--presentation` selects `studio`, `paper`
or `diorama`; `--quality` selects `fast` or `high`. `--start` and `--stop` are
inclusive recording indices, not native step numbers. Omit them for the whole
recording. `--browser` and `--ffmpeg` accept explicit executable paths.

The existing `terra.viewer3d` manual-play application remains available. Both
paths share its scene and machine renderer. Developers rebuild the native and
postprocessed bundles together using `npm ci && npm run build` in
`terra/viewer3d/web`; `npm run check` runs the frontend tests.

## Two excavators or an excavator and skid

```bash
terra-postprocess fleet terra/postprocess/examples/fleet_two_excavators.json --out fleet_example
terra-postprocess fleet fleet_source.json --out fleet_review
```

Each output contains the report, compiled result, original and postprocessed
timelines, and both offline pages. Exit 0 means a complete geometric candidate;
exit 2 means unresolved material, workspace-refinement or fixed-path constraints.
The rejected candidate remains available for review. Neither result is a native
execution certificate. The packaged example is synthetic.

The processor preserves machine identity, every material event, global soil
order, partial pickups, load/setup state and the exact final terrain. It can
replace closed motion loops with WAIT, refine recorded excavation support into
continuous radial bands and retime retained paths against held-machine
reservations. It does not create replacement holding poses, routes or stations.
Use `--no-cleanup` or `--no-retime` for a controlled comparison.

Import a trace only when it contains the required exact substeps:

```bash
terra-postprocess import-fleet trace.npz --profile fleet_geometry.json --out sources
```

The explicit profile declares machine geometry, coordinate conversion and soil
units. Round-only snapshots are rejected: one machine can pick up soil just
released by the other inside a round, which an endpoint delta cannot attribute.
The newer joint-round viewer recordings therefore render directly but need an
exact native substep capture before fleet postprocessing.

`terra.postprocess.fleet.NativeFleetRecorder` records an explicitly ordered
native round. For guarded runtimes, provide requested actions, effective actions
and workspace-blocked flags from that same native step. It replays accepted
primitives and checks the resulting canonical state before storing substeps.
Requested blocked commands remain visible as such. The supplied native runtime
must support the recorder's joint-step API; the postprocessor does not substitute
a different environment or infer action order.

## Saved solo conversions and ROS integration

```bash
terra-postprocess solo conversion_output --routes checked_routes --out solo_review
terra-postprocess solo conversion_output --ros-repo /path/to/moleworks_ros --out solo_review
```

The portable path replays the saved terrain forecast and route evidence. ROS
schema/preflight and station-arrival checks run only through an explicitly
selected ROS adapter; otherwise their missing evidence stays unverified.
Run adapter commands in the normal sourced ROS container.

To perform fresh robot-specific conversion and then render the result:

```bash
terra-postprocess evaluate saved_source --profile profile.yaml --ros-repo /path/to/moleworks_ros --out evaluated --repair --routes
```

ROS retains machine profiles, tool geometry, radial conversion, Nav2 checking,
GridMap export, preflight and execution. Its existing `TerraMapMaker/tmm.py plan`
commands forward portable replay/rendering operations to this package and supply
the ROS validators. There is one portable processor and renderer implementation;
the robot runtime remains the authority for field admission.

For the older 2D solo comparison:

```bash
terra-postprocess animate evaluated --out plan.mp4 --mode side-by-side --stills 4
```

This command accepts a schema-v2 native source directory or a solo evaluation
directory, not a native snapshot recording or fleet metric timeline. Its
`native`, `machine` and `side-by-side` modes compare waypoint work with the
converted plan; the native panel reconstructs waypoint changes rather than
replaying exact rollout states. MP4 needs ffmpeg or imageio-ffmpeg; this 2D
command falls back to GIF if neither is available.

## Galleries and comparisons

```bash
terra-postprocess gallery gallery.json --out index.html
terra-postprocess dashboard comparison.json --out comparison.html
```

A gallery records native and postprocessed outcomes separately, keeps failures
visible, identifies current/legacy/synthetic cases, and reports missing replay
files. Relative links resolve from the manifest location. An evaluation failure
without a trace is not an animated failure: retain the entry and its source
metrics until an appropriate recording is available. Fresh CPU replays must be
labelled separately from historical GPU episodes.

Optional `evaluations` entries show full-panel rates separately from gallery
entry counts. Each needs `title`, integer `completed` and `total` (with
`0 <= completed <= total` and `total > 0`), and `source_url`; `note` is optional.
Panel rates stay fixed when filtering gallery entries. A case's optional
`native_label` changes its replay button text only. For a historical GPU failure
with a fresh CPU recapture, keep `native_status: "failed"`, label the button
`"CPU recapture (complete)"` or `"CPU recapture (incomplete)"`, and distinguish the
two outcomes in its metrics and note.

Example gallery entry:

```json
{
  "title": "Fleet review",
  "evaluations": [{
    "title": "Mixed u16867 · capacity 52 · greedy",
    "completed": 42,
    "total": 44,
    "note": "Saved full GPU panel; active fleet cases only.",
    "source_url": "evaluation/results.json"
  }],
  "cases": [{
    "id": "mixed-96",
    "title": "Unfinished foundation",
    "fleet": "mixed",
    "generation": "current",
    "checkpoint": "u16867, capacity 52",
    "native_status": "failed",
    "postprocessed_status": "not_run",
    "native_url": "replays/case_096.html",
    "native_label": "CPU recapture (incomplete)",
    "metrics": {"GPU soil delivered": "0 / 230", "CPU soil delivered": "0 / 230"},
    "note": "Historical GPU failure; this fresh CPU recapture also remains incomplete."
  }]
}
```

Comparison manifests group converted versions and may include a native replay:

```json
{
  "title": "Refined plans",
  "cases": [{
    "name": "Foundation",
    "terra3d": "original.json.gz",
    "versions": [{"label": "Refined", "conversion": "evaluated", "routes": "evaluated/navigation"}]
  }]
}
```

The dashboard uses the packaged studio renderer. Old `terra_viewer` source
checkout entries are no longer required. The recorded native plan and the
metric postprocessed plan keep separate modes and evidence.

## Verification

Run the portable tests without ROS or JAX, then the shared frontend checks:

```bash
python -m pytest tests/test_postprocess*.py terra/tests/test_postprocess*.py terra/tests/test_viewer3d_replay.py
npm --prefix terra/viewer3d/web run check
npm --prefix terra/viewer3d/web run build
```

ROS adapter changes additionally require the owning ROS package tests in its
container. Validate wheel-installed HTML export outside either source checkout
to check that all templates and renderer assets are packaged.
