# Terra 3D

A stylized, low-poly 3D view of Terra for manual play, map inspection, and
recorded episodes, with articulated machines, grouped boulders and containers,
stepped excavation cuts, and connected soil mounds.
The [design specification](../../VISUALIZATION_3D_DESIGN.md) describes the state
contract and visual behavior. Terra computes every action; the articulated arm
and soil particles illustrate the resulting discrete transition.

## Start

From the Terra checkout, using your existing Terra Python environment:

```bash
JAX_PLATFORMS=cpu python -m terra.viewer3d
```

This opens a built-in yard with an excavator, a partially excavated trench,
dumping targets, and obstacles. The first launch compiles Terra's CPU step and
can take a minute or two. The browser assets are bundled; normal use requires
no Node.js installation or internet access.

On the Moleworks workstation, the existing environment is:

```bash
JAX_PLATFORMS=cpu /home/lorenzo/moleworks/.venv-terra-uv/bin/python -m terra.viewer3d
```

The default address is `http://127.0.0.1:8765`. Use `--no-open` to print the
address without opening it, `--port 0` to choose a free port, or `--port 8766`
to select another port. The service only listens on loopback. For SSH forwarding,
keep the same local and remote port, for example `ssh -L 8765:127.0.0.1:8765 host`.
Stop the service with Ctrl+C.

```bash
# Multiple machines; turns follow Terra's normal active-agent sequence.
python -m terra.viewer3d --agents 0 1 2

# One wheeled excavator.
python -m terra.viewer3d --agents 0 --action-types 1

# A real single-map directory, with existing Terra companion layers.
python -m terra.viewer3d --map /path/to/single-map --seed 24

# Observe a scripted dig/slew/dump/move sequence, computed by Terra.
python -m terra.viewer3d --demo-replay
```

The map option accepts Terra's existing single-map directory formats:

- `image.npy`, `occupancy.npy`, `dumpability.npy`, `distance.npy`, optional
  `actions.npy` and `metadata.json`; or
- `images/img_1.npy`, `occupancy/img_1.npy`, `dumpability/img_1.npy`,
  `distance/img_1.npy`, optional `actions/img_1.npy` and `metadata/map.json`.

Manual maps must be square, 32–128 cells per side. The viewer uses the current
default environment configuration with the requested agent types and horizon;
it does not infer a checkpoint's reward or environment settings from a map.
For a policy episode, capture states in its matching evaluation runtime as
described under [Record from a rollout](#record-from-a-rollout).
The manual viewer checks the chosen seed's placements before starting Terra's
reset sampler. If it cannot place every machine within 4,096 proposals each,
it reports the failure; try another seed, fewer agents, or more free space.

## Pull-rule inspector

For distance, turn amount and direct work-heading controls, add
`--structured-actions --time-budget-seconds 14400 --decision-budget 450`.
This opt-in mode uses estimated machine time and a decision cap; see the
[structured-action contract](../../docs/STRUCTURED_ACTIONS.md). Its native
masks include productive soil pickup and full-load unloading. Undo and export
preserve the argument and time history.

Control an excavator from an exact saved initial state, with native pull-rule
eligibility and unload checks:

```bash
JAX_PLATFORMS=cpu python -m terra.viewer3d \
  --pull-inspector --initial-states /path/to/initial_states.pkl \
  --manual-output manual-recordings \
  --case 17411 --precision-mode precision --start 0
```

`--initial-states` is required. It is an external, trusted Python pickle
containing parallel `episodes` metadata and `initial` native State lists, not
a JSON viewer replay. Only load a bank you trust. The inspector selects the
saved `greedy` entries and offers their cases, starts, and bulk/precision modes.
Each state must use the pull rule, one tracked excavator, and a 450-action
horizon. Configurations must match across the bank except for the precision
requirement (`enforce_foundation_border_alignment`). No pose is recreated.

While the bucket is empty, green marks the exact fresh cells a native `DO`
would excavate now; amber marks additional cells available after cabin rotation
at the same base pose; red marks remaining targets unavailable at every cabin
heading from this pose. The purple outline is the immutable precision-required
band. The footprint is native; the workspace cone is a geometric preview,
not an eligibility promise. While loaded, fresh eligibility is suppressed.
Use Q/E to swing toward a direction reported as unloadable, then Space to
unload; the status distinguishes accepted disposal from off-zone unloading.

Undo restores the native state and action history. Reset and the case controls
return to the selected saved start. At 450 actions the episode freezes;
**Continue exploring** explicitly permits further actions without changing
the 450-action configuration. A finish after the budget is exploratory, not a
450-action episode success.
Export writes the replay, action tape, and complete native physical states under
`--manual-output`; resetting also saves a nonempty tape. Local policy-observation
caches are not refreshed by this interactive path: rebuild them with the normal
Terra wrapper before using an exported state as policy input.

The main integration passed 64 JavaScript checks and four Python lifecycle
checks (`python -m unittest terra.tests.test_pull_viewer_session -v`, with
`TERRA_PULL_TEST_INITIAL_STATES=/path/to/initial_states.pkl` set). A live
native HTTP check matched the green mask to all 44 excavated cells in a
ten-action dig/unload tape, verified 44 units of accepted disposal, loaded-mask
suppression, exact undo, all listed case/mode resets, and native-state export.
These checks do not establish full-map solvability or browser acceptance;
the rendered browser interaction was not verified.

## Play and inspect

| Control | Manual play | Recorded playback |
| --- | --- | --- |
| Arrows / W A S D | Move base / turn or steer | Arrows select previous/next frame |
| Q / E | Turn cabin | — |
| Space | Dig, dump, or transfer (`DO`) | Play/pause |
| N | Wait one turn | — |
| R | Reset and start a new recording | — |
| H / T / F | Home / top / follow camera | Same |
| G | Rich lighting on/off (ambient occlusion, outlines) | Same |
| P | Paper figure style / stylized diorama | Same |
| Mouse drag / right drag / wheel | Orbit / pan / zoom | Same |

Buttons expose the same controls. The timeline can inspect earlier manual
snapshots; return to the latest frame to continue playing. Download a recording
before reset if you want to keep it. Terra's terminal state remains visible until
reset; the UI distinguishes success and timeout.

Layer switches show dig targets, designated dump targets, static no-dump areas,
current dumpability, interaction workspaces, and the cell grid. The interaction
map is Terra's union of active workspaces, not a promise that a `DO` action is
valid. Connected obstacle regions become large boulders or containers, with
their geometry kept within the blocked mask. These are visual styles, not
additional obstacle types in Terra. Clicking terrain or an obstacle reveals its
raw map values. Displayed heights and carried quantities are abstract soil
units, not calibrated cubic meters. Horizontal dimensions use the map's tile
size. Height exaggeration changes only the picture.

Three presentation styles share the same data. **Studio** (the default) is
meant for papers and talks that need motion: the grid stands as a block of
banded earth on a studio floor that shows only its shadow, under a soft
backdrop, with natural soil colors, coated machine paint, steel buckets,
ambient occlusion and no outlines. It hides the dig overlay on cells already cut
to target, so the excavation reveals itself, and adds translucent dust. **Paper**
is meant for static figures: a white background, the plain block, muted soil
colors, grey obstacles, a colorblind-safe (Okabe-Ito) overlay palette, static
patterns, number tags and rigid machine motion. **Diorama** puts the grid on a
stylized island with turf, trees, a fence, road, office and clouds under a sky
gradient, and adds dust/exhaust puffs and body sway. The surroundings lie
outside the Terra grid and carry no map meaning. P cycles the styles. In all
styles, cut walls show one stratum band per abstract soil unit, so cut depth
can be counted, and soil clods illustrate transfers without changing the
recording. The Machine tags switch hides the number tags and the active-machine
ring. Rich lighting adds ambient occlusion and soft outlines (none in studio);
the viewer drops to fast graphics once if the first frames are slow, G toggles
it, and both choices are remembered in the browser. Capture PNG renders the
scene at twice the viewport resolution (at least the screen's pixel density)
for print.
Reduced-motion system settings disable animation and effects.

Digging and dumping are planned on the cells each machine changed. An
excavator opens its bucket above the far edge of those cells, drags the teeth
along the new cut floor toward the cab while each cell drops as the bucket
passes, curls and lifts; it dumps by holding the bucket hinge above the deposit
and opening the bucket, and the pile grows as the soil lands. A small extra
slew centers the boom on the cells and returns before the step ends. A
skid-steer loader lowers its bucket, drives into the soil and backs out to its
recorded position, or drives up with the arms raised and tips the bucket.
Joint-round recordings show discrete endpoints: their within-round motion and
per-machine soil changes cannot be reconstructed from endpoint snapshots.
For sequential recorded actions, the arm angles come from a two-link
inverse kinematics solve; the motion is illustrative and the recorded state is
unchanged.

Positive soil is rendered as a joined, slope-limited mound surface. Single-cell
dumps stay low instead of becoming spikes; wider connected piles can rise.
Neighboring cells share their edges and borders taper to the ground inside the
exact positive-soil footprint. Raw quantities remain unchanged in recordings,
metrics, and the cell inspector. The slopes are display-only shapes, not
redistributed soil, literal center heights, or calibrated mesh volume. Zone overlays
and grid lines follow the mound surface. Empty and loaded excavator buckets
face inward and attach through a separate curl hinge; loader buckets face
forward. Machine motion remains illustrative.

## Replay and export

Use **Export JSON** to save manual play. **Open replay** reads a local JSON
recording; **Return to live session** reconnects to the current manual episode.
**Capture PNG** downloads the current scene at 2× resolution.

```bash
python -m terra.viewer3d --replay episode.json
python -m terra.viewer3d --replay episode.json.gz

# Generate a real example without starting the viewer service.
python -m terra.viewer3d --demo-replay --export example.json.gz

# Shared export for native recordings and postprocessed metric timelines.
terra-postprocess render episode.json.gz --out episode.html
terra-postprocess render episode.json.gz --out episode.mp4
```

Reading JSON/gzip and rendering saved recordings do not initialize JAX. WebGL2
is required to draw the scene. HTML opens offline without a server or network;
video needs the optional browser and encoder dependencies in the
[media workflow](../postprocess/README.md#install). Video holds exact recorded
endpoints for the selected duration, without interpolating motion. Full snapshots
make interactive seeking exact but use more RAM than a video; use one episode
per file and gzip for storage/transfer. The viewer's `--replay ... --export ...`
and `ReplayRecorder.save_html()` APIs remain available for native recordings.

## Record from a rollout

From a Python rollout, explicitly select one environment from a batch:

```python
from terra.viewer3d import ReplayRecorder

recording = ReplayRecorder(metadata={"title": "Policy episode", "source": "evaluation"})
recording.append(timestep, env_index=0)  # reset state

# During each step, keep the preceding actor's stable state-slot ID.
actor = int(timestep.state.agent.current_agent[0])
timestep = env.step_no_reset(timestep, actions, keys)
recording.append(timestep, action=int(action[0]), actor_id=actor, env_index=0)

recording.save("episode.json.gz")
recording.save_html("episode.html")
```

For an unbatched timestep omit `env_index`. For `[device, env, ...]` input use
`env_index=(device, env)`. Stop at the first terminal frame. Recording is a
host-side operation and should be used for selected evaluation episodes, not
every training environment inside a JIT loop.

Use the Terra and terra-baselines revisions paired with the checkpoint. The
evaluation adapter still owns model loading, observation construction,
recurrent state, reset/step RNG and action selection. `ReplayRecorder` records
the resulting native states; it is not a checkpoint loader. A newly captured
episode can differ from an earlier GPU evaluation even when the checkpoint and
seed match, so retain those outcomes separately. See the baselines
[inference guide](https://github.com/leggedrobotics/terra-baselines/blob/main/inference/README.md)
for the existing policy diagnostics.

For a joint-step runtime, record the full requested/effective transition and
the native reservation polygons explicitly:

```python
from terra.viewer3d import ReplayRecorder

recording = ReplayRecorder(metadata={"title": "Joint policy episode", "source": "evaluation"})
recording.append_joint(
    timestep,
    requested_actions=None,
    effective_actions=None,
    order=None,
    workspace_blocked=None,
    workspace_polygons=initial_native_polygons[world],
    env_index=world,
)

# Inside the matching evaluator, after one native joint step:
recording.append_joint(
    timestep,
    requested_actions=requested_actions,
    effective_actions=timestep.info["effective_actions"],
    order=execution_order,
    workspace_blocked=timestep.info["workspace_blocked"],
    workspace_polygons=current_native_polygons[world],
    env_index=world,
)
recording.save("joint_episode.json.gz")
```

The reset call is at native step zero. Transition vectors use stable agent
slots; they may be selected vectors or carry the timestep's batch axes, selected
with `env_index`. `execution_order` is the actual native order, not an inferred
agent sequence. Reservation polygons must already be selected for that world
and include the native body/work geometry of its active slots, for example
`{"id": 0, "component": "body", "vertices": [[x, y], ...]}`. Vertices use
native cell-edge grid coordinates without a half-cell shift. Missing guard
results, order or geometry must be supplied by the matching runtime adapter.

Joint endpoint recordings are sufficient for rendering. Fleet postprocessing
additionally needs exact ordered substeps from
`terra.postprocess.fleet.NativeFleetRecorder`; it cannot recover those substeps
from an endpoint delta. See the
[fleet workflow](../postprocess/README.md#two-excavators-or-an-excavator-and-skid).

## Joint recordings and postprocessed plans

Joint-round recordings can include `joint_actions`, `effective_joint_actions`,
`workspace_blocked` and native `workspace_polygons`. The viewer shows requested
and rejected actions for each stable machine slot. Solid work and dashed body
outlines use cell-edge coordinates. Joint frames remain discrete endpoints,
without inferred action attribution or interpolated work.

Use `terra-postprocess render RECORDING --out replay.html` (or `.mp4`/`.gif`)
for either native recordings or metric processed timelines. Fleet cleanup,
refined workspaces, comparison dashboards and galleries share this renderer;
their workflow lives
in the [postprocessing README](../postprocess/README.md). A native task outcome,
a processed-plan verdict and physical execution evidence are separate results.

## Development

```bash
cd terra/viewer3d/web
npm ci
npm run check
npm run build
```

The lockfile pins Three.js and esbuild. The build regenerates
`../static/viewer.js` and `../static/postprocessed.js`; include both bundles when sharing/installing
the Python package. Models are constructed in `web/models.js` from repository
code, with no downloaded meshes, textures, or remote fonts. Three.js's MIT
license is retained with the bundled assets. Existing Pygame rendering remains
available through its original commands.

Focused checks from the Terra root:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s terra/tests -p 'test_viewer3d_*.py' -v
```

The snapshot format is `terra.viewer3d.v1`. New checkpoint/model features belong
in terra-baselines; environment rules belong in Terra; display interpolation
belongs in the viewer. Raw world maps and stable state-slot IDs are the boundary.
