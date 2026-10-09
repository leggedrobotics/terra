# Terra 3D viewer

Date: 2026-09-07. This specification precedes implementation.

## Purpose

Make Terra's discrete earthwork decisions easy to understand during manual play,
policy replay, and map inspection. Show the machine moving, turning its cabin,
lifting soil, and placing soil in a compact, readable 3D construction scene.
Terra remains the sole owner of transitions, rewards, legality, and termination.
Arm motion and particles illustrate a workspace action; they do not represent
physical bucket trajectories, timing, or soil mechanics.

## Deliverable and visual direction

A local browser viewer served by Python from `terra.viewer3d`, with a bundled
Three.js renderer. Use an earth-colored diorama, stepped soil surfaces and cut
walls, soft directional shadows, an articulated yellow excavator with tracks,
cab, counterweight, boom, stick, hydraulic rods and bucket. Trucks and skid
steers have distinct simplified models. Models are repository-owned procedural
geometry, so there is no external asset download or uncertain asset license.

The visual target is a stylized low-poly construction game: chunky machine
silhouettes, warm soil, faceted rock obstacles, distinct colored zones, and
readable motion. Surface detail should make cuts, piles, and carried soil clear
at ordinary play zoom. Keep obstacle graphics inside the cells Terra marks as
occupied. Favor consistent shapes, colors, and lighting over physical realism.

Revision 2026-09-27 adds two presentation styles over the same scene. The
*paper* style (then the default) targets journal figures: white background, the grid on a
plain banded-earth block (a block diagram), muted soil, grey obstacles,
Okabe-Ito overlays with static patterns and outlines, plain number tags,
neutral tone mapping and rigid machine motion, with 2× PNG capture. The
*diorama* style, inspired by Islanders, Townscaper and Tiny Glade, places the
grid on a floating island with turf, trees, a fence, road, site office and
clouds, adds dust and exhaust puffs, beacons, body sway and a vignette. Both
use optional ambient occlusion and soft outlines, one stratum band per soil
unit on cut walls, animated track shoes and wheels, and keyframed work actions
(anticipation, action, follow-through) with soil clods. These are illustrations
only; decoration never enters a Terra cell.

Revision 2026-10-05 adds a default *studio* style for papers and talks that
need motion: the plain earth block on a shadow-only studio floor under a soft
backdrop, natural soil colors with fine grain, coated machine paint, steel
buckets, ambient occlusion without outlines, translucent dust and the dig
overlay hidden on cells already cut to target. The excavator bucket is rebuilt
as a curved back plate, side plates, wear straps, a steel cutting edge with
teeth and pinned brackets; the loader carries a general-purpose bucket. Dig and
dump motions are now planned on the changed cells with inverse kinematics:
changed cells follow the bucket instead of easing together, and joint frames
animate every machine that worked.

Keep most of the viewport for the scene. Use compact controls for the current
machine, soil carried, step, reward, episode outcome, and selected cell. Dig
targets are orange, designated dump targets teal, obstacles slate, and the
interaction workspaces yellow. Every overlay can be toggled. Task
overlays follow the actual terrain surface; turning them off leaves the terrain
and machines visible. Static prohibited dumping and current dumpability are
distinct layers. Show a legend and explain the abstract height units.

Orbit, pan, zoom, top view, isometric home view, machine follow, and height
exaggeration are display settings. Replay has play/pause, previous/next,
timeline seek, speed, and local JSON import/export. Manual play offers buttons
and keyboard actions, reset, and download of the played episode. A bundled
playable example needs neither a training dataset nor a checkpoint. An optional
map path uses the existing Terra dataset loader. A PNG export captures the scene.

## Architecture

```text
Terra State / TimeStep -> snapshot adapter -> ReplayRecorder -> JSON / HTML
        ^                       |                              |
        |                       +-> local HTTP server -------->+-> 3D viewer
 keyboard / buttons -> manual session -> Terra.step_no_reset
terra-baselines checkpoint -> single-episode inference -> ReplayRecorder
```

The renderer is opt-in. Existing Pygame rendering and training entry points keep
their current defaults. Browser code has no JAX, checkpoint loader, or alternate
implementation of dig/dump rules. Python serializes selected unbatched states
on the host, outside compiled training loops. Only explicit recording introduces
device-to-host transfers. Replay files contain data, never Python pickle/code.
Checkpoint loading stays in terra-baselines under its existing trusted-input
contract.

Three.js uses instanced terrain/overlay meshes to keep draw calls small. See
[InstancedMesh](https://threejs.org/docs/pages/InstancedMesh.html) and
[OrbitControls](https://threejs.org/docs/pages/OrbitControls.html).
The JavaScript bundle, styles, and models are shipped with the Python package;
normal use needs no CDN, npm, or network connection. npm is a development-only
build tool. The service binds to loopback and serves only its own static assets
and selected session. It must reject malformed actions, serialize state updates,
reject cross-origin mutations, and preserve terminal states until explicit reset.

## Snapshot contract: `terra.viewer3d.v1`

An episode is `{schema, metadata, frames}`. Metadata includes a title and source;
callers may add checkpoint, seed, map, and revision information. Frames are full
snapshots for direct seeking and simple inspection. JSON gzip is supported for
long recordings. Bound the supported scene to 128 x 128 cells and validate shape,
finite values, active IDs, required fields, and version before rendering.

Each frame contains:

- `step`, `action` (null at reset, otherwise 0..7), `actor_id` (the preceding
  acting agent), `current_agent`, `reward`, `done`, and `task_done`.
- `grid`: `rows`, `cols`, `tile_size_m`. Heights are integer soil units, with a
  separate viewer-only exaggeration control; horizontal scale uses tile size.
- `maps`: `action`, `target`, `padding`, `dumpability`, `dumpability_static`,
  `interaction`, `traversability`; each is a rows-by-cols array. Preserve raw
  positive pile heights rather than clipped policy features. Optional unavailable
  diagnostic layers are null, not invented all-zero masks. Terra's interaction
  mask is the union of active agents' workspaces; label it accordingly.
- `agents`: stable original-slot `id`, `type` (0 excavator, 1 truck, 2 skid
  steer), `action_type` (0 tracked, 1 wheeled), `position` as `[row, column]`,
  `base_yaw` and relative `cabin_yaw` in radians, footprint `width` (across
  local forward) and `height` (along local forward) in cells, integer `loaded`,
  `wheel_angle`, `shovel_lifted`, and `reach` as `[inner, outer]` in cells.

Read agent states from `state.agent.agent_states` and filter `agent_active`;
never use the acting-first observation rows as persistent IDs. Grid coordinates
are the existing array indexing: `action[row][column]`, `pos_base=[row,column]`.
For the scene, X is column, Y is elevation, Z is row. Center tile coordinates
with a half-cell offset. With the machine modeled facing +X, scene rotation
about Y is `2*pi*angle_base/angles_base`; cabin rotation is relative to the
base. At zero heading Terra moves toward increasing column. This comes from
the transition code, not the older four-heading AgentState docstring.

## Motion and truthfulness

Interpolate machine position and rotations using the shortest angular path.
Animate a dig only when terrain decreases and the actor gains material; animate
a dump when material leaves the actor and terrain increases. Handle direct
agent-to-agent transfer by load changes with no terrain change. Terrain changes
are always derived from consecutive recorded maps, including collapse effects.
Represent failed actions with an unchanged scene and an explicit no-effect
message. Seeking, reset, loading another recording, and crossing episode
boundaries snap to the chosen snapshot and clear transient animation.

All displayed metrics come from the selected frame. Excavated fraction may be a
descriptive terrain statistic, but it must not be labeled task success. Only
Terra's terminal `task_done` determines the outcome; timeout is separate.

## Integration

`python -m terra.viewer3d` starts the built-in manual scene. `--map` selects a
Terra single-map path. `--replay` opens an existing recording without initializing
JAX. `--no-open` prints a URL for manual opening or SSH forwarding. The server
initializes/warmups the environment before accepting play actions.

Python callers use `ReplayRecorder(metadata=...)`, `append(timestep,
action=..., actor_id=..., env_index=...)`, and `save(path)` / `save_html(path)`.
The adapter requires an explicit index for batched input and handles an
unbatched `TimeStep` directly. Recorded arrays are detached copies.

The baseline integration exports one initial episode and uses `step_no_reset`.
It reuses existing checkpoint model/config preprocessing. Existing metrics and
GIF defaults must not be silently changed to support the viewer. Unsupported
checkpoint formats receive the existing compatibility error, not guessed
architecture or config values. Full-state 3D export from old diagnostic traces
is possible only if those traces actually contain the required terrain states.

## Acceptance

1. Record real Terra reset, motion, rotation, dig, dump, and terminal states;
   assert stable agent identity, array orientation, raw heights, and unchanged
   underlying state. Check saved/reloaded snapshots and malformed replay errors.
2. Exercise manual actions through HTTP against Terra on CPU, including invalid
   action input, reset, replay download, and refusal to step a terminal episode.
3. Build the bundled frontend. Inspect the live scene and a replay: terrain,
   machine geometry, colors, overlay toggles, camera, timeline, and load feedback.
   Check narrow-window layout and visible errors when graphics/data fail.
4. Run focused existing environment tests and relevant baseline export tests.
   This visualization task does not launch training or alter environment rules.
5. Document launch, replay export, controls, asset/build provenance, verified
   capabilities, and current limits. Preserve pre-existing changes in both repos.

## Later extensions

External glTF machine assets, multi-episode comparisons, video encoding, live
training dashboards, and ROS/Newton visualization can build on this snapshot
interface. They are separate work after the local manual/replay viewer is usable.

## Visual refinement: machine attachment and grouped terrain props

The second visual pass keeps the miniature construction-game style and all
recorded state unchanged:

- The excavator bucket faces inward after a 180-degree yaw correction. Its
  curl remains a separate hinged rotation; the skid-steer bucket still faces
  forward. Add a visible attachment pin and connected linkage, and bevel the
  major machine panels without rounding away the silhouette.
- Bucket proportion correction: reduce the excavator bowl, teeth, mounting
  ears, payload, and local linkage to 65% of their first-refinement size.
  Keep the stick-end hinge fixed and retain stick-side mount widths. The
  ear spacing must clear the unchanged stick-eye housing, and both mounting
  pins must span the ears even though the bowl is smaller. The
  resulting bucket is roughly one quarter of the machine footprint width;
  this is an art-direction ratio, not a physical capacity calibration. Loader
  buckets and Terra's soil capacity remain unchanged.
- Connected obstacle masks become a few large boulders or corrugated
  containers. Fit every prop within a fully blocked rectangular portion of
  the mask. Preserve holes, narrow gaps, disconnected components, and irregular
  boundaries; a visual bounding box must never bridge traversable cells.
- Positive terrain becomes a connected, triangulated soil surface with shared
  edges and tapered borders. Limit display slopes to 1.6 soil-height units per
  horizontal cell before height exaggeration. A separable slope envelope lowers
  unsupported peaks: single-cell dumps remain low, while broad connected piles
  can rise. Keep the exact positive footprint, including holes and obstacle
  boundaries. Dig holes keep their stepped cut walls. Raw soil quantities remain
  unchanged in snapshots, metrics, and the cell inspector; the mound is not a
  calibrated volume surface or literal per-cell height plot.
- Overlays and cell grid lines follow the mound surface. Soil geometry grows
  and disappears by interpolating the capped endpoint profiles, bounded by
  the current positive support during transitions. Large single-cell deposits
  therefore grow smoothly instead of reaching the display cap immediately.
  Ray picking continues to report the original underlying grid cell.
- Machine elevation, selection highlighting, and soil-particle landing points
  use the same display surface, so reducing a spike does not leave floating
  highlights, machines, or particles at the old raw height.

Verification uses real existing reset, loaded, and dumped snapshots, plus
synthetic geometry-only edge cases for obstacle holes and soil boundaries.
Check the bucket empty and loaded from side/top views, hinge attachment over
the full animation, grouped-obstacle containment, and replay seek/reset with
height exaggeration and layers. Do not reset an existing user's live episode
to perform these display checks.
