# Viewer frontend development

The normal Python viewer serves the checked-in `../static/` bundle. Node and
npm are only needed when changing the frontend:

```sh
cd terra/viewer3d/web
npm ci
npm run check
npm run build
```

Three.js 0.185.1 and esbuild 0.28.2 are pinned in `package-lock.json`. The bundled
renderer has no CDN or other network dependencies. The full Three.js MIT notice
is included in the JavaScript, including self-contained HTML exports, and in
`../static/THIRD_PARTY_NOTICES.txt`.

- `app.js`: manual HTTP requests, replay playback/import/export, controls, and UI.
- `data.js`: snapshot validation and facts calculated from recorded changes.
- `scene.js`: instanced soil and layers, cameras, picking, recorded animation
  and its timed display events (clods, puffs), the paper/diorama looks and
  print-resolution capture.
- `models.js`: original procedural tracked/wheeled excavators, trucks, skid
  steers, with animated treads, wheels, beacons and choreographed work poses.
- `materials.js`: world-space soil strata, turf, loose-soil and zone-pattern shaders.
- `environment.js`: the paper block, or the diorama island with fence, trees,
  road, office and clouds, all outside the grid (checked by `environment.test.js`).
- `effects.js`: pooled soil clods and dust/exhaust puffs.
- `post.js`: optional ambient occlusion, soft outlines and (diorama) vignette.
- `../static/index.html` and `style.css`: the local viewer layout and styles.

The browser never decides whether an action is legal and never advances Terra.
`/api/action` returns the authoritative next frame. Missing diagnostic layers
are disabled. Recorded `wheel_angle` is a discrete steering index; the model
illustrates its sign using a bounded display angle. Soil heights are abstract
units scaled vertically by the display slider. Work particles and articulated
arm motion illustrate recorded load/terrain changes and are not soil physics.

Snapshots support grids up to 128 × 128 and 1–4 stable agent slots. Replay data
is held in browser memory; JSON imports are bounded to 256 MB and 100,000 frames.
Larger recordings can be selected with Python `--replay`, subject to the same
browser memory limit. JSON gzip is handled by Python, not the local file picker.
WebGL2 is required. PNG capture exports the rendered scene, without HTML panels.
`window.terraViewer` exposes the scene and `show(index, options)` for scripted
captures and debugging; the viewer itself does not use it.

## Postprocessed metric plans

`postprocessed.js` provides `PostprocessedEpisode` for the separate
`terra.postprocessed.v1` playback format produced by
[`terra.postprocess`](../../postprocess/README.md). `plan-view.js` builds the
standalone player and dashboard adapter with this same studio scene. `npm run
build` updates both checked-in bundles; normal HTML export uses those packaged
assets without Node or a viewer checkout. The native-replay file picker still
uses `terra.viewer3d.v1`.

The Terra postprocessing README owns dashboard export, standalone playback and
the runnable two-excavator example. Saved traces need no policy inference.
Capturing new joint-action traces requires the matching native Terra runtime;
the viewer does not provide those APIs. Joint-round recordings preserve stable
machine slots, requested/effective actions and rejection flags. Their stationary
reservation outlines use native cell-edge coordinates, and playback shows exact
endpoints without interpolating joint motion or assigning soil to one machine.

Metric playback retains the original grid (up to 1,048,576 cells), separate native
and loose soil heights in metres, exact route samples, stable agent IDs and
explicit ownership of every changed cell. Array rows advance plan Y and columns
plan X. `origin_xy_m` is the map position of cell `[0,0]`'s centre; optional grid
`yaw_rad` rotates these axes into map coordinates. Machine and cabin headings
are absolute map headings, and steering is supplied in radians. Reverse seeking
reconstructs the same terrain without quantizing heights or using uint16 indices.

`metric-terrain.js` merges exactly coplanar cell faces and exposed wall spans;
it never resamples the grid. Picking still returns individual grid cells.
Metric motion interpolates only saved route segments with bounded heading
interpolation. Explicit `relocate` frames snap, making missing routes visible.
Workspace, reservation and route overlays belong to `postprocessed-view.js`.
Arm animation and display durations remain illustrative. Rendering a rejected
proposal does not change its validation status or establish native/physical
execution.
