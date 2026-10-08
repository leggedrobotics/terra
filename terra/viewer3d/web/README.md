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
