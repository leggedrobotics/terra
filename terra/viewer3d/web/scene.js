import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { makeMachine, SLOT_COLORS } from './models.js';
import { shortestAngle, transitionFacts, diggingView, isJointFrame, workspaceVertexToWorld } from './data.js';
import { SoilPiles } from './piles.js';
import { createObstacleProps } from './obstacles.js';
import { createEnvironment } from './environment.js';
import { metricTerrainGeometry } from './metric-terrain.js';
import { Effects } from './effects.js';
import { PostPipeline } from './post.js';
import { PALETTE, PALETTES, earthMaterial, shared, skyTexture, studioBackdrop, zoneMaterial } from './materials.js';

const LAYERS = {
  dig: { color: 0xe69f00, opacity: .55, map: 'target', pattern: 'hatch', test: v => v < 0 },
  dump: { color: 0x009e73, opacity: .5, map: 'target', pattern: 'dots', test: v => v > 0 },
  restricted: { color: 0xd55e00, opacity: .42, map: 'dumpability_static', pattern: 'cross', test: v => !v },
  dumpability: { color: 0x0072b2, opacity: .28, map: 'dumpability', pattern: 'solid', test: v => !!v },
  interaction: { color: 0x56b4e9, opacity: .20, map: '_workspace', pattern: 'solid', test: v => !!v },
  footprint: { color: 0x274c59, opacity: .12, map: 'footprint', pattern: 'solid', test: v => !!v },
  precision: { color: 0x7553b5, opacity: .08, map: 'precision_required_band', pattern: 'solid', test: v => !!v },
  eligibleNow: { color: 0x009b68, opacity: .80, map: '_digging', pattern: 'solid', test: v => v === 1 },
  eligibleSwing: { color: 0xe4a317, opacity: .82, map: '_digging', pattern: 'dots', test: v => v === 2 },
  eligibleBlocked: { color: 0xc74d45, opacity: .58, map: '_digging', pattern: 'hatch', test: v => v === 3 },
  eligibleLoaded: { color: 0x929ba0, opacity: .85, map: '_digging', pattern: 'solid', test: v => v === 4 },
};
const ELIGIBILITY_LAYERS = ['eligibleNow', 'eligibleSwing', 'eligibleBlocked', 'eligibleLoaded'];
const matrix = new THREE.Matrix4(), dummy = new THREE.Object3D(), color = new THREE.Color();
const smooth = t => t * t * (3 - 2 * t);
const easeInOut = t => t < .5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
const backOut = t => { const c = 1.4; return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2; };
const mix = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const WORK_KINDS = ['dig', 'dump', 'transfer'];
const PRESENTATIONS = ['studio', 'paper', 'diorama'];
// Studio zones: a static hatch for the cut and a light tint for disposal.
const STUDIO_LAYERS = { dig: { color: 0xd98a1c, opacity: .42 }, dump: { color: 0x2f8f6a, opacity: .2, pattern: 'solid' } };
const layersFor = presentation => presentation === 'studio' ? Object.fromEntries(Object.entries(LAYERS).map(([name, settings]) => [name, { ...settings, ...STUDIO_LAYERS[name] }])) : LAYERS;
const QUALITY_KEY = 'terra-viewer3d-quality', PRESENTATION_KEY = 'terra-viewer3d-presentation';
function disposeProps(group) {
  if (!group) return;
  const geometries = new Set(), materials = new Set();
  group.traverse(item => { if (item.geometry) geometries.add(item.geometry); if (item.material) for (const material of Array.isArray(item.material) ? item.material : [item.material]) materials.add(material); });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
  group.removeFromParent(); group.clear();
}

// Metric forecasts already contain their pile model. Their columns render the
// supplied surface directly; none of Terra's illustrative pile capping applies.
class MetricColumns extends THREE.Group {
  update() {}
  setLayer() {}
}
function stored(key) { try { return localStorage.getItem(key); } catch { return null; } }
function store(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } }

export class TerraScene {
  constructor(element, { onPick, onCameraChange, onError, onQualityChange } = {}) {
    this.element = element; this.onPick = onPick; this.onCameraChange = onCameraChange; this.onError = onError; this.onQualityChange = onQualityChange;
    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.pixelRatio = Math.min(window.devicePixelRatio || 1, 2); this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.shadowMap.enabled = true; this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.appendChild(this.renderer.domElement);
    this.renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); this.onError?.(new Error('The graphics context was lost. Reload the viewer to reconnect to the scene. Your live episode remains on the server.')); });
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), .04).texture; this.scene.environmentIntensity = .28; pmrem.dispose();
    this.camera = new THREE.PerspectiveCamera(32, 1, .1, 2000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement); this.controls.enableDamping = true; this.controls.dampingFactor = .085; this.controls.maxPolarAngle = Math.PI * .47; this.controls.minPolarAngle = .001; this.controls.screenSpacePanning = true;
    this.controls.addEventListener('start', () => { this.tween = null; if (this.follow) { this.follow = false; this.onCameraChange?.({ follow: false }); } });
    this.hemi = new THREE.HemisphereLight(0xcfe4ff, 0x9a7650, .8); this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffe9c9, 2.7); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048); this.sun.shadow.bias = -.0004; this.sun.shadow.radius = 3; this.scene.add(this.sun); this.scene.add(this.sun.target);
    this.fill = new THREE.DirectionalLight(0xa9c9ff, .35); this.scene.add(this.fill);
    this.world = new THREE.Group(); this.scene.add(this.world); this.machines = new Map(); this.heightScale = 1;
    this.visibility = { dig: true, dump: true, restricted: false, dumpability: false, interaction: true, workspace: true, footprint: true, precision: true, eligibility: true, eligibleNow: true, eligibleSwing: true, eligibleBlocked: true, eligibleLoaded: true, grid: false, tags: true };
    this.raycaster = new THREE.Raycaster(); this.pointer = new THREE.Vector2(); this.selected = null;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.effects = new Effects({ groundHeight: (x, z) => this.groundAt(x, z) }); this.scene.add(this.effects);
    const quality = stored(QUALITY_KEY); this.quality = quality === 'fast' ? 'fast' : 'high';
    // Without a stored choice, fall back once to fast graphics on slow GPUs.
    this.perf = quality ? null : { frames: 0, elapsed: 0 };
    const presentation = stored(PRESENTATION_KEY); this.presentation = PRESENTATIONS.includes(presentation) ? presentation : 'studio';
    try { this.post = new PostPipeline(this.renderer, this.scene, this.camera); } catch (error) { console.warn('Post-processing unavailable', error); this.post = null; this.quality = 'fast'; }
    this.lineMaterials = new Set(); this.applyLook();
    let pointerStart = null;
    element.addEventListener('pointerdown', event => { pointerStart = { x: event.clientX, y: event.clientY, button: event.button }; });
    element.addEventListener('pointerup', event => { if (pointerStart?.button === 0 && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) < 5) this.pick(event); pointerStart = null; });
    this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(element); this.resize();
    this.clock = { last: performance.now(), idle: 0 };
    this.renderer.setAnimationLoop(time => { this.update(time); this.controls.update(); this.render(); });
  }

  render() { if (this.quality === 'high' && this.post) this.post.render(); else this.renderer.render(this.scene, this.camera); }
  measure(dt) {
    if (!this.perf || !this.frame || this.quality !== 'high' || document.hidden) return;
    // Skip the first frames (shader compilation), then average about 2.5 s.
    if (++this.perf.frames > 20) this.perf.elapsed += dt;
    if (this.perf.frames < 20 + 90) return;
    const average = this.perf.elapsed / (this.perf.frames - 20); this.perf = null;
    if (average > 1 / 24) { this.quality = 'fast'; this.onQualityChange?.('fast'); }
  }
  setQuality(value) { this.quality = value === 'fast' || !this.post ? 'fast' : 'high'; store(QUALITY_KEY, this.quality); return this.quality; }
  /** 'studio': the grid as an earth block on a studio floor with realistic
   * materials; 'paper': neutral figure style on white; 'diorama': the stylized island. */
  applyLook() {
    const look = this.presentation, paper = look === 'paper', studio = look === 'studio', diorama = look === 'diorama'; this.palette = PALETTES[look];
    // An HDR white clear color tone-maps to (near) pure white in the rich pipeline.
    this.scene.background = paper ? new THREE.Color(12, 12, 12) : studio ? (this.backdrop ||= studioBackdrop()) : (this.sky ||= skyTexture());
    this.scene.fog = diorama && this.span ? new THREE.Fog(new THREE.Color(PALETTE.sky[1]), this.span * 3.2, this.span * 7.5) : null;
    this.renderer.toneMapping = paper ? THREE.NeutralToneMapping : THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1;
    this.scene.environmentIntensity = studio ? .5 : .28;
    this.hemi.color.set(paper ? 0xffffff : studio ? 0xe1eaf3 : 0xcfe4ff); this.hemi.groundColor.set(paper ? 0x8d8880 : studio ? 0x7a6a58 : 0x9a7650); this.hemi.intensity = paper ? .9 : studio ? .6 : .8;
    this.sun.color.set(paper ? 0xffffff : studio ? 0xfff0dc : 0xffe9c9); this.sun.intensity = paper ? 2.3 : studio ? 2.9 : 2.7; this.sun.shadow.radius = studio ? 5 : 3;
    this.fill.color.set(paper ? 0xffffff : studio ? 0xc6d9f5 : 0xa9c9ff); this.fill.intensity = paper ? .45 : studio ? .4 : .35;
    this.effects.puffsEnabled = diorama; this.effects.dustEnabled = studio; this.effects.palette = this.palette.clods; shared.uMotion.value = diorama ? 1 : 0;
    this.post?.setLook({ vignette: diorama ? .16 : studio ? .12 : 0, ink: studio ? 0 : .92 });
    if (this.element.ownerDocument?.body) this.element.ownerDocument.body.dataset.presentation = this.presentation;
  }
  setPresentation(value) {
    this.presentation = PRESENTATIONS.includes(value) ? value : 'studio'; store(PRESENTATION_KEY, this.presentation);
    this.applyLook();
    if (this.frame) {
      // Rebuild materials, surroundings and machines for the new look; keep the view.
      const position = this.camera.position.clone(), target = this.controls.target.clone();
      this.setFrame(this.frame, { reset: true });
      this.tween = null; this.camera.position.copy(position); this.controls.target.copy(target); this.controls.update();
    }
    return this.presentation;
  }

  resize() {
    const width = this.element.clientWidth || 1, height = this.element.clientHeight || 1;
    this.renderer.setSize(width, height, false); this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    this.post?.setSize(width, height, this.pixelRatio);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2()); this.effects.setViewport(size.y, this.camera.fov);
    for (const material of this.lineMaterials ?? []) material.resolution.copy(size);
  }
  point(row, col, height = 0) { const { rows, cols, tile_size_m: tile } = this.frame.grid; return new THREE.Vector3((col + .5 - cols / 2) * tile, height * this.unitHeight, (row + .5 - rows / 2) * tile * (this.frame.metric ? -1 : 1)); }
  heightAt(row, col, frame = this.frame) {
    row = clamp(Math.round(row), 0, frame.grid.rows - 1); col = clamp(Math.round(col), 0, frame.grid.cols - 1);
    const height = frame.maps.action[row][col];
    return !frame.metric && height > 0 && !frame.maps.padding[row][col]
      ? this.piles.endpointHeight(row, col, frame === this.piles.previous)
      : height * this.unitHeight;
  }
  displayHeightAt(row, col) {
    row = clamp(Math.round(row), 0, this.frame.grid.rows - 1); col = clamp(Math.round(col), 0, this.frame.grid.cols - 1);
    const height = this.displayHeights?.[row]?.[col] ?? this.frame.maps.action[row][col];
    return !this.frame.metric && height > 0 && !this.frame.maps.padding[row][col] ? this.piles.nodeHeight(row * 2 + 1, col * 2 + 1) : height * this.unitHeight;
  }
  surfacePoint(row, col) { const point = this.point(row, col); point.y = this.displayHeightAt(row, col); return point; }
  /** Displayed surface height at a world position (0 on the surrounding turf). */
  groundAt(x, z) {
    if (!this.frame) return 0;
    const { rows, cols, tile_size_m: tile } = this.frame.grid, col = Math.floor(x / tile + cols / 2), row = Math.floor((this.frame.metric ? -z : z) / tile + rows / 2);
    if (row < 0 || col < 0 || row >= rows || col >= cols) return 0;
    if (this.frame.maps.padding[row][col]) return this.obstacleTop ?? 0;
    return this.displayHeightAt(row, col);
  }

  buildWorld(frame) {
    if (this.terrain) this.disposeWorld();
    const { rows, cols, tile_size_m: tile } = frame.grid, count = rows * cols;
    this.span = Math.max(rows, cols) * tile; shared.uTile.value = tile;
    this.camera.near = Math.max(tile * .025, this.span / 200); this.camera.far = this.span * 20; this.camera.updateProjectionMatrix();
    this.controls.minDistance = Math.max(tile * 2, this.span * .08); this.controls.maxDistance = this.span * 4.5;
    this.applyLook();
    const reach = this.span * .5 + THREE.MathUtils.clamp(this.span * .2, 6, 22) + 2;
    this.sun.position.set(-this.span * .75, this.span * 1.35, -this.span * .45);
    Object.assign(this.sun.shadow.camera, { left: -reach, right: reach, top: reach, bottom: -reach, near: .1, far: this.span * 4 }); this.sun.shadow.camera.updateProjectionMatrix();
    this.sun.shadow.normalBias = tile * .04; this.fill.position.set(this.span * .8, this.span * .6, this.span * .9);
    this.post?.configure({ span: this.span, tile });
    // Adjacent columns share coplanar side faces. Bias those faces behind the
    // surface so float precision cannot produce dotted seams across flat soil.
    const sides = earthMaterial('soil', { vertexColors: !!frame.metric, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 2 }, this.palette);
    const top = earthMaterial('soil', { vertexColors: !!frame.metric, color: 0xffffff, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }, this.palette);
    const bottom = new THREE.MeshStandardMaterial({ color: 0x8c7153, roughness: 1 });
    // Front faces cast, so cut walls shade trenches without the slab self-shadowing.
    for (const material of [sides, top, bottom]) material.shadowSide = THREE.FrontSide;
    this.terrain = frame.metric ? new THREE.Mesh(new THREE.BufferGeometry(), [top, sides]) : new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), [sides, sides, top, bottom, sides, sides], count);
    if (frame.metric) bottom.dispose(); else this.terrain.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.terrain.castShadow = true; this.terrain.receiveShadow = true; this.world.add(this.terrain);
    this.environment = createEnvironment(frame, { style: this.presentation }); this.world.add(this.environment);
    this.layers = {}; this.boundaries = {}; this.boundaryEntries = new Map();
    this.layerSettings = layersFor(this.presentation);
    for (const [name, settings] of Object.entries(this.layerSettings)) {
      const geometry = new THREE.PlaneGeometry(1, 1); geometry.rotateX(-Math.PI / 2);
      const material = zoneMaterial({ color: settings.color, opacity: settings.opacity, pattern: settings.pattern, polygonOffset: true, polygonOffsetFactor: -2 });
      let capacity = count, slots = null;
      if (frame.metric) {
        slots = new Int32Array(count).fill(-1); capacity = 0;
        const map = this.layerMaps[settings.map];
        if (map != null) for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) if (!frame.maps.padding[row][col] && settings.test(map[row][col])) slots[row * cols + col] = capacity++;
      }
      const layer = new THREE.InstancedMesh(geometry, material, capacity); layer.userData.cellSlots = slots; layer.instanceMatrix.setUsage(THREE.DynamicDrawUsage); layer.visible = this.visibility[name]; layer.renderOrder = 3 + Object.keys(this.layers).length; layer.frustumCulled = false; layer.userData.skipAO = true; this.layers[name] = layer; this.world.add(layer);
      if (name === 'dig' || name === 'dump' || name === 'interaction' || name === 'precision' || name === 'footprint') {
        const material = new LineMaterial({ color: new THREE.Color(settings.color).multiplyScalar(name === 'precision' ? 1 : .82), linewidth: name === 'precision' ? 3.5 : 2.6, transparent: true, opacity: .95, depthWrite: false });
        material.resolution.copy(this.renderer.getDrawingBufferSize(new THREE.Vector2())); this.lineMaterials.add(material);
        const line = new LineSegments2(new LineSegmentsGeometry(), material);
        line.renderOrder = name === 'precision' ? 18 : 14; line.visible = this.visibility[name]; line.frustumCulled = false; this.boundaries[name] = line; this.world.add(line);
      }
    }
    this.gridLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x6b5132, transparent: true, opacity: .22, depthWrite: false })); this.gridLines.renderOrder = 10; this.gridLines.visible = this.visibility.grid; this.gridLines.frustumCulled = false; this.world.add(this.gridLines);
    this.selection = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(tile * .99, tile * .04, tile * .99)), new THREE.LineBasicMaterial({ color: 0xfffbe0, depthTest: false })); this.selection.renderOrder = 20; this.selection.visible = false; this.world.add(this.selection);
  }

  disposeWorld() {
    this.clearMotion(); this.effects.clear();
    for (const machine of this.machines.values()) { this.scene.remove(machine.root); machine.dispose(); } this.machines.clear();
    const geometries = new Set(), materials = new Set();
    this.world.traverse(item => { if (item.geometry) geometries.add(item.geometry); if (item.material) for (const material of Array.isArray(item.material) ? item.material : [item.material]) materials.add(material); });
    for (const geometry of geometries) geometry.dispose(); for (const material of materials) { material.map?.dispose(); material.dispose(); } this.world.clear(); this.selected = null; this.piles = null; this.obstacleProps = null; this.environment = null;
    this.lineMaterials.clear();
  }

  setFrame(frame, { animate = false, duration = 650, reset = false } = {}) {
    const previous = this.frame, previousUnitHeight = this.unitHeight;
    // A playback timer can advance just before the final animation frame. Finish
    // metric cuts before reusing their terrain through a long sequence of poses.
    if (previous?.metric && this.motion?.facts.changed.length) { this.setFloor(this.finalFloor); this.populate(previous); }
    const dimensionsChanged = !previous || !!previous.metric !== !!frame.metric || previous.grid.rows !== frame.grid.rows || previous.grid.cols !== frame.grid.cols || previous.grid.tile_size_m !== frame.grid.tile_size_m;
    this.clearMotion(); this.frame = frame; this.unitHeight = (frame.metric ? 1 : frame.grid.tile_size_m * .48) * this.heightScale; shared.uUnit.value = this.unitHeight;
    // Native inspector masks are absent from metric plans; avoid scanning their large grids.
    const digging = frame.maps.fresh_dig_current != null && frame.maps.fresh_dig_swing != null ? diggingView(frame) : null;
    this.layerMaps = { ...frame.maps, _workspace: frame.maps.work_cone ?? frame.maps.interaction, _digging: digging?.cells ?? null };
    if (dimensionsChanged || reset) this.buildWorld(frame);
    const facts = transitionFacts(previous, frame);
    const mayAnimate = !isJointFrame(frame) && animate && !reset && !dimensionsChanged && !this.reducedMotion && previous && !previous.done && frame.step === previous.step + 1;
    let lowest = 0;
    for (const row of frame.maps.action) for (const cell of row) lowest = Math.min(lowest, cell);
    this.finalFloor = lowest * this.unitHeight - frame.grid.tile_size_m * .85;
    if (mayAnimate) for (const row of previous.maps.action) for (const cell of row) lowest = Math.min(lowest, cell);
    this.setFloor(lowest * this.unitHeight - frame.grid.tile_size_m * .85);
    if (!frame.metric || dimensionsChanged || reset || previousUnitHeight !== this.unitHeight || !this.obstacleProps) {
      disposeProps(this.obstacleProps);
      this.obstacleProps = createObstacleProps(frame, { unitHeight: this.unitHeight, style: this.presentation }); this.world.add(this.obstacleProps);
      if (frame.metric) this.obstacleProps.scale.z = -1;
    }
    const unchangedTerrain = frame.metric && previous?.metric && !dimensionsChanged && !reset && previousUnitHeight === this.unitHeight && !facts.changed.length && !frame.metric.terrain_changes.length && previous.metric.native.every((row, i) => row === frame.metric.native[i]) && previous.metric.loose.every((row, i) => row === frame.metric.loose[i]);
    if (!unchangedTerrain) {
      disposeProps(this.piles);
      this.piles = frame.metric ? new MetricColumns() : new SoilPiles({ ...frame, maps: this.layerMaps }, { previous: mayAnimate ? previous : null, unitHeight: this.unitHeight, layerSettings: this.layerSettings, visibility: this.visibility, palette: this.palette, roughness: this.presentation === 'studio' ? .16 : 0 });
      this.world.add(this.piles); this.piles.update(mayAnimate ? 0 : 1);
      this.populate(frame);
    }
    this.populateReservations(frame);
    if (isJointFrame(frame) && this.follow) this.setFollow(false);
    const liveIds = new Set(frame.agents.map(agent => agent.id));
    for (const [id, machine] of this.machines) if (!liveIds.has(id)) { this.scene.remove(machine.root); machine.dispose(); this.machines.delete(id); }
    for (const agent of frame.agents) {
      let machine = this.machines.get(agent.id);
      if (machine && (machine.agent.type !== agent.type || machine.agent.action_type !== agent.action_type || machine.agent.width !== agent.width || machine.agent.height !== agent.height || machine.agent.reach.some((v, i) => v !== agent.reach[i]))) { this.scene.remove(machine.root); machine.dispose(); this.machines.delete(agent.id); machine = null; }
      if (!machine) { machine = makeMachine(agent, frame.grid.tile_size_m, { style: this.presentation }); machine.setTags(this.visibility.tags); this.machines.set(agent.id, machine); this.scene.add(machine.root); }
      if (!mayAnimate) machine.lastMove = null;
      this.poseMachine(machine, agent, frame, 1);
    }
    if (mayAnimate) {
      // Every machine that changed animates in the same transition: joint team
      // rounds move several machines at once. Dig and dump motions are planned
      // on the cells each machine changed, which then follow the bucket.
      const actors = this.actorWork(previous, frame), timing = new Map();
      for (const actor of actors.values()) {
        const machine = this.machines.get(actor.id);
        if ((actor.kind === 'dig' || actor.kind === 'dump') && machine?.plan) actor.plan = machine.plan({ kind: actor.kind, from: actor.from, to: actor.to, cells: actor.cells.map(cell => this.workCell(cell, previous, frame)) });
        for (const [key, window] of actor.plan?.timing ?? []) timing.set(key, window);
        actor.events = this.planEvents(actor); actor.fired = new Set();
      }
      // Work actions get a little more time for anticipation and follow-through.
      const work = [...actors.values()].some(actor => WORK_KINDS.includes(actor.kind)), span = work ? duration * 1.3 : duration;
      this.motion = { previous, frame, facts, actors, timing, start: performance.now(), duration: clamp(span, 100, 900) };
      for (const cell of facts.changed) this.updateCell(cell.row, cell.col, previous.maps.action[cell.row][cell.col]); this.dirtyInstances();
      this.update(performance.now());
    }
    if (this.selected) this.highlight(this.selected.row, this.selected.col);
    if (dimensionsChanged || reset) this.home({ instant: true });
  }

  populate(frame) {
    const { rows, cols, tile_size_m: tile } = frame.grid;
    const gridPositions = []; this.boundaryEntries.clear(); this.gridEntries = new Map();
    this.displayHeights = frame.maps.action.map(row => [...row]);
    const dug = this.palette.dug.map(hex => new THREE.Color(hex)), sand = new THREE.Color(this.palette.sand), loose = new THREE.Color(this.palette.loose);
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const index = row * cols + col, height = frame.maps.action[row][col];
      this.updateCell(row, col, height);
      if (!frame.metric) {
        const noise = ((row * 71 + col * 29 + (row * col) % 47) % 31) / 31;
        color.copy(height > 0 ? loose : height < 0 ? dug[Math.max(0, Math.min(dug.length - 1, Math.floor(-height) - 1))] : sand).multiplyScalar(.98 + noise * .04); this.terrain.setColorAt(index, color);
      }
      if (this.visibility.grid) { this.gridEntries.set(index, gridPositions.length); gridPositions.push(...this.flatGridCell(row, col, height)); }
    }
    this.dirtyInstances();
    if (!frame.metric) { this.terrain.instanceColor.needsUpdate = true; this.terrain.computeBoundingSphere(); }
    this.gridLines.geometry.dispose(); this.gridLines.geometry = new THREE.BufferGeometry(); this.gridLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(gridPositions, 3));
    for (const [name, layer] of Object.entries(this.layers)) layer.visible = this.visibility[name] && this.layerMaps[LAYERS[name].map] != null;
    for (const [name, boundary] of Object.entries(this.boundaries)) {
      const positions = [], test = LAYERS[name].test, map = this.layerMaps[LAYERS[name].map];
      const member = (row, col) => map != null && row >= 0 && row < rows && col >= 0 && col < cols && !frame.maps.padding[row][col] && test(map[row][col]);
      if (map != null) for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) if (member(row, col)) {
        const index = row * cols + col;
        const edges = [[row - 1, col, [[0, 0], [0, 1], [0, 2]]], [row + 1, col, [[2, 0], [2, 1], [2, 2]]], [row, col - 1, [[0, 0], [1, 0], [2, 0]]], [row, col + 1, [[0, 2], [1, 2], [2, 2]]]];
        for (const [neighborRow, neighborCol, nodes] of edges) if (!member(neighborRow, neighborCol)) {
          if (!this.boundaryEntries.has(index)) this.boundaryEntries.set(index, []);
          for (const node of [nodes[0], nodes[1], nodes[1], nodes[2]]) {
            const row2 = row * 2 + node[0], col2 = col * 2 + node[1];
            this.boundaryEntries.get(index).push({ name, y: positions.length + 1, row2, col2 });
            positions.push((col2 / 2 - cols / 2) * tile, this.boundaryHeight(row, col, row2, col2), (row2 / 2 - rows / 2) * tile * (frame.metric ? -1 : 1));
          }
        }
      }
      boundary.geometry.dispose(); boundary.geometry = new LineSegmentsGeometry();
      if (positions.length) boundary.geometry.setPositions(positions);
      boundary.visible = this.visibility[name] && positions.length > 0;
    }
  }

  populateReservations(frame) {
    if (!this.reservationOutlines && !frame.workspace_polygons?.length) return;
    if (this.reservationOutlines) {
      this.reservationOutlines.traverse(line => { if (line.material) this.lineMaterials.delete(line.material); });
      disposeProps(this.reservationOutlines);
    }
    this.reservationOutlines = new THREE.Group();
    this.reservationOutlines.name = 'recorded-stationary-reservations';
    this.reservationOutlines.visible = this.visibility.workspace;
    this.world.add(this.reservationOutlines);
    for (const polygon of frame.workspace_polygons ?? []) {
      const positions = [], tile = frame.grid.tile_size_m;
      for (let i = 0; i < polygon.vertices.length; i++) {
        positions.push(...workspaceVertexToWorld(frame.grid, polygon.vertices[i], tile * .06));
        positions.push(...workspaceVertexToWorld(frame.grid, polygon.vertices[(i + 1) % polygon.vertices.length], tile * .06));
      }
      const body = polygon.component === 'body';
      const material = new LineMaterial({ color: SLOT_COLORS[polygon.id], linewidth: body ? 1.7 : 2.8, dashed: body,
        dashSize: tile * .32, gapSize: tile * .2, transparent: true, opacity: body ? .75 : 1, depthWrite: false, depthTest: false });
      material.resolution.copy(this.renderer.getDrawingBufferSize(new THREE.Vector2())); this.lineMaterials.add(material);
      const geometry = new LineSegmentsGeometry(); geometry.setPositions(positions);
      const line = new LineSegments2(geometry, material); line.computeLineDistances();
      line.name = `machine-${polygon.id}-${polygon.component}-reservation`;
      line.renderOrder = 18; line.frustumCulled = false; line.userData.skipAO = true;
      this.reservationOutlines.add(line);
    }
  }

  flatGridCell(row, col, height) {
    const tile = this.frame.grid.tile_size_m, point = this.point(row, col, height), y = point.y + tile * .022;
    // Positive cells use the soil mesh's draped grid instead of a floating quad.
    const half = !this.frame.metric && height > 0 && !this.frame.maps.padding[row][col] ? 0 : tile / 2;
    const x = point.x, z = point.z;
    return [x - half, y, z - half, x + half, y, z - half, x + half, y, z - half, x + half, y, z + half, x + half, y, z + half, x - half, y, z + half, x - half, y, z + half, x - half, y, z - half];
  }

  setFloor(height) { this.floor = height; this.environment?.setFloor(height); }

  boundaryHeight(row, col, row2, col2) {
    const height = this.displayHeights[row][col];
    return (!this.frame.metric && height > 0 ? this.piles.nodeHeight(row2, col2) : height * this.unitHeight) + this.frame.grid.tile_size_m * .06;
  }

  updateCell(row, col, height) {
    const frame = this.frame, tile = frame.grid.tile_size_m, index = row * frame.grid.cols + col;
    const soil = !frame.metric && height > 0 && !frame.maps.padding[row][col], point = this.point(row, col, height);
    this.displayHeights[row][col] = height;
    // Soil mounds replace the positive box cap; negative excavation keeps its
    // exact stepped cut walls and there is still a solid ground slab below soil.
    const thickness = Math.max(tile * .02, (soil ? 0 : point.y) - this.floor);
    dummy.rotation.set(0, 0, 0);
    if (frame.metric) this.metricTerrainDirty = true;
    else { dummy.position.set(point.x, this.floor + thickness / 2, point.z); dummy.scale.set(tile, thickness, tile); dummy.updateMatrix(); this.terrain.setMatrixAt(index, dummy.matrix); }
    let offset = 0;
    for (const [name, layer] of Object.entries(this.layers)) {
      const slot = frame.metric ? layer.userData.cellSlots[index] : index;
      if (slot < 0) continue;
      // Studio shows only the cut still to be made: finished target cells reveal the excavated soil.
      const settings = LAYERS[name], map = this.layerMaps[settings.map], finished = name === 'dig' && this.presentation === 'studio' && map != null && height <= map[row][col];
      const visible = !soil && !finished && map != null && settings.test(map[row][col]) && !frame.maps.padding[row][col];
      dummy.position.set(point.x, point.y + tile * (.008 + offset * .003), point.z); dummy.scale.set(visible ? tile : 0, 1, visible ? tile : 0); dummy.updateMatrix(); layer.setMatrixAt(slot, dummy.matrix); offset++;
    }
    if (this.gridEntries.has(index)) this.gridLines.geometry.attributes.position.array.set(this.flatGridCell(row, col, height), this.gridEntries.get(index));
  }
  dirtyInstances() {
    if (this.frame.metric && this.metricTerrainDirty) {
      const geometry = metricTerrainGeometry(this.frame, this.displayHeights, this.unitHeight, this.floor, this.palette);
      this.terrain.geometry.dispose(); this.terrain.geometry = geometry; this.metricTerrainDirty = false;
    } else if (!this.frame.metric) this.terrain.instanceMatrix.needsUpdate = true;
    for (const layer of Object.values(this.layers)) layer.instanceMatrix.needsUpdate = true;
    const cols = this.frame.grid.cols;
    for (const [index, entries] of this.boundaryEntries) for (const entry of entries) {
      const array = this.boundaries[entry.name].geometry.attributes.instanceStart?.data.array;
      if (array) array[entry.y] = this.boundaryHeight(Math.floor(index / cols), index % cols, entry.row2, entry.col2);
    }
    for (const boundary of Object.values(this.boundaries)) { const data = boundary.geometry.attributes.instanceStart?.data; if (data) data.needsUpdate = true; }
    if (this.gridLines.geometry.attributes.position) this.gridLines.geometry.attributes.position.needsUpdate = true;
  }

  poseMachine(machine, state, frame, progress, oldState, oldFrame, kind = '', plan = null) {
    const from = oldState || state, turning = kind === 'turn' && !frame.metric ? backOut(progress) : progress;
    const position = state.position.map((v, i) => mix(from.position[i], v, progress));
    const point = this.point(position[0], position[1]); point.y = mix(this.displayHeightAt(...from.position), this.displayHeightAt(...state.position), progress);
    machine.root.position.copy(point); machine.root.rotation.y = shortestAngle(from.base_yaw, state.base_yaw, turning);
    machine.setPose({ ...state, previous_loaded: from.loaded, cabin_yaw: shortestAngle(from.cabin_yaw, state.cabin_yaw, turning), wheel_angle: mix(from.wheel_angle, state.wheel_angle, progress) }, !isJointFrame(frame) && state.id === frame.current_agent, progress, kind, plan);
    machine.drive?.(machine.root.position, machine.root.rotation.y);
  }

  /** Per-machine work in one transition. Metric plans carry explicit ownership;
   * native joint snapshots use load changes and proximity for display only. */
  actorWork(previous, frame) {
    const before = new Map(previous.agents.map(agent => [agent.id, agent]));
    if (frame.metric) {
      const work = new Map(frame.metric.work.map(item => [item.agent_id, item])), cols = frame.grid.cols;
      return new Map(frame.agents.map(agent => {
        const old = before.get(agent.id) ?? agent, explicit = work.get(agent.id);
        const moved = agent.position.some((v, i) => v !== old.position[i]), turned = agent.base_yaw !== old.base_yaw || agent.cabin_yaw !== old.cabin_yaw || agent.wheel_angle !== old.wheel_angle || agent.shovel_lifted !== old.shovel_lifted;
        const cells = (explicit?.changed_indices ?? []).map(index => { const row = Math.floor(index / cols), col = index % cols; return { row, col, delta: frame.maps.action[row][col] - previous.maps.action[row][col] }; });
        const kind = explicit?.kind === 'collect' ? 'dig' : explicit?.kind ?? (moved ? 'move' : turned ? 'turn' : '');
        return [agent.id, { id: agent.id, kind, cells, load: agent.loaded - old.loaded, from: old, to: agent, swing: kind === 'turn' && agent.cabin_yaw !== old.cabin_yaw, recipient: frame.agents.find(other => other.id === explicit?.recipient_id) ?? null }];
      }));
    }
    const loads = new Map(frame.agents.map(agent => [agent.id, agent.loaded - (before.get(agent.id)?.loaded ?? agent.loaded)]));
    const owned = new Map(frame.agents.map(agent => [agent.id, []]));
    for (const cell of transitionFacts(previous, frame).changed) {
      const explaining = frame.agents.filter(agent => cell.delta < 0 ? loads.get(agent.id) > 0 : loads.get(agent.id) < 0);
      let owner = null, distance = Infinity;
      for (const agent of explaining.length ? explaining : frame.agents) {
        const d = Math.hypot(agent.position[0] - cell.row, agent.position[1] - cell.col);
        if (d < distance) { distance = d; owner = agent; }
      }
      owned.get(owner.id).push(cell);
    }
    const actors = new Map();
    for (const agent of frame.agents) {
      const old = before.get(agent.id); if (!old) continue;
      const cells = owned.get(agent.id), load = loads.get(agent.id);
      const recipient = frame.agents.find(other => other.id !== agent.id && loads.get(other.id) > 0 && !owned.get(other.id).some(cell => cell.delta < 0));
      const moved = agent.position.some((v, i) => v !== old.position[i]), slewed = agent.cabin_yaw !== old.cabin_yaw;
      const turned = slewed || agent.base_yaw !== old.base_yaw || agent.wheel_angle !== old.wheel_angle || agent.shovel_lifted !== old.shovel_lifted;
      let kind = '';
      if (cells.some(cell => cell.delta < 0) && load > 0) kind = 'dig';
      else if (cells.some(cell => cell.delta > 0) && load < 0) kind = 'dump';
      else if (load < 0 && recipient) kind = 'transfer';
      else if (cells.length) kind = 'terrain';
      else if (moved) kind = 'move';
      else if (turned) kind = 'turn';
      actors.set(agent.id, { id: agent.id, kind, cells, load, from: old, to: agent, swing: kind === 'turn' && slewed && agent.base_yaw === old.base_yaw, recipient: kind === 'transfer' ? recipient : null });
    }
    return actors;
  }

  /** A changed cell in world space with its displayed surface height before and after. */
  workCell(cell, previous, frame) {
    const point = this.point(cell.row, cell.col), blocked = frame.maps.padding[cell.row][cell.col];
    const display = (value, start) => !frame.metric && value > 0 && !blocked ? this.piles.endpointHeight(cell.row, cell.col, start) : value * this.unitHeight;
    return { key: cell.row * frame.grid.cols + cell.col, row: cell.row, col: cell.col, delta: cell.delta, x: point.x, z: point.z, before: display(previous.maps.action[cell.row][cell.col], true), after: display(frame.maps.action[cell.row][cell.col], false) };
  }

  /** Timed display events for one machine's transition, in normalized motion time. */
  planEvents(actor) {
    const events = [], { kind, plan, from, to } = actor;
    if (!kind || kind === 'terrain') return events;
    const moved = to.position.some((v, i) => v !== from.position[i]);
    events.push({ at: 0, once: 'exhaust-start' });
    if (moved) events.push({ from: .05, to: .9, stream: 'tracks', rate: 16 });
    const cells = sign => actor.cells.filter(cell => sign < 0 ? cell.delta < 0 : cell.delta > 0);
    if (kind === 'dig') {
      const bite = plan?.events.bite ?? (to.type === 2 ? .38 : .32), [start, end] = plan?.events.drag ?? [bite, bite + .22];
      events.push({ at: bite, once: 'bite', cells: cells(-1) });
      events.push({ from: start, to: end, stream: 'scoop', cells: cells(-1), rate: plan ? 34 : 70 });
      if (plan?.events.breakout) events.push({ at: plan.events.breakout, once: 'spill' });
    } else if (kind === 'dump') {
      const [start, end] = plan?.events.pour ?? (to.type === 1 ? [.32, .72] : to.type === 2 ? [.34, .62] : [.44, .74]);
      events.push({ from: start, to: end, stream: to.type === 1 ? 'bed' : 'pour', cells: cells(1), rate: plan ? 230 : 60 });
      events.push({ at: Math.min(.95, (start + end) / 2 + .1), once: 'landing', cells: cells(1) });
    } else if (kind === 'transfer') {
      events.push({ from: .44, to: .72, stream: 'transfer', rate: 55 });
    }
    return events;
  }

  centroid(cells) {
    const point = new THREE.Vector3(); if (!cells?.length) return null;
    for (const cell of cells) point.add(this.surfacePoint(cell.row, cell.col));
    return point.multiplyScalar(1 / cells.length);
  }

  runEvents(motion, t, dt) {
    const tile = motion.frame.grid.tile_size_m, fx = this.effects, studio = this.presentation === 'studio';
    for (const actor of motion.actors.values()) {
      const machine = this.machines.get(actor.id); if (!machine || !actor.events.length) continue;
      machine.root.updateMatrixWorld(true);
      for (const [index, event] of actor.events.entries()) {
        if (event.once) {
          if (actor.fired.has(index) || t < event.at) continue;
          actor.fired.add(index);
          if (event.once === 'exhaust-start') { if (!studio) fx.puff(machine.exhaust(), { count: 4, size: tile * .28, rise: 1.4, spread: tile * .15, color: 0x5e636b, life: 1.1 }); }
          else if (event.once === 'bite') {
            const at = studio ? machine.teeth() : this.centroid(event.cells) ?? machine.tip();
            fx.burst(at, { count: studio ? 9 : 14, speed: studio ? 1.6 : 2.4, size: tile * (studio ? .06 : .09) });
            fx.puff(at, { count: 7, size: tile * .38, spread: tile * .6, rise: .5 }); fx.dust(at, { count: 6, size: tile * 1.1, spread: tile * .5, life: 1.6 });
          } else if (event.once === 'landing') {
            const at = this.centroid(event.cells); if (at) { fx.puff(at, { count: 8, size: tile * .42, spread: tile * .7, rise: .4 }); fx.dust(at, { count: 9, size: tile * 1.5, spread: tile * .8, life: 2 }); }
          } else if (event.once === 'spill') {
            fx.throwClods(machine.lip(), machine.lip().setY(this.groundAt(machine.lip().x, machine.lip().z)), { count: 5, flight: .4, spread: tile * .3, size: tile * .06 });
          }
        } else if (t >= event.from && t <= event.to) {
          event.carry = (event.carry ?? 0) + event.rate * dt;
          let count = Math.floor(event.carry); event.carry -= count;
          while (count-- > 0) this.emitStream(event, machine, motion, tile);
        }
      }
    }
  }

  emitStream(event, actor, motion, tile) {
    const fx = this.effects, studio = this.presentation === 'studio';
    const pick = cells => cells?.length ? this.surfacePoint(...Object.values(cells[Math.floor(Math.random() * cells.length)]).slice(0, 2)) : null;
    if (event.stream === 'tracks') {
      const state = motion.frame.agents.find(agent => agent.id === actor.agent.id);
      const back = new THREE.Vector3(-actor.agent.height * tile * .45, 0, (Math.random() < .5 ? -1 : 1) * actor.agent.width * tile * .35).applyAxisAngle(new THREE.Vector3(0, 1, 0), actor.root.rotation.y).add(actor.root.position);
      if (state && Math.random() < .5) { fx.puff(back, { count: 1, size: tile * .3, spread: tile * .2, rise: .35, life: .8 }); if (Math.random() < .4) fx.dust(back, { count: 1, size: tile * .9, spread: tile * .3, life: 1.3, opacity: .16 }); }
      if (Math.random() < .25) fx.puff(actor.exhaust(), { count: 1, size: tile * .2, rise: 1.3, spread: tile * .08, color: 0x6a6f77, life: 1 });
    } else if (event.stream === 'scoop') {
      if (studio) {
        // Loose soil rolls off the teeth and spills beside the bucket.
        const teeth = actor.teeth(), ground = teeth.clone().add(new THREE.Vector3((Math.random() - .5) * tile * 1.2, 0, (Math.random() - .5) * tile * 1.2));
        ground.y = this.groundAt(ground.x, ground.z);
        fx.throwClods(teeth.clone().add(new THREE.Vector3(0, tile * .12, 0)), ground, { count: 1, flight: .28, spread: tile * .15, size: tile * .055, jitter: tile * .2 });
        if (Math.random() < .12) fx.dust(teeth, { count: 1, size: tile * .8, spread: tile * .3, life: 1.2, opacity: .14 });
      } else { const from = pick(event.cells); if (from) fx.throwClods(from, actor.tip(), { count: 1, flight: .22, spread: 0, size: tile * .08, settle: false, jitter: tile * .3 }); }
    } else if (event.stream === 'pour' || event.stream === 'bed') {
      const to = pick(event.cells); if (!to) return;
      const from = event.stream === 'bed' ? actor.bedLip() : studio ? actor.lip() : actor.tip();
      fx.throwClods(from, to, { count: 1, flight: studio ? .42 : .34, spread: tile * .45, size: tile * (studio ? .06 : .095), jitter: tile * (studio ? .22 : .12) });
      if (studio && Math.random() < .06) fx.dust(from, { count: 1, size: tile * .9, spread: tile * .3, life: 1.4, opacity: .12 });
    } else if (event.stream === 'transfer') {
      const recipient = this.machines.get(motion.actors.get(actor.agent.id)?.recipient?.id); if (!recipient) return;
      recipient.root.updateMatrixWorld(true);
      fx.throwClods(actor.tip(), recipient.tip(), { count: 1, flight: .3, spread: tile * .2, size: tile * .09, settle: false, jitter: tile * .1 });
    }
  }

  update(time) {
    const dt = Math.min(.1, Math.max(0, (time - (this.clock?.last ?? time)) / 1000)); if (this.clock) this.clock.last = time;
    shared.uTime.value = time / 1000;
    if (!this.frame) return;
    this.measure(dt);
    this.environment?.update(time / 1000);
    const moving = new Map();
    if (this.motion) {
      const { previous, frame, facts, start, duration, actors, timing } = this.motion, t = clamp((time - start) / duration, 0, 1), eased = smooth(t), cols = frame.grid.cols;
      // Planned cells change while the bucket passes them; others ease together.
      const progressAt = (row, col) => { const window = timing.get(row * cols + col); return window ? smooth(clamp((t - window[0]) / (window[1] - window[0]), 0, 1)) : eased; };
      if (facts.changed.length) this.piles.update(timing.size ? progressAt : eased);
      for (const cell of facts.changed) this.updateCell(cell.row, cell.col, mix(previous.maps.action[cell.row][cell.col], frame.maps.action[cell.row][cell.col], progressAt(cell.row, cell.col)));
      for (const agent of frame.agents) {
        const old = previous.agents.find(a => a.id === agent.id), actor = actors.get(agent.id);
        let kind = actor?.kind === 'terrain' ? '' : actor?.kind ?? '';
        if (!kind && [...actors.values()].some(other => other.recipient?.id === agent.id)) kind = 'receive';
        if (kind === 'turn' || kind === 'move') kind = old && agent.position.some((v, i) => v !== old.position[i]) ? 'move' : 'turn';
        this.poseMachine(this.machines.get(agent.id), agent, frame, actor?.plan || frame.metric?.event.phase === 'drive' ? t : eased, old, previous, kind, actor?.plan);
        if (old && agent.position.some((v, i) => v !== old.position[i])) {
          const forward = new THREE.Vector2(Math.cos(agent.base_yaw), Math.sin(agent.base_yaw)), delta = new THREE.Vector2(agent.position[1] - old.position[1], (agent.position[0] - old.position[0]) * (frame.metric ? -1 : 1));
          moving.set(agent.id, { move: t, direction: Math.sign(forward.x * delta.x - forward.y * delta.y) || 1 });
        }
      }
      if (facts.changed.length) { this.dirtyInstances(); if (this.selected) this.highlight(this.selected.row, this.selected.col); }
      if (!this.reducedMotion) this.runEvents(this.motion, t, dt);
      if (t >= 1) {
        this.clearMotion();
        if (this.floor !== this.finalFloor) { this.setFloor(this.finalFloor); this.populate(this.frame); }
      }
    }
    const seconds = time / 1000;
    for (const machine of this.machines.values()) machine.tick?.(seconds, { ...(moving.get(machine.agent.id) || {}), reducedMotion: this.reducedMotion });
    // An idle active machine breathes a little exhaust now and then.
    this.clock.idle -= dt;
    if (!this.reducedMotion && this.clock.idle <= 0) {
      this.clock.idle = 1.3 + Math.random() * .8;
      const active = this.machines.get(this.frame.current_agent), tile = this.frame.grid.tile_size_m;
      if (active && !this.frame.done) { active.root.updateMatrixWorld(true); this.effects.puff(active.exhaust(), { count: 1, size: tile * .18, rise: 1.1, spread: tile * .05, color: 0x767b83, life: 1.2 }); }
    }
    this.effects.update(dt);
    if (this.tween) {
      const { from, to, start, duration } = this.tween, u = easeInOut(clamp((time - start) / duration, 0, 1));
      this.camera.position.lerpVectors(from.position, to.position, u); this.controls.target.lerpVectors(from.target, to.target, u);
      if (u >= 1) this.tween = null;
    }
    if (this.follow) {
      const machine = this.machines.get(this.frame.current_agent);
      if (machine) { const destination = machine.root.position.clone(); destination.y += this.frame.grid.tile_size_m; const delta = destination.sub(this.controls.target).multiplyScalar(.055); this.camera.position.add(delta); this.controls.target.add(delta); }
    }
  }

  clearMotion() {
    if (this.motion) for (const agent of this.motion.frame.agents) { const machine = this.machines.get(agent.id); if (machine) machine.tick?.(performance.now() / 1000, { reducedMotion: this.reducedMotion }); }
    this.motion = null;
  }
  setLayer(name, visible) {
    this.visibility[name] = visible;
    if (name === 'workspace') { if (this.reservationOutlines) this.reservationOutlines.visible = visible; return; }
    if (name === 'eligibility') { for (const layer of ELIGIBILITY_LAYERS) this.setLayer(layer, visible); return; }
    if (name === 'tags') { for (const machine of this.machines.values()) machine.setTags(visible); return; }
    if (!this.frame) return; this.piles?.setLayer(name, visible); if (name === 'grid') { this.gridLines.visible = visible; this.populate(this.frame); } else if (this.layers[name]) this.layers[name].visible = visible && this.layerMaps[LAYERS[name].map] != null; if (this.boundaries[name]) this.boundaries[name].visible = visible && !!this.boundaries[name].geometry.attributes.instanceStart; }
  setHeight(value) { this.heightScale = value; if (this.frame) this.setFrame(this.frame); }
  flyTo(position, target, { instant = false } = {}) {
    if (instant || this.reducedMotion) { this.tween = null; this.camera.position.copy(position); this.controls.target.copy(target); this.controls.update(); return; }
    this.tween = { from: { position: this.camera.position.clone(), target: this.controls.target.clone() }, to: { position, target }, start: performance.now(), duration: 750 };
  }
  home({ instant = false } = {}) {
    if (!this.frame) return; this.follow = false;
    const aspect = this.camera.aspect, fit = this.presentation === 'diorama' ? 1.75 : 1.45, distance = this.span * (aspect < 1 ? fit / aspect : fit);
    this.flyTo(new THREE.Vector3(distance * .72, distance * .66, distance * .84), new THREE.Vector3(0, -this.span * .04, 0), { instant });
    this.onCameraChange?.({ view: 'home', follow: false });
  }
  top({ instant = false } = {}) {
    if (!this.frame) return; this.follow = false; this.camera.up.set(0, 1, 0);
    const distance = this.span * .5 / Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2) / Math.min(this.camera.aspect, 1) * 1.3;
    this.flyTo(new THREE.Vector3(0, distance, this.span * .001), new THREE.Vector3(0, 0, 0), { instant });
    this.onCameraChange?.({ view: 'top', follow: false });
  }
  setFollow(value) {
    if (value && isJointFrame(this.frame)) return;
    this.follow = value; this.tween = null;
    if (value && this.frame) {
      const machine = this.machines.get(this.frame.current_agent);
      if (machine) {
        const state = this.frame.agents.find(agent => agent.id === this.frame.current_agent), tile = this.frame.grid.tile_size_m;
        const destination = machine.root.position.clone(); destination.y += tile;
        const verticalFov = THREE.MathUtils.degToRad(this.camera.fov), horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * this.camera.aspect);
        const radius = Math.max(state.reach[1], Math.hypot(state.width, state.height) * .65) * tile;
        const distance = Math.max(this.span * .5, radius * 1.15 / Math.sin(Math.min(verticalFov, horizontalFov) / 2));
        const offset = this.camera.position.clone().sub(this.controls.target).normalize().multiplyScalar(distance);
        this.flyTo(destination.clone().add(offset), destination);
      }
    }
    this.onCameraChange?.({ follow: value });
  }
  pick(event) {
    if (!this.terrain) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    this.camera.updateMatrixWorld(); this.world.updateMatrixWorld(true); this.raycaster.setFromCamera(this.pointer, this.camera);
    const objects = [this.terrain, this.obstacleProps]; if (this.piles.surface?.visible) objects.push(this.piles.surface);
    const hit = this.raycaster.intersectObjects(objects.filter(Boolean), true)[0];
    let cell;
    if (hit && hit.object === this.piles.surface) cell = this.piles.cellForHit(hit);
    else if (hit?.object === this.terrain && hit.instanceId !== undefined) cell = { row: Math.floor(hit.instanceId / this.frame.grid.cols), col: hit.instanceId % this.frame.grid.cols };
    else if (hit) { const { rows, cols, tile_size_m: tile } = this.frame.grid; cell = { row: clamp(Math.floor((this.frame.metric ? -hit.point.z : hit.point.z) / tile + rows / 2), 0, rows - 1), col: clamp(Math.floor(hit.point.x / tile + cols / 2), 0, cols - 1) }; }
    if (cell) { this.highlight(cell.row, cell.col); this.onPick?.(cell); }
  }
  highlight(row, col) { if (row >= this.frame.grid.rows || col >= this.frame.grid.cols) { this.selected = null; this.selection.visible = false; return; } this.selected = { row, col }; this.selection.position.copy(this.surfacePoint(row, col)); this.selection.position.y += this.frame.grid.tile_size_m * .03; this.selection.visible = true; }
  /** Render one frame at `scale`× the viewport (at least the device ratio) for figures. */
  capture({ scale = 2 } = {}) {
    const width = this.element.clientWidth || 1, height = this.element.clientHeight || 1;
    const ratio = Math.min(Math.max(scale, this.pixelRatio), this.renderer.capabilities.maxTextureSize / Math.max(width, height));
    // Multisampled HDR targets at print size are memory-heavy; FXAA covers edges.
    this.renderer.setPixelRatio(ratio); this.renderer.setSize(width, height, false);
    this.post?.setSamples(0); this.post?.setSize(width, height, ratio);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    for (const material of this.lineMaterials) material.resolution.copy(size);
    try { this.render(); return { url: this.renderer.domElement.toDataURL('image/png'), width: size.x, height: size.y }; }
    finally { this.renderer.setPixelRatio(this.pixelRatio); this.post?.setSamples(4); this.resize(); }
  }
}
