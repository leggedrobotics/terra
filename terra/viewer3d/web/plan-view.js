/*!
 * Three.js 0.185.1 — The MIT License
 * Copyright © 2010-2026 three.js authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
// Shared native-plan and metric-plan player for offline pages and dashboards.
//
// One recorded Terra episode (a terra.viewer3d.v1 replay, compacted by terra.postprocess.dashboard.terra3d_data) plays in the Terra
// viewer's own scene, TerraScene: terrain, soil piles, the excavator and its dig and dump motion. The page decodes
// the episode and drives this view. Frames are built on demand from the first frame's maps and the cells each frame
// changes. On top of the scene, every Terra step (a dig/dump pair, numbered like the dashboard's "Terra step") gets:
// - a label over the soil it dumped, white on a final dump zone and teal elsewhere (soil Terra collects later),
//   hollow teal once a later step has lifted that soil; steps that dump at the same place share one label;
// - an arc from its dig to its dump.
// The current step is blue. "All dumps" also shows the steps still to come, in grey.
// npm run build packages this player with the shared scene and Three.js.
import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { TerraScene } from './scene.js';
import { actionName } from './data.js';
import { PostprocessedEpisode, POSTPROCESSED_SCHEMA } from './postprocessed.js';
import { PostprocessedOverlay } from './postprocessed-view.js';

// Screen time of one action at 1x, by what it does (the pacing of the studio video harness).
const PACE = { dig: 1500, dump: 1400, terrain: 900, move: 620, turn: 560, swing: 520, idle: 300 };
const INK = '#1c1917', TEAL = '#0f766e', BLUE = '#2563eb';
const ARC_COLOR = { final: 0x3f3a36, temp: 0x0f766e, current: 0x2563eb, future: 0x9b968e };
const LABEL_PX = 17;  // label row height on screen
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** The frames of one compacted episode, built on demand in the viewer's snapshot format. */
class Episode {
  constructor(data) {
    this.data = data; this.count = data.count;
    const { rows, cols } = data.grid;
    this.nested = flat => Array.from({ length: rows }, (_, r) => Array.from(flat.subarray(r * cols, (r + 1) * cols)));
    this.fixed = { target: this.nested(data.target), padding: this.nested(data.padding), dumpability_static: data.dump_static ? this.nested(data.dump_static) : null };
    this.layers = {};
    for (const [name, layer] of Object.entries(data.layers)) if (layer) this.layers[name] = { ...layer, cur: layer.init.slice(), at: 0 };
  }

  layerAt(name, index) {
    const layer = this.layers[name];
    if (!layer) return null;
    if (index < layer.at) { layer.cur.set(layer.init); layer.at = 0; }
    for (let f = layer.at + 1; f <= index; f++) for (let i = layer.off[f - 1]; i < layer.off[f]; i++) layer.cur[layer.idx[i]] = layer.val[i];
    layer.at = index;
    return this.nested(layer.cur);
  }

  frame(index) {
    const d = this.data, last = index === this.count - 1;
    const maps = { ...this.fixed, action: this.layerAt('action', index), dumpability: this.layerAt('dumpability', index), interaction: this.layerAt('interaction', index), traversability: null };
    const agents = d.agents.fixed.map((fixed, a) => {
      const [row, col, base_yaw, cabin_yaw, loaded, wheel_angle, shovel_lifted] = d.agents.track[index][a];
      return { ...fixed, position: [row, col], base_yaw, cabin_yaw, loaded, wheel_angle, shovel_lifted };
    });
    return {
      step: d.step[index], action: d.action[index] < 0 ? null : d.action[index], actor_id: d.actor[index] < 0 ? null : d.actor[index],
      current_agent: d.current[index], reward: 0, done: last && d.done, task_done: last && d.task_done,
      grid: { rows: d.grid.rows, cols: d.grid.cols, tile_size_m: d.grid.tile }, maps, agents,
      ...d.frame_extras?.[index],
    };
  }
}

/** A label texture: the step numbers of one dump place as pills, wrapped six per row. */
function drawLabel(entries) {
  const scale = 2, h = 18 * scale, pad = 5 * scale, gap = 3 * scale, perRow = 6;
  const canvas = document.createElement('canvas'), c = canvas.getContext('2d');
  c.font = `700 ${12 * scale}px system-ui, sans-serif`;
  const widths = entries.map(e => Math.max(h, c.measureText(String(e.p)).width + 2 * pad));
  const rows = Math.ceil(entries.length / perRow);
  const rowWidths = Array.from({ length: rows }, (_, r) => widths.slice(r * perRow, (r + 1) * perRow).reduce((a, w) => a + w + gap, -gap));
  canvas.width = Math.ceil(Math.max(...rowWidths)) + 4; canvas.height = rows * (h + gap) - gap + 4;
  c.font = `700 ${12 * scale}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle';
  entries.forEach((e, i) => {
    const r = Math.floor(i / perRow), k = i % perRow, w = widths[i];
    const x = 2 + (canvas.width - 4 - rowWidths[r]) / 2 + widths.slice(r * perRow, r * perRow + k).reduce((a, v) => a + v + gap, 0), y = 2 + r * (h + gap);
    const [fill, ink, line] = e.current ? [BLUE, '#ffffff', BLUE] : e.future ? ['#ece9e4', '#8a847c', '#c9c4bc']
      : e.final ? ['#ffffff', INK, INK] : e.gone ? ['#ffffff', TEAL, TEAL] : [TEAL, '#ffffff', TEAL];
    c.beginPath(); c.roundRect(x, y, w, h, h / 2); c.fillStyle = fill; c.fill();
    c.lineWidth = 1.5 * scale; c.strokeStyle = line; c.stroke();
    c.fillStyle = ink; c.fillText(String(e.p), x + w / 2, y + h / 2 + scale);
  });
  return { canvas, rows };
}

class PlanView {
  constructor(element, { onFrame, onPick, onError } = {}) {
    this.element = element; this.onFrame = onFrame;
    this.scene = new TerraScene(element, { onError, onPick: cell => onPick?.(this.cell(cell)) });
    this.loop = time => { this.scene.update(time); this.scene.controls.update(); this.scene.render(); };
    // The studio look; a viewer without it keeps its own default.
    this.scene.setPresentation('studio');
    // No machine number tags; the workspace overlay starts hidden (the page has a switch for it).
    this.scene.setLayer('tags', false); this.scene.setLayer('interaction', false);
    this.overlay = new THREE.Group(); this.overlay.name = 'terra-plan-dumps'; this.scene.scene.add(this.overlay);
    this.lineMaterials = new Set();
    this.observer = new ResizeObserver(() => this.resize()); this.observer.observe(element);
    this.index = 0; this.playing = false; this.speed = 1; this.allDumps = false; this.active = true;
  }

  get count() { return this.episode ? this.episode.count : 0; }

  /** Show a new episode from its first frame, with the map-aligned camera. */
  load(data) {
    this.pause();
    this.metricOverlay?.dispose(); this.metricOverlay = null;
    this.postprocessed = data.schema === POSTPROCESSED_SCHEMA;
    if (this.postprocessed) {
      this.data = data; this.episode = new PostprocessedEpisode(data); this.index = 0;
      this.workspacesVisible = true; this.routesVisible = true;
      this.scene.setLayer('tags', data.agents.length > 1);
      this.scene.setFrame(this.episode.frame(0), { reset: true });
      this.buildOverlay(); this.bounds = this.metricOverlay.bounds();
      this.view('map', { instant: true }); this.update(); this.onFrame?.(this.index);
      return;
    }
    this.scene.setLayer('tags', !!data.joint);
    this.data = data; this.episode = new Episode(data); this.index = 0;
    this.eventAt = new Map(data.events.map(e => [e.frame, e]));
    this.kinds = Array.from({ length: data.count }, (_, f) => (f ? this.kind(f) : 'idle'));
    this.scene.setFrame(this.episode.frame(0), { reset: true });
    this.buildOverlay();
    this.bounds = this.worksite();
    this.view('map', { instant: true });
    this.update();
    this.onFrame?.(this.index);
  }

  /** What the action into frame f does, for pacing. */
  kind(f) {
    if (this.data.joint) return 'idle';
    const event = this.eventAt.get(f);
    if (event) return event.kind;
    const [a] = this.data.agents.track[f], [b] = this.data.agents.track[f - 1];
    return a[0] !== b[0] || a[1] !== b[1] ? 'move' : a[2] !== b[2] ? 'turn' : a[3] !== b[3] ? 'swing' : 'idle';
  }

  pace(f) { return (this.postprocessed ? this.episode.duration(f) : this.data.joint ? 900 : PACE[this.kinds[f]]) / this.speed; }

  description(index = this.index) { return actionName(this.episode.frame(index)); }

  show(index, { animate = false, duration = null } = {}) {
    if (!this.episode) return;
    index = clamp(Math.round(index), 0, this.count - 1);
    const forward = animate && index === this.index + 1 && (!this.postprocessed || this.episode.event(index).phase !== 'relocate');
    this.scene.setFrame(this.episode.frame(index), { animate: forward, duration: duration ?? 650 });
    // The scene caps its own motion at 0.9 s; at 1x a dig takes longer.
    if (forward && duration && this.scene.motion) this.scene.motion.duration = Math.max(80, duration);
    this.index = index;
    this.update();
    this.onFrame?.(index);
  }

  /** Frame `target` animated from the frame before it. */
  arrive(target, scale = 1) {
    target = clamp(target, 0, this.count - 1);
    if (target === 0) { this.show(0); return; }
    if (this.index !== target - 1) this.show(target - 1);
    this.show(target, { animate: true, duration: this.pace(target) * scale });
  }

  /** The previous or next dig or dump. */
  jump(direction) {
    if (this.data?.joint) { this.pause(); this.show(this.index + direction); return; }
    if (!this.data) return;
    this.pause();
    const frames = this.postprocessed ? this.data.frames.flatMap((e, i) => ['dig', 'cut', 'dump', 'collect', 'transfer'].includes(e.phase) ? [i + 1] : []) : this.data.events.map(e => e.frame);
    const target = direction > 0 ? frames.find(f => f > this.index) : [...frames].reverse().find(f => f < this.index);
    if (target == null) this.show(direction > 0 ? this.count - 1 : 0); else this.arrive(target);
  }

  /** The dig or dump of Terra step p. */
  showPair(p, phase = 'dig', { scale = 1 } = {}) {
    const pair = this.data?.pairs[p];
    if (!pair) return;
    this.pause();
    this.arrive(this.data.events[phase === 'dump' ? pair.dump : pair.dig].frame, scale);
  }

  showWorkspace(id, phase = null) {
    if (!this.postprocessed) return;
    const index = this.data.frames.findIndex(f => f.workspace_id === id && (phase == null || f.phase === phase));
    if (index >= 0) { this.pause(); this.show(index + 1); }
  }

  setWorkspaces(on) { this.workspacesVisible = !!on; this.update(); }
  setRoutes(on) { this.routesVisible = !!on; this.update(); }

  /** The Terra step the frame belongs to: the last one whose dig is done, -1 before the first dig. */
  pairAt(index = this.index) {
    if (this.postprocessed) return this.data.workspaces.findIndex(w => w.id === this.episode.event(index).workspace_id);
    let current = -1;
    this.data?.pairs.forEach((pair, p) => { if (this.data.events[pair.dig].frame <= index) current = p; });
    return current;
  }

  play() {
    if (!this.episode) return;
    if (this.index >= this.count - 1) this.show(0);
    this.playing = true; this.onFrame?.(this.index); this.advance();
  }

  advance() {
    clearTimeout(this.timer);
    if (!this.playing) return;
    if (this.index >= this.count - 1) { this.pause(); return; }
    const ms = this.pace(this.index + 1);
    this.show(this.index + 1, { animate: true, duration: ms * (this.postprocessed ? 1 : .92) });
    this.timer = setTimeout(() => this.advance(), ms);
  }

  pause() { const was = this.playing; this.playing = false; clearTimeout(this.timer); if (was) this.onFrame?.(this.index); }
  setSpeed(speed) { this.speed = speed; }
  setAllDumps(on) { this.allDumps = on; this.update(); }

  /** Render only while the page shows the view. */
  setActive(on) {
    if (on === this.active) return;
    this.active = on;
    if (!on) this.pause();
    this.scene.renderer.setAnimationLoop(on ? this.loop : null);
  }

  /** A clicked cell: Terra row and column. */
  cell({ row, col }) { return { row, col }; }

  /* ---------- dumps and arcs ---------- */

  buildOverlay() {
    for (const child of [...this.overlay.children]) { child.geometry?.dispose(); child.material?.map?.dispose(); child.material?.dispose(); this.overlay.remove(child); }
    this.lineMaterials.clear();
    if (this.postprocessed) { this.places = []; this.arcs = []; this.metricOverlay = new PostprocessedOverlay(this, this.data); return; }
    const { events, pairs } = this.data, tile = this.data.grid.tile;
    // Dump places: steps whose dumps lie within 1.6 cells of a place's first dump share its label.
    this.places = [];
    pairs.forEach((pair, p) => {
      const dump = events[pair.dump];
      const place = this.places.find(c => Math.hypot(c.row - dump.row, c.col - dump.col) <= 1.6);
      if (place) place.pairs.push(p); else this.places.push({ row: dump.row, col: dump.col, pairs: [p], key: '' });
    });
    for (const place of this.places) {
      place.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, depthWrite: false, transparent: true, sizeAttenuation: false }));
      place.sprite.center.set(.5, 0); place.sprite.renderOrder = 40; place.sprite.userData.skipAO = true;
      this.overlay.add(place.sprite);
    }
    this.arcs = pairs.map(() => {
      const material = new LineMaterial({ color: ARC_COLOR.final, linewidth: 2.4, transparent: true, depthTest: false, depthWrite: false, dashSize: tile * .45, gapSize: tile * .3 });
      this.lineMaterials.add(material);
      const line = new Line2(new LineGeometry(), material);
      line.renderOrder = 30; line.frustumCulled = false; line.userData.skipAO = true; line.userData.key = '';
      this.overlay.add(line);
      return line;
    });
    this.resize();
  }

  /** Displayed surface height of a cell at the end of the current transition (pile tops, cut floors). */
  height(row, col) {
    const frame = this.scene.frame, r = clamp(Math.round(row), 0, frame.grid.rows - 1), c = clamp(Math.round(col), 0, frame.grid.cols - 1);
    const soil = frame.maps.action[r][c];
    return soil > 0 && !frame.maps.padding[r][c] ? this.scene.piles.endpointHeight(r, c) : Math.max(0, soil * this.scene.unitHeight);
  }

  update() {
    if (!this.data) return;
    if (this.postprocessed) { this.metricOverlay?.update(this.index); return; }
    const { events, pairs } = this.data, f = this.index, current = this.pairAt(f), tile = this.data.grid.tile;
    const dumped = p => events[pairs[p].dump].frame <= f;
    // A later step has lifted (some of) this temporary dump.
    const gone = p => { const q = events[pairs[p].dump].collected_by; return q != null && events[pairs[q].dig].frame <= f; };
    for (const place of this.places) {
      const entries = place.pairs.filter(p => dumped(p) || this.allDumps)
        .map(p => ({ p, current: p === current && dumped(p), future: !dumped(p), final: events[pairs[p].dump].zone >= .5, gone: gone(p) }));
      place.sprite.visible = entries.length > 0;
      if (!entries.length) continue;
      const key = JSON.stringify(entries);
      if (key !== place.key) {
        const { canvas, rows } = drawLabel(entries);
        place.sprite.material.map?.dispose();
        place.sprite.material.map = new THREE.CanvasTexture(canvas); place.sprite.material.map.colorSpace = THREE.SRGBColorSpace;
        place.sprite.material.needsUpdate = true; place.key = key; place.aspect = canvas.width / canvas.height; place.rows = rows;
      }
      const point = this.scene.point(place.row, place.col);
      point.y = Math.max(...place.pairs.map(p => this.height(events[pairs[p].dump].row, events[pairs[p].dump].col))) + tile * .9;
      place.sprite.position.copy(point);
    }
    this.scaleLabels();
    pairs.forEach((pair, p) => {
      const line = this.arcs[p], dig = events[pair.dig], dump = events[pair.dump];
      const state = p === current ? 'current' : dumped(p) ? (dump.zone >= .5 ? 'final' : 'temp') : this.allDumps ? 'future' : null;
      line.visible = state !== null && (state !== 'current' || dig.frame <= f);
      if (!line.visible) return;
      const a = this.scene.point(dig.row, dig.col), b = this.scene.point(dump.row, dump.col);
      a.y = tile * .25; b.y = this.height(dump.row, dump.col) + tile * .15;
      const lift = Math.max(a.y, b.y) + Math.max(tile * 1.2, a.distanceTo(b) * .3);
      const middle = a.clone().lerp(b, .5).setY(lift), points = [];
      for (let i = 0; i <= 28; i++) {
        const t = i / 28, u = 1 - t;
        points.push(u * u * a.x + 2 * u * t * middle.x + t * t * b.x, u * u * a.y + 2 * u * t * middle.y + t * t * b.y, u * u * a.z + 2 * u * t * middle.z + t * t * b.z);
      }
      const key = `${points.map(v => v.toFixed(3)).join(',')}|${state}|${dumped(p)}|${gone(p)}`;
      if (key === line.userData.key) return;
      line.userData.key = key;
      line.geometry.setPositions(points);
      line.computeLineDistances();
      const material = line.material;
      material.color.setHex(ARC_COLOR[state]); material.linewidth = state === 'current' ? 4 : 2.4;
      material.opacity = state === 'future' ? .55 : state === 'current' ? 1 : state === 'temp' && gone(p) ? .35 : .8;
      // The current step's arc is dashed until its soil lands.
      material.dashed = state === 'future' || (state === 'current' && !dumped(p));
      material.needsUpdate = true;
    });
  }

  /** Labels keep LABEL_PX per row on screen at any zoom (sprites without size attenuation). */
  scaleLabels() {
    const height = this.element.clientHeight || 1, projection = this.scene.camera.projectionMatrix.elements[5] || 1;
    const unit = 2 * LABEL_PX / (projection * height);
    for (const place of this.places ?? []) if (place.aspect) place.sprite.scale.set(unit * place.rows * place.aspect, unit * place.rows, 1);
  }

  resize() {
    const size = this.scene.renderer.getDrawingBufferSize(new THREE.Vector2());
    for (const material of this.lineMaterials) material.resolution.copy(size);
    this.scaleLabels();
  }

  /* ---------- camera ---------- */

  /** World box of the work: the dig target, every dig and dump, and the machine where it digs and dumps. */
  worksite() {
    const d = this.data, { rows, cols, tile } = d.grid, box = new THREE.Box3(), add = (row, col, r) => {
      const p = this.scene.point(row, col);
      box.expandByPoint(new THREE.Vector3(p.x - r, 0, p.z - r)); box.expandByPoint(new THREE.Vector3(p.x + r, 0, p.z + r));
    };
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) if (d.target[row * cols + col] < 0) add(row, col, tile);
    for (const event of d.events) {
      add(event.row, event.col, tile * 2);
      for (const [row, col] of d.agents.track[event.frame]) add(row, col, tile * 2);
    }
    box.min.y = -tile; box.max.y = tile * 2;
    return box;
  }

  /** Camera distance that fits `box` seen from `direction` (unit vector from target to camera). */
  fit(box, direction, fill = .86) {
    const camera = this.scene.camera, target = box.getCenter(new THREE.Vector3());
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), direction);
    if (right.lengthSq() < 1e-8) right.set(0, 0, 1); else right.normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize(), tangent = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    let radius = 0;
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
      const offset = new THREE.Vector3(x, y, z).sub(target);
      radius = Math.max(radius, offset.dot(direction) + Math.abs(offset.dot(up)) / (tangent * fill), offset.dot(direction) + Math.abs(offset.dot(right)) / (tangent * camera.aspect * fill));
    }
    return { target, radius };
  }

  /**
   * 'map': oblique, aligned with the dashboard's 2D map (map x to the right, map y away from the camera; Terra
   * rows run along map x and the viewer puts rows on world z). 'top': from above, same orientation. 'step': the
   * current Terra step's dig, dump and machine. 'follow': the viewer's follow camera.
   */
  view(name, { instant = false } = {}) {
    if (!this.data) return;
    if (name === 'follow') { this.scene.setFollow(true); return; }
    this.scene.setFollow(false);
    // The step view looks down steeply, so the machine hides little of its own dig and dump.
    const elevation = THREE.MathUtils.degToRad(name === 'top' ? 89.5 : name === 'step' ? 64 : 52);
    const gridYaw = this.postprocessed ? this.data.grid.yaw_rad ?? 0 : 0;
    const direction = this.postprocessed ? new THREE.Vector3(-Math.sin(gridYaw) * Math.cos(elevation), Math.sin(elevation), Math.cos(gridYaw) * Math.cos(elevation)) : new THREE.Vector3(-Math.cos(elevation), Math.sin(elevation), 0).normalize();
    let box = this.bounds;
    if (this.postprocessed && name === 'step') {
      box = new THREE.Box3();
      const event = this.episode.event(this.index), current = this.data.workspaces.find(w => w.id === event.workspace_id);
      const states = event.agents.filter(a => !current || a.id === current.agent_id);
      for (const state of states) { const p = this.metricOverlay.point(state.pose[0], state.pose[1]), agent = this.data.agents.find(a => a.id === state.id), r = Math.max(...agent.reach_m, agent.length_m); box.expandByPoint(p.clone().add(new THREE.Vector3(-r, -.5, -r))); box.expandByPoint(p.clone().add(new THREE.Vector3(r, 3, r))); }
    }
    if (!this.postprocessed && name === 'step') {
      const p = Math.max(0, this.pairAt()), pair = this.data.pairs[p], tile = this.data.grid.tile;
      const extent = Math.max(...this.data.agents.fixed.map(a => Math.hypot(a.width, a.height))) * tile * .55;
      box = new THREE.Box3();
      const add = (row, col, r) => { const point = this.scene.point(row, col); box.expandByPoint(point.clone().addScalar(-r)); box.expandByPoint(point.clone().addScalar(r)); };
      if (pair) for (const e of [this.data.events[pair.dig], this.data.events[pair.dump]]) add(e.row, e.col, tile * 4);
      for (const [row, col] of this.data.agents.track[this.index]) add(row, col, extent);
      box.min.y = -tile; box.max.y = tile * 6;
    }
    const { target, radius } = this.fit(box, direction);
    this.scene.flyTo(target.clone().addScaledVector(direction, radius), target, { instant });
  }
}

window.TerraPlan3D = { create: (element, options) => new PlanView(element, options) };
window.TerraPostprocessedView = window.TerraPlan3D;
