import * as THREE from 'three';
import { PALETTE, earthMaterial, zoneMaterial } from './materials.js';

const RING = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]];
// Display soil units per horizontal cell. A narrow deposit stays low; broad
// connected deposits may form taller mounds. This does not redistribute soil.
export const PILE_RISE_PER_CELL = 1.6;

// Progress is a number for the whole map, or a per-cell function (row, col).
const progressAt = (progress, row, col) => typeof progress === 'function' ? progress(row, col) : progress;

function soilHeight(frame, row, col, previous, progress) {
  if (row < 0 || col < 0 || row >= frame.grid.rows || col >= frame.grid.cols || frame.maps.padding[row][col]) return 0;
  const after = frame.maps.action[row][col];
  const before = previous ? previous.maps.action[row][col] : after;
  return Math.max(0, before + (after - before) * progressAt(progress, row, col));
}

// A half-cell node blends the progress of the cells that share it.
function nodeProgress(frame, row2, col2, progress) {
  if (typeof progress !== 'function') return progress;
  const rows = row2 % 2 ? [(row2 - 1) / 2] : [row2 / 2 - 1, row2 / 2];
  const cols = col2 % 2 ? [(col2 - 1) / 2] : [col2 / 2 - 1, col2 / 2];
  let sum = 0, count = 0;
  for (const row of rows) for (const col of cols) if (row >= 0 && col >= 0 && row < frame.grid.rows && col < frame.grid.cols) { sum += progress(row, col); count++; }
  return count ? sum / count : 1;
}

// Half-cell coordinates: odd/odd is a cell center, even/even a shared corner.
// All contributors must contain soil, so mounds cannot bridge empty cells or
// spill into an obstacle/hole. The minimum is continuous as a neighbor appears
// or disappears; averaging positive contributors would pop up a shared edge.
function supportedNodeHeight(frame, row2, col2, previous, progress) {
  const rows = row2 % 2 ? [(row2 - 1) / 2] : [row2 / 2 - 1, row2 / 2];
  const cols = col2 % 2 ? [(col2 - 1) / 2] : [col2 / 2 - 1, col2 / 2];
  let height = Infinity;
  for (const row of rows) for (const col of cols) {
    height = Math.min(height, soilHeight(frame, row, col, previous, progress));
  }
  return height;
}

// Exact separable Manhattan slope envelope on the half-cell lattice. Start
// from the raw support, then only lower display vertices until neighboring
// heights differ by at most half a cell's allowed rise. Empty/negative/blocked
// cells stay empty; snapshot values and recorded soil totals are never edited.
export function pileHeightField(frame, previous = null, progress = 1, endpoints = null) {
  const rows = frame.grid.rows * 2 + 1, cols = frame.grid.cols * 2 + 1;
  const heights = new Float64Array(rows * cols), rise = PILE_RISE_PER_CELL / 2;
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    heights[row * cols + col] = supportedNodeHeight(frame, row, col, previous, progress);
  }
  for (let row = 0; row < rows; row++) {
    const start = row * cols;
    for (let col = 1; col < cols; col++) heights[start + col] = Math.min(heights[start + col], heights[start + col - 1] + rise);
    for (let col = cols - 2; col >= 0; col--) heights[start + col] = Math.min(heights[start + col], heights[start + col + 1] + rise);
  }
  for (let row = 1; row < rows; row++) for (let col = 0; col < cols; col++) {
    const index = row * cols + col; heights[index] = Math.min(heights[index], heights[index - cols] + rise);
  }
  for (let row = rows - 2; row >= 0; row--) for (let col = 0; col < cols; col++) {
    const index = row * cols + col; heights[index] = Math.min(heights[index], heights[index + cols] + rise);
  }
  if (previous && (typeof progress === 'function' || (progress > 0 && progress < 1))) {
    const start = endpoints?.start ?? pileHeightField(frame, previous, 0);
    const end = endpoints?.end ?? pileHeightField(frame);
    // Morph the capped profiles, not the unbounded quantities: a 127-unit dump
    // should grow across the whole animation rather than instantly hit its cap.
    // The minimum of two slope-bounded fields is also slope-bounded. Keeping
    // the current support envelope prevents bridging a newly exposed hole.
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
      const i = row * cols + col, p = nodeProgress(frame, row, col, progress);
      heights[i] = Math.min(heights[i], start.heights[i] + (end.heights[i] - start.heights[i]) * p);
    }
  }
  return { rows, cols, heights };
}

export function pileNodeHeight(frame, row2, col2, previous = null, progress = 1) {
  const field = pileHeightField(frame, previous, progress);
  return field.heights[row2 * field.cols + col2] ?? 0;
}

export function pileTopology(frame, previous = null) {
  const nodes = [], cells = [], lookup = new Map(), stride = frame.grid.cols * 2 + 1;
  const node = (row2, col2) => {
    const key = row2 * stride + col2;
    if (!lookup.has(key)) { lookup.set(key, nodes.length); nodes.push([row2, col2]); }
    return lookup.get(key);
  };
  for (let row = 0; row < frame.grid.rows; row++) for (let col = 0; col < frame.grid.cols; col++) {
    if (soilHeight(frame, row, col, previous, 0) <= 0 && soilHeight(frame, row, col, previous, 1) <= 0) continue;
    const center = node(row * 2 + 1, col * 2 + 1);
    const ring = RING.map(([r, c]) => node(row * 2 + r, col * 2 + c));
    const triangles = ring.flatMap((index, i) => [center, index, ring[(i + 1) % ring.length]]);
    cells.push({ row, col, center, ring, triangles });
  }
  return { nodes, cells };
}

// A continuous surface, not a collection of per-cell soil props. Each cell fan
// shares its boundary vertices with its neighbors and tapers at support edges.
export class SoilPiles extends THREE.Group {
  constructor(frame, { previous = null, unitHeight = frame.grid.tile_size_m * .48, layerSettings = {}, visibility = {}, palette = PALETTE, roughness = 0 } = {}) {
    super();
    this.frame = frame; this.previous = previous; this.unitHeight = unitHeight;
    this.endHeights = pileHeightField(frame);
    this.startHeights = previous ? pileHeightField(frame, previous, 0) : this.endHeights;
    this.endpoints = { start: this.startHeights, end: this.endHeights };
    this.progress = 1; this.topology = pileTopology(frame, previous); this.activeCells = [];
    const { rows, cols, tile_size_m: tile } = frame.grid;
    const positions = new Float32Array(this.topology.nodes.length * 3);
    const colors = new Float32Array(positions.length), shade = new THREE.Color();
    for (let i = 0; i < this.topology.nodes.length; i++) {
      const [row2, col2] = this.topology.nodes[i];
      // Optional stable jitter breaks the cell-aligned outline of loose soil.
      const jitter = (seed, edge) => edge ? 0 : (((Math.sin(row2 * 12.9898 + col2 * 78.233 + seed) * 43758.5453) % 1 + 1) % 1 - .5) * 2 * roughness * tile;
      positions[i * 3] = (col2 / 2 - cols / 2) * tile + jitter(1.7, col2 === 0 || col2 === cols * 2);
      positions[i * 3 + 2] = (row2 / 2 - rows / 2) * tile + jitter(5.3, row2 === 0 || row2 === rows * 2);
      const noise = ((row2 * 37 + col2 * 61 + row2 * col2 * 7) % 29) / 29;
      shade.set(palette.loose).multiplyScalar(.95 + noise * .1); shade.toArray(colors, i * 3);
    }
    this.positions = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', this.positions);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    this.surface = new THREE.Mesh(geometry, earthMaterial('pile', { vertexColors: true, flatShading: true, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -2 }, palette));
    this.surface.name = 'connected-soil-piles'; this.surface.castShadow = true; this.surface.receiveShadow = true;
    this.surface.userData.soilPiles = this; this.add(this.surface);
    this.layerSettings = layerSettings; this.overlays = {};
    for (const [name, settings] of Object.entries(layerSettings)) {
      const overlayGeometry = new THREE.BufferGeometry(); overlayGeometry.setAttribute('position', this.positions);
      const material = zoneMaterial({ color: settings.color, opacity: settings.opacity * .7, pattern: settings.pattern, polygonOffset: true, polygonOffsetFactor: -2 });
      const layer = new THREE.Mesh(overlayGeometry, material);
      layer.position.y = tile * (.008 + Object.keys(this.overlays).length * .003);
      layer.renderOrder = 3 + Object.keys(this.overlays).length; layer.visible = !!visibility[name];
      layer.frustumCulled = false; layer.userData.skipAO = true; this.overlays[name] = layer; this.add(layer);
    }
    const gridGeometry = new THREE.BufferGeometry(); gridGeometry.setAttribute('position', this.positions);
    this.gridLines = new THREE.LineSegments(gridGeometry, new THREE.LineBasicMaterial({ color: 0x71512f, transparent: true, opacity: .25, depthWrite: false }));
    this.gridLines.position.y = tile * .022; this.gridLines.renderOrder = 12;
    this.gridLines.visible = !!visibility.grid; this.gridLines.frustumCulled = false; this.add(this.gridLines);
    this.update(1);
  }

  nodeHeight(row2, col2) { return (this.heights.heights[row2 * this.heights.cols + col2] ?? 0) * this.unitHeight; }

  endpointHeight(row, col, start = false) {
    const field = start ? this.startHeights : this.endHeights;
    return (field.heights[(row * 2 + 1) * field.cols + col * 2 + 1] ?? 0) * this.unitHeight;
  }

  update(progress = 1) {
    this.progress = progress;
    this.heights = typeof progress === 'function' ? pileHeightField(this.frame, this.previous, progress, this.endpoints) : progress <= 0 ? this.startHeights : progress >= 1 ? this.endHeights : pileHeightField(this.frame, this.previous, progress, this.endpoints);
    const positions = this.positions.array;
    for (let i = 0; i < this.topology.nodes.length; i++) {
      const [row2, col2] = this.topology.nodes[i]; positions[i * 3 + 1] = this.nodeHeight(row2, col2);
    }
    this.positions.needsUpdate = true;
    const active = this.topology.cells.filter(cell => soilHeight(this.frame, cell.row, cell.col, this.previous, progress) > 0);
    const changed = active.length !== this.activeCells.length || active.some((cell, i) => cell !== this.activeCells[i]);
    this.activeCells = active; this.surface.visible = active.length > 0;
    if (changed || !this.surface.geometry.index) {
      this.surface.geometry.setIndex(active.flatMap(cell => cell.triangles));
      for (const [name, layer] of Object.entries(this.overlays)) {
        const settings = this.layerSettings[name], map = this.frame.maps[settings.map];
        layer.geometry.setIndex(active.filter(cell => map != null && settings.test(map[cell.row][cell.col])).flatMap(cell => cell.triangles));
      }
      this.gridLines.geometry.setIndex(active.flatMap(cell => cell.ring.flatMap((index, i) => [index, cell.ring[(i + 1) % cell.ring.length]])));
    }
    this.surface.geometry.computeVertexNormals(); this.surface.geometry.computeBoundingSphere();
  }

  setLayer(name, visible) {
    if (name === 'grid') this.gridLines.visible = visible;
    else if (this.overlays[name]) this.overlays[name].visible = visible && this.frame.maps[this.layerSettings[name].map] != null;
  }

  cellForHit(hit) {
    const cell = this.activeCells[Math.floor(hit.faceIndex / 8)];
    return cell ? { row: cell.row, col: cell.col } : null;
  }

  dispose() {
    this.traverse(object => { object.geometry?.dispose(); if (object.material) object.material.dispose(); });
    this.clear();
  }
}
