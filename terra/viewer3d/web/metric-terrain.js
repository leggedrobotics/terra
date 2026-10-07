import * as THREE from 'three';

/** Exact stepped surface with adjacent coplanar faces merged; no resampling. */
export function metricTerrainGeometry(frame, heights, unitHeight, floor, palette) {
  const { rows, cols, tile_size_m: tile } = frame.grid;
  const positions = [], colors = [];
  const sand = new THREE.Color(palette.sand), loose = new THREE.Color(palette.loose), dug = palette.dug.map(hex => new THREE.Color(hex)), wall = new THREE.Color(0xffffff);
  const kindAt = (r, c) => frame.metric.loose[r][c] > 1e-6 ? 1 : heights[r][c] < 0 ? 2 + Math.max(0, Math.min(dug.length - 1, Math.floor(-heights[r][c]) - 1)) : 0;
  const inks = [sand, loose, ...dug];
  const quad = (a, b, c, d, color, reverse = false) => {
    for (const p of reverse ? [a, d, b, b, d, c] : [a, b, d, b, c, d]) {
      positions.push(...p); colors.push(color.r, color.g, color.b);
    }
  };
  const top = rect => {
    const x0 = (rect.col - cols / 2) * tile, x1 = (rect.end - cols / 2) * tile;
    const z0 = (rows / 2 - rect.bottom) * tile, z1 = (rows / 2 - rect.row) * tile, y = rect.height * unitHeight;
    quad([x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0], inks[rect.kind]);
  };
  let active = new Map();
  for (let row = 0; row < rows; row++) {
    const next = new Map();
    for (let col = 0; col < cols;) {
      const height = heights[row][col], kind = kindAt(row, col), start = col++;
      while (col < cols && heights[row][col] === height && kindAt(row, col) === kind) col++;
      const key = `${start}:${col}:${height}:${kind}`, rect = active.get(key) ?? { row, col: start, end: col, height, kind };
      rect.bottom = row + 1; next.set(key, rect); active.delete(key);
    }
    for (const rect of active.values()) top(rect);
    active = next;
  }
  for (const rect of active.values()) top(rect);
  const topCount = positions.length / 3, base = floor / unitHeight;
  // Only exposed vertical faces exist. Internal faces between equal cells are absent.
  for (let row = 0; row <= rows; row++) for (let col = 0; col < cols;) {
    const a = row > 0 ? heights[row - 1][col] : base, b = row < rows ? heights[row][col] : base, start = col++;
    if (a === b) continue;
    while (col < cols && (row > 0 ? heights[row - 1][col] : base) === a && (row < rows ? heights[row][col] : base) === b) col++;
    const x0 = (start - cols / 2) * tile, x1 = (col - cols / 2) * tile, z = (rows / 2 - row) * tile;
    const low = Math.min(a, b) * unitHeight, high = Math.max(a, b) * unitHeight;
    quad([x0, low, z], [x1, low, z], [x1, high, z], [x0, high, z], wall, a > b);
  }
  for (let col = 0; col <= cols; col++) for (let row = 0; row < rows;) {
    const a = col > 0 ? heights[row][col - 1] : base, b = col < cols ? heights[row][col] : base, start = row++;
    if (a === b) continue;
    while (row < rows && (col > 0 ? heights[row][col - 1] : base) === a && (col < cols ? heights[row][col] : base) === b) row++;
    const z0 = (rows / 2 - row) * tile, z1 = (rows / 2 - start) * tile, x = (col - cols / 2) * tile;
    const low = Math.min(a, b) * unitHeight, high = Math.max(a, b) * unitHeight;
    quad([x, low, z0], [x, low, z1], [x, high, z1], [x, high, z0], wall, a > b);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.addGroup(0, topCount, 0); geometry.addGroup(topCount, positions.length / 3 - topCount, 1);
  geometry.computeVertexNormals(); geometry.computeBoundingSphere();
  return geometry;
}
