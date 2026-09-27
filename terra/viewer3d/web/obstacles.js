import * as THREE from 'three';

// Original procedural scenery. Footprints are planned entirely inside padding;
// these props never turn free terrain or holes into visual obstacles.
function hash(value, salt = 0) {
  let result = (value ^ Math.imul(salt + 1, 0x9e3779b1)) >>> 0;
  result = Math.imul(result ^ (result >>> 16), 0x85ebca6b);
  result = Math.imul(result ^ (result >>> 13), 0xc2b2ae35);
  return (result ^ (result >>> 16)) >>> 0;
}
const noise = (seed, salt) => hash(seed, salt) / 0xffffffff;
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function inspectMask(padding) {
  if (!Array.isArray(padding) || !padding.length || !Array.isArray(padding[0]) || !padding[0].length) throw new Error('Obstacle padding must be a nonempty rectangular array.');
  const rows = padding.length, cols = padding[0].length;
  if (rows > 128 || cols > 128 || padding.some(row => !Array.isArray(row) || row.length !== cols || row.some(value => ![0, 1, false, true].includes(value)))) throw new Error('Obstacle padding must contain aligned 0/1 cells, at most 128 × 128.');
  return { rows, cols };
}

function componentsOf(padding, rows, cols) {
  const visited = new Uint8Array(rows * cols), components = [];
  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    const first = row * cols + col;
    if (!padding[row][col] || visited[first]) continue;
    const cells = [[row, col]]; visited[first] = 1;
    let minRow = row, maxRow = row, minCol = col, maxCol = col;
    for (let cursor = 0; cursor < cells.length; cursor++) {
      const [r, c] = cells[cursor];
      minRow = Math.min(minRow, r); maxRow = Math.max(maxRow, r); minCol = Math.min(minCol, c); maxCol = Math.max(maxCol, c);
      for (const [rr, cc] of [[r - 1, c], [r, c - 1], [r, c + 1], [r + 1, c]]) {
        if (rr < 0 || cc < 0 || rr >= rows || cc >= cols) continue;
        const index = rr * cols + cc;
        if (padding[rr][cc] && !visited[index]) { visited[index] = 1; cells.push([rr, cc]); }
      }
    }
    // Cell-based seeds keep existing components stable if another component is
    // added elsewhere. Neither animation time nor terrain heights affect style.
    let seed = 2166136261;
    for (const [r, c] of [...cells].sort((a, b) => a[0] - b[0] || a[1] - b[1])) seed = Math.imul(seed ^ (r * 131 + c), 16777619) >>> 0;
    components.push({ cells, minRow, maxRow, minCol, maxCol, seed });
  }
  return components;
}

function largestRectangle(mask, rows, cols) {
  const heights = new Uint16Array(cols);
  let best = null, bestArea = 0;
  for (let row = 0; row < rows; row++) {
    const stack = [];
    for (let col = 0; col < cols; col++) heights[col] = mask[row * cols + col] ? heights[col] + 1 : 0;
    for (let col = 0; col <= cols; col++) {
      const height = col < cols ? heights[col] : 0;
      let start = col;
      while (stack.length && stack[stack.length - 1].height > height) {
        const entry = stack.pop(), area = entry.height * (col - entry.start);
        const candidate = { row: row - entry.height + 1, col: entry.start, rows: entry.height, cols: col - entry.start };
        if (area > bestArea || (area === bestArea && (candidate.row < best.row || (candidate.row === best.row && candidate.col < best.col)))) { best = candidate; bestArea = area; }
        start = entry.start;
      }
      if (height && (!stack.length || stack[stack.length - 1].height < height)) stack.push({ start, height });
    }
  }
  return best;
}

function expandInsideComponent(rectangle, member, rows, cols) {
  let rectangleNow = { ...rectangle };
  function filled(rect) {
    if (rect.row < 0 || rect.col < 0 || rect.row + rect.rows > rows || rect.col + rect.cols > cols) return false;
    for (let row = rect.row; row < rect.row + rect.rows; row++) for (let col = rect.col; col < rect.col + rect.cols; col++) if (!member[row * cols + col]) return false;
    return true;
  }
  for (;;) {
    const r = rectangleNow;
    const candidates = [
      { ...r, row: r.row - 1, rows: r.rows + 1 },
      { ...r, col: r.col - 1, cols: r.cols + 1 },
      { ...r, rows: r.rows + 1 },
      { ...r, cols: r.cols + 1 },
    ].filter(filled).sort((a, b) => b.rows * b.cols - a.rows * a.cols || a.row - b.row || a.col - b.col);
    if (!candidates.length) return rectangleNow;
    rectangleNow = candidates[0];
  }
}

/** Plan a deterministic rectangle cover; every covered cell is actually blocked.
 * Rectangles may overlap within a component. That lets thin branches join a
 * neighboring boulder instead of leaving one miniature rock per leftover cell.
 */
export function planObstacleFootprints(padding) {
  const { rows, cols } = inspectMask(padding), plans = [];
  const components = componentsOf(padding, rows, cols);
  for (const [componentIndex, component] of components.entries()) {
    const height = component.maxRow - component.minRow + 1, width = component.maxCol - component.minCol + 1;
    const member = new Uint8Array(height * width);
    for (const [row, col] of component.cells) member[(row - component.minRow) * width + col - component.minCol] = 1;
    const uncovered = member.slice(); let remaining = component.cells.length;
    const rectangularity = component.cells.length / (height * width);
    while (remaining) {
      const rect = expandInsideComponent(largestRectangle(uncovered, height, width), member, height, width);
      for (let row = rect.row; row < rect.row + rect.rows; row++) for (let col = rect.col; col < rect.col + rect.cols; col++) {
        const index = row * width + col;
        if (uncovered[index]) { uncovered[index] = 0; remaining--; }
      }
      const global = { row: rect.row + component.minRow, col: rect.col + component.minCol, rows: rect.rows, cols: rect.cols };
      const seed = hash(component.seed, global.row * 131 + global.col);
      const shortSide = Math.min(rect.rows, rect.cols), longSide = Math.max(rect.rows, rect.cols);
      const siteStyle = (component.minRow + component.minCol + height + width) % 3;
      const containerSite = rectangularity >= .9 && shortSide >= 3 && longSide >= 6 && rect.rows * rect.cols >= 24 && siteStyle !== 0;
      if (containerSite) {
        // Broad rectangular compounds fit a small row of conventional long
        // containers, rather than one implausibly square shipping container.
        const acrossRows = rect.cols >= rect.rows;
        const count = Math.min(3, Math.floor(shortSide / 3), Math.max(1, Math.round(shortSide / (longSide * .37))));
        for (let part = 0; part < count; part++) {
          const start = Math.floor(shortSide * part / count), end = Math.floor(shortSide * (part + 1) / count);
          const stripe = { ...global };
          if (acrossRows) { stripe.row += start; stripe.rows = end - start; } else { stripe.col += start; stripe.cols = end - start; }
          plans.push({ ...stripe, kind: 'container', component: componentIndex, seed: hash(seed, part) });
        }
      } else plans.push({ ...global, kind: 'boulder', component: componentIndex, seed });
    }
  }
  return plans;
}

function boulderGeometry(width, depth, height, seed, mossy = true) {
  const sides = 9, rings = [], positions = [], colors = [];
  const stone = new THREE.Color().setHex([0x9a978c, 0xa39a88, 0x8d948e][seed % 3]);
  const moss = new THREE.Color(0x86a857), edge = new THREE.Vector3(), other = new THREE.Vector3(), normal = new THREE.Vector3();
  for (let ring = 0; ring < 3; ring++) {
    const points = [];
    for (let index = 0; index < sides; index++) {
      const angle = (index + noise(seed, index) * .13) * Math.PI * 2 / sides;
      const radius = ring === 2 ? .44 + noise(seed, index + 20) * .22 : .85 + noise(seed, index + ring * sides + 40) * .14;
      points.push(new THREE.Vector3(Math.cos(angle) * width / 2 * radius, ring === 0 ? 0 : height * (ring === 1 ? .38 + noise(seed, index + 70) * .1 : .76 + noise(seed, index + 90) * .18), Math.sin(angle) * depth / 2 * radius));
    }
    rings.push(points);
  }
  function triangle(a, b, c) {
    // Stylized stone: sunlit, sometimes mossy tops over cooler flanks.
    normal.crossVectors(edge.subVectors(b, a), other.subVectors(c, a)).normalize();
    const facing = Math.abs(normal.y), shade = stone.clone().multiplyScalar(.84 + noise(seed, positions.length) * .2 + facing * .12);
    if (mossy && facing > .72 && noise(seed, positions.length + 7) < .45) shade.lerp(moss, .55);
    for (const point of [a, b, c]) { positions.push(point.x, point.y, point.z); colors.push(shade.r, shade.g, shade.b); }
  }
  for (let index = 0; index < sides; index++) {
    const next = (index + 1) % sides;
    for (let ring = 0; ring < 2; ring++) {
      triangle(rings[ring][index], rings[ring + 1][index], rings[ring + 1][next]);
      triangle(rings[ring][index], rings[ring + 1][next], rings[ring][next]);
    }
    triangle(new THREE.Vector3(0, 0, 0), rings[0][index], rings[0][next]);
    triangle(rings[2][index], new THREE.Vector3(0, height, 0), rings[2][next]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3)); geometry.computeVertexNormals(); geometry.computeBoundingBox();
  return geometry;
}

function addBox(parent, geometry, material, x, y, z, width, height, depth) {
  const mesh = new THREE.Mesh(geometry, material); mesh.position.set(x, y, z); mesh.scale.set(width, height, depth); mesh.castShadow = true; mesh.receiveShadow = true; parent.add(mesh); return mesh;
}

function addContainer(parent, plan, tile, rise, context) {
  const long = Math.max(plan.rows, plan.cols) * tile, across = Math.min(plan.rows, plan.cols) * tile;
  const length = long * .94, width = Math.min(across * .88, length * .42), height = clamp(width * .94, tile * .7, tile * 3.6);
  const container = new THREE.Group(); container.rotation.y = plan.rows > plan.cols ? Math.PI / 2 : 0; parent.add(container);
  const base = rise + tile * .12;
  context.box ||= new THREE.BoxGeometry(1, 1, 1);
  context.metal ||= new THREE.MeshStandardMaterial({ color: 0x77807b, roughness: .64, metalness: .25 });
  context.foundation ||= new THREE.MeshStandardMaterial({ color: 0x8e9184, roughness: 1 });
  const paint = new THREE.MeshStandardMaterial({ color: (context.paper ? [0x8d918f, 0x7f8584, 0x989a95] : [0xa45f43, 0x517e7b, 0x627f91])[plan.seed % 3], roughness: .77, metalness: .15 });
  const light = paint.clone(); light.color.multiplyScalar(1.12);
  addBox(container, context.box, context.foundation, 0, base / 2, 0, length + tile * .06, base, width + tile * .09);
  addBox(container, context.box, paint, 0, base + height / 2, 0, length, height, width);
  addBox(container, context.box, light, 0, base + height + tile * .025, 0, length, tile * .05, width);
  const ribCount = Math.max(4, Math.round(length / (tile * .45))), ribs = new THREE.InstancedMesh(context.box, light, ribCount * 2), dummy = new THREE.Object3D();
  ribs.castShadow = true; ribs.receiveShadow = true;
  for (let side = 0; side < 2; side++) for (let rib = 0; rib < ribCount; rib++) {
    dummy.position.set(length * (-.46 + .92 * rib / (ribCount - 1)), base + height / 2, (side ? 1 : -1) * (width / 2 + tile * .013));
    dummy.scale.set(tile * .065, height * .94, tile * .033); dummy.updateMatrix(); ribs.setMatrixAt(side * ribCount + rib, dummy.matrix);
  }
  container.add(ribs);
  for (const side of [-1, 1]) {
    addBox(container, context.box, light, length / 2 + tile * .02, base + height * .50, side * width * .237, tile * .04, height * .88, width * .45);
    addBox(container, context.box, context.metal, length / 2 + tile * .047, base + height * .50, side * width * .17, tile * .028, height * .79, tile * .038);
    for (const end of [-1, 1]) addBox(container, context.box, context.metal, end * (length / 2 - tile * .045), base + height / 2, side * (width / 2 - tile * .036), tile * .09, height, tile * .075);
  }
}

/** Create scene props in Terra's centered X=column, Z=row coordinates.
 * Add the group to the world and dispose its geometry/materials with the world.
 * Recreate with the current unitHeight when height exaggeration changes.
 */
export function createObstacleProps(frame, options = {}) {
  if (typeof options === 'number') options = { tile: options };
  const tile = options.tile ?? frame.grid.tile_size_m, unitHeight = options.unitHeight ?? tile * .48;
  if (!Number.isFinite(tile) || tile <= 0 || !Number.isFinite(unitHeight) || unitHeight <= 0) throw new Error('Obstacle display scale must be finite and positive.');
  const { rows, cols } = inspectMask(frame.maps.padding);
  if (rows !== frame.grid.rows || cols !== frame.grid.cols || !Array.isArray(frame.maps.action) || frame.maps.action.length !== rows || frame.maps.action.some(row => !Array.isArray(row) || row.length !== cols || row.some(value => !Number.isFinite(value)))) throw new Error('Obstacle terrain must match the frame grid and contain finite heights.');
  const plans = planObstacleFootprints(frame.maps.padding), root = new THREE.Group(), context = { paper: options.style === 'paper' };
  root.name = 'Terra obstacle props'; root.userData.footprints = plans;
  for (const plan of plans) {
    const prop = new THREE.Group(); prop.name = `${plan.kind}-${plan.row}-${plan.col}`; prop.userData.footprint = { ...plan };
    let low = Infinity, high = -Infinity;
    for (let row = plan.row; row < plan.row + plan.rows; row++) for (let col = plan.col; col < plan.col + plan.cols; col++) { const height = frame.maps.action[row][col] * unitHeight; low = Math.min(low, height); high = Math.max(high, height); }
    prop.position.set((plan.col + plan.cols / 2 - cols / 2) * tile, low + tile * .008, (plan.row + plan.rows / 2 - rows / 2) * tile);
    if (plan.kind === 'container') addContainer(prop, plan, tile, high - low, context);
    else {
      context.stone ||= new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, flatShading: true });
      const height = clamp(Math.min(plan.rows, plan.cols) * tile * .62, tile * .52, tile * 3.4) + high - low;
      const mesh = new THREE.Mesh(boulderGeometry(plan.cols * tile * .96, plan.rows * tile * .96, height, plan.seed, !context.paper), context.stone); mesh.castShadow = true; mesh.receiveShadow = true; prop.add(mesh);
    }
    root.add(prop);
  }
  return root;
}
