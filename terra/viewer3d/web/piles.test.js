import test from 'node:test';
import assert from 'node:assert/strict';
import { SoilPiles, pileNodeHeight, pileTopology, pileHeightField, PILE_RISE_PER_CELL } from './piles.js';

function frame(action, padding = action.map(row => row.map(() => 0))) {
  return { grid: { rows: action.length, cols: action[0].length, tile_size_m: 1 }, maps: { action, padding, target: action.map(row => row.map(() => 1)) } };
}

test('adjacent soil cells share slope-limited vertices without changing recorded heights', () => {
  const state = frame([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]]);
  const original = structuredClone(state), topology = pileTopology(state);
  assert.equal(topology.cells.length, 2);
  const common = topology.cells[0].ring.filter(index => topology.cells[1].ring.includes(index));
  assert.equal(common.length, 3);
  assert.equal(pileNodeHeight(state, 3, 3), .8); assert.equal(pileNodeHeight(state, 3, 5), .8);
  assert.equal(pileNodeHeight(state, 3, 4), .8);
  assert.deepEqual(state, original);
});

test('pile boundaries taper to zero and never cover holes or obstacle cells', () => {
  const state = frame([[1, 1, 1], [1, 0, 1], [1, 1, 2]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]]);
  const topology = pileTopology(state);
  assert.equal(topology.cells.length, 7);
  assert.ok(!topology.cells.some(cell => cell.row === 1 && cell.col === 1));
  assert.ok(!topology.cells.some(cell => cell.row === 0 && cell.col === 2));
  assert.equal(pileNodeHeight(state, 0, 1), 0);
  assert.equal(pileNodeHeight(state, 2, 2), 0);
  assert.equal(pileNodeHeight(state, 1, 4), 0);
});

test('dump grows one connected surface; reverse seek removes it without stale triangles', () => {
  const before = frame([[0, 0], [0, 0]]), after = frame([[2, 1], [0, 0]]);
  const settings = { dump: { map: 'target', test: value => value > 0, color: 0x00ff00, opacity: .5 } };
  const piles = new SoilPiles(after, { previous: before, unitHeight: .5, layerSettings: settings, visibility: { dump: true, grid: true } });
  piles.update(0); assert.equal(piles.surface.geometry.index.count, 0); assert.equal(piles.surface.visible, false);
  piles.update(.25); assert.equal(piles.nodeHeight(1, 1), .1);
  piles.update(.5); assert.equal(piles.surface.geometry.index.count, 48); assert.equal(piles.nodeHeight(1, 1), .2);
  assert.equal(piles.overlays.dump.geometry.attributes.position, piles.surface.geometry.attributes.position);
  assert.deepEqual(piles.cellForHit({ faceIndex: 9 }), { row: 0, col: 1 });
  piles.update(1); assert.equal(piles.nodeHeight(1, 1), .4);
  piles.update(0); assert.equal(piles.surface.geometry.index.count, 0); assert.equal(piles.overlays.dump.geometry.index.count, 0);
  piles.dispose();
});

test('removing positive soil exposes the negative cut instead of leaving a flat cap', () => {
  const before = frame([[1]]), after = frame([[-1]]), piles = new SoilPiles(after, { previous: before });
  piles.update(.25); assert.ok(piles.surface.visible);
  piles.update(.5); assert.equal(piles.surface.visible, false);
  piles.update(1); assert.equal(piles.surface.geometry.index.count, 0);
  piles.dispose();
});

test('pile geometry remains finite with upward faces and vertical exaggeration', () => {
  const piles = new SoilPiles(frame([[3, 1], [1, 2]]), { unitHeight: 2 });
  const geometry = piles.surface.geometry;
  assert.ok([...geometry.attributes.position.array].every(Number.isFinite));
  assert.ok([...geometry.attributes.normal.array].every(Number.isFinite));
  assert.equal(piles.nodeHeight(1, 1), 1.6);
  const index = geometry.index.array, p = geometry.attributes.position.array;
  for (let i = 0; i < index.length; i += 3) {
    const [a, b, c] = [index[i], index[i + 1], index[i + 2]].map(vertex => [p[vertex * 3], p[vertex * 3 + 2]]);
    assert.ok((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1]) > 0);
  }
  piles.dispose();
});

test('a maximum single-cell dump cannot form a tall spike, at any animation sample', () => {
  const before = frame([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
  const after = frame([[0, 0, 0], [0, 127, 0], [0, 0, 0]]), original = structuredClone(after);
  const piles = new SoilPiles(after, { previous: before });
  for (let sample = 0; sample <= 20; sample++) {
    piles.update(sample / 20);
    assert.ok(piles.nodeHeight(3, 3) <= .384 + 1e-12);
    assert.ok(Math.abs(piles.nodeHeight(3, 3) - .384 * sample / 20) < 1e-12);
    assert.ok([...piles.positions.array].every(Number.isFinite));
  }
  assert.deepEqual(after, original); assert.equal(after.maps.action[1][1], 127);
  assert.equal(piles.endpointHeight(1, 1), .384); assert.equal(piles.endpointHeight(1, 1, true), 0);
  piles.dispose();
});

test('wide supported piles rise above narrow piles without a global height cap', () => {
  const state = frame(Array.from({ length: 7 }, (_, row) => Array.from({ length: 7 }, (_, col) => row && row < 6 && col && col < 6 ? 8 : 0)));
  assert.ok(pileNodeHeight(state, 7, 7) > pileNodeHeight(state, 3, 3));
  assert.equal(pileNodeHeight(state, 7, 7), 4);
});

test('shared edges grow continuously when a high pile gains a low neighboring cell', () => {
  const before = frame([[127, 0], [0, 0]]), after = frame([[127, 2], [0, 0]]);
  assert.equal(pileNodeHeight(after, 1, 2, before, 0), 0);
  assert.ok(pileNodeHeight(after, 1, 2, before, 1e-6) <= 2e-6);
});

test('slope envelope obeys local bounds on irregular support and preserves holes', () => {
  const action = Array.from({ length: 13 }, (_, r) => Array.from({ length: 17 }, (_, c) => ((r * 41 + c * 23) % 11) < 3 ? -1 : (r * c * 7 + 13) % 128));
  const padding = action.map((row, r) => row.map((_, c) => (r + c * 3) % 19 === 0 ? 1 : 0));
  const state = frame(action, padding), original = structuredClone(state);
  const field = pileHeightField(state), delta = PILE_RISE_PER_CELL / 2 + 1e-12;
  for (let r = 0; r < field.rows; r++) for (let c = 0; c < field.cols; c++) {
    const i = r * field.cols + c, height = field.heights[i];
    assert.ok(Number.isFinite(height) && height >= 0);
    if (r) assert.ok(Math.abs(height - field.heights[i - field.cols]) <= delta);
    if (c) assert.ok(Math.abs(height - field.heights[i - 1]) <= delta);
  }
  for (let r = 0; r < action.length; r++) for (let c = 0; c < action[0].length; c++) {
    const height = field.heights[(r * 2 + 1) * field.cols + c * 2 + 1];
    assert.ok(height <= Math.max(0, action[r][c]));
    if (action[r][c] <= 0 || padding[r][c]) assert.equal(height, 0);
  }
  assert.deepEqual(state, original);
});
