import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { createObstacleProps, planObstacleFootprints } from './obstacles.js';

const mask = (rows, cols) => Array.from({ length: rows }, () => Array(cols).fill(0));
function fill(padding, row, col, rows, cols) {
  for (let r = row; r < row + rows; r++) for (let c = col; c < col + cols; c++) padding[r][c] = 1;
  return padding;
}
function verifyCover(padding) {
  const original = structuredClone(padding), plans = planObstacleFootprints(padding), covered = mask(padding.length, padding[0].length);
  for (const plan of plans) {
    assert.ok(plan.rows > 0 && plan.cols > 0);
    assert.ok(plan.row >= 0 && plan.col >= 0 && plan.row + plan.rows <= padding.length && plan.col + plan.cols <= padding[0].length);
    for (let row = plan.row; row < plan.row + plan.rows; row++) for (let col = plan.col; col < plan.col + plan.cols; col++) {
      assert.equal(!!padding[row][col], true, `Prop bridges free cell ${row},${col}`);
      covered[row][col] = 1;
    }
    if (plan.rows * plan.cols === 1) {
      const { row, col } = plan;
      for (const [r, c] of [[row - 1, col], [row + 1, col], [row, col - 1], [row, col + 1]]) if (r >= 0 && c >= 0 && r < padding.length && c < padding[0].length) assert.equal(!!padding[r][c], false, 'Only isolated cells should receive cell-sized props');
    }
  }
  assert.deepEqual(covered, padding.map(row => row.map(Number)), 'Every blocked cell should belong to a planned footprint');
  assert.deepEqual(padding, original, 'Planning must not mutate the map');
  assert.deepEqual(plans, planObstacleFootprints(padding), 'Planning must be deterministic');
  return plans;
}
function frameFor(padding) {
  return { grid: { rows: padding.length, cols: padding[0].length, tile_size_m: .6 }, maps: { padding, action: mask(padding.length, padding[0].length) } };
}
function dispose(group) {
  const geometries = new Set(), materials = new Set();
  group.traverse(object => { if (object.geometry) geometries.add(object.geometry); if (object.material) materials.add(object.material); });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
}

test('large rectangular blocked components become a few coherent assets', () => {
  const padding = fill(fill(mask(64, 64), 7, 47, 8, 9), 47, 7, 8, 10);
  const plans = verifyCover(padding);
  assert.ok(plans.length <= 6, `Expected a few props, got ${plans.length}`);
  assert.ok(plans.every(plan => plan.rows * plan.cols >= 24));
  assert.ok(plans.some(plan => plan.kind === 'container'));
  assert.ok(plans.some(plan => plan.kind === 'boulder'));
  assert.ok(new Set(plans.map(plan => plan.component)).size === 2);
});

test('ring holes, concave bays and disconnected regions remain traversable', () => {
  const padding = mask(15, 20);
  fill(padding, 1, 1, 8, 1); fill(padding, 1, 1, 1, 8); fill(padding, 8, 1, 1, 8); fill(padding, 1, 8, 8, 1);
  fill(padding, 2, 12, 10, 2); fill(padding, 10, 12, 2, 7);
  const plans = verifyCover(padding);
  assert.ok(plans.length < 12);
  assert.ok(plans.every(plan => !(plan.row <= 4 && plan.row + plan.rows > 4 && plan.col <= 4 && plan.col + plan.cols > 4)));
  assert.equal(new Set(plans.map(plan => plan.component)).size, 2);
});

test('narrow branches form long rocks while truly isolated singles remain separate', () => {
  const padding = mask(13, 21);
  fill(padding, 1, 1, 1, 13); fill(padding, 1, 7, 9, 1); fill(padding, 5, 3, 1, 9);
  padding[11][17] = 1; padding[12][18] = 1; // Diagonal contact is not a connected obstruction.
  const plans = verifyCover(padding);
  const singles = plans.filter(plan => plan.rows * plan.cols === 1);
  assert.equal(singles.length, 2);
  assert.ok(plans.length <= 7);
});

test('staircase branches do not produce one rock per remaining cell', () => {
  const padding = mask(12, 12);
  for (let i = 1; i < 10; i++) { padding[i][i] = 1; padding[i][i + 1] = 1; }
  const plans = verifyCover(padding);
  assert.ok(plans.every(plan => plan.rows * plan.cols >= 2));
  assert.ok(plans.length < 18);
});

test('component style is stable when terrain or unrelated obstacles change', () => {
  const padding = fill(mask(20, 30), 8, 15, 8, 10);
  const before = planObstacleFootprints(padding);
  const added = structuredClone(padding); added[0][0] = 1;
  const after = planObstacleFootprints(added).filter(plan => plan.row >= 8).map(({ component, ...plan }) => plan);
  assert.deepEqual(after, before.map(({ component, ...plan }) => plan));
  const frame = frameFor(padding), original = structuredClone(frame);
  const props = createObstacleProps(frame, { unitHeight: .9 });
  assert.deepEqual(frame, original);
  dispose(props);
});

test('all geometry remains finite and inside the occupied-cell rectangles', () => {
  const padding = fill(fill(mask(24, 34), 2, 2, 8, 9), 12, 22, 10, 8);
  fill(padding, 13, 2, 1, 9); fill(padding, 13, 10, 7, 1); padding[22][1] = 1;
  const frame = frameFor(padding); frame.maps.action[2][2] = -2; frame.maps.action[3][3] = 4;
  const original = structuredClone(frame), tile = .73, group = createObstacleProps(frame, { tile, unitHeight: 1.4 });
  group.updateMatrixWorld(true);
  assert.equal(group.children.length, group.userData.footprints.length);
  for (const prop of group.children) {
    const plan = prop.userData.footprint, bounds = new THREE.Box3().setFromObject(prop), epsilon = 1e-5;
    assert.ok(bounds.min.x >= (plan.col - frame.grid.cols / 2) * tile - epsilon);
    assert.ok(bounds.max.x <= (plan.col + plan.cols - frame.grid.cols / 2) * tile + epsilon);
    assert.ok(bounds.min.z >= (plan.row - frame.grid.rows / 2) * tile - epsilon);
    assert.ok(bounds.max.z <= (plan.row + plan.rows - frame.grid.rows / 2) * tile + epsilon);
    assert.ok([...bounds.min.toArray(), ...bounds.max.toArray()].every(Number.isFinite));
    prop.traverse(object => {
      if (object.geometry) assert.ok([...object.geometry.attributes.position.array].every(Number.isFinite));
      assert.ok(object.matrixWorld.elements.every(Number.isFinite));
    });
  }
  assert.deepEqual(frame, original);
  dispose(group);
});

test('empty and irregular generated masks preserve the occupancy contract', () => {
  assert.deepEqual(verifyCover(mask(3, 8)), []);
  let value = 871;
  for (let sample = 0; sample < 20; sample++) {
    const padding = mask(12, 16);
    for (const row of padding) for (let col = 0; col < row.length; col++) { value = (Math.imul(value, 1664525) + 1013904223) >>> 0; row[col] = value % 5 < 3 ? 1 : 0; }
    verifyCover(padding);
  }
});
