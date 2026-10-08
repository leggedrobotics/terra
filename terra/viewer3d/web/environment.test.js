import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createEnvironment } from './environment.js';
import { Effects } from './effects.js';

const frame = (rows, cols, tile) => ({ grid: { rows, cols, tile_size_m: tile } });

test('surroundings stay outside every Terra cell', () => {
  for (const style of ['diorama', 'paper']) for (const [rows, cols, tile] of [[64, 64, .5714], [32, 48, .5], [128, 128, 1.25]]) {
    const root = createEnvironment(frame(rows, cols, tile), { style }), hx = cols * tile / 2, hz = rows * tile / 2;
    try {
      root.setFloor(-2); root.updateMatrixWorld(true);
      const box = new THREE.Box3(), matrix = new THREE.Matrix4(), point = new THREE.Box3();
      root.traverse(object => {
        if (!object.isMesh || object.name === 'island-turf' || object.name === 'island-underside') return;
        const count = object.isInstancedMesh ? object.count : 1;
        for (let i = 0; i < count; i++) {
          if (object.isInstancedMesh) { object.getMatrixAt(i, matrix); matrix.premultiply(object.matrixWorld); } else matrix.copy(object.matrixWorld);
          object.geometry.computeBoundingBox(); point.copy(object.geometry.boundingBox).applyMatrix4(matrix);
          box.copy(point);
          const inside = box.max.x > -hx && box.min.x < hx && box.max.z > -hz && box.min.z < hz && box.max.y > -1;
          assert.ok(!inside, `${object.name || object.type} ${i} overlaps the ${rows}×${cols} grid: ${box.min.toArray().map(v => v.toFixed(2))} ${box.max.toArray().map(v => v.toFixed(2))} (hx ${hx})`);
        }
      });
      // Clouds orbit outside the island, so a top view never hides the grid.
      root.update(123.4);
    } finally { root.dispose(); }
  }
});

test('paper plinth sits exactly under the grid, below the terrain floor', () => {
  const root = createEnvironment(frame(32, 48, .5), { style: 'paper' });
  try {
    root.setFloor(-3); root.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(root.getObjectByName('plinth'));
    assert.ok(Math.abs(bounds.max.y + 3) < 1e-9 && bounds.min.y < -3.3);
    assert.ok(Math.abs(bounds.max.x - 12) < 1e-9 && Math.abs(bounds.max.z - 8) < 1e-9, 'footprint equals the grid');
    assert.equal(root.children.length, 1, 'no decoration in the paper style');
  } finally { root.dispose(); }
});

test('island walls extend below the deepest displayed cut', () => {
  const root = createEnvironment(frame(64, 64, .5));
  try {
    root.setFloor(-9); root.updateMatrixWorld(true);
    const turf = root.getObjectByName('island-turf'), bounds = new THREE.Box3().setFromObject(turf);
    assert.ok(bounds.min.y < -9, `island bottom ${bounds.min.y} must enclose the cut floor`);
    assert.ok(Math.abs(bounds.max.y) < 1e-6, 'turf top stays at the neutral ground height');
  } finally { root.dispose(); }
});

test('effects expire and never outlive their pools', () => {
  const effects = new Effects({ groundHeight: () => 0 });
  try {
    for (let i = 0; i < 50; i++) {
      effects.throwClods(new THREE.Vector3(0, 2, 0), new THREE.Vector3(1, 0, 0), { count: 10 });
      effects.burst(new THREE.Vector3(), { count: 10 });
      effects.puff(new THREE.Vector3(), { count: 10 });
    }
    assert.ok(effects.clods.length <= 320 && effects.puffs.length <= 260);
    for (let t = 0; t < 200; t++) effects.update(1 / 30);
    assert.equal(effects.clods.length, 0); assert.equal(effects.puffs.length, 0);
    assert.equal(effects.clodMesh.count, 0); assert.equal(effects.puffMesh.count, 0);
  } finally { effects.dispose(); }
});
