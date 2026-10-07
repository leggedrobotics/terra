import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { metricTerrainGeometry } from './metric-terrain.js';
import { PALETTES } from './materials.js';

function frame(heights, loose = heights.map(row => row.map(() => 0))) {
  return { grid: { rows: heights.length, cols: heights[0].length, tile_size_m: .1 }, metric: { loose } };
}

test('fine flat metric terrain merges faces without resampling its extent', () => {
  const heights = Array.from({ length: 446 }, () => Array(446).fill(0));
  const geometry = metricTerrainGeometry(frame(heights), heights, 1, -.085, PALETTES.studio);
  assert.equal(geometry.getAttribute('position').count, 30); // one top and four outside walls
  geometry.computeBoundingBox();
  assert.ok(Math.abs(geometry.boundingBox.min.x + 22.3) < 1e-5);
  assert.ok(Math.abs(geometry.boundingBox.max.z - 22.3) < 1e-5);
  assert.ok(Math.abs(geometry.boundingBox.min.y + .085) < 1e-7);
  geometry.dispose();
});

test('merged metric terrain preserves fractional cuts, fills and outward wall normals', () => {
  const heights = [[0, 0, .25], [0, -.375, .125]], loose = [[0, 0, .25], [0, .125, .5]];
  const geometry = metricTerrainGeometry(frame(heights, loose), heights, 1, -.5, PALETTES.studio);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  mesh.updateMatrixWorld(true);
  const ray = new THREE.Raycaster();
  for (let row = 0; row < 2; row++) for (let col = 0; col < 3; col++) {
    ray.set(new THREE.Vector3((col - 1) * .1, 2, (.5 - row) * .1), new THREE.Vector3(0, -1, 0));
    const hit = ray.intersectObject(mesh)[0];
    assert.ok(hit); assert.ok(Math.abs(hit.point.y - heights[row][col]) < 1e-7);
    assert.ok(hit.face.normal.y > .99);
  }
  ray.set(new THREE.Vector3(.2, .1, .05), new THREE.Vector3(-1, 0, 0));
  const side = ray.intersectObject(mesh)[0];
  assert.ok(side.face.normal.x > .99);
  geometry.dispose(); mesh.material.dispose();
});
