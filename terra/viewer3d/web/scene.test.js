import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { TerraScene } from './scene.js';
import { SoilPiles } from './piles.js';

function frame(action) {
  return { grid: { rows: 3, cols: 3, tile_size_m: 1 }, current_agent: 0, maps: { action, padding: action.map(row => row.map(() => 0)) } };
}

test('surface anchors follow capped piles while raw cuts and stored values stay exact', () => {
  const scene = Object.create(TerraScene.prototype);
  const before = frame([[0, 0, 0], [0, 0, -2], [0, 0, 0]]);
  scene.frame = frame([[0, 0, 0], [0, 127, -2], [0, 0, 0]]);
  scene.unitHeight = .48; scene.piles = new SoilPiles(scene.frame, { previous: before });
  scene.selection = new THREE.Object3D();
  assert.equal(scene.heightAt(1, 1), .384);
  assert.equal(scene.heightAt(1, 1, before), 0);
  assert.equal(scene.heightAt(1, 2), -.96);
  assert.equal(scene.surfacePoint(1, 1).y, .384);
  scene.highlight(1, 1); assert.equal(scene.selection.position.y, .384 + .03);
  assert.deepEqual(scene.selected, { row: 1, col: 1 });
  assert.equal(scene.frame.maps.action[1][1], 127);
  const state = { id: 0, position: [1, 1], base_yaw: 0, cabin_yaw: 0, wheel_angle: 0 };
  const machine = { root: new THREE.Object3D(), setPose() {} };
  scene.piles.update(.5);
  scene.displayHeights = before.maps.action.map((row, r) => row.map((height, c) => (height + scene.frame.maps.action[r][c]) / 2));
  scene.poseMachine(machine, state, scene.frame, .5, state, before);
  assert.equal(machine.root.position.y, scene.piles.nodeHeight(3, 3));
  assert.equal(scene.surfacePoint(1, 1).y, scene.piles.nodeHeight(3, 3));
  scene.highlight(1, 1); assert.equal(scene.selection.position.y, scene.piles.nodeHeight(3, 3) + .03);
  scene.piles.update(1); scene.displayHeights = scene.frame.maps.action;
  scene.poseMachine(machine, state, scene.frame, 1);
  assert.equal(machine.root.position.y, .384);
  scene.piles.dispose();
});

test('anchors match the current surface while digging through zero into a cut', () => {
  const scene = Object.create(TerraScene.prototype);
  const before = frame([[0, 0, 0], [0, 1, 0], [0, 0, 0]]);
  scene.frame = frame([[0, 0, 0], [0, -1, 0], [0, 0, 0]]);
  scene.unitHeight = .48; scene.piles = new SoilPiles(scene.frame, { previous: before });
  scene.selection = new THREE.Object3D();
  const state = { id: 0, position: [1, 1], base_yaw: 0, cabin_yaw: 0, wheel_angle: 0 };
  const machine = { root: new THREE.Object3D(), setPose() {} };
  for (const progress of [.25, .5, .75, 1]) {
    scene.piles.update(progress);
    scene.displayHeights = before.maps.action.map((row, r) => row.map((height, c) => height + (scene.frame.maps.action[r][c] - height) * progress));
    scene.poseMachine(machine, state, scene.frame, progress, state, before);
    scene.highlight(1, 1);
    const expected = progress < .5 ? scene.piles.nodeHeight(3, 3) : (1 - progress * 2) * .48;
    assert.equal(machine.root.position.y, expected);
    assert.equal(scene.surfacePoint(1, 1).y, expected);
    assert.equal(scene.selection.position.y, expected + .03);
  }
  scene.piles.dispose();
});
