import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { validateFrame, actionName, jointRequests, transitionFacts, workspaceVertexToWorld } from './data.js';
import { TerraScene } from './scene.js';

function frame() {
  const map = () => Array.from({ length: 4 }, () => Array(6).fill(0));
  const agent = (id, type) => ({ id, type, action_type: 0, position: [1, 2], base_yaw: 0,
    cabin_yaw: 0, wheel_angle: 0, width: 1, height: 1, loaded: 0, shovel_lifted: 0, reach: [0, 2] });
  return { grid: { rows: 4, cols: 6, tile_size_m: 2 }, maps: { action: map(), target: map(), padding: map(), dumpability: map() },
    agents: [agent(0, 0), agent(1, 2)], step: 0, reward: 0, done: false, task_done: false,
    current_agent: 1, action: null, actor_id: null, joint_actions: null, workspace_blocked: null };
}

test('joint labels use requested stable slots, including blocked and reset, without naming one actor', () => {
  const reset = frame(); assert.equal(validateFrame(reset), reset);
  assert.equal(actionName(reset), 'Initial joint state');
  const next = structuredClone(reset); next.step = 1; next.joint_actions = [6, 7]; next.workspace_blocked = [true, false];
  next.agents.reverse(); validateFrame(next);
  assert.deepEqual(jointRequests(next), [{ id: 1, name: 'Wait', blocked: false }, { id: 0, name: 'Work (dig / dump)', blocked: true }]);
  assert.equal(actionName(next), 'Requested: 2 Wait · 1 Work (dig / dump) [blocked]');
  assert.throws(() => validateFrame({ ...next, actor_id: 0 }), /cannot name one action or actor/);
  assert.throws(() => validateFrame({ ...next, joint_actions: [6] }), /stable machine slots/);
  assert.throws(() => validateFrame({ ...next, workspace_blocked: [1, 0] }), /rejection flags/);
  assert.throws(() => validateFrame({ ...next, effective_joint_actions: [6] }), /effective joint actions/);
  assert.throws(() => validateFrame({ ...reset, effective_joint_actions: [7, 7] }), /requested slots/);
});

test('native cell-edge polygon geometry uses X=column and Z=row without a half-cell shift', () => {
  const snapshot = frame();
  const vertices = [[0, 0], [0, 6], [4, 6], [4, 0]];
  assert.deepEqual(vertices.map(v => workspaceVertexToWorld(snapshot.grid, v)), [[-6, 0, -4], [6, 0, -4], [6, 0, 4], [-6, 0, 4]]);
  assert.deepEqual(workspaceVertexToWorld(snapshot.grid, [2, 3]), [0, 0, 0]);
  snapshot.workspace_polygons = [{ id: 0, component: 'work', vertices }, { id: 1, component: 'body', vertices }];
  validateFrame(snapshot);
  const scene = Object.assign(Object.create(TerraScene.prototype), { world: new THREE.Group(), lineMaterials: new Set(),
    visibility: { workspace: true }, renderer: { getDrawingBufferSize: v => v.set(800, 600) } });
  scene.populateReservations(snapshot);
  const [work, body] = scene.reservationOutlines.children;
  assert.equal(work.material.dashed, false); assert.equal(body.material.dashed, true);
  const start = work.geometry.getAttribute('instanceStart'), end = work.geometry.getAttribute('instanceEnd');
  assert.deepEqual([start.getX(0), start.getZ(0), end.getX(0), end.getZ(0)], [-6, -4, 6, -4]);
  scene.setLayer('workspace', false); assert.equal(scene.reservationOutlines.visible, false);
  assert.throws(() => validateFrame({ ...snapshot, workspace_polygons: [{ id: 3, component: 'work', vertices }] }), /unknown workspace/);
});

test('joint soil and load changes are reported jointly, without inferred digging or transfer attribution', () => {
  const before = frame(), after = structuredClone(before);
  after.step = 1; after.joint_actions = [6, 0]; after.workspace_blocked = [false, false];
  after.maps.action[0][0] = -1; after.agents[1].loaded = 1;
  const facts = transitionFacts(before, after);
  assert.equal(facts.kind, 'joint'); assert.equal(facts.removed, 1);
  assert.equal(facts.message, 'Joint round · 1 terrain cells changed');
  assert.equal(facts.recipient, undefined); assert.equal(facts.loadDelta, undefined);
  const stalled = structuredClone(before); stalled.step = 1; stalled.joint_actions = [2, 2];
  assert.equal(transitionFacts(before, stalled).message, 'Joint round · no recorded state change');
});

test('joint follow cannot invent a single active actor', () => {
  const scene = Object.assign(Object.create(TerraScene.prototype), { frame: frame(), follow: false });
  scene.setFollow(true);
  assert.equal(scene.follow, false);
});
