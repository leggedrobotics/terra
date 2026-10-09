import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { PostprocessedEpisode, validatePostprocessed, metricAgent, metricCell, postprocessedRoutes } from './postprocessed.js';
import { TerraScene } from './scene.js';

function fixture(rows = 3, cols = 5) {
  const agents = [0, 2].map(id => ({ id, type: id === 0 ? 0 : 2, action_type: 0, width_m: 2, length_m: 4, reach_m: [1, 6] }));
  const states = agents.map((a, i) => ({ id: a.id, pose: [10 + i * .5, 20.5, 0], cabin_yaw: Math.PI / 2, load: 0, wheel_angle: 0, shovel_lifted: 0 }));
  return { schema: 'terra.postprocessed.v1', grid: { rows, cols, resolution_m: .5, origin_xy_m: [10, 20] }, agents,
    initial: { native_m: Array(rows * cols).fill(0), loose_m: Array(rows * cols).fill(0), agents: states },
    workspaces: [{ id: 'A', agent_id: 0, kind: 'excavate', masks: { dig: [[1, 1, 2]], finish: [[1, 1, 1]], dump: [[0, 2, 1]], deposit: [[0, 2, 2]] } }],
    frames: [],
  };
}
function append(data, change = {}) {
  data.frames.push({ phase: 'cut', workspace_id: 'A', agent_id: 0, agents: structuredClone(data.initial.agents), terrain_changes: [], work: [], ...change });
}

test('metric frames preserve fractional native and loose terrain, reverse seek and sparse original IDs', () => {
  const data = fixture();
  append(data, { terrain_changes: [[6, -.375, .125]], work: [{ agent_id: 0, kind: 'dig', changed_indices: [6] }] });
  append(data, { phase: 'dump', terrain_changes: [[6, -.375, .625]], work: [{ agent_id: 0, kind: 'dump', changed_indices: [6] }] });
  const copy = structuredClone(data), episode = new PostprocessedEpisode(data);
  const before = episode.frame(0), cut = episode.frame(1), dumped = episode.frame(2);
  assert.equal(cut.maps.action[1][1], -.25); assert.equal(dumped.maps.action[1][1], .25);
  assert.equal(cut.metric.native[1][1], -.375); assert.equal(cut.metric.loose[1][1], .125);
  assert.equal(before.maps.action[1][1], 0); assert.equal(episode.frame(0).maps.action[1][1], 0);
  assert.equal(episode.frame(2).maps.action[1][1], .25);
  assert.deepEqual(dumped.agents.map(a => a.id), [0, 2]); assert.deepEqual(data, copy);
});

test('fine grids keep indices above uint16 and exact workspace support', () => {
  const data = fixture(260, 300), index = 259 * 300 + 299;
  data.workspaces[0].masks.finish = [[259, 299, 1]];
  append(data, { terrain_changes: [[index, -.5, .023]], work: [{ agent_id: 0, kind: 'dig', changed_indices: [index] }] });
  const frame = new PostprocessedEpisode(data).frame(1);
  assert.equal(frame.maps.action[259][299], -.477);
  assert.equal(frame.maps.target[259][299], -1); assert.equal(frame.maps.target[41][155], 0);
  assert.equal(frame.metric.work[0].changed_indices[0], index);
});

test('rotated metric grid transforms world poses without a half-cell or heading error', () => {
  const data = fixture(); data.grid.yaw_rad = Math.PI / 2;
  // row=1, col=3: R(pi/2) * [1.5,.5] + [10,20] = [9.5,21.5].
  const state = { ...data.initial.agents[0], pose: [9.5, 21.5, Math.PI], cabin_yaw: Math.PI * 1.25, wheel_angle: Math.PI / 18 };
  const a = metricAgent(data.agents[0], state, data.grid);
  assert.ok(Math.abs(a.position[0] - 1) < 1e-12); assert.ok(Math.abs(a.position[1] - 3) < 1e-12);
  assert.equal(a.base_yaw, Math.PI / 2); assert.equal(a.cabin_yaw, Math.PI / 4); assert.equal(a.wheel_angle, .5);
  assert.deepEqual(metricCell(10, 20, data.grid), [0, 0]);
  const scene = Object.create(TerraScene.prototype); scene.frame = { metric: {}, grid: { rows: 3, cols: 5, tile_size_m: .5 } }; scene.unitHeight = 1;
  const p = scene.point(1, 3, -.375); assert.deepEqual(p.toArray(), [.5, -.375, -0]);
  const forward = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), a.base_yaw);
  assert.ok(Math.abs(forward.x) < 1e-12); assert.ok(Math.abs(forward.z + 1) < 1e-12);
});

test('positive metric surfaces and fills below grade keep their supplied heights', () => {
  const data = fixture(); data.initial.native_m[6] = -.5; data.initial.loose_m[6] = .2; data.initial.loose_m[7] = 2.75;
  const scene = Object.create(TerraScene.prototype); scene.frame = new PostprocessedEpisode(data).frame(0); scene.unitHeight = 1;
  assert.equal(scene.heightAt(1, 1), -.3); assert.equal(scene.surfacePoint(1, 2).y, 2.75);
  assert.equal(scene.groundAt(scene.point(1, 1).x, scene.point(1, 1).z), -.3);
});

test('explicit work ownership wins over nearest machine and load heuristics', () => {
  const data = fixture();
  append(data, { terrain_changes: [[6, -.4, 0]], work: [{ agent_id: 0, kind: 'dig', changed_indices: [6] }] });
  const episode = new PostprocessedEpisode(data), scene = Object.create(TerraScene.prototype);
  const actors = scene.actorWork(episode.frame(0), episode.frame(1));
  // Machine 2 sits exactly on the changed cell and loads are both unchanged.
  assert.equal(actors.get(0).kind, 'dig'); assert.deepEqual(actors.get(0).cells.map(c => [c.row, c.col]), [[1, 1]]);
  assert.equal(actors.get(2).cells.length, 0);
});

test('checked drive retains each reversal pose and interpolates the yaw seam by its short path', () => {
  const data = fixture(), states = structuredClone(data.initial.agents);
  for (const [x, yaw] of [[10.5, 3.12], [11, 3.13], [10.75, -3.13]]) {
    const a = structuredClone(states); a[0].pose = [x, 20.5, yaw]; a[0].cabin_yaw = yaw;
    append(data, { phase: 'drive', route_status: 'checked', agents: a });
  }
  const episode = new PostprocessedEpisode(data), scene = Object.create(TerraScene.prototype);
  const before = episode.frame(2), after = episode.frame(3); scene.frame = after; scene.unitHeight = 1;
  const machine = { root: new THREE.Object3D(), setPose() {} };
  scene.poseMachine(machine, after.agents[0], after, .5, before.agents[0], before, 'move');
  assert.ok(Math.abs(machine.root.rotation.y - Math.PI) < .02);
  assert.equal(machine.root.position.x, -.125); // exact midpoint of a reversing leg
  assert.equal(episode.event(3).agents[0].pose[0], 10.75);
});

test('fleet routes without workspaces keep machine ownership and split missing connections', () => {
  const data = fixture(), agents = structuredClone(data.initial.agents);
  for (const [id, x, y, phase, status] of [[0, 11, 20.5, 'drive', 'checked'], [2, 10.5, 21, 'drive', 'checked'], [0, 15, 20.5, 'relocate', 'missing'], [0, 16, 20.5, 'drive', 'checked']]) {
    agents.find(a => a.id === id).pose = [x, y, 0];
    append(data, { agent_id: id, agents: structuredClone(agents), workspace_id: null, route_id: 'leg', phase, route_status: status });
  }
  const routes = postprocessedRoutes(data), first = routes.byFrame.get(1)[0], second = routes.byFrame.get(2)[0];
  assert.notEqual(first, second); assert.equal(routes.routes.size, 2); assert.equal(routes.byWorkspace.size, 0);
  const a = routes.routes.get(first), b = routes.routes.get(second);
  assert.equal(a.agent_id, 0); assert.equal(b.agent_id, 2);
  assert.deepEqual(a.segments.map(s => s.status), ['checked', 'missing', 'checked']);
  assert.deepEqual(a.segments.map(s => s.points.map(p => p[0])), [[10, 11], [11, 15], [15, 16]]);
  assert.deepEqual(b.segments[0].points.map(p => p.slice(0, 2)), [[10.5, 20.5], [10.5, 21]]);
});

test('metric fleet turns remain within the declared endpoint headings', () => {
  const episode = new PostprocessedEpisode(fixture()), frame = episode.frame(0), scene = Object.create(TerraScene.prototype);
  scene.frame = frame; scene.unitHeight = 1;
  const before = { ...frame.agents[0], base_yaw: 0, cabin_yaw: 0 }, after = { ...before, base_yaw: 1, cabin_yaw: .6 };
  let cabin = null; const machine = { root: new THREE.Object3D(), setPose(state) { cabin = state.cabin_yaw; } };
  for (const progress of [0, .2, .5, .8, .95, 1]) {
    scene.poseMachine(machine, after, frame, progress, before, frame, 'turn');
    assert.ok(machine.root.rotation.y >= 0 && machine.root.rotation.y <= 1);
    assert.ok(cabin >= 0 && cabin <= .6);
  }
});

test('malformed attribution and metric values fail before playback', () => {
  for (const mutate of [
    d => { d.initial.loose_m[0] = -1; },
    d => { d.grid.yaw_rad = Infinity; },
    d => { d.agents[0].id = -1; },
    d => { append(d, { terrain_changes: [[6, -.5, 0]] }); },
    d => { append(d, { terrain_changes: [[6, -.5, 0]], work: [{ agent_id: 0, kind: 'dig', changed_indices: [6] }, { agent_id: 2, kind: 'dig', changed_indices: [6] }] }); },
    d => { d.agents[0].reach_m = [3, 1]; },
    d => { d.initial.agents[0].pose[0] = NaN; },
  ]) { const data = fixture(); mutate(data); assert.throws(() => validatePostprocessed(data)); }
});
