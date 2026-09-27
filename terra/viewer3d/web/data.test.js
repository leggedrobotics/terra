import test from 'node:test';
import assert from 'node:assert/strict';
import { validateReplay, validateFrame, transitionFacts, actionName, shortestAngle, formatReward } from './data.js';

function fixture() {
  return {
    schema: 'terra.viewer3d.v1', metadata: { title: 'Validation fixture', source: 'unit test' },
    frames: [{ step: 0, action: null, actor_id: null, current_agent: 2, reward: 0, done: false, task_done: false,
      grid: { rows: 2, cols: 3, tile_size_m: .5 },
      maps: { action: [[0, 0, 0], [0, 0, 3]], target: [[0, -1, 0], [1, 0, 0]], padding: [[0, 0, 0], [0, 0, 1]], dumpability: [[1, 1, 1], [1, 1, 0]], dumpability_static: null, interaction: null, traversability: [[0, 0, 0], [0, -1, 1]] },
      agents: [{ id: 2, type: 0, action_type: 0, position: [1, 1], base_yaw: 0, cabin_yaw: 0, width: 1, height: 1, loaded: 0, wheel_angle: 0, shovel_lifted: 0, reach: [0, 2] }],
    }],
  };
}
test('accepts stable sparse machine slots and raw pile heights without changing input', () => {
  const value = fixture(), original = structuredClone(value); assert.equal(validateReplay(value), value); assert.deepEqual(value, original);
});
test('rejects malformed import shapes, nonfinite data, invalid masks and outcome', () => {
  for (const change of [r => { r.metadata.title = 1; }, r => { r.frames[0].maps.action[0][0] = .5; }, r => { r.frames[0].maps.padding[0][0] = 2; }, r => { r.frames[0].maps.dumpability = null; }, r => { r.frames[0].maps.target[0].pop(); }, r => { r.frames[0].reward = Infinity; }, r => { r.frames[0].agents[0].id = 4; }, r => { r.frames[0].agents[0].shovel_lifted = 2; }, r => { r.frames[0].task_done = true; }, r => { r.frames[0].agents[0].position = [3, 1]; }]) { const value = fixture(); change(value); assert.throws(() => validateReplay(value)); }
});
test('missing optional diagnostic layer remains unavailable', () => {
  const value = fixture(); delete value.frames[0].maps.interaction; assert.equal(validateFrame(value.frames[0]), value.frames[0]);
});
test('DO without a terrain or load delta never invents a dig animation', () => {
  const before = fixture().frames[0], after = structuredClone(before); after.action = 6; after.actor_id = 2; after.step = 1;
  assert.equal(transitionFacts(before, after).kind, 'unchanged');
  after.maps.action[0][1] = -1; assert.equal(transitionFacts(before, after).kind, 'terrain');
  after.agents[0].loaded = 1; assert.equal(transitionFacts(before, after).kind, 'dig');
});
test('load transfer is separate from dumping onto terrain and retains recipient identity', () => {
  const before = fixture().frames[0]; before.agents[0].loaded = 3; before.agents.push({ ...before.agents[0], id: 0, loaded: 0, type: 1 });
  const after = structuredClone(before); after.step = 1; after.action = 6; after.actor_id = 2; after.agents[0].loaded = 0; after.agents[1].loaded = 3; after.current_agent = 0;
  const facts = transitionFacts(before, after); assert.equal(facts.kind, 'transfer'); assert.equal(facts.recipient.id, 0); assert.equal(facts.placed, 0);
});
test('action labels use the preceding actor, not the next active machine', () => {
  const before = fixture().frames[0]; before.agents[0].action_type = 1;
  const after = structuredClone(before); after.action = 2; after.actor_id = 2;
  assert.equal(actionName(after, before), 'Steer left');
  before.agents[0].action_type = 0; assert.equal(actionName(after, before), 'Turn clockwise');
});
test('rotation crosses the angle seam by the short path', () => {
  const halfway = shortestAngle(Math.PI * 1.9, Math.PI * .1, .5); assert.ok(Math.abs(halfway - Math.PI * 2) < 1e-10);
});
test('small policy rewards remain distinguishable from zero', () => {
  assert.equal(formatReward(0), '0.00');
  assert.equal(formatReward(-.0042857146), '-0.00429');
  assert.equal(formatReward(.0000012), '+0.00000120');
  assert.equal(formatReward(.68), '+0.68');
});
