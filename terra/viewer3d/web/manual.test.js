import test from 'node:test';
import assert from 'node:assert/strict';
import { manualControls, unloadGuidance } from './manual.js';

function live(overrides = {}) {
  return { replay: { frames: [{ done: false, task_done: false }, { done: false, task_done: false }] }, index: 1, mode: 'manual', session: { cases: [{ id: '17411' }], can_undo: true }, ...overrides };
}
test('manual actions require the latest live frame; history and imported replays cannot mutate the session', () => {
  assert.equal(manualControls(live()).action, true);
  for (const override of [{ index: 0 }, { imported: true }, { playing: true }, { busy: true }, { mode: 'replay' }, { replay: undefined }]) {
    const state = manualControls(live(override));
    assert.equal(state.action, false); assert.equal(state.undo, false);
  }
});
test('450-step timeout freezes actions until explicit server exploration, while keeping undo and reset', () => {
  const input = live({ replay: { frames: [{ step: 450, done: true, task_done: false }] }, index: 0 });
  const frozen = manualControls(input);
  assert.equal(frozen.action, false); assert.equal(frozen.undo, true); assert.equal(frozen.reset, true); assert.equal(frozen.continueVisible, true);
  input.session.exploring = true;
  assert.equal(manualControls(input).action, true); assert.equal(manualControls(input).continueVisible, false);
  input.imported = true;
  assert.equal(manualControls(input).action, false); assert.equal(manualControls(input).reset, false);
});
test('undo requires authoritative native history and completed tasks do not offer timeout continuation', () => {
  const input = live(); input.session.can_undo = false;
  assert.equal(manualControls(input).undo, false);
  input.replay.frames[1] = { done: true, task_done: true };
  assert.equal(manualControls(input).action, false); assert.equal(manualControls(input).continueVisible, false);
});
test('unload guidance wraps cabin offsets and chooses the fewest actual Q/E turns', () => {
  const d = { loaded: 5, accepted_unload_now: false, accepted_unload_any: true, accepted_unload_by_cabin_offset: Array(12).fill(false) };
  d.accepted_unload_by_cabin_offset[1] = true;
  assert.match(unloadGuidance(d), /^Q × 1, then Space/);
  d.accepted_unload_by_cabin_offset[1] = false; d.accepted_unload_by_cabin_offset[11] = true; d.accepted_unload_by_cabin_offset[4] = true;
  assert.match(unloadGuidance(d), /^E × 1, then Space/);
  d.accepted_unload_now = true; assert.match(unloadGuidance(d), /^Press Space/);
  d.loaded = 0; assert.match(unloadGuidance(d), /^Empty bucket/);
});
test('off-target native unloading is not described as accepted disposal', () => {
  assert.match(unloadGuidance({ loaded: 5, dump_status: 'off_zone_only' }), /will not count as accepted disposal/);
  assert.match(unloadGuidance({ loaded: 5, dump_status: 'no_unload_at_this_base' }), /loaded base cannot move/);
});
