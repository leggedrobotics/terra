import test from 'node:test';
import assert from 'node:assert/strict';
import { manualControls, unloadGuidance, structuredMode, manualActionRequest, manualActionAvailable, moveDistanceControls, currentCabinHeading, cabinHeadingLabel, structuredBudget } from './manual.js';

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

const structured = { action_mode: 'structured_v1' };
test('structured requests send only relevant arguments while legacy requests stay identical', () => {
  const choices = { distance: 2, turn: 4, heading: 9 };
  for (let action = 0; action < 8; action++) assert.deepEqual(manualActionRequest(action, {}, choices), { action });
  assert.deepEqual(manualActionRequest(0, structured, choices), { action: 0, amount: 2 });
  assert.deepEqual(manualActionRequest(1, structured, choices), { action: 1, amount: 2 });
  assert.deepEqual(manualActionRequest(2, structured, choices), { action: 2, amount: 4 });
  assert.deepEqual(manualActionRequest(3, structured, choices), { action: 3, amount: 4 });
  assert.deepEqual(manualActionRequest(6, structured, choices), { action: 6, heading: 9 });
  for (const action of [4, 5, 7]) assert.deepEqual(manualActionRequest(action, structured, choices), { action });
  assert.deepEqual(manualActionRequest(6, structured), { action: 6 });
  assert.deepEqual(manualActionRequest(0, structured), { action: 0, amount: 5 });
  assert.equal(structuredMode({ action_mode: 'future_unknown' }), false);
});
test('native masks distinguish direction, distance, turn amount, and the chosen work heading', () => {
  const snapshot = { diagnostics: { structured_actions: { current_heading: 11, move_mask: [[true, false, true, true, true], [false, true, false, false, false]], turn_mask: [[true, true, false, false, false, false], [false, false, false, true, true, true]], do_mask: Array(12).fill(false) } } };
  const available = request => manualActionAvailable(request, snapshot, structured);
  assert.equal(available({ action: 0, amount: 2 }), false);
  assert.equal(available({ action: 1, amount: 2 }), true);
  assert.equal(available({ action: 2, amount: 4 }), false);
  assert.equal(available({ action: 3, amount: 4 }), true);
  assert.equal(available({ action: 6 }), false);
  snapshot.diagnostics.structured_actions.do_mask[11] = true;
  assert.equal(available({ action: 6 }), true);
  assert.equal(available({ action: 6, heading: 0 }), false);
  for (const action of [4, 5, 7]) assert.equal(available({ action }), true);
  assert.equal(manualActionAvailable({ action: 0, amount: 2 }, snapshot, {}), true);
  assert.equal(manualActionAvailable({ action: 6 }, {}, structured), true);
});
test('native work mask permits relifting and off-zone unload even without fresh or accepted work', () => {
  const snapshot = { diagnostics: { do_kind: 'relift', fresh_dig_current_count: 0, accepted_unload_now: false, dump_status: 'off_zone_only', structured_actions: { current_heading: 0, do_mask: [true, ...Array(11).fill(false)] } } };
  assert.equal(manualActionAvailable({ action: 6 }, snapshot, structured), true);
});
test('oblique heading replaces a rejected one-cell choice with two, preserving all larger user choices', () => {
  const snapshot = { current_agent: 0, agents: [{ id: 0, base_yaw: Math.PI / 6, loaded: 0 }], diagnostics: { structured_actions: { move_mask: [[false, true, true, true, true], [false, false, false, false, false]] } } };
  for (let bin = 0; bin < 12; bin++) {
    snapshot.agents[0].base_yaw = bin * Math.PI / 6;
    const controls = moveDistanceControls(snapshot, 1);
    assert.equal(controls.distance, bin % 3 ? 2 : 1);
    assert.equal(controls.oneCellDisabled, true);
    assert.equal(controls.hint, bin % 3 ? 'Angled headings need at least 2 cells on this grid.' : '');
    for (const amount of [2, 3, 4, 5]) assert.equal(moveDistanceControls(snapshot, amount).distance, amount);
  }
  // The corrected selector value drives the same request path used by keyboard.
  const distance = moveDistanceControls(snapshot, 1).distance;
  const request = manualActionRequest(0, structured, { distance });
  assert.deepEqual(request, { action: 0, amount: 2 });
  assert.equal(manualActionAvailable(request, snapshot, structured), true);
});
test('loaded, blocked, cardinal, and missing-mask states do not invent a viable two-cell move', () => {
  const snapshot = { current_agent: 0, agents: [{ id: 0, base_yaw: Math.PI / 6, loaded: 3 }], diagnostics: { structured_actions: { move_mask: [[false, true, true, true, true], [false, false, false, false, false]] } } };
  assert.deepEqual(moveDistanceControls(snapshot, 1), { distance: 1, oneCellDisabled: true, hint: '' });
  snapshot.agents[0].loaded = 0;
  snapshot.diagnostics.structured_actions.move_mask = [Array(5).fill(false), Array(5).fill(false)];
  assert.equal(moveDistanceControls(snapshot, 1).distance, 1);
  snapshot.diagnostics.structured_actions.move_mask[1][2] = true;
  assert.equal(moveDistanceControls(snapshot, 1).distance, 3);
  snapshot.diagnostics.structured_actions.move_mask[0][0] = true;
  assert.deepEqual(moveDistanceControls(snapshot, 1), { distance: 1, oneCellDisabled: false, hint: '' });
  delete snapshot.diagnostics;
  assert.deepEqual(moveDistanceControls(snapshot, 1), { distance: 1, oneCellDisabled: false, hint: '' });
});
test('heading labels and current selection use chassis-relative cabin angles, wrapping Q/E correctly', () => {
  const snapshot = { current_agent: 1, agents: [{ id: 1, base_yaw: Math.PI, cabin_yaw: 11 * Math.PI / 6 }] };
  assert.equal(currentCabinHeading(snapshot), 11);
  assert.match(cabinHeadingLabel(0, 11), /0° · forward · Q 30°/);
  assert.match(cabinHeadingLabel(11, 0), /30° right · E 30°/);
  assert.match(cabinHeadingLabel(6, 0), /180° · rear · Q 180°/);
  snapshot.diagnostics = { structured_actions: { current_heading: 2 } };
  assert.equal(currentCabinHeading(snapshot), 2);
});
test('structured budget shows seconds and decisions independently of native step count', () => {
  const diagnostics = { elapsed_time_s: 102.5, time_budget_s: 150, remaining_time_s: 47.5, decisions: 2, decision_budget: 80 };
  const budget = structuredBudget(diagnostics, { step: 9 });
  assert.match(budget.time, /102[.,]5 \/ 150 s/);
  assert.equal(budget.decisions, '2 / 80');
  assert.match(budget.note, /47[.,]5 s remaining/);
  assert.equal(structuredBudget({ ...diagnostics, termination_reason: 'decision_budget' }, { done: true }).note, 'Decision budget reached · episode frozen');
  assert.equal(structuredBudget(diagnostics, { done: true, task_done: true }).note, 'Completed within the episode');
  assert.equal(structuredBudget(diagnostics, { done: true }, true).note, 'Exploration · outside the episode budget');
});
