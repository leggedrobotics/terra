import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';

// Pure labels from the actual packaged capture adapter; no DOM or WebGL needed.
const capture = runInNewContext(readFileSync(new URL('../../postprocess/assets/capture.js', import.meta.url), 'utf8'));

test('video keeps native evaluation and exploration outcomes distinct', () => {
  assert.equal(capture.caption({ task_done: true, done: true }, 10, 11, false), 'Task complete');
  assert.equal(capture.caption({ task_done: false, done: true }, 10, 11, false), 'Episode ended without success');
  assert.equal(capture.caption({ task_done: false, done: false }, 10, 11, false), 'Recording ended before termination');
  assert.equal(capture.caption({ task_done: true, done: true, diagnostics: { exploring: true } }, 500, 501, false), 'Exploration outside episode budget · task complete');
  assert.equal(capture.caption({ task_done: false, diagnostics: { exploring: true } }, 500, 501, false), 'Exploration outside episode budget · task incomplete');
});

test('video retains CPU source and both processed verdicts', () => {
  const native = capture.title({ metadata: { title: 'Case 20', source: 'Native Terra CPU replay' } }, false);
  assert.match(native, /Native Terra CPU replay/);
  const metric = capture.title({ metadata: { title: 'Case 20', variant: 'cleaned', plan_status: 'INCOMPLETE_WORKSPACE_REFINEMENT', schedule_status: 'FIXED_PATH_CONFLICT' } }, true);
  assert.match(metric, /INCOMPLETE WORKSPACE REFINEMENT/);
  assert.match(metric, /FIXED PATH CONFLICT/);
});
