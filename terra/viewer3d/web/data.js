/** Pure snapshot validation and display facts. Never implements Terra transitions. */
export const SCHEMA = 'terra.viewer3d.v1';
export const TYPES = ['Excavator', 'Truck', 'Skid steer'];
export const ACTIONS = ['Forward', 'Backward', 'Turn clockwise', 'Turn anticlockwise', 'Cabin clockwise', 'Cabin anticlockwise', 'Work', 'Wait'];
const requiredMaps = ['action', 'target', 'padding', 'dumpability'];
const diagnosticMaps = ['dumpability_static', 'interaction', 'traversability', 'precision_required_band', 'fresh_dig_current', 'fresh_dig_swing', 'remaining_target', 'footprint', 'work_cone', 'pull_permission'];
const finite = (value) => typeof value === 'number' && Number.isFinite(value);
const integer = (value) => Number.isSafeInteger(value);
function expect(condition, message) { if (!condition) throw new Error(message); }

export function validateReplay(replay) {
  expect(replay && typeof replay === 'object' && replay.schema === SCHEMA, `Expected a ${SCHEMA} replay.`);
  expect(replay.metadata && typeof replay.metadata.title === 'string' && typeof replay.metadata.source === 'string', 'Replay metadata must include title and source strings.');
  expect(Array.isArray(replay.frames) && replay.frames.length > 0, 'The replay contains no frames.');
  expect(replay.frames.length <= 100000, 'This viewer supports at most 100,000 frames.');
  for (const [index, frame] of replay.frames.entries()) validateFrame(frame, `Frame ${index}`);
  return replay;
}

export function validateFrame(frame, context = 'Frame') {
  expect(frame && typeof frame === 'object', `${context}: expected an object.`);
  const { grid, maps, agents } = frame;
  expect(grid && integer(grid.rows) && integer(grid.cols) && grid.rows > 0 && grid.cols > 0 && grid.rows <= 128 && grid.cols <= 128, `${context}: grid must be between 1 and 128 cells on each side.`);
  expect(finite(grid.tile_size_m) && grid.tile_size_m > 0, `${context}: invalid tile size.`);
  expect(maps && typeof maps === 'object', `${context}: missing maps.`);
  for (const key of [...requiredMaps, ...diagnosticMaps]) {
    const map = maps[key];
    if (map == null && diagnosticMaps.includes(key)) continue;
    expect(Array.isArray(map) && map.length === grid.rows, `${context}: ${key} has the wrong row count.`);
    const soil = key === 'action' || key === 'target';
    const allowed = key === 'traversability' ? [-1, 0, 1, false, true] : [0, 1, false, true];
    for (const row of map) expect(Array.isArray(row) && row.length === grid.cols && row.every(v => soil ? integer(v) : allowed.includes(v)), `${context}: ${key} has invalid cells or columns.`);
  }
  expect(integer(frame.step) && frame.step >= 0 && finite(frame.reward), `${context}: invalid step or reward.`);
  expect(frame.action === null || (integer(frame.action) && frame.action >= 0 && frame.action <= 7), `${context}: invalid action.`);
  expect(typeof frame.done === 'boolean' && typeof frame.task_done === 'boolean', `${context}: invalid episode outcome.`);
  expect(!frame.task_done || frame.done, `${context}: task_done requires done.`);
  expect(Array.isArray(agents) && agents.length > 0 && agents.length <= 4, `${context}: expected 1–4 active agents.`);
  const ids = new Set();
  for (const agent of agents) {
    expect(agent && integer(agent.id) && agent.id >= 0 && agent.id <= 3 && !ids.has(agent.id), `${context}: agent IDs must be unique original slots from 0 to 3.`);
    ids.add(agent.id);
    expect(integer(agent.type) && agent.type >= 0 && agent.type <= 2 && (agent.action_type === 0 || agent.action_type === 1), `${context}: unknown machine type.`);
    expect(Array.isArray(agent.position) && agent.position.length === 2 && agent.position.every(finite) && agent.position[0] >= 0 && agent.position[0] < grid.rows && agent.position[1] >= 0 && agent.position[1] < grid.cols, `${context}: agent position is outside the map.`);
    expect(finite(agent.base_yaw) && finite(agent.cabin_yaw) && integer(agent.wheel_angle), `${context}: invalid machine angle.`);
    expect(finite(agent.width) && finite(agent.height) && agent.width > 0 && agent.height > 0, `${context}: invalid machine footprint.`);
    expect(integer(agent.loaded) && agent.loaded >= 0 && (agent.shovel_lifted === 0 || agent.shovel_lifted === 1), `${context}: invalid machine load or shovel state.`);
    expect(Array.isArray(agent.reach) && agent.reach.length === 2 && agent.reach.every(finite) && agent.reach[0] >= 0 && agent.reach[1] >= agent.reach[0], `${context}: invalid machine reach.`);
  }
  expect(ids.has(frame.current_agent), `${context}: the active agent does not exist.`);
  expect(frame.actor_id === null || ids.has(frame.actor_id), `${context}: the preceding actor does not exist.`);
  if (isJointFrame(frame)) {
    expect(frame.action === null && frame.actor_id === null, `${context}: a joint round cannot name one action or actor.`);
    const validSlots = (values, test) => Array.isArray(values) && values.length <= 4 && [...ids].every(id => id < values.length) && values.every(test);
    expect(frame.joint_actions === null || validSlots(frame.joint_actions, v => integer(v) && v >= 0 && v <= 7), `${context}: joint requests must use stable machine slots.`);
    expect(frame.workspace_blocked == null || validSlots(frame.workspace_blocked, v => typeof v === 'boolean'), `${context}: invalid per-machine workspace rejection flags.`);
    if (frame.effective_joint_actions != null) {
      expect(validSlots(frame.effective_joint_actions, v => integer(v) && v >= 0 && v <= 7), `${context}: invalid effective joint actions.`);
      expect(frame.joint_actions !== null && frame.effective_joint_actions.length === frame.joint_actions.length, `${context}: effective actions must match requested slots.`);
    }
    if (frame.workspace_blocked != null) expect(frame.joint_actions !== null && frame.workspace_blocked.length === frame.joint_actions.length, `${context}: rejection flags must match requested slots.`);
  }
  if (frame.workspace_polygons != null) {
    expect(Array.isArray(frame.workspace_polygons), `${context}: workspace polygons must be an array.`);
    const components = new Set();
    for (const polygon of frame.workspace_polygons) {
      expect(polygon && ids.has(polygon.id) && ['body', 'work'].includes(polygon.component), `${context}: unknown workspace component or machine.`);
      const key = `${polygon.id}:${polygon.component}`;
      expect(!components.has(key), `${context}: repeated workspace component.`); components.add(key);
      expect(Array.isArray(polygon.vertices) && polygon.vertices.length >= 3 && polygon.vertices.every(v => Array.isArray(v) && v.length === 2 && v.every(finite)), `${context}: invalid workspace vertices.`);
    }
  }
  return frame;
}

export function isJointFrame(frame) { return !!frame && Object.prototype.hasOwnProperty.call(frame, 'joint_actions'); }

/** Requests use original machine slots, not observation order. No effect is inferred. */
export function jointRequests(frame) {
  return frame.agents.map(agent => {
    const action = frame.joint_actions?.[agent.id] ?? null;
    let name = action === null ? 'No request · reset' : ACTIONS[action];
    if (agent.action_type === 1 && (action === 2 || action === 3)) name = action === 2 ? 'Steer left' : 'Steer right';
    if (action === 6) name = agent.type === 2 ? 'Shovel action' : 'Work (dig / dump)';
    if (agent.type === 2 && (action === 4 || action === 5)) name = 'Cabin request (no-op)';
    return { id: agent.id, name, blocked: frame.workspace_blocked?.[agent.id] ?? null };
  });
}

/** Projection-bound vertices are cell EDGES, unlike the cell-centre pose API. */
export function workspaceVertexToWorld(grid, [row, col], height = 0) {
  return [(col - grid.cols / 2) * grid.tile_size_m, height, (row - grid.rows / 2) * grid.tile_size_m];
}

export function actionName(frame, previous) {
  if (isJointFrame(frame)) return frame.joint_actions === null ? 'Initial joint state' : 'Requested: ' + jointRequests(frame).map(r => `${r.id + 1} ${r.name}${r.blocked ? ' [blocked]' : ''}`).join(' · ');
  if (frame.action === null) return 'Initial state';
  const actor = (previous || frame).agents.find(a => a.id === frame.actor_id) || frame.agents[0];
  if (actor.action_type === 1 && (frame.action === 2 || frame.action === 3)) return frame.action === 2 ? 'Steer left' : 'Steer right';
  if (frame.action === 6) {
    if (actor.type === 2) return 'Shovel action';
    return actor.loaded > 0 ? 'Dump / transfer' : (actor.type === 1 ? 'Dump' : 'Dig');
  }
  return ACTIONS[frame.action];
}

/** Classify backend masks for display; never infer a native action's legality. */
export function diggingView(frame) {
  const { maps, grid } = frame;
  const loaded = (frame.agents.find(agent => agent.id === frame.current_agent)?.loaded ?? 0) > 0;
  const available = maps.fresh_dig_current != null && maps.fresh_dig_swing != null;
  const counts = { current: 0, swing: 0, blocked: 0, remaining: 0, precision: 0 };
  const cells = Array.from({ length: grid.rows }, () => Array(grid.cols).fill(0));
  for (let row = 0; row < grid.rows; row++) for (let col = 0; col < grid.cols; col++) {
    if (maps.padding[row][col]) continue;
    if (maps.precision_required_band?.[row][col]) counts.precision++;
    const remaining = maps.remaining_target != null ? !!maps.remaining_target[row][col] : maps.target[row][col] < 0 && maps.action[row][col] > maps.target[row][col];
    if (!remaining) continue;
    counts.remaining++;
    if (!available) continue;
    if (loaded) { cells[row][col] = 4; continue; }
    if (maps.fresh_dig_current[row][col]) { cells[row][col] = 1; counts.current++; }
    else if (maps.fresh_dig_swing[row][col]) { cells[row][col] = 2; counts.swing++; }
    else { cells[row][col] = 3; counts.blocked++; }
  }
  return { available, loaded, cells, counts };
}

export const DIGGING_LABELS = ['No remaining dig target', 'Dig now', 'Swing cabin only', 'Not diggable from this base', 'Unload first'];

export function transitionFacts(previous, frame) {
  if (!previous || frame.grid.rows !== previous.grid.rows || frame.grid.cols !== previous.grid.cols) return { kind: 'snapshot', changed: [], removed: 0, placed: 0, message: 'Initial state' };
  const changed = [];
  let removed = 0, placed = 0;
  for (let row = 0; row < frame.grid.rows; row++) for (let col = 0; col < frame.grid.cols; col++) {
    const delta = frame.maps.action[row][col] - previous.maps.action[row][col];
    if (delta) { changed.push({ row, col, delta }); if (delta < 0) removed -= delta; else placed += delta; }
  }
  if (isJointFrame(frame)) {
    const machineChanged = frame.agents.some(agent => {
      const old = previous.agents.find(a => a.id === agent.id);
      return !old || agent.position.some((v, i) => v !== old.position[i]) || ['loaded', 'base_yaw', 'cabin_yaw', 'wheel_angle', 'shovel_lifted'].some(key => agent[key] !== old[key]);
    });
    const message = changed.length ? `Joint round · ${changed.length} terrain cells changed` : machineChanged ? 'Joint round · machine states changed' : 'Joint round · no recorded state change';
    return { kind: 'joint', changed, removed, placed, message };
  }
  const actor = frame.agents.find(a => a.id === frame.actor_id);
  const oldActor = previous.agents.find(a => a.id === frame.actor_id);
  const loadDelta = actor && oldActor ? actor.loaded - oldActor.loaded : 0;
  const recipient = frame.agents.find(a => a.id !== frame.actor_id && a.loaded > (previous.agents.find(b => b.id === a.id)?.loaded ?? a.loaded));
  let kind = 'unchanged', message = 'No visible state change';
  if (removed > 0 && loadDelta > 0) { kind = 'dig'; message = `Picked up ${loadDelta} soil units · ${changed.length} cells changed`; }
  else if (placed > 0 && loadDelta < 0) { kind = 'dump'; message = `Placed ${-loadDelta} soil units · ${changed.length} cells changed`; }
  else if (loadDelta < 0 && recipient) { kind = 'transfer'; message = `Transferred soil to machine ${recipient.id + 1}`; }
  else if (changed.length) { kind = 'terrain'; message = `${changed.length} terrain cells changed`; }
  else if (actor && oldActor && actor.position.some((v, i) => v !== oldActor.position[i])) { kind = 'move'; message = 'Machine moved'; }
  else if (actor && oldActor && (actor.base_yaw !== oldActor.base_yaw || actor.cabin_yaw !== oldActor.cabin_yaw || actor.wheel_angle !== oldActor.wheel_angle || actor.shovel_lifted !== oldActor.shovel_lifted)) { kind = 'turn'; message = 'Machine configuration changed'; }
  return { kind, changed, removed, placed, loadDelta, recipient, message };
}

export function terrainFacts(frame) {
  let cut = 0, fill = 0, target = 0;
  for (let row = 0; row < frame.grid.rows; row++) for (let col = 0; col < frame.grid.cols; col++) {
    const height = frame.maps.action[row][col];
    cut += Math.max(0, -height); fill += Math.max(0, height);
    target += Math.max(0, -frame.maps.target[row][col]);
  }
  return { cut, fill, target, carried: frame.agents.reduce((sum, agent) => sum + agent.loaded, 0) };
}

export function shortestAngle(from, to, t) {
  const difference = Math.atan2(Math.sin(to - from), Math.cos(to - from));
  return from + difference * t;
}

export function formatReward(value) {
  if (value === 0) return '0.00';
  const digits = Math.abs(value) < .01 ? value.toPrecision(3) : value.toFixed(2);
  return `${value > 0 ? '+' : ''}${digits}`;
}
