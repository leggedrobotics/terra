/** Metric plan playback. This is a modeled plan, never a Terra environment episode. */
export const POSTPROCESSED_SCHEMA = 'terra.postprocessed.v1';
const finite = value => typeof value === 'number' && Number.isFinite(value);
const require = (value, message) => { if (!value) throw new Error(`Postprocessed plan: ${message}`); };
const phaseKind = phase => ({ cut: 'dig', collect: 'dig', collection: 'dig', excavate: 'dig', drive: 'move', arrive: 'move', arrival: 'move' }[phase] ?? phase);

export function maskFromRuns(runs, rows, cols, fill = 0) {
  const out = Array.from({ length: rows }, () => Array(cols).fill(fill));
  for (const run of runs ?? []) {
    require(Array.isArray(run) && run.length === 3 && run.every(Number.isInteger), 'mask runs must be [row, column, width]');
    const [row, col, width] = run;
    require(row >= 0 && row < rows && col >= 0 && width > 0 && col + width <= cols, 'mask run is outside the grid');
    out[row].fill(1, col, col + width);
  }
  return out;
}

export function validatePostprocessed(data) {
  require(data?.schema === POSTPROCESSED_SCHEMA, `expected ${POSTPROCESSED_SCHEMA}`);
  const g = data.grid, size = g?.rows * g?.cols;
  require(g && Number.isInteger(g.rows) && Number.isInteger(g.cols) && g.rows > 0 && g.cols > 0 && size <= 1048576, 'grid must contain 1 to 1,048,576 cells');
  require(finite(g.resolution_m) && g.resolution_m > 0 && Array.isArray(g.origin_xy_m) && g.origin_xy_m.length === 2 && g.origin_xy_m.every(finite), 'grid needs metric resolution and its first cell centre');
  require(g.yaw_rad == null || finite(g.yaw_rad), 'grid yaw must be finite');
  require(Array.isArray(data.agents) && data.agents.length > 0 && data.agents.length <= 4, 'expected one to four machines');
  const ids = new Set();
  for (const a of data.agents) {
    require(Number.isInteger(a.id) && a.id >= 0 && !ids.has(a.id) && [0, 1, 2].includes(a.type) && [0, 1].includes(a.action_type), 'machine identities and types must be explicit');
    require(finite(a.width_m) && a.width_m > 0 && finite(a.length_m) && a.length_m > 0, 'machine footprint must be in metres');
    require(Array.isArray(a.reach_m) && a.reach_m.length === 2 && a.reach_m.every(finite) && a.reach_m[0] >= 0 && a.reach_m[1] >= a.reach_m[0], 'machine reach must be [inner, outer] metres');
    ids.add(a.id);
  }
  const states = states => {
    require(Array.isArray(states) && states.length === ids.size && new Set(states.map(a => a.id)).size === ids.size, 'every frame must keep every machine once');
    for (const a of states) {
      require(ids.has(a.id) && Array.isArray(a.pose) && a.pose.length === 3 && a.pose.every(finite), 'machine pose must be [map x, map y, yaw]');
      require(finite(a.cabin_yaw) && finite(a.load) && a.load >= 0 && finite(a.wheel_angle) && [0, 1, false, true].includes(a.shovel_lifted), 'invalid machine configuration');
    }
  };
  for (const name of ['native_m', 'loose_m']) require(Array.isArray(data.initial?.[name]) && data.initial[name].length === size && data.initial[name].every(value => finite(value) && (name !== 'loose_m' || value >= 0)), `${name} must contain one metric height per cell`);
  states(data.initial.agents);
  const workspaces = new Set();
  for (const w of data.workspaces ?? []) {
    require(w.id != null && !workspaces.has(w.id) && ids.has(w.agent_id), 'workspace identities and owners must be explicit');
    workspaces.add(w.id);
    for (const mask of Object.values(w.masks ?? {})) maskFromRuns(mask, g.rows, g.cols);
  }
  for (const name of ['known', 'obstacle', 'final', 'target']) if (g[name]) maskFromRuns(g[name], g.rows, g.cols);
  if (g.design_m) require(Array.isArray(g.design_m) && g.design_m.length === size && g.design_m.every(finite), 'design heights must match the metric grid');
  require(Array.isArray(data.frames), 'frames must be a list');
  for (const f of data.frames) {
    states(f.agents);
    require(typeof f.phase === 'string' && (f.agent_id == null || ids.has(f.agent_id)), 'frame phase and owner are required');
    require(f.workspace_id == null || workspaces.has(f.workspace_id), 'frame workspace does not exist');
    require(f.route_status == null || ['checked', 'failed', 'missing', 'unverified'].includes(f.route_status), 'unknown route status');
    require(f.duration_s == null || finite(f.duration_s) && f.duration_s > 0, 'display duration must be positive');
    const changed = new Set();
    for (const c of f.terrain_changes ?? []) {
      require(Array.isArray(c) && c.length === 3 && Number.isInteger(c[0]) && c[0] >= 0 && c[0] < size && finite(c[1]) && finite(c[2]) && c[2] >= 0 && !changed.has(c[0]), 'invalid or duplicate terrain change');
      changed.add(c[0]);
    }
    const owned = new Set();
    for (const w of f.work ?? []) {
      require(ids.has(w.agent_id) && typeof w.kind === 'string' && (w.recipient_id == null || ids.has(w.recipient_id)), 'invalid work owner or recipient');
      for (const i of w.changed_indices ?? []) { require(changed.has(i) && !owned.has(i), 'work cells must name a changed cell exactly once'); owned.add(i); }
    }
    require(owned.size === changed.size, 'changed terrain needs explicit ownership for every cell');
  }
  return data;
}

/** Inverse of origin + R(grid yaw) * [column, row] * resolution. */
export function metricCell(x, y, grid) {
  const yaw = grid.yaw_rad ?? 0, c = Math.cos(yaw), s = Math.sin(yaw), dx = x - grid.origin_xy_m[0], dy = y - grid.origin_xy_m[1];
  return [(c * dy - s * dx) / grid.resolution_m, (c * dx + s * dy) / grid.resolution_m];
}

/** Map origin is cell [0,0]'s centre. Grid X -> scene X; grid Y -> -Z. */
export function metricAgent(agent, state, grid) {
  const tile = grid.resolution_m, [x, y, yaw] = state.pose;
  return { id: agent.id, type: agent.type, action_type: agent.action_type,
    position: metricCell(x, y, grid),
    base_yaw: yaw - (grid.yaw_rad ?? 0), cabin_yaw: state.cabin_yaw - yaw, width: agent.width_m / tile, height: agent.length_m / tile,
    reach: agent.reach_m.map(v => v / tile), loaded: state.load, wheel_angle: state.wheel_angle / (Math.PI / 9),
    shovel_lifted: Number(state.shovel_lifted), metric_pose: [...state.pose],
  };
}

/** Preserve route ownership and discontinuities, including fleet legs with no workspace. */
export function postprocessedRoutes(data) {
  const routes = new Map(), byFrame = new Map(), byWorkspace = new Map();
  let previous = new Map(data.initial.agents.map(agent => [agent.id, agent]));
  data.frames.forEach((frame, i) => {
    if (['drive', 'relocate'].includes(frame.phase)) {
      const keys = [];
      for (const agent of frame.agents) {
        const before = previous.get(agent.id);
        if (agent.id !== frame.agent_id && agent.pose[0] === before.pose[0] && agent.pose[1] === before.pose[1]) continue;
        const key = JSON.stringify([agent.id, frame.route_id ?? null, frame.workspace_id ?? null]);
        let route = routes.get(key);
        if (!route) { route = { agent_id: agent.id, workspace_id: frame.workspace_id ?? null, segments: [] }; routes.set(key, route); }
        const status = frame.route_status ?? 'unverified', last = route.segments.at(-1), end = last?.points.at(-1);
        const connected = end && Math.hypot(end[0] - before.pose[0], end[1] - before.pose[1]) < 1e-7;
        const segment = connected && last.status === status && !last.relocate && frame.phase !== 'relocate'
          ? last : { points: [[...before.pose]], status, relocate: frame.phase === 'relocate' };
        if (segment !== last) route.segments.push(segment);
        segment.points.push([...agent.pose]); keys.push(key);
        if (frame.workspace_id != null) {
          if (!byWorkspace.has(frame.workspace_id)) byWorkspace.set(frame.workspace_id, new Set());
          byWorkspace.get(frame.workspace_id).add(key);
        }
      }
      byFrame.set(i + 1, keys);
    }
    previous = new Map(frame.agents.map(agent => [agent.id, agent]));
  });
  return { routes, byFrame, byWorkspace };
}

export class PostprocessedEpisode {
  constructor(data) {
    this.data = validatePostprocessed(data); this.count = data.frames.length + 1;
    const { rows, cols } = data.grid;
    this.nested = flat => Array.from({ length: rows }, (_, row) => Array.from(flat.slice(row * cols, (row + 1) * cols)));
    this.native = this.nested(data.initial.native_m); this.loose = this.nested(data.initial.loose_m);
    this.at = 0; this.cache = null;
    const known = maskFromRuns(data.grid.known, rows, cols, data.grid.known == null ? 1 : 0);
    const obstacle = maskFromRuns(data.grid.obstacle, rows, cols), final = maskFromRuns(data.grid.final, rows, cols);
    const target = maskFromRuns(data.grid.target ?? (data.workspaces ?? []).flatMap(w => w.masks?.finish ?? []), rows, cols);
    this.maps = { target: target.map((line, row) => line.map((v, col) => v ? (data.grid.design_m?.[row * cols + col] ?? -1) : final[row][col])), padding: obstacle.map((line, row) => line.map((v, col) => Number(v || !known[row][col]))), dumpability: final, dumpability_static: null, interaction: null, traversability: null };
  }

  event(index) { return index ? this.data.frames[index - 1] : { phase: 'initial', agents: this.data.initial.agents }; }

  frame(index) {
    require(Number.isInteger(index) && index >= 0 && index < this.count, 'frame index is outside the playback');
    if (this.cache?.step === index) return this.cache;
    if (index < this.at) { this.native = this.nested(this.data.initial.native_m); this.loose = this.nested(this.data.initial.loose_m); this.at = 0; }
    const cols = this.data.grid.cols;
    // Copy only touched rows; previous scene frames keep their exact endpoint heights.
    for (let f = this.at; f < index; f++) {
      const native = [...this.native], loose = [...this.loose], copied = new Set();
      for (const [i, n, l] of this.data.frames[f].terrain_changes ?? []) {
        const row = Math.floor(i / cols), col = i % cols;
        if (!copied.has(row)) { native[row] = [...native[row]]; loose[row] = [...loose[row]]; copied.add(row); }
        native[row][col] = n; loose[row][col] = l;
      }
      this.native = native; this.loose = loose;
    }
    this.at = index;
    const event = this.event(index), states = new Map(event.agents.map(a => [a.id, a]));
    const agents = this.data.agents.map(a => metricAgent(a, states.get(a.id), this.data.grid));
    const changes = event.terrain_changes ?? [], work = (event.work ?? []).map(w => ({ ...w, kind: phaseKind(w.kind) }));
    const owned = new Set(work.flatMap(w => w.changed_indices ?? []));
    const unowned = changes.map(c => c[0]).filter(i => !owned.has(i));
    if (event.agent_id != null && (unowned.length || !work.some(w => w.agent_id === event.agent_id))) work.push({ agent_id: event.agent_id, kind: phaseKind(event.phase), changed_indices: unowned });
    this.cache = { step: index, action: null, actor_id: event.agent_id ?? null, current_agent: event.agent_id ?? agents[0].id,
      reward: 0, done: false, task_done: false, grid: { rows: this.data.grid.rows, cols, tile_size_m: this.data.grid.resolution_m },
      maps: { ...this.maps, action: this.native.map((line, row) => line.map((v, col) => v + this.loose[row][col])) }, agents,
      metric: { native: this.native, loose: this.loose, work, terrain_changes: changes, event },
    };
    return this.cache;
  }

  duration(index) {
    const event = this.event(index);
    if (event.duration_s) return event.duration_s * 1000;
    if (event.phase === 'drive') {
      const before = new Map(this.event(index - 1).agents.map(a => [a.id, a]));
      const distance = Math.max(...event.agents.map(a => Math.hypot(a.pose[0] - before.get(a.id).pose[0], a.pose[1] - before.get(a.id).pose[1])));
      return Math.max(90, Math.min(1400, distance / 1.2 * 1000));
    }
    return ({ dig: 1500, dump: 1400, transfer: 1400, move: 350 }[phaseKind(event.phase)] ?? 650);
  }
}
