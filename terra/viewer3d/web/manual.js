/** UI availability only. Native action legality always belongs to the server. */
export function manualControls({ replay, index = 0, mode, imported = false, busy = false, playing = false, session = {} }) {
  const frame = replay?.frames[index], live = mode === 'manual' && !imported;
  const latest = !!frame && index === replay.frames.length - 1;
  return {
    action: live && latest && !busy && !playing && (!frame.done || !!session.exploring),
    reset: live && !busy,
    undo: live && latest && !busy && !playing && !!session.can_undo,
    continueVisible: live && latest && !!session.cases?.length && !!frame.done && !frame.task_done && !session.exploring,
    continueEnabled: !busy && !playing,
  };
}

/** Offsets are supplied by Terra: positive is Q, negative is E. */
export function unloadGuidance(diagnostics) {
  if (!diagnostics?.loaded) return 'Empty bucket. Cabin turns keep the base in place.';
  if (diagnostics.accepted_unload_now) return 'Press Space to unload into the accepted dump area.';
  const offsets = diagnostics.accepted_unload_by_cabin_offset;
  if (Array.isArray(offsets) && offsets.length) {
    const choices = offsets.flatMap((allowed, offset) => !allowed || offset === 0 ? [] : [{ key: offset <= offsets.length / 2 ? 'Q' : 'E', turns: Math.min(offset, offsets.length - offset) }]);
    choices.sort((a, b) => a.turns - b.turns);
    if (choices.length) return `${choices[0].key} × ${choices[0].turns}, then Space to unload. Base stays in place.`;
  }
  if (diagnostics.accepted_unload_any) return 'Swing the cabin to an accepted dump direction, then press Space.';
  if (diagnostics.dump_status === 'off_zone_only') return 'Only off-target unloading is available at this base. It will not count as accepted disposal.';
  return 'No unloading direction at this base. Undo is available; the loaded base cannot move.';
}

export const structuredMode = metadata => metadata?.action_mode === 'structured_v1';

/** Only arguments used by the selected action are sent. Legacy payloads stay unchanged. */
export function manualActionRequest(action, metadata, { distance = 5, turn = 1, heading = -1 } = {}) {
  const request = { action };
  if (!structuredMode(metadata)) return request;
  if (action === 0 || action === 1) request.amount = distance;
  if (action === 2 || action === 3) request.amount = turn;
  if (action === 6 && heading !== -1) request.heading = heading;
  return request;
}

export function currentCabinHeading(snapshot) {
  const native = snapshot?.diagnostics?.structured_actions?.current_heading;
  if (Number.isInteger(native)) return native;
  const agent = snapshot?.agents.find(item => item.id === snapshot.current_agent);
  return ((Math.round((agent?.cabin_yaw ?? 0) * 12 / (2 * Math.PI)) % 12) + 12) % 12;
}

/** Native masks include rehandling and off-target unloading; fresh-dig counts are not masks. */
export function manualActionAvailable(request, snapshot, metadata) {
  if (!structuredMode(metadata)) return true;
  const masks = snapshot?.diagnostics?.structured_actions;
  let value;
  if (request.action === 0 || request.action === 1) value = masks?.move_mask?.[request.action]?.[(request.amount ?? 5) - 1];
  if (request.action === 2 || request.action === 3) value = masks?.turn_mask?.[request.action - 2]?.[(request.amount ?? 1) - 1];
  if (request.action === 6) value = masks?.do_mask?.[request.heading ?? currentCabinHeading(snapshot)];
  // Missing masks in older exports are unknown, never an invented rejection.
  return value !== false && value !== 0;
}

/** One-cell translations cannot retain oblique headings on the integer grid. */
export function moveDistanceControls(snapshot, selectedDistance) {
  const masks = snapshot?.diagnostics?.structured_actions?.move_mask;
  const rejected = value => value === false || value === 0;
  const oneCellDisabled = rejected(masks?.[0]?.[0]) && rejected(masks?.[1]?.[0]);
  const agent = snapshot?.agents?.find(item => item.id === snapshot.current_agent);
  const oblique = Number.isFinite(agent?.base_yaw) && Math.round(agent.base_yaw * 12 / (2 * Math.PI)) % 3 !== 0;
  const angledMinimum = oneCellDisabled && oblique && agent.loaded === 0;
  const nextAllowed = angledMinimum ? [2, 3, 4, 5].find(amount => masks?.some(row => row[amount - 1] === true || row[amount - 1] === 1)) : undefined;
  return {
    distance: selectedDistance === 1 && nextAllowed !== undefined ? nextAllowed : selectedDistance,
    oneCellDisabled,
    hint: angledMinimum ? 'Angled headings need at least 2 cells on this grid.' : '',
  };
}

export function cabinHeadingLabel(heading, current = 0) {
  const direction = heading === 0 ? '0° · forward' : heading === 6 ? '180° · rear' : heading < 6 ? `${heading * 30}° left` : `${(12 - heading) * 30}° right`;
  const offset = (heading - current + 12) % 12;
  return `${direction} · ${offset === 0 ? 'current' : offset <= 6 ? `Q ${offset * 30}°` : `E ${(12 - offset) * 30}°`}`;
}

export function structuredBudget(diagnostics = {}, snapshot = {}, exploring = false) {
  const format = value => Number.isFinite(value) ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value) : '—';
  const reason = { time_budget: 'Time budget reached', decision_budget: 'Decision budget reached', completed: 'Task complete', task_done: 'Task complete' }[diagnostics.termination_reason] || 'Episode ended';
  const remaining = diagnostics.remaining_time_s ?? Math.max(0, diagnostics.time_budget_s - diagnostics.elapsed_time_s);
  return {
    time: `${format(diagnostics.elapsed_time_s)} / ${format(diagnostics.time_budget_s)} s`,
    decisions: `${format(diagnostics.decisions)} / ${format(diagnostics.decision_budget)}`,
    note: exploring ? 'Exploration · outside the episode budget' : snapshot.task_done ? 'Completed within the episode' : snapshot.done ? `${reason} · episode frozen` : `${format(remaining)} s remaining`,
  };
}
