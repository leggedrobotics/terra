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
