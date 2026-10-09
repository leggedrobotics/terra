# Structured solo excavator actions

`terra.structured_actions` is an opt-in functional environment interface. It
preserves the legacy `State`, config, saved-state pickle and eight-action API.
It supports one tracked excavator with twelve chassis and cabin headings.

Structured actions do not enable the optional working-strip rule. Saved states
carry that rule independently, and its width defaults to zero (disabled).
The manual game's 1 m by 1.3 m strip has a known training-bank compatibility
problem; see [the reported coverage audit](PULL_DIRECTION_ALIGNMENT.md#bank-compatibility)
before selecting an initial-state bank for training.

| Type | Argument | Execution |
| --- | --- | --- |
| 0 / 1 forward / backward | `amount=1..5` cells | Existing swept translation and turn-keeping rule; angled headings require at least 2 cells. |
| 2 / 3 clockwise / counterclockwise | `amount=1..6` | Sequential native 30-degree turns; stop at the first blocked orientation. |
| 6 dig or unload | `heading=0..11`, or `-1` for current | Aim cabin and execute the actual native DO. Dig and unload remain separate decisions. |
| 4 / 5 cabin rotation | none | Retained for manual controls and legacy tapes. |
| 7 wait | none | Decision-cap fallback. |

Heading is the stored cabin index **relative to the chassis**, not a world
heading or an offset from the current cabin. Zero is chassis-forward; each
positive increment is 30 degrees. A selected work heading uses the shortest
cabin swing. The native abstraction does not model arm sweep collisions.

Move, turn and work masks evaluate actual native outcomes. Productive relifting
and complete off-zone unloading remain available. A failed full-load dump
cannot become valid merely because its candidate placement mask is nonempty.
An infeasible work request leaves cabin and material unchanged; it never
silently substitutes another heading. The game rejects disabled requests.
Clipped movement/rotation requests are valid if they execute any progress,
except a one-cell result at an oblique chassis heading. Rounding a one-cell
30/60-degree displacement to the integer grid can erase its minor component
and produce repeated motion along the wrong direction. Structured actions
therefore allow one-cell moves only at cardinal headings, and reject longer
requests clipped to such a one-cell endpoint. This does not change legacy
movement. Larger oblique moves retain the native grid approximation.

## Time and reward

The caller carries `StructuredClock(visit_open, moved)` and elapsed seconds
alongside the native state. `structured_transition` returns the new native
state, reward, duration and diagnostics; one call increments `env_steps` once.
`structured_termination` supplies the shared time/decision-budget result and
terminal reward. Callers apply that reward once and reset all clocks together.

The `material_time_v1` reward is the undiscounted material-potential difference,
existing optional behavior costs, and `-3.6 * duration / time_budget`. Success
adds 6; budget exhaustion adds -1. It bypasses the legacy 450-step reward guard
and per-action cost. The trainer's discount/GAE settings are a separate choice,
recorded with each experiment; these returns are a new protocol.

Time estimates use 226 s/m³ for loading work (including dumping), 415 s per
workspace visit, 15 s relocation overhead, 0.5 m/s travel, 5 s/rad chassis
turning, and cabin angular speed 0.28 rad/s. Material volume uses `tile_size³`
per native unit. Loading/setup are based on the October 7 field estimate;
navigation, relocation and chassis-turn values include assumptions. This is
modeled time, not a prediction validated across real tasks. Dumping is not
charged a second per-volume work cost. Effective chassis motion closes the
current visit, including a move away and back to the same position.

The manual default is a provisional 14,400-second budget and a 450-decision
cap. Waits and blocked requests have zero modeled duration but consume a
decision. Actions finish atomically; completing after the time budget is a
timeout. Exact completion at the budget boundary is allowed. Remaining time
is exposed to the policy and viewer. Budget comparisons must use the same
time model and report completion, disposal, modeled time and decisions.

## Manual game

```bash
JAX_PLATFORMS=cpu python -m terra.viewer3d \
  --pull-inspector --structured-actions \
  --initial-states /path/to/initial_states.pkl \
  --time-budget-seconds 14400 --decision-budget 450 --port 8768
```

The selectors control distance, turn amount and work heading. Keyboard
movement and Space use the selected arguments. Q/E remain direct cabin
controls. Masks apply to keyboard and button requests. Undo restores native
state, elapsed time, setup/relocation flags, arguments and terminal status.
Exports include replay frames, native states, argument-bearing action records
and `structured_clock.json`. Reset saves a nonempty recording first. Use
`--resume-recording /path/to/export_directory` to restore a trusted export
with its board, complete action history and modeled clock after a restart.

The structured trainer is `train_structured.py` in the paired
`terra-baselines` checkout. Its explicit checkpoint/action protocol is
independent of the legacy `train_mixed.py` categorical-action runs. See its
CLI and structured training documentation for supported warm starts and
validation. Partial digging, dump-placement parameters and fused dig/unload
are not part of this version.
