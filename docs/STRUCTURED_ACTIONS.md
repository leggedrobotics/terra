# Structured solo excavator actions

`terra.structured_actions` is an opt-in functional environment interface. It
preserves the legacy `State`, config, saved-state pickle and eight-action API.
It supports one tracked excavator with twelve chassis and cabin headings.

Native digging has no bucket-width constraint; trench width is handled during
postprocessing. The temporary rectangular working-strip gate has been removed.
Saved states still carry their pull-length setting independently of the action
space; the manual game uses 1 m. See [pull admission](PULL_DIRECTION_ALIGNMENT.md).

## Precision safety and observations (October 12)

`EnvConfig.precision_safety` defaults off for historical states. New structured
training enables it explicitly. Soil relaxation cannot spread into the acting
excavator's unreachable inner radius, and fresh excavation cannot violate the
current chassis's dug-clearance distance. Existing soil is conserved; loose-soil
recovery remains available. Oblique move endpoint selection ignores one-cell
candidates before applying turn preference. None of these rules proves egress.

`LocalMapWrapper.wrap(..., precision_observation_v2=True)` builds remaining-edge
and executable fresh-edge inputs without changing target boundary geometry.
Edge errors remain float32, and all local heading arrays are indexed by the
chassis-relative DO heading. The wrapped state is an observation copy, not the
native rollout state. Historical preprocessing remains available for old weights.

`StructuredEpisodeConfig` additionally records `precision_safety`,
`precision_edge_weight` (`-1` means legacy volume weighting), and
`terminate_unloaded_dead_ends`. The viewer reads them from trainer protocol JSON,
preserves them on reset/export/restore, and reports `unloaded_dead_end` separately
from `loaded_no_unload`. Missing fields in historical recordings retain their
legacy values. The shared episode helper charges the existing failure penalty
once; it does not subtract lost potential again.

The unloaded certificate retains any possible radial fresh work or misplaced
soil repair, even if current heading constraints reject it. It terminates only
when holes/unreachable inner soil block all geometric move/turn exits and no
such work remains. It deliberately misses uncertain traps rather than stopping
recoverable episodes. Complete precision excavation/disposal and learned policy
quality remain separate qualification results.

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
`terra.structured_episode` supplies shared map-dependent limits and terminal
rewards for the trainer and viewer. Callers apply that reward once and reset
all clocks together. Its lower-level `structured_termination` remains available
for callers using fixed limits.

`structured_loaded_dead_end(state, do_mask)` recognizes a loaded solo tracked
excavator with no executable full-load unload at any heading. It consumes the
already computed exact DO row, including off-zone staging. Pass that result as
`dead_end` to `structured_termination` to end the episode with -1 and a separate
dead-end flag; it is not a budget timeout. This detects a trap after it occurs;
it does not prevent a dig from creating one. Empty-machine planning failures
are not inferred from this check.

The manual viewer checks the post-action row immediately, reports
`loaded_no_unload`, and keeps undo/export available. The paired trainer reuses
its next observation's row and terminates after one zero-duration fallback
WAIT. That adds one recorded decision compared with immediate manual
termination but no modeled time. Undiscounted return is unchanged; for lambda
below one, GAE still includes the value estimate of the intermediate loaded
state. The trainer saves this option in its checkpoint protocol, preserving
explicit legacy evaluation/resume behavior. See the
[ordered fix checklist](ENVIRONMENT_FIXES_20261010.md).

The default `material_time_v1` reward is the undiscounted material-potential difference,
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
These are the manual defaults; an explicit protocol replaces all timing
coefficients, including the dense time-cost coefficient and setup-per-dig flag.

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

To match a structured training run, supply its saved protocol explicitly:

```bash
JAX_PLATFORMS=cpu python -m terra.viewer3d \
  --pull-inspector --structured-actions \
  --initial-states /path/to/matching_initial_states.pkl \
  --structured-protocol /path/to/training_run/config.json --port 8768
```

The JSON can be a trainer `config.json` with a `training` object, or a flat
object. It must include a complete `timing` object and explicit
`time_budget_s`, `time_budget_factor`, `time_budget_offset_s`, `decision_limit`,
`decisions_per_dig_unit`, `success_time_bonus`, `success_decision_bonus`, plus
`time_limit` (or the trainer's inverse `no_time_limit`). Set
`terminate_loaded_dead_ends` explicitly; omission preserves old trainer
behavior with that option disabled. The protocol flag cannot be combined with
the fixed-budget CLI flags. It does not alter the native geometry saved in
the initial-state bank; use the same bank/rules as the intended evaluation.

Each reset derives the time reference and decision cap from its selected map.
The cap is `max(decision_limit, ceil(decisions_per_dig_unit * dig_units))`.
When time scaling is enabled, the reference is the larger of `time_budget_s` and
`time_budget_offset_s + time_budget_factor * target_volume * dig_s_per_m3`.
For `no_time_limit: true`, the display says **Time reference**; exceeding it
does not end the episode. The reference still scales time-based rewards and
policy observations. The decision cap still applies. Success-only bonuses are included in the shared terminal
reward and paid once, including when continuing exploration after completion.

Exports include the resolved episode settings and all timing coefficients in
replay/native metadata and `structured_clock.json`, alongside each map's
limits. Restore rejects mismatched protocols or limits. Old exports without
protocol metadata can only be restored with their fixed budgets and the
original default timing/reward settings. Recordings from before the dead-end
fix require `terminate_loaded_dead_ends: false`, inferred from their historical
diagnostics. Changing protocols requires a fresh
episode rather than reinterpreting its old clock and rewards.

The structured trainer is `train_structured.py` in the paired
`terra-baselines` checkout. Its explicit checkpoint/action protocol is
independent of the legacy `train_mixed.py` categorical-action runs. See its
CLI and structured training documentation for supported warm starts and
validation. Partial digging, dump-placement parameters and fused dig/unload
are not part of this version.
