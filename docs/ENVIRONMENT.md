# Terra environment, rewards and episode contract

Audited from local source and experiment records on September 8, 2026.
This is the methods reference for repository documentation; it does not change
the paper or the training implementation. Dataset composition is documented in
[DATASET.md](DATASET.md). Optimization and experiment-specific policy inputs
are documented in the sibling
[training protocol](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/TRAINING_PROTOCOL.md).

Source links identify public code where available. References marked **local**
are paths relative to the original `/home/lorenzo/moleworks` workspace. Input
banks and unpublished experiment snapshots are separate from this documentation.
The September optional behaviors describe those recorded snapshots and may not
be implemented in the published main branch.

## Which environment this describes

Terra is a JAX environment for global excavation planning. It represents an
excavator choosing base poses, cabin directions, workspace excavation and spoil
placement on a discrete grid. A `DO` action can remove or deposit a workspace's
worth of material. It is not a simulation of an individual bucket trajectory,
hydraulics, contact forces or measured excavation time.

There are several experiment generations. Their results must retain the
environment revision and resolved configuration:

| Scope | Terra source | Environment differences that matter |
| --- | --- | --- |
| Local checkout used during this audit | `b87c70f6` | Older source; its default dense reward and historical completion implementation do not define the newer experiments below. |
| Historical V8 relay / recurrent comparisons | `25f855db` | Solo tracked excavator, corrected visible-dump success and reward-v2 support; no fresh-trench alignment gate. The relay feed-forward run additionally applied `ebdc3ad7`, preventing excavation beneath the base and preserving terrain blockers in the observation. |
| September 7 restart of the v2 generalist and trench specialist | `46b140f8` | Per-cell trench admission, corrected footprint/soil containment and matching admissibility observations. |
| September 7–8 easy-foundation reward screen | `fa8d5d13` | Same current mechanics, with optional costs for fresh lateral excavation, actual base travel and actual base rotation, plus the optional executable-dig observation. |

The detailed source anchors below use `fa8d5d13`; the source index identifies
the corresponding local worktree. The shared reward-v2 constants and exact
completion formula also exist in the historical V8 runtime. Later geometry
fixes do not retroactively validate older trajectories.

The [V8 handover](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/research/V8_PAPER_EXPERIMENT_HANDOVER_20260818.md)
records the historical runtime differences. The
[experiment ledger](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/EXPERIMENTS_RUNNING.md) and
foundation run record (local: `.artifacts/terra_foundation_sweep_20260907/RUNS.md`)
record newer source pairs and resolved treatments. This document reports those
recorded configurations, not live scheduler status.

## State, maps and units

The world state includes the immutable target map, obstacle/padding mask,
initial dumpability mask, geometric metadata and distance-to-dump field; the
mutable action map, current dumpability and previous excavation mask; agent
poses and carried material; reward bookkeeping; and the elapsed decision count.

| Field | Meaning |
| --- | --- |
| `target_map < 0` | Required excavation. Its magnitude is required discrete depth. |
| `target_map > 0` | Declared acceptable final spoil region. This is a location label, not a required final pile height. |
| `target_map == 0` | No required excavation or final spoil acceptance. Some such cells can permit temporary staging. |
| `action_map < 0` | Excavated depth relative to the initial flat surface. |
| `action_map > 0` | Loose or deposited material above that surface. |
| `padding_mask == 1` | Static obstacle or padded map region. |
| `dumpability_mask_init` | Where material may physically be placed, before dynamic clearance rules. It is distinct from final spoil acceptance. |
| `last_dig_mask` | Recent excavation area used by anti-cycle action filters; it is part of transition state. |

For the audited V8, v2 and easy-foundation protocols, the map is `64 × 64`
cells spanning approximately `36.5714 × 36.5714 m`, giving
`s = 0.5714285714 m/cell`. A unit-depth target cell contributes one abstract
material unit and `s² ≈ 0.326531 m²` of plan area. Neither a material unit nor
one depth level should be labelled a cubic metre without an explicitly declared
vertical scale. These tasks use unit excavation depth; the sampler's difficulty
`depth` is a separate dataset label.

The nominal machine dimensions are `6.08 × 3.5 m`. The environment rounds each
dimension to an odd cell extent, producing stored extents `11 × 7` cells
(`6.29 × 4.00 m` before rotation and footprint rasterization). State positions
are row/column grid coordinates. Cabin yaw is relative to the chassis; the
workspace direction combines chassis and cabin yaw. Coordinate conversions
must preserve that convention.

Source: `config.py:9–11,64–110`; `env.py:826–860`;
`state.py:1238–1271` and `_accepted_dump_mask`.

## Actions and transition mechanics

The paper-relevant reward-v2 experiments use one tracked excavator. The broader
environment supports up to four active excavators, trucks or skid steers and
tracked or wheeled motion. Those capabilities are not evidence that the solo
experiments trained or evaluated multi-agent policies.

| Index | Tracked action | Solo-excavator effect |
| ---: | --- | --- |
| 0 | `FORWARD` | Translate in the chassis heading, if empty and the destination footprint is valid. |
| 1 | `BACKWARD` | Translate against the heading, subject to the same restrictions. |
| 2 | `CLOCK` | Rotate the chassis one clockwise bin, if empty and the resulting footprint is valid. |
| 3 | `ANTICLOCK` | Rotate the chassis one anticlockwise bin under the same conditions. |
| 4 | `CABIN_CLOCK` | Rotate the cabin one clockwise bin. |
| 5 | `CABIN_ANTICLOCK` | Rotate the cabin one anticlockwise bin. |
| 6 | `DO` | If empty, attempt excavation or positive-soil pickup; if loaded, attempt unloading. |
| 7 | `DO_NOTHING` | Leave the physical state unchanged. |

Base and cabin orientations each have 12 bins, so one rotation action is
30 degrees. A move proposes five cells and rounds its endpoint to the integer
grid. Its actual metric displacement depends on heading and rounding; five
cells is a nominal step, not a fixed measured travel distance. A loaded solo
excavator can rotate its cabin but cannot translate or rotate its chassis.
Distant soil transport therefore requires reachable placement, repositioning
while empty and, where needed, relifting staged soil.

Every attempted action, including a rejected action and explicit no-op,
increments the decision count. Invalid motion is a physical no-op rather than
a separate collision termination. Translation and base rotation validate the
candidate footprint, not a continuous swept path. Cabin motion does not model
arm collision or dynamic stability.

For current motion feasibility, holes (`action_map < 0`), piles above one unit,
and static obstacles block motion. A nonzero terrain cell also blocks motion
when at least six of its `3 × 3` neighborhood cells are nonzero. Some isolated
unit-height spoil is therefore traversable. The source comments mentioning an
eight-cell threshold are stale; the implemented threshold is six.

Source: `actions.py:13–101`; `state.py:286–318,538–675,703–731,897–1199`.

### Workspace excavation

The excavator's workspace is a cabin-centered angular sector with radial reach
from `3.642857 m` to `6.5 m` and angular range `±30°`. These radii follow from
the resolved footprint, cell size, fixed `0.5 m` extension and five-cell radial
width. Additional footprint and connected-workspace filters narrow the sector.

An empty `DO` selects target cells still eligible for excavation or existing
positive soil. If the raw workspace contains positive soil, the implemented
selection prefers positive-soil pickup over fresh excavation; this can prevent
fresh digging even when target cells are also present. Recent-workspace,
depth, base-footprint, optional foundation-edge and trench-admission filters
also apply. Any static obstacle in the cleaned dig cone vetoes the complete
dig, even if some selected target cells would otherwise be valid.

Fresh excavation removes one depth unit per selected cell in the unit-depth
protocol. The resulting load is the actual material removed from the action
map. Singleton excavation is permitted in the newer V8 and current runtimes.
The excavator has no separately configured 52-unit bucket capacity: fresh load
must fit signed eight-bit load storage (`≤127`), and positive-soil pickup is
bounded at 127 units, preserving the remainder. The number 52 appears as the
truck/skid-steer capacity default and as a legacy workspace-efficiency
reference, not a physical solo-excavator bucket volume.

Source: `state.py:1436–1612,2155–2276,2796–2916`; `settings.py`;
`config.py:234–238`.

### Dumping, temporary staging and material conservation

The final accepted mask is

```text
A = (target_map > 0) AND (padding_mask != 1).
```

A dump also needs physical reach, current dumpability, terrain/traversability
clearance and the recent-workspace restrictions. The dynamic dumpability mask
removes a `5 × 5` dilation of excavated holes from initially dumpable cells.
Consequently, a cell can belong to the final accepted mask while being
temporarily unavailable for a particular dump.

The solo excavator first uses reachable accepted cells. It considers off-zone
staging only when no accepted cell survives the physical filters. A dump places
the complete load or leaves the state unchanged. If some accepted cells exist
but the complete deposit fails, the implementation does not then retry the
off-zone alternative. Global capacity or reset admission cannot establish that
every future workspace can accept every load.

The load is initially concentrated within two cells of the selected dump-mask
centroid, with a nearest-valid-cell fallback. Integer division and a
deterministic remainder allocation distribute the complete volume. A local
relaxation applies three passes over four cardinal neighbor directions, moving
one material unit when the height difference is at least two. This is a bounded
grid redistribution rule, not calibrated soil mechanics or a physical
angle-of-repose model.

Current relaxation is contained to the selected accepted/off-zone region on
dump and to accepted cells on excavation/relift. A dump commits only if the map
gain equals the removed load, changes stay in the permitted region and values
fit storage. Transition diagnostics separately check total action-map sum plus
active-agent loads. Final task success alone is not a substitute for those
integrity checks.

Source: `map.py:32–54`; `state.py:1614–1797,2070–2117,2796–2916,3034–3155`;
`env.py:229–311`.

### Current trench admission versus historical V8

The historical V8 experiments did not enforce fresh-trench alignment. The
current v2 specialist and generalist enable a geometric gate for fresh trench
excavation. A section accepts a pose when the chassis is parallel to its axis
within `0.2619 rad` (approximately 15 degrees, with numerical slack) and the
base center is within `2 m` perpendicular distance of that axis. Radial
working distance is imposed by the workspace itself.

Finite-section membership assigns target cells to one or more trench axes.
At a junction, any owning section that accepts the pose can admit the cell.
The current gate removes inadmissible fresh cells from the candidate dig;
it does not reject an otherwise useful dig solely because another section in
the cone is misaligned. This per-cell rule replaced the earlier whole-action
junction veto. Positive-soil pickup, dumping and foundation-only cells do not
receive a fresh-trench alignment constraint.

The older `trench_dig_standoff_enforced=True` variant instead required a
perpendicular band of `3.5–7 m`. It is retained for historical replay and is
not the current v2 setting. Foundation-edge alignment is separately optional
and is disabled in the audited V8, v2 and easy-foundation recipes. The
foundation reward screen's lateral preference is a soft cost; it does not
turn sideways excavation into a forbidden action.

Source: `config.py:248–309`; `state.py:2325–2779`;
[restart record](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/EXPERIMENTS_RUNNING.md).

## Observations and reset state

Terra exposes an observation dictionary, and the trainer selects and encodes a
subset. The dictionary must not be described as the exact policy tensor.
The current interface provides:

| Component | Representation and meaning |
| --- | --- |
| Agent state | Four padded rows of nine scalars: row, column, base bin, relative cabin bin, steering bin, load, agent type, shovel state and carried relocation work divided by required volume. The acting agent is first; an active mask and count identify padding. |
| Global maps | Action, target, observed traversability, optional reachability, padding/obstacles, dumpability and current interaction workspace. |
| Local features | Nine 12-entry vectors describing positive/negative action and target material, dumpability, obstacles and foundation-edge workspace/alignment/diggability across cabin directions. These are aggregated sectors, not image crops. |
| Optional current features | Static relocation-distance map; a 12-entry admissible-dig vector; normalized material-stall age; the two reset baselines; trench alignment diagnostics; and explicitly enabled movement-feasibility or previous-outcome feedback. |

The foundation screen can replace the historical local admissibility counts
with counts of fresh material actually executable by `DO` at each relative
cabin heading. This uses the actual cleaned cone and all dig filters, returns
zero while loaded or when the operation would relift spoil, and preserves the
existing 12-entry input width. Runs A and B differ in this replacement, so
their observation semantics must remain explicit.

The observed traversability map marks every nonzero action-map cell as blocked,
which is stricter than the selective-spoil movement rule above. Reachability is
optional and disabled by default. Current PPO recipes clip the global action
map to `[-1,1]`, so positive pile heights are aliased in that channel. The
previous-action history is added by `terra-baselines`; its standard length in
the current recipes is five. The normal policy samples unmasked logits.
`info["action_mask"]` remains an all-zero informational placeholder rather
than a usable validity mask.

The policy does not receive elapsed or remaining episode time in the audited
recipes. `last_dig_mask` is also not directly observed, although it affects
future digging and dumping. Recurrence or finite action history does not by
itself establish full observability. Carry-work, reset-context, distance and
admissibility flags vary by experiment and must be reported with the model.

Standard resets sample a valid base location and heading, with empty load,
zero relative cabin/steering state and reset decision count. Spawn filtering
rejects static obstacles, nonzero action terrain and forbidden footprint
locations. This is rejection sampling of initial geometry, not a proof of
episode solvability. Fixed evaluations instead supply reproducible initial
agent states.

Partial-reset tiers `1,2,3` denote generated `90%,75%,50%` completion;
tier 0 selects a full reset. Partial action maps preserve the original target
and source identity, are material-conserving, and can include staged spoil.
The reset stores initial excavation fraction `q_reset` and material work
`H_reset` for reward accounting. The historical relay experiment used a
96-source `fnd-slab-apron-d16` partial bank, not partial states spanning all V8
categories. Current specialist and easy-foundation recipes use full starts.

`TerraEnv.step()` returns a reset state/observation immediately after episode
end, while `done`, `task_done` and transition diagnostics describe the ended
transition. Replays requiring the terminal map must use `step_no_reset()` or
preserve that map before resetting. The audited current and August trainers
assert `env_steps == 0` at initial reset; they do not shorten initial episodes
by randomizing elapsed steps. Remaining time is still absent from the policy
observation. See the [training protocol](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/TRAINING_PROTOCOL.md)
for action-history and recurrent-state reset behavior.

Source: `env.py:130–163,423–614,616–731`; `agent.py:218–271,314–445`;
`wrappers.py:90–101,342–394,480–619`; `state.py:130–222,4436–4492`;
[partial-reset contract](https://github.com/leggedrobotics/terra/blob/25f855db3d913fd638c4e56b1740437a2b7122ca/terra/env_generation/PARTIAL_COMPLETION_RESETS.md).

## Reward-v2: the implemented objective

The current v2 specialist/generalist and foundation screen use
`reward_stage=reward_v2`, timing variant 0. The main historical V8 relay and
recurrent lines also use this reward; older V8 dense controls are distinct
experiments.

Let cell index be `i`, target depth requirement be
`v_i = max(-target_i, 0)`, and `V = Σ_i v_i > 0`. For action-map value `h_i`,
define completed excavation and positive soil as

```text
e_i = min(max(-h_i, 0), v_i)
z_i = max(h_i, 0).
```

Let `d_i` be the static obstacle-aware eight-neighbor distance to the accepted
dump mask, expressed in metres and divided by `16 m`. Cardinal moves cost one
cell and diagonal moves cost `sqrt(2)` cells. The field treats obstacles as
blocked cell centers; it does not model vehicle footprint, dynamic holes or
complete motion feasibility. Diagonal steps do not add a corner-cutting test.
Accepted cells have zero distance, and obstacle entries are stored as zero.
All non-obstacle cells must connect to accepted cells, and `max(d_i) ≤ 2.5`.
The bound is an admission check: the generator rejects violations rather than
clipping or rescaling each map.

`C(s)` is total active-carrier relocation credit. A lift transfers the removed
source's distance-weighted work into this ledger; accepted placement clears it.
The remaining material-work quantity and reset-relative progress are

```text
H(s)   = Σ_i (v_i - e_i) d_i + Σ_{i outside A} z_i d_i + C(s)
Q(s)   = (Σ_i e_i)/V - q_reset
P(s)   = (H_reset - H(s))/V
Phi(s) = Q(s) + 1.5 [P(s) + 2.5].
```

At a full untouched reset `q_reset=0` and `H_reset=Σ_i v_i d_i`; at a partial
reset they are the latched initial excavation and material-work values. A lift
does not earn transport progress merely by moving material into the carrier:
the carry ledger preserves its remaining work. Useful staging can reduce `H`;
moving material farther from the accepted region can increase it.

For one physical transition `s → s'`, let `S` indicate exact task success and
`F` indicate episode end without success. The baseline reward is

```text
r(s,a,s') = 6 S - F - 1/450 + 0.9984 Phi(s') - Phi(s).
```

| Constant | Value |
| --- | ---: |
| Success component | `+6` |
| Horizon-failure component | `-1` |
| Explicit step cost | `-1/450` on every transition |
| Excavation weight | `1` |
| Transport weight | `1.5` |
| Potential discount | `0.9984` |
| Shaping weight | `1` |
| Global distance reference | `16 m` |
| Admitted normalized distance bound | `2.5` |

These are the rewards passed to PPO. Reward-v2 replaces the legacy normalized
reward; it is **not divided by 70** and does not add legacy dig, collision,
trench, dump or graded-timeout rewards. Shaping is evaluated on the physical
terminal state too; the implementation does not set terminal `Phi` to zero.
Consequently, the total terminal-transition reward is not exactly `+6` or `-1`.
No policy-invariance claim follows from the use of a potential alone.

At an unchanged full-reset state `Phi=3.75`, timing 0 produces approximately
`-0.008222` per decision: `-1/450 - (1-0.9984)×3.75`. The potential's additive
constant therefore contributes discount-dependent time pressure. A selectable
timing variant 1 instead uses `Phi(s')-Phi(s)` and explicit step cost `-3.6/450`.
That is a separate treatment, not the setting of the current specialist or
foundation screen.

Runtime guards require a nonempty target, one tracked excavator, horizon 450,
finite constants/distances and bounded progress. A violation produces NaN for
the trainer's finite-value guard to reject; it is not a new task-termination
category. R2 distance generation uses the protocol identifier
`obstacle_geodesic_8_physical_global_v1`.

Source: `config.py:19–56`; `state.py:4064–4137,4763–4982`;
`env_generation/distance.py:44–131`.

### Optional foundation behavior costs

The September foundation screen adds

```text
r_behavior = -c_side (ΔV_fresh / V) sin²(theta_cabin)
             -c_travel × executed_base_distance_m
             -c_turn × abs(wrapped_executed_base_rotation_rad).
```

`theta_cabin` is relative to the chassis before excavation. `ΔV_fresh` includes
only newly removed required target material, so dumping and positive-soil
pickup do not incur the lateral cost. The cost is zero fore/aft and maximal
sideways. It expresses a geometric preference, not calculated tipping risk.
Travel/turn terms use actual state change, so rejected motion has zero added
travel/turn cost. There is no additional cabin-swing cost in this screen.

| Recipe | `c_side` | `c_travel` per m | `c_turn` per rad |
| --- | ---: | ---: | ---: |
| A/B controls | 0 | 0 | 0 |
| C: lateral only | 0.25 | 0 | 0 |
| D: motion only | 0 | 0.005 | 0.02 |
| E: combined | 0.25 | 0.005 | 0.02 |
| F: doubled combined | 0.50 | 0.010 | 0.04 |
| September 8 E ×4 | 1 | 0.02 | 0.08 |
| September 8 E ×8 | 2 | 0.04 | 0.16 |
| September 8 E ×16 | 4 | 0.08 | 0.32 |
| September 8 E ×32 | 8 | 0.16 | 0.64 |

These coefficients are distinct experimental settings. None is established
here as the preferred final method. Raw Terra motion costs are learning
surrogates; deployment retains productive work poses and lets the navigation
stack plan motion between them.

Source: `state.py:4985–5036`;
resolved foundation recipes (local: `.artifacts/terra_foundation_sweep_20260907/parent_recipe_comparison.json`),
original screen (local: `.artifacts/terra_foundation_sweep_20260907/RUNS.md`) and
upper-cost screen (local: `.artifacts/terra_foundation_sweep_20260908_upper/PLAN.md`).

### Legacy dense and terminal-only rewards

The `Rewards.dense()` named tuple remains the native default, but does not
describe reward-v2. In corrected V8 dense mode, the sum of action, existence,
optional trench and terminal rewards is divided by 70. Its base constants are
existence `-0.25`, move `-0.1`, collision move `-0.2`, base turn `-0.05`,
collision turn `-0.05`, cabin turn `-0.02`, invalid dig `-0.12`, invalid dump
`-1`, configured dump weight `1`, and terminal base `200`.

Fresh excavation has a hard-coded `+1` event reward, irrespective of volume;
the configured `dig_correct=0.6` does not supply that event reward. A successful
world dump earns signed carrier-plus-off-zone relocation progress times
`relocation_progress_mult` (default `1.5`), times
`clip(170 / max(number_of_target_cells,1), 2,5)/2`, times the configured dump
weight. Accepted-soil relifting incurs `-1.2` times accepted volume removed,
times that dump weight. Exact solo completion's terminal component is
`200 × 1.2 × 2 / 70 ≈ 6.857`; a timeout's terminal component is
`200 × 0.1 × 2 / 70 × absolute_completion²`.

The separate terminal-only mode gives zero reward before termination, `-1`
for failure, and an exact-success reward with workspace- and step-efficiency
bonuses. With the dense constants its success expression is
`(400/70) [1 + 0.15 E_workspace + 0.05 E_step]`, where
`E_step=clip(1-t/450,0,1)` and `E_workspace` compares productive loading cycles
against `max(1,ceil(V/52))`. Annealed-objective mode mixes dense and terminal-only
rewards. Legacy `Rewards.sparse()` is yet another preset and still retains
action costs and the hard-coded fresh-dig event; it must not be called a
strictly sparse terminal objective.

Source: `config.py:114–194`; `state.py:3438–3517,3756–4060,4724–4754`.
The [historical reward audit](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/research/V8_REWARD_TERMINATION_AUDIT.md)
explains why these reward generations are not interchangeable.

## Exact success and termination

For admitted excavation-and-dump maps, exact success requires every target cell
fully excavated, all displaced material in the accepted mask, no off-zone
positive soil, and an empty carrier. The implemented scalar exposes the exact
prerequisites:

```text
dig_completion = sum(completed required depth) / V
dump_purity = accepted_positive_volume / total_positive_volume
dump_volume_completion = clip(accepted_positive_volume / V, 0, 1)
unloaded = all active carrier loads equal zero
integrity = no declared positive-target cell is an obstacle
absolute_completion = min(task_present, unloaded, integrity,
                          optional_edge_requirement, dig_completion,
                          dump_purity, dump_volume_completion)
task_done = absolute_completion >= 1 - 1e-6
done = task_done OR (env_steps >= max_steps_in_episode).
```

When a task declares a dump region but no positive soil has yet been placed,
dump purity is zero. Empty tasks do not succeed. Foundation-edge completion is
an additional requirement only when its alignment treatment is enabled; it is
not enabled in the protocols summarized here. The code also has compatibility
branches for excavation-only and relocation-only tasks; the R2 experiments
documented here require nonempty excavation-and-dump maps.

The horizon is 450 decision steps. Exact success at decision 450 is counted as
success and receives no horizon-failure penalty. There are no separate
collision, invalid-dig, no-effect, stall-age or low-progress terminations in
this contract; those events can lead to failure by exhausting the horizon.
Runtime/data-integrity failures are errors, not successful or timed-out
episodes.

`absolute_completion` is a strict conjunction expressed numerically. Because
the unloaded component is binary, one unit remaining in the carrier makes
it zero even after nearly all work is complete. It must not be used as a
smooth material-progress score. Report exact success together with excavated,
accepted, off-zone and loaded fractions and the termination reason. At the PPO
boundary both success and horizon failure zero-bootstrap under the fixed
within-450 objective; a 32-step rollout-buffer boundary is not an episode end.

Source: `state.py:4146–4314`; bootstrap and optimizer details are in the
[training protocol](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/TRAINING_PROTOCOL.md).

## Source index and publication boundaries

The source index identifies public files and local experiment snapshots.
An unpublished revision requires the corresponding saved local repository; for
example, `git show fa8d5d13:terra/state.py` only works where that commit exists.
Do not substitute a different runtime. No environment or training code was
changed for this audit. Public links for unchanged helpers use byte-identical
earlier revisions; the recorded runtime remains `fa8d5d13`.

| Subject | Audited source |
| --- | --- |
| Configuration, constants and units | config.py (local: `.worktrees/terra_foundation_sweep_20260907/terra/terra/config.py`) at `fa8d5d13` |
| Actions | [actions.py](https://github.com/leggedrobotics/terra/blob/25f855db3d913fd638c4e56b1740437a2b7122ca/terra/actions.py) at `fa8d5d13` |
| Mechanics, rewards and exact success | state.py (local: `.worktrees/terra_foundation_sweep_20260907/terra/terra/state.py`) at `fa8d5d13` |
| Observation and auto-reset interface | env.py (local: `.worktrees/terra_foundation_sweep_20260907/terra/terra/env.py`), wrappers.py (local: `.worktrees/terra_foundation_sweep_20260907/terra/terra/wrappers.py`) at `fa8d5d13` |
| Spawn and map-derived constraints | [agent.py](https://github.com/leggedrobotics/terra/blob/25f855db3d913fd638c4e56b1740437a2b7122ca/terra/agent.py), [map.py](https://github.com/leggedrobotics/terra/blob/171cf116f09160299140cebd012eda6f323c4f5d/terra/map.py) at `fa8d5d13` |
| R2 geodesic field | [distance.py](https://github.com/leggedrobotics/terra/blob/25f855db3d913fd638c4e56b1740437a2b7122ca/terra/env_generation/distance.py) at `fa8d5d13` |
| Historical V8 runtime | [source tree](https://github.com/leggedrobotics/terra/tree/25f855db3d913fd638c4e56b1740437a2b7122ca/terra) at `25f855db` |
| Current v2 restart runtime | source tree (local: `.worktrees/terra_training_restart_20260907/terra/terra`) at `46b140f8` |

A publication should name the task bank and split, source pair, resolved
geometry/actions, observation treatment, reset treatment, reward stage/timing,
termination horizon and policy/evaluation mode. Frozen historical results can
be reported under their original environment; comparisons across geometry,
reset, observation or reward changes need explicit labels. Grid-level
completion is evidence about this planner abstraction. It does not establish
continuous collision clearance, executed ROS plans, measured retained payload
or hardware excavation performance.
