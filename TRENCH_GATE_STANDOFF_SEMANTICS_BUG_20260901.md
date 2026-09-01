# The fresh-trench gate applied its working-distance band sideways

Date: 2026-09-01. Branch `epoch/terra-footprint-and-soil-containment-20260831`.
Status: design error confirmed in code, v2 semantics implemented and unit-tested,
completed pilot re-scored under v2. Re-validation of feasibility under v2 and the
oracle solvability run are recorded in §6 as they land.

## 1. The error

`State._get_fresh_trench_dig_alignment_details` (terra/state.py) computed

    standoffs_m = |A*col + B*row + C| / ||(A,B)|| * tile_size

the PERPENDICULAR distance from the base centre to the section axis, and
required it in [3.5, 7.0] m together with chassis yaw parallel to the axis
(<= 0.2619 rad). Jointly those two clauses force a lateral offset lane and
refuse the on-axis pose outright: on the trench line the perpendicular distance
is ~0 < 3.5.

But the machine's working distance is already enforced RADIALLY by the dig
cone: `_dig_cone_radius_bounds()` gives r_min = 0.5 + 0.5714*5.5 = 3.643 m and
r_max = 6.500 m, machine -> cell, within +-30 deg of the cabin heading. A
machine standing on the trench line, aligned, digging the cells ahead of it at
3.64-6.50 m radial distance and retreating backward satisfies every physical
requirement and was refused for a reason no physics supports. 3.5 m ~= 3.643 m
is the tell: the "don't dig at your feet" idea was re-imposed machine -> line
instead of machine -> cell.

Found by Lorenzo playing the gate manually (see
`TRENCH_STANDOFF_FLOOR_HUMAN_TRIAL_20260901.md`): 457 actions, 27 DO presses,
0 of 185 cells dug, every refusal at yaw 0.0 deg with fresh cells in the cone,
refused solely on the floor. His objection, verbatim: *"the 3.5 to 7 is along
the chassis major axis, not lateral distance -- otherwise how are we supposed to
dig the trench if we are not on top of the trench?"*

## 2. Provenance: a spec-level error, not a coding slip

The implementation matches its specification. The research note
(`TRENCH_FRESH_DIG_ALIGNMENT_RESEARCH_NOTE_20260818.md`, unchanged since it was
written) says, line 121: *"A section is pose-valid when chassis yaw is parallel
to its axis within the tolerance and base-center perpendicular standoff lies in
the configured band"*, and motivates it at line 70 by rejecting the old reward's
centreline attraction because *"the chassis needs a safe parallel offset lane,
not attraction to the excavation centerline"*. That reasoning conflated
lane-keeping with working distance. Every later check -- the 64-map feasibility
witness, the 2,400-map preflight, the gate over-restriction audit, both pilot
readouts -- tested the gate against ITS SPEC and passed. None tested the spec
against physical intent. It took a human driving the machine to catch it.

Minimal reproduction (now `test_v1_refuses_the_on_axis_dig_ahead_pose_that_v2_admits`):
straight trench on row 24, machine at (24, 32), base heading 0, cabin 0, empty.
Terra's own cone selects five fresh cells at cols 39-43, i.e. 4.00-6.29 m
radial. v1: yaw 0.0 deg, standoff error -1.0 (saturated too close), REFUSED,
nothing dug. v2: same pose, ADMITTED, five cells removed, loaded = 5.

## 3. v2 semantics (implemented)

`EnvConfig.trench_dig_standoff_enforced: bool = False` (appended last; positional
checkpoint compatibility preserved; `frozen_benchmark_protocol` pops it so the
frozen SHA is unchanged).

- v2 (default): a section is pose-valid when chassis yaw is parallel within
  `trench_dig_yaw_tolerance_rad`. That is the whole positional clause. Working
  distance is the dig cone's job; cell scoping is the untouched
  membership / junction all-or-nothing logic.
- v1 (`=True`): additionally requires perpendicular standoff in
  [`trench_dig_standoff_min_m`, `trench_dig_standoff_max_m`]. Kept only so the
  C0/T1 pilot replays.

Observation semantics change: `fresh_trench_dig_standoff_error` was the
band-relative error (exactly 0 anywhere in band -- no gradient). Under v2 it is
the SIGNED perpendicular offset / r_max (6.50 m), clipped to [-1, 1]: 0.0 on the
line, |1| when the axis is at or beyond the far edge of reach. The sign follows
each section's line-equation convention (constant within an episode), so it
says "which side of this section am I on", not "left/right of the machine". The
diagnostic-axis tie-break is unchanged in form; under v2 it reports the nearest
equally-aligned section.

Alternative recorded, NOT implemented: a small max-offset bound (e.g. <= 2 m,
"be near the line"). Not chosen because yaw + cone + membership already scope
digs, a machine > 7 m out cannot reach anyway, and any bound re-introduces an
unshaped positional cliff. It is the fallback if v2 admits stations the ROS
stack later rejects.

## 4. The completed pilot re-scored under v2

`tools/rescore_trench_pilot_v2_admissibility.py` over the frozen u85,000 probe
traces (no new rollouts). Exact where `fresh_axis_count == 1`; bracketed where
a v1-refused step had >= 2 owning sections (the junction clause needs per-cell
ownership the traces lack): pessimistic = all ambiguous inadmissible,
optimistic = all admissible.

| arm | executed digs | v1-admissible | v2-admissible (exact) | v2 ambiguous | v2-inadmissible (exact) |
|---|---:|---:|---:|---:|---:|
| C0 | 1,142 | 159 (13.9%) | 376 (32.9%) | 287 (25.1%) | 479 (41.9%) |
| T1 | 713 | 713 | 713 | 0 | 0 |

| admissible exact completion, 176 slots | raw | v1 | v2 pessimistic | v2 optimistic |
|---|---:|---:|---:|---:|
| C0 | 167 (94.89%) | 2 (1.14%) | 10 (5.68%) | 19 (10.80%) |
| T1 | 110 (62.50%) | 110 (62.50%) | 110 (62.50%) | 110 (62.50%) |

The 217 C0 digs that v2 rescues are exactly the on-line aligned pattern v1
wrongly refused: median standoff 0.59 m, 155 of them under 1 m, yaw 0.0 deg
(max 0.03 deg). C0 does dig that way sometimes -- but 41.9% of its digs are
yaw-misaligned outright, which no standoff semantics forgives.

**The pilot's qualitative verdict survives the semantics fix.** The gate arm's
completions are admissible; the control's are mostly not. The margin moves
from +61.4 pp (v1) to +51.7 .. +56.8 pp (v2 bracket).

Honesty constraint: T1 was TRAINED under v1, so its behaviour is shaped by the
wrong band. Re-scoring measures both arms under a different ruler; it does not
say what a v2-trained policy would do. Any clean v2 claim needs a new matched
pair trained under v2.

## 5. What this changes in the earlier documents

- `TRENCH_STANDOFF_FLOOR_HUMAN_TRIAL_20260901.md`: the positional trap
  (22.8% of T1's tail below the 3.5 m floor) disappears under v2 -- there is
  no floor. Its other findings stand and become more important: the junction
  veto accounts for 69.1% of T1's refused attempts (all at exactly 60 deg yaw to
  the blocking section); dig-and-dump is a PAIR problem (a legal dig can
  deadlock the machine loaded with no legal dump); section selection is hard
  for humans too.
- `TRENCH_ALIGNMENT_PILOT_U85000_READOUT_20260825.md` §(b): the
  "standoff never binds on admitted digs" observation was correct but measured
  the wrong side; the floor was the only side that bound on REFUSALS. Now moot.
- The research note itself is deliberately left unchanged as the record of
  what was specified and trained against.

**Replay trap, closed.** `eval_fixed_bank.py` rebuilds the eval env from the
checkpoint's `train_config` (`load_env_from_checkpoint = False`), not from its
stored `env_config`. The pilot checkpoints predate the selector, so on this
branch they would have been evaluated under v2 silently -- a different ruler
from the one they were trained under. The baselines side now threads
`trench_dig_standoff_enforced` through preset -> `MixedAgentTrainConfig` ->
`create_mixed_agent_env_config`, prints the effective semantics at startup,
records it in the treatment fingerprint (only when the field is present, so
pre-v2 fingerprints stay byte-identical), and `eval_fixed_bank.py --gate-v1`
forces v1 with a loud warning whenever a gate-on checkpoint lacks the field.
The two pilot presets are pinned `trench_dig_standoff_enforced: true`.

## 6. Re-validation under v2 (filled as results land)

PENDING: over-restriction audit, order-independent station cover, axis sweep
(on-axis lane completability per family), full 15-condition preflight incl. net4,
scripted oracle with persistent-pose navigation and dig+dump pairing.

## 7. Recommendation

1. New matched training pair under v2 semantics (this is the only way to make a
   clean v2 claim). Both arms carry the v2 observation semantics.
2. Re-admit net4 to the bank if the v2 preflight shows complete covers.
3. ROS physical acceptance remains the ground-truth arbiter of which stations
   are executable and is STILL unmeasured. v2 widens what the simulator admits;
   it does not prove the real stack accepts on-axis stations. Measure it before
   any promotion claim.
4. Open design question for Lorenzo, not acted on: the junction all-or-nothing
   veto. Approaching a T head-on along axis 0, the +-30 deg cone sweeps
   exclusive cells of the perpendicular branch up to ~3 m laterally at 3.6-6.5 m
   ahead, so the last stretch before every junction can only be dug from the
   cross-branch's own lane. That is the mechanism behind 69% of T1's lost digs.
   Whether digging a few cross-branch cells while aligned to the main axis is
   physically bad is a real-machine question; the veto is the spec's intent and
   the audit found coverage survives it, so it stays until ROS says otherwise.
