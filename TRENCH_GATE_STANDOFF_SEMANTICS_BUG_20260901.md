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

### 3.1 The on-the-line clause (added 2026-09-02 after manual play)

Yaw-parallel alone was not enough. Playing v2, Lorenzo could still dig from the
old sideways lane: chassis parallel (yaw 0.0 deg) but 3.84 m and 6.52 m to the
side of the line with the cabin swung 60-120 deg to reach in (session
`manual_trench_slot455_20260902_091337.jsonl`, seq 203 and 221). Dropping the
band had removed the *prohibition* of the on-axis pose without adding the
*requirement* for it. His intent is "on top of the trench", which is a MAXIMUM
perpendicular offset, not a minimum.

v2 pose validity is now: chassis yaw parallel within tolerance AND perpendicular
offset of the base centre to the axis `<= EnvConfig.trench_dig_max_offset_m`
(appended last; `<= 0` disables the clause; ignored under v1). The exported
standoff observation is unchanged (signed offset / cone reach). On the integer
lattice the admitted offsets are 0, 0.57, 1.14, 1.71 m (0-3 cells) at 2.0 m;
2.29 m (4 cells) and beyond are refused. The default 2.0 m is provisional: a
coverage sweep over {1.14, 1.71, 2.29, 2.86, 3.43, disabled} m picks the
smallest bound that loses no coverage (oblique 30-deg lanes drift up to
~1.5 m over a traverse, so the bound cannot be arbitrarily tight).

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

## 6. Re-validation under v2

All numbers v1 vs v2 on identical code; receipts in
`tools/trench_align_v2_revalidation_20260901/` (tables in its README) and
`tools/trench_align_oracle_receipts_20260831/`.

**The gate never bound coverage under either semantics.** Every one of the
13,428 panel cells and 86,933 pooled-training cells is admissibly diggable under
v1 and v2; `cells_reachable_but_never_admissible = 0`. What v2 changes is the
size of the admissible set: 2.4x the applicable candidates, 2.2x the admissible
stations (204,835 -> 459,383 on the panel). Persistent station cover 176/176
under both. Axis sweep with 12 cabins 176/176 under both.

**The on-axis lane, quantified.** Dig-ahead-and-retreat alone (perpendicular
~0, cabin ahead/behind, forward/backward only, dumping removed):

| family | v1 cells | v2 cells | v2 maps complete (tol 1.14 m) |
|---|---:|---:|---:|
| straight | 0 | 4,408/4,408 (100%) | 64/64 |
| tee | 0 | 2,417/2,465 (98.1%) | 5/32 |
| network | 0 | 2,121/2,193 (96.7%) | 6/32 |
| road | 0 | 1,353/1,394 (97.1%) | 0/16 |
| segmented | 0 | 2,754/2,968 (92.8%) | 5/32 |

Under v1 the on-axis lane digs exactly zero cells on every row -- that is the
design error in one number. Under v2 straight trenches are fully completable by
the retreat pattern alone; the multi-section residual is the junction
all-or-nothing veto (straight is the single-section control), not standoff.
Note the on-axis model uses a padding-only blocked set: a machine on the line
stands in its own hole under the order-independent set, but retreat is
provably safe because the cone starts at 3.64 m while the chassis reaches only
3.14 m ahead.

**net4 verdict: re-admitted.** Full 2,400-map preflight, net4 included: maps
without a complete fresh cover old-v1 61, branch-v1 89, **v2 0**
(`preflight_passed = True`; net4 conditions 160/160/160). The matched v1 on the
same branch is worse than the original, so the gain is the semantics, not the
footprint fix. The generalist training pool is therefore the full 40-condition
set (25 foundation + 15 trench, 3,840 maps); v7-trn stays out (no finite
provenance).

**Scripted oracle** (persistent-pose navigation, airtight dig+dump pairs,
49,551 per-step checks against Terra's exported alignment scalars with zero
divergence): v2 134/176 within horizon 450 (median 80.5 steps, max 183 -- no
success needed the extended horizon), v1 132/176 with the identical fixed
controller, net4 15/48 under v2. The old 147/176 was inflated: it counted
completions reachable only through poses the controller's own excavation
deletes. Holding the controller fixed, v2 beats v1 by +2 slots and adds 382
on-axis stations (37.3% of 1,025) that v1 forbids outright. Residual stalls are
pose-graph connectivity (476 of 489 remaining cells at `pose_reachable`), i.e.
the oracle's conservative navigation, not any gate clause; the static covers
above are the ceiling evidence. One loaded-no-legal-dump deadlock survives in
176 (slot 455).

**Spoil leak root-caused.** Every surviving illegal-spoil event in all three
oracle runs has `loaded_before == 0`: the dig relaxes TWICE (`_apply_dig_mask`,
then `_handle_dig`), and only the second pass had been contained. Both passes
now carry the accepted-dump-zone containment. Confirming v2 oracle run on the
same 176 slots: **146/176 within horizon (83.0%)**, median 78.5 steps, p90
130, **illegal spoil 0 units / 0 slots, loaded-no-legal-dump deadlocks 0**
(the slot-455 deadlock was the uncontained relaxation consuming the
planner's dump cells), 366 of 988 stations on-axis. Receipt:
`tools/trench_align_oracle_receipts_20260831/oracle_176slot_v2_contained_dig.json`.

**Two tool corrections worth knowing.** The tools' two footprint models had
been inverted relative to Terra after commit 566867db (verified against
`State._is_valid_move` over 4,000 poses; the right model agrees 0.9995), and
the shared module's Terra selfcheck built Terra's state from the config default
while the replica forced v1, producing 8/512 phantom mismatches under
`--gate-v1`. Both fixed; zero mismatches under both flags everywhere.

## 7. Recommendation

1. New matched training pair under v2 semantics (this is the only way to make a
   clean v2 claim). Both arms carry the v2 observation semantics.
2. net4 re-admitted (v2 preflight: 0 of 2,400 maps without a complete cover).
   The v2 generalist trains on the full 40-condition pool, 3,840 maps.
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
