# Fresh-trench gate: junction admission (all-or-nothing veto vs per-cell), 2026-09-03

## Finding

At update 10,000 the v2 generalist (Euler job 12508156, gate on, v2 yaw-parallel
and on-the-line semantics, bound 2.0 m) completed 226/384 foundation and 53/64
straight-trench panel episodes but 0/160 junction episodes (tee, seg, net3,
net4) on the gate_main/development panel. A rollout probe over the 224 trench
slots (v2 clause records per step, scratchpad `gen_u10000_junction_probe/`)
showed:

- The gate almost never refuses (61 refusals in 224 episodes). It deters: the
  policy presses DO when the exported valid bit is 0 at ~0% of steps.
- On junction maps the machine reaches a dig-opportunity pose (empty, fresh
  cell in cone) as often as on straights (26-34% of steps), but the gate marks
  71 / 86 / 83 / 97% of those poses invalid (tee / seg / net3 / net4;
  straights 62%).
- The junction all-or-nothing clause is the junction-specific excess: 41 / 45 /
  43 / 22% of invalid poses, 0% on straights by construction. Between two
  oblique segments (seg2) it is total: the +-30 deg cone straddles both
  sections and no yaw is parallel to both (slot 389: 0/68 cells dug, 437/438
  opportunities vetoed).
- Three classes among the 160 junction episodes: A (19) parked the whole
  horizon against the junction veto; B (36) parked against the yaw clause;
  C (105) rarely gate-blocked but deadlocked by traversability after digging
  one branch (dug cells are impassable, 7x11 chassis).
- Illegal spoil (8-15 units/episode) lands on neutral ground, not on trench
  cells; secondary.

## Scratch A/B on the same checkpoint (no EnvConfig field, not poolable)

Per-cell admission (dig the pose-valid fresh cells in the cone, leave the
others, valid bit on when at least one is admissible):

| | straight | tee | seg | net3 | net4 | class A |
|---|---|---|---|---|---|---|
| dig fraction, veto | 0.98 | 0.45 | 0.39 | 0.35 | 0.31 | 0.24 |
| dig fraction, per-cell | 0.98 | 0.61 | 0.55 | 0.52 | 0.43 | 0.57 |
| completions, veto -> per-cell | 53 -> 53 | 0 -> 1 | 0 -> 1 | 0 -> 0 | 0 -> 0 | 0 -> 0 |

Junction gate-invalid poses 18,101 -> 9,693; all-or-nothing share 36% -> 0%;
yaw share 54% -> 95%; episodes at dig fraction >= 0.8: 0 -> 14 of 160.
Straight control: 0/64 episodes changed (the clause is inert on single-section
maps). Residual for this policy: yaw deterrence, RL competence (DO pressed at
16-35% of admitted junction poses vs 80% on straights), traversability
deadlock.

## Implementation

2026-09-03, Terra main: the all-or-nothing junction veto is REMOVED (user
decision after the geometric check below). `State._get_fresh_trench_dig_alignment_details`
now always admits per cell: the admitted dig mask is
`dig_mask & (not fresh_trench_target | fresh_cell_pose_valid)` and the
exported `fresh_trench_dig_alignment_valid` bit is 1 when at least one fresh
cell in the cone is admissible. Yaw and on-the-line clauses are unchanged.
There is no configuration switch; the pilot (v1) and the veto-era v2 runs
are replayable only from their own Terra revisions (veto era ends at
c703c4eb). Tests: `test_intersection_digs_the_aligned_cell_and_leaves_the_perpendicular_one`,
`test_v2_junction_dig_from_an_on_axis_pose_admits_only_the_aligned_cells`.
terra-baselines: no field; the Euler launcher arms `genpc` / `specpc` and
the run contract's `gate_semantics=v2_yaw_parallel_on_the_line_per_cell_admission`
label the epoch, and the Terra revision pin carries the semantics.

2026-09-07: `local_map_admissible_dig` now uses the same per-cell admission
helper as the DO mask. Its former whole-cone veto incorrectly reported zero
when an aligned section and an unaligned branch shared a cone. Each heading
now counts the fresh cells admitted by any owning section. A shared junction
cell remains diggable from either aligned approach; cells owned only by an
unaligned section remain untouched. The focused observation regressions check
all 12 cabin headings against the prospective gate and actual DO volume from
both section approaches, plus rejection when neither section accepts the yaw.
This corrects the observation without changing the existing dynamics, yaw or
offset tolerances, or rewards.

## Status

Adopted. Veto-era runs cancelled on 2026-09-03 (Euler generalist 12508156
at u14,000, CSCS specialist 4586880 at u18,000); both arms relaunched under
per-cell admission.

## Geometric check (2026-09-03, `tools/check_trench_axis_sweep_feasibility.py`
## on the gate_main/development panel, 176 maps, same code and configuration
## for both rules, on-the-line 2.0 m, transcription self-checked against
## Terra's gate: 0 mismatches over 371 applicable probes)

On-axis lane, section C: stand on the section line, chassis parallel, cabin
straight ahead or behind, move forward/backward only, one axis at a time
(blocked = padding, the monotone-retreat model). Maps complete at tolerances
0.5 / 1.0 / 2.0 tiles:

| family | all-or-nothing veto | per-cell admission |
|---|---|---|
| straight (64) | 64 / 64 / 64 | 64 / 64 / 64 |
| tee (32) | 2 / 4 / 5 | 32 / 32 / 32 |
| segmented (32) | 0 / 2 / 5 | 32 / 32 / 32 |
| net3 road (16) | 0 / 0 / 0 | 16 / 16 / 16 |

Sections complete under the veto: tee 2-13/64, segmented 0-13/80, net3
0/48; under per-cell every section. So under the removed veto a junction
cannot be dug by driving along one of its axes and digging ahead: the other
branch's cells inside the cone veto the whole action. That contradicts the
intended semantics (a junction is diggable from any of its axes; the policy
chooses the order). Per-cell admission restores it without changing any
straight map. Receipts: session scratchpad `junction_geometry/`
(`axis_sweep_veto.json`, `axis_sweep_percell.json`, patched tool copy
`axis_sweep_percell.py`, env `TRENCH_PER_CELL=1`).
