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

`EnvConfig.trench_dig_per_cell_admission` (default False = the all-or-nothing
veto of the pilot and v2 contracts). `State._get_fresh_trench_dig_alignment_details`
computes both validities and returns the admitted dig mask; under per-cell the
mask is `dig_mask & (not fresh_trench_target | fresh_cell_pose_valid)`. Yaw and
on-the-line clauses are unchanged. Test:
`test_per_cell_admission_digs_only_the_aligned_cells_at_a_junction`.
`benchmark_protocol` pops the field like the other gate fields. terra-baselines
threads it as `trench_dig_per_cell_admission` (presets
`trench_align_v2pc_generalist_gen` / `trench_align_v2pc_specialist_spec`, Euler
launcher arms `genpc` / `specpc`, eval fingerprint records it).

## Status

Not adopted by default. Decision pending: restart the two arms under per-cell
admission (new epoch) or continue under the veto. The u10000 checkpoint's
residual junction failures are not addressed by this flag alone.
