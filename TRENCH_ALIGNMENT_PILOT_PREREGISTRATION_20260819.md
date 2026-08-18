# Goal: C0/T1 fresh-trench dig-alignment pilot (2026-08-18)

Decide causally whether the Terra fresh-trench dig-admissibility gate
(chassis-yaw alignment + 3.5–7.0 m standoff, all-or-nothing macro no-op)
improves trench execution quality without a material completion regression.

## Arms (matched seed, fresh init)

| Arm | Finite metadata | 3 alignment obs in policy input | Gate | Legacy trench reward |
|---|---|---|---|---|
| C0 control | required | yes | off | off |
| T1 treatment | required | yes | on | off |

Only difference: Terra `enforce_trench_dig_alignment` (frozen defaults:
yaw tol 0.2619 rad, standoff 3.5–7.0 m).

## Recipe (both arms, from current fresh-training lineage)

- Terra: branch `experiment/trench-fresh-dig-alignment-20260818`
  (base 25f855db + gate/metadata/obs implementation), committed revision TBD.
- terra-baselines: branch `experiment/trench-pose-alignment-20260818`
  (+ obs wiring), committed revision TBD.
- Configs: `trench_align_c0_v1` / `trench_align_t1_v1` YAML presets —
  pure-trench training (f0 trench lineage: DENSE rewards, curriculum 20/80,
  max_steps 450, `apply_trench_rewards: false`), differing ONLY in
  `enforce_trench_dig_alignment`. Rationale: the accepted-bank machinery has
  no trench-only scope (`--accepted-bank-scope` = full/47-condition only), and
  a mixed run would dilute the mechanism signal; minimal causal experiment
  wins.
- Model: v6.1 spatial encoder (resnet_spatial_8x8_se_sa_xattn, fused width
  704 — required by the alignment-embedding injection), feed-forward actor
  (NO GRU, no stall-age, no partial resets, no reset-context, no action mask),
  carry-work obs on, + the width-3 trench-alignment vector via zero-init
  (3,704) actor/critic embeddings (both arms).
- Bank: `/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819`
  — deterministic finite-metadata enrichment of the shipped V8 R2 release
  (`terra_v8_v6_constraints_v7_adjacent_train96_v5`, R2 geodesic
  materialization). 2,400 trench maps enriched from P5 provenance (endpoint
  residual ≤ 4.4e-14 tiles); the 7 unprovenanced `v7-trn-*` conditions kept
  but fail-closed. Reward stage `reward_v2` (timing variant 0, protocol
  `obstacle_geodesic_8_physical_global_v1`) because the legacy path rejects
  R2 datasets. Train scope = pooled 12 conditions × 96 = 1,152 maps.
- Preregistered scope exclusion: preflight found 61/2,400 maps without a
  complete strict-gate cover — ALL 4-axis net4 junctions, tolerance-
  independent (12-heading yaw quantization; identical at 15/20/25°). This is
  the note's "more than N sections must fail preflight, not truncate" case.
  The 3 net4 conditions (`trn-net4-side2{,-s}`, `trn-net4-side1-road`) are
  excluded from BOTH training and the pilot's primary endpoint panel, and
  reported separately. net3/road/T/seg junctions remain (100% cover:
  2,339/2,339 in-scope maps, all cells).
- Same bank archive, single YAML anchor consumed by both arms.
- Fail-closed: `require_trench_alignment_metadata` runs the canonical-loader
  finite-segment contract + Terra's array-level validator at startup for BOTH
  arms (so C0 can never train on a bank T1 cannot use).
- Preflight: canonical loader with finite-segment requirement over every
  trench map + all-map strict-gate feasibility cover. Any failure blocks
  launch (preflight stop from the research note).
- 4× RTX 4090, 4 devices × 512 envs × 32 steps, 32 minibatches, 2 epochs,
  65,536 transitions/update; lr 3e-4, vf 2.0, ent 0.15→0.02/20k;
  seed 20260818 both arms; target update 100,000 (beyond one allocation,
  wall-time exit with checkpoint = CONTINUABLE); checkpoints every 500;
  partition gpuhe.120h, account es_hutter, 119:45.

## Preregistered decision points (from research note)

- Pilot early checkpoint: u10,000. Mechanism check: T1 invalid fresh-DO
  attempt fraction must fall ≥50% from its first evaluation.
- Pilot stop: T1 exact completion >5 pp below C0 at two successive scheduled
  evaluations AND invalid-DO fraction not halved → stop; next step is the
  broad-to-strict tolerance curriculum, NOT a reward term.
- Promotion (later, ≥3 matched seeds): seed-stratified paired bootstrap on the
  frozen full-start panel; 95% LCB(T1−C0 exact) > −2 pp AND
  95% LCB(raw ROS physical acceptance) > 0.
- Primary endpoint: strict exact completion on untouched frozen full-start
  panel. Mechanism: invalid fresh-DO rate, raw fresh-dig yaw/standoff,
  completion by family/section.

## Non-interference

Running jobs 10777230 (relay u100), 10777232 (pending u200), 10991006
(gru64r) are untouched; this pilot shares no run dirs, W&B ids, or banks
in place (enriched bank is a new copy).
