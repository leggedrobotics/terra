# Fresh-trench gate v2 revalidation, 2026-09-01

v1 = perpendicular standoff band [3.5, 7.0] m enforced on top of the yaw-parallel clause (`--gate-v1`).  v2 = yaw-parallel only, the shipped default (`EnvConfig.trench_dig_standoff_enforced=False`); working distance is left to the dig cone (3.64-6.50 m radial, +-30 deg).  Every numpy replica in these tools is asserted against Terra's exported `fresh_trench_dig_alignment_valid` under BOTH semantics; mismatch counts are reported below and are all zero.

## Verdicts

1. **net4 becomes fully coverable under v2.**  The full 2,400-map preflight goes from 89 maps without a complete order-independent fresh cover (all net4) to 0, preflight_passed=True.  net4 can re-enter the training bank on this evidence.
2. **The gate was never the binding constraint on coverage in the gate_main panel or the pooled training bank.**  Under both semantics every target cell is admissibly diggable from some pose and `cells_reachable_but_never_admissible` is 0.  What v2 changes is the SIZE of the admissible set: 2.40x the applicable candidates and 2.24x the admissible stations on gate_main/development.
3. **The on-axis lane works, but only for a single section.**  Standing on the trench line, cabin straight ahead or behind, digging and backing up completes 64/64 straight maps at every tolerance tested, and 0 maps under v1.  No multi-section family is completable that way alone: 93-98% of cells, and the residual sits at the junctions, which is the all-or-nothing veto, not the standoff.
4. The retired v1 band stays reproducible with `--gate-v1`, and the numpy replica matches Terra's exported verdict under both semantics on every map checked.

## (a) Over-restriction audit - evaluation/gate_main/development

| metric | v1 | v2 |
|---|---|---|
| maps | 176 | 176 |
| target cells | 13,428 | 13,428 |
| cells admissibly diggable | 13,428 | 13,428 |
| cells admissibly diggable % | 100.00 | 100.00 |
| cells reachable but never admissible | 0 | 0 |
| applicable candidates | 1,110,285 | 2,663,512 |
| vetoed candidates | 538,238 | 1,372,226 |
| veto rate over applicable candidates | 0.4848 | 0.5152 |
| admissible stations | 204,835 | 459,383 |
| admissible stations Terra-legal | 150,760 | 231,978 |
| replica-vs-Terra mismatches | 0 | 0 |

## (a) Over-restriction audit - train_pilot_pooled_12cond

| metric | v1 | v2 |
|---|---|---|
| maps | 1,152 | 1,152 |
| target cells | 86,933 | 86,933 |
| cells admissibly diggable | 86,933 | 86,933 |
| cells admissibly diggable % | 100.00 | 100.00 |
| cells reachable but never admissible | 0 | 0 |
| applicable candidates | 6,945,246 | 16,741,854 |
| vetoed candidates | 3,218,796 | 8,294,508 |
| veto rate over applicable candidates | 0.4635 | 0.4954 |
| admissible stations | 1,305,071 | 2,941,127 |
| admissible stations Terra-legal | 980,141 | 1,502,153 |
| replica-vs-Terra mismatches | 0 | 0 |

## (b) Order-independent persistent station cover - evaluation/gate_main/development

| metric | v1 | v2 |
|---|---|---|
| maps | 176 | 176 |
| target cells | 13,428 | 13,428 |
| cells admissible from any pose | 13,428 | 13,428 |
| cells from a Terra-legal persistent station | 13,428 | 13,428 |
| cells from a legacy-mirror persistent station | 13,408 | 13,424 |
| cells the legacy mirror would have lost | 20 | 4 |
| maps with a complete cover (any pose) | 176 | 176 |
| maps with a complete cover (Terra footprint today) | 176 | 176 |
| maps with a complete cover (legacy mirror) | 169 | 175 |
| Terra-legal admissible stations | 150,760 | 231,978 |
| replica-vs-Terra mismatches | 0 | 0 |

## (c) Axis sweep - evaluation/gate_main/development

All 12 cabin headings, FORWARD/BACKWARD lanes, dumping removed.

| footprint | semantics | maps complete (exact) | cells covered (exact) | min single-lane share |
|---|---|---|---|---|
| terra | v1 | 176/176 | 13,428/13,428 | 0.786 |
| terra | v2 | 176/176 | 13,428/13,428 | 0.800 |
| legacy_mirror | v1 | 167/176 | 13,406/13,428 | 0.444 |
| legacy_mirror | v2 | 175/176 | 13,424/13,428 | 0.481 |

### (c2) The ON-AXIS LANE: perpendicular ~ 0, cabin ahead or behind, forward/backward only

Is every section completable by dig-ahead-and-retreat alone?  Blocked space is `padding` only (`fresh`): that is the correct model for a monotone retreat, because the cone starts 3.64 m from the base centre while the chassis reaches only 3.14 m ahead, so a machine that only ever backs up digs strictly ahead of every pose it will occupy and never stands on a cell it dug.  Footprint: `terra`.

**Under v1 every cell of this table is 0** -- the lane is empty by construction, since perpendicular <= 2 tiles = 1.14 m is below the 3.5 m floor.  The v1 rows, the pessimistic `persistent` blocked model (`padding | all target<0`, under which a machine on the line stands in its own hole) and the `legacy_mirror` footprint are all in summary.json.

| family | tol tiles (m) | maps complete | sections complete (own lane) | cells covered | cell % | worst section share |
|---|---|---|---|---|---|---|
| network | 0.5 (0.29 m) | 0/32 | 0/96 | 1,888/2,193 | 86.1% | 0.400 |
| network | 1.0 (0.57 m) | 1/32 | 1/96 | 1,969/2,193 | 89.8% | 0.400 |
| network | 2.0 (1.14 m) | 6/32 | 9/96 | 2,121/2,193 | 96.7% | 0.438 |
| road | 0.5 (0.29 m) | 0/16 | 0/48 | 1,217/1,394 | 87.3% | 0.476 |
| road | 1.0 (0.57 m) | 0/16 | 0/48 | 1,268/1,394 | 91.0% | 0.476 |
| road | 2.0 (1.14 m) | 0/16 | 0/48 | 1,353/1,394 | 97.1% | 0.538 |
| segmented | 0.5 (0.29 m) | 0/32 | 0/80 | 2,448/2,968 | 82.5% | 0.359 |
| segmented | 1.0 (0.57 m) | 2/32 | 2/80 | 2,552/2,968 | 86.0% | 0.467 |
| segmented | 2.0 (1.14 m) | 5/32 | 13/80 | 2,754/2,968 | 92.8% | 0.600 |
| straight | 0.5 (0.29 m) | 64/64 | 64/64 | 4,408/4,408 | 100.0% | 1.000 |
| straight | 1.0 (0.57 m) | 64/64 | 64/64 | 4,408/4,408 | 100.0% | 1.000 |
| straight | 2.0 (1.14 m) | 64/64 | 64/64 | 4,408/4,408 | 100.0% | 1.000 |
| tee | 0.5 (0.29 m) | 2/32 | 2/64 | 2,235/2,465 | 90.7% | 0.552 |
| tee | 1.0 (0.57 m) | 4/32 | 4/64 | 2,299/2,465 | 93.3% | 0.552 |
| tee | 2.0 (1.14 m) | 5/32 | 13/64 | 2,417/2,465 | 98.1% | 0.621 |

v1 on-axis rows with any coverage: 0 of 60 (expected 0).

## (d) Full 15-condition preflight, net4 included

| condition | v1 complete | v2 complete | v1 cells | v2 cells |
|---|---|---|---|---|
| trn-net3-side1-road | 160/160 | 160/160 | 13,861/13,861 | 13,861/13,861 |
| trn-net3-side2 | 160/160 | 160/160 | 13,861/13,861 | 13,861/13,861 |
| trn-net3-side2-s | 160/160 | 160/160 | 8,009/8,009 | 8,009/8,009 |
| trn-net4-side1-road | 145/160 | 160/160 | 22,003/22,051 | 22,051/22,051 |
| trn-net4-side2 | 145/160 | 160/160 | 22,003/22,051 | 22,051/22,051 |
| trn-net4-side2-s | 101/160 | 160/160 | 11,258/11,372 | 11,372/11,372 |
| trn-seg2-side2 | 160/160 | 160/160 | 12,962/12,962 | 12,962/12,962 |
| trn-seg3-side2 | 160/160 | 160/160 | 16,902/16,902 | 16,902/16,902 |
| trn-straight-allfree | 160/160 | 160/160 | 10,902/10,902 | 10,902/10,902 |
| trn-straight-altsides | 160/160 | 160/160 | 10,902/10,902 | 10,902/10,902 |
| trn-straight-side1 | 160/160 | 160/160 | 10,902/10,902 | 10,902/10,902 |
| trn-straight-side1-tight | 160/160 | 160/160 | 10,902/10,902 | 10,902/10,902 |
| trn-straight-side2 | 160/160 | 160/160 | 10,902/10,902 | 10,902/10,902 |
| trn-tee-side2 | 160/160 | 160/160 | 14,851/14,851 | 14,851/14,851 |
| trn-tee-side2-s | 160/160 | 160/160 | 10,021/10,021 | 10,021/10,021 |

- v1: 2400 maps, 89 without a complete fresh cover, preflight_passed=False, 9,554,553 candidate macro actions, 650 s
- v2: 2400 maps, 0 without a complete fresh cover, preflight_passed=True, 18,046,816 candidate macro actions, 1091 s

**net4 verdict: yes.**  Every one of the 480 net4 maps has a complete order-independent fresh cover under v2.  The matched v1 run on the same branch and the same code leaves 89 maps incomplete, all of them net4, so the gain is the semantics, not the branch: the earlier 2026-08-19 v1 receipt had 61 (the difference is terra commit 566867db, which moved the pose graph and made net4-side2-s slightly WORSE under v1, 123/160 -> 101/160).

## v1 reproduction check - review-v4 bank (tools/trench_alignment_feasibility_20260818.json)

| family | v1 complete | v1 cells | v2 complete | v2 cells |
|---|---|---|---|---|
| network | 16/16 | 2,310 | 16/16 | 2,310 |
| road | 16/16 | 2,379 | 16/16 | 2,379 |
| straight | 16/16 | 1,894 | 16/16 | 1,894 |
| tee | 16/16 | 2,392 | 16/16 | 2,392 |

## Correctness checks

### `--gate-v1` reproduction of `tools/trench_alignment_feasibility_20260818.json`

- fresh-cover summary identical: **True** (0 of 64 maps differ on the fresh cover)
- `persistent_pose_count` differs on 61 maps and the same-base dump probe on 11: persistent_pose_count and the dump probe are built from the agent footprint and do not read the standoff band at all, so their movement is terra commit 566867db (the compute_polygon_mask raster fix), not the gate semantics

### Which footprint model is Terra

terra commit 566867db fixed `compute_polygon_mask` to rasterise `(row, col)`, which INVERTS what these tools called "Terra's footprint".  Checked directly against `State._is_valid_move` over 4000 random poses (0.717 legal):

| model | agreement |
|---|---|
| fp_true_untransposed (the tools' 'terra' model) | 0.9995 |
| fp_masked_transposed (the tools' 'legacy_mirror' model) | 0.5270 |
| fp_masked_untransposed | 0.8832 |
| fp_true_transposed | 0.5450 |

The tools now use the 0.9995 model as `terra` and keep the retired one as `legacy_mirror`; the residual is the corner-in-bounds clause `free_poses` does not model.

## Exact commands

```
cd /home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
export JAX_PLATFORMS=cpu
export PYTHONPATH=/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
PY=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
BANK=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819
R=tools/trench_align_v2_revalidation_20260901

# junction contract, both semantics
$PY tools/check_trench_gate_multiowner.py [--gate-v1]

# v1 reproduction of the review-v4 witness (and its v2 counterpart)
$PY tools/audit_trench_alignment_feasibility.py --workers 8 [--gate-v1] --output $R/feasibility_reviewv4_<sem>.json

# (a) over-restriction audit
$PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \
    --dataset evaluation/gate_main/development --workers 24 [--gate-v1] \
    --output $R/overrestriction_gate_main_dev_<sem>.json
$PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \
    --dataset train_pilot_pooled_12cond --workers 24 [--gate-v1] \
    --output $R/overrestriction_train_pooled_12cond_<sem>.json

# (b) order-independent persistent station cover
$PY tools/check_trench_persistent_station_cover.py --bank-root $BANK \
    --dataset evaluation/gate_main/development --workers 24 [--gate-v1] \
    --output $R/station_cover_gate_main_dev_<sem>.json

# (c) axis sweep + the on-axis lane
$PY tools/check_trench_axis_sweep_feasibility.py --bank-root $BANK \
    --dataset evaluation/gate_main/development --workers 24 [--gate-v1] \
    --output $R/axis_sweep_gate_main_dev_<sem>.json

# (d) full 21-dataset preflight, net4 included
$PY tools/audit_trench_alignment_feasibility.py --bank $BANK --layout exact \
    --dataset train/018__trn-net3-side1-road ... --dataset evaluation/capability_floor/sealed \
    --workers 24 --witness-actions counts [--gate-v1] \
    --output $R/preflight_full_<sem>.json

# footprint model check and this summary
$PY $R/check_footprint_model.py
$PY $R/summarize_receipts.py
```

The full dataset list for (d) is in each preflight receipt's `contract.reproduction_command`.

