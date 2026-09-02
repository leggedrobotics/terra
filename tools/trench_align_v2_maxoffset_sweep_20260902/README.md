# v2 "on the line" bound: coverage sweep

Date: 2026-09-02. Branch `epoch/terra-footprint-and-soil-containment-20260831`,
core clause committed as 6b6924b5 (`EnvConfig.trench_dig_max_offset_m`, default
2.0 m); every run below was made against that exact clause.
Bank `/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819`.

The clause under test: under v2 (`trench_dig_standoff_enforced=False`) a section
is pose-valid iff chassis yaw is parallel within tolerance **and** the base
centre's perpendicular distance to the axis is `<= trench_dig_max_offset_m`.
A value `<= 0` disables the clause and reproduces the 2026-09-01 yaw-only v2
semantics; that is the `disabled` row everywhere below and it is the control.

Every numpy replica in these tools was asserted against Terra's own exported
`fresh_trench_dig_alignment_valid` at every bound: **zero mismatches, every run**
(the `replica mismatches` columns). `tools/manual_trench_debugger.py
--replica-sweep 300` is 0 divergences under v2 at 2.00 m, under v2 with the
clause disabled, and under `--gate-v1`.

## TL;DR

**No bound is free.** Which loss you get depends on which instrument you ask.

| bound | (a) gate coverage, both panels | (b) order-independent persistent cover | (c) 2,400-map preflight | (d) on-axis retreat lane | oracle, 176 slots |
|---|---|---|---|---|---|
| 1.14 m | **-9 cells / 5 maps** (panel), **-70 / 30 maps** (pooled) | 0/224 | 0/2400 | -2 tee, -2 network, -7 segmented cells | not run |
| 1.15 m | **-9 cells / 5 maps** (panel), **-66 / 30 maps** (pooled) | 0/224 | not run | identical to disabled | not run |
| 1.71 m | 0 (panel), **-1 cell / 1 map** (pooled) | 0/224 | 0/2400 | identical to disabled | 0/176 |
| 1.72 m | 0 (panel), **-1 cell / 1 map** (pooled) | 0/224 | not run | identical to disabled | not run |
| **2.00 m** (current default) | **0 / 0** | 14/224 | not run | identical to disabled | not run |
| 2.29 m | 0 / 0 | 53/224 | 326/2400 | identical to disabled | 28/176 |
| 2.86 m | 0 / 0 | 146/224 | 1642/2400 | identical to disabled | 44/176 |
| **3.43 m** | **0 / 0** | 221/224 | **2400/2400, preflight_passed** | identical to disabled | 69/176 |
| disabled | 0 / 0 | 224/224 | 2400/2400 | (reference) | 146/176 |

* **Smallest bound that costs the gate no coverage at all (a):** **2.00 m** --
  the value already in `EnvConfig`. 1.72 m loses exactly one cell on one
  `trn-net4-side2-s` map of the pooled bank; 1.15 m loses 66.
* **Smallest bound that also leaves the full preflight intact (c):**
  **3.43 m** -- the only finite bound with `preflight_passed = True`, byte-equal
  to the disabled control on all 2,400 maps.
* **No finite bound preserves the order-independent persistent station cover
  (b)**, and none preserves the scripted oracle's completion rate. Both are
  consequences of the clause, not of the gate: see "The collision" below.

## Recommendation

**Ship 3.43 m** (6 tiles) if a bound ships at all, and treat 2.00 m as the floor
below which the *gate itself* starts refusing cells.

Why 3.43 m and not smaller:

1. It costs **nothing** on the gate's own coverage (0 cells on 224 panel maps
   and on 1,440 pooled maps), nothing on the on-axis retreat lane, and nothing
   on the full 2,400-map preflight -- which is the check that gates the training
   bank, and which every tighter bound fails.
2. It still does exactly the job it was added for. Lorenzo's two sideways digs
   were at **3.84 m** and **6.52 m**; the retired v1 lane is **[3.5, 7.0] m**.
   A 3.43 m ceiling excludes the entire v1 lane and both observed poses, by
   0.07 m at the tight end. The clause remains "stand on the trench".
3. Everything below it is bought with executable coverage, not with better
   alignment: 2.86 m already leaves 758 maps without a complete preflight cover,
   2.29 m leaves 2,074, and 1.71 m leaves all 2,400.

Why not smaller even though 2.00 m is lossless on (a): (a) asks "does some pose
exist", with the base centre merely off padding. It does not ask whether that
pose is one a machine can stand in once the trench is open, nor whether it can
be driven to. As soon as either is required -- (b), (c), or the oracle -- 2.00 m
is far worse than 3.43 m, and it buys nothing in return.

## The collision the sweep exposes

Two instruments assume a machine that never stands where it will later dig.
`check_trench_persistent_station_cover.py` and the
`audit_trench_alignment_feasibility.py` preflight both build their pose space
from `blocked = padding | all target<0`, the order-independent worst case, so a
station is a pose that stays legal with the WHOLE trench already dug. That is
what makes their verdicts order-free and what stops a controller walling itself
off behind its own excavation.

The on-the-line clause requires the opposite. The chassis is 7 cells across, the
trench band is ~3 cells wide, so standing within ~5 cells of the axis puts the
footprint on target cells -- blocked in that model. The tighter the bound, the
fewer persistent stations exist. **Not because the gate refuses the dig, but
because the pose the gate now demands is not one the pessimistic model allows.**
Below roughly (trench half width + chassis half width) tiles the two are
mutually exclusive by construction, which is why (b) is 0/224 at every bound
<= 1.72 m and only recovers at 2.86-3.43 m.

The model that matches the intended manoeuvre is the monotone retreat: the cone
starts 3.64 m from the base centre while the chassis reaches only 3.14 m ahead,
so a machine that only ever backs up digs strictly ahead of every pose it will
occupy and never stands on a cell it dug. That is the `fresh` blocked model, and
under it (table (d)) **every bound >= 1.15 m is identical to the disabled
control on every family** -- the retreat lane does not care about the bound.
Only 1.14 m dents it, and by 11 cells in total (2 tee, 2 network, 7 segmented).

`(b)` and `(c)` disagree on 3 maps at 3.43 m (`trn-seg2-side2` slots 385, 390,
394 of gate_main/development). The footprint tables and the 144 cone tables of
the two tools are byte-identical (checked); the difference is the criterion.
`(b)` is single-shot and order-independent -- the whole trench is fresh at once,
so a junction cone always contains the perpendicular branch's exclusive cells.
`(c)` is a monotone chain, so it may finish one branch and then the other, which
is the resolution the gate contract explicitly allows. `(c)` is the more
permissive and the more Terra-faithful of the two here.

## What the oracle says, and what it does not

The scripted oracle navigates over **persistent poses only**. At 3.43 m it
completes 69/176 against the control's 146/176, with zero deadlocks and zero
illegal spoil at every bound. The stall attribution says the cause precisely:
across the 107 failed slots at 3.43 m, of 2,137 remaining cells

* 2,137 have an in-cone, yaw-parallel, on-the-line, non-vetoed pose,
* 1,911 have a footprint-legal one, 1,888 a persistent one,
* but only **70** have a *reachable* one.

The loss is **pose-graph connectivity**, not the gate: the on-line lane is a
narrow corridor along the trench, and reaching it from the spawn over poses that
stay legal with the whole trench dug is what fails. The control run at
`--max-offset-m 0` reproduces the 2026-09-01 receipt exactly (146/176, median
78.5, p90 130, max 183, 988 stations, 366 on-axis, 0 deadlocks, 0 spoil), so
the drop is the clause and nothing else.

**Consequence for the next step.** If the clause ships, the controller and the
preflight have to adopt the retreat pattern (enter the lane, dig ahead, back
out) rather than the "never stand on your target" policy; the oracle's number is
a property of that policy, not a ceiling on the task. `stations_on_axis ==
stations` in every bounded run is the receipt that the clause did bind.

## Exactly which cells are lost, and why

Every cell any bound removes from the gate's admissible set is a **junction
veto**, on a 4-axis `trn-net4-*` map, in the map interior:

| bound | panel | cells lost | maps | mechanism | owners | nearest padding | nearest cross-section cell |
|---|---|---|---|---|---|---|---|
| 1.14 m | gate_main | 9 | 5 (`net4-side1-road` 2, `net4-side2` 2, `net4-side2-s` 1) | junction_veto 9/9 | 1 (single-owner) | > 20 cells (none nearby) | 1-3 cells |
| 1.71 m | pooled | 1 | 1 (`trn-net4-side2-s` slot 2242) | junction_veto | 1 | > 20 cells | 1 cell |

Not padding-adjacent, not border cells. The machine is pinned within 1-2 cells
of the axis, so every cone that reaches these cells also sweeps an exclusive
cell of the crossing branch, and the all-or-nothing DO is refused. Widening the
lane by one cell (1.72 m -> 2.00 m) restores them. Receipts:
`lost_cells_gate_main_b114.json`, `lost_cells_pooled_net4_b171.json`.

## A rounding trap in the requested grid

`tile = 0.5714285969734192 m`, so `2 tiles = 1.1428...`, `3 tiles = 1.7142...`,
`4 tiles = 2.2857...`, `5 tiles = 2.8571...`, `6 tiles = 3.4285...`. The
requested values 1.14 and 1.71 fall just *below* 2 and 3 tiles and therefore
admit only 0-1 and 0-2 integer-cell lanes; 2.29, 2.86 and 3.43 fall just above 4,
5 and 6 tiles and do admit them. 1.15 and 1.72 were added so the "2 tiles" and
"3 tiles" rows exist as intended, and 2.00 (the current config default) was added
because it sits between them. The `lanes admitted` column in every table states
what each value actually admits for an axis-aligned section; oblique sections
take fractional offsets, which is why 1.15 and 1.71 differ despite admitting the
same integer lanes.

### (a) Over-restriction audit -- evaluation/gate_main/development

| bound | lanes admitted | scope | maps | target cells | cells admissibly diggable | % | reachable-but-never-admissible | maps losing cells | applicable candidates | admissible stations | replica mismatches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.14 m | 0-1 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 439,094 | 88,736 | 0 |
| 1.14 m | 0-1 cells | net4 | 48 | 5,515 | 5,506 | 99.84 | 9 | 5 | 276,616 | 35,300 | 0 |
| 1.14 m | 0-1 cells | all | 224 | 18,943 | 18,934 | 99.95 | 9 | 5 | 715,710 | 124,036 | 0 |
| 1.15 m | 0-2 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 442,506 | 89,388 | 0 |
| 1.15 m | 0-2 cells | net4 | 48 | 5,515 | 5,506 | 99.84 | 9 | 5 | 278,080 | 35,471 | 0 |
| 1.15 m | 0-2 cells | all | 224 | 18,943 | 18,934 | 99.95 | 9 | 5 | 720,586 | 124,859 | 0 |
| 1.71 m | 0-2 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 645,572 | 129,273 | 0 |
| 1.71 m | 0-2 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 410,122 | 50,793 | 0 |
| 1.71 m | 0-2 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 1,055,694 | 180,066 | 0 |
| 1.72 m | 0-3 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 649,967 | 130,048 | 0 |
| 1.72 m | 0-3 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 411,562 | 50,968 | 0 |
| 1.72 m | 0-3 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 1,061,529 | 181,016 | 0 |
| 2.00 m | 0-3 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 764,887 | 151,761 | 0 |
| 2.00 m | 0-3 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 480,244 | 59,967 | 0 |
| 2.00 m | 0-3 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 1,245,131 | 211,728 | 0 |
| 2.29 m | 0-4 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 887,717 | 174,806 | 0 |
| 2.29 m | 0-4 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 545,329 | 69,387 | 0 |
| 2.29 m | 0-4 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 1,433,046 | 244,193 | 0 |
| 2.86 m | 0-5 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 1,125,603 | 220,299 | 0 |
| 2.86 m | 0-5 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 667,133 | 88,394 | 0 |
| 2.86 m | 0-5 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 1,792,736 | 308,693 | 0 |
| 3.43 m | 0-6 cells | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 1,362,538 | 264,992 | 0 |
| 3.43 m | 0-6 cells | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 775,359 | 107,070 | 0 |
| 3.43 m | 0-6 cells | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 2,137,897 | 372,062 | 0 |
| disabled | every lane | no_net4 | 176 | 13,428 | 13,428 | 100.00 | 0 | 0 | 2,663,512 | 459,383 | 0 |
| disabled | every lane | net4 | 48 | 5,515 | 5,515 | 100.00 | 0 | 0 | 1,681,025 | 187,606 | 0 |
| disabled | every lane | all | 224 | 18,943 | 18,943 | 100.00 | 0 | 0 | 4,344,537 | 646,989 | 0 |

### (a) Over-restriction audit -- train_v2_pooled_generalist (trench slots)

| bound | lanes admitted | scope | maps | target cells | cells admissibly diggable | % | reachable-but-never-admissible | maps losing cells | applicable candidates | admissible stations | replica mismatches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.14 m | 0-1 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 2,744,340 | 564,461 | 0 |
| 1.14 m | 0-1 cells | net4 | 288 | 33,546 | 33,476 | 99.79 | 70 | 30 | 1,660,148 | 211,461 | 0 |
| 1.14 m | 0-1 cells | all | 1440 | 120,479 | 120,409 | 99.94 | 70 | 30 | 4,404,488 | 775,922 | 0 |
| 1.15 m | 0-2 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 2,763,039 | 568,137 | 0 |
| 1.15 m | 0-2 cells | net4 | 288 | 33,546 | 33,480 | 99.80 | 66 | 30 | 1,672,838 | 212,907 | 0 |
| 1.15 m | 0-2 cells | all | 1440 | 120,479 | 120,413 | 99.95 | 66 | 30 | 4,435,877 | 781,044 | 0 |
| 1.71 m | 0-2 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 4,031,671 | 824,618 | 0 |
| 1.71 m | 0-2 cells | net4 | 288 | 33,546 | 33,545 | 100.00 | 1 | 1 | 2,460,159 | 303,294 | 0 |
| 1.71 m | 0-2 cells | all | 1440 | 120,479 | 120,478 | 100.00 | 1 | 1 | 6,491,830 | 1,127,912 | 0 |
| 1.72 m | 0-3 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 4,052,215 | 828,348 | 0 |
| 1.72 m | 0-3 cells | net4 | 288 | 33,546 | 33,545 | 100.00 | 1 | 1 | 2,472,534 | 304,902 | 0 |
| 1.72 m | 0-3 cells | all | 1440 | 120,479 | 120,478 | 100.00 | 1 | 1 | 6,524,749 | 1,133,250 | 0 |
| 2.00 m | 0-3 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 4,778,570 | 967,824 | 0 |
| 2.00 m | 0-3 cells | net4 | 288 | 33,546 | 33,546 | 100.00 | 0 | 0 | 2,876,980 | 358,578 | 0 |
| 2.00 m | 0-3 cells | all | 1440 | 120,479 | 120,479 | 100.00 | 0 | 0 | 7,655,550 | 1,326,402 | 0 |
| 2.29 m | 0-4 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 5,540,469 | 1,113,674 | 0 |
| 2.29 m | 0-4 cells | net4 | 288 | 33,546 | 33,546 | 100.00 | 0 | 0 | 3,271,380 | 415,979 | 0 |
| 2.29 m | 0-4 cells | all | 1440 | 120,479 | 120,479 | 100.00 | 0 | 0 | 8,811,849 | 1,529,653 | 0 |
| 2.86 m | 0-5 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 7,046,240 | 1,405,249 | 0 |
| 2.86 m | 0-5 cells | net4 | 288 | 33,546 | 33,546 | 100.00 | 0 | 0 | 3,986,638 | 530,121 | 0 |
| 2.86 m | 0-5 cells | all | 1440 | 120,479 | 120,479 | 100.00 | 0 | 0 | 11,032,878 | 1,935,370 | 0 |
| 3.43 m | 0-6 cells | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 8,550,008 | 1,690,903 | 0 |
| 3.43 m | 0-6 cells | net4 | 288 | 33,546 | 33,546 | 100.00 | 0 | 0 | 4,634,445 | 642,892 | 0 |
| 3.43 m | 0-6 cells | all | 1440 | 120,479 | 120,479 | 100.00 | 0 | 0 | 13,184,453 | 2,333,795 | 0 |
| disabled | every lane | no_net4 | 1152 | 86,933 | 86,933 | 100.00 | 0 | 0 | 16,741,854 | 2,941,127 | 0 |
| disabled | every lane | net4 | 288 | 33,546 | 33,546 | 100.00 | 0 | 0 | 10,107,534 | 1,123,497 | 0 |
| disabled | every lane | all | 1440 | 120,479 | 120,479 | 100.00 | 0 | 0 | 26,849,388 | 4,064,624 | 0 |

### (b) Order-independent persistent station cover -- evaluation/gate_main/development

| bound | lanes admitted | maps | complete cover, any pose | complete cover, Terra-legal PERSISTENT station | cells from a persistent station | replica mismatches |
|---|---|---|---|---|---|---|
| 1.14 m | 0-1 cells | 224 | 219/224 | 0/224 | 10,786/18,943 | 0 |
| 1.15 m | 0-2 cells | 224 | 219/224 | 0/224 | 10,788/18,943 | 0 |
| 1.71 m | 0-2 cells | 224 | 224/224 | 0/224 | 10,884/18,943 | 0 |
| 1.72 m | 0-3 cells | 224 | 224/224 | 0/224 | 10,886/18,943 | 0 |
| 2.00 m | 0-3 cells | 224 | 224/224 | 14/224 | 12,055/18,943 | 0 |
| 2.29 m | 0-4 cells | 224 | 224/224 | 53/224 | 14,077/18,943 | 0 |
| 2.86 m | 0-5 cells | 224 | 224/224 | 146/224 | 18,287/18,943 | 0 |
| 3.43 m | 0-6 cells | 224 | 224/224 | 221/224 | 18,936/18,943 | 0 |
| disabled | every lane | 224 | 224/224 | 224/224 | 18,943/18,943 | 0 |

### (c) Full 2,400-map preflight (net4 included)

| bound | lanes admitted | maps | maps WITHOUT a complete fresh cover | preflight_passed | cells covered | wall s |
|---|---|---|---|---|---|---|
| 1.14 m | 0-1 cells | 2400 | 2400 | False | 91,898/200,451 | 682 |
| 1.71 m | 0-2 cells | 2400 | 2400 | False | 94,474/200,451 | 688 |
| 2.29 m | 0-4 cells | 2400 | 2074 | False | 131,050/200,451 | 663 |
| 2.86 m | 0-5 cells | 2400 | 758 | False | 190,555/200,451 | 813 |
| 3.43 m | 0-6 cells | 2400 | 0 | True | 200,451/200,451 | 712 |
| disabled | every lane | 2400 | 0 | True | 200,451/200,451 | 959 |

Per condition, maps with a complete cover:

| condition | 1.14 m | 1.71 m | 2.29 m | 2.86 m | 3.43 m | disabled |
|---|---|---|---|---|---|---|
| trn-net3-side1-road | 0/160 | 0/160 | 1/160 | 87/160 | 160/160 | 160/160 |
| trn-net3-side2 | 0/160 | 0/160 | 1/160 | 87/160 | 160/160 | 160/160 |
| trn-net3-side2-s | 0/160 | 0/160 | 0/160 | 103/160 | 160/160 | 160/160 |
| trn-net4-side1-road | 0/160 | 0/160 | 0/160 | 64/160 | 160/160 | 160/160 |
| trn-net4-side2 | 0/160 | 0/160 | 0/160 | 64/160 | 160/160 | 160/160 |
| trn-net4-side2-s | 0/160 | 0/160 | 25/160 | 104/160 | 160/160 | 160/160 |
| trn-seg2-side2 | 0/160 | 0/160 | 0/160 | 74/160 | 160/160 | 160/160 |
| trn-seg3-side2 | 0/160 | 0/160 | 0/160 | 79/160 | 160/160 | 160/160 |
| trn-straight-allfree | 0/160 | 0/160 | 52/160 | 148/160 | 160/160 | 160/160 |
| trn-straight-altsides | 0/160 | 0/160 | 52/160 | 148/160 | 160/160 | 160/160 |
| trn-straight-side1 | 0/160 | 0/160 | 52/160 | 148/160 | 160/160 | 160/160 |
| trn-straight-side1-tight | 0/160 | 0/160 | 52/160 | 148/160 | 160/160 | 160/160 |
| trn-straight-side2 | 0/160 | 0/160 | 52/160 | 148/160 | 160/160 | 160/160 |
| trn-tee-side2 | 0/160 | 0/160 | 13/160 | 99/160 | 160/160 | 160/160 |
| trn-tee-side2-s | 0/160 | 0/160 | 26/160 | 141/160 | 160/160 | 160/160 |

### (d) Axis sweep -- evaluation/gate_main/development

Main sweep (all 12 cabins, FORWARD/BACKWARD lanes, dumping removed, Terra footprint, PERSISTENT blocked model):

| bound | lanes admitted | maps complete (exact) | cells covered (exact) | replica mismatches |
|---|---|---|---|---|
| 1.14 m | 0-1 cells | 0/224 | 10,648/18,943 | 0 |
| 1.15 m | 0-2 cells | 0/224 | 10,650/18,943 | 0 |
| 1.71 m | 0-2 cells | 0/224 | 10,734/18,943 | 0 |
| 1.72 m | 0-3 cells | 0/224 | 10,737/18,943 | 0 |
| 2.00 m | 0-3 cells | 14/224 | 11,928/18,943 | 0 |
| 2.29 m | 0-4 cells | 52/224 | 13,960/18,943 | 0 |
| 2.86 m | 0-5 cells | 143/224 | 18,202/18,943 | 0 |
| 3.43 m | 0-6 cells | 219/224 | 18,934/18,943 | 0 |
| disabled | every lane | 224/224 | 18,943/18,943 | 0 |

The ON-AXIS LANE (perpendicular <= 2 tiles of the axis, cabin straight ahead or behind, FORWARD/BACKWARD only, `fresh` blocked model = padding only, Terra footprint):

| family | metric | 1.14 m | 1.15 m | 1.71 m | 1.72 m | 2.00 m | 2.29 m | 2.86 m | 3.43 m | disabled |
|---|---|---|---|---|---|---|---|---|---|---|
| straight | maps complete | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 |
| straight | cells covered | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 | 4408/4408 |
| tee | maps complete | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 |
| tee | cells covered | 2415/2465 | 2417/2465 | 2417/2465 | 2417/2465 | 2417/2465 | 2417/2465 | 2417/2465 | 2417/2465 | 2417/2465 |
| network | maps complete | 8/64 | 8/64 | 8/64 | 8/64 | 8/64 | 8/64 | 8/64 | 8/64 | 8/64 |
| network | cells covered | 5300/5506 | 5302/5506 | 5302/5506 | 5302/5506 | 5302/5506 | 5302/5506 | 5302/5506 | 5302/5506 | 5302/5506 |
| road | maps complete | 0/32 | 0/32 | 0/32 | 0/32 | 0/32 | 0/32 | 0/32 | 0/32 | 0/32 |
| road | cells covered | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 | 3486/3596 |
| segmented | maps complete | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 | 5/32 |
| segmented | cells covered | 2747/2968 | 2754/2968 | 2754/2968 | 2754/2968 | 2754/2968 | 2754/2968 | 2754/2968 | 2754/2968 | 2754/2968 |

### Scripted oracle -- 176 gate_main/development slots

`--horizon 450 --extended-horizon 900 --verify-action-mask`, checkpoint `oracle_t1_arm_train_config_only.pkl`, terra revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4.

| bound | completions | rate | median steps | p90 | mean dig fraction | stations | on-axis stations | loaded-no-dump deadlocks | illegal spoil | per-step alignment checks |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.71 m | 0/176 | 0.000 | - | - | 0.445 | 818 | 818 | 0 | 0u / 0 slots | 158,400 |
| 2.29 m | 28/176 | 0.159 | 95.5 | 130.0 | 0.626 | 1,056 | 1,056 | 0 | 0u / 0 slots | 135,912 |
| 2.86 m | 44/176 | 0.250 | 101.0 | 157.7 | 0.746 | 1,130 | 1,130 | 0 | 0u / 0 slots | 123,445 |
| 3.43 m | 69/176 | 0.392 | 104.0 | 168.8 | 0.861 | 1,266 | 1,266 | 0 | 0u / 0 slots | 103,948 |
| disabled | 146/176 | 0.830 | 78.5 | 130.0 | 0.969 | 988 | 366 | 0 | 0u / 0 slots | 39,449 |

## Reproduction

```bash
W=/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
BASE=/home/lorenzo/moleworks/.worktrees/terra_baselines_trench_pose_alignment_20260818
PY=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
BANK=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819
R=$W/tools/trench_align_v2_maxoffset_sweep_20260902
export JAX_PLATFORMS=cpu PYTHONPATH=$W

# the whole sweep (resumable; .done markers skip finished jobs)
WORKERS=22 bash $R/driver_all.sh
# one bound, one phase:
WORKERS=22 BOUNDS="b343:3.43" PHASES="cover overres axis pooled preflight" bash $R/run_sweep.sh

# why a bound loses a cell
$PY $R/classify_lost_cells.py --bank-root $BANK \
    --dataset evaluation/gate_main/development --max-offset-m 1.14 \
    --workers 24 --output $R/lost_cells_gate_main_b114.json

# the scripted oracle at one bound (176 slots)
bash $R/run_oracle.sh b343 3.43        # tag, metres; 0 = clause disabled

# tables and summary
python3 $R/summarize_sweep.py && python3 $R/make_tables.py

# replica-vs-Terra assertions in the manual debugger
$PY $W/tools/manual_trench_debugger.py --headless --replica-sweep 300 --slot 455 \
    [--max-offset-m X | --gate-v1]
# junction contract, three configurations
$PY $W/tools/check_trench_gate_multiowner.py [--gate-v1 | --max-offset-m 0]
```

Receipts in this directory: `station_cover_*`, `overrestriction_gate_main_dev_*`,
`overrestriction_train_v2_pooled_*`, `axis_sweep_*`, `preflight_full_*`,
`oracle_176slot_*`, `lost_cells_*` (`.json` receipt + `.log` stdout for each),
`run_log.txt` (per-job wall times), `summary.json`, `tables.md`. Verification
logs: `multiowner_v2_default.log`, `multiowner_maxoffset0.log`,
`replica_v2_final.log` (clause at 2.00 m), `replica_v2_off.log` (clause
disabled), `replica_v1_final.log` (`--gate-v1`), `v1_bound_inert.log` (the bound
is provably inert under v1: identical cells and stations at 1.14 m and 3.43 m).
