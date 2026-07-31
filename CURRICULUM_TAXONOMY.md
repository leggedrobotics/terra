# Terra Curriculum Taxonomy

- Current executable release: `v6-main`
- Executable registry:
  [`tools/map_generation/curriculum_taxonomy.py`](tools/map_generation/curriculum_taxonomy.py)
- Supported generator:
  [`tools/map_generation/generate_curriculum_bank.py`](tools/map_generation/generate_curriculum_bank.py)
- Bank, review, and experiment plan:
  [`D5_D7_IMPLEMENTATION_PLAN.md`](D5_D7_IMPLEMENTATION_PLAN.md)

This is the canonical human-readable reference for the current condition IDs,
factors, tiers, anchors, and bank counts. The executable registry is the source
of truth; a focused test checks the 32-row table below against it. Historical
`v3`, `v4`, and `v5` tables remain in the registry for provenance. The supported
generator explicitly selects `v6-main`; the module's historical default release
must not be read as the current curriculum.

`MAP_BENCHMARK_SPEC.md` remains the broader benchmark-design and reviewer-log
document. Physical construction and admission gates remain in the supported
generator. This document does not create a second generator or sampler.

## 1. A factor graph, not a difficulty ladder

The current review support is:

```text
T0 selected anchors                         9 conditions
T1 one-factor siblings                    19 conditions
   capacity 4 | distance 2 | layout 3 | geometry 7 | site 3
T2 composed changes                        4 conditions
                                             ----------
                                             32 total
```

`T0`, `T1`, and `T2` are **factor depth**, not measured policy hardness and not
a sequential unlock order. `T0` means a selected comparison baseline with no
`anchorConditionId`; it does not mean globally easiest. For example,
`trn-tee-side2` is a T0 anchor, while a policy may empirically find a particular
T1 condition easier. Likewise, `c2x`, `c1p6`, and `c1p2` are parallel points on
one capacity branch: all are T1, not three successive tiers.

The five T1 branches are siblings. Gallery folder order is organizational only.
An anchor names the factor-delta reference for a controlled comparison; it is
not proof of mastery, a promotion prerequisite, a strict scheduler edge, or a
guarantee that two realized maps are a literal pair.

The old `M0`-`M6` labels in `TRAINING_DESIGN.md` and historical sections of
`TRAINING_TASKS.md` describe earlier failed or proposed experiments. They are
not the current taxonomy.

## 2. Identity and condition grammar

The ID grammar has fixed factor order:

```text
<family>-<geometry>-<dump>[-<capacity>][-<distance>][-<site>][-<scale>]
```

Defaults `generous`, `clean`, `unspec`, and `std` are omitted. `near` is emitted
because it identifies the controlled direct-service distance baseline. The
taxonomy schema version (`taxonomy_version = 1`) and the selected release
(`v6-main`) are separate fields.

| Factor | Current tokens and meaning | Tier role |
|---|---|---|
| family | `fnd` foundation; `trn` trench | Identity only; never scored. |
| geometry | Foundation: `slab`, `slab-lg`, `proc`, `strips`. Trench: `straight`, `tee`, `seg2`, `seg3`, `net3`, `net4`. | `slab`, `slab-lg`, `proc`, `straight`, and `tee` are selected easy levels. `strips`, segmented trenches, and networks score one factor. |
| dump | `ring3x` capped gapped ring; `apron` nearby apron; `side1` one side; `side2` both trench sides; `altsides` alternating trench banks; `split` separated zones. | Family-dependent: foundation `ring3x`/`apron` and trench `side1`/`side2` are easy. Other current layouts score one factor. |
| capacity | Ratio of **reachable designated dump cells to dig cells** for controlled apron/tight bands: `c3x` 2.90-3.10, `c2x` 1.90-2.10, `c1p6` 1.55-1.75, `c1p2` 1.15-1.25, and trench `tight` 1.30-1.65. `ring3x` separately gates both designated and reachable ring capacity to 3-4x. | `generous` and `c3x` are easy in `v6-main`; `c2x`, `c1p6`, `c1p2`, and `tight` score one factor. |
| site | `clean`; `obj1` 1-2 light scattered objects; `obj` 2-5 scattered objects; `road` access road. | Only `clean` is easy. |
| distance | `unspec`; `near` direct service; `d12` and `d16` target median Euclidean raster distance from dig cells to the nearest designated dump cell. The distance-bin tolerance is +/-2 tiles. | `unspec` and `near` are easy; `d12` and `d16` score one factor. |
| scale | `std` standard extent; `s` short-arm mini. Minis also record their standard variant separately. | A real mitigating factor, but both levels are easy; scale never raises the tier. |

The live 64 x 64 map spans `36.5714285714 m`, so one tile is
`0.5714285714 m`. Distance tokens and generator distance receipts are in tiles;
the manifest also records the live metre conversion.

Tier is the count of factors at non-easy levels among geometry, dump, capacity,
site, distance, and scale. The family, source provenance, map identity, and
split are not scored. Work volume or physical extent is not assumed to be
harder: the large slab and the short tee are both T0, and short versus standard
scale contributes zero. Static validity, generator rejection rate, centered
IoU, human review disposition, reward design, partial-reset treatment, and
runtime sampling weight are also not difficulty factors.

For mini trenches, `standardVariantConditionId` says which standard condition
was shrunk. That relation is not the taxonomy anchor: the T1 mini networks
anchor to the T0 mini tee so that the anchor is strictly lower tier.

## 3. Current `v6-main` conditions

The current review source contains 64 generator-accepted scenarios for every
condition below: 2,048 scenarios total. The gallery shows 16 representative
graphics per condition (512 total), plus 32 condition overview sheets. Those
counts describe the review artifact, not a human-accepted training bank.

The rows between the markers are checked against `SPEC_TABLE_V6_MAIN` for ID,
order, tier, and anchor.

<!-- taxonomy:v6-main:start -->
| Condition | Depth | Review branch | Anchor | Controlled change from anchor |
|---|---:|---|---|---|
| `fnd-slab-ring3x` | T0 | Anchor / easy | - | baseline |
| `fnd-proc-ring3x` | T0 | Anchor / easy | - | baseline |
| `fnd-slab-lg-ring3x` | T0 | Anchor / easy | - | baseline |
| `fnd-slab-apron-near` | T0 | Anchor / easy | - | baseline |
| `fnd-slab-apron-c3x` | T0 | Anchor / easy | - | baseline |
| `fnd-slab-apron-c2x` | T1 | Dump capacity | `fnd-slab-apron-c3x` | `capacity: c3x -> c2x` |
| `fnd-slab-apron-c1p6` | T1 | Dump capacity | `fnd-slab-apron-c3x` | `capacity: c3x -> c1p6` |
| `fnd-slab-apron-c1p2` | T1 | Dump capacity | `fnd-slab-apron-c3x` | `capacity: c3x -> c1p2` |
| `fnd-slab-apron-d12` | T1 | Dump distance | `fnd-slab-apron-near` | `distance: near -> d12` |
| `fnd-slab-apron-d16` | T1 | Dump distance | `fnd-slab-apron-near` | `distance: near -> d16` |
| `fnd-slab-side1` | T1 | Dump layout | `fnd-slab-ring3x` | `dump: ring3x -> side1` |
| `fnd-slab-split` | T1 | Dump layout | `fnd-slab-ring3x` | `dump: ring3x -> split` |
| `fnd-strips-ring3x` | T1 | Geometry / topology | `fnd-slab-ring3x` | `geometry: slab -> strips` |
| `fnd-slab-ring3x-obj1` | T1 | Site constraints | `fnd-slab-ring3x` | `site: clean -> obj1` |
| `fnd-slab-ring3x-obj` | T1 | Site constraints | `fnd-slab-ring3x` | `site: clean -> obj` |
| `fnd-slab-ring3x-road` | T1 | Site constraints | `fnd-slab-ring3x` | `site: clean -> road` |
| `fnd-proc-side1-road` | T2 | Composed | `fnd-proc-ring3x` | `dump: ring3x -> side1`<br>`site: clean -> road` |
| `fnd-slab-side1-obj` | T2 | Composed | `fnd-slab-ring3x` | `dump: ring3x -> side1`<br>`site: clean -> obj` |
| `trn-straight-side2` | T0 | Anchor / easy | - | baseline |
| `trn-straight-side1` | T0 | Anchor / easy | - | baseline |
| `trn-tee-side2` | T0 | Anchor / easy | - | baseline |
| `trn-straight-side1-tight` | T1 | Dump capacity | `trn-straight-side1` | `capacity: generous -> tight` |
| `trn-straight-altsides` | T1 | Dump layout | `trn-straight-side2` | `dump: side2 -> altsides` |
| `trn-seg2-side2` | T1 | Geometry / topology | `trn-straight-side2` | `geometry: straight -> seg2` |
| `trn-seg3-side2` | T1 | Geometry / topology | `trn-straight-side2` | `geometry: straight -> seg3` |
| `trn-net3-side2` | T1 | Geometry / topology | `trn-straight-side2` | `geometry: straight -> net3` |
| `trn-net4-side2` | T1 | Geometry / topology | `trn-straight-side2` | `geometry: straight -> net4` |
| `trn-net3-side1-road` | T2 | Composed | `trn-straight-side1` | `geometry: straight -> net3`<br>`site: clean -> road` |
| `trn-net4-side1-road` | T2 | Composed | `trn-straight-side1` | `geometry: straight -> net4`<br>`site: clean -> road` |
| `trn-tee-side2-s` | T0 | Anchor / easy | - | baseline |
| `trn-net3-side2-s` | T1 | Geometry / topology | `trn-tee-side2-s` | `geometry: tee -> net3` |
| `trn-net4-side2-s` | T1 | Geometry / topology | `trn-tee-side2-s` | `geometry: tee -> net4` |
<!-- taxonomy:v6-main:end -->

There are 18 foundation and 14 trench conditions. The table has 9 T0 anchors,
19 T1 conditions, and 4 T2 compositions; the current main track has no T3
condition.

## 4. Matched counterfactuals and splits

`anchorConditionId` describes a factor delta. Literal counterfactual pairing is
enforced separately by generation and materialization:

| Pair family | Intended shared support | Changed factor |
|---|---|---|
| `fnd-slab-apron-{c3x,c2x,c1p6,c1p2}` | same slab dig and apron azimuth at a retained pair slot | capacity |
| `fnd-slab-apron-{near,d12,d16}` | same slab dig and apron orientation at a retained pair slot | distance |
| `trn-straight-side1` / `trn-straight-side1-tight` | same straight dig and one-side layout at a retained pair slot | capacity |

Only declared compatible dump, capacity, distance, or site variants may retain
a common pair/source. A topology change necessarily changes the realized dig
raster and is only a taxonomy comparison unless a future contract explicitly
pairs it.
`source_group_id` identifies the raw OSM footprint or realized procedural dig;
`pair_slot_id` identifies the intended counterfactual slot. If a
condition-specific reroll changes the realized dig inside a pair slot,
`materialize_splits.py` drops that slot rather than calling it matched. Every
retained variant of a source group and pair slot stays in one split.

The remaining identities have distinct roles:

- `condition_id`: factor combination in the table above;
- `scenario_id`: hash of the five reset-consumed arrays (target, action state,
  dumpability, occupancy, and distance);
- `episode_id`: hash of `scenario_id`, frozen reset seed, and environment
  protocol hash; and
- `split`: `train`, `promotion`, `development`, or `sealed`.

## 5. Review bank, accepted bank, and training bank

The current diverse-64 artifact and its 16-example gallery are **review-only**.
Their long generation began before the final source-group/pair-slot repair.
After Accept/Reject/Quarantine decisions are exported, selected conditions must
be regenerated with the committed generator and then split-frozen. The review
artifact itself must never be used for P5 training.

For every accepted condition, the pilot contract is:

| Split | Source-disjoint layouts | Use |
|---|---:|---|
| train | 64 | P5 scratch training screens |
| promotion | 16 | checkpoint promotion only |
| development | 16 | diagnosis and treatment selection |
| sealed | 32 | one final selected-policy evaluation |

This is 128 retained source groups per accepted condition. Review images are a
view of candidate scenarios, not an additional split. P6 expands only the train
split from 64 to 256 source groups per selected condition (192 additions); the
three evaluation splits remain frozen.

## 6. Runtime sampling is a separate treatment

The taxonomy fixes support and labels. It does not prescribe the training
weights. P5 compares four scratch arms on the regenerated accepted bank:

- `F-ANCHOR`: accepted foundation T0 conditions, uniform by condition;
- `T-ANCHOR`: accepted trench T0 conditions, uniform by condition;
- `G-UNIFORM`: every accepted condition, uniform by condition; and
- `G-ADAPTIVE`: the identical all-condition support, adaptively weighted.

The loader serializes tier depth as `Anchor`, `One-axis`, or `Composed`, but
those labels do not unlock maps. `G-ADAPTIVE` exposes every accepted condition
from update zero. Every 150 PPO updates it uses completion EMA to form

```text
q = 0.20 * Uniform(conditions) + 0.80 * Frontier
```

`Frontier` favors the most advanced conditions below the 0.75 mastery threshold
with a 0.15 per-condition cap. Mastered conditions remain in the uniform floor
and re-enter the frontier if competence falls. There is no strict stage unlock
or per-environment demotion. Fixed promotion/development evaluation remains on
the frozen target bank and is never reweighted to match training exposure.

Implementation details and launch gates live in the
[terra-baselines P5 document](/home/lorenzo/moleworks/.worktrees/terra_baselines_simple_mapbank_reward_20260730/docs/research/P5_ACCEPTED_BANK_EXPERIMENTS.md),
not in the taxonomy registry.
