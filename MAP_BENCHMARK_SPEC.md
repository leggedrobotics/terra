# TerraMap-Bench specification

Status: proposed v0.2, for review before implementation

This document specifies a public, versioned benchmark for Terra excavation
maps. It covers the map format, difficulty taxonomy, curriculum levels,
feasibility gates, train/evaluation splits, policy submissions, metrics, and a
website for inspecting maps, distributions, and policy failures.

No benchmark bank is frozen by this document. The existing local review set and
B0 feasibility panels are design inputs, not benchmark releases.

Normative terms `must`, `should`, and `may` describe release requirements.

## 1. Purpose

TerraMap-Bench should answer five questions that the current pooled training
metrics cannot:

1. Which exact geometry, dumping, site, work, or reset conditions does a policy
   solve?
2. Is a map statically valid and dynamically completable under the frozen Terra
   action contract?
3. Is a training distribution diverse and balanced, or is it dominated by
   repeated identities and a few easy cells?
4. Does performance transfer to source-disjoint maps and unseen combinations?
5. Can another policy be evaluated under the same maps, horizon, completion
   semantics, and reporting contract?

The benchmark is for excavation planning, initially foundations and trenches.
Relocation, grading, multiple agents, and other embodiments can become later
versioned suites without changing the v1 meaning.

## 2. Design principles

### 2.1 Factorized conditions, named progression levels

A map is described by its factor vector. It is never defined only as "easy",
"medium", or "hard".

TerraMap-Bench retains the familiar `M0` through `M5` progression, but gives
each level a descriptive, single-axis name and derives membership from
versioned factor rules. The deployment mixture is reported separately because
it is a mixture, not another ordered difficulty:

| Tier | Display name | Main change |
|---|---|---|
| `M0` | Nearby Generous | Simple geometry and large nearby dump regions |
| `M1` | Nearby Geometry | Connected geometry variation while dumping stays generous and nearby |
| `M2` | Nearby Dump Constraints | Side access, fragmentation, and irregular nearby masks on qualified geometry |
| `M3` | Complex Geometry | Structural foundations and intersecting or disconnected trench networks |
| `M4` | Site Access | Objects, roads, barriers, and then combined site constraints |
| `M5` | Remote Hauling | Medium/far hauling and later capacity pressure |
| `D0` | Deployment Mix | Frozen realistic mixture of already qualified factors; not a curriculum tier |

These tiers are a curriculum ordering, not a claim that every policy finds
every `M1` map easier than every `M2` map. Empirical difficulty remains
policy-dependent and is reported separately.

### 2.2 Separate map, reset, dynamics, and policy evaluation

The benchmark must not hide changes to the task behind a map label:

- a **map** fixes excavation geometry, obstacles, legal dumping, and all
  environment-consumed map layers;
- a **scenario** adds the exact initial soil, excavator state, and environment
  reset seed;
- a **protocol** fixes Terra dynamics, action and observation contracts,
  completion semantics, and horizon; and
- an **episode result** is one policy evaluated on one scenario under one
  protocol.

Full-task and partial-reset scenarios are separate tracks. Reward schedules are
training choices and are not part of map difficulty.

### 2.3 Exact visible dumping

The accepted dump region is the exact visible dump mask everywhere: actions,
completion, termination, reward, capacity validation, and evaluation.

The starter benchmark freezes mass-conserving contained-pile dynamics: a legal
dump action may spread soil, but its support is restricted to the exact
accepted mask and all volume is redistributed there without clipping or
deletion. Any off-mask positive soil explicitly present in a scenario must
still be cleaned before success. Success uses the frozen
`exact_visible_dump_v1` predicate. No hidden one-cell tolerance is permitted.

### 2.4 Diagnose with cells, rank with balanced cells

An exact **cell** is a normalized factor combination. Every evaluation result
must be available per cell. The default aggregate macro-averages cell success
rates so a large or easy cell cannot hide a failed condition.

### 2.5 Static-first and simple

The first implementation extends the existing exact-loader dataset rather than
introducing a new runtime loader or generic curriculum framework.

It uses:

- JSON/JSONL manifests and SHA-256 checksums;
- the current indexed NumPy sidecars;
- small offline build, static-validation, and export scripts; and
- one generated local `index.html`.

It does not initially require a database, a learned generator, a live
curriculum service, dynamic witness replay, policy evaluation, model
submission, arbitrary Python uploads, or arbitrary containers. Those later
publication stages reuse the same manifest rather than expanding the first
implementation.

## 3. Canonical objects and identity

The standard distinguishes identities that are currently easy to conflate:

| Object | Meaning |
|---|---|
| `source_group_id` | Underlying OSM footprint or procedural base source before counterfactual variants |
| `geometry_id` | Canonical excavation geometry |
| `map_id` | Physical target, occupancy, dumpability, and map metadata |
| `scenario_id` | Map plus exact initial soil, agent state, and reset seed |
| `treatment_id` | Scenario plus reward-distance/reward-treatment digest used for a training or evaluation artifact |
| `slot_id` | One sampling/evaluation slot referencing a scenario |
| `condition_id` | Mechanically generated normalized factor cell |
| `suite_id` | Versioned set of conditions and split counts |
| `release_id` | Immutable maps, protocol, scoring, and checksum tree |

A repeated slot is not a new map. Manifests must report slot count, unique
scenario count, unique map count, and unique source-group count.

All variants derived from one source group, such as one-side and both-side dump
masks on the same trench, must stay in the same split.

## 4. Existing-compatible release format

```text
terramap-bench-v1/
  benchmark.json
  conditions.jsonl
  scenarios.jsonl
  audit.jsonl
  source_registry.jsonl
  provenance.json
  coverage.json
  validation.json
  files.sha256
  splits/
    public_train/
      <exact-loader datasets>
    public_dev/
      <exact-loader datasets>
    sealed_pilot/
      <local held-out exact-loader datasets>
    private_test/
      <server-only exact-loader datasets>
    compositional_dev/
      <public held-out-factor-combination datasets>
    compositional_test/
      <held-out-factor-combination datasets>
  galleries/
    <generated public images and metadata>
```

Each exact-loader dataset keeps the current indexed sidecars:

```text
images/          target values {-1, 0, 1}
occupancy/       obstacles
dumpability/     legal dumping surface
actions/         initial soil/reset state
distance/        dense-reward distance field
metadata/        trench/foundation metadata
dataset.json
manifest.jsonl
```

`benchmark.json` freezes:

- schema, taxonomy, suite, and release versions;
- Terra and terra-baselines commits;
- generator, validator, and evaluator hashes;
- map shape, tile size, coordinate/frame convention, and dtypes;
- map-layer meanings;
- exact dump, mass, dynamics, action, observation, and completion contracts;
- evaluation horizon and deterministic action rule;
- split counts and cell weights;
- similarity/leakage policy; and
- the root checksum.

Changing a frozen item creates a new scored benchmark version.

## 5. Canonical scenario schema

The scenario manifest contains authored identity, provenance, declared
condition inputs, and immutable layer hashes. Generator-authored metadata is
not accepted as proof of capacity, distance, connectivity, or feasibility.
Those values are recomputed from saved arrays by the frozen validator and
written once to `audit.jsonl`.

The website joins the two records by `scenario_id`. Condition membership is
validated from audit values; a stale authored label fails validation.

```yaml
schema: terra_benchmark_scenario_v1
release_id: terramap-bench-v1.0.0

scenario_id: ...
map_id: ...
geometry_id: ...
source_group_id: ...
source_id: ...
condition_id: ...
split: public_train | public_dev | sealed_pilot | private_test | compositional_dev | compositional_test

family: foundation | trench
tier_id: M0 | M1 | M2 | M3 | M4 | M5
suite_ids: [...]

layers:
  target_sha256: ...
  occupancy_sha256: ...
  dumpability_sha256: ...
  initial_soil_sha256: ...
  metadata_sha256: ...
  shape: [64, 64]
  tile_size_m: 0.6875

generator:
  name: ...
  revision: ...
  config_sha256: ...
  seed: ...
  attempt: ...

declared_condition:
  geometry_class: ...
  topology: ...
  dump_layout: ...
  side_access: ...
  site_class: ...
  volume_band: ...
  reset_mode: ...

initial_condition:
  environment_reset_seed: ...
  initial_agent_state:
    base_position: [...]
    base_orientation: ...
    loaded_volume: ...
    current_agent: ...
  serialized_state_sha256: ...
  initial_soil_sha256: ...
  initial_negative_volume: ...
  initial_positive_volume: ...
  completion_fraction: ...

reward_treatment:
  treatment_id: ...
  reward_distance_sha256: ...
  reward_contract_sha256: ...
```

The corresponding validator-owned record is:

```yaml
schema: terra_benchmark_audit_v1
release_id: terramap-bench-v1.0.0
scenario_id: ...
validator_revision: ...

geometry:
  source_class: ...
  topology: ...
  dig_components: ...
  dig_cells: ...
  required_volume: ...
  segment_count: ...
  axis_count: ...
  junction_count: ...
  junction_degrees: [...]

dump:
  layout: ...
  side_access: ...
  side_sign: ...
  component_count: ...
  cells_per_component: [...]
  accepted_cells: ...
  legal_free_coverage: ...
  path_distance_tiles: {p50: ..., p95: ..., max: ...}
  path_distance_m: {p50: ..., p95: ..., max: ...}
  single_layer_area_ratio: ...
  reachable_capacity_ratio: ...
  representable_remaining_volume: ...
  minimum_cell_headroom: ...

site:
  class: ...
  object_count: ...
  obstacle_fraction: ...
  nondump_fraction: ...
  traversable_components: ...
  spawn_component_fraction: ...
  minimum_access_width_tiles: ...
  pre_dig_workspace_coverage: ...
  post_dig_workspace_coverage: ...

work:
  volume_band: ...
  required_volume: ...
  transport_work_proxy: ...

reset:
  mode: full | partial_in_zone | partial_mixed | partial_near_zone
  completion_fraction: ...
  initial_negative_volume: ...
  initial_positive_volume: ...
  remaining_components: ...
  mass_balance: ...

validation:
  format_valid: true
  static_valid: true
  witnessed: true
  witness_type: constructive | planner | policy
  witness_trace_sha256: ...
  witness_horizon: 450
  witness_initial_state_sha256: ...
```

`map_id` changes when a physical map layer changes. `scenario_id` additionally
includes initial soil, the serialized initial excavator state, and environment
reset seed. `treatment_id` changes when the reward-distance or reward contract
changes. Environment/reset seeds and policy-sampling seeds are different
namespaces and must never be substituted for each other.

The benchmark may evaluate more than one fixed admissible initial excavator
state per physical map. Each is a separate scenario that shares `map_id`; the
release declares the exact number and clusters uncertainty by map and
`source_group_id`.

## 6. Factor taxonomy

### 6.1 Foundation geometry

| Canonical class | Meaning |
|---|---|
| `osm_connected` | Connected footprint derived from the OSM/source bank |
| `procedural_connected` | Generated connected footprint with varied orientation, aspect, wings, or segmentation |
| `structural_disconnected` | Bearing strips, pads, pillars, or mixed disconnected foundation elements |

Foundation records must additionally include component count, area, perimeter,
aspect ratio, orientation, compactness, holes, bearing strips, and pad/pillar
counts when applicable.

### 6.2 Trench geometry

| Canonical topology | Meaning |
|---|---|
| `straight` | One segment/axis, no junction |
| `segmented_2` | Two end-to-end segments, no branch junction |
| `segmented_3` | Three end-to-end segments, no branch junction |
| `T` | One degree-3 junction |
| `X` | One degree-4 junction |
| `multi_junction` | Two or more declared junctions |
| `disconnected` | Two or more trench components |

Trench records must include segment count, component count, axis count,
junction count, ordered junction degrees, total length, width, global angle,
and relative angles.

Curved trenches are outside v1. A future curve suite must not silently label a
rasterized curve as another `segmented_n` cell.

The v1 runtime can admit at most the topology represented exactly by its frozen
metadata contract. Arbitrary `N`-junction support requires extending the
current three-axis metadata representation and therefore a new compatible
protocol version.

### 6.3 Dump access

Dump constraints are represented by separate fields rather than one overloaded
style name.

Canonical layouts:

- `all_around`;
- `apron`;
- `side_cast`;
- `separated`;
- `irregular_near`;
- `remote_one_side`; and
- `haul_edge`.

Canonical side access:

- `all`;
- `both`;
- `one`; and
- `per_segment`.

Every case records exact accepted cells, components, cells per component,
side balance, legal-free coverage, obstacle-aware path distance, and capacity.
For `one` and `per_segment`, the accepted side or side vector is explicit.

Distance is the shortest traversable 8-connected path from dig-boundary work
cells to accepted dump cells, using cardinal cost `1`, diagonal cost
`sqrt(2)`, and obstacles as blocked cells. It is recorded in tiles and metres.
The dense-reward distance sidecar remains separately hashed because changing
it changes training reward semantics.

Capacity names must distinguish:

- single-layer accepted-area ratio;
- reachable representable capacity ratio;
- remaining representable volume; and
- minimum local cell headroom.

The capacity gate uses the exact accepted mask and proves that complete valid
bucket loads fit under the frozen integer soil representation.

### 6.4 Site constraints

Canonical classes:

- `none`;
- `light`;
- `scattered_objects`;
- `access_road`;
- `gapped_barrier`; and
- `combined`.

The class is only a display label. Quantitative obstacle, non-dump, access,
connectivity, spawn, and workspace measurements are mandatory.

Current B0 panels have no site constraints. They cannot provide evidence for
object, road, wall, or combined-site generalization.

### 6.5 Work amount

Every scenario records exact required excavation volume, dig cells, remaining
volume, and a transport-work proxy. A release defines its volume-band
boundaries in `conditions.jsonl`; labels such as `low` and `normal` are not
global constants.

The transport-work proxy is diagnostic. It must not be used as a reward or
success substitute.

### 6.6 Reset state

`full` is the primary benchmark reset. Partial-reset modes are separately
scored diagnostic suites with exact starting completion and mass.

Results from full and partial resets must never be pooled into one leaderboard
score.

## 7. Conditions and cells

`condition_id` is generated mechanically from normalized fields. For example:

```text
trench.straight.s1.j0.c1
__sidecast.both.dnear.cap3plus
__site.none
__vol.normal
__reset.full
```

Exact numeric values live in the validator-owned audit record. The release owns
a condition registry that defines:

- included categorical values;
- numeric bands and closed/open boundaries;
- expected scenario and source-group counts per split;
- cell evaluation weight; and
- derived `M0`-`M5` tier membership.

No generator may hand-author only a tier label and omit its factor vector.
Every condition belongs to exactly one progression tier. A control condition
may be evaluated in several suites, but its canonical `tier_id` does not
change.

## 8. Progression-tier rules

The first release should use the following intent:

| Tier | Geometry | Dump/access | Site | Admission intent |
|---|---|---|---|---|
| `M0 Nearby Generous` | OSM/simple foundation; straight trench | All-around/apron foundation; large both-side or one-side trench zones | None | Establish full-task anchors |
| `M1 Nearby Geometry` | Connected procedural foundation; two/three end-to-end trench segments | Same generous nearby access as M0 | None | Change geometry only |
| `M2 Nearby Dump Constraints` | Geometry already witnessed in M0/M1 | One-side, irregular, or separated but generous and nearby | None | Change dump access only |
| `M3 Complex Geometry` | Structural/disconnected foundation; T, X, multi-junction, or disconnected trench | Nearby and generous | None | Change complex topology only |
| `M4 Site Access` | Previously witnessed geometry | Nearby and capacity-matched | Objects, road, barrier, then combined | Change site access only |
| `M5 Remote Hauling` | Previously witnessed geometry/site | Medium/far/haul; capacity matched first, tight capacity later | Qualified site classes | Isolate distance before combining distance and capacity |

`D0 Deployment Mix` is a frozen realistic cross-product of already witnessed
conditions. It has no tier membership and introduces no new primitive factor.

### 8.1 Proposed initial M0-M2 condition registry

This is the concrete candidate registry for the first local inspector. It is
specific enough to build without silently inventing new research choices, but
does not become normative until its candidate gallery is approved.

All cells use full resets, no site obstacles, reachable capacity ratio at least
3, and the nearby path envelope below. For the one-layer-deep M0-M2 target,
required excavation volume equals the number of dig cells. The candidate
volume bands are frozen as:

| Family | `low` | `normal` |
|---|---:|---:|
| foundation | 90-160 cell-volumes, inclusive | 161-190 cell-volumes, inclusive |
| trench | 55-110 cell-volumes, inclusive | 111-155 cell-volumes, inclusive |

A candidate outside these closed intervals is rejected rather than silently
rebinned. Variable-depth targets require a new condition registry based on
exact volume rather than cell count.

| Tier | Condition ID | Geometry | Dump access | Volume |
|---|---|---|---|---|
| M0 | `f.osm.all.low` | OSM connected foundation | all-around | low |
| M0 | `f.osm.all.normal` | OSM connected foundation | all-around | normal |
| M0 | `f.osm.apron.near.low` | OSM connected foundation | large nearby apron | low |
| M0 | `f.osm.apron.near.normal` | OSM connected foundation | large nearby apron | normal |
| M0 | `t.straight.both.near.low` | straight trench | large both-side | low |
| M0 | `t.straight.both.near.normal` | straight trench | large both-side | normal |
| M0 | `t.straight.one.near.low` | straight trench | large one-side | low |
| M0 | `t.straight.one.near.normal` | straight trench | large one-side | normal |
| M1 | `f.procedural.all.normal` | connected procedural foundation | all-around | normal |
| M1 | `f.procedural.apron.near.normal` | connected procedural foundation | large nearby apron | normal |
| M1 | `t.segmented2.both.near.normal` | two end-to-end segments | large both-side | normal |
| M1 | `t.segmented3.both.near.normal` | three end-to-end segments | large both-side | normal |
| M2 | `f.osm.one.near.normal` | OSM connected foundation | large one-side | normal |
| M2 | `f.osm.separated.near.normal` | OSM connected foundation | separated nearby | normal |
| M2 | `f.procedural.one.near.normal` | qualified procedural foundation | large one-side | normal |
| M2 | `f.procedural.separated.near.normal` | qualified procedural foundation | separated nearby | normal |
| M2 | `t.straight.irregular_one.near.normal` | qualified straight trench | irregular large one-side | normal |
| M2 | `t.straight.separated.near.normal` | qualified straight trench | separated nearby | normal |
| M2 | `t.segmented2.one.near.normal` | qualified two-segment trench | large one-side | normal |
| M2 | `t.segmented3.one.near.normal` | qualified three-segment trench | large one-side | normal |

M1 geometry candidates reuse the M0 dump contracts. M2 candidates use only
geometry classes that have already produced M0/M1 witnessed cases. T/X and
disconnected trenches do not enter M2; they begin at M3.

Recommended quantitative local-anchor envelope:

- path-distance p50 at most 6 tiles;
- p95 at most 10 tiles;
- maximum at most 12 tiles; and
- reachable representable capacity ratio at least 3.

Recommended local-constraint envelope:

- path-distance p50 at most 10 tiles;
- p95 at most 14 tiles;
- maximum at most 18 tiles; and
- reachable representable capacity ratio at least 3.

These values are release parameters and candidate rejection gates, not reward
terms. M5 defines separate medium/far bands from measured geodesic distance
rather than names such as `d08` alone.

## 9. Suites

The benchmark exposes complementary views rather than one undifferentiated
bank.

### 9.1 Axis panels

Axis panels hold a base source group fixed and vary one factor. They are the
main causal diagnostic for current policies:

- foundation source geometry;
- foundation dump layout/distance;
- trench dump distance;
- trench one-side versus both-side access;
- trench topology;
- site constraint;
- work amount; and
- full versus partial reset.

The current B0 builder already contains 16 unique foundation/trench cells
covering the first five items, but lacks site and reset panels. It becomes a
pilot input, not the final benchmark.

### 9.2 Tier suites

Each `M0`-`M5` suite contains all admitted cells for that progression tier and
is reported separately. A curriculum can train on these tiers in order, mix
them, or ignore the tiering; evaluation remains fixed.

### 9.3 Core leaderboard

The proposed v1 Core score covers full-reset, dynamically witnessed cells from
`M0` through `M4`. `M5 Remote Hauling` has its own leaderboard until enough
methods establish that it is a useful solved-but-difficult regime.

`D0 Deployment Mix` is a separate realistic-mixture score. It must not replace
the balanced per-cell Core result.

### 9.4 Compositional transfer

Source-disjoint maps within familiar cells test new geometry sources, not
unseen combinations. A separate compositional suite therefore holds out entire
factor combinations from every model-selection-visible split (`public_train`
and `public_dev`) while keeping every primitive factor represented in public
training.

Example: public training includes procedural-foundation x all-around and
OSM-foundation x separated, while compositional test contains
procedural-foundation x separated. The split registry must prove:

- the held-out combination is absent from public training and development;
- each constituent factor has public-training support;
- source groups remain disjoint;
- no condition is simultaneously in within-cell and compositional test; and
- its score is never pooled with ordinary within-cell source generalization.

The website displays a train-support x test-combination matrix and reports
compositional success separately.

If compositional model selection is needed, a separately frozen
`compositional_dev` registry uses different held-out combinations from
`compositional_test`.

### 9.5 Unverified Challenge

Statically valid maps without a dynamic completion witness may be published in
an explicitly unranked Challenge suite. Failure on an unverified map is not
evidence of policy failure.

## 10. Split and diversity standard

The local pilot uses the already motivated 256-source training diversity while
keeping evaluation small enough to iterate:

| Split | Unique source groups per cell | Visibility |
|---|---:|---|
| `public_train` | 256 | Maps, seeds, metadata, and examples public |
| `public_dev` | 32 | Maps and per-map diagnostics public |
| `sealed_pilot` | 64 | Held-out local maps; no public-server claim |

A publishable private-test size is selected only after the pilot records
evaluation cost and preregisters a target cell-success confidence width and
minimum paired effect size. The likely range is 128-256 source groups per cell;
it is not frozen by intuition alone. The release reports total episodes, fixed
initial states per map, maximum transitions, and measured wall-clock cost.

For planning only, the approximate worst-case 95% Wilson half-width at 50%
success is 11.9 percentage points for 64 independent source groups, 8.6 for
128, and 6.1 for 256. The final choice must also account for clustered initial
states and paired variants. The 256 public-training count is a diversity
hypothesis from current generalization failures, not a statistical test-size
argument.

Paired counterfactual conditions share source groups. A source geometry
rendered as both-side and one-side is one source group in each cell, not two
independent sources.

A release is invalid unless:

1. source groups are disjoint across all splits;
2. geometry, map, and scenario hashes are disjoint across splits;
3. all counterfactual variants from one source group remain in one split;
4. OSM derivatives are grouped by underlying footprint, not transform or
   filename;
5. procedural seed namespaces are disjoint, while seed disjointness alone is
   not accepted as identity proof;
6. exact counts and frozen evaluation weights are declared per condition;
7. repeated training slots expose their identity and multiplicity;
8. centered/dihedral geometry similarity is audited within and across splits;
9. the release declares and enforces its family-specific near-duplicate
   threshold;
10. diversity/support summaries include more than source count: nearest-
    neighbour similarity, geometry descriptors, and factor-distribution
    coverage; and
11. private arrays and seeds remain private.

The website publishes within/cross-split nearest-neighbour distributions and
all quarantined similarities. It must not summarize split safety as a single
unchecked boolean.

## 11. Feasibility and integrity

Every scenario receives one of three plainly named badges:

| Badge | Meaning |
|---|---|
| `Format-valid` | Files, shapes, ranges, dtypes, metadata, loader, and hashes pass |
| `Static-valid` | Exact dump, capacity, footprint, spawn, access, workspace, and split checks pass |
| `Witnessed` | A legal, mass-conserving completion trace replays from the exact initial state under the frozen protocol and horizon |

Only `Witnessed` scenarios count in a ranked suite.

For a publication-ranked suite, the witness must come from constructive
generation or a frozen method-neutral planner. The validator replays the
certificate from the exact initial state. A learned-policy witness may qualify
a map for the Reference Pilot or Unverified Challenge, but cannot by itself
select the primary test distribution: that would favor the reachable subset of
one method. A future multi-method admission rule must be preregistered before
candidate generation if it is used instead.

The witness type is disclosed, but sealed-test traces remain private.

A witness proves feasibility, not map ease. It cannot be used as a submitted
policy's score or leaked as a sealed-test demonstration.

Static validation includes:

- exact-loader contract, shape, dtype, finite, and layer consistency;
- target/obstacle/road/dump-mask disjointness and semantics;
- exact accepted-mask capacity and integer headroom;
- traversable access and reachable accepted dump components;
- valid spawn, base-centre connectivity, and pre/post-dig workspace coverage;
- declared geometry components, axes, segments, junctions, and degrees;
- mass balance;
- split, hash, source, pair, and near-duplicate audits; and
- deterministic rejection reasons.

If a released map later fails integrity or feasibility, it is removed only in
a new scored release. Existing results remain archived with the original
release status.

## 12. Distribution and curriculum audit

Every release automatically exports:

- exact scenario, map, source-group, and slot counts;
- weights and effective sample size per condition;
- geometry x dump, dump x site, family x reset, and tier x family heatmaps;
- distributions of dig volume, distance p50/p95/max, capacity, dump
  components, side balance, angles, aspect ratio, obstacle fraction, legal
  dump coverage, and transport work;
- paired-group completeness;
- exact and near-duplicate reports;
- train/development/sealed support overlap without revealing sealed identities;
- generator rejection-reason histograms;
- format/static/witness badge counts; and
- warnings for empty cells, too few unique sources, support gaps, or
  repeated-slot dominance.

The curriculum inspector accepts either a generated bank manifest or a small
declared mixture JSON and compares it to a chosen benchmark suite. It shows:

- intended versus materialized cell probability;
- expected exposures per cell;
- unique sources and repeat concentration;
- missing and overrepresented cells;
- progression-tier coverage;
- train-to-evaluation support gaps; and
- current-policy success/completion by cell, when receipts are supplied.

Training balance is not required to be uniform: curricula intentionally change
weights. The tool reports the declared target, actual exposure, and mismatch
instead of silently deciding that all skew is bad.

### 12.1 Difficulty warnings

Static descriptors are called complexity or work proxies, not empirical
difficulty.

Empirical difficulty is calibrated from a frozen panel of hashed reference
receipts and displayed as a policy x cell matrix. A release names:

- a uniform valid-action diagnostic baseline;
- one foundation specialist;
- one trench specialist;
- one joint generalist; and
- any deterministic planner/heuristic that is independently evaluated.

The constructive feasibility witness is not a reference policy and cannot
calibrate difficulty. A learned-method warning requires at least two applicable
non-witness reference receipts from independent training runs; otherwise the
website says
`insufficient reference evidence`.

The site may flag:

- **anchor-saturated**: all qualified references exceed 95% success;
- **currently unsupported**: all references have zero success despite
  `Witnessed`
  feasibility;
- **curriculum cliff**: most new exposure is on cells far below the current
  policy's prerequisite-cell performance; and
- **non-monotonic tier**: a later named tier is empirically easier than an
  earlier one for the selected policy.

These are review warnings, not automatic map deletion or promotion rules.

## 13. Evaluation protocol

The primary v1 protocol freezes:

- one tracked excavator;
- the official observation schema and action values `0..7`;
- one untouched full-task reset from the scenario's serialized initial agent
  and soil state;
- a separately named environment reset seed and policy seed;
- at most 450 calls to `step_no_reset`, numbered 1 through 450;
- exact manifest enumeration;
- deterministic policy action selection;
- `exact_visible_dump_v1` completion;
- the code predicate `absolute_completion >= 1.0 - 1e-6`;
- stop on the first transition satisfying that predicate;
- success on call 450 if that call first satisfies the predicate, otherwise
  horizon censoring after call 450;
- success or horizon as the only normal episode endings; and
- zero integrity failures.

The local pilot pins one canonical admissible initial agent state per map. A
publishable release should evaluate four fixed admissible states per map unless
the measured cost review justifies another frozen number. Results and
uncertainty cluster those states by `map_id` and `source_group_id`; changing
their count creates a new score version.

Evaluator conformance tests must include early success, a task first completed
on call 449, a task first completed on call 450, and an incomplete task after
call 450. They must prove that terminal state/diagnostics are retained, no
automatic reset occurs, and no call 451 is executed.

Reward and online training return are never benchmark ranking metrics.

### 13.1 Primary metrics

The release freezes a hierarchy of semantic weights rather than allowing cell
proliferation to change the score. Core uses equal family weight, equal
progression-tier weight within family, and equal condition weight within each
family x tier group:

```text
Balanced Condition Success
  = mean_family(
      mean_tier_in_family(
        mean_condition_in_family_tier(success_rate)
      )
    )
```

All resulting condition weights are materialized in `conditions.jsonl`.
Adding, splitting, or removing a scored condition creates a new score version.
The site also publishes the raw unweighted cell macro and a sensitivity table,
but neither silently replaces the frozen primary score.

Submission eligibility requires:

- 100% expected scenario coverage;
- one valid action at every evaluated step;
- matching map/scenario slots and hashes;
- no non-finite values;
- no mass, target, obstacle, or completion-contract integrity failure; and
- complete evaluation receipts.

The leaderboard displays, at minimum:

- Balanced Condition Success;
- raw unweighted cell macro;
- foundation and trench macro success;
- every cell success;
- worst-quartile cell success;
- minimum family success;
- full success/completion heatmaps by declared factor and tier; and
- evaluated, valid, censored, and expected counts.

Default tie-break order:

1. worst-quartile cell success;
2. minimum of foundation and trench macro success;
3. mean terminal absolute completion on failures; and
4. median steps among successful episodes.

There is no reward-based tie-break.

### 13.2 Diagnostic metrics

Per scenario:

- success and termination reason;
- episode steps;
- terminal absolute, dig-edge, dig-inner, dump-volume, dump-purity, and
  unloaded completion;
- action and physical-effect counts;
- mass residual;
- target/obstacle mutation;
- non-finite state;
- termination/completion disagreement; and
- slot/hash mismatch.

Public-development results may expose per-map diagnostics and trajectories.
Sealed-test results expose only family, cell, tier, and declared-axis
aggregates.

### 13.3 Uncertainty

- For the one-state-per-map pilot, report a 95% Wilson interval for each cell.
- For multi-state releases, report source-group-clustered intervals rather than
  treating initial-state variants as independent Bernoulli trials.
- Use a fixed-seed, map/source-group-clustered bootstrap for Balanced Condition
  Success, family scores, worst-quartile success, and paired-condition
  differences.
- Treat initial-state variants and paired map variants from the same source
  geometry as one bootstrap cluster.

### 13.4 Stochastic robustness

A secondary evaluator-owned robustness pass uses four fixed hidden policy seeds
per scenario. It reports mean and spread per cell and never best-of-N.

Deterministic and sampled results have separate tabs and are never pooled.

## 14. Submission and result contracts

### 14.1 Native reference-pilot track

The first public upload path is deliberately narrow:

```text
submission.tar.zst
  submission.json
  weights.msgpack or weights.npz
  MODEL_CARD.md
```

`submission.json` declares:

- submission schema and benchmark release;
- method and team;
- observation/action/completion contract digests;
- allowlisted official architecture name and declarative configuration;
- weight filename, dtype/shape tree, and SHA-256;
- deterministic action rule;
- Terra and terra-baselines training commits;
- training-bank release and data track;
- training seeds and compute summary;
- code URL/commit when public;
- license; and
- model-card digest.

The server constructs an allowlisted model, validates every named weight, and
then loads non-executable bytes. Pickled Python objects are not accepted.

This track is enough to publish map data and comparable official baselines, but
it is labeled `Reference Pilot`; it is not an architecture-neutral public
method leaderboard. An evaluation-only planner that does not instantiate the
official model is run directly by the benchmark owner and receives the same
receipt without pretending to be a native-weight submission.

A general public method leaderboard requires an architecture-neutral adapter
or sandboxed OCI evaluator and becomes a separately versioned publication
track. It does not block the initial local inspector or static data release.

### 14.2 Data tracks

Results must declare one of:

- `closed_data`: only the released public-train scenario bytes and official
  reset-state sampler;
- `open_data`: any training maps, generator, or additional data; or
- `evaluation_only`: no training claim, for planners and heuristics.

Tracks have separate leaderboard filters. Open-data results never silently
compete as closed-data results.

For v1 closed data, public-development scenarios are forbidden as gradient or
training data but are explicitly allowed for hyperparameter, checkpoint, and
model selection. Sealed scenarios are forbidden for both training and
selection. New procedural/OSM maps, raster rotations or reflections that create
new map identities, and generator-derived variants move a result to
`open_data`. Observation-only noise, dropout, masking, and other
transformations that do not add map content are allowed but must be declared
in the model card.

### 14.3 Evaluation receipt

The evaluator seals:

- submission/policy digest;
- Terra, terra-baselines, evaluator, generator, and validator commits;
- benchmark root hash and split-manifest hash;
- completion, observation, action, dynamics, and horizon contracts;
- deterministic and sampled seeds;
- expected/evaluated/valid counts;
- integrity results;
- aggregate and per-cell metrics;
- hardware/runtime metadata; and
- receipt digest.

The public receipt omits private scenario IDs, seeds, outcomes, traces, and
arrays.

## 15. Sealed-test protection and versioning

- Public-development evaluation is unlimited and includes per-map diagnostics.
- A team selects one final artifact digest using public development, before
  seeing any sealed result.
- The official sealed score is one-shot per team and release. A rerun is
  allowed only when the evaluator, not the model, was invalid.
- Any additional adaptive server board is labeled `adaptive development`, has
  a finite release-wide query budget, and is never called an untouched test.
- Identical artifact digests reuse the cached receipt.
- Sealed feedback for the final artifact is released with the leaderboard; no
  per-map identities, seeds, outcomes, or trajectories are exposed.
- The release publishes a cryptographic commitment to the sealed bank before
  accepting submissions and reveals it when the release is retired.

Any change to scored map bytes, cells, weights, Terra dynamics, completion,
observation, action, horizon, or scoring creates a new major leaderboard
version. Documentation, rendering, or website fixes may be patches only when
they provably preserve evaluation bytes.

Old releases and leaderboards remain accessible.

## 16. Website specification

The website is generated from versioned JSON, thumbnails, and optional result
receipts. It can be hosted as static files. Sealed evaluation runs elsewhere
and exports public-safe receipts; the website never executes submitted models.

The first deliverable is one local `index.html`, not a web service. It shows:

- the condition x policy heatmap with worst cells first;
- ordered `M0` through `M5` review;
- click-through stratified galleries;
- distance, capacity, uniqueness, format/static/witness warnings; and
- training exposure versus benchmark exposure.

The separate pages below are the publication phase. Browser uploads, persistent
review state, and submission handling do not block the first inspector.

### 16.1 Overview

Show:

- release and protocol versions;
- ranked suites, cells, splits, and scenario counts;
- format/static/witness status;
- Terra and evaluator revisions;
- download links, checksum, license, changelog, and sealed commitment; and
- clear labels for Core, Remote Hauling, Deployment Mix, and Unverified
  Challenge.

### 16.2 Map Explorer

Filters:

- family and `M0`-`M5` tier;
- foundation source/structure;
- trench topology, segments, components, junctions, and junction degree;
- dump layout, side access, components, distance, and capacity;
- site class and quantitative obstacle/access ranges;
- work volume;
- reset mode;
- feasibility badge; and
- public split.

Each card shows:

- a colored composite;
- target, occupancy, dumpability, initial-soil, and distance layer toggles;
- complete factor vector and exact measurements;
- capacity, distance, and format/static/witness badges;
- source/generator provenance and nearest-neighbour audit;
- cell/tier membership; and
- public submission outcomes when available.

Recommended colors:

- orange: excavation target;
- green: exact legal dump mask;
- grey: traversable non-dump surface/road;
- black: obstacle;
- brown/blue diverging overlay: initial positive/negative soil.

Sealed-test pages show only cell definitions, aggregate distributions, and
separate public examples generated from public source groups.

### 16.3 Distribution Dashboard

Show exact counts and weights, not only normalized bars:

- factor marginals and joint heatmaps;
- train/development/sealed aggregate histograms;
- source-group and effective-sample-size counts;
- paired-condition coverage;
- distance, capacity, volume, topology, obstacle, and dump-coverage
  distributions;
- near-duplicate and split-leakage audit;
- rejection reasons; and
- missing, underrepresented, or support-mismatched cells.

The dashboard can compare any uploaded training manifest with the frozen
benchmark target.

### 16.4 Cell Page

Show:

- normative condition definition and numeric bands;
- tier and suite membership;
- split counts and source-group counts;
- feasibility requirements and witness coverage;
- fixed stratified public gallery;
- distributions and balance warnings;
- reference-policy performance with uncertainty; and
- common failure modes.

### 16.5 Curriculum Inspector

Accept a bank manifest or mixture JSON. Show:

- progression-level exposure;
- expected repeats per identity/source;
- declared versus actual weights;
- missing and overrepresented conditions;
- benchmark support coverage;
- policy success/completion overlaid on training exposure; and
- likely curriculum cliffs without automatically changing the curriculum.

This is the main interface for deciding whether a proposed training bank is too
hard or unbalanced.

### 16.6 Leaderboard and Compare

Leaderboard columns:

- Balanced Condition Success with confidence interval;
- worst-quartile success;
- foundation and trench macros;
- Remote Hauling and Deployment Mix scores when present;
- deterministic/sampled mode;
- data track; and
- integrity badge.

Expanding a row shows the full cell/tier heatmap. The Compare page shows two
submissions' paired cell delta heatmap, factor marginals, clustered uncertainty,
and worst-regressed/worst-improved public cells.

### 16.7 Submission and provenance

Show the public model card, artifact hash, declared training data/compute,
architecture contract, code/version information, evaluation receipt, and all
applicable benchmark versions.

## 17. Human map-review workflow

Before freezing a release:

1. Generate a large candidate pool by source group.
2. Run format/static validation and duplicate/leakage audits.
3. Export a local static review site and fixed galleries.
4. Review `M0 Nearby Generous` first, then change one axis at a time.
5. Inspect distribution tails and all rejected/quarantined candidates.
6. Reject semantic errors; record aesthetic preferences separately.
7. Produce and replay exact-initial-state witnesses.
8. Select balanced source groups without moving variants across splits.
9. Freeze public train/development and commit the sealed split.
10. Evaluate frozen reference methods and publish the first receipts.

The publication site may later add a "review queue" mode with:

- paired base/counterfactual maps side by side;
- next/previous cell and keyboard navigation;
- approve, reject, quarantine, and note export;
- deterministic ordering and stable URLs; and
- an outlier queue for extreme distance, capacity, volume, obstacle, or
  similarity values.

Human review decisions are exported as data. They are not stored only in a
browser or screenshot.

## 18. Minimal implementation plan

The immediate implementation is a local review bundle with one golden-path
command:

```bash
python tools/build_map_review.py \
  --conditions benchmark/m0_m2_v1.json \
  --out .artifacts/terramap_bench_v1

xdg-open .artifacts/terramap_bench_v1/site/index.html
```

Required outputs:

```text
.artifacts/terramap_bench_v1/
  manifest.jsonl
  audit.jsonl
  summary.json
  review_decisions.jsonl
  thumbnails/
  site/index.html
  policy_results.jsonl       # optional input/output when receipts are supplied
```

The script must fail before exporting the site when a format, identity, split,
capacity, or declared-condition check fails. It is a thin orchestration entry
point over two immediate tools, not a general framework:

1. `build_benchmark_bank.py`
   - wraps existing generators;
   - writes exact-loader data and normalized scenario records.
2. `validate_benchmark_bank.py`
   - recomputes `audit.jsonl`;
   - validates format, semantics, capacity, splits, similarity, and checksums.
3. `export_benchmark_site.py`
   - creates static JSON, thumbnails, galleries, dashboards, and result pages.

S1-S2 stop here. Dynamic witness replay, policy evaluation, safe model
submission, uncertainty, sealed-server operation, and leaderboard publication
are S3-S5 work. Later tools are:

- `replay_benchmark_witnesses.py`;
- `evaluate_benchmark.py`; and
- the publication-mode extension of `export_benchmark_site.py`.

Reuse current capacity validation, exact-dataset validation, B0 generation,
fixed-bank evaluation, receipts, and trajectory replay. Do not create a
parallel environment implementation.

Repository ownership stays narrow:

- Terra owns conditions, generators, exact map/scenario manifests, validation,
  and public map release artifacts.
- terra-baselines owns policy adapters, evaluation, uncertainty, and receipts.
- the static site consumes exported data from both and owns no scientific
  truth.

Suggested delivery gates:

| Gate | Deliverable | Pass condition |
|---|---|---|
| `S0 Spec` | This document and reviewed tier/cell registry | All normative/open choices resolved |
| `S1 Schema` | Manifest v2 normalizer and validator | Existing B0 bank round-trips without semantic loss |
| `S2 Review` | Candidate pools and local static site | Counts, distributions, pairs, and maps are human-reviewed |
| `S3 Feasibility` | Witness store and replay | Every ranked scenario replays from its exact initial state within horizon |
| `S4 Evaluation` | Native bundle, evaluator, uncertainty, receipt | Reference submissions reproduce byte-identical receipts |
| `S5 Publication` | Versioned data, docs, static site, sealed commitment | Downloads and leaderboard independently verifiable |

## 19. Explicit non-goals for v1

- A generic adaptive curriculum framework.
- PLR, ALP-GMM, PAIRED, ACCEL, or learned adversarial generation.
- Automatic promotion/demotion policy.
- A single learned scalar map-difficulty predictor.
- Curved trenches.
- Arbitrary `N`-axis runtime support without a metadata-contract change.
- A database-backed web application.
- Arbitrary executable Python or container submissions.
- Mixing reward-curriculum claims with map-curriculum claims.
- Calling static validity proof of dynamic feasibility.

## 20. Decisions to review before implementation

Recommended defaults:

1. Public name: `TerraMap-Bench`.
2. Retain `M0`-`M5` with the single-axis descriptive names in Section 2, and
   keep `D0 Deployment Mix` separate.
3. Build the local pilot with 256 public-train, 32 public-development, and 64
   sealed source groups per admitted cell; select publication test size from a
   preregistered precision/cost target.
4. Rank Core by deterministic Balanced Condition Success; expose every cell and
   worst-quartile performance.
5. Keep `M5 Remote Hauling` separately ranked for v1.
6. Require exact-initial-state replay witnesses for ranked maps.
7. Label the native safe-weight board a Reference Pilot; require an
   architecture-neutral adapter before claiming a general method leaderboard.
8. Generate a static local/public website from the same manifests and receipts.

Still to decide:

- the M3-M5 and compositional condition registries after candidate galleries
  are reviewed;
- the family-specific cross-split similarity thresholds;
- the publication test confidence/effect target and number of fixed initial
  states per map;
- which current official architecture configurations are allowlisted;
- benchmark data and submission licenses; and
- public hosting and sealed-evaluator ownership.

These decisions do not require changing Terra dynamics or launching PPO. The
next step after spec approval is `S1 Schema`, followed by a local review site
over candidate maps before any benchmark release is frozen.
