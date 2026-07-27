# TerraMap-Bench specification

Status: design-frozen v0.3; S1 implementation pending

Last accepted-review update: 2026-07-27

This is the canonical current map-benchmark and map-curriculum design. Section
20 is the append-only decision log for reviewer comments: an accepted or
modified comment must be recorded there in the same change that updates the
normative sections. `TRAINING_TASKS.md` owns execution evidence and links back
to this document; chat history is never the only record of an accepted design
change.

This document specifies a public, versioned benchmark for Terra excavation
maps. It covers the map format, difficulty taxonomy, admission graph, training
mixtures, feasibility gates, train/evaluation splits, policy submissions,
metrics, and a website for inspecting maps, distributions, and policy
failures.

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

### 2.1 Factorized conditions, three display depths

A map is described by its factor vector. It is never defined only as "easy",
"medium", or "hard".

The benchmark uses three human-readable display depths, not an intrinsic total
order over maps:

| Display depth | Meaning |
|---|---|
| `Anchor` | A full-reset condition that has passed its source-disjoint promotion gate |
| `One-axis` | Exactly one geometry, work, dump, or site factor changes from a passed direct parent |
| `Composed` | Two or more individually admitted factors are combined |

Display depth is organizational metadata. Conditions are admitted
independently through an explicit `requires` list. Sibling one-axis conditions
are unordered, and no stage-wide pass over all siblings is required.

The deployment mixture and forced-rehandling Challenge are separate suites,
not additional depths. A scalar depth must never drive Terra's per-environment
`curriculum.level` controller or silently select reward, horizon, or reset
semantics.

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
    promotion/
      <frozen source-disjoint promotion datasets>
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
- map shape `64 x 64`, `edge_length_m = 36.5714285714`,
  `tile_size_m = 0.571428571428125`, coordinate/frame convention, and dtypes;
- tracked-excavator dimensions and their derived `7 x 11` tile footprint,
  `move_tiles = 5`, `dig_radius_tiles = 5`, `dig_depth = 1`, twelve base
  orientations, twelve cabin orientations, and the exact workspace rule;
- map-layer meanings;
- exact dump, mass, dynamics, action, observation, and completion contracts;
- `max_steps_in_episode = 450`, `rewards_type = DENSE`,
  `apply_trench_rewards = false`, the corrected dense reward hash, and the
  deterministic evaluation action rule;
- split counts and cell weights;
- the uniform-over-materialized-slots sampler contract, common slot count,
  per-condition multiplicities, and slot-to-scenario mapping;
- similarity/leakage policy; and
- the root checksum.

Changing a frozen item creates a new scored benchmark version.

The validator must assert
`tile_size_m == edge_length_m / edge_length_px`. The live physical scale is
authoritative. No benchmark repair may change `edge_length_m` merely to match
stale metadata.

## 5. Canonical scenario schema

The scenario manifest contains authored identity, provenance, declared
condition inputs, and immutable layer hashes. Generator-authored metadata is
not accepted as proof of capacity, separation, connectivity, or feasibility.
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
split: public_train | promotion | public_dev | sealed_pilot | private_test | compositional_dev | compositional_test

family: foundation | trench
display_depth: anchor | one_axis | composed
requires: [...]
suite_ids: [...]

layers:
  target_sha256: ...
  occupancy_sha256: ...
  dumpability_sha256: ...
  initial_soil_sha256: ...
  metadata_sha256: ...
  shape: [64, 64]
  edge_length_m: 36.5714285714
  tile_size_m: 0.571428571428125

generator:
  name: ...
  revision: ...
  config_sha256: ...
  seed: ...
  attempt: ...

declared_condition:
  source_family: ...
  geometry_class: ...
  topology: ...
  dump_layout: ...
  side_access: ...
  dig_dump_separation_band:
    metric: p50_tiles
    lower_inclusive: ...
    upper_inclusive: ...
  capacity_band:
    metric: single_layer_area_ratio
    lower_inclusive: ...
    upper_inclusive: ...
  site_class: ...
  volume_band: ...
  reset_mode: ...

initial_condition:
  environment_reset_seed: ...
  initial_agent_state:
    schema: terra_agent_state_v1
    width: 7
    height: 11
    max_agents: 4
    num_agents: 1
    current_agent: 0
    moving_dumped_dirt: false
    agent_active: [true, false, false, false]
    agent_states:
      pos_base: [[...], [...], [...], [...]]
      angle_base: [...]
      angle_cabin: [...]
      wheel_angle: [...]
      loaded: [...]
      agent_type: [...]
      action_type: [...]
      shovel_lifted: [...]
      carry_baseline_potential: [...]
      carry_potential_after_lift: [...]
  initial_agent_state_sha256: ...
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
  dig_dump_separation_tiles: {p50: ..., p95: ..., max: ...}
  dig_dump_separation_m: {p50: ..., p95: ..., max: ...}
  any_direct_transfer_pose_exists_initial: ...
  direct_service_coverage_initial: ...
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
  initial_workspace_coverage: ...
  admissible_pose_count_initial: ...

work:
  required_volume: ...
  separation_work_proxy: ...
  forced_rehandling: ...

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
  witness_steps: ...
  witness_step_fraction: ...
  witness_initial_scenario_sha256: ...
  witness_terminal_semantic_state_sha256: ...
  terminal_workspace_coverage: ...
  terminal_admissible_pose_count: ...
  minimum_admissible_pose_count_during_witness: ...
```

`map_id` changes when a physical map layer changes. `scenario_id` additionally
includes initial soil, the explicit initial excavator state, and environment
reset seed. S1 exposes an explicit admissible `initial_agent_state` input on
the single and batched reset paths; a seed plus commit is not a portable
serialized state.
`initial_agent_state_sha256` covers the canonical byte encoding of every
reset-consumed `Agent` and `AgentState` field, including the active mask,
current-agent index, fixed four-slot state tree, footprint dimensions,
moving-dirt flag, wheel/action/shovel state, and both carry-potential caches.
Inactive slots use declared canonical bytes rather than being omitted. The v1
full reset records zero-valued caches rather than silently deriving them.
Adding or removing a state field requires a new state schema and protocol
hash.

The state codec is part of S1 rather than an implied host serialization. It
hashes fields in declared schema order; prefixes every array with its field
path, dtype, rank, and shape; canonicalizes numeric leaves to declared
little-endian integer or IEEE-754 representations; and hashes C-order bytes.
The release records the codec revision and test vectors. Witness
`initial_scenario` hashes cover the scenario record and referenced bytes.
Witness terminal semantic-state hashes additionally cover the frozen RNG key,
all mutable `GridWorld` arrays/scalars, the complete `Agent` tree, and
`env_steps`; the protocol hash owns immutable `EnvConfig`.

The implemented `terra_agent_state_v1` codec asserts exact coverage of the
live `Agent`/`AgentState` field trees, preserves the live `int8[4]` active
mask, and hashes all four slots. Explicit reset bypasses `Agent.new`, cache
derivation, current-agent randomization, and reset-RNG consumption. The
canonical one-agent test vector digest is
`debd22b6ff2c8b31d263ceb843e524d5bf9ae1ffe186e26291f1e5ec3d18fb1a`.

`treatment_id` changes when the reward-distance or reward contract changes.
Environment/reset seeds and policy-sampling seeds are different namespaces and
must never be substituted for each other.

The benchmark may evaluate more than one fixed admissible initial excavator
state per physical map. Each is a separate scenario that shares `map_id`; the
release declares the exact number and clusters uncertainty by map and
`source_group_id`.

## 6. Factor taxonomy

### 6.1 Foundation geometry

| Field/class | Meaning |
|---|---|
| `source_family=osm` | Provenance: connected footprint derived from the OSM/source bank |
| `source_family=procedural` | Provenance: generated connected footprint |
| `connected` | One connected excavation footprint |
| `structural_disconnected` | Bearing strips, pads, pillars, or mixed disconnected foundation elements |

Foundation records must additionally include component count, area, perimeter,
aspect ratio, orientation, compactness, holes, bearing strips, and pad/pillar
counts when applicable.

OSM versus procedural is a source/coverage slice, not an intrinsic difficulty
axis: Terra consumes the resulting raster without a source tag. The pilot
keeps matched OSM and procedural all-around cells separate because the current
generators are not distribution-matched and the completed B0-GEO-F run
qualified neither source. They may be pooled inside a future condition only
after matched volume/compactness support and separate source-slice gates pass.

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
- `remote_edge`.

Canonical side access:

- `all`;
- `both`;
- `one`; and
- `per_segment`.

Every case records exact accepted cells, components, cells per component,
side balance, legal-free coverage, obstacle-aware separation, direct-service
coverage, and capacity. For `one` and `per_segment`, the accepted side or side
vector is explicit.

The existing `d02`, `d04`, `d06`, and `d08` quantity is the shortest
traversable 8-connected path from dig-boundary cells to accepted dump cells,
using cardinal cost `1`, diagonal cost `sqrt(2)`, and obstacles as blocked
cells. It is recorded in tiles and metres and is named **dig/dump
separation**, not transport or hauling distance.

The live tracked-excavator **radial envelope** is an annulus from `6.375` to
`11.375` tiles (`3.642857` to `6.5` metres) around a base centre. Each cabin
cone is `+/-30` degrees (60 degrees total). This envelope is not the exact
action mask: the validator must call or bit-match Terra's cartesian body
exclusion, inner-tooth cleanup, obstacle veto, and all dig/dump filters for
each base/cabin orientation.

A base pose is `(row, column, angle_base)`. It is reachable only if Terra's
frozen empty-excavator forward, backward, and collision-checked base-rotation
transitions can reach it from the explicit initial pose. An 8-connected
base-centre component is not accepted as a substitute because movement is in
five-tile rounded steps.

A direct transfer is an exact runtime replay from one fixed reachable base
pose: a legal dig of positive target volume, cabin-only rotations if needed,
then a legal complete-load dump into the exact accepted mask with sufficient
headroom. The replay includes the dig-induced hole dilation, traversability,
load, integer-capacity, and contained-pile updates. Direct-service coverage is
the fraction of required excavation volume for which at least one such
sequence exists in the initial scenario. The boolean
`any_direct_transfer_pose_exists_initial` means that fraction is nonzero; it
does not claim that one pose serves the whole map.

The static validator reports this boolean and fraction plus initial exact-cone
workspace coverage and reachable admissible-pose count. A `d02`-`d08` pair can
often have direct service, so separation alone is not transport difficulty.
The tracked excavator cannot move or rotate its base while loaded.

There is no generator-defined generic "post-dig" state. After S3, replay of the
canonical witness defines an exact terminal state and reports workspace
coverage and reachable admissible-pose count there plus the minimum count
encountered during the trace. Terminal direct-service coverage is undefined
because no target volume remains. Witness-derived values are keyed by the
terminal semantic-state hash and cannot be used as S1 static-validation
inputs.

Separation remains a useful policy diagnostic. The conservative
`forced_rehandling` label is allowed only when initial direct-service coverage
is exactly zero and an action-level witness proves completion within the
frozen horizon. Coverage strictly between zero and one is labeled
`mixed_service`; a witness that happens to rehandle soil does not prove that
rehandling was necessary.

Minimum relay hops remains a possible later witness-derived diagnostic. It is
not a v1 S1 field or Static-valid gate until its temporary-spoil nodes,
headroom, transition, and cost semantics are separately frozen.

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
volume, and a separation/rehandling work proxy. Global family-wide `low` and
`normal` bands are forbidden: they created structurally empty segmented-trench
cells.

Dump, capacity, side, site, and separation counterfactuals reuse the exact dig
raster and therefore exact work volume. Geometry-source and topology
comparisons use matched overlapping volume distributions. Volume becomes a
separate later one-axis condition only after a supported matched pair and a
450-step witness exist.

Work proxies are diagnostic. They must not be used as reward or success
substitutes.

### 6.6 Reset state

`full` is the primary benchmark reset. Partial-reset modes are separately
scored diagnostic suites with exact starting completion and mass.

Results from full and partial resets must never be pooled into one leaderboard
score.

## 7. Conditions and cells

`condition_id` is generated mechanically from normalized fields. For example:

```text
trench.straight.c1.j0
__source.procedural
__side_cast.both.sep02.slcap03_04.v68_77
__site.none
__reset.full
```

Exact numeric values live in the validator-owned audit record. The release owns
a condition registry that defines:

- included categorical values;
- numeric bands and closed/open boundaries;
- expected scenario and source-group counts per split;
- cell evaluation weight; and
- display depth and explicit prerequisite condition IDs.

No generator may hand-author only a display depth and omit its factor vector.
A control condition may be evaluated in several suites, but its physical
condition identity does not change.

The initial pilot uses closed single-layer accepted-area capacity bands:

| Token | Validator metric and closed interval |
|---|---|
| `slcap03_04` | `single_layer_area_ratio in [3, 4]` |
| `slcap07_10` | `single_layer_area_ratio in [7, 10]` |
| `slcap20_45` | `single_layer_area_ratio in [20, 45]` |

`sep02` means achieved dig/dump-separation p50 in `[1.25, 2.75]` tiles;
`sep00_02` means `[0, 2]`. Capacity tokens always name their metric: the
historical B0 value `3.25` is the single-layer area ratio and must not be
silently substituted for reachable representable capacity.

OSM/procedural provenance, orientation, and aspect remain audit/slicing fields
unless a matched experiment establishes a residual source condition. Until
then, the pilot keeps OSM/procedural source cells separately gated rather than
pooling their `6/8` result.

Legacy generator metadata value `broad_side_cast` is retained only in raw
provenance. The S1 normalizer maps it to the one canonical taxonomy token
`side_cast`; no benchmark condition ID uses `sidecast` or
`broad_side_cast`.

## 8. Admission graph and map-curriculum rules

The curriculum has four measured controls:

1. work volume;
2. geometry/topology;
3. dump access, capacity, separation, and rehandling burden; and
4. site access.

`Anchor`, `One-axis`, and `Composed` are display depths over an admission graph,
not three monolithic Terra levels. Each condition has a small literal
`requires` list. A one-axis condition changes exactly one control relative to
its direct parent. A composed condition becomes eligible only when every
primitive parent has passed.

Sibling axes may be witnessed independently. Their policies or optimizer
states are not merged: a generalist still adds admitted conditions
sequentially to one cumulative recorded lineage.

### 8.1 Frozen map-curriculum protocol

Every matched map-curriculum run uses:

- untouched full resets with `env_steps == 0`;
- `max_steps_in_episode = 450`;
- `rewards_type = DENSE`;
- `apply_trench_rewards = false`;
- the same `corrected_dense_v1` coefficients and reward hash;
- the same action, observation, dynamics, exact-dump, and completion contract;
  and
- one immutable materialized map level per recorded run.

Map, reward, reset, and architecture curricula are separate named treatments.
Partial resets are excluded from this comparison. PR0 later compares 100% full
resets against a declared partial-reset mixture after the map sampler is
selected.

### 8.2 Promotion, retention, and recovery

Promotion uses a separate frozen source-disjoint bank every 100 updates:

- each new condition and its direct parent require at least `6/8` successes at
  two consecutive checkpoints;
- a four-cell family qualification additionally requires at least `26/32` in
  that family at both checkpoints;
- every separately gated source slice must pass rather than being hidden by a
  pooled cell;
- every previously mastered condition and qualified family panel must pass the
  count-based retention rule below; and
- every gate requires zero integrity failures.

At admission, freeze each panel's mastery reference as the lower success count
from its two consecutive passing evaluations. Retention is evaluated only on
unique identities in the same frozen panel version:

- an eight-map condition retains at
  `successes >= max(6, reference_successes - 1)`; and
- a fixed 32-map four-cell family panel retains at
  `successes >= max(26, reference_successes - 1)`.

This makes the evaluation quantum explicit: one map is `12.5` percentage
points for a condition and `3.125` points for a family panel. Slot duplicates
never increase either denominator. A separately gated source slice must
declare its own fixed `n` and integer threshold before use. Later peaks do not
ratchet the frozen mastery reference upward, and success-set Jaccard remains a
diagnostic rather than another gate.

There is no per-environment promotion or automatic demotion. Promotion starts
a new recorded run at a checkpoint boundary. Two consecutive scheduled,
complete, integrity-valid retention failures for any condition or qualified
family panel trigger recovery; a passing evaluation resets that panel's
failure streak. An incomplete or integrity-invalid evaluation blocks
promotion and is repaired, but does not count as evidence of policy
regression. On a retention trigger:

1. stop the current run;
2. restore the last checkpoint that passed all prior gates;
3. relaunch the previous mixture as a new recorded treatment; and
4. do not call that relaunch an exact continuation unless model, optimizer,
   schedule, RNG, environment, and action-history state are all restored.

Development is diagnostic and cannot drive promotion. The sealed split is
opened once after model selection.

### 8.3 Family balance and rehearsal treatment

Sampling is hierarchical so a generalist never loses the accepted `50%`
foundation / `50%` trench balance:

1. a specialist fixes its one family; a generalist samples foundation/trench
   `50/50`;
2. inside the sampled family, if that family has an active frontier, sample
   `50%` from its frontier and `50%` uniformly from its admitted conditions;
3. if that family has no active frontier, sample uniformly from its admitted
   conditions; and
4. while bootstrapping the first anchor, sample that frontier at `100%` because
   no admitted parent exists.

Multiple frontier or admitted conditions are uniform within their bucket. The
resulting global frontier share may be below `50%` when only one family is
advancing; it is receipted rather than disguised by breaking family balance.
Exact weights, slot multiplicities, unique identities, and realized exposure
are frozen per run.

Uniform per-condition replay decays as more conditions are admitted; that
arithmetic is reported but is not itself evidence of failure. Two retention
failures invoke the rollback contract in Section 8.2. Any alternative replay
treatment requires a new explicit spec row and causal comparison; v1 does not
pre-authorize an ambiguous "recent" bucket or an adaptive priority sampler.

### 8.4 Eight-condition local pilot

The first S2 review bank is deliberately small: `32` public-train, `8`
promotion, `8` public-development, and `8` sealed scenarios per condition,
for 448 scenarios total. All counterfactual variants from a source group stay
in one split.

These labels are candidate aliases. S1 expands every alias into the full
mechanical `condition_id`; no S2 record may retain `vmatch`. Numeric volume
bands are frozen only after the pair-specific train-only support audits below.

| Candidate alias | Controlled purpose |
|---|---|
| `f.all.osm.sep00_02.slcap20_45.v140_189` | OSM/source-bank all-around slice in the frozen matched support |
| `f.all.procedural.sep00_02.slcap20_45.v140_189` | procedural all-around slice paired at exact volume and perimeter |
| `f.apron.osm.sep02.slcap07_10.vmatch` | new moderate-capacity OSM apron capability |
| `f.apron.osm.sep02.slcap03_04.vmatch` | exact-dig paired constrained-capacity OSM counterfactual |
| `t.straight.both.sep02.slcap03_04.v68_77` | straight, both-side local trench candidate |
| `t.straight.one.sep02.slcap03_04.v68_77` | exact-dig one-side counterfactual |
| `t.segmented2.both.sep02.slcap03_04.v68_77` | no-junction, volume-conditioned topology comparison |
| `t.segmented3.both.sep02.slcap03_04.v68_77` | three-segment no-junction, volume-conditioned topology comparison |

`slcap07_10` and the generous apron are new generator work; the current bank
jumps from `3.25x` constrained capacity to roughly `20.6x`-`41.2x`
all-around capacity. The moderate apron must be visually reviewed and must
match the constrained pair's dig raster and separation distribution.

The completed B0-GEO-F run remains evidence against pooling the two source
cells today: both held-out cells stayed `0/8` across all 50 checkpoints, while
training support differed. A later matched and diverse source experiment may
retire the separate cells.

Foundation matching uses two contracts, not one global four-cell interval:

1. the OSM/procedural all-around source pair uses the train-audited closed
   `v140_189` interval and compactness support `[0.30, 0.65]`. Each matched
   pair has identical required unit-depth volume and identical exposed
   four-neighbour perimeter, so compactness
   `4*pi*volume/perimeter_4_edges^2` is identical rather than merely close;
   and
2. the `slcap03_04`/`slcap07_10` apron pair reuses the exact same OSM dig
   raster within each source group, so pairwise target identity and volume
   equality are the gate.

The fixed train-only source audit partitions canonical source groups before
distribution measurement. It loads reserve masks only to canonicalize source
identity and apply the fixed factory-eligibility rule; it computes no reserve
distribution metrics and performs no reserve support or matching selection.
From 256 audit-train source groups, 111 lie in the frozen support, 98 have at
least two unique exact procedural candidates, and 32 unique exact pairs are
materialized. The retuned procedural sampler changes only its main
length/width ranges to `U[13,22)` and `U[7,13)` and yields 3,732 supported
proposals (3,721 unique) from 20,000. The byte-reproduced receipt is
`/home/lorenzo/moleworks/.artifacts/terra_pilot_foundation_source_support_20260727_v3/`
(`support_summary.json` SHA-256
`e0867f67669c5f7f2e6ce15890c2c824783666056ddb58ebe21e56107191261f`).
The historical B0 builder remains byte-identical. The apron pair need not
share this source-pair interval.

Exact volume/perimeter matching does not equalize every shape descriptor:
selected OSM/source-bank and procedural bbox-aspect medians are `1.065` and
`1.125`, while moment-aspect medians are `1.231` and `1.344`. This is a
volume/compactness-matched source-geometry comparison, not a source-only
causal intervention. Those residual covariates remain website audit slices.

This receipt proves raster-level train support, not publication provenance.
The current source artifact does not preserve raw OSM feature IDs or an
attribution manifest. Recover and pin those records, or rename the public
slice as an unattributed source bank, before a downloadable benchmark release.

The existing B0 trench identities also do not support a common frozen volume
band: only `2/16` straight, `2/16` segmented-2, and `0/16` segmented-3
identities fall in the previously proposed `65-74` interval. This is evidence
against carrying over that bank, not proof that the current raster generator
has zero support.

S1 therefore adds one narrow trench-support deliverable:

- keep the straight generator fixed;
- audit candidate segmented-length ranges, beginning with approximately
  `U[10, 13)` tiles for segmented-2 and `U[7, 9.5)` for segmented-3;
- use one fixed train-only seed namespace and at least 20,000 raster-valid
  proposals per topology;
- compare closed ten-integer-cell intervals, retaining the width of the
  withdrawn `v65_74` proposal so the audit selects location rather than
  jointly optimizing location and width; freeze one only if it contains at
  least 10% of proposals from each topology and its midpoint lies inside each
  topology's empirical 10th-90th-percentile range; and
- receipt raw and accepted segment lengths, turn angles, achieved volumes,
  uniqueness, and every rejection category.

The train-only S1 audit froze `U[10,13)` for segmented-2, `U[7,9.5)` for
segmented-3, fixed radius-one width, and `v68_77`. Across 20,000 proposals per
topology, support was `21.46%` straight, `52.795%` segmented-2, and `79.50%`
segmented-3, with 3,077, 8,130, and 15,420 unique accepted rasters. The
byte-reproduced receipt is
`/home/lorenzo/moleworks/.artifacts/terra_pilot_trench_volume_support_20260727_v2/`
(`support_summary.json` SHA-256
`a0927ef98e13774edfe0c500184ba1dd88b23e35bb67b14b0628c852bb8fb01c`).
S2 materializes fresh source-disjoint `32/8/8/8` splits without retuning on
promotion, development, or sealed identities. Volume conditioning shifts
segmented-2 toward longer segments; the website must show the receipted
covariates, and the result is named a volume-conditioned topology comparison,
not a pure geometry-only causal estimate.

The existing `d02`-`d08` panels remain useful separation diagnostics but do
not occupy a pilot curriculum rung. A forced-rehandling candidate is added
only after the exact direct-service validator exists and an action witness
fits the horizon.

Segmented-2/3 have no branch junction and remain one-axis geometry; their
comparisons use a frozen overlapping volume interval rather than the rejected
global trench volume bands. T, X, multi-junction, disconnected,
site-constrained, combined, and forced-rehandling cases are later candidate
nodes. T/X and disconnected cases are Composed/complex geometry and are never
introduced together with a new dump or site constraint.

## 9. Suites

The benchmark exposes complementary views rather than one undifferentiated
bank.

### 9.1 Axis panels

Axis panels hold a base source group fixed and vary one factor. They are the
main causal diagnostic for current policies:

- matched foundation source coverage;
- foundation dump layout/capacity;
- dig/dump separation and direct-service coverage;
- trench one-side versus both-side access;
- trench topology;
- site constraint;
- work amount; and
- full versus partial reset.

The current B0 builder already contains 16 unique foundation/trench cells
covering the first five items, but lacks site and reset panels. It becomes a
pilot input, not the final benchmark.

### 9.2 Display-depth suites

Anchor, One-axis, and Composed views contain all admitted conditions at that
display depth and are reported separately. They are review views, not a
required total-order training schedule.

### 9.3 Core leaderboard

The proposed v1 Core score covers full-reset, dynamically witnessed Anchor,
One-axis, and admitted Composed conditions. Forced-rehandling maps have their
own leaderboard until enough methods establish that they are a useful
solved-but-difficult regime.

The Deployment Mix is a separate realistic-mixture score. It must not replace
the balanced per-cell Core result.

### 9.4 Compositional transfer

Source-disjoint maps within familiar cells test new geometry sources, not
unseen combinations. A separate compositional suite therefore holds out entire
factor combinations from every model-selection-visible split (`public_train`,
`promotion`, and `public_dev`) while keeping every primitive factor
represented in public training.

Example: public training includes procedural-foundation x all-around and
OSM-foundation x separated, while compositional test contains
procedural-foundation x separated. The split registry must prove:

- the held-out combination is absent from public training, promotion, and
  development;
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

The S2 toolchain/visual pilot uses:

| Split | Unique source groups per cell | Visibility |
|---|---:|---|
| `public_train` | 32 | Maps, seeds, metadata, and examples public |
| `promotion` | 8 | Frozen local gate; never used for gradients or public model selection |
| `public_dev` | 8 | Maps and per-map diagnostics public |
| `sealed_pilot` | 8 | Held-out local maps; opened once |

After schema, validator, live-geometry revalidation, and human review pass,
only active training conditions initially expand to 256 unique
public-training source groups. Promotion remains separate; public development
expands to 32 and sealed pilot to 64 only for a selected larger treatment.
This avoids generating roughly 7,000 witnessed scenarios before the small
toolchain and condition definitions are accepted.

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

1. source groups are disjoint across train, promotion, development, sealed,
   private-test, and compositional splits;
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

Pilot admission requires `witness_steps <= 450`, matching the frozen horizon.
`witness_step_fraction` and the distribution of remaining horizon margin are
reported. A stricter publication-Core anti-censoring margin, such as `225`
steps, is not a ratified gate and must be chosen from S3 evidence in a later
spec revision.

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
- the live `0.571428571428125` m/tile scale, runtime-derived `7 x 11`
  footprint, `6.375`-`11.375` tile annulus, and twelve base/cabin
  orientations;
- target/obstacle/road/dump-mask disjointness and semantics;
- exact accepted-mask capacity and integer headroom;
- traversable access and reachable accepted dump components;
- valid spawn, exact action-reachable base-pose graph, per-volume initial
  workspace coverage, and direct-service replay;
- declared geometry components, axes, segments, junctions, and degrees;
- mass balance;
- split, hash, source, pair, and near-duplicate audits; and
- deterministic rejection reasons.

S1 materializes exactly one pilot initial state per source group and split. A
counterfactual group shares one state across all of its variants; an unpaired
map is its own group. The generator derives a `uint32` reset seed from the
first four SHA-256 bytes in big-endian/network order:
`SHA256("terra_initial_state_seed_v1\0" + release_id + "\0" + split + "\0" +
source_group_id + "\0" + state_index)`, using `state_index=0` for the pilot.
It calls the live one-tracked-agent `Agent.new` path once against the
intersection of the variants' admissible spawn masks, then validates and
serializes that exact state against every variant. The internal live spawn
rejection sampler is part of this call. There is no resampling after observing
workspace, direct-service, witness, or policy outcomes: a state or group that
fails is listed and replaced under the ordinary admission rules. Promotion,
development, and sealed namespaces are distinct.

Initial direct service is an exact transition outcome, not an annulus
approximation. Starting from the serialized state, the validator builds the
reachable `(row, column, angle_base)` graph using the real unloaded forward,
backward, clockwise, and anticlockwise transitions. For each reachable pose
and all twelve cabin headings, it replays the real dig transition and all
twelve loaded cabin headings. A dump is legal direct service only when the
complete pre-dump load is unloaded, mass is conserved, the entire positive
soil delta lies inside the exact accepted dump mask, and the off-mask delta is
zero. The intentionally executable C1a wrong-dump transition is counted as
`wrong_complete_dump_attempts` and never as legal service.

Overlapping dig cones are unioned per required cell: for each cell, retain the
maximum target progress from any legal dig, and separately the maximum from
any dig followed by at least one legal complete dump. Cap both by that cell's
initial remaining required volume before summing. These sums divided by total
initial remaining required volume define `initial_workspace_coverage` and
`direct_service_coverage_initial`. This prevents overlapping hypothetical
digs from double-counting work.

The frozen B0a rasters were generated for the live Terra runtime, but their
metre fields and static audit were computed with stale `0.6875` m/tile,
`5 x 9` footprint assumptions. Repair does not change Terra
`edge_length_m` and does not invalidate F0/F0R runtime witnesses. S1 must
recompute metre fields and every static receipt over unchanged rasters using
live geometry, preserve identities that pass, replace only failures, and
refresh affected hashes/manifests. Editing metre labels without revalidation
is insufficient.

An exploratory 2026-07-27 live-geometry probe was not saved with a versioned
script, output, or failing-identity list, so its numerical count is withdrawn
and is not benchmark evidence. S1 must instead write one migration receipt
over the unchanged frozen B0a inputs containing input, validator, protocol,
and script hashes plus every per-identity old/new outcome and rejection
reason. Static-valid status is adjudicated only by the new initial-state
metrics; S1 must not recreate an undefined generic post-dig state merely to
match an unreceipted count.

Plan-first constructive generation is one candidate witness supplier for
procedural maps, not a universal requirement. Its traces require exact replay
and distribution-bias audit. OSM cases without a method-neutral witness remain
unranked rather than being selected by one learned policy.

If a released map later fails integrity or feasibility, it is removed only in
a new scored release. Existing results remain archived with the original
release status.

## 12. Distribution and curriculum audit

Every release automatically exports:

- exact scenario, map, source-group, and slot counts;
- weights and effective sample size per condition;
- geometry x dump, dump x site, family x reset, and display-depth x family
  heatmaps;
- distributions of dig volume, separation p50/p95/max, direct-service
  coverage, capacity, dump components, side balance, angles,
  aspect ratio, obstacle fraction, legal dump coverage, and work;
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
- prerequisite/display-depth coverage;
- train-to-evaluation support gaps; and
- current-policy success/completion by cell, when receipts are supplied.

It also verifies that train, promotion, development, and sealed source groups
are disjoint; promotion uses full resets; and every materialized map level has
the frozen horizon/reward/action/observation/dynamics contract.

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
  policy's prerequisite-cell performance;
- **unsupported prerequisite**: a composed cell is exposed before one or more
  required one-axis parents has passed; and
- **source-slice gap**: matched OSM/procedural or other provenance slices have
  materially different performance.

These are review warnings, not automatic map deletion or promotion rules.

## 13. Evaluation protocol

The primary v1 protocol freezes:

- one tracked excavator;
- the official observation schema and action values `0..7`;
- one untouched full-task reset from the scenario's explicit validated initial
  agent and soil state;
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

The release freezes a hierarchy of semantic weights rather than allowing
scenario counts to change the score. Core uses equal family weight and equal
condition weight within family:

```text
Balanced Condition Success
  = mean_family(
      mean_condition_in_family(success_rate)
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
- full success/completion heatmaps by declared factor and display depth; and
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
Sealed-test results expose only family, cell, display depth, and declared-axis
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
- Anchor, One-axis, and Composed review with prerequisite links;
- click-through stratified galleries;
- separation, direct-service, capacity, uniqueness, and
  format/static/witness warnings; and
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
- clear labels for Core, Forced Rehandling, Deployment Mix, and Unverified
  Challenge.

### 16.2 Map Explorer

Filters:

- family and Anchor/One-axis/Composed display depth;
- foundation source/structure;
- trench topology, segments, components, junctions, and junction degree;
- dump layout, side access, components, dig/dump separation, and capacity;
- site class and quantitative obstacle/access ranges;
- work volume;
- reset mode;
- feasibility badge; and
- public split.

Each card shows:

- a colored composite;
- target, occupancy, dumpability, initial-soil, and reward-distance layer
  toggles;
- complete factor vector and exact measurements;
- capacity, dig/dump-separation, and format/static/witness badges;
- source/generator provenance and nearest-neighbour audit;
- condition/display-depth membership and prerequisites; and
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
- dig/dump-separation, capacity, volume, topology, obstacle, and dump-coverage
  distributions;
- near-duplicate and split-leakage audit;
- rejection reasons; and
- missing, underrepresented, or support-mismatched cells.

The dashboard can compare any uploaded training manifest with the frozen
benchmark target.

### 16.4 Cell Page

Show:

- normative condition definition and numeric bands;
- display-depth, prerequisite, and suite membership;
- split counts and source-group counts;
- feasibility requirements and witness coverage;
- fixed stratified public gallery;
- distributions and balance warnings;
- reference-policy performance with uncertainty; and
- common failure modes.

### 16.5 Curriculum Inspector

Accept a bank manifest or mixture JSON. Show:

- condition and display-depth exposure;
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
- Forced Rehandling and Deployment Mix scores when present;
- deterministic/sampled mode;
- data track; and
- integrity badge.

Expanding a row shows the full condition/display-depth heatmap. The Compare page shows two
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
4. Review Anchor candidates first, then each one-axis counterfactual and its
   direct parent side by side.
5. Inspect distribution tails and all rejected/quarantined candidates.
6. Reject semantic errors; record aesthetic preferences separately.
7. Produce and replay exact-initial-state witnesses.
8. Select balanced source groups without moving variants across splits.
9. Freeze source-disjoint train/promotion/development splits and commit the
   sealed split.
10. Evaluate frozen reference methods and publish the first receipts.

The publication site may later add a "review queue" mode with:

- paired base/counterfactual maps side by side;
- next/previous cell and keyboard navigation;
- approve, reject, quarantine, and note export;
- deterministic ordering and stable URLs; and
- an outlier queue for extreme separation, capacity, volume, obstacle, or
  similarity values.

Human review decisions are exported as data. They are not stored only in a
browser or screenshot.

## 18. Minimal implementation plan

The immediate implementation is a local review bundle with one golden-path
command:

```bash
python tools/build_map_review.py \
  --conditions benchmark/pilot_v03.json \
  --out .artifacts/terramap_bench_v1

xdg-open .artifacts/terramap_bench_v1/site/index.html
```

Required outputs:

```text
.artifacts/terramap_bench_v1/
  manifest.jsonl
  audit.jsonl
  summary.json
  migration_validation.jsonl
  validation_cost.json
  review_decisions.jsonl
  thumbnails/
  site/index.html
  policy_results.jsonl       # optional input/output when receipts are supplied
```

The script must fail before exporting the site when a format, identity, split,
live-geometry workspace, direct-service, capacity, or declared-condition check
fails. It is a thin orchestration entry point, not a general framework:

1. `build_benchmark_bank.py`
   - wraps existing generators;
   - writes exact-loader data and normalized scenario records.
2. `validate_benchmark_bank.py`
   - recomputes `audit.jsonl`;
   - derives scale/footprint/workspace from the frozen runtime protocol;
   - validates format, semantics, capacity, splits, similarity, exact
     direct-service fields, and checksums.
3. `export_benchmark_site.py`
   - creates static JSON, thumbnails, galleries, dashboards, and result pages.

S1-S2 stop here. The local site labels maps Format-valid or Static-valid but
does not rank them. Dynamic witness replay, policy evaluation, safe model
submission, uncertainty, sealed-server operation, and leaderboard publication
are S3-S5 work. Later tools are:

- `replay_benchmark_witnesses.py`;
- `evaluate_benchmark.py`; and
- the publication-mode extension of `export_benchmark_site.py`.

Reuse current capacity validation, exact-dataset validation, B0 generation,
fixed-bank evaluation, receipts, and trajectory replay. Do not create a
parallel environment implementation.

Before exact direct service becomes a universal 448-scenario export gate, S1
first writes a non-admission `validation_cost_probe.json` for one fixed
public-train B0a identity and explicit state. It runs the complete exact
movement graph and prefilter, cold-compiles the unchanged exact service kernel
on one real batch, then times a deterministic warmed subset with synchronized
outputs. The subset produces cost evidence only: it emits no coverage or
validity decision. The receipt separates logical attempts from padded kernel
rows and records machine/software/input/protocol/validator hashes, JAX device,
cold compile, first execution, steady per-row p50/p95, candidate counts, peak
memory, and explicit 1/256/448-scenario projections.

If that estimate is viable, run one complete exact 64 x 64 scenario and compare
observed versus projected cost. Only then profile all 256 frozen B0a identities
and write the admission `validation_cost.json`, including per-scenario
p50/p95/max and total wall time. Review both receipts before S2. If cost is
prohibitive, optimize the one exact validator path or revise the spec
explicitly; never substitute an unreceipted geometric approximation.
Content-addressed caching is added only if the profile shows it is needed.

Repository ownership stays narrow:

- Terra owns conditions, generators, exact map/scenario manifests, validation,
  and public map release artifacts.
- terra-baselines owns policy adapters, evaluation, uncertainty, and receipts.
- the static site consumes exported data from both and owns no scientific
  truth.

Suggested delivery gates:

| Status | Gate | Deliverable | Pass condition |
|---|---|---|---|
| `[x]` | `S0 Spec` | v0.3 accepted plan and reviewer-decision log | No stale M0-M5 or protocol claim is normative |
| `[ ]` | `S1 Schema` | Manifest normalizer, explicit initial-state reset, live validator, migration receipt, and cost profile | Existing rasters are re-audited at 0.5714 m/tile; full hashes/outcomes and projected 448-scenario validation cost are reviewed; failures are listed, not hidden |
| `[ ]` | `S1 Capacity` | adjustable apron generator | Exact-dig `slcap03_04`/`slcap07_10` pair matches separation and passes visual/static review |
| `[ ]` | `S1 Support` | pair-specific foundation matching and topology-aware trench length sampler | Train-only audits freeze supported numeric bands/covariates; no `vmatch` reaches S2 |
| `[ ]` | `S2 Review` | 448-scenario eight-condition pool and local site | Counts, splits, pairs, distributions, and maps are human-reviewed |
| `[ ]` | `S3 Feasibility` | witness store and replay | Every pilot-ranked scenario replays from explicit initial state in at most 450 steps; margin is reported |
| `[ ]` | `S4 Evaluation` | Native bundle, evaluator, uncertainty, receipt | Reference submissions reproduce byte-identical receipts |
| `[ ]` | `S5 Publication` | Versioned data, docs, static site, sealed commitment | Downloads and leaderboard are independently verifiable |

## 19. Explicit non-goals for v1

- A generic adaptive curriculum framework.
- A generic learned admission-graph scheduler.
- PLR, ALP-GMM, PAIRED, ACCEL, or learned adversarial generation.
- Automatic promotion/demotion policy.
- A single learned scalar map-difficulty predictor.
- Partial resets inside the map-curriculum causal comparison.
- Changing Terra `edge_length_m` to repair stale benchmark metadata.
- Curved trenches.
- Arbitrary `N`-axis runtime support without a metadata-contract change.
- A database-backed web application.
- Arbitrary executable Python or container submissions.
- Mixing reward-curriculum claims with map-curriculum claims.
- Calling static validity proof of dynamic feasibility.

## 20. Reviewer decision log and remaining choices

This table is append-only. A future review change must add a row and update the
normative section in the same commit. `Accepted with correction` records the
part that was retained and the factual correction, so stale reviewer text
cannot silently regain authority.

| ID | Disposition | Durable decision |
|---|---|---|
| `R-20260727-01` | Accepted | Freeze the map/scenario/protocol/result identity split, source-group atomicity, validator-owned audit, balanced per-condition score, deterministic/sampled separation, and v1 non-goals. |
| `R-20260727-02` | Accepted | Delete the M0-M5 total order and unsupported 20-cell registry. Use Anchor/One-axis/Composed display depths plus literal per-condition prerequisites. |
| `R-20260727-03` | Accepted with correction | Live Terra is self-consistent at 36.5714285714 m / 64 = 0.571428571428125 m/tile and must not change. Correct metre metadata without regenerating rasters solely for scale; F0/F0R runtime witnesses remain valid. |
| `R-20260727-04` | Accepted with correction | d02-d08 is dig/dump separation, not loaded transport. Add action-reachable exact direct-service coverage. Defer relay-hop scoring until its state graph is defined. The predicted flat effect is not accepted as fact: existing foundation and trench results decline with separation. |
| `R-20260727-05` | Accepted | Map-curriculum runs use 100% untouched full resets. The 25% partial-reset mixture remains the separate PR0 treatment after map-sampler selection. |
| `R-20260727-06` | Accepted | Promotion uses a separate source-disjoint bank, 6/8 per new/direct-parent cell twice, 26/32 for a four-cell family panel, zero integrity failures, retention, checkpoint-bounded promotion, and stop/restore/relaunch recovery. |
| `R-20260727-07` | Accepted with current evidence | OSM/procedural is provenance in principle, but cells stay separately matched and gated. B0-GEO-F has completed, contrary to the stale review: both held-out cells remained 0/8 across all 50 checkpoints and their training support differed. |
| `R-20260727-08` | Accepted | Segmented-2/3 have no junction and are one-axis geometry. T, X, multi-junction, and disconnected cases are later complex/composed nodes. |
| `R-20260727-09` | Accepted | Adjustable moderate apron capacity is the first new generator capability. Build an exact-dig, separation-matched `slcap03_04` versus `slcap07_10` pair before defining capacity progression. These tokens name single-layer accepted-area ratio. Capacity is a high-value missing causal axis, not a proven sole cause. |
| `R-20260727-10` | Accepted with restraint | Preserve 50/50 foundation/trench sampling for a generalist, then use 50% frontier / 50% uniform-admitted within each active family. Uniform-share decay is reported but not presumed harmful. Retention failure rolls back; no ambiguous recent-cell or adaptive replay treatment is pre-authorized. |
| `R-20260727-11` | Accepted | Hold exact volume fixed for dump/capacity/site counterfactuals and match overlapping volume support for source/topology comparisons. Do not revive global family low/normal bands. |
| `R-20260727-12` | Accepted | Train, promotion, development, and sealed splits are source-disjoint; counterfactual variants stay in one split. Every run pins horizon 450, DENSE rewards, trench absolute shaping off, and one protocol hash. |
| `R-20260727-13` | Accepted with restraint | Pilot-ranked scenarios require exact initial-state replay within the frozen 450-step horizon. Record witness margin; a 225-step publication-Core gate remains unratified until S3 evidence. Forced-rehandling and unwitnessed cases remain separately labeled Challenge cases. |
| `R-20260727-14` | Accepted with simplification | Start with a balanced four-foundation/four-trench, 448-scenario review pilot. Replace the unsupported `vhigh` cell with volume-matched segmented-3; add a work-volume cell only after numeric support and witness gates pass. Expand only active accepted training conditions to 256 identities after schema/static/human review. |
| `R-20260727-15` | Accepted implementation-audit amendment | Keep every raster unchanged initially, but recompute static receipts because the old validator used the stale 5 x 9 footprint. Preserve passing identities and replace only proven failures; do not attribute this stronger revalidation rule to the original metadata-only review. |
| `R-20260727-16` | Accepted implementation-audit correction | Exact initial state means the versioned canonical bytes of every reset-consumed Agent and AgentState field, not a partial pose tuple or a reset seed. |
| `R-20260727-17` | Accepted implementation-audit correction | S1 workspace/direct-service metrics describe the exact initial scenario using reachable Terra action transitions and runtime dig/dump masks. There is no undefined generic post-dig state; terminal and during-trace access metrics are derived only by canonical S3 witness replay. |
| `R-20260727-18` | Accepted implementation-audit correction | Full condition IDs include source, achieved separation, explicitly named capacity metric/band, and numeric volume support. Freeze trench `v65_74`; a train-only audit must replace every foundation `vmatch` alias before S2. |
| `R-20260727-19` | Accepted with factual correction; supersedes R-18's trench band | Withdraw frozen trench `v65_74`: the existing bank has only 2/16 straight, 2/16 segmented-2, and 0/16 segmented-3 identities in that band. Raster sampling shows nonzero generator support, so "structurally impossible" is too strong. S1 audits candidate topology-specific length ranges, then freezes one supported volume-conditioned comparison; all trench pilot identities are fresh. |
| `R-20260727-20` | Accepted | Replace the unquantized five-percentage-point retention rule with integer gates. Freeze the lower of two passing counts; retain at `max(6, reference-1)` on 8-map cells and `max(26, reference-1)` on the fixed 32-map family panel. Only two consecutive complete integrity-valid failures trigger rollback. |
| `R-20260727-21` | Accepted | Foundation volume matching is pair-specific: retune procedural all-around generation toward fixed OSM support for the source comparison, while the apron capacity pair shares exact OSM dig rasters and pairwise volume. No single interval spans all four foundation cells. |
| `R-20260727-22` | Accepted | Remove the unreceipted 23/256 probe count. S1 emits a hashed per-identity migration receipt and profiles exact direct-service validation before S2. Normalize legacy `broad_side_cast` provenance to canonical `side_cast`. |
| `R-20260727-23` | Accepted and executed | Keep the proposed ten-integer-cell trench-band width fixed, audit only its location under the preregistered support/interior rule, and freeze `v68_77`. The tracked pilot sampler leaves the hash-pinned B0 builder unchanged and receipts generator retries, duplicates, and conditioning covariates. |
| `R-20260727-24` | Accepted and executed | Add an optional complete `Agent` input to single/batched reset, preserve the supplied tree and RNG exactly, and freeze `terra_agent_state_v1` as path/dtype/rank/shape-prefixed canonical bytes over every live field and all four slots. Admissibility remains a separate host-side gate. |
| `R-20260727-25` | Accepted | Materialize one deterministic initial state per source group and split from a hash-derived seed and the live `Agent.new` sampler over the variants' intersected spawn contract. Share it across counterfactuals and never resample after seeing feasibility or policy outcomes. |
| `R-20260727-26` | Accepted | Compute initial direct service by exact action replay from the serialized state. A complete off-zone wrong dump remains a valid C1a mistake but never counts as legal service. Union overlapping hypothetical digs by capped per-cell maximum progress before computing coverage. |
| `R-20260727-27` | Accepted and executed | Freeze foundation source support at `v140_189` and compactness `[0.30,0.65]`; pair at exact required volume and four-neighbour perimeter. The audit uses a fixed 256-group train partition, leaves reserve groups identity-and-eligibility-only, retunes only procedural main length/width, and preserves distinct source IDs. Raw OSM attribution remains a publication gate rather than being inferred from raster filenames. |
| `R-20260727-28` | Accepted implementation hardening | Treat the foundation result as a volume/compactness-matched source-geometry comparison, not source-only causal isolation. Reject cross-family canonical-raster collisions explicitly and receipt Python/NumPy/SciPy versions; tracked generator portability is required before S2 materialization. |
| `R-20260727-29` | Accepted implementation staging | The exact direct-service kernel remains the validator contract, but cost discovery is staged: one non-admission 64 x 64 probe with full pose graph/prefilter and a warmed exact-service subset, then one complete scenario, then the required 256-identity profile. Logical attempts and padded kernel executions are receipted separately; a subset never emits feasibility. |
| `R-20260727-30` | Accepted migration identity contract | In the legacy B0a migration, canonical `source_group_id` is the existing `source_id`; this yields 144 groups across 256 rows (112 singleton, 16 four-map foundation-distance, 16 five-map straight-trench). `paired_source_group_id` remains counterfactual-panel metadata and `topology_match_group_id` never shares reset state. Use release ID `terramap-bench-v1.0.0`; direct-service coverage is diagnostic, while static dump access gates reachable capacity rather than requiring every optional dump component. |
| `R-20260727-31` | Accepted and executed | The exact initial direct-service validator uses real tracked movement, dig, cabin, and dump transitions; the exact visible accepted mask; diagnostic entirely off-zone complete dumps; hard failure for mixed-boundary complete dumps; and capped per-cell union across hypothetical digs. Logical/padded counters and a prefilter differential guard are mandatory. The semantic suite passes, while 64 x 64 cost remains a separate open gate. |

Still to decide through S1-S2 evidence:

- the exact moderate-apron construction that changes capacity without changing
  the dig raster or separation distribution;
- the conservative relay-hop algorithm and method-neutral witness supplier;
- whether S3 evidence supports a stricter publication-Core witness margin than
  the frozen 450-step pilot horizon;
- family-specific cross-split similarity thresholds;
- whether a future matched, diverse OSM/procedural experiment permits pooling;
- the publication test confidence/effect target and number of fixed initial
  states per map;
- recovery of raw source feature IDs and attribution for the current
  OSM-labeled raster bank, or a public rename that makes the missing
  provenance explicit;
- migration of the currently hash-pinned generator helpers out of local
  `.artifacts` into the tracked S2 builder/release source;
- benchmark data/submission licenses, public hosting, and sealed-evaluator
  ownership.

No open choice authorizes PPO. The next implementation is S1 Schema/live
revalidation, S1 Capacity, and S1 Support, followed by the S2 local review
site.
