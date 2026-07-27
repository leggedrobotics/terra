# TerraMap-Bench specification

Status: design-frozen v0.3; local B0a preview complete; S1 implementation
pending

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
The active delivery goal and its verification checklist are pinned in
[`DIGGING_BENCHMARK_SITE_GOAL.md`](DIGGING_BENCHMARK_SITE_GOAL.md); that goal
is subordinate to this scientific contract and
[`TRAINING_TASKS.md`](TRAINING_TASKS.md).

The first deliverable is one locally served static client, not a scientific
service. It shows:

- the condition x policy heatmap with worst cells first;
- Anchor, One-axis, and Composed review with prerequisite links;
- click-through stratified galleries;
- separation, direct-service, capacity, uniqueness, and
  format/static/witness warnings; and
- training exposure versus benchmark exposure.

The separate pages below are the publication phase. Browser uploads, persistent
review state, and submission handling do not block the first inspector.

The B0a preview is a static client application served locally, not a
scientific service. It is an implementation precursor; the fail-closed S1-S2
golden-path builder and standalone release bundle remain pending.

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
and write the bank-profile `validation_cost.json`. Each identity has exactly
one synchronized end-to-end duration, measured from the start of exact
map/state loading through synchronized outcome computation and atomic scenario
receipt publication using the existing atomic no-replace hard-link pattern.
Because a receipt cannot contain the duration of its own final publication, the
worker records that duration afterward in its final shard ledger. The aggregate
receipt reports nearest-rank p50, p95, and maximum over those 256 identity
durations plus total observed makespan; it does not claim repeated per-identity
quantiles. Review both receipts before S2. If cost is prohibitive, optimize the
one exact validator path or revise the spec explicitly; never substitute an
unreceipted geometric approximation. Content-addressed caching is added only if
the profile shows it is needed.

The operational go/no-go rule is frozen before observing the probe. Proceed to
one complete scenario only when the cost-only probe completes without error,
its synchronized one-scenario p95 projection is at most 60 minutes, and peak
host/device use is at most 80% of capacity. Proceed from that confirmation to
the 256-identity profile only when observed wall time is within `[0.5, 2.0]`
times the probe's one-scenario p50 projection, calibrated 256/448-scenario p95
costs are at most 24/48 hours, and the same memory headroom holds. Sequential
execution is the baseline. After the exact GPU and hybrid paths failed
cross-backend parity, the only allowed concurrency amendment is the fixed
four-process CPU treatment in `R-20260727-58`: each process executes the
unchanged exact entrypoint and owns a deterministic scenario shard. No
in-process vectorization, alternate kernel, dynamic worker count, or result
approximation is implied. A failed gate selects optimization or corrected
accounting, not weaker validation.

The conditional 256-identity profile is itself one fixed treatment. It consumes
the frozen B0a live-migration receipt in lexicographic `legacy_map_id` order and
assigns global index `i` to worker `i % 4`, yielding four disjoint 64-scenario
shards. Each long-lived worker loads the migrated record's explicit serialized
agent state and calls the unchanged exact CPU entrypoint once per assigned
identity. Worker zero evaluates the pinned confirmation identity first within
its ordinary shard and must reproduce its complete typed outcome exactly.
There is no retry, resume, result cache, backend option, worker-count option, or
dynamic scheduling.

Let `D_j` be the one end-to-end duration for identity `j`, including its atomic
no-replace scenario-receipt publication. Nearest-rank p50/p95/max are reported
across all 256 values. For worker `i`, let `Q95_warm_i` be the nearest-rank p95
over its 63 durations after excluding that worker's first identity. With 64
observed identities per worker and 112 needed for a 448-scenario bank, freeze
the cost-only extrapolation
`P_448 = observed_256_makespan + 48 * max_i(Q95_warm_i)`. The observed
256-scenario makespan begins immediately before the first worker spawn and ends
only after all four workers exit successfully, all 256 scenario receipts are
verified, and a canonical candidate merge is atomically persisted and
re-verified. Only after every numeric and integrity gate passes is that
candidate published at the success filename `direct_service_results.jsonl`
using the same no-replace primitive. The profile passes only when all 256
canonical identities complete exactly once, the confirmation sentinel matches
bit-for-bit, every identity takes at most 3,600 seconds, observed makespan is at
most 86,400 seconds, `P_448` is at most 172,800 seconds, all workers retain
their fixed affinity and CPU backend, swap and cgroup OOM deltas are zero,
coordinator plus conservative summed worker peak RSS is at most 80% of physical
memory, and pre/post code and input receipts match. Zero direct service remains
valid measured data rather than a profile failure. Any failure emits partial
evidence but no authoritative merged result and authorizes no Static claim,
bank admission, witness, or PPO. Any corrected rerun requires a new append-only
decision and fresh output directory.

The R-58 receipt's narrow
`authorizes_one_deterministic_four_worker_256_profile: true` is the only
authority consumed by this profile. Its generic `authorizes_bank_profile:
false` remains required and means that no reusable or alternate bank-profile
authority was granted; it does not negate the named one-shot authorization.

Migration and execution provenance remain separate. Each result preserves the
frozen migration protocol at Terra revision `affc0d921...`, environment
protocol SHA-256 `15e4d45f...`, and EnvConfig SHA-256 `02863f62...`. The
profile separately derives the full execution protocol from its own clean
committed Terra revision and receipts that revision, protocol hash, EnvConfig
hash, and exact code bundle. The revision-dependent protocol hashes are not
required to equal; the EnvConfig hash and all environment-consumed constants
must. All workers must agree on both receipts. This distinction prevents the
cost sidecar from silently rewriting migration identity or claiming Static
validity. Before publishing success, re-hash the authorization and
confirmation receipts plus the frozen B0a identities/checksum/source/provenance
manifests and migration JSONL/summary and require exact equality with their
pre-run values. Re-verify every file named by the frozen `files.sha256` against
its recorded digest and reject missing, changed, or unmanifested consumed
files; matching the checksum-manifest bytes alone is insufficient.

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
| `[x]` | `B0a Preview` | Local design-input inspector and organized gallery | Hash-verified public-train/development input is reviewable; zero sealed assets; no Static, Witness, admission, or ranking claim |
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
| `R-20260727-32` | Accepted operational gate | Cost escalation is staged with preregistered limits: probe-to-full requires at most 60 minutes projected p95 and 20% memory headroom; full-to-256 requires factor-two projection agreement, at most 24/48 hours calibrated p95 for 256/448 sequential scenarios, and the same headroom. Failure selects optimization or corrected accounting, never an approximate validator. |
| `R-20260727-33` | Accepted and executed | The fixed cost-only probe passes its first escalation gate: one full scenario is projected at `1,060.204 s` p95 with `2,946,580 KiB` process peak. Its provisional 256/448 p95 estimates are about `60.0/105.0 h`, so only the full-scenario calibration is authorized; no bank-scale run or feasibility claim follows from the subset. |
| `R-20260727-34` | Accepted and executed | One complete exact scenario took `1,086.615 s`, `1.0324x` the probe p50, and `3,558,344 KiB`; it achieved exact workspace/direct coverage `104/104`. The projection is calibrated, but 256/448 p95 is about `61.9/108.4 h`, failing both operational limits. Optimize and re-profile the same exact path; do not launch the bank or weaken validation. |
| `R-20260727-35` | Rejected after exact null | The focused parity test preserved outputs after replacing counterfactual dump/cabin wrapping with raw real transitions, but warmed batch latency was effectively unchanged (`0.9986x` p50 and `1.0041x` p95, new/old). This is consistent with compiler elimination or amortization, not proof of it. Revert the extra semantic proof surface. Next test only fixed service batch sizes `4/8/16` with exact output parity before considering another algorithmic change. |
| `R-20260727-36` | Accepted optimization gate; amended before execution | Sweep only exact service batch sizes `4/8/16` on the same pinned state: first 16 accepted rows for timing, first 18 for padded-tail parity. Give every arm one equal 16-row warmup and 12 timed repeats; bracket with a closing batch-4 control whose p50 drift must be at most 5%. Exact parity means matching dtype/shape/content hashes. Persistent compilation caching fails closed; padding and cumulative shared-process memory remain explicit. Select away from 4 only when the eligible winner's projected 256-scenario p95 is strictly below the smaller opening/closing batch-4 p50, breaking ties toward the smaller batch. Selection authorizes a fresh staged probe, not a bank run. |
| `R-20260727-37` | Accepted and executed | The controlled sweep passed exact output parity, memory, and the 2.27% opening/closing drift gate, but batches 8 and 16 were slower over the same 16 logical rows. Their projected 256-scenario p95 costs are 75.5 and 75.1 hours versus 65.4-65.5 hours for batch 4; neither beats the strict 62.8-hour batch-4 p50 threshold. Retain batch 4. Do not re-probe, re-confirm, or run 256 scenarios from this treatment. |
| `R-20260727-38` | Accepted delivery priority | Standardize the training-distribution review contract and build the graphical benchmark inspector now, while unresolved S1 admission work continues. Its first input is the current B0a bank and must be labelled design input, not S2 or a frozen release. The owner-only site exports human decisions as data, exposes no sealed maps, executes no models, and becomes the same exporter/UI used for the admitted S2 bank rather than a parallel dashboard. |
| `R-20260727-39` | Accepted delivery simplification | Host the current review build locally rather than deploying it. Also export a Nautilus-friendly image tree organized by provisional Anchor/One-axis/Composed review depth, then family and condition, so map distributions can be reviewed outside the application. The legacy B0a depth grouping remains visibly provisional. A hosting project may be reserved, but no production version is saved or deployed until explicitly requested. |
| `R-20260727-40` | Accepted and executed as non-admission tooling | The local B0a preview exports 256 design-input scenarios across 16 legacy cells and 144 source groups, with 1,792 deterministic layer graphics, 16 overview sheets, 256 individual composites, and zero sealed assets. It provides pair/layer navigation, distribution, curriculum-exposure, and receipt views plus deterministic review-decision JSONL import/export bound to exact release and scenario hashes. Format validation passes; live-geometry Static validation, Witness validation, canonical S2 identities, policy results, and Lorenzo's actual review remain pending. The provisional depth folders are not the S2 pilot or a frozen release. |
| `R-20260727-41` | Accepted and executed as review-UX correction | A map decision without an obvious comment path is not an adequate human-review record. Put the decision-and-comment editor first, label the primary field `Map comment`, state when it becomes editable, and preserve its text as the hash-bound `semanticNote` in deterministic JSONL export/import. |
| `R-20260727-42` | Accepted as an owner distribution finding | The current source-bank foundation examples look small relative to the available 64 x 64 site. This is quantitatively consistent with the 16 `f_osm_all` preview identities: 97-179 dig cells (median 155.5), only 2.37%-4.37% of the 4,096-cell raster. Keep small jobs as useful anchors, but do not claim foundation work-volume coverage from this preview. Before S2 foundation coverage is accepted, show target-area/required-volume support explicitly and add a deliberately larger-footprint candidate review slice. It becomes an admitted work-volume condition only after numeric support and the existing 450-step witness gate; this does not authorize PPO. |
| `R-20260727-43` | Accepted and executed; corrects R-41 interaction | A comment must not require a prior accept/reject/quarantine decision. The editor creates a hash-bound `terra-map-review-record-v2` with an optional decision, labels it decision-pending, persists and exports it immediately, and preserves the comment when a later decision is added. Import remains fail-closed on release, manifest, and scenario hashes. |
| `R-20260727-44` | Accepted bounded optimization probe; pending timing | On the frozen one-excavator protocol, skip construction and dilation of the truck-transfer cone when no other active truck exists. This is an exact early exit from an empty candidate set, not an approximate validator. Commit `9f7b95fa` preserves eager/JIT no-truck behavior and the active-truck transfer path. Run one new cost-only probe on the same pinned B0a identity, source group, state, rows, batch size, CPU host, and exact service kernel. Keep the change only if both warmed batch p50 and p95 improve by at least 5% over the pinned v1 control; otherwise revert it under the simplicity rule. A passing timing result authorizes the ordinary R-32 staged confirmation only, never the 256-scenario profile or PPO. |
| `R-20260727-45` | Rejected after controlled timing | The no-truck empty-candidate shortcut failed both preregistered timing gates on the same identity, source group, explicit state, 39,606 candidate rows, batch size, and CPU host: warm batch p50 worsened 14.1% (`0.072704` to `0.082937 s`) and p95 worsened 23.0% (`0.073481` to `0.090367 s`). Commit `90c83808` reverts `9f7b95fa` under the simplicity rule. The receipt is non-admission; do not run a confirmation, bank-scale profile, Static admission, or PPO from this treatment. |
| `R-20260727-46` | Accepted and executed as migration evidence | The B0a normalizer verifies the frozen 3,392-file checksum tree, derives the live protocol, preserves raw dtype-sensitive target identity while separately checking exact-loader values, and materializes one serialized reset state per canonical source group. Two clean runs at `affc0d92` are byte-identical across 256 rows and 144 groups. All affordable semantics, capacity, state, and migration-record checks pass, but every row remains design input with canonical Format false, Static null, and exact direct service pending; this is not S2 admission. |
| `R-20260727-47` | Accepted and executed as controlled visual evidence | The fresh train-only apron review contains 32 exact-dig `slcap03_04`/`slcap07_10` pairs and 64 maps. Capacity spans 3.2500-3.2547x versus 8.5000-8.5035x with zero paired p50/p95 separation delta; all 427 manifested files and the independent verifier pass. The 140-189-cell work range occupies only 3.42%-4.61% of the map, confirming R-42. Treat this as a capacity comparison only; exact Static validation, Lorenzo's review decision, the larger-footprint slice, and S1 Capacity completion remain open. |
| `R-20260727-48` | Accepted bounded device treatment; pending parity | Before more validator code, test the unchanged exact service kernel on the local single RTX 4090. First replay the pinned 18 accepted rows and require exact dtype/shape/leaf-count/content parity with CPU candidate SHA-256 `b19744ab...dd56` and output SHA-256 `fa8dd4f8...106`. Leave `JAX_PLATFORMS` unset and assert exactly one NVIDIA GPU because explicit `JAX_PLATFORMS=gpu` currently enters a broken ROCm registration path. Only parity authorizes one batch-4 cost-only probe on the same identity, source group, state, and inputs. Continue to ordinary R-32 confirmation only when one-scenario p95 is at most 3,600 s, sequential 256/448 p95 is at most 24/48 h, and host plus normalized accelerator peak memory are each at most 80%. Any parity/cost failure rejects the device path; do not change exact semantics, revise the sequential gate, vectorize headings, shard scenarios, run bank admission, or launch PPO from this treatment. |
| `R-20260727-49` | Accepted and executed as local review tooling | Site commit `91ccf87` adds the verified 64-map/32-pair controlled-capacity bank as the default local review dataset while retaining B0a behind a bank selector. The UI labels it exact-loader-valid, Static-pending, non-admitted, controlled-capacity-only, and small-footprint. Comments are enabled before a decision, persist across reload, and export/import against exact release/manifest/scenario hashes. This changes no scientific receipt and authorizes no admission, witness, evaluation, or PPO claim. |
| `R-20260727-50` | Exact GPU parity passed; cost treatment pending | Tool commit `1856f92a` replayed the pinned first 18 accepted rows on exactly one RTX 4090 with `JAX_PLATFORMS` unset. All three `int32` leaves matched the frozen CPU dtype/shape/leaf/content hash `fa8dd4f8...106`, with candidate hash `b19744ab...dd56`; diagnostic GPU/host peaks were below 80%. The first interactive launch received external exit `143` before emitting a receipt, while the successful retry ran under a user service. A Newton benchmark started after that launch, so exact parity remains valid but the receipt is not cost evidence. Receipt `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_gpu_parity_20260727_v1/direct_service_gpu_parity.json`, SHA-256 `6eee6e354c77924867a43f5311cabd029cc50e736f2ebd3360e25a0aea9baef1`, authorizes only one unchanged batch-4 cost probe after the GPU is uncontended; it authorizes no confirmation, bank profile, Static admission, or PPO. |
| `R-20260727-51` | Accepted and executed as narrow visual evidence | Commit `f60d0e52` builds a fresh train-only procedural large-foundation review slice after excluding 112 canonical digs used by the matched-source audit, B0a foundations, and the controlled-capacity bank. Sixteen selected groups come from 160 eligible fresh candidates and cover 328-339 required cells (8.01-8.28% of the 64 x 64 site), with all-around dumping, no obstacles, and 11.08-11.49x single-layer capacity. A deterministic rebuild exact-compares every JSON, NPY, PNG, README, provenance, and manifest file, followed by exact-loader smoke checks; the 123-file manifest has SHA-256 `1829c92b13119fcf726a1f572b87bf6096fdb5b7819eeabb191043b59a6febda`. Artifact `/home/lorenzo/moleworks/.artifacts/terra_pilot_large_foundation_review_20260727_v1/` is Static/witness pending and non-admitted. It is a narrow 8.0-8.3% generator tail, not broad 8-12% support, not an OSM-matched causal comparison, and leaves a true >=10% slice pending. |
| `R-20260727-52` | Pure-GPU exact validator rejected after cost receipt; hybrid treatment accepted next | The batch-4 cost probe passed all numeric gates: one-scenario p95 315.702 s, sequential 256/448 p95 6.99/12.19 h, normalized GPU/host peak 0.8653%/4.5182%, with the host denominator supplied by the same-host parity receipt. Identity, inputs, protocol, stable reset state, and all twelve core validator hashes match the CPU control. Nevertheless, the full graph/prefilter population differs by device before padding: reachable poses are 31,366/31,418 CPU/GPU, pose/cabin rows are 376,392/377,016, and accepted service candidates are 39,606/39,520. Thus R-50's first-18-row service check was insufficient for complete-validator parity. Receipt `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_gpu_cost_probe_20260727_v1/validation_cost_probe.json`, SHA-256 `1250bdf0a723f5d739fce1dd6ccf90c9ea4eece5ece580da08fd3fd18e2971e1`, is useful timing evidence but authorizes no confirmation, bank profile, Static admission, or PPO. The next bounded treatment keeps graph traversal and dig prefilter on canonical CPU, hashes and transfers the ordered unpadded `int32` candidates unchanged, and runs only `_service_batch` on GPU. Full-population candidate/counter/output/final-result parity against CPU confirmation `f4bc393a...aae7` is required before timing; the new receipt must also hash imported `terra/benchmark_protocol.py`. On failure, retain CPU exactness and consider process-level CPU scenario parallelism rather than changing geometry or weakening the gate. |
| `R-20260727-53` | Accepted and executed as separate local visual bank | Site commit `21d5526` adds the 16-map train-only narrow large-foundation slice as a third review bank while preserving the B0a and capacity JSON releases byte-identically. The UI calls it `Provisional work-size review candidate (not a level)`, exposes comments before decisions, and retains the source artifact's Static/witness/non-admission limits. Source manifest SHA-256 is `1829c92b...6febda`; review-data, review-manifest, release, scenario-manifest, and layer-tree SHA-256 values are `e3cf3cf8...7e59e`, `36835be8...38db7`, `6394895f...a648d`, `57dbef9a...ea8e`, and `cf341e4a...1c0a19`. Nine Python tests, TypeScript, production build, four browser workflows, source verification, independent review, and the loopback server pass. This visual slice is not broad 8-12% support, not causal to OSM, and leaves Static, witness, true >=10%, and admission pending. Its inherited metre separation uses stale 0.6875 m/tile metadata and must be regenerated at live 0.571428571428125 m/tile before admission; the site exposes tile separation only. |
| `R-20260727-54` | Accepted and executed as metadata-only repair | Commit `7af1d4f5` derives large-foundation p50/p95/max metre fields from the frozen live `edge_length_m / edge_length_px = 0.571428571428125` receipt and fails closed if benchmark and environment map-geometry receipts disagree. The clean v2 rebuild at commit `09893079` passes 16 focused/protocol tests, deterministic full-tree verification, exact-loader checks, Black, and diff checks. Across all 16 maps the 48 corrected metre scalars equal tile values times live tile size. Relative to v1, only identities, provenance, summary protocol hash, and checksum manifest change; all NPY/PNG bytes, selection, map/dig hashes, initial states, and non-metre identity content remain identical. Artifact `/home/lorenzo/moleworks/.artifacts/terra_pilot_large_foundation_review_20260727_v2/` has `files.sha256` SHA-256 `3def81e558ff9776bdb0e9c8e17969d0c2c91f6ee43d98e80b2d227cc01a79b0`, identities SHA-256 `bb689353...e4b80`, and environment-protocol SHA-256 `36e2035b...0664`. This closes only the stale conversion defect; Static, witness, broad size support, causal matching, and admission remain pending. |
| `R-20260727-55` | Accepted and executed as review-navigation clarification | Site commit `8340fbb` makes the supplied display-depth groups permanently visible above the review queue, orders the B0a preview as Anchor, One-axis, then Composed, shows exact map counts, and supports one-click group filtering. The same surface labels the groups `Visual depth only - not admitted levels`; isolated capacity and work-size banks remain review candidates rather than implied curriculum stages. TypeScript, nine exporter tests, the production build, four browser workflows, desktop visual inspection, narrow-layout inspection, and the loopback service pass. This is a review-UX clarification only and changes no map, receipt, admission state, or PPO authorization. |
| `R-20260727-56` | Accepted and executed as the S1 condition-registry freeze | Commit `847322e5` materializes `benchmark/pilot_v03.json` and its canonical `conditions.jsonl` projection: exactly eight conditions, balanced four/four by family, three Anchor roots, five One-axis nodes, literal prerequisite IDs, equal `0.125` evaluation weights, and the approved source/capacity/side/topology panels. Full mechanical IDs use foundation `v140_189` and trench `v68_77`; `vmatch` remains only in non-identity alias provenance. The verifier pins exact factor dictionaries, source gates, suite and panel membership, support receipts, weights, and all 435 files in the three external evidence trees. It names `maximum_centered_dihedral_iou` while leaving both family thresholds `UNSET` and blocking S2. The receipt-tree verifier, 32 focused/protocol/support tests, Black, byte compilation, whitespace checks, and independent re-review pass. Scenario materialization, Static, Witness, and PPO flags remain false. Registry/file SHA-256 values are `fe161ff7...0721b3`, `a69ca3d9...06b593`, `b942958a...584d1`, and `13c543e4...1b5f2`. |
| `R-20260727-57` | Hybrid exact validator rejected at the preregistered full-population gate | The clean `d66969fc` replay ran the confirmed CPU graph/prefilter and only `_service_batch` on one RTX 4090. Both full replays completed; the CPU final outcome reproduced confirmation `f4bc393a...aae7`, all CPU/hybrid population counters matched it, and ordered CPU candidates matched GPU dispatch and round-trip exactly. Full ordered GPU service outputs did not match CPU, and the hybrid final outcome therefore differed. The process exited `1` after 1,328 s wall, 42 min 3.754 s CPU, 5.5 GiB peak, and no swap/OOM. Because the harness writes only after success, no validator receipt or candidate artifact exists; the timestamped journal-derived failure record is `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_hybrid_parity_20260727_v1/hybrid_parity_failure.json`, SHA-256 `49638f9b48fdc479cba5a01554c0afdbd71c6429886bacc33c877d883b1bc110`. Reject hybrid timing and do not introduce float tolerances or a GPU-specific geometry contract. Retain the exact CPU validator and evaluate only external scenario-level CPU process parallelism next. No bank profile, Static admission, or PPO is authorized. |
| `R-20260727-58` | Preregistered fixed CPU-process scaling gate | Test one orchestration-only treatment: one fresh cohort of four long-lived spawned CPU workers, each making exactly two complete calls on the pinned confirmed scenario. Every call rematerializes the exact state; call one measures cold import/load/compile/execute and call two measures process-local warm throughput. Workers use `JAX_PLATFORMS=cpu`, no persistent compilation cache or new XLA/thread flags, and fixed disjoint affinity sets `0-3,16-19`, `4-7,20-23`, `8-11,24-27`, and `12-15,28-31` on the pinned 32-CPU starship topology. All eight complete outcomes and every counter must exactly equal CPU confirmation `f4bc393a...aae7`; state, protocol, input, dependency, harness, backend, PID, and affinity receipts are mandatory. For worker `i`, let `C_i` be launch through synchronized first result and `W_i` be rematerialization through synchronized second result. Freeze `P_N = max_i(C_i + (ceil(N/4)-1)*W_i)` and require `P_256 <= 86,400 s`, `P_448 <= 172,800 s`, every call <=3,600 s, four distinct successful PIDs, zero worker swap/OOM, and coordinator plus conservative summed worker peak RSS <=80% physical memory. One failure rejects the treatment without retry or width sweep. A pass authorizes only one deterministic four-worker 256-identity exact profile; that profile must itself finish within 24 hours and project 448 within 48 hours. Static, bank admission, and PPO remain false. |
| `R-20260727-59` | First R-58 service bootstrap classified execution-null; one corrected bootstrap allowed | Receipt `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cpu_process_probe_20260727_v1/cpu_process_probe.json`, SHA-256 `e09bb9719039754a64e8ca50286d2b1cbfd24d55334fe38c3f2f395bb1d8ce21`, records all four workers exiting with the same `ModuleNotFoundError` because file-path execution placed `tools/` rather than the pinned worktree root first on `sys.path`. The coordinator stopped after `0.417 s`; there are no ready receipts, start barrier, call receipts, state/protocol/projection/gate/resource records, or exact entrypoint calls. Treat v1 as a null execution bootstrap, preserve it, and exclude all of its timing and memory values. This append-only decision overrides only v1's mechanical `authorizes_retry: false`: freeze module launch as `python -m tools.probe_b0a_direct_service_cpu_processes` from the same clean repository root, assert and receipt that direct-service, confirmation, and profile modules resolve to that root, and permit one fresh v2 output directory. Worker count, affinities, environment, state, exact entrypoint, timing equation, and gates remain unchanged. Once any corrected worker reaches readiness, no further cohort rerun is allowed. |
| `R-20260727-60` | Preregistered conditional 256-identity CPU profile | Run only after a passing R-58 v2 receipt is hash-pinned. Consume the frozen live-migration rows in lexicographic `legacy_map_id` order, assign index `i` to worker `i % 4`, and keep four persistent fixed-affinity CPU workers with 64 identities each. Decode each row's explicit serialized state; never regenerate development states through a public-train namespace. Each identity gets one duration from exact load through synchronized outcome and atomic receipt rename, stored afterward in its final worker ledger. Report nearest-rank p50/p95/max across 256 durations. For worker `i`, exclude its first identity and define `Q95_warm_i` over its 63 warm durations; freeze `P_448 = observed_256_makespan + 48 * max_i(Q95_warm_i)`. Makespan runs from immediately before first spawn through successful worker exits, verification of all 256 receipts, and atomic canonical merge. Require all 256 identities exactly once, exact confirmation-sentinel equality, every identity <=3,600 s, observed makespan <=86,400 s, `P_448 <=172,800 s`, fixed CPU backend/affinity, zero swap/OOM, <=80% conservative memory, and unchanged code/input receipts. Consume only the named one-shot R-58 authorization; require generic `authorizes_bank_profile` to remain false. No retries, resume, cache, alternate worker count, Static claim, admission, witness, or PPO. A failure writes partial evidence but no merged success output; any corrected rerun needs a new append-only decision and output directory. |
| `R-20260727-61` | Pre-execution persistence correction to R-60 | Replace R-60's imprecise word `rename` with the existing atomic no-replace hard-link publication primitive. Identity duration ends after the scenario receipt is durably published and is recorded later in the worker ledger. After all workers exit, write and verify a clearly named canonical candidate merge; include that work in observed makespan and compute every cost/resource/integrity gate before publishing the success filename. A failed profile may retain the candidate as partial evidence but must never emit `direct_service_results.jsonl`. This changes no population, timing equation, threshold, retry rule, or scientific authority. |
| `R-20260727-62` | Pre-execution protocol-provenance clarification | Preserve the migration protocol receipt (`affc0d921...`, `15e4d45f...`, EnvConfig `02863f62...`) as historical identity provenance and separately derive the execution protocol from the profile's clean committed revision. Require all workers to agree and require the EnvConfig/constants, code, packages, executable, authorization, confirmation, B0a identities/checksum/source/provenance manifests, and migration JSONL/summary to remain exact pre/post. Do not require the revision-dependent migration and execution protocol hashes to equal and do not let this cost sidecar mark a migration row Static-valid. This changes no map, state, outcome, threshold, or authority. |
| `R-20260727-63` | Pre-execution manifest-integrity clarification | R-62's pre/post manifest equality is necessary but not sufficient. Before success publication, re-verify every path named by frozen `files.sha256` against its recorded digest and reject missing, changed, or unmanifested consumed files. The no-replace hard-link publication is durable only because the source file is fsynced before linking and the containing directory is fsynced afterward; require both operations. This adds no new artifact format, map, timing threshold, or authority. |
| `R-20260727-64` | Pre-execution two-file commit clarification | Under the simple one-directory writer, `direct_service_results.jsonl` is never authoritative by filename alone. A usable sidecar requires both that file and a passing no-replace `validation_cost.json` whose recorded result hash matches it. Result publication is the final data-path operation after every scientific gate; the validation receipt is the commit marker. If the hard-link succeeds but its directory fsync or final commit-marker publication fails, preserve and receipt the orphan as non-authoritative partial evidence rather than destructively deleting it. This narrowly supersedes R-61's literal claim that a failure can never leave the success filename; it does not weaken any scientific gate or authorize retry, Static, admission, witness, or PPO. |

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
