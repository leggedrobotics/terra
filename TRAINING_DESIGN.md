# Terra Training Design

- Status: first paired screen complete; recovery contract ratified
- Version: historical `training_design_v4` plus `recovery_v1`
- Date: 2026-07-25 recovery update
- First target: one 64 x 64 tracked-excavator policy for foundations and
  trenches
- Production training authorized by this document: no

The original v4 sections below are retained as the preregistered experiment
contract. The completed result, critical review, corrected decisions, and
dependency-ordered backlog are in
[`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) and
[`TRAINING_TASKS.md`](TRAINING_TASKS.md). Those companion documents supersede
the original map ladder and execution order for new work; they do not rewrite
the historical treatment after seeing the result.

Post-screen result:

- flat peaked at M0 `24/64`, M1 `7/64`, and M2 `1/64`, then regressed to
  M0 `11/64`;
- staged peaked at M0 `13/64`, M1 `2/64`, and M2 `0/64`; and
- neither arm passed a family, cell, retention, or joint mastery gate.

## 0. Ratified recovery contract

The original M0-M2 ladder below is retained only as the frozen description of
the failed screen. It is outdated as a future map curriculum. The active
dependency-ordered plan is in [`TRAINING_TASKS.md`](TRAINING_TASKS.md).

The following decisions were ratified after reviewing the failure evidence:

1. **One visible dump mask.** The exact visible target dump mask is the only
   accepted dump region for action interpretation, capacity validation,
   completion, termination, reward, logging, and evaluation. There is no
   hidden one-cell tolerance.
2. **Contained, mass-conserving starter physics.** For the first scratch
   teachers and initial quantitative map cells, a dump aimed into the accepted
   region may redistribute soil only inside that region. It may form piles,
   but no mass may be clipped, deleted, or spilled across the boundary. A dump
   made entirely outside the accepted region remains a physically possible
   mistake; its soil remains outside and must be recovered. Boundary-spill
   dynamics are a later, separately named difficulty treatment.
3. **No greedy action veto.** Relocation potential may shape reward, but it
   must not prohibit a physically valid dump merely because the predicted
   potential increases.
4. **Scratch small-policy recovery.** E8 remains a historical zero-shot and
   lineage reference only. Recovery begins with independent
   `resnet_spatial_8x8` base policies, approximately 994,825 parameters,
   initialized from scratch: one foundation policy and one trench policy.
   Neither is called a teacher until it passes a source-disjoint family gate.
5. **Adequate scratch and long-run budgets.** Treat 500, 1,000, 2,000, and
   5,000 PPO updates as review milestones. Continue the exact lineage whenever
   a deterministic fixed bank gains one successful identity or `0.01` median
   terminal completion; do not use reward alone. Once the family/cell recipe
   passes twice, selected policies receive at least 20,000 updates on
   `gpuhe.120h` with a five-day request and continue in 20,000-update chunks
   while the fixed task metrics still improve.
6. **Separate causal programs.** Map progression, dense-reward design,
   dense-to-terminal reward progression, and partial-reset progression remain
   separate experiments. The first feasibility treatment uses one frozen
   corrected dense reward; it does not change reward and map difficulty
   together.

The first corrected dense reward removes the action veto and uses the exact
mask and contained transition above. A possible second dense treatment would
replace the current dump-time relocation term with one per-step
mass-distance potential over both off-zone soil and carried load. Its exact
equation and distance metric are still open decisions; it is not authorized
for implementation or training yet.

## 1. Decision

The final target is one policy that completes feasible foundation and trench
jobs across unfamiliar geometry, dump access, dump distance, site obstacles,
and partially completed states.

The program has two progressive curricula:

1. a **map curriculum**, from easy local earthmoving to complex geometry and
   constrained sites; and
2. a **reward curriculum**, from dense skill acquisition toward the true
   terminal task objective.

They are separate experiments. The first campaign changes only the map
distribution while holding the current dense reward and full-task resets
fixed. After selecting a map design, the first reward experiment freezes one
qualified foundation-only family and follows the current local draft
`/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_VALIDATION_PLAN.md`, after
the W0 corrections in `TRAINING_TASKS.md`.
Only a later confirmation run combines selected treatments.

The minimum first map experiment is:

```text
completed E8 parameters
        |
        +-- NEW-MAPS-FLAT
        |
        `-- NEW-MAPS-STAGED: M0 -> M1 -> M2
```

Initially launch only these two matched training arms. A family specialist or
far-dump probe is conditional on a specific failure. Do not begin with a large
family-certification ladder, a learned curriculum teacher, PLR, partial
resets, or simultaneous reward changes.

## 2. Corrections to the earlier proposal

### 2.1 What E8 is

E1-E10 are experiment runs, not curriculum stages or automatically useful
specialists. E8 is the only relevant multitask parent because it trains one
policy on foundations and trenches.

E8 is **not** a close-dump specialist. Its old levels are:

```text
foundations -> trenches/double -> trenches/double_diagonal
```

Those datasets permit dumping on nearly all legal free ground. They are
outdated for this study and must not appear in either new arm. E8 contributes
only compatible model parameters and the common model/PPO template.

The completed E8 checkpoint must be loaded as a **parameters-only warm start**:

- fresh optimizer and learning-rate schedule;
- update counter reset to zero;
- current Terra environment configuration;
- fresh RNG and curriculum state at M0; and
- no inherited old map paths.

Do not use the normal resume path. It restores the optimizer, update counter,
and checkpoint environment configuration, which would turn this into an
invalid continuation of the old campaign.

### 2.2 There is no predefined "close specialist"

The earlier "close specialist" meant a hypothetical new family-only run. It
does not refer to E1-E10. The term is removed.

If fixed evaluation later shows that one family fails while the other passes,
train exactly one conditional `SPECIALIST-FOUNDATION` or
`SPECIALIST-TRENCH`. Until then, a specialist adds cost without answering the
first question.

### 2.3 Update and transition units

Use PPO updates as the primary budget unit. At the E8 rollout shape:

```text
32 rollout steps x 1024 environments/device x 4 devices
    = 131,072 global environment transitions/update
```

Therefore:

| PPO updates | Global transitions |
|---:|---:|
| 1,000 | 131,072,000 |
| 2,000 | 262,144,000 |
| 4,000 | 524,288,000 |
| 20,000 | 2,621,440,000 |

The earlier `2.5B` meant approximately 2.5 billion global transitions, or
19,073 updates at this rollout shape. That is almost a full E8 run and is
removed from the initial feasibility campaign.

### 2.4 Existing E8 metrics are not the new-map gate

E8's inline SWHiR samples the training environments' current curriculum
distribution and uses a shorter default evaluation horizon than the 450-action
training horizon. A pooled value near one does not certify foundations,
trenches, or the new constrained maps separately.

Every decision below uses fixed, source-disjoint, family-stratified evaluation
banks with a declared 450-action horizon. Reward return and pooled inline SWHiR
remain diagnostics only.

## 3. Frozen causal contract

The flat and staged arms share:

- the same completed E8 parameters-only parent;
- current Terra and terra-baselines revisions;
- model architecture, PPO settings, action mask, and observation contract;
- current dense reward with `apply_trench_rewards: false`, matching E8;
- untouched full-task resets;
- 450-action horizon;
- generator revision and terminal map universe;
- training seed;
- development and sealed-test manifests;
- 4,000-update exploratory budget; and
- evaluation cadence.

Only map exposure differs.

Changing trench shaping, reward coefficients, partial-reset probability,
horizon, architecture, optimizer state, or action masking creates a different
experiment. In particular, do not enable `apply_trench_rewards` in only one
new arm.

Before a launch, record hashes for:

```text
Terra revision
terra-baselines revision
generator and validator
train/dev/test manifests
E8 checkpoint
environment and PPO configurations
```

## 4. Map curriculum

### 4.1 Difficulty axes

Map difficulty is factored into four independently recorded axes:

1. **geometry**: foundation connectivity or trench segment, axis, and junction
   structure;
2. **dumping**: side access, zone fragmentation, path distance, and capacity;
3. **site**: objects, roads, walls, and combined constraints; and
4. **work amount**: excavation volume and required soil transport.

Do not call a map simply "easy", "medium", or "hard" without its axis metadata.
In particular:

- trench hardness records segment count, axis count, junction count, and the
  degree of each junction; T and X cases are separate strata;
- curved trenches are excluded from the first program;
- disconnected foundation strips and pads are a named structural class;
- dump distance is traversable shortest-path distance, not straight-line
  distance through obstacles; and
- dump capacity is recorded separately from distance.

Every admitted map must pass:

- strict Terra loading and 64 x 64 shape checks;
- target, obstacle, road, and dump-mask consistency;
- mass and reachable dump-capacity checks;
- valid spawn and static workspace coverage;
- declared geometry and junction/component counts; and
- train/dev/test source-disjointness.

Static validity is not dynamic feasibility. A hard family enters the terminal
deployment distribution only after a bounded policy or planner produces a
legal, mass-conserving completion witness. M0-M2 are candidate training
families whose first screen provides that evidence; static validation alone
does not qualify them for deployment.

Foundation geometry labels mean:

- `OSM-like`: a footprint derived from the existing source-shape bank;
- `connected procedural`: a generated connected footprint with varied
  orientation, wings, or segmentation; and
- `structural`: disconnected bearing strips, isolated pads, or pillar-like
  footings.

### 4.2 First local ladder

The first comparison uses only nearby, generously sized dump regions. Each
difficulty stratum is 50% foundations and 50% trenches.

| Stratum | Geometry | Dumping | Site | Explicitly excluded |
|---|---|---|---|---|
| `M0 easy local` | OSM-like foundations; straight one-axis trenches with zero junctions | foundation all-around or large nearby apron; trench broad both-side or one large nearby side | light | separated/tight zones, procedural or structural foundations, intersections, obstacles, far and haul-away |
| `M1 local access` | connected procedural foundations; straight or segmented trenches with zero junctions | large nearby apron, broad one-side, or separated nearby zones | light or scattered objects | disconnected structural foundations, trench intersections, combined sites, far and haul-away |
| `M2 local constrained` | connected procedural foundations with greater shape variation; T or X trenches with exactly one junction | all-around, broad nearby, one-side, irregular nearby, and separated nearby; still generous | scattered objects, access road, or gapped wall individually | structural foundations, two-or-more-junction trenches, curved trenches, combined sites, tight capacity, far and haul-away |

Each train, development, and test manifest freezes exact counts for every
listed geometry x dump-access x site subtype. Use no undefined "small share."
The first development and sealed banks use exactly four primary cells per task
family and eight episodes per cell:

| Stratum/family | Four separately gated primary cells |
|---|---|
| M0 foundation | all-around x low volume; all-around x normal volume; large apron x low volume; large apron x normal volume |
| M0 trench | one straight segment x both-side x low volume; one straight segment x both-side x normal volume; one straight segment x one-side x low volume; one straight segment x one-side x normal volume |
| M1 foundation | connected procedural x apron x light; connected procedural x one-side x light; connected procedural x separated x light; connected procedural x one-side x scattered objects |
| M1 trench | straight x one-side x light; end-to-end segmented x both-side x light; end-to-end segmented x one-side x scattered objects; end-to-end segmented x separated x scattered objects |
| M2 foundation | connected irregular x apron x scattered objects; connected irregular x one-side x access road; connected irregular x separated x gapped wall; connected irregular x irregular-nearby x access road |
| M2 trench | T x both-side x scattered objects; T x one-side x access road; X x both-side x gapped wall; X x one-side x gapped wall |

An end-to-end segmented trench may bend but has no branch junction. T and X
cells both have one junction but are separated by junction degree. Training
uses the same four-cell balance within each stratum/family. Report and gate
every cell; do not let the pooled task-family result hide one.

Starting capacity contract:

- every F0 starter identity and every later B0 feasibility-panel identity has
  at least `3x` reachable single-layer soil capacity;
- the capacity validator additionally proves complete valid bucket loads fit
  the contained-pile `int8` representation;
- foundations with all-around dumping retain all legal non-dig free ground;
  and
- tighter 2.0x/2.5x capacity variants are later isolated treatments, never
  silently mixed into the initial geometry or distance curriculum.

Starting distance contract:

- for every dig-boundary work cell, compute the shortest 8-connected path to
  its nearest permitted dump cell through traversable non-obstacle cells, with
  cardinal cost 1 and diagonal cost `sqrt(2)`;
- M0 median traversable dig-to-dump path no more than 6 tiles, 95th percentile
  no more than 10, and maximum no more than 12;
- M1 and M2 median no more than 10 tiles, 95th percentile no more than 14, and
  maximum no more than 18;
- within the applicable maximum radius, reachable dump capacity must still
  meet the capacity ratio in the preceding contract;
- record reachability and capacity for every declared dump component or side,
  but do not reject an easy map merely because it also permits harmless distant
  cells that are not needed to meet the nearby-capacity contract; and
- record both tiles and metres using the frozen map scale.

These are generator rejection gates, not reward terms. Capacity is intentionally
loose in this campaign so that distance, geometry, or site access is not
confounded with a barely large-enough dump zone.

### 4.3 Fixed banks

The local visual audit under
`../.artifacts/terra_map_audit_20260723/procedural_v1/review_set_006/`
is a design reference, not a training dataset. Regenerate source-disjoint banks
from the accepted procedural contracts after the final visual review.

Use one frozen terminal training pool:

| Split | Size | Composition |
|---|---:|---|
| train | 256 map identities | 64 M0, 96 M1, 96 M2 |
| development | 64 per stratum | 32 foundations, 32 trenches |
| sealed test | 64 per stratum | 32 foundations, 32 trenches |

The 256 training identities form the common terminal universe. Materialize
equal-size level directories because the current loader stacks levels and
samples uniformly within each one:

| Directory | Slots | Sampling content |
|---|---:|---|
| `local_M0` | 256 | M0 identities only |
| `local_M1` | 256 | M1 frontier identities only |
| `local_M2_terminal` | 256 | 25% M0, 37.5% M1, 37.5% M2 |
| `local_flat` | 256 | byte-identical to `local_M2_terminal` |

When a directory needs repeated slots, preserve the source `map_id` in the
manifest and report both slot count and unique-identity count. Do not silently
substitute newly generated maps between arms.

Generation must shuffle deterministically before writing because the current
loader takes the first configured number of maps. All level arrays must have
the same count and dimensions.

## 5. First experiment set

### 5.1 Preflight evaluation

After E8 finishes and passes its terminal run audit:

1. freeze one E8 checkpoint and hash it;
2. run parameters-only inference on all fixed M0 development maps;
3. report foundations and trenches separately at horizon 450; and
4. run a finite 10-update warm-start smoke on each new arm.

This zero-shot screen establishes the transfer gap. It does not need to pass
80% to permit adaptation, but any observation/action incompatibility,
non-finite update, stale map path, or loader failure blocks training.

### 5.2 Primary matched arms

| ID | Treatment | Initial budget | Question |
|---|---|---:|---|
| `NEW-MAPS-FLAT` | `local_flat` from update zero | 4,000 updates | Can E8 adapt by ordinary sampling? |
| `NEW-MAPS-STAGED` | per-environment `local_M0 -> local_M1 -> local_M2_terminal` | 4,000 updates | Does progressive exposure improve learning speed or worst-family success? |

The first screen uses one paired seed. Add two more paired 4,000-update seeds
only if at least one arm reaches the development gate. If neither arm reaches
it by update 4,000, this screen fails and triggers the conditional
family-specific diagnosis. Do not add an unregistered post-result extension.

### 5.3 Online exposure promotion and demotion

Use Terra's existing per-environment curriculum manager:

```yaml
increase_level_threshold: 3
decrease_level_threshold: 3
last_level_type: none
```

Interpretation:

- three consecutive successful episodes promote that environment one level;
- three consecutive failed episodes demote it one level;
- M0 and M1 promotion evidence comes only from their frontier directories; and
- the final level never randomly sends environments back to arbitrary levels.

This symmetric `3/3` controller keeps environments near their competence
frontier. The M2 terminal directory restores the earlier-family mixture. This
replaces E8's sticky `20/80/random` defaults for this experiment.

> Post-screen correction (2026-07-24): the competence-frontier interpretation
> is rejected. The controller is a per-environment streak-based exposure
> heuristic over randomly sampled identities. Its final occupancy promoted
> most environments to M1 while fixed held-out M1 remained unsolved. It is not
> the selected curriculum; see `TRAINING_TASKS.md`.

Before using this controller, fix the current terminal/reset ordering so the
terminal outcome updates the level **before** the replacement map is sampled.
Add deterministic tests showing that:

- an M0 promotion makes the immediately following episode sample M1; and
- an M1 demotion makes the immediately following episode sample M0.

The staged launch is blocked until this transition is atomic.

The online controller changes training exposure only. It does **not** make a
scientific mastery claim. Log normalized level occupancy over all devices; use
the fixed evaluator for per-level and per-family success. The existing
first-device-only histogram is insufficient. Online map-family logging is
optional until map provenance is carried correctly through terminal
transitions.

Do not add a second recovery controller in the first implementation. Online
demotion and the final M2 replay mixture are the recovery mechanisms. If
held-out earlier levels regress, the run fails its retention gate; do not
repair it with an unregistered schedule change.

### 5.4 Scientific mastery and retention

Evaluate the fixed development banks every 500 PPO updates. For each stratum,
evaluate 32 foundations and 32 trenches as untouched initial episodes at the
450-action horizon.

A stratum is mastered after two consecutive evaluations with:

- at least 26/32 successes for foundations;
- at least 26/32 successes for trenches;
- at least 6/8 successes in every declared primary subtype;
- no previously mastered family more than five percentage points below its
  recorded mastery value; and
- zero integrity failures.

Integrity failures include invalid action execution, mass mismatch, target or
obstacle corruption, non-finite state, and evaluator disagreement about true
termination.

Primary comparison metrics are:

1. worst stratum x family success;
2. updates to joint foundation-and-trench mastery;
3. retention on earlier strata; and
4. final sealed-test success after model selection.

Return, SWHiR, episode length, and curriculum occupancy are diagnostics.

Across the three paired seeds, a treatment qualifies only if at least two
seeds reach joint mastery by update 4,000. Select staged for the next program
only if it qualifies and either:

- its paired median updates-to-mastery is at least one 500-update evaluation
  interval lower; or
- its paired median worst-stratum success is at least 10 percentage points
  higher;

and no paired family median regresses by more than five percentage points.
Otherwise select flat only if flat qualifies. If neither qualifies, select
neither. One exploratory seed decides whether replication is warranted; it is
not a state-of-the-art performance claim.

### 5.5 Conditional diagnostics

Run no specialist initially.

If one task family is below the gate while the other passes, train only the
failed family for at most 2,000 updates from the same E8 parameters-only
parent:

| Outcome | Interpretation |
|---|---|
| specialist learns | likely multitask interference or mixture imbalance |
| specialist also fails | family, horizon, action feasibility, reward, or representation remains unresolved |

Use E8 as the trench-specialist parent. E6 may be evaluated as a
foundation-only reference, but its old all-around maps do not make it a
constrained-foundation specialist.

If both families fail, do not launch two full specialists. First overfit one
fixed M0 foundation and one fixed M0 straight trench for at most 500 updates
each:

| Outcome | Next action |
|---|---|
| either fixed identity fails | stop and diagnose environment, action feasibility, horizon, dense reward, or warm start |
| both fixed identities learn | inspect the weakest held-out stratum and authorize at most one 2,000-update family specialist |

A fixed identity is learned only after two evaluations with at least 29/32
successful held-out reset episodes at horizon 450 and zero integrity failures.
A family specialist must pass the applicable family and subtype gates in
Section 5.4.

After local M0-M2 succeeds, run `FAR-ONE-MAP`:

1. choose one locally solved identity;
2. create a paired remote version with identical dig geometry, site, spawn,
   horizon, and reachable dump capacity;
3. change only traversable dump path distance;
4. verify that the near counterpart is solved; and
5. allow at most 1,000 updates to overfit the far identity.

The far identity uses the same fixed-identity `29/32` gate.

Only a successful fixed far identity justifies a small far family. Only a
successful small family justifies adding remote dumping to the main
curriculum. Natural tight-capacity remote zones come after the
capacity-matched distance test.

## 6. Path to the final map distribution

The first M0-M2 comparison deliberately stops before the hardest
compositions. Later stages are admitted one at a time:

| Later stage | Added difficulty | Admission evidence |
|---|---|---|
| `M3 topology` | larger share of disconnected structural foundations and trenches with two or more junctions | family-stratified local-dump success |
| `M4 constrained site` | stronger individual obstacles, then combined road/wall/object scenes | capacity, access, and dynamic completion witnesses |
| `M5 remote transport` | medium then far one-side dumping with matched capacity; tight natural zones last | paired near/far fixed-map and small-family success |
| `M6 deployment mixture` | frozen realistic mixture of qualified geometry x dump x site factors | sealed cross-product evaluation with no failed family hidden by pooling |

At each addition:

- retain 20-30% sampling from mastered easier strata;
- compare against the frozen target mixture;
- promote foundations and trenches separately;
- stop rather than silently relaxing capacity, horizon, or success thresholds;
  and
- keep rejected or unproven families in a named challenge bank.

> Post-screen correction (2026-07-24): `20-30%` is an untested rehearsal
> hypothesis, not a default. Future mixture weights are selected by the
> retention fork and global promotion-bank experiment in `TRAINING_TASKS.md`.

"Arbitrary" in the final objective means broad procedural coverage inside a
declared feasible support. It does not mean sampling maps that violate dump
capacity, access, or Terra action reachability and asking PPO to absorb the
invalidity.

## 7. Progressive reward curriculum

Map and reward progression answer different questions:

| Curriculum | Changes | Holds fixed | Primary decision |
|---|---|---|---|
| map | geometry, dump, and site exposure | current dense reward, full resets, PPO/model | which maps can be learned and whether staging helps |
| reward | dense-to-terminal objective sequence | one qualified fixed foundation family, full resets, PPO/model | whether the bridge improves the true objective on foundations |

After the map experiment:

1. select one qualified, fixed foundation-only family from the chosen map
   design;
2. freeze that family and qualify its dense parent;
3. execute the authoritative paired experiment:

```text
A: dense_skill continuation
B: dense_skill -> terminal_objective
C: dense_skill -> terminal_margin -> terminal_objective
```

4. select checkpoints lexicographically by true success, productive workspace
   cycles, then steps; and
5. label the result foundation-only;
6. validate the selected reward sequence separately on trenches and the
   multitask mixture; and
7. run a combined confirmation only after both independent experiments decide.

The equations, correctness gates, cohort rules, and first foundation-only
scope remain in the local drafts
`/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_CURRICULUM.md` and
`/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_VALIDATION_PLAN.md`, subject
to the W0 corrections in `TRAINING_TASKS.md`.
Reward return never promotes a map level.

## 8. Partial-completion resets

Partially completed geometry is a reset/start-state curriculum, not a map
difficulty level. Keep it out of the first flat-versus-staged comparison and
the first reward validation.

After selecting the map sampler, compare full resets against a declared
20-30% minority of mass-conserving partial resets, initially emphasizing 50%
and 75% `in_zone` completion. Keep untouched full-task evaluation separate and
primary. A policy that solves only partial resets has not demonstrated
full-task feasibility.

## 9. Minimum implementation

The first campaign requires only:

1. three generated map directories, with the flat preset pointing directly to
   `local_M2_terminal`;
2. train/dev/sealed manifests with capacity and shortest-path metadata;
3. two YAML training presets;
4. a new explicit parameters-only warm-start path, because the current resume
   flags still restore optimizer, update, and campaign state;
5. an exact-manifest fixed-bank evaluator that preserves level identity,
   enumerates every requested map once, and uses horizon 450, because the
   current inline and offline evaluators do not implement this gate;
6. all-device level occupancy logging and fixed-evaluation per-family success
   reporting; and
7. small tests for loading, source disjointness, `3/3` transitions, no random
   final level, and finite warm-start updates.

Also fail loudly unless every generated `distance/img_*.npy` exists, has the
declared shape, finite values, metric, and normalization. Keep the new
traversable shortest-path eligibility statistics as manifest metadata
separate from Terra's reward distance field. Replacing or rescaling the reward
field would change the dense reward and invalidate the map-only comparison;
silently substituting zeros for a missing distance file is not acceptable.

Do not implement for the first campaign:

- PLR or a global adaptive sampler;
- a generic curriculum framework;
- ALP-GMM, PAIRED, ACCEL, or a learned map generator;
- simultaneous reset and reward schedules;
- curved trenches;
- remote or haul-away maps in M0-M2; or
- automatic relaxation after failure.

This follows the useful part of
[self-paced deep RL](https://proceedings.neurips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html):
approach a fixed target distribution gradually and compare against sampling
that target directly.

[Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
is the first adaptive method worth testing later if staged exposure wins but
uniform within-level sampling becomes the measured bottleneck. It must retain
a uniform floor and fixed target-distribution evaluation because learning
priority does not prove physical feasibility or deployment relevance.

## 10. Execution order

1. Freeze the accepted generator contract and visually review regenerated M0,
   M1, and M2 galleries.
2. Generate and statically validate source-disjoint manifests.
3. Wait for E8 to finish; audit and hash the selected checkpoint.
4. Run fixed M0 zero-shot evaluation.
5. Implement only the minimum paths in Section 9 and pass CPU tests.
6. Run paired 10-update finite smokes.
7. With separate authorization, launch one paired exploratory seed of
   `NEW-MAPS-FLAT` and `NEW-MAPS-STAGED`.
8. Apply the preregistered development gates and conditional diagnostics.
9. Confirm the selected map treatment with paired seeds.
10. Freeze one qualified foundation-only family and begin the separate first
    progressive-reward validation.
11. Evaluate partial resets separately.
12. Run one combined confirmation before claiming the final curriculum.

## 11. Launch gate

No production run starts until all answers are yes:

- Are every old E8 map path and old curriculum level absent?
- Is the E8 checkpoint loaded parameters-only with a fresh optimizer and
  update counter?
- Are flat and staged terminal banks identical?
- Are train, development, and sealed sources disjoint?
- Do all maps meet declared capacity, path-distance, and integrity contracts?
- Are fixed evaluations stratified by foundation/trench and map difficulty?
- Does evaluation use the declared 450-action horizon?
- Are all devices included in curriculum logging?
- Did both 10-update smokes produce finite updates?
- Is the exact launch separately authorized?
