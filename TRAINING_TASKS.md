# Terra Training Tasks

- Status: active diagnostic and implementation backlog
- Date: 2026-07-25 failure-audit update
- Governing design: [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md)
- Failure evidence: [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)
- Current policy baseline: E8-compatible `resnet_spatial_8x8_se`
- Production training authorized by this document: no

## 1. Current decision

The first paired map-curriculum screen is complete. Neither treatment is
selected:

| Treatment | Best M0 | Best M1 | Best M2 | Decision |
|---|---:|---:|---:|---|
| flat terminal mixture | 24/64 | 7/64 | 1/64 at any one checkpoint | reject |
| per-environment `3/3` staged | 13/64 | 2/64 | 0/64 | reject |

The flat policy learned useful source-disjoint M0 behavior and then regressed
from 24/64 at update 1,000 to 11/64 at update 4,000. The staged policy promoted
most training environments to M1 but never demonstrated held-out M1
competence. This is evidence against the tested exposure scheduler, not
evidence that curriculum learning in general is ineffective.

Do not launch another broad M0-M2 generalist, reward-curriculum run, partial
reset treatment, or architecture sweep yet. The next work must answer, in
order:

1. Is the current task/reward contract internally consistent?
2. Are existing policies memorizing training identities or merely evaluated
   differently online and offline?
3. Can the current action and observation contract learn one easy foundation
   and one easy trench?
4. Can each M0 family generalize when trained alone?
5. Does heterogeneous exposure cause the observed M0 regression?
6. Only then: what global map curriculum and observation/model treatment
   should be tested?

## 2. Critical assessment of the Oracle review

The Oracle review is useful because it separates confirmed code defects from
hypotheses and does not respond to poor learning with an architecture sweep.
Its central recommendation—prove semantics and dynamic feasibility first—is
sound. It needs the following corrections and additions.

| Oracle claim | Assessment | Consequence |
|---|---|---|
| Termination and terminal completion use different dump masks. | Confirmed. Termination accepts a one-cell dilation while completion and reward count exact target cells. | This is a correctness blocker, but the repair must first decide which mask is legally authoritative. Blindly widening reward would hide the map's visible dump constraint. |
| The `3/3` controller is not a competence curriculum. | Confirmed and strengthened by the final staged results. It pools outcomes from random identities, families, and cells. | Retire it as the candidate curriculum. Replace it with global checkpoint-bounded stages and fixed promotion-bank gates. |
| M0-M2 static validation does not prove dynamic feasibility. | Confirmed. | Run fixed-identity learning probes before another family or generalist campaign. |
| Flat suffered catastrophic forgetting. | Too strong. It showed held-out regression without ever mastering M0, under a stationary mixture. | Call it retention/generalization regression. Separate memorization, mixture interference, and PPO drift. |
| Positive soil height clipping may alias important states. | The clipping is confirmed; its causal importance is not. Local features may recover some nearby volume. | Require a paired-state alias test before changing observation preprocessing. |
| Random initial `env_steps` violates the full-reset contract. | Confirmed, but it affects only the first episode of each training process and cannot explain late regression alone. | Remove it for future full-reset experiments; do not claim it explains the failed screen. |
| No architecture experiment should be next. | Correct. Existing old-map results are saturated and do not rank architectures on the new distribution. | Retain `_se`; xattn is the first conditional architecture ablation only after semantic, feasibility, and retention gates. |
| Progressive reward v2 is ready after dense qualification. | Not yet. Its current Stage-1 legacy-parity requirement conflicts with its corrected completion invariant. | Revise Stage 1 to use a newly frozen corrected dense contract before implementing Stage 2 or Stage 3. |
| The staged result was incomplete. | Stale at review time. The evaluator later completed. | The final result strengthens rejection: staged peaks at M0 13/64, M1 2/64, M2 0/64. |

The review also underweights two cheap, high-information diagnostics:

- evaluate exact training identities and source-disjoint development identities
  with the same checkpoint and action-selection mode; and
- compare deterministic argmax with repeated sampled-action evaluation on a
  small selected checkpoint set.

Both should precede new PPO training.

The current data volume is another material weakness: training has only eight
unique M0 maps and twelve unique M1/M2 maps per primary cell. Repeated slots
balance sampling but do not create procedural diversity. Larger source-disjoint
training banks become mandatory after single-map feasibility is established.

## 3. Experimental rules

1. Preserve the historical revisions and reward semantics for retrospective
   diagnosis. Do not mix a semantic fix into a continuation intended to explain
   the old run.
2. Use the corrected contract for every future-policy feasibility, specialist,
   curriculum, reward, reset, and architecture experiment.
3. Change one causal factor per comparison.
4. Use untouched 450-step full resets with `env_steps == 0` unless reset
   distribution is the named treatment.
5. Keep map, reward, reset, and architecture curricula separate.
6. Use fixed source-disjoint banks for decisions. Online success, return,
   entropy, and curriculum occupancy are diagnostics.
7. Treat single-map learning as a dynamic feasibility witness, never as a
   generalization claim.
8. Fail loudly on missing maps, metadata, distance fields, or evaluator
   integrity fields. Do not add fallback loaders or a generic curriculum
   framework.
9. Do not use the development or sealed bank to drive automatic promotion.
   Promotion uses a separate frozen source-disjoint gate bank.
10. Every training task below requires its own update-1 finite smoke and exact
    run receipt before a production launch.

## 4. Dependency graph

```text
D0 close first screen
 |
 +--> D1 semantic replay audit --------+
 |                                     |
 +--> D2 train/dev + action-mode audit |
                                       v
                              C0 legal dump decision
                                       |
                              C1 unified completion
                                       |
                         C2 reset/loader/evaluator gates
                                       |
                              O0 observation alias tests
                                       |
                         F0 two fixed-identity probes
                           |                         |
                        fail                       pass
                           |                         |
                 diagnose environment        +-----+------+
                                             |            |
                                      R0 historical   B0 larger banks
                                      retention fork       |
                                                   F1 family-bank probes
                                                           |
                                                  G0 corrected M0 parent
                                                           |
                                        M0 global checkpoint-gated ladder
                                      /              |             \
                              reward study     partial resets   architecture
```

`R0` is the sole historical-revision experiment. `F0` and every other
future-policy task use the corrected contract.

Task index:

| ID | Priority | Cost class | Depends on | State |
|---|---|---|---|---|
| D0 | P0 | documentation | completed evaluators | complete |
| D1 | P0 | evaluation only | D0 | open |
| D2 | P0 | evaluation only | D0 | open |
| C0 | P0 | decision | D1 | open |
| C1 | P0 | Terra code/tests | C0 | open |
| C2 | P0 | baselines code/test | C1 | open |
| C3 | P0 | Terra loader/tests | C1 | open |
| C4 | P0 | evaluator/tests | C1-C3 | open |
| C5 | P0 | training receipts/tests | C1-C4 | open |
| O0 | P1 | deterministic tests | C1-C5 | open |
| F0 | P0 | two bounded PPO probes | C1-C5, O0 | blocked |
| R0 | P1 | two 500-update historical forks | D1, D2, F0 | blocked |
| B0 | P1 | generation/validation | F0 | blocked |
| F1 | P1 | two bounded family specialists | B0, F0 | blocked |
| G0 | P1 | one corrected M0 generalist | F1 | blocked |
| M0 | P2 | staged generalist campaign | G0 | blocked |
| A0 | P3 | one conditional A/B | G0 plus representation evidence | blocked |
| W0-W2 | P2 | specification then reward A/B/C | G0 | blocked |
| P0 | P3 | reset A/B | selected map sampler | blocked |

## 5. Phase D — close and diagnose the completed screen

### D0 — Freeze the first-screen receipt

Status: complete.

Record:

- Terra revision `d37e780480c0fae64a4b9e4ba6638b4499748761`;
- terra-baselines revision `2722d832c8381a68d594d8bf8298ba3aec7f4c6a`;
- flat job `8398905`, W&B `mwn0d0cr`;
- staged job `8398906`, W&B `kh3pf9tw`;
- evaluator jobs `8398907` and `8398908`;
- evaluator JSON hashes; and
- the complete family/cell checkpoint cadence.

Acceptance:

- both arms are recorded as rejected;
- neither is called mastered or selected;
- the sealed bank remains unopened; and
- no second paired seed is authorized.

### D1 — Audit the dump-mask semantic mismatch

Hypothesis:

> Buffer-only legal terminations are receiving a different completion and
> terminal reward from equivalent exact-zone terminations.

Implementation:

- add read-only counterfactual metrics to fixed evaluation;
- do not change transitions, actions, task termination, or training;
- evaluate flat update 1,000, flat update 4,000, and staged update 4,000;
- use development M0-M2 at horizon 450; and
- record exact-target and current accepted-buffer completion on every terminal
  state and top-quartile timeout.

Required output by family and primary cell:

- `task_done && exact_completion < 1`;
- exact-mask completion;
- accepted-mask completion;
- current versus counterfactual terminal reward;
- positive soil volume in buffer-only cells;
- mass residual; and
- timeout completion delta.

Decision:

- any `task_done && completion < 1` confirms the contract violation;
- call it a material contributor only if at least 10% of successes or at least
  10% of top-quartile timeouts change completion by `>= 0.05` or terminal
  reward by `>= 0.1 * terminal_reward`;
- otherwise fix it as correctness debt without claiming it caused the broad
  regression.

Budget: at most 259,200 evaluation transitions and no gradients.

### D2 — Separate memorization, policy mode, and held-out regression

Hypotheses:

1. online success is high because the policy fits repeated training identities;
2. deterministic argmax under-reports a diffuse but useful sampled policy; or
3. both train and development behavior regress because optimization or
   heterogeneous exposure moves the policy away from M0.

Implementation:

- build an exact manifest over unique training identities, excluding repeated
  slots;
- evaluate E8 zero-shot, flat updates 1,000 and 4,000, and staged updates 1,000
  and 4,000;
- run deterministic evaluation on train identities and development M0-M2;
- for flat updates 1,000 and 4,000 only, repeat sampled-action evaluation with
  eight declared seeds on M0;
- report family and primary-cell results; and
- include policy entropy, action-logit margin, and deterministic-versus-sampled
  action disagreement.

Interpretation:

- support memorization when train success is at least 60% and exceeds
  source-disjoint development by at least 20 percentage points in the same
  family;
- support shared optimization drift when train and development success both
  fall by at least 10 percentage points from update 1,000 to update 4,000;
- support diffuse-policy sensitivity when mean sampled M0 success exceeds
  deterministic M0 success by at least 10 percentage points and at least six
  of eight sampled seeds improve;
- none of these outcomes alone establishes architecture insufficiency.

Acceptance:

- train and development use identical horizon, reset, and action-selection
  semantics;
- repeated training slots are not double-counted; and
- the report states which hypotheses remain viable.

## 6. Phase C — establish one corrected future-policy contract

### C0 — Ratify the legal dump-mask definition

Recommended contract:

> The explicit map dump mask is the legal dump region. Any tolerance region
> must be materialized by the generator, visible in the map/gallery, included
> in capacity calculations, and stored in the manifest. There is no hidden
> one-cell legal buffer.

For the current representation, the direct definition is
`accepted_dump_mask = (target_map > 0) & ~obstacle_mask`, with generation
rejecting any target/obstacle overlap.

This recommendation preserves the meaning of arbitrary dump constraints and
prevents the environment from silently accepting soil outside the reviewed
zone. If the dilated region is retained instead, it must become the explicit
accepted mask everywhere, including visualization and generator capacity. Do
not retain an implicit mixed contract.

Deliverable:

- one named `accepted_dump_mask` definition;
- a short design decision in `TRAINING_DESIGN.md`; and
- no second termination-specific or reward-specific dump mask.

### C1 — Unify termination, completion, reward, and evaluation

Implement the smallest pure task-completion path needed by the current dense
experiment:

- compute dig, dump, unloaded, and applicable edge requirements once;
- reduce active requirements with a minimum so one completed component cannot
  hide an unfinished terminal prerequisite;
- use the same accepted dump mask for termination, completion, reward logging,
  and evaluation;
- require `task_done <=> absolute_completion == 1` within tolerance;
- create a new named corrected dense contract;
- do not claim numerical parity with the inconsistent legacy terminal
  calculation; and
- do not implement the full progressive reward framework in this task.

Primary files:

- `terra/state.py`;
- focused Terra completion tests; and
- the fixed evaluator's completion reporting.

Required deterministic cases:

- exact-zone soil;
- soil in the former hidden buffer;
- soil outside the legal region;
- obstacle-overlapping buffer cells;
- relocation-only and combined dig-and-dump tasks;
- incomplete and complete digging;
- loaded and unloaded agents;
- foundation edge incomplete/complete; and
- mass-conserving partial states.

Acceptance:

- every success has completion one;
- every completion-one state succeeds;
- no terminal reward path recomputes completion differently;
- eager, `jit`, and `vmap` cases agree; and
- legacy checkpoints remain evaluable under a clearly labeled legacy contract.

### C2 — Restore the full-reset horizon contract

For all future full-task training and evaluation:

- remove `randomize_initial_env_steps` from the initial reset path;
- assert `env_steps == 0` after reset;
- keep the 450-step horizon identical across train and fixed evaluation; and
- treat randomized remaining horizon as a future named treatment, not a
  startup optimization.

This is a correctness cleanup, not the proposed explanation for late
regression.

Acceptance:

- a focused reset test observes zero elapsed steps;
- the training receipt logs the effective horizon; and
- no hidden countdown randomization remains in a full-reset preset.

### C3 — Make dataset loading exact and fail loud

Change the current short-dataset warning into an error. Validate before JAX
compilation:

- exact expected map count;
- contiguous indices;
- target/action/occupancy/dumpability/distance files for every index;
- finite arrays and declared shapes;
- source-disjoint split IDs;
- manifest slot and unique-identity counts; and
- distance metric and normalization metadata.

Acceptance:

- a missing map or sidecar fails before environment construction;
- no zero or default distance map is substituted; and
- M1 slot multiplicity is explicit rather than silently weighted.

### C4 — Complete the fixed evaluator contract

Keep one direct fixed-bank evaluator and add:

- explicit deterministic or sampled mode in every receipt;
- map ID, source ID, family, stratum, primary cell, and slot weight;
- selected slot index carried through reset and terminal info so training
  outcomes can be joined to manifest provenance;
- exact verification of target, initial action, occupancy, dumpability,
  distance, and relevant metadata at reset;
- mass residual;
- invalid or no-op action counts;
- target/obstacle mutation;
- non-finite state;
- environment/evaluator termination disagreement; and
- an offline history aggregator for two-consecutive mastery and retention.

Do not turn this into a generic evaluation framework. The output remains one
versioned JSON record per checkpoint and stratum.

Acceptance:

- `mastery_gate.passed` cannot be true with an integrity failure;
- a single-checkpoint result is not labeled two-consecutive mastery; and
- all gate inputs can be recomputed from the saved JSON.

### C5 — Make reward and termination histories auditable

The completed runs expose a logging defect: scalar reward-component fields are
taken from the final state of one environment, while terminal completion fields
cover device 0 and pool successes with timeouts.

Add one globally reduced terminal-episode receipt containing:

- map ID, family, primary cell, stage, and termination reason;
- exact-target, accepted-mask, buffer-only, and illegal dump volumes;
- dig, dump, and combined completion;
- episode return and every reward component summed over the episode;
- terminal reward before and after normalization;
- steps, action counts, invalid/no-op actions, and productive workspace cycles;
- mass and immutable-map integrity fields; and
- enough raw counts to recompute every W&B rate offline.

Do not log an arbitrary environment element as an aggregate. Preserve
machine-readable per-episode or histogram receipts and use W&B only for reduced
totals and bounded rates.

Acceptance:

- global counts agree with a single-device reference fixture;
- success, timeout, and simultaneous success/timeout are separately labeled;
- reward-component sums reproduce total episodic return within tolerance;
- no field silently pools success and timeout; and
- all fixed-evaluator and training receipt fields use C1's completion source.

## 7. Phase O — test observation sufficiency before changing the model

### O0 — Construct paired-state alias tests

Try to construct current-model inputs that are identical while relevant state
differs:

1. positive pile height `1` versus `>1` outside the local workspace;
2. identical spatial state with different remaining horizon;
3. different last-dig or loaded-soil provenance with different next-step
   legality or reward; and
4. disabled reachability information on an obstacle-constrained map.

For an alias to count, show both:

- equality of every tensor returned by `obs_to_model_input`; and
- a difference in transition legality, reward, or a clearly defined optimal
  action.

Decision:

- if no consequential pair is found, retain the current observation;
- if raw positive height is consequential, change only its bounded
  preprocessing while preserving channel shape;
- add remaining budget or provenance only when its paired test succeeds;
- enable reachability only in a separate treatment; and
- do not add recurrence for state that can be represented directly.

Acceptance:

- deterministic regression tests capture every confirmed alias; and
- any observation-v2 change has a checkpoint compatibility or explicit
  checkpoint-growth decision.

## 8. Phase F — prove dynamic feasibility

### F0 — Overfit one easy foundation and one easy trench

Dependencies: C0-C5 and any required O0 observation correction.

Select and visually record:

- one M0 low-volume foundation with all-around dumping; and
- one M0 low-volume straight trench with broad both-side dumping.

For each identity:

- start from the same E8 parameters-only checkpoint;
- use a fresh optimizer and schedule;
- use the corrected dense contract;
- use full 450-step resets with `env_steps == 0`;
- use no partial resets, map curriculum, reward curriculum, or architecture
  change;
- evaluate deterministically on the same 32 declared reset seeds every 50
  updates; and
- use a budget ladder of 50, 100, 250, then at most 500 updates, stopping when
  the gate passes.

Pass gate:

- at least 29/32 successes in two consecutive evaluations;
- zero integrity failures; and
- at least one saved legal action trajectory.

If a probe fails, stop broader training and classify the failure:

- no legal action sequence within the horizon;
- action mask or transition blocks required work;
- reward gives the wrong local incentive;
- observation alias;
- PPO/value instability; or
- map itself violates the intended dynamic-feasibility contract.

Single-map success proves only that the current dynamics can learn that
identity.

### F1 — Train M0 family-bank specialists

Dependencies: both F0 identities pass and the larger B0 training bank is
available. Family generalization must not be judged from the current
eight-identity-per-cell M0 pool.

Train two parameters-only E8 adaptations:

- `M0-FOUNDATION-SPECIALIST`;
- `M0-TRENCH-SPECIALIST`.

Hold PPO, model, reward, horizon, reset, and evaluation fixed. Use only the
named family as the treatment. Evaluate deterministically on the
source-disjoint M0 family bank every 100 updates.

Budget ladder: 500, 1,000, then at most 2,000 updates while performance is
improving.

Family pass gate:

- at least 26/32 family successes;
- at least 6/8 in every primary cell;
- two consecutive evaluations; and
- zero integrity failures.

Interpretation:

- both pass: the families are learnable separately; multitask interference or
  mixture design becomes plausible;
- one fails: diagnose that family before a generalist;
- both fail: do not run the map curriculum.

These specialists are feasibility instruments, not final deployment policies.

## 9. Phase R — explain the historical M0 regression

### R0 — Fork the flat update-2,000 checkpoint

Run only if D1 does not show that the historical semantic mismatch dominates
the measured trajectories and both F0 probes pass.

Use the exact historical Terra and terra-baselines revisions and the full
update-2,000 checkpoint, including optimizer state. The historical control
falls from 22/64 at update 2,000 to 15/64 at update 2,500, making a 500-update
fork the shortest useful reproduction window.

Arms:

- control: continue the historical terminal mixture;
- treatment: continue M0 only.

Both arms restart environment/RNG state through the identical resume path and
use paired seeds. Do not include the corrected dump contract, observation
change, entropy change, or partial resets.

Evaluate every 100 updates:

- unique train identities and development M0;
- foundation, trench, and every primary cell;
- M1 at fork start and finish;
- policy KL and greedy-action disagreement from the parent;
- action-logit margin;
- entropy, actor loss, value loss, and explained variance.

Preregistered decision:

- support heterogeneous-exposure regression if the control reaches `<=16/64`
  while M0-only remains `>=20/64` in two consecutive evaluations;
- reject exposure as a sufficient explanation if both reach `<=16/64` or the
  final gap is `<4/64`;
- call it inconclusive if the control remains `>=20/64` and fails to reproduce
  the historical decline.

Budget: 500 updates per arm.

## 10. Phase G — establish a corrected M0 multitask parent

### G0 — Train foundations and trenches together on corrected M0

Dependency: both F1 specialists pass.

Start from E8 parameters only, not from either specialist. Train a 50/50 M0
foundation/trench mixture with the corrected contract and the expanded B0
training bank.

Evaluate every 100 updates. Initial budget 1,000 updates; extend once to 2,000
only if the preregistered learning curve is still improving.

Pass gate:

- at least 26/32 foundations;
- at least 26/32 trenches;
- at least 6/8 in every M0 primary cell;
- two consecutive evaluations;
- zero integrity failures.

The train-development gap remains a required diagnostic but is not an
additional post-hoc mastery threshold.

If specialists pass and G0 fails, the next treatment is sampling/gradient
interference, not a larger encoder by default.

This checkpoint is the first candidate corrected dense parent. No reward
curriculum begins before it passes.

## 11. Phase B — increase procedural diversity

### B0 — Rebuild quantitative cells, then expand source-disjoint banks

Dependency: both F0 probes pass.

Do not enlarge the failed M0-M2 bank unchanged. Its all-around and large-apron
foundation cells differ by roughly an order of magnitude in relative dump
area, its procedural foundation cells change geometry and dump layout
together, and its topology/site cells are not a monotonic ladder.

First build small paired feasibility panels that change one axis at a time:

- OSM versus procedural foundation geometry under identical all-around dumping;
- broad-apron dump distance centered near 2, 4, 6, and 8 tiles under fixed
  geometry, volume, capacity, and site;
- straight, two/three end-to-end segment, T, X, and disconnected trench
  topology under easy side-cast dumping; and
- site constraints only after the corresponding geometry/dump cell passes.

Use eight unique train and eight source-disjoint development identities per
candidate cell. Admit a cell to the large bank only after a bounded specialist
provides a dynamic witness. Remote haul at 12 or more tiles remains a separate
conditional feasibility track.

Then use offline procedural generation; do not add online generation or PLR.

Initial target:

- at least 64 unique training identities per primary cell;
- no repeated training slots;
- separate fixed promotion, development, and sealed banks with eight maps per
  primary cell each;
- disjoint generator seeds and source geometry IDs across every split; and
- frozen quantitative geometry, dump-distance, reachable-capacity, and site
  contracts for every admitted cell.

For each cell, save:

- geometry and junction/component metadata;
- dump side/components, capacity ratio, and shortest-path statistics;
- site constraints and work volume;
- generator revision and seed;
- source ID and split; and
- accepted dump-mask definition.

Validation:

- generate a random contact sheet per cell for visual review;
- reject templated duplicates using exact hashes plus a declared geometric
  similarity check;
- run static capacity/access validation; and
- admit a family only after F0/F1 provide dynamic witnesses.

Before fixing the bank size, measure loader memory and first-update compile
with the intended 64x64 arrays. Prefer 512 unique maps per stratum if it fits;
do not silently reduce diversity after launch.

## 12. Phase M — replace `3/3` with a global map curriculum

### M0 — Implement checkpoint-bounded global stages

Do not add a learned teacher or generic adaptive scheduler. Materialize one
directory per stage and start a new recorded run at each promotion boundary.

Proposed stages:

```text
C0: M0 only
C1: M0 rehearsal + M1
C2: M0 rehearsal + M1 rehearsal + M2
```

The C1/C2 rehearsal weights are treatments, not constants. Select their first
values from R0 and G0 retention evidence. Do not retain the current unproven
20-30% rule as fact.

Promotion:

- evaluate a separate fixed promotion bank every 250 updates;
- require family and primary-cell gates in two consecutive evaluations;
- require retention gates on all earlier strata; and
- promote only at a checkpoint/run boundary;
- carry the full model and optimizer state and preserve the schedule position;
  environment, RNG, and history restart identically for every compared arm;
  and
- create a new run identity and immutable mixture receipt.

Demotion/recovery:

- if an earlier stratum fails retention twice, stop the current stage;
- restore the last checkpoint that passed all earlier gates;
- relaunch the previous mixture as a new recorded treatment; and
- never mutate exposure silently inside the compiled PPO run.

Development evaluation remains every 500 updates and is not used to drive the
scheduler. The sealed bank is opened once after model selection.

Required logging:

- stage and mixture weights;
- unique map IDs and per-cell exposure;
- promotion-bank results;
- residence updates in the stage;
- promotion/recovery events; and
- fixed development retention.

Acceptance:

- no per-environment `3/3` promotion remains in the selected treatment;
- no promotion can occur from pooled success alone;
- gate, development, and sealed sources are disjoint; and
- a failed cell cannot be hidden by a pooled family score.

## 13. Conditional architecture work

### A0 — Keep `_se` until a representation-specific failure exists

No v5, 128-resolution, transformer-core, or recurrent sweep is authorized.

Authorize `resnet_spatial_8x8_se` versus
`resnet_spatial_8x8_se_xattn` only if:

- C0-C4 pass;
- F0 and F1 pass;
- G0 passes or fails specifically on location-conditioned constrained maps;
- train-identity learning is strong;
- observation alias tests are resolved; and
- trajectory/error analysis shows incorrect selection among spatially
  separated legal work or dump regions.

The ablation must use the same corrected observation, map bank, PPO settings,
parent policy decision, and budget. Primary metrics remain source-disjoint
success and retention, not old-map SWHiR.

Recurrence is considered only after a consequential alias cannot be represented
by a compact explicit state feature.

## 14. Separate progressive reward curriculum

### W0 — Correct the reward specification before implementation

Revise `PROGRESSIVE_REWARD_CURRICULUM.md` so:

- Stage 1 parity is defined against the newly frozen corrected dense contract;
- task completion and termination share C1's single source of truth;
- legacy inconsistent dense behavior remains replayable but is not the v2
  parent;
- map level cannot change reward stage;
- terminal duplication/backfill is disabled for the single-agent v2 path; and
- reward-independent success, workspace-cycle, and step metrics decide.

Apply the same corrected Stage-1 parent and gate language to
`PROGRESSIVE_REWARD_VALIDATION_PLAN.md`; the two specifications must not define
different qualification contracts.

Do not implement a compatibility framework beyond the one legacy replay path
needed to evaluate existing checkpoints.

### W1 — Qualify one foundation dense parent

Dependency: G0 passes and one fixed corrected foundation family passes its
family/cell gates in three scheduled evaluations.

Freeze:

- map family and bank;
- full resets;
- model and PPO;
- corrected reward contract;
- checkpoint hash; and
- evaluation protocol.

Only this qualified parent authorizes the reward experiment.

### W2 — Run the matched reward experiment

Arms:

```text
A: corrected dense continuation
B: corrected dense -> terminal objective
C: corrected dense -> terminal margin -> terminal objective
```

Keep map and reset distributions identical. Select lexicographically by:

1. fixed-bank success;
2. productive workspace cycles on episodes solved by both;
3. steps on episodes solved by both; and
4. completion margin on failures.

Validate the selected sequence separately on trenches and only then on the
multitask mixture. Do not combine it with a new map stage.

## 15. Separate partial-reset curriculum

### P0 — Test partial resets after the map sampler is selected

Compare:

- control: 100% untouched full resets;
- treatment: 75% full resets and 25% mass-conserving partial resets.

Start the partial share with equal 50% and 75% `in_zone` states. Add 25% states
only as a bridge if needed; defer `mixed`, `near_zone`, and 90% states.

Hold map, reward, model, PPO, and budget fixed. Primary evaluation remains
untouched full tasks; report partial-reset success separately by completion
fraction and pile mode.

Support the treatment only if it improves late-state competence without
reducing full-task family/cell success or retention.

## 16. Later map expansion

Admit one axis at a time after M0-M2 passes:

1. disconnected structural foundations and multi-junction trenches;
2. stronger single obstacles;
3. combined road/wall/object sites;
4. medium then far dump distance with matched capacity;
5. tight natural capacity only after distance is solved; and
6. the final realistic deployment mixture.

For remote dumping, first create a paired near/far identity that changes only
traversable dump distance. A successful near map is a prerequisite. One far
identity must pass the F0 gate before generating a far family.

Unsolved or statically valid but dynamically unproven families remain in a
named challenge bank; they are not mixed into training and called curriculum
difficulty.

## 17. Immediate execution queue

Do these next, in order:

1. D1 semantic replay audit.
2. D2 train/development and deterministic/sampled audit.
3. Ratify C0's legal dump-mask decision.
4. Implement and test C1-C5.
5. Run O0 paired-state alias tests and only the observation fixes they prove.
6. Run the two F0 fixed-identity probes.
7. Decide from D1/F0 whether the historical R0 fork is authorized.
8. Build the small B0 feasibility panels, expand only passing cells, then
   decide whether F1 is authorized.

Stop after each decision gate. Do not pre-build later curriculum, reward, or
architecture machinery while an earlier result can invalidate it.

## 18. Definition of done for this backlog

The training redesign is ready for a generalist confirmation only when:

- termination, completion, reward, and evaluation share one legal task
  contract;
- reward and termination histories are globally reduced, stratified by
  outcome, and recomputable from machine-readable receipts;
- full-reset and evaluator integrity contracts pass;
- one foundation and one trench identity are dynamically learnable;
- both M0 families generalize separately;
- a corrected M0 multitask parent passes family/cell and retention gates;
- the map curriculum uses source-disjoint global promotion gates;
- procedural training diversity is sufficient and contains no silent repeated
  weighting;
- reward and partial-reset treatments remain separate;
- any architecture change is supported by a representation-specific failure;
  and
- the sealed bank remains untouched until model selection.
