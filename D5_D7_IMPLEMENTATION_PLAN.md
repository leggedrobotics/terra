# Terra Map-Diversity and Reward-Semantics Implementation Plan

- Status: active
- Date: 2026-07-30
- Terra branch: `experiment/simple-mapbank-reward-v3`
- Terra base: `952522631104f1e051b8ea46963d305074437d28`
- terra-baselines branch: `experiment/simple-mapbank-reward-v3`
- terra-baselines base: `edd9827180a57d983f0d7633b8b0a9874bc86f7d`
- Governing research-code workflow:
  [`$simple-research-code`](/home/lorenzo/git/codex_skills/skills/simple-research-code/SKILL.md)
- Training authority: [`TRAINING_TASKS.md`](TRAINING_TASKS.md)
- Design context: [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md)
- Canonical current curriculum taxonomy:
  [`CURRICULUM_TAXONOMY.md`](CURRICULUM_TAXONOMY.md)
- Executable P5 baseline plan:
  [`P5_ACCEPTED_BANK_EXPERIMENTS.md`](/home/lorenzo/moleworks/.worktrees/terra_baselines_simple_mapbank_reward_20260730/docs/research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
- Local visual-review adapter:
  [`export_diverse64_gallery_review.py`](/home/lorenzo/moleworks/.worktrees/terra_digging_benchmark_diverse64_review_20260730/scripts/export_diverse64_gallery_review.py)
  (site commit `240f38f`)

## 1. Objective

Deliver the shortest trustworthy path from the reviewed v6 map distribution to
a larger, source-disjoint training bank and then to controlled PPO experiments.
In parallel, replace the incomplete reward-v3 proposal with one clean,
agent-neutral reward contract before interpreting multi-agent transport runs.

The final sequence is:

1. generate many valid maps without treating all-pairs centred IoU as an
   admission wall;
2. inspect a deliberately diverse visual subset;
3. freeze train, promotion, development, and sealed splits by source group;
4. run the map-curriculum experiment with one unchanged reward contract;
5. validate any reward change separately on a fixed map bank; and
6. promote a learning recipe to a long run as soon as fixed-bank task progress
   is repeatable and integrity-clean.

## 2. `$simple-research-code` constraints

The linked skill is normative for this implementation:

- one supported generator command, not another versioned generator chain;
- one manifest schema for training and review;
- loud failures for invalid masks, duplicate scenario identities, split
  leakage, insufficient capacity, and missing files;
- no all-pairs similarity admission rule;
- no compatibility layer for the rejected reward-v3 API;
- no new scheduler framework, planner, witness solver, or general validation
  platform;
- only focused deterministic tests that protect the experiment's conclusions;
- diagnostics are written once to compact artifacts and are not recomputed by
  several wrappers; and
- stop each validation stage as soon as it answers its declared decision.

## 3. Accepted decisions

### 3.1 Map diversity

1. The current `centred_iou < 0.60` foundation and `< 0.70` trench rule is not
   a hard training-bank admission gate.
2. Centred IoU remains a diagnostic and may be used to select a small visual
   showcase.
3. Similar excavation masks are allowed when the complete scenarios differ in
   policy-relevant ways such as reset pose, dump geometry, capacity, distance,
   obstacles, access, or local boundary variation.
4. Reusing one base excavation across controlled counterfactual conditions is
   intentional. Every variant of one base layout stays in the same split.
5. The first scale target is 64 training layouts per active condition. The
   accepted long-run bank target is 256 training layouts per active condition.
6. The website and image folders show 16 deliberately diverse examples per
   condition; the showcase is not the training bank.

### 3.2 Hard map gates

Only these properties block bank materialization:

- generator output has the expected shapes and dtypes;
- target, dump, obstacle, occupancy, and reset-state fields are internally
  consistent;
- accepted dump capacity is sufficient for the map's excavation volume under
  the frozen contained-dump physics;
- the initial state is admissible;
- a full scenario identity is not duplicated within one condition and split;
- no source group appears in more than one split; and
- all controlled variants of a source group are assigned to the same split.

Basic static workspace/reachability quantities remain reported. They are not
promoted into a constructive 450-step solver in this task.

### 3.3 Reward semantics

The uncommitted reward-v3 patch in
`terra_v5m_screen_20260730` is evidence only and is not a merge source.

The intended rule is agent-neutral:

- fresh target excavation receives the existing extraction bonus once, and
  only when required dig depth actually decreases;
- merely picking up previously dumped soil receives no extraction bonus;
- every agent uses the same signed relocation progress:
  `carry_credit + potential_before - potential_after`;
- carry credit is stored once per carrying agent, moves rather than copies
  during a handoff, and is cleared after a dump;
- negative relocation progress is retained rather than clipped away;
- required rehandling by a skid steer or truck is not multiplied by an
  arbitrary machine- or material-specific `0.2`; and
- a closed dig/dump/reload cycle with no net task progress must have
  non-positive net shaped return.

The common potential is the sum, over positive off-zone soil, of soil mass
times normalized distance to the accepted dump mask. Fresh target extraction
adds the corresponding source credit to the carrier. Rehandling adds only the
change in common potential. A handoff pays no relocation reward by itself.
Partial transfers are out of scope: the first implementation transfers the
whole carried load and credit atomically.

Map-curriculum and reward comparisons remain separate. Every initial map
experiment uses the committed agent-neutral contract
(`relocation_progress_mult=1.5`). Any later reward ablation is a separate,
current-schema treatment on the selected fixed maps and sampler; historical
reward-v1/v2 checkpoints are not resumed under the new environment schema.

## 4. Bank design

### 4.1 Unit of identity

- `source_group_id`: the base excavation/layout identity shared by deliberate
  counterfactual variants.
- `scenario_id`: hash of the five map arrays consumed by reset: target, action
  state, dumpability, occupancy/obstacles, and relocation distance.
- `episode_id`: hash of `scenario_id`, fixed reset seed, and frozen environment
  protocol hash. Evaluation pins this identity explicitly; training may sample
  fresh reset seeds.
- `condition_id`: the human-readable factor combination.
- `split`: one of `train`, `promotion`, `development`, or `sealed`.

Split assignment is deterministic from the atomic pair/source grouping and the
frozen per-split target counts. Its complete assignment and hash are written
before policy evaluation and never changed to rescue a metric.

For P5, the exact JAX/JAXlib lock is fixed and the allocation preflight must
hash the reset agent state for every evaluation seed and compare CPU/GPU
receipts before policy evaluation. If those hashes differ, seed-bound
`episode_id` is not portable and evaluation stops until explicit initial-state
injection is implemented. A public benchmark release remains gated on folding
the verified initial-state hash into episode identity. Trench/foundation
metadata is nonsemantic only under the frozen protocol's disabled absolute
trench and foundation-border shaping; enabling either requires an identity
schema bump.

### 4.2 Pilot sizes

For every active condition:

| Split | Base layouts | Purpose |
|---|---:|---|
| train | 64 | first training/generalization pilot |
| promotion | 16 | checkpoint promotion only |
| development | 16 | diagnosis and design comparison |
| sealed | 32 | final selected-policy evaluation |

This is 128 source groups per condition. Counterfactual variants may share the
same source groups, so this does not necessarily require 128 independent
geometry builds for every condition.

After the pilot:

- expand only accepted train conditions from 64 to 256 source groups;
- keep promotion/development/sealed identities frozen;
- add only new source-disjoint groups to the train split under an expansion
  receipt; and
- never move a seen source group into evaluation.

### 4.3 Diversity report

For each condition and split, write:

- number of files, scenarios, and source groups;
- exact duplicate count;
- nearest-neighbour centred-IoU p10/p50/p90/max;
- dig area, perimeter, compactness, components, aspect, and width summaries;
- dump area/capacity, distance, component count, and side/layout summaries;
- obstacle count and blocked-area summaries; and
- rejection counts by hard-gate reason.

No diversity statistic decides policy success. It only exposes collapse,
imbalance, or unexpected generator behaviour.

## 5. Implementation tasks

### P0 — isolate and freeze authority

- [x] Create clean Terra worktree from `95252263`.
- [x] Create clean terra-baselines worktree from `edd9827`.
- [x] Preserve the dirty v5m reward worktree without copying its diff.
- [x] Link this plan from `TRAINING_TASKS.md` and `TRAINING_DESIGN.md`.
- [x] Record the exact generator source selected below.

Exit gate: both implementation worktrees are clean before the first edit, and
all later changes are explainable by this plan.

### P1 — one generator path

- [x] Select the smallest current generator implementation that reproduces one
  v6 condition byte-for-byte for its first seed.
- [x] Put that path under version control.
- [x] Collapse the version-chain entry point to one supported command.
- [x] Make the bank map count an explicit required argument. Split counts are
  a P2 materialization argument, not a geometry-generation concern.
- [x] Remove the bank-capacity-probe path and hard all-pairs IoU rejection from
  normal generation.
- [x] Retain exact full-scenario duplicate rejection.
- [x] Emit one manifest and one rejection/diversity report.

Focused tests:

1. [x] manual smoke receipt reproduces the pinned reference scenario;
2. [x] 64 scenarios can be generated for a representative slab and trench
   condition without IoU exhaustion;
3. [x] an exact duplicate scenario fails loudly; and
4. [x] variants of one source group cannot cross splits (P2).

Exit gate: the representative two-condition command generated 64 valid slab
and 64 valid trench scenarios. Two independent trench runs had identical
manifests. The first slab scenario is byte-identical to reviewed v6 across all
five arrays. Receipt:
[`tools/map_generation/SMOKE_RECEIPT_20260730.md`](tools/map_generation/SMOKE_RECEIPT_20260730.md).

### P2 — split-ready pilot

- [x] Give every row a raw-source or realized-dig `source_group_id` and a
  separate declared `pair_slot_id`.
- [x] Drop a pair slot when condition-specific rerolls produce different dig
  identities inside it.
- [x] Implement exact deterministic `64/16/16/32`
  train/promotion/development/sealed materialization.
- [x] Fail instead of leaking one realized source across splits.
- [x] Generate an oversized candidate bank (initially 160 maps per condition)
  so at least 128 exact pair slots remain after reroll drops.
- [x] Materialize `64/16/16/32` retained pair slots for representative
  foundation and trench anchor conditions.
- [x] Validate hard gates on the final-code 32-condition acceptance smoke and
  representative split-ready bank.
- [x] Write placed, translation-normalized, and dihedral-normalized dig counts
  as diagnostics, not admission gates.
- [x] Fail if any condition has fewer than its requested retained pair slots.
- [ ] Select 16 review examples per condition using descriptor coverage and
  nearest-neighbour diversity, without changing the training bank.
- [x] Export an image-folder gallery grouped by explicit sibling branches:
  anchor/easy, capacity, distance, dump layout, geometry/topology, site, and
  composed.
- [x] Export the full candidate gallery into the isolated local review website
  on port `4174`.
- [x] Separate map comments from one explicit manifest-bound
  Accept/Reject/Quarantine disposition per condition; never infer condition
  admission from map votes.
- [x] Add a fail-loud compiler that requires all condition dispositions and
  accepted foundation/trench anchors before emitting the generator `--only`
  list (`tools/map_generation/compile_condition_review.py`).
- [ ] Record Lorenzo's comments/accept/reject decisions without changing
  scenario identity.

Implementation receipt:
[`tools/map_generation/SPLIT_PILOT_RECEIPT_20260730.md`](tools/map_generation/SPLIT_PILOT_RECEIPT_20260730.md).
The first real split-ready probe produced 160 candidates for one foundation
anchor and one trench anchor, then materialized exact `64/16/16/32` splits with
256/256 unique scenarios and zero source leakage. This validates the mechanism;
all conditions still require visual acceptance before the curriculum is frozen.

Visual review authority:

- durable generation/site receipt:
  [`FULL_REVIEW_RECEIPT_20260730.md`](tools/map_generation/FULL_REVIEW_RECEIPT_20260730.md);
- image folders and editable index:
  [`terra_diverse64_full_review_20260730`](/home/lorenzo/moleworks/.artifacts/terra_diverse64_full_review_20260730);
- local site: [http://127.0.0.1:4174/](http://127.0.0.1:4174/);
- site branch/commit: `diverse64-gallery-review` at `885f56b` (diverse-64
  default; explicit condition dispositions and atomic combined import/export).

The visual source bank is explicitly review-only because its long process
started before the final source/pair identity repair. Accepted conditions must
be regenerated with the committed generator before split freezing.

Exit gate: every selected condition has complete split counts, zero leakage,
zero exact scenario duplicates, a visual subset, and an accepted or explicitly
deferred review disposition.

### P3 — reward evidence harness

- [x] Revert conceptually to committed reward-v2 as the control; do not import
  the dirty reward-v3 diff.
- [ ] Add deterministic traces for:
  1. [x] excavator fresh dig and correct dump;
  2. [x] excavator dump/re-dig/dump closed cycle;
  3. [x] excavator-to-truck productive transfer and dump;
  4. [x] skid-steer pickup of an excavator pile and correct dump;
  5. [x] transport pickup/drop/re-pickup closed cycle; and
  6. [x] a handoff and dump that exposes the current double payment and copied
     carry caches.
- [x] Report action reward, step-cost-adjusted reward, dig/dump progress,
  potential, task completion, load, world mutation, and conserved mass.
- [x] After P4 separates the terms, report extraction and relocation reward
  components independently.

Evidence:
[`terra/tests/test_relocation_reward_contract.py`](terra/tests/test_relocation_reward_contract.py).
The committed control pays a terrain-unchanged excavator-to-truck handoff and
then pays the truck's dump again. A no-progress excavator rehandle cycle is raw
break-even (`+1/-1`) and becomes negative only through existence costs.

Exit gate: the harness drives the real action paths and reproduces the current
control numbers without hand-setting reward flags.

### P4 — corrected agent-neutral reward

- [x] Pay the extraction bonus from fresh target progress, not merely a
  `0 -> loaded` transition.
- [x] Remove the transport-versus-excavator multiplier branch.
- [x] Replace the three machine/material relocation multipliers with one
  `relocation_progress_mult` for every agent.
- [x] Remove obsolete machine-specific multiplier arguments and update known
  executable presets directly. Historical audit/receipt fields remain pinned
  to their old revisions and are not launch inputs.
- [x] Replace the global material flag and two carry-potential caches with one
  per-agent `carry_relocation_credit`.
- [x] Pay signed relocation progress on dump; do not clip negative progress.
- [x] Transfer the whole load and carry credit atomically and never pay the
  handoff itself as a dump.
- [x] Ensure skid-steer auto-load and truck transfer use the same mass and
  reward accounting as direct excavation.
- [x] Treat positive soil only as auto-loadable material; a negative target
  hole is not a pile.

Focused gates:

- fresh target work retains positive incentive;
- re-pickup alone receives no extraction bonus;
- productive transport receives positive relocation credit;
- every no-progress closed cycle is non-positive;
- mass is conserved; and
- the full focused Terra and preset suites pass.

Implementation: Terra commit `64deed22` (`Implement agent-neutral relocation
reward`) in
[`terra/state.py`](terra/state.py),
[`terra/agent.py`](terra/agent.py), and
[`terra/config.py`](terra/config.py). The serialized agent state is intentionally
`terra_agent_state_v2`; the benchmark release is `terramap-bench-v1.0.1`.
Forty-seven focused reward, state-codec, protocol, and migration checks pass.

Exit gate: the six P3 traces satisfy the rule without an agent-type reward
branch. The Terra implementation and baseline configuration gates are complete.
Current executable baseline presets are new agent-neutral reward treatments;
they do not reproduce the old reward-v1/v2 split. The complete CPU
terra-baselines suite passes against the paired Terra worktree (173 tests).

The complete Terra suite has 243 passing tests and two expected provenance
failures. Those guards correctly reject the 2026-07-27 CPU/GPU direct-service
receipts because their pinned execution-code hashes predate the AgentState,
EnvConfig, State, benchmark-schema, and maps-buffer changes. Do not rewrite the
expected hashes. Re-run that historical parity measurement only if a future
decision needs it; it is not an admission gate for visual review or PPO.

### P5 — controlled experiments

Map and reward treatments never change together.

Euler authorization recorded 2026-07-30: after Lorenzo's review decisions are
exported, accepted conditions are regenerated and split-frozen, and local plus
allocated-GPU first-update gates pass, sync the paired commits to the isolated
Euler workspace and submit the bounded map experiments without another
confirmation. This authorization covers the declared 2,000-update screens and
one 20,000-update `gpuhe.120h` promotion that satisfies the fixed-bank gate. It
does not authorize unplanned ablations or training on rejected/review-only maps.

#### P5a Map sampler/curriculum

Implementation:
[terra-baselines P5 accepted-bank experiments](/home/lorenzo/moleworks/.worktrees/terra_baselines_simple_mapbank_reward_20260730/docs/research/P5_ACCEPTED_BANK_EXPERIMENTS.md)
(sampler/evaluator base `64fc4a9`; reviewed immutable Euler screen path
`18322cb`, with exact 64-layout declaration and staged-payload gates at
`e847d51`, `98d0055`, and `af5ca6a`, plus review-admission binding at
`d35d9cf` and pinned review-release validation at `cee4de9`).

- fixed reward: the committed agent-neutral contract
  (`relocation_progress_mult=1.5`) for every arm;
- maps: the same accepted 64-layout-per-condition training bank;
- evaluation: identical frozen promotion/development panels;
- reset, horizon, observation, action, dynamics, architecture, and PPO:
  identical.

Minimal one-seed screen:

| Arm | Training distribution | Question |
|---|---|---|
| `F-ANCHOR` | accepted foundation anchors, uniform | are the new foundation maps learnable from scratch? |
| `T-ANCHOR` | accepted trench anchors, uniform | are the new trench maps learnable from scratch? |
| `G-UNIFORM` | all accepted conditions, uniform | can one generalist learn the target distribution directly? |
| `G-ADAPTIVE` | same all-condition bank, adaptive progressive sampler | does progressive exposure improve the generalist? |

All four use scratch parameters with the same E8 architecture recipe
(`resnet_spatial_8x8_se`, medium model, MLP, bf16), not an E8 checkpoint.
`G-ADAPTIVE` is not a strict stage unlock: it retains a 20% aggregate uniform
floor from update zero and spends 80% on the highest-competence unmastered
conditions. Call the comparison adaptive-progressive versus uniform and log
exposure by branch depth.

Before any submission:

0. [x] compile an explicit condition-level review receipt bound to the current
   review release and manifest; incomplete, duplicate, stale, unknown, or
   map-vote-inferred condition sets fail; the exact review-data SHA and
   canonical 32-condition registry are pinned, the loader bank copies and
   hashes the receipt, and the Euler launcher requires its accepted IDs to
   equal the train condition IDs (`d35d9cf`, pinned to the exact review
   release at `cee4de9`);
1. [x] materialize loader-ready contiguous arrays plus `dataset.json`,
   `manifest.jsonl`, and `source_registry.jsonl`
   (`7f9fd4ee`, strengthened by `c6d894cf` so published arrays are re-hashed
   and exact scenario/pair/source counts are consumer-verified;
   [`materialize_loader_bank.py`](tools/map_generation/materialize_loader_bank.py));
2. [x] pin `reset_seed` and `episode_id` on every
   promotion/development/sealed row through the live `MapsBuffer` selection
   path;
3. [x] port and migrate the adaptive sampler from the isolated experimental
   worktree onto the sole agent-neutral reward API (`64fc4a9`);
4. [x] report exact and condition-macro graded completion, micro p10, worst
   condition, family, and cell metrics;
5. [x] use continuous promotion evidence rather than hard-coded counts from an
   old panel size;
6. [x] implement a fail-closed CPU/GPU initial-agent-state comparison for every
   fixed promotion/development seed under the exact runtime lock before a
   screen trains (`18322cb`); its allocated execution on the final frozen bank
   remains pending; and
7. [x] add a content-addressed immutable Euler smoke/screen launch path and
   receipt (`18322cb`, strengthened at `e847d51`, `98d0055`, and `af5ca6a` to
   require and record exactly 64 training layouts per condition, require local
   `slot_count = 64`, and validate the staged manifests plus all five reset-array
   file sets before upload, at `d35d9cf` to require the archived review
   receipt and exact equality between its accepted IDs and the training-bank
   conditions, and at `cee4de9` to reject stale release, manifest, or
   review-data identity). Existing
   v5m/v6m launch scripts are historical inputs, not submission authority. The
   20,000-update P6 path deliberately fails closed until the separate
   256-training-layouts-per-condition expansion contract exists.

Execution:

1. [complete] migrate terra-baselines to the sole
   `relocation_progress_mult`, remove old carry-field reads and reward-v2 guard
   arguments, and pass its preset/config tests (`3ce0e84`, 21 focused tests);
2. [complete] CPU/config/manifest/launch validation (50 focused
   accepted-bank/Euler tests at `18322cb`; 26 focused staged-bank tests at
   `af5ca6a`; current 35 focused and 232 full review-bound launch tests at
   `cee4de9`;
   ShellCheck, syntax, and deterministic `SUBMIT=0` packaging);
3. [pending final bank] CUDA conv/backward and NCCL preflight in the
   allocation;
4. [pending final bank] W&B-disabled first-update smoke for all four arms;
5. [pending smoke receipts] one unblinded 2,000-update seed per arm
   (`262,144,000` global transitions each at `4 x 1024 x 32`);
6. [P6, fail-closed] promote a learning generalist arm to one fresh continuous
   20,000-update `gpuhe.120h` run on 256 training layouts per accepted
   condition when
   two fixed evaluations show either one additional exact success or at least
   `+0.01` macro condition-balanced terminal completion without guard
   regression; and
7. use paired seeds for the final scheduler claim, not to decide whether a
   clearly learning recipe deserves enough compute.

Current execution blockers, checked 2026-07-31:

- Lorenzo's current diverse-64 map comments and all 32 explicit condition
  dispositions have not yet been exported from the local review site;
- the accepted 64/16/16/32 source-disjoint bank therefore has not been
  regenerated and frozen from those decisions; and
- allocated CUDA/NCCL, reset-parity, and completed-update-1 receipts do not yet
  exist.

The purge-damaged scratch environment has been replaced by one dependency-only
project environment:
`/cluster/project/rsl/lterenzi/terra_curriculum_20260730_c14bd7d_3ce0e84_py312_jax0426`.
Its exact 90-package lock, CPU imports, CUDA package presence, absence of
editable source bindings, quota footprint, and 19-file provenance ledger pass;
ledger SHA-256 is
`853871aef55efe34a64474660109673c0d48b9a34cba333e368600a11b126d5c`.

Source provenance and launch isolation are implemented at `18322cb`,
`e847d51`, `98d0055`, `af5ca6a`, `d35d9cf`, and `cee4de9`: the launcher
requires clean committed source, copies
the bank into a private stage, validates local `slot_count = 64`, exactly 64
ordered manifest slots, and all five reset-array file sets for every training
condition, validates the copied review receipt and exact accepted-condition
set, then hashes that same staged payload. It creates one content-addressed
archive, unpacks it into
allocation-local storage, and keeps only logs/checkpoints/W&B on scratch. It rejects
`REVIEW_ONLY.md` and `NON_ADMISSION.md`, checks the dependency-ledger hash,
home quota, four RTX 4090 devices, JAX conv/backward/NCCL health, exact source
imports, and a finite completed update 1 with clean transition integrity.
Screens additionally require CPU/GPU equality of every fixed
promotion/development initial-agent-state hash before training.

#### P5b Reward comparison

Deferred until P5a selects a viable map recipe. The old reward-v1/v2 EnvConfig
and checkpoint schema are not executable in the current branch, so they are not
a valid in-place control. Any later reward ablation must use a named,
current-schema alternative on the frozen P5a maps and sampler; it must not be
folded into the initial curriculum screen.

### P6 — 256-layout long-run bank

- [ ] Expand selected training conditions to 256 source groups.
- [ ] Re-run only hard validation and diversity reporting.
- [ ] Keep the promotion/development/sealed banks unchanged.
- [ ] Materialize exact sampler slots and record effective condition weights.
- [ ] Launch the selected scratch specialist/generalist recipes independently.
- [ ] Continue long runs while fixed-bank task metrics improve.

## 6. Metrics and promotion

Primary:

- exact success within 450 steps;
- macro condition-balanced terminal completion;
- per-family and per-condition completion;
- worst-condition completion; and
- retention on previously passed conditions.

Guards:

- micro completion p10 does not regress by more than 0.05;
- worst-condition completion does not regress by more than 0.05;
- no source leakage or manifest mismatch;
- finite parameters, optimizer state, losses, rollout tensors, and evaluation;
- no change in reward/horizon/action/observation/dynamics inside a map
  comparison; and
- no change in maps/sampler/reset/architecture inside a reward comparison.

Online return and pooled online success are diagnostics only.

## 7. Stop conditions

Stop and revise the generator if:

- it cannot fill a requested condition after removing similarity rejection;
- exact duplicates dominate;
- descriptor distributions collapse to a small template set;
- capacity, spawn, or static workspace validity repeatedly rejects one
  condition; or
- visual review identifies a systematic unrealistic pattern.

Stop a training arm if:

- fixed-bank macro completion is flat through the declared bounded screen;
- task progress improves only on trained identities;
- guard metrics regress persistently;
- reward rises while task completion does not; or
- integrity/provenance is invalid.

Do not add a planner, learned curriculum teacher, partial resets, new encoder,
or reward schedule to rescue the same run. Each is a separate named treatment.

## 8. Execution log

| Date | Item | Evidence | Status |
|---|---|---|---|
| 2026-07-30 | Clean Terra worktree | branch/base above | complete |
| 2026-07-30 | Clean terra-baselines worktree | branch/base above | complete |
| 2026-07-30 | D5/D7 plan linked to `$simple-research-code` | this document | complete |
| 2026-07-30 | Generator implementation selection | reviewed v6 dependency closure, one public CLI | complete |
| 2026-07-30 | Representative 64-map generation | [`SMOKE_RECEIPT_20260730.md`](tools/map_generation/SMOKE_RECEIPT_20260730.md) | complete |
| 2026-07-30 | Generator and split unit contract | 16 focused tests pass | complete |
| 2026-07-30 | Source/pair identity repair | raw OSM hash, realized dig hash, pair-slot reroll audit | complete |
| 2026-07-30 | Exact split materializer | pair-slot grouping plus realized-source leakage failure | complete |
| 2026-07-30 | Reward semantic path audit | signed common potential plus one per-agent carry credit | complete |
| 2026-07-30 | Reward-v2 real-path control harness | 3 new + 7 existing focused tests pass | complete |
| 2026-07-30 | Branch-organized review gallery exporter | 3 focused tests; editable decision/comment CSV | complete |
| 2026-07-30 | Final-code 32-condition acceptance smoke | [`SPLIT_PILOT_RECEIPT_20260730.md`](tools/map_generation/SPLIT_PILOT_RECEIPT_20260730.md) | complete |
| 2026-07-30 | Real `160 -> 64/16/16/32` split probe | 256 unique scenarios; zero source leakage | complete |
| 2026-07-30 | Agent-neutral relocation reward | Terra `64deed22`; 47 focused checks | complete |
| 2026-07-30 | Benchmark direct-service parity | 5 tests + 4 subtests, normal CPU JIT | complete |
| 2026-07-30 | Terra-baselines reward-field migration | baseline `3ce0e84`; 21 focused checks | complete |
| 2026-07-30 | Terra-baselines CPU regression | 173 tests against paired Terra `64deed22` | complete |
| 2026-07-30 | Full Terra CPU regression | 243 pass; 2 stale-receipt provenance guards fail as intended | complete with explicit stale receipts |
| 2026-07-30 | Clean local review-site adapter | site `240f38f`; 13 Python + build + 6 Playwright pass | complete |
| 2026-07-30 | 32-condition × 64 candidate review generation | 2048 maps; zero unsatisfied constraints; review-only | complete |
| 2026-07-30 | Full seven-branch review site | site `4a1c1a2`; 512 hash-bound graphics at `127.0.0.1:4174` | running for review |
| 2026-07-30 | Euler curriculum-validation authorization | post-review/freeze smokes, 2k screens, gated 20k promotion | authorized, not yet launchable |
| 2026-07-30 | Euler read-only readiness audit | no Terra jobs; invalid copied-worktree Git metadata; damaged venv; scratch soft inode quota exceeded | repair required |
| 2026-07-30 | Dependency-only Euler runtime repair | project venv; exact 90-package lock; ledger `853871ae...`; no GPU job | complete |
| 2026-07-30 | Minimal experiment matrix | F-ANCHOR, T-ANCHOR, G-UNIFORM, G-ADAPTIVE | implementation complete; review/freeze gates pending |
| 2026-07-30 | Strict accepted-bank identity/count verification | Terra `c6d894cf`; 81 relevant tests + 4 subtests; real split pilot | complete |
| 2026-07-31 | Accepted-bank sampler/evaluator and immutable Euler screens | baseline `af5ca6a`; 223 full tests; staged 64-layout manifest/array gate with required local `slot_count = 64`; deterministic local packaging; no remote mutation | implementation complete; human review/final bank and allocated receipts pending |
| 2026-07-31 | Explicit condition admission contract | site `885f56b`; Terra compiler + loader binding; baseline `d35d9cf` + `cee4de9`; 7 browser, 40 Terra curriculum, 35 focused baseline, and 232 full baseline checks | complete; Lorenzo dispositions pending |
| 2026-07-30 | Oversized split-ready candidate and review export | P2 | pending |
