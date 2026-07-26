# Terra Training Tasks

- Status: active recovery execution; C0-C5 and C1b complete; D1 diagnosis
  complete with a failed historical mass-integrity gate and material completion
  mismatch; D2 diagnosis complete with memorization and action-mode evidence;
  F0 foundation feasibility passed with a terminal retention failure; F0 trench
  failed cleanly; bounded diagnosis selected F0R; F0R passed; B0a paired-panel
  generation is active
- Date: 2026-07-26 execution update
- Governing design: [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md)
- Failure evidence: [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)
- Historical reference: E8 `resnet_spatial_8x8_se`
- Recovery scratch topology: base `resnet_spatial_8x8`, approximately 994,825
  parameters
- Production training authorized by this document: yes, only for declared
  tasks whose gates pass; independent declared arms may run concurrently

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
4. Can each regenerated quantitative easy family generalize when trained
   alone?
5. Does heterogeneous exposure cause the observed M0 regression?
6. Only then: what global map curriculum and observation/model treatment
   should be tested?

### 1.1 Ratified decisions from the design review

These decisions supersede stale choices later in the historical v4 design:

| Topic | Ratified recovery decision |
|---|---|
| Legal dump region | The exact visible target dump mask is authoritative everywhere. There is no hidden one-cell buffer. |
| Starter soil physics | A correctly aimed dump is contained inside the exact target region and conserves mass. No boundary clipping or deletion is permitted. |
| Wrong dumps | An entirely off-zone dump remains physically possible, remains off-zone, earns no legal completion, and must be recovered. |
| Later spill difficulty | Physical boundary spill is deferred to a separately named dynamics treatment after contained-map competence. |
| Action validity | Remove every relocation-potential veto on a physically valid dump. Potential belongs in reward, not action legality. |
| Initialization | Train new small policies from scratch on the corrected distribution. E8 is evaluation context only, never the initializer or teacher for recovery runs. |
| Family separation | Foundation and trench feasibility/specialist policies are two independent runs. |
| Scratch budget | Treat 500/1,000/2,000/5,000 updates as review milestones, not hard ceilings. Continue the same checkpoint lineage whenever the fixed task metrics show even slight preregistered improvement; stop a feasibility screen once its gate passes twice. |
| Qualified long runs | Once both F1 specialists establish the recipe, run selected F1/G0/S0/K0 treatments as at least 20,000-update continuations on `gpuhe.120h` with a five-day wall-time request. Continue in 20,000-update chunks while the fixed bank still improves. |
| Curriculum separation | Map, dense-reward, dense-to-terminal reward, and partial-reset treatments never advance in the same causal comparison. |

The first recovery dense reward, named `corrected_dense_v1`, is the current
dense reward with one exact completion contract, contained mass-conserving
dumping, and no potential-based action veto. A candidate `transport_potential`
dense arm is only a recorded design question for now: if authorized, it must
replace rather than double-count the current dump-time relocation term and
must account for both off-zone soil and carried load. Its equation and distance
metric remain undecided.

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
11. Treat
    `sum(world action-map soil) + sum(all active carried loads)` relative to
    the reset state as a transition invariant. A map, dump, soil-relaxation, or
    partial-reset path that clips, deletes, creates, or overflows soil is
    invalid.
12. Use the small base architecture for recovery feasibility and teacher
    training. Do not mix an architecture comparison into those runs.
13. Short budgets are decision milestones, not evidence that learning has
    saturated. On a deterministic source-disjoint bank, "slight improvement"
    means either one additional successful identity in the family or current
    worst cell, or at least `0.01` absolute improvement in median terminal task
    completion, relative to the previous best scheduled evaluation. Reward,
    loss, or online success alone cannot trigger continuation.
14. At a milestone, continue the exact model, optimizer, RNG, and schedule
    lineage to the next `500 -> 1,000 -> 2,000 -> 5,000` milestone whenever
    slight improvement exists and integrity remains clean. A five-evaluation
    fixed-bank plateau with no such improvement is the stop rule.
15. A recipe is considered qualified for long training only after its
    source-disjoint family/cell gate passes twice and no semantic, integrity,
    or causal blocker remains. Qualified continuations use Euler
    `gpuhe.120h`, request `5-00:00:00`, run for at least 20,000 updates, save
    fixed-bank checkpoints at a declared cadence, and continue in additional
    20,000-update chunks while rule 13 still shows improvement. Long training
    never changes map, reward, reset, PPO, or architecture treatment in place.

## 4. Dependency graph

```text
D0 frozen first-screen receipt
 |
 +--> D1 historical semantic attribution -----> optional R0 historical fork
 |
 `--> D2 train/dev and action-mode audit ------^

C0 exact visible dump mask (ratified)
 |
 +--> C1 one completion contract
 |
 `--> C1a contained mass-conserving dump transition
          |
          +--> C2 full-reset horizon
          +--> C3 exact loader
          +--> C1b exact footprint rasterization
          `--> C4 minimal fixed evaluator
                    |
                    +--> C5 auditable training aggregates
                    |
                    `--> F0 two scratch fixed-identity probes
                            | fail
                            `--> O0/transition/reward diagnosis
                                      |
                                     F0R one-factor trench repair
                                      |
                                     B0 orthogonal feasibility cells
                               |
                              F1 two scratch family specialists
                               |
                              G0 scratch easy small generalist
                               |
                              S0 grow and qualify medium student
                               |
                              K0 global checkpoint-gated map ladder
                             /                 \
                  reward curriculum       partial resets

Architecture work remains conditional on a representation-specific failure.
```

`R0` is the sole historical-revision experiment. `F0` and every other
future-policy task use the corrected contract. D1 and D2 improve historical
attribution but do not block the already-ratified future semantic correction.
C5 and C1b are now complete and are required launch gates for every new PPO
run.

Task index:

| ID | Priority | Cost class | Depends on | State |
|---|---|---|---|---|
| D0 | P0 | documentation | completed evaluators | [x] complete |
| D1 | P0 | evaluation only | D0 | [x] diagnosis complete; historical mass-integrity gate failed |
| D2 | P0 | evaluation only | D0 | [x] 20 deterministic plus 16 sampled records adjudicated |
| C0 | P0 | decision | design review | [x] complete |
| C1 | P0 | Terra code/tests | C0 | [x] complete |
| C1a | P0 | Terra transition/tests | C0 | [x] complete |
| C1b | P0 | Terra geometry/tests | C1a | [x] complete |
| C2 | P0 | baselines code/test | C1, C1a | [x] complete |
| C3 | P0 | Terra loader/tests | C1, C1a | [x] complete |
| C4 | P0 | evaluator/tests | C1-C3, C1a | [x] complete |
| C5 | P0 | training receipts/tests | C1-C4 | [x] complete |
| O0 | P1 | conditional deterministic tests | failed F0 or direct alias evidence | [x] alias test not authorized: trajectory evidence implicates action/reward attractors |
| F0 | P0 | two scratch bounded PPO probes | C0-C5, C1a, C1b | [x] foundation passed; trench failed |
| F0R | P0 | one scratch trench reward repair | failed trench F0, diagnosis | [x] passed with terminal retention |
| R0 | P1 | two 500-update historical forks | D1, D2, F0 | [x] not authorized: shared train-and-development drift rejected |
| B0 | P1 | paired generation, five bounded panel probes, bank expansion | foundation F0, trench F0R | [ ] B0a active |
| F1 | P1 | two scratch family specialists | B0, foundation F0, trench F0R, C5 | [ ] blocked |
| G0 | P1 | one scratch small easy generalist | F1 | [ ] blocked |
| S0 | P1 | one grown medium qualification | G0 | [ ] blocked |
| K0 | P2 | global staged map campaign | S0 | [ ] blocked |
| A0 | P3 | one conditional architecture A/B | representation-specific evidence | [ ] blocked |
| W0 | P2 | corrected reward specification | C1, C1a | [ ] blocked |
| W0a | P2 | conditional dense transport A/B | F0 plus transport-specific evidence | [ ] blocked |
| W1-W2 | P2 | dense-to-terminal reward A/B/C | S0 | [ ] blocked |
| PR0 | P3 | reset A/B | selected K0 map sampler | [ ] blocked |

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

Execution receipt, retry submitted 2026-07-26:

- observer-only implementation:
  terra-baselines `d049107` plus the terminal-threshold default fix
  `1aeb1a6` (`audit_historical_curriculum.py`);
- frozen source copy:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/source`;
- historical Terra and baseline revisions remain
  `d37e780480c0fae64a4b9e4ba6638b4499748761` and
  `2722d832c8381a68d594d8bf8298ba3aec7f4c6a`;
- initial preflight `8623160` failed before producing evidence because the
  observer directly accessed an optional historical terminal-threshold field;
  its dependent jobs `8623162` and `8623163` were cancelled automatically;
- first replacement `8624492` completed the audit but the shell gate rejected
  a maximum `1.43e-6` float32 reconstruction difference against a `1e-6`
  threshold; that JSON is preserved under `failed_attempts/`;
- the explicit, receipt-recorded float32 tolerance is now `1e-5`
  (`7b5d52d`), with regression coverage;
- replacement preflight `8626340`, completed in `00:09:09` with exit code
  `0:0`;
- replacement full deterministic job `8626341` produced all 20 declared records
  and exact reset receipts, then exited `1:0` because its final shell gate found
  historical dump-observer transition mismatches;
- deterministic JSON SHA-256:
  `22abef7ca13006d31abf2bba2d268581ea5298532b5fe1fd8ece0c976e69df1b`;
- 15 M1/M2 timeout episodes have both a nonzero mass residual and at least one
  dump attempt for which the observer did not reproduce the old transition;
  their dump-veto and boundary-flow diagnostics are not admissible yet;
- bounded diagnostic implementation `04edebf` records which branch mismatched
  plus the first-step load and terrain deltas;
- diagnostic job `8632822` targets only `flat_u1000` on development M1
  deterministically, at most 28,800 transitions, under immutable diagnostic
  root
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/diagnostics/observer_mismatch_v1`;
- diagnostic launch-receipt SHA-256:
  `ffbeca6347e73bcfd44599521623599a1708cd91e79c3b702dcd231b9b315f88`;
- diagnostic job `8632822` completed in `00:10:22` with exit code `0:0`;
- diagnostic JSON SHA-256:
  `ca61654b1e1acdefdc61c7ccc888e6d854b03669e83fc81439ac4a21efb4e526`;
- all three mismatches in the selected `flat_u1000` development-M1 record were
  non-veto executed dumps, not load-state or veto-branch classification errors:
  carried loads 43, 37, and 33 became zero while actual world-soil deltas were
  only 42, 36, and 32;
- the corresponding independently reconstructed candidate deltas conserved all
  43, 37, and 33 units and differed from the actual maps by L1 values 3, 1,
  and 1. The historical transition therefore deleted one soil unit in each
  selected episode;
- every one of the original 15 transition-observer mismatches co-occurs with a
  nonzero historical mass residual. The bounded diagnostic identifies the
  selected three but does not assume that all remaining twelve share the exact
  same internal path;
- observer-derived potential-veto and boundary-flow counts are inadmissible for
  those 15 episodes. Actual terminal state, success, exact/buffer completion,
  terminal reward, and independently measured mass residual remain observable;
  and
- the deterministic command hard-limits D1 attribution to the three declared
  checkpoints over development M0-M2: exactly 259,200 maximum transitions.

D1 is a completed diagnosis but not a passed clean causal audit: the historical
transition itself fails mass integrity. Do not erase or waive that gate.
Materiality conclusions below use the fully clean M0 records and a sensitivity
view that excludes every integrity-failing M1/M2 episode.

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
- dump attempts rejected solely because predicted relocation potential
  increased;
- positive-soil delta moved across the visible target boundary by local soil
  relaxation;
- mass residual; and
- timeout completion delta.

Decision:

- any `task_done && completion < 1` confirms the contract violation;
- call it a material contributor only if at least 10% of successes or at least
  10% of top-quartile timeouts change completion by `>= 0.05` or terminal
  reward by `>= 0.1 * terminal_reward`;
- report the veto and boundary-crossing rates separately rather than folding
  them into the completion threshold; and
- otherwise fix it as correctness debt without claiming it caused the broad
  regression.

Budget: at most 259,200 evaluation transitions and no gradients.

Adjudication:

- the exact-versus-buffer completion mismatch is a material contributor under
  the preregistered threshold, but it is not established as the sole cause of
  poor generalization;
- on clean development M0, `flat_u1000` changes 11/24 successes overall:
  1/11 foundation successes (9.09%, below the family threshold) and 10/13
  trench successes (76.9%);
- on clean development M0, `flat_u4000` changes 2/11 successes (18.2%);
- on clean development M0, `staged_u4000` changes 1/11 successes and 3/14
  top-quartile timeouts, so the timeout criterion is material;
- after excluding all integrity-failing rows, the aggregate materiality
  decision remains positive for each declared M1 and M2
  checkpoint/dataset record;
- potential-veto attempts and boundary relaxation are present in clean rows,
  but their rates must be reported separately and cannot be extrapolated from
  the 15 integrity-failing episodes; and
- the corrected future contract—exact visible mask everywhere, contained
  mass-conserving dumping, and no relocation-potential action veto—directly
  removes all three confounds. No historical R0 continuation may be used as
  evidence for that corrected contract.

### D2 — Separate memorization, policy mode, and held-out regression

Execution receipt, retry submitted 2026-07-26:

- deterministic train/development audit job `8626341` saved all 20 records but
  exposed the D1 observer-integrity issue described above;
- sampled M0 job `8626343` produced the first three complete records, then was
  intentionally cancelled after `01:38:03` because its serial execution rate
  could not finish the exact grid inside the four-hour allocation;
- the preserved `flat_u1000` seeds `2026072500` through `2026072502` JSON has
  SHA-256
  `a63e7b1d44aebda1d3774c11568b9bb6aaa0c929463bf14e7a911a2a08b3158a`;
- execution-only sharding at terra-baselines `b34c122` leaves checkpoints,
  maps, seeds, horizon, action sampling, and historical source unchanged;
- immutable shard root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/sampled_shards_v1`;
- shard launch-receipt SHA-256:
  `9019d01b7515554df086ecc18ca0fafee8cada5c40aea94287eb8512384e3cff`;
- submitted-jobs receipt SHA-256:
  `db112eb390acf8722cc0d5144196a10ff70577854f93916179a4d8a111fad50e`;
- jobs `8633209`, `8633211`, and `8633220` cover, respectively,
  `flat_u1000` seeds 3-7 and `flat_u4000` seeds 0-3 and 4-7, with no overlap;
- those jobs completed with exit code `0:0` in `00:17:35`, `00:33:57`, and
  `00:34:00`;
- shard JSON SHA-256 values, in input order, are
  `a63e7b1d44aebda1d3774c11568b9bb6aaa0c929463bf14e7a911a2a08b3158a`,
  `ac0b6f799172f21bbb063df501eb8cee205f8492b04d96fa4f860d732408326d`,
  `b92bea1538a6939dd3271a9652f94be2f8b230b31eb4a89ce337aa96becdacb6`,
  and
  `c0cbd9ca9b3fc0226a3013d4db5978e40c25d5abc23a14a72decf2658a6063d1`;
- the exact 16-record merge passed uniqueness and static-contract checks and
  has SHA-256
  `3ad4805fb64b4bb8b08ed25b32c2903233e2ff551cad87c982a3b37cd9759a74`;
- declared sampled seeds `2026072500` through `2026072507`;
- exact training-identity view:
  `train/local_M2_terminal`, whose 256 slots must verify as 256 unique source
  IDs and 256 unique map IDs before evaluation; and
- checkpoint labels `e8_u20000`, `flat_u1000`, `flat_u4000`,
  `staged_u1000`, and `staged_u4000`, each guarded by its frozen SHA-256.

D2 execution and diagnosis are complete, but the historical integrity gate did
not pass. The deterministic records contain 48/2,240 episodes with nonzero
historical mass residual (maximum 3), and sampled records contain 3/1,024
episodes with one-unit residuals. None of those 51 episodes succeeded; all
reset hashes, target/obstacle immutability, finite-state, and terminal-reward
reconstruction checks passed. Results below retain the failures explicitly and
include the clean-row sensitivity where relevant.

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

Adjudication:

- the exact training view passed with 256 unique source IDs and 256 unique map
  IDs, so repeated slots were not double-counted;
- the memorization criterion is supported for the trench family at
  `flat_u4000`: training reaches 82/128 (64.1%) while source-disjoint
  development M0 reaches only 3/32 (9.38%), a 54.7-point gap even against the
  easiest development level. Foundation and overall training do not reach the
  60% threshold;
- shared train-and-development drift is rejected. From `flat_u1000` to
  `flat_u4000`, unique-training success rises from 54/256 to 122/256 while
  development M0 falls from 24/64 to 11/64. Both families improve on training
  identities, so the preregistered joint-decline condition is false;
- diffuse-policy sensitivity is supported at both checkpoints. Update 1,000
  sampled counts are 31, 29, 36, 34, 30, 35, 34, and 29 (mean 32.25/64,
  +12.9 points over deterministic 24/64); update 4,000 counts are 21, 19, 20,
  24, 20, 20, 22, and 19 (mean 20.625/64, +15.0 points over deterministic
  11/64). All 8/8 seeds improve at each checkpoint;
- sampled action disagreement averages 35.1% at update 1,000 and 20.0% at
  update 4,000, consistent with useful probability mass outside the argmax;
- the three sampled mass-integrity failures all occur at update 1,000, are
  unsuccessful episodes, and do not change either sampled-action decision
  under clean-row sensitivity; and
- the viable explanation for the old screen is family-specific identity fit
  plus held-out generalization regression, compounded by a material task
  contract mismatch and deterministic action-mode sensitivity. It is not
  evidence for architecture insufficiency or for shared optimization drift.
  R0 is therefore not authorized.

## 6. Phase C — establish one corrected future-policy contract

### C0 — Legal dump-mask decision

Status: complete.

Ratified contract:

> The explicit map dump mask is the legal dump region. Any tolerance region
> must be materialized by the generator, visible in the map/gallery, included
> in capacity calculations, and stored in the manifest. There is no hidden
> one-cell legal buffer.

For the current representation, the direct definition is
`accepted_dump_mask = (target_map > 0) & ~obstacle_mask`, with generation
rejecting any target/obstacle overlap.

This recommendation preserves the meaning of arbitrary dump constraints and
prevents the environment from silently accepting soil outside the reviewed
zone. An off-zone dump may remain a physically executable mistake; it is never
part of the accepted mask or successful completion.

Deliverable:

- one named `accepted_dump_mask` definition;
- a short design decision in `TRAINING_DESIGN.md`; and
- no second termination-specific or reward-specific dump mask.

Decision receipt: ratified with Lorenzo on 2026-07-25. No further mask choice
is required before implementation.

### C1 — Unify termination, completion, reward, and evaluation

Status: complete.

Verified implementation receipt, 2026-07-25:

- Terra exposes the named `exact_visible_dump_v1` contract and computes dig,
  exact-dump purity/volume, unloaded, task-present, dump-mask-integrity, and
  applicable edge components once;
- termination, terminal reward, reward components, and the compatibility
  completion accessor use the same minimum-reduced absolute completion;
- exact-zone, former-buffer, off-zone, relocation-only, combined, partial,
  loaded, empty-task, obstacle-overlap, edge, terminal-reward, eager, `jit`,
  and `vmap` tests pass; and
- the C4 evaluator now records the same named contract, hard-fails any
  `task_done <=> absolute_completion == 1` disagreement, and labels the
  imported environment as either corrected or legacy; and
- all 111 baseline tests and the real corrected-environment C4 smoke pass.

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

### C1a — Make dumping contained, mass-conserving, and non-greedy

Status: complete.

Verified implementation receipt, 2026-07-25:

- tracked-excavator dumps prefer physically reachable exact-mask cells and
  otherwise permit an entirely off-zone, recoverable mistake;
- correctly aimed and wrong dumps constrain local soil relaxation to opposite
  sides of the accepted-mask boundary;
- complete-load mass, containment, and `int8` representability are checked
  before changing world soil or carried load;
- reward-potential transition vetoes were removed from active dump and
  implicit reverse-dump paths;
- obstacles and non-dumpable cells are excluded per tile instead of vetoing an
  otherwise usable dump workspace; and
- 16 focused contract tests plus all 39 Terra tests passed,
  including repeated dump/re-lift, overflow rejection, potential increase,
  and eager/`jit`/`vmap` agreement.

Capacity-validator receipt, 2026-07-25:

- the loader computes accepted cells from the exact visible mask, rejects
  target/obstacle and target/non-dumpable overlap, checks the declared
  single-layer ratio, and verifies total and maximum-bucket `int8` headroom;
- focused insufficient-area and unplaceable-bucket fixtures pass;
- the old `terra_training_design_v1_20260724` bank was audited and has minima
  near 2.0x, so it is explicitly ineligible for the new 3x starter contract;
  and
- regenerated F0 receipts now prove 63.0x capacity and one-tile path distance
  for the all-around foundation, and 9.27x capacity with two-tile p95 path
  distance for the broad both-side trench. Both are obstacle-free, exact-mask
  datasets and load successfully through the strict C3 path; the frozen bank,
  gallery, and validation JSON are under
  `.artifacts/terra_curriculum_recovery_20260725/f0_starters_v1/`.

Scope the first implementation to the single tracked-excavator recovery path.
Do not build a configurable spill framework.

Transition rule:

1. Build the physically reachable dump workspace using the existing geometry,
   obstacle, traversability, and dumpability constraints.
2. If that workspace intersects `accepted_dump_mask`, interpret the action as
   a correctly aimed dump. Deposit the complete carried load only on the
   reachable accepted cells and restrict local soil relaxation to the accepted
   mask.
3. If the workspace contains no accepted cell but contains physically
   dumpable off-zone cells, permit the wrong dump and restrict its deposition
   and relaxation to off-zone valid cells. The resulting soil remains illegal
   until recovered.
4. Never move soil across the accepted-mask boundary during either path.
5. Never clip or delete an unplaceable remainder. Fail the dump without
   changing world soil or carried load if the complete load cannot be
   represented.

Remove relocation-potential comparisons from action validity and transition
acceptance. In particular, the active `_handle_dump` path must not return the
unchanged state merely because predicted relocation potential is higher. Any
equivalent veto in an action-availability or implicit reverse-dump path must
also be removed for the active recovery agent.

The map validator must compute capacity from the exact accepted mask. Starter
cells use at least `3x` reachable single-layer-equivalent capacity and must
also prove that all valid bucket loads stay within the action-map numeric
range under the contained pile rule.

Required deterministic cases:

- an interior legal dump;
- a legal dump whose unconstrained soil relaxation would cross the boundary;
- a workspace overlapping both legal and neutral cells;
- an entirely off-zone wrong dump;
- a legal region with insufficient representable capacity;
- a dump adjacent to an obstacle or non-dumpable tile;
- a dump that increases relocation potential but is physically valid; and
- repeated dig/lift/dump/re-lift sequences.

Acceptance:

- `sum(action_map) + sum(active loaded soil)` is exactly conserved on every
  successful transition and unchanged on every rejected transition;
- a legal dump creates no positive-soil delta outside
  `accepted_dump_mask`;
- an off-zone dump creates no positive-soil delta inside
  `accepted_dump_mask`;
- carried load decreases by exactly the soil added to the world;
- no integer overflow, wraparound, clipping, or silent remainder loss occurs;
- the former potential-increase veto has no effect on transition legality; and
- eager, `jit`, and `vmap` cases agree.

Physical boundary spill becomes a later named dynamics treatment only after
the contained contract passes family and retention gates.

### C1b — Preserve the exact excavator footprint

Status: complete.

Verified implementation receipt, 2026-07-26:

- the pre-existing polygon rasterizer sampled integer cell corners under
  strict half-plane tests, shrinking an axis-aligned `W x H` footprint to
  `(W-1) x (H-1)`, and its x/y grid construction was transposed;
- Terra commit `f3eeca6a` samples cell centers in `[x, y]` order and returns a
  `(map_height, map_width)` mask;
- a regression built from the production `get_agent_corners` path proves that
  an odd `5 x 3` excavator occupies exactly 15 correctly oriented cells;
- the focused footprint, dump-contract, and partial-loading set passes
  29 tests; and
- the full Terra suite passes 51 tests plus 6 subtests, with formatting,
  linting, and whitespace checks clean.

This issue was discovered while closing the training-integrity gates and was
fixed before any corrected-contract PPO production launch.

Acceptance:

- centered and boundary-touching footprints preserve their declared cells;
- non-square footprints are not transposed;
- odd production dimensions occupy exactly `width * height` cells at zero
  rotation; and
- dump and partial-loading transition tests remain green.

### C2 — Restore the full-reset horizon contract

Status: complete.

Verified implementation receipt, 2026-07-25:

- the initial full-task reset no longer randomizes `env_steps`;
- the training path asserts that every initial `env_steps` value is zero;
- the run metrics record the minimum and maximum configured effective
  horizon, while the direct fixed evaluator rejects any horizon other than
  450;
- no `randomize_initial_env_steps` reference remains; and
- all 42 focused training-utility tests passed on CPU, including zero and
  nonzero reset fixtures.

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

Status: complete.

Verified implementation receipt, 2026-07-25:

- future multi-map loading requires the named
  `terra_exact_map_dataset_v1` contract before constructing JAX arrays;
- `DATASET_SIZE`, declared slot count, manifest rows, contiguous indices, and
  all target/action/occupancy/dumpability/distance/metadata sidecars must
  agree exactly;
- the contract records unique identities, explicit per-slot weight and
  identity multiplicity, shape, distance metric/normalization, exact dump
  contract, and an optional capacity floor;
- a hashed source registry is verified and rejects a source ID assigned to
  more than one split;
- missing sidecars, count/multiplicity mismatches, source overlap, invalid
  distance data, and a violated capacity floor all have deterministic failure
  fixtures; and
- all 44 Terra tests pass. The partial-reset generator uses one explicit
  legacy-contract test opt-out until its separate PR0 bank gate is active.

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

Status: complete.

Verified implementation receipt, 2026-07-25:

- direct policy evaluation can preserve each first terminal state instead of
  auto-resetting it; inactive environments are frozen for the rest of the
  batch rollout;
- exact reset verifies target, initial action, occupancy, initial
  dumpability, reward distance, trench/foundation metadata, and zero elapsed
  steps, with aggregate layer hashes in the JSON receipt;
- deterministic versus sampled mode, manifest provenance and slot weights,
  verified reset slot, completion components, mass residual, no-effect action
  count, immutable-map mutation, non-finite state, and termination/slot
  disagreement are saved per map;
- any integrity failure blocks `mastery_gate.passed`, while legacy
  environments remain labeled and cannot earn corrected-contract mastery
  without the integrity fields;
- `aggregate_fixed_bank_history.py` requires two adjacent passing
  checkpoints and evaluates the five-percentage-point family retention rule;
- focused fixtures prove that one perfect-performing map with a mass error
  cannot pass and that one checkpoint cannot claim consecutive mastery;
- all 111 baseline tests pass; and
- a real one-step foundation rollout with a scratch
  `resnet_spatial_8x8` model reported supported integrity, slot 0, zero mass
  residual, no target/obstacle mutation, and no non-finite state.

Keep one direct fixed-bank evaluator and add:

- explicit deterministic or sampled mode in every receipt;
- map ID, source ID, family, stratum, primary cell, and slot weight;
- selected slot index verified from the exact reset key and preserved in the
  fixed evaluator's terminal accumulator so outcomes join to manifest
  provenance; C5 separately carries provenance through high-volume training;
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

Status: complete.

Verified implementation receipt, 2026-07-25:

- Terra commit `f6bfc007` exposes pre-reset timeout, action-effect,
  productive-workspace-cycle, mass-residual, immutable-target/obstacle, and
  exact manifest-provenance diagnostics;
- terra-baselines commit `475ae47` carries per-environment episode state
  across PPO rollout boundaries and writes one bounded
  `terra_training_episode_aggregate_v1` JSON grouped by stage, family, primary
  cell, and separate `task_done`, `timeout`, `both`, and `other` reasons;
- terra-baselines commit `6c56525` additionally reduces mass residual and
  immutable-target/obstacle mutation over every transition and every device,
  aborting before any checkpoint even when no episode terminates in that PPO
  update;
- additive values use global device sums while integrity maxima use global
  maxima; W&B receives only reduced totals and rates, never an arbitrary
  environment element;
- fixtures cover a two-window episode, population-equivalent shard reduction,
  all terminal labels, and checkpoint-blocking mass, mutation, and reward
  reconstruction failures;
- the full terra-baselines suite passes 130 tests after the F0 launch and
  checkpoint-lineage gates; the final focused aggregate suite passes 5 tests;
- a one-update strict-F0 terminal-path CPU smoke at horizon one records exact
  `foundation / all_around_low_volume` provenance, one timeout, one action,
  return `-0.005` exactly reconstructed by its components, and zero mass,
  mutation, or reward-integrity failures; and
- its exact saved checkpoint reload has 50 finite model leaves, finite
  optimizer state, and `next_update == 1`.

A second real PPO integration smoke used the exact regenerated F0 foundation
manifest with a deliberately reduced CPU shape (one device, two environments,
one step, one update). It exercised the final per-transition hard abort and
checkpoint schema, recorded exact
`foundation / all_around_low_volume` provenance, and produced zero transition
integrity failures. This is implementation evidence only; it does not replace
either production-shaped four-GPU F0 smoke.

Machine-readable smoke receipt:

- aggregate:
  `.artifacts/terra_curriculum_recovery_20260725/c5_terminal_smoke/episode_aggregates/c5-terminal-smoke_update_000001.json`,
  SHA-256
  `29f4cdc2910f43a52781392be19a651883b3243ddd225079ccc5ca5e6cd5ed91`;
- checkpoint:
  `.artifacts/terra_curriculum_recovery_20260725/c5_terminal_smoke/c5-terminal-smoke_FINAL.pkl`,
  SHA-256
  `b57789dbca7fca20ff6e5cb8144444c89d5920f3d4d86f782b774c2b6f46c60a`.
- F0-path CPU checkpoint:
  `.artifacts/terra_curriculum_recovery_20260725/f0_launch_cpu_smoke/f0-launch-cpu-smoke-local-2026-07-26-01-30-27_FINAL.pkl`,
  SHA-256
  `4c67c470cbba240f6ddf00bac7f09aaffb4892fc0e38fc956ca35d664d8c3f0b`;
- F0-path CPU aggregate:
  `.artifacts/terra_curriculum_recovery_20260725/f0_launch_cpu_smoke/episode_aggregates/f0-launch-cpu-smoke-local-2026-07-26-01-30-27_update_000001.json`,
  SHA-256
  `5ca79f8bdd1e37030f4374923a631f75a3fe1ae58e6f51b21b5b41802a1e77fa`.

C5 corrective amendment, 2026-07-26:

- the first foundation production run exposed a false hard failure in schema v1:
  reward and reward components were accumulated independently for as many as
  450 signed float32 steps and only compared at episode end;
- terra-baselines `c58ad23` changes the hard invariant to reward reconstruction
  on every transition, while retaining the independently accumulated
  episode-level difference as an explicitly informational drift metric;
- `terra_training_episode_aggregate_v2` records both quantities separately, and
  the checkpoint-blocking gate uses only
  `step_reward_residual_violation_count`;
- a deterministic 450-step regression with 225 rewards of `+0.8`, 225 of
  `-0.8`, and exact `0.005` existence components reproduces a v1 episode drift
  violation while every per-transition residual is exactly zero;
- a true per-transition missing-component case still fails;
- the full terra-baselines suite passes 131 tests with 69 warnings, the focused
  aggregate suite passes 6 tests, shell syntax, byte-compilation, and critical
  Ruff checks pass;
- a real reduced F0-path PPO checkpoint reloaded with 92 finite model leaves,
  185 finite optimizer leaves, schema v2, and zero transition-level violations:
  checkpoint SHA-256
  `ec48f2058d64f7888745068f4bbff91ac66c17ddf1c9566997ccafea417c1064`
  and aggregate SHA-256
  `4475d5ffc870a64255fd426329e876c8af7f87c4112bf6630e8675d184a6171e`;
  and
- a separate horizon-one PPO smoke forced one completed timeout episode whose
  return and components reconstruct exactly, with zero mass, mutation, or
  step-reward violations: checkpoint SHA-256
  `8805fcf070776b95672d3e025cb0742801348ef6dc0aacd2ea473b45633c71b8`
  and aggregate SHA-256
  `6cc302e0a03a19212050f20962e8206ab7078e6777e047192821322074693411`.

This amendment repairs the receipt gate; it does not change PPO, reward values,
map identity, reset, model, or the preregistered F0 feasibility treatment.

The completed runs expose a logging defect: scalar reward-component fields are
taken from the final state of one environment, while terminal completion fields
cover device 0 and pool successes with timeouts.

Add one bounded full-rollout aggregate keyed by family, primary cell, active
stage, and termination reason. Reduce across every environment and device
before host logging. For each key, preserve raw count and sum fields for:

- exact-target, accepted-mask, buffer-only, and illegal dump volumes;
- dig, dump, and combined completion;
- episodic return and every reward component summed over terminal episodes;
- terminal reward before and after normalization;
- steps, action counts, invalid/no-op actions, and productive workspace cycles;
- mass and immutable-map integrity fields; and
- enough raw counts to recompute every W&B rate offline.

Carry reward-component and episode-stat accumulators per environment across
rollout boundaries. Snapshot and reset them only on that environment's
terminal transition; otherwise an episode spanning two PPO rollouts is
silently truncated.

Do not log an arbitrary environment element as an aggregate. Save the bounded
aggregate as machine-readable JSON at each logging interval and use W&B only
for its reduced totals and rates. Per-map, per-episode receipts remain a fixed
evaluator responsibility; do not stream every training episode or build a
generic event-logging framework.

Acceptance:

- global counts agree with a single-device reference fixture;
- a fixture spanning two rollout windows produces one complete episode sum;
- success, timeout, and simultaneous success/timeout are separately labeled;
- reward-component sums reproduce total episodic return within tolerance;
- no field silently pools success and timeout; and
- all fixed-evaluator and training receipt fields use C1's completion source.

## 7. Phase O — test observation sufficiency before changing the model

### O0 — Construct paired-state alias tests

Do not block the first corrected fixed-identity probes on a broad observation
audit. Trigger this task when a direct paired state is already known or when an
F0 failure implicates missing state rather than transition or reward behavior.

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

Dependencies: C0-C5, C1a, and C1b. A broad O0 audit is not a prerequisite for
this bounded probe.

Select and visually record:

- one regenerated low-volume foundation with all-around dumping; and
- one regenerated low-volume straight trench with broad both-side dumping.

These are new quantitative starter identities under the corrected contract,
not reused historical M0 maps.

Run two independent policies, one per identity. For each:

- initialize a base `resnet_spatial_8x8` policy from scratch;
- use an independent declared initialization seed, fresh optimizer, and fresh
  schedule;
- use only `corrected_dense_v1`;
- use full 450-step resets with `env_steps == 0`;
- use no partial resets, map curriculum, reward curriculum, or architecture
  change;
- save a checkpoint every 100 updates, then evaluate all ten checkpoints
  deterministically on the same 32 declared reset seeds; and
- run the preregistered 1,000 PPO updates. This first implementation does not
  claim online early stopping. Extend once to 2,000 only if no two-checkpoint
  pass exists and the fixed-seed curve is still improving.

At the current four-device, 1,024-environment-per-device, 32-rollout-step
shape, one update is 131,072 global transitions; 1,000 and 2,000 updates are
131,072,000 and 262,144,000 transitions. Record both units and recompute them
if the probe shape changes.

Frozen F0 launch receipt:

- implementation: terra-baselines `6c56525`;
- foundation initialization seed: `2026072601`;
- trench initialization seed: `2026072602`;
- common evaluation reset seeds: integers `2026072600` through `2026072631`;
- one independent four-RTX-4090 job per identity, with 1,024 environments per
  device, 32 rollout steps, two update epochs, and 32 minibatches;
- explicit learning rate `3e-4`;
- entropy coefficient cosine-annealed from `0.15` to `0.005` over 950 updates,
  so the bounded probe reaches its low-exploration regime before update 1,000;
- base `resnet_spatial_8x8`, float32 encoder, flat minibatch shuffle, no value
  clipping, and finite checks every update;
- no resume, warm start, teacher, map-stage transition, reward-stage
  transition, or partial reset;
- one exact production-shaped update-1 smoke per arm before its production
  command, including reload of the saved model and optimizer plus validation of
  the C5 aggregate;
- 1,000 mandatory per-update aggregate receipts, ten periodic checkpoints, and
  one final checkpoint before a training-complete marker can exist; and
- the evaluator rejects a checkpoint whose seed, treatment, optimizer lineage,
  model, integrity receipt, cadence, or map identity differs from this frozen
  declaration.

The new remote root is
`/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0`.
It must contain immutable source and bank SHA-256 manifests before submission.
The two training arms may run concurrently because they answer independent
fixed-identity feasibility questions. Their dependent evaluators may run only
after the corresponding training job completes successfully.

Submission receipt, 2026-07-26 01:47 CEST:

- launch receipt SHA-256:
  `bc1ab0f3808727b2e0ce13305860a95d7dbe1b4f167f2688642b75dc47fab6e4`;
- submitted-jobs receipt SHA-256:
  `975c0852391b96a73a2f1aa202856b52214811406217351d50adf27fa3832bb4`;
- foundation train job `8629884`, requesting exactly four RTX 4090 GPUs,
  four CPUs, and 32 GB;
- trench train job `8629885`, with the same independent resource request;
- foundation evaluator `8629886`, `afterok:8629884`, requesting one RTX 4090;
- trench evaluator `8629887`, `afterok:8629885`, requesting one RTX 4090; and
- both training jobs were `PENDING (Priority)` at the first scheduler audit.

Submission is not a passed smoke or a training result. Keep F0 and checklist
items 7-8 open until the corresponding machine-readable gates exist.

Foundation update-1 GPU smoke receipt, job `8629884`:

- runtime preflight saw exactly four RTX 4090 devices and passed cuDNN and NCCL;
- the exact 4 x 1,024 x 32 update ran from seed `2026072601`;
- the FINAL checkpoint reloaded with 92 finite model leaves, 185 finite
  optimizer leaves, `next_update == 1`, and the complete frozen config;
- transition mass residual, target mutation, and obstacle mutation are all
  zero;
- smoke gate SHA-256:
  `daa0257ba1e17a195ea8654d9ebbdfbe9344d9b385d8a258990cb3ba0aeb2c36`;
- FINAL checkpoint SHA-256:
  `68b5ab8f6de6a5c211a4dbe407a831aa7e993e888237f99ab3d810a708a2d2ba`;
- update-1 aggregate SHA-256:
  `a5e8b4ef26fa97f1a2f0f870cfe7fef9b68d9c4c72459d33dbfa7733b9473b33`;
  and
- production continued as W&B run `u7hhtnrh`.

The independent trench update-1 smoke in job `8629885` also passed the same
four-RTX-4090 runtime, finite-checkpoint, configuration, and transition gates:

- smoke gate SHA-256:
  `5ba0469aa8d5ef2b7faeffe86461682c194e9d7c60d241575c23f0397a2ea888`;
- FINAL checkpoint SHA-256:
  `69c95ba12de91e7260520c7800d6d2be88b2c7955b4ba9ba01423100417adedc`;
  and
- update-1 aggregate SHA-256:
  `f0b8a2fa7aa3055abb49c0748deb3ec8ce52b4ac07613a5f96b012446455ae7b`.

First-attempt incident receipt:

- foundation production job `8629884` completed 56 valid updates at roughly
  30.5k global transitions/s, then failed before update 57 could be written or
  before the first update-100 checkpoint;
- the sole reported failure was
  `reward_residual_violation_count=1`; log SHA-256
  `5bc005ec3172f3b9fc51713e5b06f8246b18a4dacd99457b04f7195ef5da2d1e`;
- the deterministic long-episode regression above proves this was a C5
  float32-association false positive, not evidence of a missing reward component
  or failed map feasibility;
- dependent foundation evaluation `8629886` was cancelled by `afterok`;
- trench job `8629885` was intentionally cancelled after its smoke and before
  production execution under the known-defective v1 gate; dependent evaluator
  `8629887` was cancelled; and
- all first-attempt source, logs, smokes, and 56 foundation aggregate receipts
  remain preserved under the original immutable F0 root.

No first-attempt production result is admissible as F0 evidence. Replacement
jobs must start from scratch at the same declared seeds and treatment under
terra-baselines `c58ad23`, use a distinct immutable `f0_retry1` root, and repeat
both exact production-shaped update-1 GPU smokes under schema v2. Checklist
items 7-8 therefore remained open until the retry evidence below.

Corrected retry submission receipt, 2026-07-26 02:33 CEST:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0_retry1`;
- Terra revision `188362cd34de6757bc93fff1931561e173a9f2a8`;
- terra-baselines revision `c58ad233e4478316e8aded57361bc29850af6317`;
- source-manifest SHA-256:
  `d1d28bedecfe26767fe902664a76d11bad82bd53c5968b9308416a5152856787`;
- unchanged bank-manifest SHA-256:
  `0e02471987700460b37e42a88bd13548a5cc68ce812f3986514a2dffd3c53d2b`;
- bank-validation SHA-256:
  `40cab18be4527490e531c6289236077e553ce4e322ddb4ebe379d8b56c5c51cc`;
- launch-receipt SHA-256:
  `43681f81faccb8729938b8df07eb90b29f7ffa3421f085bcde16ee40ad751a60`;
- submitted-jobs receipt SHA-256:
  `543a88804305a0350b30d0659cef75922055b2239938d18e5c85663e23a2fc3c`;
- foundation train/eval jobs `8632268` and `8632273`;
- trench train/eval jobs `8632271` and `8632307`; and
- both training jobs were `PENDING (Priority)` at the first scheduler audit,
  with evaluators held by `afterok`.

The launch scripts pass the retry root explicitly through Slurm and override the
log path at submission, preventing an accidental write back into the
first-attempt root.

Corrected retry update-1 GPU smoke receipts:

- foundation job `8632268` passed the exact four-RTX-4090,
  4 x 1,024-environment x 32-step update under aggregate schema
  `terra_training_episode_aggregate_v2`;
- foundation smoke-gate, reloaded FINAL checkpoint, and update-1 aggregate
  SHA-256 values are
  `d193f9e9d94b9299664dab235a2cdcf074b4b0ad39d1cde3c4c9ab6005c60adb`,
  `072f9bd402dc8be697c578058a4b85f32fd72e1bf21fc9e2bc27dd219d00bb82`,
  and
  `921cf537c1a62e4634fdc12adb46d22a5e16ae7430c81004e49699c20a347308`;
- trench job `8632271` independently passed the same production-shaped gate
  with seed `2026072602`;
- trench smoke-gate, reloaded FINAL checkpoint, and update-1 aggregate SHA-256
  values are
  `9e87b0fc93f14bb36d61e4150ee175fcf4b6906d429638dc5b84e63647d39b06`,
  `713d1f9761faa6f33e3b95c31ee7db8f99c4dc6051a831f93ba1d5536059c60a`,
  and
  `7102bfc8f1dbce1248d2265082e996de16f32fb6e1a3fdf099354fed47edbf42`;
- both gates reloaded 92 finite model leaves and 185 finite optimizer leaves,
  matched the frozen configuration and lineage, and reported zero mass
  residual, target mutation, and obstacle mutation; and
- both jobs then entered fresh 1,000-update production runs. No production
  checkpoint or feasibility result is inferred from the smoke.

Corrected retry terminal training receipts:

- foundation job `8632268` completed 1,000 updates in `01:34:41` with exit code
  `0:0`; trench job `8632271` completed in `01:31:48`, also `0:0`;
- each arm has exactly 1,000 schema-v2 aggregates, checkpoints at updates
  100–1,000 in increments of 100, one FINAL, and `TRAINING_COMPLETE`;
- independent terminal gates reload both FINAL and update-1,000 with 92 finite
  model leaves and 185 finite optimizer leaves, prove their model and optimizer
  trees equal, and verify every hard aggregate field;
- foundation and trench training-gate JSON SHA-256 values are
  `800ea2648faaa584988523647576b9d7d93b7c367782bc59ef4facc26380ae19`
  and
  `0379637fd80925911c9aee11e75f514c9a4dc718f4128d1b9e64067de00b94c2`;
- foundation FINAL and update-1,000 SHA-256 values are
  `8ab3cf96f8de4543e8cc071e61e68fc60b4adf779acf92358e3d698b54b505c9`
  and
  `7c3344c270a3e0d25d847810a43fe9b996c5778ec480521a20ba9cf051aea00a`;
- trench FINAL and update-1,000 SHA-256 values are
  `ff76ad33b0db7bbc4455affafb66b329ec1ef99529baba1dbc19930e6f2eaa3e`
  and
  `cbdbe766a208002dbf6ed16fef22da810e3ace4957e084020f5ed85e0b512a0c`;
- no transition has a mass, target, obstacle, or per-step reward-reconstruction
  violation. The retained episode-level float drift is informational: 16
  foundation episodes and one trench episode over the entire run;
- foundation records 2,690,368 online successes and 94,750 timeouts; trench
  records only 8 successes and 290,808 timeouts; and
- both arms executed the full declared 131,072,000 global transitions.

Corrected retry fixed-evaluation receipts:

- foundation evaluator `8632273` completed in `00:11:48`, exit `0:0`; trench
  evaluator `8632307` completed in `00:19:07`, exit `0:0`;
- foundation and trench evaluation JSON SHA-256 values are
  `a05286563e8fdd42e51b67b6b3f12304dbbc95b7f2fdfa183ee2b2b45a774ad0`
  and
  `16b5893fbba396dcfa8adc7751384a1207345238e4317172e3b554c12f298479`;
- every one of the 640 fixed rollouts passes reset, mass, immutable-map, finite
  state, completion, and checkpoint-lineage integrity;
- foundation successes at updates 100–1,000 are
  `0, 0, 16, 32, 32, 32, 32, 32, 32, 0` out of 32. Passing pairs are
  400/500 through 800/900, and a legal successful trajectory is saved;
- update 900 is the foundation feasibility witness, SHA-256
  `68a57f34e0e1cc3f806e8746de27a7e607d3852ec18c8aee1656b1a8fb44c721`.
  Update 1,000 is not promotable: deterministic behavior regresses to 0/32 and
  usually performs no excavation despite its finite, lineage-valid state;
- trench is 0/32 at every checkpoint. Its best individual fixed resets reach
  48/66 legal moved units (72.7% completion), while many initial poses make no
  progress or settle into no-effect behavior; and
- no 2,000-update extension is authorized. Foundation already passes, whereas
  the trench fixed curve is flat, its productive-cycle rate declines, and the
  extension criterion requires continued fixed-bank improvement.

F0 adjudication:

- the corrected action/observation/dynamics contract can learn the foundation
  identity, so broad architecture insufficiency and universal PPO failure are
  rejected;
- the foundation terminal collapse is a retention/action-selection failure,
  not a feasibility failure, and must be protected by checkpoint-bounded
  promotion rather than final-checkpoint selection;
- the tested trench treatment fails cleanly. Its partial legal progress rules
  out a completely broken dump transition, but does not yet distinguish
  geometry/pose reachability, local reward incentive, or policy cycling; and
- B0, F1, and every broad descendant remain blocked until the bounded trench
  trajectory/reward diagnosis selects and validates one minimal repair.

Bounded trajectory-diagnosis launch receipt:

- observer-only code is sealed at terra-baselines `9935c67` under
  `f0_retry1/diagnostics/trajectory_v1`; it does not modify either immutable F0
  arm;
- launch-receipt and source-manifest SHA-256 values are
  `75eb0edb621db6d9d7ab4f086b5bd9d08adf561c998f3b976e508428e2bca645`
  and
  `af5338d6b209e24d6a1de70b2f63a87183b4c4f8ca06119775eac776c9c3cb52`;
- the four preregistered replays are foundation updates 900/1,000 and trench
  updates 900/1,000. Each uses the original 32-reset batch and transition RNG,
  for at most 57,600 transitions and zero gradient updates;
- compact traces are retained for the foundation retention control
  `(900, 2026072600)` versus `(1000, 2026072600)`, the trench progressing/stalled
  pair `(900, 2026072600/2026072601)`, and two distinct update-1,000 failures
  `(2026072602/2026072611)`;
- the diagnostic is inadmissible unless all 128 replayed rows match the sealed
  evaluator in success, termination, length, no-effect count, return, and every
  completion field. Only then may policy logits, action-effect opportunities,
  reward components, repeated states, and counterfactual DO rewards select a
  repair; and
- one-RTX-4090 job `8642618` was submitted at `2026-07-26T02:49:32Z`;
  submitted-job receipt SHA-256 is
  `e2062250a9153eda0968250238abd401f476e94dde77c5edfac82dcc893acfe8`.
  It failed after `00:00:26`, exit `1:0`, because the shared Euler venv has no
  `pytest`; runtime and both immutable manifests passed, and no replay began;
- the failed root remains sealed. Retry root `trajectory_v1_retry1` changes no
  treatment or diagnostic logic and invokes the same two focused test
  functions through Python `runpy`;
- retry launch-receipt and source-manifest SHA-256 values are
  `259e78fc95227daca3629126b3d2a5c6f592f437aa7a56882ee3ce7b526f1236`
  and
  `63c103e5e261671a09e6bf8d2d2ea529f903b17100e0b6b11ba52ab57ce948a1`;
  and
- retry job `8642753` was submitted at `2026-07-26T02:52:14Z`; its
  submitted-job receipt SHA-256 is
  `e46e03b26e126c17ae250ed2344f7d442160f8c1c1b75a7ee92a7916318fab1a`.
  It completed in `00:16:40`, exit `0:0`;
- all four replays and all 128 sealed evaluator rows match exactly, including
  zero maximum float error. The 57,600-transition observer gate passed with
  zero gradient updates; output and log SHA-256 values are
  `8b313453fb3c420d2730195252d82c8ad1afb1c9b88d35a888e2fc7ab6115d3f`
  and
  `be49a22907108ec32039e3ed80dcf170e22231ee4b0a3a8b1e2a7aca9e4dc7ed`;
- the successful foundation update-900 control finishes in 21 effective steps.
  Update 1,000 instead alternates 220 forward and 222 backward actions, visits
  only 11 physical states, and never digs. DO would have an effect on 227
  steps, has positive immediate reward advantage on all 227, yet is ranked
  eighth by the policy every time;
- trench update 900 seed `2026072600` makes 20 effective actions, moves 48/66
  units legally, then selects 430 explicit no-ops. Seed `2026072601` makes nine
  effective moves and then selects 441 no-ops; movement and rotation remain
  effective, but it never reaches a DO-effective workspace;
- trench update 1,000 seed `2026072602` cycles through only six physical states
  for 450 effective movement actions without reaching a DO-effective workspace.
  Seed `2026072611` digs 41/66, places 19, remains loaded with 22, then
  oscillates between 218/219 cabin rotations. DO is effect-capable on 441
  steps but selected only three times; 437/438 missed DO actions have higher
  immediate reward, by `+0.270` on average; and
- the 100-update training-receipt bins independently show the trench no-op
  fraction rising from `13.3%` to `92.2%`, no-effect actions from `35.8%` to
  `92.7%`, and productive cycles falling from `6.73` to `2.09` per episode
  while mean return improves from `-7.91` to `-1.11`. The foundation control
  learns while its no-op/no-effect rates fall.

Diagnosis:

- the map is statically valid and supports repeated legal dig/dump progress;
  transition, capacity, mass, and exact-mask contracts are not the blocker;
- no trace requires different hidden outcomes from an identical model input.
  This is not a broad alias search, so it supplies no authorization for O0
  feature or recurrence work;
- the failed policies exploit idling or short motion/cabin cycles after easy
  progress. The foundation control rejects a universal PPO or base-architecture
  failure, but its update-1,000 collapse makes checkpoint-bounded retention
  mandatory; and
- the first implicated treatment is the trench-only absolute distance/alignment
  reward. It becomes less negative as the population settles and return
  improves while task work collapses. This is a causal hypothesis, not yet a
  selected reward: it requires the one-factor F0R ablation below.

### F0R — Remove absolute trench shaping as a one-factor repair

Dependency: the completed failed-trench diagnosis above. Foundation is not
rerun because it already supplies the matched PPO/model feasibility control.

Freeze every F0 trench choice, including:

- the same exact trench identity, reset bank, horizon, initialization seed
  `2026072602`, fresh optimizer, 4 x 1,024 x 32 PPO shape, learning rate,
  entropy schedule, model, checkpoint cadence, and 1,000-update budget;
- `corrected_dense_v1` action rewards, exact completion, terminal reward,
  transition, and integrity gates; and
- independent update-1 GPU smoke, ten fixed evaluations, and the same
  29/32-at-two-consecutive-checkpoints pass gate.

Change exactly one environment field: set `apply_trench_rewards=false`. This
removes the absolute per-step distance/alignment term; it does not add progress
reward, change action costs, change the map, alter PPO, or begin W0a/W1.
Record the treatment as `corrected_dense_v1_trench_absolute_off`.

Implementation and launch seal:

- terra-baselines `b203d8a922742ec62dfd62cd9b2e24cd7b6eaa3e`
  adds the treatment-specific preset, independent smoke/evaluation contract,
  and `terra_f0_training_gate_v1`. The latter reloads every numbered
  checkpoint and `FINAL`, requires exact update-1,000 model and optimizer
  equality, and audits all 1,000 population receipts before marking training
  complete;
- a field-for-field regression test proves that the control and F0R presets
  differ only in their name, description, and
  `maps[0].apply_trench_rewards`; the semantic treatment changes only that
  Boolean. The seven F0 evaluator tests and two terminal-verifier tests passed
  in the sealed Euler source environment;
- immutable run root
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0r_trench_absolute_off_v1`
  reuses the exact F0 retry bank without regeneration. Its source manifest is
  `6b32132f67fd33021094119bd6f9ab320290276f5fd4c0d19eddfc3aada71116`,
  its unchanged bank manifest is
  `0e02471987700460b37e42a88bd13548a5cc68ce812f3986514a2dffd3c53d2b`,
  and its preregistered launch receipt is
  `48ebe22b248accfa9915c53db99d1f306f95dcb464681c7131cba4f72ae78bf1`;
  and
- sealed source revisions are Terra
  `200e30d55e7999bb6c2466343c5000bca37be3e1` (tree
  `6ba53bd7e0603ebb2ff5458acd1345e97f14cd62`) and terra-baselines
  `b203d8a922742ec62dfd62cd9b2e24cd7b6eaa3e` (tree
  `b0a32bc07a490c009c4a43c6d33eb39a9d78e958`). Source and bank are
  read-only after their manifests passed.

F0R was submitted at `2026-07-26T05:24:28+02:00`: training job `8643810`
owns the production-shaped update-1 smoke, the 1,000-update treatment, and the
terminal receipt/checkpoint gate; fixed-seed evaluation job `8643812` has an
`afterok` dependency on it. The submission receipt SHA-256 is
`cb5ac9cad89246723fd963edbdeb1e02e8dcddb2a9a819ec2a16eace171ebb07`.
Both jobs were initially pending for priority/dependency, so this records a
valid submission rather than a passed smoke or experiment result.

Job `8643810` subsequently failed in `17 s` on `eu-g6-064` before the source
tests or PPO: Slurm and `nvidia-smi` allocated four RTX 4090s, but the
independent runtime gate saw only JAX devices `0`, `1`, and `3`. Evaluator
`8643812` was therefore cancelled by dependency. The untouched failure log is
`c0b9e513c2cd5e4da03ca4ac8e66633d27457b2276f5ba8f485accc80137de24`;
this is infrastructure evidence, not an update-1 smoke or treatment result.

The no-treatment-change retry uses immutable root
`f0r_trench_absolute_off_v1_retry1`, launch receipt
`49a8e0ae582fbfb45075cc841ca4fd4ffbe2ebe1af9e295ee8513c7a6c557e46`,
and the same source/bank manifests. Training/evaluation jobs
`8643823`/`8643824` were submitted at `2026-07-26T05:26:31+02:00`;
their submission receipt is
`8c7431a0a5c44723ac7ea48637692870406576e99a5369cf40a9d542ba9a6668`.
Both pending jobs have `eu-g6-064` in the scheduler's explicit exclusion list.

Retry job `8643823` then passed the four-device CUDA/cuDNN/NCCL preflight, all
nine source-contract tests, and the exact saved update-1 smoke on
`eu-g6-062`. `SMOKE_GATE.json` is
`172a86f1449d5f10241480cd279a68fc2f1d42352e1641348fd167bafbd35dd5`;
it reloads checkpoint
`2a3779a326a045d346e3918e2dbde297b1ac90d93a0a6a83a65402954a106a2f`,
checks 92 finite model and 185 finite optimizer leaves, verifies the exact
treatment/configuration, and reports zero mass, target, or obstacle
violations. Its update-1 population receipt is
`307556881f098f99972e52538876cc290b674951494e105ad8b370166dc310f7`.
The independent 1,000-update production initialization then started as W&B run
`nosra33p`; neither smoke completion nor a running production job is an F0R
mastery result.

At update 100, the first numbered production checkpoint
`0efaf2e7dd2e5ea185098314a6ecc18b9e02eb8dc4a93e1802b1b704027b3cfc`
independently reloaded on CPU with the exact treatment/configuration, 92 finite
model leaves, 185 finite optimizer leaves, and zero checkpoint transition
integrity counters. Updates 1-100 contain 28,674 completed online episodes,
81 successes, and no hard aggregate failures. The synchronized timeout
cohorts' mean completion rose from `0.293` at update 15 to `0.584` at update
99, while no-op/no-effect rates remained about `13-15%`/`34-35%` instead of
forming the historical late idling attractor. These are healthy intermediate
training signals only; fixed greedy mastery remains unmeasured.

The online treatment response is material by checkpoint 400. Checkpoint 300 is
`4dc6b365851a300244a9b4fbb48457ab091b76c992bdda9050ca0ebb03fbe0fb`;
updates 201-300 completed 130,602/138,269 online episodes successfully
(`94.5%`) while no-op/no-effect rates fell to `11.9%`/`26.5%`. Checkpoint 400
is `fbbb23faf80e217b3ef0f78772532ef109580720d66191f82161f93306c722f2`;
updates 301-400 reached 334,558/334,793 online successes (`99.93%`) with
`4.9%` no-ops, `10.6%` no-effect actions, and zero hard failures. Thus F0R
removes the measured online idling attractor, but only the complete
ten-checkpoint fixed evaluation can establish mastery and late retention.

F0R final adjudication: **PASS**.

- training job `8643823` completed 1,000 updates in `01:34:25`, exit `0:0`;
  fixed evaluator `8643824` completed in `00:11:19`, also `0:0`;
- terminal training receipt
  `212f2da2f8cd7c1305f9ad7c500f7ba5e4964e17bdd5cd9685bd3296e0f868fb`
  certifies ten checkpoints, all 1,000 schema-v2 aggregates, 92 finite model
  and 185 finite optimizer leaves, and exact `FINAL`/update-1,000 model and
  optimizer equality. The checkpoint and aggregate manifest hashes are
  `ba628a2542758c34933f4c8e3deba3e2599d50ad6cd93ba94127eea63196b899`
  and
  `e677589fd5bb810ea7db55dae862bcc15848731efb34f3fe9fadbe00472e30dd`;
- `FINAL` and update-1,000 SHA-256 values are
  `e236db0ee10583a1ad1b50cf2e303791c0668bfc27f98eaecc7dc87b066254f0`
  and
  `d2b0e0bb2a36c686692ce2ba6bbdc0e5db445563ea9b69952f104b50cff4be53`.
  Across 2,892,988 online episodes, the treatment records 2,820,943 task
  completions and 72,054 timeouts. All mass, target, obstacle, and per-step
  reward-reconstruction gates are zero; four episode-sum drift flags remain
  informational under the frozen C5 contract;
- final fixed-evaluation JSON
  `1a4cc49fa8d6d5a1cb53c8c0ad98e63bf144a40b38f6de9e0f1f0865987ba672`
  evaluates the exact ten-checkpoint sequence on the 32 frozen resets. The
  success curve is `0, 7, 31, 32, 0, 32, 32, 32, 32, 32`, with zero integrity
  failures at every checkpoint;
- the required consecutive gate first passes at updates 300/400 and also
  passes at 600/700, 700/800, 800/900, and the terminal 900/1,000 pair. The
  update-300 witness solves reset `2026072600` in 35 legal effective actions;
  the terminal update-1,000 witness solves it in 30; and
- update 500 is a real isolated greedy-selection collapse (`0/32`, mean
  314.25 no-effect actions) despite strong online behavior. Recovery to
  `32/32` at update 600 and perfect retention through update 1,000 means the
  preregistered gate passes, while independently reinforcing the requirement
  to select qualified checkpoints rather than assume monotonic PPO behavior.

The selected easy-trench parent therefore uses
`corrected_dense_v1_trench_absolute_off`. No extension, O0 observation change,
second reward repair, or W0a/W1 experiment is authorized at this gate. The
retained foundation witness remains update 900
`68a57f34e0e1cc3f806e8746de27a7e607d3852ec18c8aee1656b1a8fb44c721`;
together with terminal trench witness
`d2b0e0bb2a36c686692ce2ba6bbdc0e5db445563ea9b69952f104b50cff4be53`,
it releases B0 to build and validate the orthogonal feasibility panels.

Decision:

- pass: use the shaping-off contract for the easy trench cells and proceed to
  B0; do not call the historical absolute trench term part of the corrected
  parent;
- fail with an improving fixed curve: use F0's single conditional extension;
  and
- fail flat: stop again and test a bounded completion-delta/potential reward,
  not architecture, map diversity, or a broad hyperparameter sweep.

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

### F1 — Train easy family-bank specialists

Dependencies: both F0 identities pass, C5 passes, and B0c has expanded the
eight dynamically witnessed primary easy cells into immutable large banks.
Family generalization must not be judged from the current historical
eight-identity-per-cell M0 pool or from the small B0a feasibility panels.

Train two independent base `resnet_spatial_8x8` policies from scratch:

- `EASY-FOUNDATION-SPECIALIST`;
- `EASY-TRENCH-SPECIALIST`.

Hold PPO, model, reward, horizon, reset, and evaluation fixed. Use only the
named family as the treatment. Evaluate deterministically on the
source-disjoint quantitative easy-family bank every 100 updates.

Use the global continuation rule: review at 1,000, 2,000, and 5,000 updates,
continue whenever even slight fixed-bank task progress remains, and stop the
screen once the family gate passes twice. After both family screens pass, each
selected specialist receives an at-least-20,000-update continuation on
`gpuhe.120h`; those long continuations may run concurrently with G0 and are
not substitutes for G0's multitask gate.

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

These specialists are feasibility instruments. Call each a teacher candidate
only after it passes; neither is yet the final multitask teacher or deployment
policy.

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

## 10. Phase G — establish a corrected easy multitask parent

### G0 — Train foundations and trenches together on the corrected easy bank

Dependency: both F1 specialists pass.

Initialize a third base `resnet_spatial_8x8` policy from scratch, not from E8
or either specialist. Train a 50/50 easy foundation/trench mixture with the
corrected contract and the expanded B0 training bank.

Evaluate every 100 updates. Review at 1,000, 2,000, and 5,000 updates and
continue under the global slight-improvement rule. Once G0 passes twice, its
selected checkpoint starts an at-least-20,000-update `gpuhe.120h`
continuation with the treatment held fixed.

Pass gate:

- at least 26/32 foundations;
- at least 26/32 trenches;
- at least 6/8 in every admitted easy primary cell;
- two consecutive evaluations;
- zero integrity failures.

The train-development gap remains a required diagnostic but is not an
additional post-hoc mastery threshold.

If specialists pass and G0 fails, the next treatment is sampling/gradient
interference, not a larger encoder by default.

If it passes, this checkpoint becomes the new-distribution small multitask
teacher. No medium growth or reward curriculum begins before it passes.

### S0 — Grow and qualify one medium student

Dependency: G0 passes.

Use the existing function-preserving checkpoint-growth path to initialize one
medium `resnet_spatial_8x8_se` student from the qualified small multitask
teacher. Fresh SE parameters and widened/deeper parameters follow the existing
growth contract; optimizer and training-schedule semantics must be stated in
the run receipt.

Train on the exact same corrected easy 50/50 bank used by G0. E8 remains a
zero-shot historical reference and supplies no parameters or distillation
targets.

Evaluate every 100 updates and apply the global 1,000/2,000/5,000 continuation
rule plus the same family, cell, two-consecutive, and integrity gates as G0.
Once qualified, the selected S0 checkpoint receives its own
at-least-20,000-update `gpuhe.120h` continuation.

The purpose is to establish one medium parent on the new distribution, not to
compare architectures. A scratch-medium control is conditional on failed
growth/qualification evidence; it is not part of the minimal first set.

## 11. Phase B — increase procedural diversity

### B0 — Rebuild quantitative cells, then expand source-disjoint banks

Dependency: both F0 probes pass.

Do not enlarge the failed M0-M2 bank unchanged. Its all-around and large-apron
foundation cells differ by roughly an order of magnitude in relative dump
area, its procedural foundation cells change geometry and dump layout
together, and its topology/site cells are not a monotonic ladder.

#### B0a — Build small paired feasibility panels

First build small paired feasibility panels that change one axis at a time:

- OSM versus procedural foundation geometry under identical all-around dumping;
- foundation and straight-trench broad-apron dump distance centered near 2, 4,
  6, and 8 tiles under fixed geometry, volume, capacity, and site;
- straight-trench close side access as a paired broad-both-side versus
  broad-one-side treatment with identical geometry, volume, capacity, and site;
- straight, two/three end-to-end segment, T, X, and disconnected trench
  topology under easy side-cast dumping; and
- site constraints only after the corresponding geometry/dump cell passes.

The exact B0a candidate-cell names are:

```text
foundation geometry:
  f_osm_all, f_procedural_all

foundation distance:
  f_apron_d02, f_apron_d04, f_apron_d06, f_apron_d08

trench distance:
  t_straight_both_d02, t_straight_both_d04,
  t_straight_both_d06, t_straight_both_d08

trench side access:
  t_straight_both_d02, t_straight_one_d02

trench topology:
  t_straight_both_d02, t_segmented2_both_d02,
  t_segmented3_both_d02, t_T_both_d02, t_X_both_d02,
  t_disconnected_both_d02
```

The repeated anchor cells are one immutable dataset identity set referenced by
more than one panel, not independently regenerated lookalikes.

Use eight unique train and eight source-disjoint development identities per
candidate cell. Within a paired distance or side-access panel, the same source
geometry is deliberately reused across conditions and receives one explicit
`paired_source_group_id`; source geometry and generator seeds never cross the
train/development boundary. Exact target-array duplicates are forbidden.

Remote haul at 12 or more tiles remains a separate conditional feasibility
track.

Every starter and panel map uses the exact visible accepted mask, at least
`3x` reachable single-layer-equivalent capacity, and no obstacles unless site
constraint is the isolated axis. Validate capacity under C1a's contained-pile
and numeric-range rules; a large 2-D area ratio alone is not sufficient if a
valid bucket sequence can overflow the stored height type.

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
- measure the actual requested distance statistic rather than labeling a
  generator parameter as the achieved bin.

Static acceptance of B0a requires:

- all arrays and metadata pass the exact Terra loader;
- all train/development source sets are disjoint;
- the only repeated dig geometry is a declared within-split paired source;
- every map passes C1a contained-pile capacity and numeric-range validation;
- every distance cell's dig-boundary-to-accepted-dump median is within
  `0.75` tile of its declared 2/4/6/8-tile center;
- the one-side trench cell has no accepted cells on the forbidden side, while
  the both-side anchor has material accepted capacity on both sides;
- topology metadata and connected-component counts match the saved raster; and
- all galleries, manifests, generator/source hashes, rejection counts, and
  validation receipts are sealed before PPO.

#### B0b — Five bounded dynamic panel witnesses

Run five independent scratch base-small specialists, each changing only the
named map panel:

| Run | Training cells |
|---|---|
| `B0-GEO-F` | the two foundation-geometry cells |
| `B0-DIST-F` | the four foundation-distance cells |
| `B0-DIST-T` | the four trench-distance cells |
| `B0-SIDE-T` | the paired close both-side/one-side trench cells |
| `B0-TOPO-T` | the six trench-topology cells |

Use `corrected_dense_v1` for foundations and
`corrected_dense_v1_trench_absolute_off` for trenches. Hold PPO, architecture,
450-step untouched resets, and every non-map setting at F0R. Each run receives
500 updates initially and deterministic development evaluation every 100
updates. For an unpassed cell, continue the exact lineage through the
1,000/2,000/5,000 milestones whenever the global slight-improvement rule
passes. A passing panel stops because its purpose is only a dynamic witness;
long 120-hour-queue training begins after the family recipe is qualified.

A candidate cell earns a dynamic witness only when:

- at least 6/8 development identities succeed at two consecutive scheduled
  checkpoints;
- all transition, completion, termination, and mass-integrity fields are zero;
  and
- at least one successful legal action trajectory for that cell is saved.

The panel policy is a feasibility instrument, not a curriculum parent. If one
cell fails while another cell in the same panel passes, run at most one
conditional scratch single-cell specialist for the failed cell before calling
it dynamically unproven. This conditional run uses the same budget and gate.
Do not launch all cells as an unconditional hyperparameter sweep.

#### B0c — Expand only witnessed easy cells

The primary easy bank required by F1 is:

```text
foundation:
  f_osm_all, f_procedural_all, f_apron_d02, f_apron_d04

trench:
  t_straight_both_d02, t_straight_one_d02,
  t_segmented2_both_d02, t_segmented3_both_d02
```

Every one of these eight cells must pass B0b before F1. Then regenerate each
from the same frozen algorithm, but with disjoint identities:

- 64 unique training identities per cell;
- eight promotion identities per cell;
- eight development identities per cell; and
- eight sealed identities per cell.

No small-panel identity may appear in the expanded bank. B0 is complete only
after the expanded-bank loader, split-disjointness, C1a capacity, visual,
memory, and exact production-shaped update-1 compile gates all pass. The
distance-6/8, T/X, and disconnected cells remain named K0 candidates even when
they pass B0b; they are not mixed into F1.

Before fixing the bank size, measure loader memory and first-update compile
with the intended 64x64 arrays. Prefer 512 unique maps per stratum if it fits;
do not silently reduce diversity after launch.

## 12. Phase K — global quantitative map curriculum

### K0 — Implement checkpoint-bounded family/cell stages

Do not reuse the historical M0-M2 directories or labels as the active ladder.
Do not add a learned teacher or generic adaptive scheduler. Materialize one
immutable directory per declared stage and start a new recorded run at each
promotion boundary.

The admitted axes progress independently:

```text
foundation:
  all-around OSM
  -> all-around procedural
  -> broad apron at 2, 4, 6, then 8 path-distance tiles
  -> broad one-side and separated nearby zones
  -> internal holes/strips/pads
  -> one site constraint

trench:
  straight, broad both-side
  -> straight, broad one-side
  -> two then three end-to-end segments
  -> one T then one X junction
  -> N-junction and disconnected groups
  -> one site constraint
```

Each arrow is shorthand for a separately admitted quantitative cell, not an
automatic bundled stage. A geometry/topology step holds dump layout easy; a
dump-distance or side-access step holds geometry and site fixed; a site step
uses a previously mastered geometry/dump pair. Combined constraints, tight
capacity, remote haul, and physical boundary spill remain later tracks.

At each frontier, compare only if needed:

```text
qualified parent
  +-- cumulative flat control over all admitted cells
  `-- staged frontier mixture with declared earlier-cell rehearsal
```

The rehearsal fraction is a treatment, not a fact. Select one value from G0/S0
retention evidence and R0 only if R0 is actually run. Do not inherit the
unproven historical 20-30% value.

Promotion:

- evaluate a separate fixed promotion bank every 100 updates;
- require at least 26/32 successes in each included family and at least 6/8 in
  every included cell at two consecutive evaluations;
- require zero integrity failures;
- require every previously mastered cell to remain within five percentage
  points of its recorded mastery value; and
- promote only at a checkpoint/run boundary with a new run identity and
  immutable mixture receipt.

Carry the full model and optimizer state and preserve schedule position at a
promotion. Restart environment, RNG, and history identically for any matched
control and treatment.

Demotion/recovery:

- if an earlier cell fails retention twice, stop the current stage;
- restore the last checkpoint that passed all earlier gates;
- relaunch the previous mixture as a new recorded treatment; and
- never mutate exposure silently inside the compiled PPO run.

Development evaluation remains every 500 updates and does not drive
promotion. The sealed bank is opened once after model selection.

Required logging:

- stage, family/cell frontier, and mixture weights;
- unique map IDs and per-cell exposure;
- promotion-bank results;
- residence updates in the stage;
- promotion/recovery events; and
- fixed development retention.

Acceptance:

- no per-environment `3/3` promotion remains in the selected treatment;
- no promotion can occur from pooled success alone;
- gate, development, and sealed sources are disjoint;
- a failed family or cell cannot be hidden by a pooled score; and
- only one declared difficulty axis changes at a promotion.

## 13. Conditional architecture work

### A0 — Keep role-specific base and medium models until a representation failure exists

Use base `resnet_spatial_8x8` for F0-F1-G0 and medium
`resnet_spatial_8x8_se` after S0. This is a lineage plan, not an architecture
ablation. No v5, 128-resolution, transformer-core, or recurrent sweep is
authorized.

Authorize `resnet_spatial_8x8_se` versus
`resnet_spatial_8x8_se_xattn` only if:

- C0-C4 and C1a pass;
- F0, F1, G0, and S0 pass;
- K0 fails specifically on location-conditioned constrained maps;
- train-identity learning is strong;
- observation alias tests are resolved; and
- trajectory/error analysis shows incorrect selection among spatially
  separated legal work or dump regions.

The ablation must use the same corrected observation, map bank, PPO settings,
parent-policy decision, and budget. Primary metrics remain source-disjoint
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

### W0a — Keep the dense transport ablation explicit and conditional

`corrected_dense_v1` is the only authorized F0 reward. A second dense reward is
not silently folded into C1/C1a.

Candidate `transport_potential_v1`:

```text
cost(state) =
    sum(off-zone positive soil volume * distance_to_accepted_dump)
  + carried_soil_volume * agent_distance_to_accepted_dump

potential(state) = -cost(state)
shaping = gamma * potential(next_state) - potential(state)
```

If implemented, this term replaces the current dump-time relocation-potential
reward; it is not added on top. Loading transfers the same soil mass from world
cost to carried-load cost at the same location, a legal dump drives that
mass's cost to zero, and a wrong dump leaves a positive off-zone cost. The
potential never changes action validity.

Before implementation, ratify:

- traversable shortest-path versus Euclidean distance;
- how carried-load distance is defined when the base cannot reach the dump
  mask directly; and
- normalization/capping without breaking the potential telescoping property.

Trigger a dense A/B only if `corrected_dense_v1` passes F0 but shows a
transport-specific failure on the first otherwise-feasible constrained cell,
or Lorenzo explicitly authorizes it. Use the same scratch initialization,
maps, reset seeds, PPO, 100-update evaluation cadence, and global
slight-improvement continuation rule. Do not combine this with a map-stage,
architecture, terminal-reward, or partial-reset change.

### W1 — Qualify one foundation dense parent

Dependency: S0 passes and one fixed corrected foundation family passes its
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

### PR0 — Test partial resets after the map sampler is selected

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

Admit one axis at a time after the applicable K0 quantitative cells pass:

1. disconnected structural foundations and multi-junction trenches;
2. stronger single obstacles;
3. combined road/wall/object sites;
4. medium then far dump distance with matched capacity;
5. tight natural capacity only after distance is solved; and
6. physical boundary-spill dynamics; and
7. the final realistic deployment mixture.

For remote dumping, first create a paired near/far identity that changes only
traversable dump distance. A successful near map is a prerequisite. One far
identity must pass the F0 gate before generating a far family.

Unsolved or statically valid but dynamically unproven families remain in a
named challenge bank; they are not mixed into training and called curriculum
difficulty.

## 17. Immediate execution queue

This is the live top-level checklist. A box is checked only after the
acceptance evidence in the corresponding section passes.

1. [x] Freeze D0 and reject both completed historical curriculum arms.
2. [x] Ratify and implement C0-C4 plus the contained transition C1a.
3. [x] Close the C1b excavator-footprint integrity defect before training.
4. [x] Complete C5 auditable population aggregates and exact finite-checkpoint
   terminal smoke.
5. [x] Freeze the two independent F0 launch/evaluation paths, exact
   hyperparameters, checkpoint-lineage gates, and reduced-shape PPO integration
   smoke at terra-baselines `6c56525`; correct the receipt gate without changing
   the treatment at `c58ad23`.
6. [x] Finish D1/D2, inspect every JSON integrity field, and write the
   preregistered materiality/memorization/policy-mode decisions.
7. [x] Run independent update-1 finite GPU smokes for the foundation and trench
   F0 jobs, reload each exact saved checkpoint, and verify the C5 receipt.
   The corrected v2 smokes passed in retry jobs `8632268`/`8632271`; their
   independent hashes and configuration receipts are recorded in F0.
8. [x] Launch the two scratch F0 fixed-identity probes with
   `corrected_dense_v1`; evaluate 32 fixed seeds every 100 updates. The first
   attempt is preserved as failed/cancelled infrastructure evidence. Clean
   replacements from immutable root `f0_retry1` completed: foundation passed
   feasibility but failed terminal retention, while trench failed cleanly.
9. [x] If either F0 arm fails, stop its descendants and run only the
   trajectory/O0/transition/reward diagnosis implicated by that arm. The exact
   replay gate passed and selected F0R without authorizing an observation or
   architecture change.
10. [x] Implement and run F0R. The shaping-off trench treatment passes at
    updates 300/400 and retains a terminal 900/1,000 pair; together with the
    retained foundation update-900 witness, this authorizes B0 to build and
    validate the orthogonal feasibility panels and admit only dynamically
    witnessed cells.
11. [ ] Complete B0a/B0b/B0c: seal the paired static panels, obtain dynamic
    witnesses from the five bounded panel runs (plus only conditionally needed
    cell isolates), expand the eight declared primary easy cells, and pass the
    loader/memory/update-1 gates.
12. [ ] Run the two scratch F1 family specialists; require family and per-cell
    gates twice with zero integrity failures.
13. [ ] If both specialists pass, run G0; only a twice-qualified G0 becomes the
    new-distribution small multitask teacher.
14. [ ] Grow and qualify S0 from G0, then begin the checkpoint-bounded K0 map
    ladder one isolated difficulty axis at a time.
15. [ ] After S0 qualification, execute the separate W1/W2 reward experiment;
    run PR0 only after the map sampler is selected.
16. [ ] Open the sealed bank once after model/treatment selection and publish
    the final causal, integrity, compute, and checkpoint receipts.

R0 is not in the default launch queue. Authorize it only if D2 confirms
train-and-development regression and the result would change K0's rehearsal
choice. W0a is also conditional and must not delay an F0 treatment whose
reward is already frozen.

Stop after each decision gate. Do not pre-build later curriculum, reward, or
architecture machinery while an earlier result can invalidate it.

## 18. Definition of done for this backlog

The training redesign is ready for a generalist confirmation only when:

- termination, completion, reward, and evaluation share one legal task
  contract;
- reward and termination histories are globally reduced, stratified by
  outcome, and recomputable from machine-readable receipts;
- every active transition conserves world plus carried soil with no clipping,
  deletion, creation, or overflow;
- a correctly aimed dump stays inside the exact visible region and a wrong
  dump remains outside;
- relocation potential does not veto a physically valid action;
- full-reset and evaluator integrity contracts pass;
- one foundation and one trench identity are dynamically learnable;
- both quantitative easy families generalize separately from scratch;
- a scratch small multitask teacher and its grown medium student pass
  family/cell and retention gates;
- the map curriculum uses source-disjoint global promotion gates;
- procedural training diversity is sufficient and contains no silent repeated
  weighting;
- reward and partial-reset treatments remain separate;
- any architecture change is supported by a representation-specific failure;
  and
- the sealed bank remains untouched until model selection.
