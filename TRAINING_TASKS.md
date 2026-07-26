# Terra Training Tasks

- Status: active recovery execution; C0-C5 and C1b complete; D1 diagnosis
  complete with a failed historical mass-integrity gate and material completion
  mismatch; D2 diagnosis complete with memorization and action-mode evidence;
  F0 foundation feasibility passed with a terminal retention failure; F0 trench
  failed cleanly; bounded diagnosis selected F0R; F0R passed; B0a paired-panel
  generation passed; the first B0b submission was stopped before production
  after infrastructure and receipt-gate failures; all five corrected immutable
  replacement update-1 and 500-update training gates passed; deterministic
  development evaluation passed integrity and authorized a continuous
  1,000-update confirmation for every panel; all five confirmations completed,
  three 2,000-update panels are submitted while two require diagnosis, and no
  B0 cell witness exists yet
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
| Scratch budget | Treat 500/1,000/2,000/5,000 updates as review milestones, not hard ceilings. Advance whenever fixed task metrics show even slight preregistered improvement; use an exact continuation only when the checkpoint contract preserves all process state, otherwise run the full higher milestone continuously from a declared fresh start. |
| Qualified long runs | Once both F1 specialists establish the recipe, run selected F1/G0/S0/K0 treatments for at least 20,000 continuous updates on `gpuhe.120h` with a five-day wall-time request. Grant more compute while the fixed bank improves, using exact 20,000-update extensions when available or a fresh continuous run at the full higher budget. |
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
14. At a milestone, continue the exact model, optimizer, RNG, environment,
    action-history, and schedule lineage to the next
    `500 -> 1,000 -> 2,000 -> 5,000` milestone whenever slight improvement
    exists and integrity remains clean. A five-evaluation fixed-bank plateau
    with no such improvement is the stop rule. The current checkpoint-v2
    resume path restores model, optimizer, update, and schedule position but
    explicitly restarts environment, RNG, and action history; it therefore
    does **not** satisfy this rule. Until an exact checkpoint contract exists,
    an extended confirmation must be launched as one fresh continuous run at
    the full declared milestone and reported as a repeat, not as an exact
    continuation.
15. A recipe is considered qualified for long training only after its
    source-disjoint family/cell gate passes twice and no semantic, integrity,
    or causal blocker remains. Qualified continuations use Euler
    `gpuhe.120h`, request `5-00:00:00`, run for at least 20,000 updates, save
    fixed-bank checkpoints at a declared cadence, and receive more compute
    while rule 13 still shows improvement. Until exact checkpointing is
    implemented, the first selected production treatment is one continuous
    20,000-update run; a higher milestone must be a fresh continuous run at
    the full cumulative budget rather than a checkpoint-v2 pseudo-resume.
    With an exact checkpoint contract, extend in 20,000-update chunks. Long
    training never changes map, reward, reset, PPO, or architecture treatment
    in place.

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
| B0 | P1 | paired generation, five bounded panel probes, bank expansion | foundation F0, trench F0R | [ ] B0a passed; B0b active |
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

Use the global milestone rule: review at 1,000, 2,000, and 5,000 updates,
advance whenever even slight fixed-bank task progress remains, and stop the
screen once the family gate passes twice. After both family screens pass, each
selected recipe receives an at-least-20,000-update continuous production run
on `gpuhe.120h`, following rule 15's exact-checkpoint boundary. Those long
runs may execute concurrently with G0 and are not substitutes for G0's
multitask gate.

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
advance under the global slight-improvement rule. Once G0 passes twice, its
recipe starts an at-least-20,000-update continuous `gpuhe.120h` production
run under rule 15, with the treatment held fixed.

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

Evaluate every 100 updates and apply the global 1,000/2,000/5,000 milestone
rule plus the same family, cell, two-consecutive, and integrity gates as G0.
Once qualified, the selected S0 recipe receives its own at-least-20,000-update
continuous `gpuhe.120h` production run under rule 15.

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

B0a result: **PASS** at `2026-07-26T08:00+02:00`.

- implementation commit: Terra `eaf9cf00`;
- canonical inspectable bank:
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_paired_panels_20260726`;
- builder SHA-256:
  `3a1bb66798f6a4bfc1dc5b3515c5a4485eb9a28d6ffe7c9e8413a548492b79a9`;
- identity manifest SHA-256:
  `911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e`;
- source registry SHA-256:
  `1ffe22f8c3ed4cc608fc8fc9a5106f2ecd26d8e122d042d0d2630b025a293a8d`;
- validation SHA-256:
  `aeebafae74f77d19a11617f464969f162c82385aa52a65ff246f0227bd731ca5`;
- complete file-manifest SHA-256:
  `89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347`;
- 256 unique identities and target arrays cover 16 cells, eight train plus
  eight development identities per cell; all 42 per-cell/panel directories
  reloaded through the exact Terra loader;
- all 32 declared paired source groups preserve the exact dig raster, while
  train and development source sets are disjoint;
- the generator rejected one exact dihedral straight-trench duplicate before
  sealing; accepted within-cell maximum dihedral IoU ranges from `0.439` to
  `0.936`, below the declared `0.995` ceiling;
- constrained maps provide `3.25-3.26x` single-layer-equivalent capacity.
  Achieved p50 path-distance ranges are `2.00`, `3.83-4.04`,
  `5.66-6.24`, and `8.00-8.16` tiles for the declared 2/4/6/8 cells;
- every one-side target has zero forbidden-side cells and every both-side
  target reserves at least 40% of its cells on each declared side;
- the five panel galleries and all 32 cell galleries were visually inspected;
  no obstacle/site axis is present; and
- four focused generator tests plus eight contained-transition/loader tests
  pass (`12 passed`). The file manifest verifies without error.

This is static and loader evidence only. B0 remains unchecked until B0b
supplies legal dynamic trajectories and B0c expands the eight witnessed easy
cells.

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
updates. For an unpassed cell, advance through the 1,000/2,000/5,000
milestones whenever the global slight-improvement rule passes. Because
checkpoint v2 cannot preserve environment/RNG/action-history state, the first
500-to-1,000 advance is a new continuous 1,000-update scratch confirmation
with the same panel recipe and declared seed, not a resume. The 500-update
runs and their best checkpoints remain immutable evidence. A passing panel
stops because its purpose is only a dynamic witness; long 120-hour-queue
training begins after the family recipe is qualified.

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

Frozen B0b implementation receipt:

- terra-baselines revision `c42aa61` defines the five panel presets, scratch
  seeds `2026072701` through `2026072705`, exact update-1 smoke, 500-update
  training gate, deterministic development evaluator, and paired Slurm
  launcher;
- all five runs use 4 x RTX 4090, 1,024 environments per device, 32 rollout
  steps, base `resnet_spatial_8x8`, float32 encoder compute, and the full F0R
  PPO/reward treatment except for the declared panel and independent seed;
- each panel retains checkpoints every 100 updates and evaluates every
  development identity with recorded legal action traces;
- the evaluator emits exactly one of `panel_witness_passed`,
  `continue_same_panel`, `conditional_cell_isolates`, or
  `stop_and_diagnose_panel`;
- `continue_same_panel` requires a preregistered task-metric improvement within
  the last five scheduled evaluations. A reward, loss, or online-success
  change cannot authorize more compute; and
- six B0 evaluator/config tests, two training-receipt tests, seven F0
  regression tests, Python compilation, Black, `bash -n`, ShellCheck, and
  whitespace checks pass locally. The production update-1 GPU smokes remain
  cluster gates and are not claimed by these local checks.

Submitted B0b execution receipt at `2026-07-26T08:17:28+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1`;
- source revisions: terra `5f7351a8fb13a912c15887265d70359ffe99e976`
  and terra-baselines `7474c3e954386881db435c54495005184649ceb8`;
- source-manifest SHA-256
  `a060c65f0fa33bf791ae4b11974b3276dcec8b51c3c3fd0703fb4cdd895e7f97`
  and bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- the remote source manifest, remote bank manifest, and the bank-internal
  manifest all passed before submission;
- submission-receipt SHA-256
  `823c4f21800dc03aa12cd5328f09a35b97cf9e29a215f3bd300ba282ddfcd544`;

| Panel | Training job | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8647662` | `8647663` |
| foundation distance | `8647664` | `8647665` |
| trench distance | `8647666` | `8647667` |
| trench side | `8647668` | `8647669` |
| trench topology | `8647670` | `8647671` |

This receipt records submission, not a passing smoke or learning result. Each
evaluation remains `afterok`-dependent on its matching training gate.

Infrastructure replacement receipt at `2026-07-26T08:19+02:00`:

- original training jobs `8647662`, `8647666`, and `8647670` all landed on
  `eu-g6-064`, where Slurm allocated four RTX 4090s but the frozen runtime gate
  independently saw only JAX devices `[0, 1, 3]`;
- all three failed before the update-1 smoke began, wrote zero training files,
  and their dependent evaluators `8647663`, `8647667`, and `8647671` were
  cancelled as `DependencyNeverSatisfied`;
- no GPU check was weakened and none of these jobs is scientific evidence;
- identical clean replacements exclude only `eu-g6-064`:

| Panel | Replacement training | Replacement evaluation | Receipt SHA-256 |
|---|---:|---:|---|
| foundation geometry | `8647704` | `8647705` | `0edc025cfcb4f3251cc2a13925f1fb37c32cd8078c56a2d4fcc812c380194e24` |
| trench distance | `8647719` | `8647720` | `fceb4cdaa0e58be811b98ba675dcd9d87a009b4f4165a18bc2bddbaea0b793ca` |
| trench topology | `8647721` | `8647722` | `b9b7fd320812302c0fe4b5543412cc6c5760a1b91cc5f5d33103608d93a64c9d` |

Receipt-gate correction at `2026-07-26T08:28+02:00`:

- the saved scratch checkpoints correctly serialize the inert
  `load_env_from_checkpoint` field as `false`; the first B0 verifier expected
  `true`, even though `resume_from` was `null`;
- foundation-distance job `8647664` and trench-side job `8647668` each
  completed exactly one smoke update, saved model/optimizer/aggregate state,
  then failed only this receipt comparison before production;
- the three still-compiling replacements `8647704`, `8647719`, and `8647721`
  and all remaining dependent evaluators were cancelled once the deterministic
  shared failure was known, avoiding invalid or wasted production work;
- terra-baselines `c42aa61` changes only that serialized-field expectation and
  its unit fixture;
- the corrected verifier independently reloaded both saved family examples:

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Corrected receipt SHA-256 |
|---|---|---|---|
| foundation distance | `45ae0b6c4471c11dd6b80b3fb11a5f774ec17ae3af89a3f5897bc8ae9c1a49da` | `d4998464eca8a88a703f7b3703f41404c8da0fb60c39e689f4edd0cb19bfe5b8` | `0f44b9309f4e303b29c079eab32d9d74dd2cb1fe9691fbce02d4eb221dd13782` |
| trench side | `9b0e18060643ca93d2dffbb36df69798662a19d2561e6c2b68e663e6fd66594c` | `1139b4de0fb7e332dc478bccddd705b2911876e36d17c152844ea041208d3d39` | `b381a09331fb218c7c9bbbf40afd212d95f6a842e1c353d451c4116b7b86a4db` |

Both corrected receipts have 92 finite model leaves, 185 finite optimizer
leaves, zero transition-integrity violations, exact manifests, and the frozen
foundation/trench reward contracts. These are valid update-1 smoke witnesses,
not B0 dynamic-feasibility results. The submitted source root remains
immutable; production restarts only from a new root containing `c42aa61`.

Corrected B0b submission at `2026-07-26T08:31:24+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1_retry1`;
- source revisions: terra `10cf8f03fb59a94209aa1225148d461d30817fa8`
  and terra-baselines `c42aa612af747156e0ff027a90f8c3db825d8e4a`;
- source-manifest SHA-256
  `f4b0bd57ae9c08955b56357631956da925ef10044a74d353caff40c7242efe58`
  and unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- all source, bank, and bank-internal manifests passed remotely before launch;
- submission receipt SHA-256
  `95ab8f3c28086447716a5e80a4d4c92c2e821deff5d958330ba24faebe18c01b`;
- every training job excludes only the diagnosed node `eu-g6-064`;

| Panel | Training job | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8648071` | `8648073` |
| foundation distance | `8648076` | `8648078` |
| trench distance | `8648081` | `8648084` |
| trench side | `8648088` | `8648090` |
| trench topology | `8648092` | `8648094` |

This is again a submission receipt only. No B0b result is accepted until the
new-root saved checkpoint, aggregate, training gate, and deterministic
development evaluation pass.

Corrected update-1 result: **PASS for all five panels** at
`2026-07-26T08:42+02:00`.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-receipt SHA-256 |
|---|---|---|---|
| foundation geometry | `8260b717e9239f8074b9f6fd1455506720a3e9d46f95a789f6f7b3bd69247fb5` | `72d8241ddae904f729f27c0126c68d39f687efe9a8a3f1e291ff2f26f30cd451` | `7e6c9a9d7ece50d77adc9f9f72d5ec04fc9986a33ac39ef518d4b9dbe4536b78` |
| foundation distance | `baa60b77639b34abd8e4d84a0457565bbc77dbf2bbbbe2286359ad4a6948f6ad` | `3b39cc7fb9d77e0ad29ec1c4f563b4c3f8113cbd9d83c78e25af784158846072` | `c256533dc549cfc44c65703111687130806b0cedc76eb05ac13bfeac68b3fbda` |
| trench distance | `f7a6a4ae3463c41a27a05b115ce0c0b67ea477e62ec3475f8adb210baec89c5d` | `37cdca6b460c307bf7aee05c7ed9e023463628f02dd462b55612ff012c8032c3` | `f9941ec6641cb79f92c93517253627232051e0e4aea19374f7e2756230862746` |
| trench side | `09573db08efe194fe227d7cbfc33bbaf3668e16511c47b005fda22f9ebf384ee` | `ea0fe566961f50edc0efb57efb65d1aa99971bf4103964e2d392080943cd6734` | `6f3d1ac117feb1df7e1899568b0a1581ba745c342b209f16c83f1209faef2d92` |
| trench topology | `86dae52e7af64321b182672e8ab29aa19b62d10545499fde3aee105ce31f3bcd` | `6746964f2f2e0d593f8c0248fbde90ed776e3f439b5b5734b37f7bf1660d9da0` | `9155c6c7f58ead49cd2206d30908cae1969407e0b34b6f5ce364e403eceffe2f` |

Every receipt records 92 finite model leaves, 185 finite optimizer leaves,
zero transition-integrity violations, the exact panel manifest, and its frozen
foundation or shaping-off trench reward contract. This passes the launch gate
only; it does not establish a dynamic cell witness.

Bounded 500-update training result: **PASS for all five panels** at
`2026-07-26T09:27+02:00`.

| Panel | W&B | Training-gate SHA-256 | FINAL SHA-256 | Online done / episodes |
|---|---|---|---|---:|
| foundation geometry | `q24yinzc` | `d02a6603e49e6a5c910483741033da7676d9a9b21fd0fe5982f632cda4b5de80` | `d2b259bacc8ecee46526f6b101b0f36f3ac64b51637f0f99495b79c0d2887e98` | 1,632 / 144,167 |
| foundation distance | `rryzgmnh` | `8db63c0643b8f9c2ab7f38bb2d868e75099513eeb9c37a6a11294cb7263fe878` | `8b488c292178504dec7badc37d287e64f48819a1db5451a8b217b16861965fc2` | 7,734 / 146,305 |
| trench distance | `ghs31rt3` | `a8fb706747a400f381c1cd301c7e50dc756510613420c6ec714a8f548dde938c` | `ed7387409192fbd64d7d121528cf8706529959b5c400c414f3d6f8d4afd992de` | 6,314 / 147,559 |
| trench side | `314071nj` | `2053fd48db240d53759c159281ca8ebefc49c47adcc88b618f6b563e87b764c2` | `215d2e9feb1efcf58d0ff077d73ac026cdaf9893dbc059ddf9358d334cf8740b` | 9,281 / 149,616 |
| trench topology | `jwvr6vvy` | `bcc7bff23b94e52a91fd956597867b988d077d137e94326f0ca7ac695d42c1ac` | `cba2fcbca23e2e97b08bfdcb6cfbbec8a4261207089b3ef83422efe045ce7fcc` | 2,408 / 144,577 |

Every training gate has exactly 500 aggregate receipts, numbered checkpoints
at 100/200/300/400/500, one FINAL checkpoint exactly equal to update 500,
maximum mass residual zero, and zero mass/target/obstacle/reward-residual
violation counts. The maximum sub-threshold floating reward residual was
`4.76837158203125e-07`. Online completion rose during every panel but remains
diagnostic; only the running source-disjoint development evaluations adjudicate
B0b or authorize continuation.

Bounded 500-update development result: **all five panels authorize more
bounded compute; none has a dynamic cell witness yet**.

| Panel | Evaluation job | Evaluation SHA-256 | Successful identities observed | Best scheduled median-completion evidence | Decision |
|---|---:|---|---|---|---|
| foundation geometry | `8648073` | `1c101c44d553dc4fbac48d7576ff27d61616948ae0d485804cc839c595d74783` | none | OSM `0.780 @400`; procedural `0.417 @300` | continue |
| foundation distance | `8648078` | `fd4ac9eef8769322ad8052c026061f2426a1aeaf8bf0f951889fd59130cdca46` | none | d02 `0.463 @300`; d04 `0.464 @400`; d06 `0.363 @400`; d08 `0.274 @500` | continue |
| trench distance | `8648084` | `e6308b802ce58cc62e9fa5cc70282cf97721eca01d1b106d968b8f3e7b59631e` | one d02 identity at update 400, with trajectory | d02 `0.448 @200`; d04 `0.428 @200`; d06 `0.452 @200`; d08 `0.567 @300` | continue |
| trench side | `8648090` | `a29f213839a213cf309342792c80228503c6d78bc5ca0af89ccd0678d4623f08` | one one-side identity at update 500, with trajectory | both-side `0.495 @200`; one-side `0.482 @300` | continue |
| trench topology | `8651299` | `9e3154582834080b617e8d44b9e0469e15d2ecd188f9fdb46fbea76840247a3a` | one segmented-2 and one segmented-3 identity at update 500, both with trajectories | straight `0.784 @400`; T `0.561 @400`; X `0.562 @400`; disconnected `0.644 @300`; segmented-2 `0.754 @400`; segmented-3 `0.624 @500` | continue |

All 25 scheduled fixed-bank evaluations have zero integrity failures and exact
reset-manifest verification. No cell reaches 6/8 at two consecutive
checkpoints, so no row is promotable and B0 remains unchecked. Several
terminal checkpoints regress from an earlier best; preserve all checkpoint
histories and select by fixed-bank evidence rather than assuming FINAL is
best.

Topology evaluation recovery:

- original evaluator job `8648094` failed before rollout because 48 maps were
  passed to an inherited 32-minibatch topology, a shape-only evaluator defect;
- terra-baselines `d7867a7` selects `gcd(48, 32) = 16` evaluation
  minibatches, and `d415af2` allows a pinned evaluator source without
  modifying the immutable training root;
- retry source root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_eval_retry1`;
- retry source-manifest SHA-256
  `9daffd0188a7dd5321996bdd5ea1cd4c23447b4a8453bbec75c88fb058cc3a45`
  and submission-receipt SHA-256
  `9ac2068d8f9690999d505b9fefaa18e38c80c8147ba8e4880f59cca9e243754a`;
- retry job `8651299` completed in `00:12:28` on `eu-g6-005`, using the
  unchanged five training checkpoints and bank. Its update-100 through
  update-500 success totals are `0, 0, 0, 0, 2`.

The next bounded treatment is therefore one fresh, continuous 1,000-update
scratch run for each of the five unchanged panel recipes. This deliberately
duplicates the first 500 updates: it preserves continuous process state across
the 1,000-update milestone and avoids falsely describing checkpoint-v2 resume
as exact. terra-baselines `fc93c29` parameterizes the sealed launcher,
training gate, evaluator, checkpoint count, receipt, panel subset, and
diagnosed-node exclusion for the 1,000-update treatment. `bash -n`,
ShellCheck, and whitespace checks pass. Each new run must still pass its own
remote source/bank manifest checks, update-1 GPU smoke, 1,000 aggregate
receipts, ten checkpoint gate, and deterministic development evaluation
before it contributes evidence.

Continuous 1,000-update submission receipt at
`2026-07-26T09:49:32+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_u1000_v1`;
- source revisions: Terra
  `917fbf35a0ffa468148317a42812c7e97b0f5bc1` and terra-baselines
  `fc93c294861781e19e135c82fb645465b4d8e917`;
- source-manifest SHA-256
  `562e3b883282314cc7a6441cd5d7ce614d839f04a9a74379f84f1c46ca9ba955`;
- unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- creation-receipt SHA-256
  `e5e71d8d87c2a618ac03b1b60cc0b6e47c2410744de287e4011ec2b6c2d069df`;
- submission-receipt SHA-256
  `81a7de0bb051165b9fdb52b67d563807ab06f179faa4da49c2961fe21f990b55`;
- the source manifest, bank manifest, bank-internal manifest, exact revisions,
  update target, and executable launch scripts passed before submission;
- every job excludes only the diagnosed node `eu-g6-064`.

| Panel | Continuous training | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8651897` | `8651898` |
| foundation distance | `8651899` | `8651900` |
| trench distance | `8651901` | `8651902` |
| trench side | `8651903` | `8651904` |
| trench topology | `8651905` | `8651906` |

This is a compute receipt, not a learning claim. Each dependent evaluator is
`afterok`-gated on its matching update-1 smoke and complete 1,000-update
training receipt.

Continuous-run update-1 result: **PASS for all five panels**.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-gate SHA-256 |
|---|---|---|---|
| foundation geometry | `22a2552778e259af7102959004da83c64911b0df89a6b0a3f347ec3607452dc7` | `517ea36508f89a3d87ae9f8e7353bd1e82bd774d33bead7ab84f452068db3440` | `b4dbc6aabe687e317ac337f5342c8f9a2a0bf9c4b69a10e9a831c6caa03c7ba3` |
| foundation distance | `6bac8d7df0240066070ed12bd1ccc0150061257be140d89c3f00785c6b7b8790` | `22191963a4c7c232a848113ca2a57846f7dffbdd48809d242683148fb7e31bcb` | `50e986d8b954709c8e806da1aeb5229878bab89fb6595e410b87a89f0b6cc2f7` |
| trench distance | `9f50256934d30187a0080b8b86ed9113cf880651bdf488ea492ef56e001531ec` | `dec158b290aef6064b1ab542d86d346c3921f96802c7764bac1833ac4501ba42` | `223cf7ed8e158906f405605713a0100ea5fcbb2e0213039ffda3ef37c1fd4a23` |
| trench side | `689b24819f5b8d8959b5d1e39c40aa2afe1029958f648e271a96b159034332c5` | `d88c15e35c21acc540c8ca208b89265d12fce42c39f8be9d63bf8ddcf0a9b938` | `f48287292e749335f72839a2ff228097ab157eebee0c46e10570e58c04bcd999` |
| trench topology | `889d7b7dbbb5ff786336b12608c72ef2e306128544dcfc5f3e4482bec11f4c52` | `f5b37754ba65fb2408df92e700644f9d150ccbf40358554e783061b8c355c927` | `d12efa398fc8edbe730e7202c4dca6daeda08e3764a66354e94520049f8e62e1` |

Every smoke gate independently reloads 92 finite model leaves and 185 finite
optimizer leaves, verifies the exact panel manifest and reward/completion
contract, and records zero mass, target, or obstacle integrity violations.
All five jobs passed the pinned four-GPU CUDA, cuDNN, NCCL, seven evaluator
tests, and two training-receipt tests before the smoke. This authorizes their
continuous production bodies; it is not yet task-learning evidence.

A proposed bitwise cross-process prefix gate in terra-baselines `d17d0be` was
tested and rejected, then removed by `7879be1`. All five update-1 aggregate
payloads match their original 500-update counterpart exactly apart from
`run_name`, but the first aggregate differences appear at update 15 and all 92
model leaves differ by update 100 in the first three runs. The maximum
per-leaf absolute difference is `0.585-0.869`. The only configuration
differences are run/path labels and the declared 500-versus-1,000 horizon;
learning rate, entropy schedule, PPO, seed, maps, reward, and model are
unchanged. Therefore cross-process bitwise equality is not a valid gate for
this GPU training path. Treatment/config receipts and deterministic
source-disjoint task curves remain authoritative; the continuous 1,000-update
runs are repeats of the same recipe, not exact replays of the first process.

Every update-1 hard integrity count is zero. The new W&B run IDs are:

| Panel | W&B run |
|---|---|
| foundation geometry | `a4vgn9wc` |
| foundation distance | `edjt0yxt` |
| trench distance | `ytprpw04` |
| trench side | `sxqzfr3t` |
| trench topology | `cqp6v20x` |

Interim continuous-boundary audit: **PASS at update 500 for all five runs**.
Each process crossed update 500 without restart and has exactly 500 aggregate
receipts plus checkpoints 100/200/300/400/500. Across all 2,500 receipts,
mass-residual, target-mutation, obstacle-mutation, and step-reward-residual
hard violation counts are zero; maximum mass residual is zero and the largest
sub-threshold floating reward residual is `4.76837158203125e-07`. This is an
integrity milestone only. The fixed-bank ten-checkpoint result remains the
learning gate.

Continuous 1,000-update training result: **PASS for all five panels**.

| Panel | Job / elapsed | Training-gate SHA-256 | FINAL SHA-256 | Online done / episodes |
|---|---|---|---|---:|
| foundation geometry | `8651897` / `01:32:19` | `d3f82192ae1ef8e2833d5b2a2887ce49faf8d5e6e2bc28faccc7be77a212d664` | `be0e6b524cbd51a7af05b5f74d73e80f96c277cc0fe6a8be1d9bc7d79bc7577b` | 54,222 / 331,247 |
| foundation distance | `8651899` / `01:33:57` | `08abaf5f1fe17c1ce597df7e7a9fd7a2e8f1a0e7047411f3c570fe05f87443d9` | `44a4014ed3399350c39d666c412697325dcbcbae270592cfee46e1fbe3da3723` | 320,758 / 537,739 |
| trench distance | `8651901` / `01:33:23` | `c234110f27df5bf363f3ad467545a10a159b96097a9237a178137995521a60d7` | `41c73cc28bf485b1d57565920660359609ac04ec5420ac486e792d4f7247bd3f` | 640,854 / 828,426 |
| trench side | `8651903` / `01:32:14` | `32b9faa94fdae1d3139234543d6a5756996f0309abca7221090462bbc2060e11` | `8c85be08ec71376ab41443b6a8c0967ed2704150cd17e9be0f24fc2a36ed4312` | 804,548 / 992,903 |
| trench topology | `8651905` / `01:32:06` | `354abb6e1e0896043f250c4562856448b7dfa3ec6cf21bc96f979a27225b1d32` | `b44d8b3894325e4f87314b3e042e5a150570d1b37414452304fc759721503329` | 81,282 / 348,385 |

Every gate certifies exactly 1,000 ordered aggregate receipts, ten numbered
checkpoints at updates 100 through 1,000, one FINAL exactly equal to update
1,000 across 92 model and 185 optimizer leaves, zero hard integrity counts,
maximum mass residual zero, and maximum sub-threshold floating reward residual
`4.76837158203125e-07`. All jobs exited `0:0`. The online counts above are
diagnostic only and do not rank or promote panels.

Continuous 1,000-update development result: **three panels continue; two stop
for diagnosis; no cell has a dynamic witness**.

| Panel | Eval job / elapsed | Evaluation SHA-256 | Best source-disjoint evidence | Decision |
|---|---|---|---|---|
| foundation geometry | `8651898` / `00:18:52` | `22e936a3e44e1d96ef8a01f1b3095fbcfb507c0c2b0028b1f8909996ab5dff0d` | one OSM success at 500; OSM median `0.878 @500`; procedural `0.852 @700`; both collapse by 800–1,000 | continue to 2,000 |
| foundation distance | `8651900` / `00:18:55` | `1ad753e03ee71fac00cac6c7d726c80d3f911ce8a396bd6b11eddae88ee1667a` | one d06 success at 300, d02 at 600, d08 at 700; all cells zero at 900/1,000 | continue to 2,000 |
| trench distance | `8651902` / `00:18:57` | `87b0a3b5efdd8cb88ec98598fc66f16937e12f9bfb1228057c8a1edb753868b1` | no successes; best medians d02 `0.582 @200`, d04 `0.624 @300`, d06 `0.573 @300`, d08 `0.213 @400`; all cells zero from 600 | stop and diagnose |
| trench side | `8651904` / `00:18:28` | `247595d095f353de97bf6a9db48bbdce2e50ae9c69d06cd323c7faafb4bd059a` | no successes; both-side `0.428 @200`, one-side `0.305 @200`; both zero from 500 | stop and diagnose |
| trench topology | `8651906` / `00:19:09` | `b8d0d14840a59024551b8c40f0db7af82ba6f27cb0d117c83254c67c1f0fed0d` | one segmented-2 success at 400; three total successes at 900 and four at 1,000, including two straight and two segmented-2 with trajectories | continue to 2,000 |

All 50 scheduled evaluations have zero integrity failures and exact reset
verification. Geometry and foundation-distance satisfy the frozen
five-evaluation rule because their last success/improvement at 700 is only
three evaluations old, despite terminal collapse. Topology has direct success
growth at 900/1,000. Trench distance and side have at least five consecutive
flat evaluations after their last improvement and are stopped. None reaches
6/8 in any cell at two consecutive checkpoints, so B0 remains unchecked and
no 120-hour recipe is qualified.

The next compute decision is therefore asymmetric:

- launch fresh continuous 2,000-update repeats only for foundation geometry,
  foundation distance, and trench topology, preserving all earlier best
  checkpoints;
- do not spend more unchanged PPO compute on trench distance or trench side;
  diagnose their mid-run-to-zero collapse before selecting a repair; and
- keep any entropy/reward/map repair separate from the unchanged 2,000-update
  repeats.

terra-baselines `008b5bd` extends the sealed launcher and evaluator to the
2,000-update/20-checkpoint milestone. `bash -n`, ShellCheck, and whitespace
checks pass.

Continuous 2,000-update submission receipt at
`2026-07-26T11:48:51+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_u2000_v1`;
- source revisions: Terra
  `6702cdfa4926b37e34f62501a21dff7f3460b905` and terra-baselines
  `008b5bdd11437777a44821dfe5886b5e9ac2d6ab`;
- source-manifest SHA-256
  `96852d44712c59ffe9ea26c43c38377239043183ccd6428a5c4926e47473bc20`;
- unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- creation-receipt SHA-256
  `f3419501d679ad39121a0844eef2712919eb4653c34221ca845932b92a7a5927`;
- submission-receipt SHA-256
  `11f8b2928c5465f3fdf65500c8b55b56f669fe652c224310361a7091c822cd2f`;
- source, copied bank, bank-internal manifest, exact revisions, declared panel
  subset, and 2,000-update target all passed before submission;
- every job excludes only `eu-g6-064`.

| Panel | Continuous training | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8656160` | `8656161` |
| foundation distance | `8656162` | `8656165` |
| trench topology | `8656166` | `8656167` |

Trench distance and trench side are absent by design. This is a compute
receipt only; all three new runs require fresh update-1, 2,000-receipt,
20-checkpoint, and fixed-bank gates.

Continuous 2,000-update update-1 result: **PASS for all three submitted
panels**.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-gate SHA-256 |
|---|---|---|---|
| foundation geometry | `a532a2f59fd8c90c72296167fda3c3c798664e579676e75b32230d66844eb658` | `041bc3ebb75c21b921eeaae1de44d9bb1d7973873e7adc3a97b4113800449521` | `4e1cd162962c0ff00061e065ea9da505c9a928b78a6b4002b7d915b792f62f54` |
| foundation distance | `a0dfd18de4dcd8fc41baf064caa4fbd9b8f61a036b8171b54b3627295c1bcfa4` | `7e0c97d7809879bf88fa4228726173a4f24871d71eea8860393919476777ddbf` | `9ffde7d5ec61a3d872280da7fa7713e4abcfff156ea66d806cf338797952e1e6` |
| trench topology | `e4fdb0304b8d951ec6183d937705016c2e7dece4d63f1149fe78016f4beb20b5` | `efa3c50d99ccb7ea6c4c3b60cb340328df8e0c7bbcbee74fd95b6798c0bb1c1c` | `ed84a5bfacfc40a8110d38b01b5352f034bbca10579460b3174096e3d268ee35` |

Each gate reloads 92 finite model leaves and 185 finite optimizer leaves,
matches the exact train-panel manifest and declared reward/completion contract,
and records zero mass-residual, target-mutation, or obstacle-mutation
violations. All three production bodies started after this gate. This is an
integrity authorization only; the 20-checkpoint source-disjoint task curves
remain the learning decision.

The corresponding continuous-run W&B IDs are foundation geometry
`6fbwzje9`, foundation distance `ncy5e6yo`, and trench topology `mma0bakl`.
They are operational pointers only; W&B online aggregates do not determine
continuation or promotion.

Continuous 2,000-update training result: **PASS for all three submitted
panels**.

| Panel | Job / elapsed | Training-gate SHA-256 | FINAL SHA-256 |
|---|---:|---|---|
| foundation geometry | `8656160` / `02:56:39` | `397510e7ac9242e78bfb0daab841e822a50dae497f8b278c0994b99b80d0334b` | `4e258f041186e6961dd2f8118d7850b735a68c8ad822bde98c3d056055660a74` |
| foundation distance | `8656162` / `03:26:15` | `6afa85ae40b728d4528fac547608b8b87573b6a12ac499cf0d31a149e2f618e7` | `d11122ebfa6c7dcd856aafaf98c4a10cb16e2037e9f462bdce151cd945a8cb4e` |
| trench topology | `8656166` / `02:58:07` | `949e6d8db5e91d40c41c7784c9f0eff6237f09b883e8e5f5304101af25ec26d8` | `49d1a5d2141dcefbc38193a2223076d36ea7337467ecb9b24b4a6d2e4c71531b` |

Each gate has exactly 2,000 ordered aggregate receipts and 20 numbered
checkpoints, and FINAL exactly matches update 2,000 across 92 model and 185
optimizer leaves. The hard mass, target, obstacle, and per-transition reward
counts are zero; maximum mass residual is zero and maximum sub-threshold
step-reward residual is `4.76837158203125e-07`. This is terminal training
integrity, not a learning pass.

The first two completed fixed-development decisions are deliberately
asymmetric:

| Panel | Eval job / elapsed | Evaluation SHA-256 | Source-disjoint result | Decision |
|---|---:|---|---|---|
| foundation geometry | `8656161` / `00:32:37` | `efa70d65b2acbec058502739d0d19e2aecbbb268801051632d01a4040b760d12` | zero successes at all 20 checkpoints; procedural median nevertheless rises from its prior best `0.6632` to `0.7569 @1600` | continue once to 5,000 |
| trench topology | `8656167` / `00:32:49` | `776dbdc9b903226d57e777b669b777735a4091092037dcd3da9b2e875c844bb5` | best aggregate is 5/48 at update 1,100; final-window totals are 1, 2, 0, 3, 0 and no cell exceeds 2/8 | stop and diagnose |

Both evaluators verify exact resets and zero rollout-integrity failures.
Neither panel has a 6/8 two-consecutive-checkpoint cell witness, so B0 remains
unchecked. Geometry receives the single fresh continuous 5,000-update repeat
authorized by the frozen median-completion rule; it is not promoted and does
not qualify for `gpuhe.120h`. Topology receives no more unchanged PPO compute.
Foundation-distance evaluation job `8656165` remains active and will be
adjudicated independently before its next allocation.

The geometry-only 5,000-update repeat was sealed and submitted at
`2026-07-26T15:27:11+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_geometry_u5000_v1`;
- source revisions are Terra
  `146919ffd78242f1bcf6d17091a47a20fd22b2bc` and terra-baselines
  `c418bb8bc8b57102a3d6982d85c7a3c9bd6bd85a`; the latter changes only the
  bounded launcher to admit the preregistered 5,000-update milestone and a
  16-hour train request;
- the exact prior B0a bank was copied without regeneration. Source-manifest,
  bank-manifest, creation-receipt, and submission-receipt SHA-256 values are
  `efcffedab1beb19220c96ff626548331b272b44362a9bc42e2dcdf8972ecf9a3`,
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`,
  `256a49d6f86c2952eed172a4bc18d6b7ada1ea3216fe93a17c285218731b789c`,
  and
  `0fceddc7941d308facb594bfa16a640ddd04c31681b065a204dfdf6eb260417d`;
- all 407 source and 3,393 bank files, revisions, scripts, syntax, 16 train
  identities, and 16 development identities passed before submission;
- train job `8667019` requests four RTX 4090 GPUs for `16:00:00` on
  `gpuhe.24h`, excludes only `eu-g6-064`, and runs continuously from scratch;
  and
- evaluation job `8667022` is held by `afterok:8667019` and requires all 50
  scheduled checkpoints.

This remains a compute receipt. Its own update-1 smoke, complete 5,000-update
integrity gate, and deterministic development curve are mandatory; it cannot
inherit evidence from the 2,000-update process.

The geometry-only repeat independently passed its update-1 smoke:

- checkpoint, aggregate, exact 16-map train manifest, and smoke-gate SHA-256
  values are
  `620e4586f720351652cb2430ec169408fd99f4dcd7239d6edb4d6d19425321a2`,
  `8f5b2fbd3afa2f3e09e88fc2d2689d85d255b9a41dc9d9708a007de5aaa65917`,
  `dd30a0e5d66ac6b66da2b4a1172d74b1a9d7a420829157c02e75a0d5beb14463`,
  and
  `ec61cebf89f33a44fea4b9e84611121aaaf3ae3e38406685cd1b5ba0d8c98199`;
- all 92 model and 185 optimizer leaves are finite and the declared seed,
  reward, architecture, entropy, PPO, reset, horizon, and 5,000-update target
  match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes the continuous 5,000-update body. It remains an integrity
result only.

The trusted C5 population receipts materially narrow the stopped-trench
interpretation:

| Panel | Sampled online train success at 900 | Sampled online train success at 1,000 | Deterministic source-disjoint development |
|---|---:|---:|---:|
| trench distance | 2,218/2,227 | 599/808 | 0 successes from 600 through 1,000 |
| trench side | 2,528/2,554 | 1,519/1,630 | 0 successes from 500 through 1,000 |

The update-900/update-1,000 aggregate SHA-256 pairs are respectively
`7bc262988bec5aea7c29a7229277c398182ec048a12d6132fab9bf41e2a9d12f` /
`7a12ada87ae206666fe300ebdb3499c4e9308da1566a8e3ba8ce26e77b5d8ddd`
and
`bf4fbb3c6a1b14a3435d5e948c66fb119b1f51ae0ad4a28e337645202839e9c1` /
`84405d0a1556420272a51f7df39abff9c0fb9a34a70176ddee65a5ed3d05043f`.
Therefore the development result is not evidence that the sampled policy
globally stopped moving dirt. It leaves two crossed explanations: train versus
source-disjoint identity generalization and sampled versus deterministic
action selection. Both must be measured on the same fixed checkpoint and
reset bank before selecting an entropy, reward, or map repair.

The stopped trench panels have a separate, bounded action-trace diagnosis
queued at `2026-07-26T11:56:10+02:00`. This is evaluator-only compute: it
does not add PPO updates or alter the fixed bank.

- immutable evaluator root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_collapse_diag_v1`;
- source revisions: Terra
  `9ca7bceebde5d84700290faa3a805183bb281b24` and terra-baselines
  `8a3f510e1422e8feaf73965e55886d2c1f9b0fd7`;
- source-manifest SHA-256
  `66774e156e9e640465fdb8196e15e8ec4268e3ce1583b6cf065a09373aadc8f9`;
- creation-receipt SHA-256
  `c88e2d3c023a76a93025fab499d08258198d808fd236194af33ae3dfeada76c3`;
- submission-receipt SHA-256
  `b57f4eead653b246be8a44351eafc92070d083aa25ff4ed8714a6f977de04885`;
- initial jobs `8656488` and `8656489` were cancelled at `00:05:55` after
  prematurely interpreting the compile-averaged first-rollout throughput as
  steady-state throughput. The reference evaluator shows the first replay
  rising from `0.1` to `1.3` steps/s as JIT compilation amortizes, followed by
  approximately `5.5` steps/s;
- source and treatment remain unchanged. Replacement receipt at
  `2026-07-26T12:03:12+02:00` has SHA-256
  `bfab28ce0125fd74d9ae3d705c37b7423c7e65de704e880a392223ca0619a6d1`;
- replacement trench-distance job `8656748` replays updates 200, 300, 600,
  and 1,000 over all 32 panel maps;
- replacement trench-side job `8656750` replays updates 200, 400, 600, and
  1,000 over all 16 panel maps; and
- each replay records the eight action-mode counts, effective-action counts,
  first/effective dig step, switches, run-length traces, terminal dig/dump
  mass, completion, and exact-reset verification. Both jobs exclude
  `eu-g6-064`.

Both replacements passed: distance completed in `00:11:01`, side in
`00:11:45`, with exit code `0:0`, exact-reset verification, and zero rollout
integrity failures. Their output SHA-256 values are respectively
`f86fbdc3a64a22a241d71f4c41e10bc53215614cff7524317b8d625edd107dc2`
and
`3bf209cef67fe6db659890cd853153f3b4dae04addfe5a9aa81ee7aa277925f7`.

| Panel / update | Maps issuing `DO` | `DO` / effective `DO` actions | Median action switches out of 449 | Deterministic development result |
|---|---:|---:|---:|---|
| distance / 200 | 30/32 | 993 / 177 | 438 | cell medians 0.157-0.582 |
| distance / 300 | 26/32 | 488 / 148 | 284 | cell medians 0.195-0.624 |
| distance / 600 | 4/32 | 163 / 26 | 444 | every cell median 0 |
| distance / 1,000 | 4/32 | 14 / 14 | 444 | every cell median 0 |
| side / 200 | 14/16 | 322 / 53 | 223 | both-side 0.428; one-side 0.305 |
| side / 400 | 9/16 | 318 / 36 | 442 | both-side 0.421; one-side 0 |
| side / 600 | 1/16 | 2 / 2 | 444 | both cells median 0 |
| side / 1,000 | 2/16 | 217 / 8 | 442 | both cells median 0 |

The deterministic development failure is thus an action-mode change, not a
global physics veto: all four late distance maps that issue `DO` have at least
one effective `DO`, and all 14 update-1,000 distance `DO` actions are
effective. Side also retains at least one effective `DO` path. The greedy
policy instead spends nearly every transition alternating forward, backward,
and rotation actions; the high switch count rules out a single static no-op
but identifies movement chattering. Side update 1,000 repeats many ineffective
`DO` actions on only two maps, so it has both chattering and local repeated-dig
behavior. These traces explain how the visible zero is produced; they do not
yet distinguish whether sampled action selection recovers useful behavior or
whether that behavior generalizes.

The output is diagnostic evidence for selecting a single controlled repair,
not a promotion gate by itself. After the action-trace replay, run the minimal
fixed-checkpoint train/development by deterministic/sampled cross above.
In particular, an entropy, reward, or map change must not be inferred solely
from the already observed deterministic development zero.

That crossed evaluator was implemented and queued at
`2026-07-26T12:10:52+02:00`:

- immutable evaluator root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_policy_cross_v1`;
- source revisions: Terra
  `caad73962b5ab8a916a6919ec2926f074010c4ae` and terra-baselines
  `34fcbb0d7352d1ef03019b27e3a912f8bde98662`;
- source-manifest SHA-256
  `c12b73adf6000effa70f6d3e6148fd72c2e1faaac5c881f6ecab06c310eeb44e`;
- creation-receipt SHA-256
  `cefaf3382ad5ff87e0141f9de67e8bbce71e35730743a6dac4dcae5f8696fb1c`;
- submission-receipt SHA-256
  `acefbb2a2c185948d97d449af773769bffce8f0047bc3bef199998285bf4f1ad`;
- trench-distance job `8657124` and trench-side job `8657126` each evaluate
  only their immutable update-900 checkpoint;
- each job crosses exact train and source-disjoint development identities with
  one deterministic replay and sampled seeds 2026072801-2026072804; and
- the two pure aggregation tests, byte compilation, Black, bash syntax,
  ShellCheck, source receipt, exact reset, and zero-integrity gates are
  mandatory. Both jobs exclude `eu-g6-064`.

This is an evaluator-only causal diagnosis. It does not make update 900 a
promotable checkpoint and does not authorize a training repair until its cross
is adjudicated together with the action traces.

Both crossed evaluators passed their exact-reset and zero-integrity gates.
Distance job `8657124` completed in `00:22:51`; side job `8657126` completed
in `00:18:55`; both exited `0:0`. Their output SHA-256 values are
`3a41bc64fc79cc73bb3077f34c0691e0be064583ec21745bbd026a27515d86b6`
and
`4560d16b8342c310a284a94f0aa3815b92e42ac51ecbb4db7aaf08ecfb9ebf56`.

| Panel | Train deterministic | Train sampled, four seeds | Development deterministic | Development sampled, four seeds |
|---|---:|---:|---:|---:|
| trench distance | 28/32 | 111/128 | 0/32 | 15/128 |
| trench side | 15/16 | 61/64 | 0/16 | 0/64 |

The distance sampled-development successes decrease with dump distance:
6/32 at d02, 5/32 at d04, 3/32 at d06, and 1/32 at d08. In contrast, each
distance train cell has 7/8 deterministic successes, and sampled train success
is 27-28/32 per cell. Side has 7/8 deterministic train successes for
both-side, 8/8 for one-side, sampled train rates 29/32 and 32/32, and zero
sampled development successes in both cells.

This adjudicates the main failure as source-identity memorization:

- deterministic train-minus-development gaps are 87.5 percentage points for
  distance and 93.75 points for side;
- sampled train-minus-development gaps remain 75.0 and 95.3125 points;
- side has no sampled-versus-deterministic development advantage, so an
  entropy/action-selection repair is not authorized;
- distance sampling recovers only 11.71875% development success and degrades
  monotonically with distance. This is useful policy-mode evidence, but not a
  reason to move far dumping into the starter curriculum.

The single next trench training treatment is therefore
`B0-DIVERSITY-T-SIDE`, not another unchanged run and not an entropy, reward,
or architecture ablation:

- keep exactly `t_straight_both_d02` and `t_straight_one_d02`;
- expand only training diversity from eight to 64 unique geometries per cell;
- preserve the original eight train identities as an exact subset and use the
  exact same eight source-disjoint development identities per cell;
- keep scratch seed 2026072704, `corrected_dense_v1_trench_absolute_off`,
  PPO, base `resnet_spatial_8x8`, entropy schedule, reset, and 450-step horizon
  unchanged;
- start one continuous 1,000-update run, evaluate every 100 updates, require
  6/8 per cell at two consecutive checkpoints with a saved successful
  trajectory and zero integrity failures, and apply the global slight-progress
  rule for a 2,000/5,000 extension; and
- if this diversity-only treatment still memorizes, stop before changing
  reward or architecture and inspect its train/development trajectories.

The four-cell distance panel receives no new PPO allocation now. Its d02
behavior is covered by the close side treatment; d04-d08 remain ordered later
curriculum cells, with far dumping admitted only after the close cell
generalizes.

The diversity-only bank was generated and statically accepted at
`2026-07-26T12:40+02:00`:

- local immutable candidate:
  `/home/lorenzo/moleworks/.artifacts/terra_b0_trench_side_diversity_20260726`;
- schema `terra_b0_trench_side_diversity_v1`, with 64 training identities and
  eight development identities for each of `t_straight_both_d02` and
  `t_straight_one_d02`;
- 144/144 unique map IDs and target arrays, source-disjoint train/development
  splits, preserved paired dig geometry, and all static capacity, obstacle,
  source, and loader checks passed;
- the original first eight training identities and all eight development
  identities per cell match the frozen B0a bank in all 18 declared identity,
  geometry, target, and validation fields. The 32-row reference gate used B0a
  identity-manifest SHA-256
  `911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e`;
- 31 training and one development proposals were rejected as templated
  duplicates before the declared counts were reached;
- `files.sha256`, `provenance.json`, `validation.json`, and the paired panel
  gallery have SHA-256 values
  `3e9059d7e167f8b0f054c46a9da2b3b5f1e1d6991041f57fde4e4ca3f953f4eb`,
  `4658f7611e5590cb34051bb3802ba5dc4269098157df47568262a6d242976bff`,
  `177740f1b3f9e04ce9f21c56cb467fffca3fd4ceec5fe71f283d6e8018c8a91c`,
  and
  `a6cfbe8dd7a25d1931dbaf6834a31482262d7b6ae3f721fad407001f014f9a1b`;
- visual review of the train and development galleries confirmed close,
  oversized side-cast regions, distinct both/one-side access, varied trench
  position/orientation/length, no obstacles, and no obvious impossible map;
  and
- the new generator tests plus the original B0 feasibility-panel tests pass
  (`6 passed`), as do byte compilation, Black, and whitespace checks.

This is a static bank receipt, not a learning result. It authorizes exactly
one 1,000-update `B0-DIVERSITY-T-SIDE` run after the bank and source are copied
and hash-verified under a new immutable Euler root.

That treatment was packaged and submitted at `2026-07-26T12:50+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_side_diversity_v1`;
- source revisions: Terra
  `db6bead7ee9360bb6a203c8b58f0956984bc73d6` and terra-baselines
  `3dcb7d2400c391f0630cc8d2a05ce408761187ab`;
- source-manifest, copied-bank-manifest, creation-receipt, and submission-receipt
  SHA-256 values
  `7eefb6a986a9ed0beda8e80140817321fa7f6b88bdbd85d6627b8ec36a919d9b`,
  `3711c0d6c3ff715afcbf296926ba6061dfe5d172c496d5af285b4e2929a30ace`,
  `5805ec5d83b98c6da59b1ce8dd6a9522c3251ddbb7b4b7c4db95fcced15f1a31`,
  and
  `2e695fcca670538964b256e08f618f993144e5bc6db060790667a0d9dab7e73d`;
- source, all 1,753 copied bank files, the bank-internal manifest, exact
  revisions, 128-train/16-development counts, validation status, script syntax,
  and all outer hashes passed before submission;
- training job `8658605` runs one continuous from-scratch 1,000-update
  `trench_side_diversity_v1` treatment on four RTX 4090 GPUs, with seed
  2026072704 and only `eu-g6-064` excluded; and
- fixed-development evaluation job `8658606` is held by an `afterok`
  dependency on the complete training and terminal integrity gate.

This remains a submission receipt, not evidence of learning. Update 1 must
pass the exact checkpoint/optimizer/aggregate/dataset gate before the
production body is accepted; the 10-checkpoint source-disjoint curve then
decides pass, bounded continuation, or stop.

The update-1 smoke gate subsequently passed:

- checkpoint, aggregate, 128-map train manifest, and smoke-gate SHA-256 values
  are
  `9721979acde7700327b6d6012337d8896b0572ad7ee3e8fd980e3f828da538b5`,
  `a56b0c5701cc9989e1f3821a759d1bf29ceb7f9ff93d3a3421cb3d08ed265fdc`,
  `183444a6be146b097afe56199abd9a83eda71e8dd9a5d7c1d72f3f2573c75dab`,
  and
  `a6bd892801ffea9c6cbd397ba36bab48a548aab037a892a82dbaf3564dbcb6e9`;
- all 92 model leaves and 185 optimizer leaves are finite;
- the exact seed, reward, architecture, entropy, PPO, reset, 450-step horizon,
  and one-update checkpoint configuration match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes the continuous 1,000-update body. It remains an integrity
result only, not a map-learning or promotion result.

The corresponding continuous-run W&B ID is `aknko0s4`. It is an operational
pointer only; its online aggregates do not decide continuation or promotion.

The continuous 1,000-update training result is **PASS**:

- job `8658605` completed in `01:47:52` with exit code `0:0`;
- training-gate, FINAL, last-update, numbered-checkpoint-manifest, and
  aggregate-manifest SHA-256 values are
  `9360afde2c8204f6fd77b45983efaad582344a213c49373ab0391c51c2cf515e`,
  `3295b55872a7ec0e1a10cff648dccfd749dda7e42e594fb2483adfbc6ac06893`,
  `c28907b4e7b95ee95af0794cf8b743874b766136ea93b9a7e7fe8768d23ab411`,
  `d25daf3bc60866a34e913d34acdf2ad429719ad0c279b65f6474c1297df28ba8`,
  and
  `dcb7e4d75d4881d3dfe227c72647ce9b250404530a0137107663c8d22331aeb4`;
- all 1,000 ordered aggregate receipts and all 10 numbered checkpoints at
  updates 100 through 1,000 are present; FINAL exactly equals update 1,000
  across 92 model and 185 optimizer leaves;
- the exact 128-map manifest is revalidated; all mass, target, obstacle, and
  per-transition reward hard violation counts are zero; maximum mass residual
  is zero and maximum sub-threshold step-reward residual is
  `4.76837158203125e-07`;
- 199 completed episodes exceed the informational independently accumulated
  episode-return drift tolerance. This is the explicitly non-blocking float32
  association metric documented by the C5 schema-v2 amendment; the
  transition-level reconstruction hard gate is zero; and
- the online stream contains 120,380 task completions out of 376,932 completed
  episodes. That sampled-train quantity is diagnostic only and is not evidence
  of source-disjoint generalization.

The dependent fixed-development result is **CONTINUE SAME PANEL**, not a B0
witness:

- job `8658606` completed in `00:19:05` with exit code `0:0`; its sealed
  `eval.json` has SHA-256
  `fb2fe771a387540306a27c85e14fbbfd278a2fda1b0ddea9b26966b13082d92d`;
- exact-reset verification passed for all 16 development identities at
  `env_steps == 0`, with zero integrity failures;
- the held-out success curve at updates 100 through 1,000 was:

  ```text
  update                         100 200 300 400 500 600 700 800 900 1000
  t_straight_both_d02 successes    0   0   0   0   0   1   1   1   2    3
  t_straight_one_d02 successes     0   0   0   0   0   0   0   1   2    4
  ```

- at update 1,000 the median absolute completions were 0.4044 for both-side
  and 0.5000 for one-side, and a successful action trajectory was saved for
  each cell;
- neither cell achieved 6/8 at even one checkpoint, so neither has the
  required two-consecutive-checkpoint witness and B0 remains unchecked; but
- both cells improved inside the final five-evaluation window, including
  success gains at update 1,000. The preregistered slight-progress rule
  therefore authorizes exactly one fresh, continuous 2,000-update replication
  of the same diversity-only treatment. It does not authorize B0c, a reward or
  architecture change, a 5,000-update run yet, or a 120-hour production run.

Because checkpoint-v2 does not serialize the environment, RNG, and action
history required for a bit-exact continuation, the 2,000-update treatment must
start from scratch and run continuously. Its source, bank, seed, PPO,
architecture, reward, reset, entropy, horizon, and development gate remain
unchanged. A later 5,000-update extension is allowed only if its final
five-evaluation window again satisfies the same held-out slight-progress rule;
the 120-hour queue remains reserved for a recipe that clears the repeated
family/cell gates defined below.

That bounded replication was sealed and submitted at
`2026-07-26T15:14:31+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_side_diversity_u2000_v1`;
- the exact prior source and bank were copied, rather than regenerated:
  Terra `db6bead7ee9360bb6a203c8b58f0956984bc73d6`, terra-baselines
  `3dcb7d2400c391f0630cc8d2a05ce408761187ab`, 128 train identities, and
  16 source-disjoint development identities;
- all 2,160 source/bank files passed hash verification. Source-manifest,
  bank-manifest, creation-receipt, and submission-receipt SHA-256 values are
  `7eefb6a986a9ed0beda8e80140817321fa7f6b88bdbd85d6627b8ec36a919d9b`,
  `3711c0d6c3ff715afcbf296926ba6061dfe5d172c496d5af285b4e2929a30ace`,
  `bf6149f7be1450ba74fd4df2306f43c782c58754f544e5d607dd21131f23f7ed`,
  and
  `157b136e786d66abe05b9e50411e43ad00c236767bb1cf91d4358dbe795dab33`;
- train job `8666365` requests four RTX 4090 GPUs on `gpuhe.24h`, excludes
  only `eu-g6-064`, and runs 2,000 continuous from-scratch updates with the
  unchanged `trench_side_diversity_v1` treatment; and
- fixed-development job `8666366` has an `afterok:8666365` dependency and
  applies the same 20-checkpoint witness/continuation adjudication.

This is a submission receipt only. Update 1 must pass the exact smoke gate
before the production body is accepted, and the source-disjoint development
curve—not online sampled-train success—decides the next allocation.

The 2,000-update diversity replication independently passed its update-1
smoke:

- checkpoint, aggregate, unchanged 128-map dataset manifest, and smoke-gate
  SHA-256 values are
  `0c4eb6d10b8f0da7121afebe0201d03ec8320d6e2760ded234f88babaff47f21`,
  `4a2b5a60aa35f25ee17c768a70cd94441da07b16ad55a2aa40772b69302ad6cc`,
  `183444a6be146b097afe56199abd9a83eda71e8dd9a5d7c1d72f3f2573c75dab`,
  and
  `437cce0eeb8881f354b9a6a821918b22bb2aecd6b33bee41b7e17aa8a72f2b5e`;
- all 92 model and 185 optimizer leaves are finite and the exact declared
  seed, reward, architecture, entropy, PPO, reset, and horizon match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes only the continuous 2,000-update body. It is an integrity
result, not a trench-side witness. The corresponding continuous-run W&B ID is
`d5srft2n`; it is an operational pointer only.

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
