# Terra Map-Curriculum Failure Analysis

- Date: 2026-07-25
- Status: decision-ready diagnosis and execution recommendation
- Scope: completed `NEW-MAPS-FLAT` and `NEW-MAPS-STAGED` screen
- Governing design: [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md)
- Execution backlog: [`TRAINING_TASKS.md`](TRAINING_TASKS.md)
- Production training authorized by this report: no

## 1. Executive verdict

Both completed arms are valid negative results for the treatments that were
actually tested. They are not evidence that Terra cannot learn constrained
excavation maps, and they do not justify a larger model.

The runs were numerically healthy and consumed the complete budget. The flat
arm learned source-disjoint M0 behavior early, peaked at `24/64` at update
1,000, and regressed to `11/64`. The staged arm moved most environments out of
M0 while never demonstrating held-out M1 competence and finished at
`11/64`, `0/64`, and `0/64` on M0-M2. More updates under either unchanged
treatment are not recommended.

The primary measured failures are:

1. **The map ladder is not monotonic.** True all-around foundations are
   learnable, while the nominally easy large-apron foundations are nearly
   unsolved. The cells change geometry source, dump distance, dump capacity,
   topology, and site constraints together.
2. **The staged controller is not a competence curriculum.** Three outcomes
   pooled across randomly sampled maps and families can promote an environment
   without trench or cell-level mastery.
3. **The learned policy does not retain a stable held-out solution set.** Flat
   solves 40 distinct M0 maps at least once across checkpoint cadence, but no
   M0 map at every checkpoint. Only seven of the 24 peak successes survive to
   update 4,000.
4. **Online success is not comparable to the fixed evaluator.** Online
   training is stochastic and uses repeated training identities; fixed
   evaluation is deterministic and source-disjoint. Exact-train and sampled
   fixed evaluations are still missing.
5. **Reward and termination disagree about legal dumping.** Task termination
   accepts a one-cell buffer around the visible target dump zone, while
   completion and terminal reward count only exact target cells.
6. **The reward history is not auditable from current W&B fields.** The scalar
   component logger samples the final state of one environment rather than the
   full rollout. Terminal completion diagnostics cover device 0 and pool
   successes with timeouts.

The next training must be bounded feasibility work on one easy foundation and
one easy trench after semantic and instrumentation corrections. The next map
curriculum must use family-specific, quantitative cells and global
checkpoint-bounded promotion. Progressive rewards and partial resets remain
separate later experiments.

## 2. Frozen run receipt

### 2.1 Source and parent

| Item | Value |
|---|---|
| Terra revision | `d37e780480c0fae64a4b9e4ba6638b4499748761` |
| terra-baselines revision | `2722d832c8381a68d594d8bf8298ba3aec7f4c6a` |
| E8 parent update | 20,000 |
| E8 parent SHA-256 | `f364a5dbfe3329542317273819b65cf5fc12a7329fef2d9126d8e3f251a9f674` |
| Bank identity manifest SHA-256 | `301c68fc2327708b2913bcc07a53a72b8173415a65a44b855eba1696244e930d` |
| Bank provenance SHA-256 | `18141d54ea3bb933eef9652546a05086e70b706c2cca14e39719cfcc25802d6f` |
| Development horizon | 450 actions |
| Training seed | 42 |

Both arms used a parameters-only E8 warm start. Model parameters were copied,
while optimizer state, learning-rate schedule position, update counter,
environment, curriculum state, RNG, and histories were fresh.

### 2.2 Jobs and W&B

| Treatment | Slurm | W&B | Result | Wall time |
|---|---:|---|---|---:|
| flat | `8398905` | `mwn0d0cr` | `COMPLETED`, `0:0` | 06:09:04 |
| staged | `8398906` | `kh3pf9tw` | `COMPLETED`, `0:0` | 06:15:17 |
| flat fixed eval | `8398907` | disabled | `COMPLETED`, `0:0` | 00:45:31 |
| staged fixed eval | `8398908` | disabled | `COMPLETED`, `0:0` | 00:59:52 |

Each training arm used four RTX 4090 GPUs, 4,096 environments, 32 rollout
steps, and 4,000 updates. That is 131,072 transitions per update and
524,288,000 transitions per arm.

The complete W&B history contains exactly 4,000 rows for each run, steps
`0..3999`. All logged finite-fraction diagnostics remained `1.0`; the logs
contain no training traceback, non-finite failure, or interrupted checkpoint.

### 2.3 Evaluation artifacts

| Artifact | SHA-256 |
|---|---|
| flat development cadence | `10fe8be0e7d633fe2ef08aa9e53a72411aef9b0ad8becea35dc3992cac9d820b` |
| staged development cadence | `c8377a8e7a44d635c7b696b95fbb2d0c25aab16bc00cdc7cbc2644d95079c42c` |

Local copies are under:

```text
/home/lorenzo/moleworks/.artifacts/terra_training_design_v1_20260724/
  oracle_review_20260724/final_eval/
```

The development evaluator enumerates 64 fixed source-disjoint identities in
each stratum, uses deterministic argmax actions, and evaluates each initial
episode for at most 450 actions.

## 3. What each treatment actually sampled

### 3.1 Flat

`train/local_M2_terminal` contains 256 unique identities:

| Stratum | Foundations | Trenches | Total |
|---|---:|---:|---:|
| M0 | 32 | 32 | 64 |
| M1 | 48 | 48 | 96 |
| M2 | 48 | 48 | 96 |

Flat therefore exposed every environment to 25% M0, 37.5% M1, and 37.5% M2
from update zero. It was family-balanced and had no repeated slots, but it had
only eight unique M0 identities and twelve unique M1/M2 identities per primary
cell.

### 3.2 Staged

Every environment began in `local_M0` and used a `3`-success promotion and
`3`-failure demotion rule.

| Directory | Slots | Unique identities | Weighting |
|---|---:|---:|---|
| `local_M0` | 256 | 64 | every identity repeated 4 times |
| `local_M1` | 256 | 96 | 64 identities repeated 3 times, 32 repeated 2 times |
| `local_M2_terminal` | 256 | 256 | no repeats |

`local_M1` is not family-balanced after materialization: it has 144 foundation
slots and 112 trench slots. Individual M1 cells have 24, 28, or 36 slots. This
is a treatment defect, not just a reporting detail.

Staged occupancy evolved as follows:

| Update | M0 | M1 | M2 |
|---:|---:|---:|---:|
| 0 | 4,096 | 0 | 0 |
| 500 | 3,154 | 942 | 0 |
| 1,000 | 1,702 | 2,363 | 31 |
| 1,500 | 1,213 | 2,781 | 102 |
| 2,000 | 963 | 2,854 | 279 |
| 3,000 | 718 | 3,012 | 366 |
| 4,000 | 706 | 2,965 | 425 |

This occupancy cannot be interpreted as mastery. At update 1,000, 58% of
training environments were already at M1 while deterministic development M0
trench success was `0/32`. Random easy-foundation streaks can move an
environment past unsolved trench cells.

## 4. Training history

### 4.1 Global online terminations

`train/completed_episodes` and `train/successful_episodes` are globally reduced
across all devices and are the trustworthy online termination counters.
A completed non-success episode is a timeout under the current environment.

| Updates | Flat success | Flat completed/update | Staged success | Staged completed/update |
|---|---:|---:|---:|---:|
| 1-500 | 7.8% | 310.9 | 28.4% | 383.1 |
| 501-1,000 | 19.3% | 346.6 | 46.6% | 476.3 |
| 1,001-1,500 | 28.5% | 380.4 | 49.0% | 497.1 |
| 1,501-2,000 | 34.6% | 407.2 | 49.9% | 504.5 |
| 2,001-2,500 | 39.5% | 432.2 | 50.2% | 506.6 |
| 2,501-3,000 | 43.9% | 457.8 | 50.4% | 511.9 |
| 3,001-3,500 | 45.6% | 467.8 | 50.3% | 512.5 |
| 3,501-4,000 | 49.8% | 494.3 | 50.3% | 512.4 |

Across the full run:

| Treatment | Completed | Successes | Timeouts | Success rate |
|---|---:|---:|---:|---:|
| flat | 1,648,646 | 585,884 | 1,062,762 | 35.5% |
| staged | 1,952,206 | 926,975 | 1,025,231 | 47.5% |

The staged online rate plateaued by roughly update 1,000. Flat continued to
improve on its stochastic training stream even while deterministic
source-disjoint M0 success regressed. This divergence is the central
generalization signal.

### 4.2 Terminal completion diagnostics

The current completion fields are averaged over all terminal events on device
0; they mix successes and timeouts. They are still useful as directional
diagnostics.

| Treatment/window | Dig completion | Exact dump purity | Exact moved-to-dump completion | Remaining edge tiles | Remaining inner tiles |
|---|---:|---:|---:|---:|---:|
| flat, updates 1-500 | 0.514 | 0.488 | 0.374 | 61.8 | 19.9 |
| flat, updates 3,501-4,000 | 0.954 | 0.949 | 0.901 | 7.1 | 1.1 |
| staged, updates 1-500 | 0.571 | 0.590 | 0.482 | 50.3 | 9.3 |
| staged, updates 3,501-4,000 | 0.947 | 0.945 | 0.899 | 7.8 | 1.7 |

Both policies learned dense bulk progress. The residual is concentrated near
the final excavation/dump cleanup, especially boundary tiles. Because the
logger does not separate successful terminations from timeouts, these numbers
cannot establish the exact failed-episode bottleneck. The next evaluator must
stratify them by termination reason, family, cell, and map identity.

### 4.3 Optimization history

There is no evidence of numerical failure:

- every finite diagnostic remained exactly `1.0`;
- 500-update mean explained variance rose to `0.993` for flat and `0.996` for
  staged;
- final-window mean value loss was `0.0179` for flat and `0.0117` for staged;
- actor loss remained near zero, as expected for the averaged PPO surrogate;
- final-window gradient norms were approximately `0.99` and `1.05`; and
- no parameter, optimizer, rollout, target, advantage, log-probability, or
  ratio non-finite value was logged.

Policy entropy declined from its early peak to final values of `1.187` for
flat and `0.966` for staged. The entropy coefficient was still `0.01482` at
update 4,000 because the cosine schedule spans 10,000 updates. This leaves
action-selection sensitivity as a live hypothesis, but not a reason to extend
the same runs: held-out regression occurred while entropy was already falling.

## 5. Fixed-bank failure anatomy

### 5.1 Checkpoint cadence

| Treatment/update | M0 | M1 | M2 |
|---|---:|---:|---:|
| E8 zero-shot | 12/64 | not run | not run |
| flat 500 | 14/64 | 2/64 | 0/64 |
| flat 1,000 | **24/64** | 3/64 | 0/64 |
| flat 1,500 | 23/64 | 4/64 | 1/64 |
| flat 2,000 | 22/64 | 6/64 | 0/64 |
| flat 2,500 | 15/64 | 6/64 | 0/64 |
| flat 3,000 | 17/64 | **7/64** | 1/64 |
| flat 3,500 | 12/64 | 6/64 | 0/64 |
| flat 4,000 | 11/64 | 6/64 | 0/64 |
| staged 500 | 12/64 | 1/64 | 0/64 |
| staged 1,000 | **13/64** | 1/64 | 0/64 |
| staged 1,500 | **13/64** | 1/64 | 0/64 |
| staged 2,000 | 9/64 | 1/64 | 0/64 |
| staged 2,500 | 12/64 | 1/64 | 0/64 |
| staged 3,000 | 12/64 | **2/64** | 0/64 |
| staged 3,500 | **13/64** | **2/64** | 0/64 |
| staged 4,000 | 11/64 | 0/64 | 0/64 |

Every failed fixed episode terminated at the 450-step timeout. Every success
completed in at most 144 steps; typical successful medians were 55-70 steps.
The current 450-step horizon is therefore generous for policies that discover
a working mode on these maps. This does not establish that remote-hauling maps
fit the same horizon.

### 5.2 M0 is two different difficulty regimes

The table reports successes over all eight checkpoint evaluations, so each
cell has 64 checkpoint-map opportunities.

| M0 cell | Flat | Staged | Median dig cells | Median dump/dig area | Median dig-dump distance |
|---|---:|---:|---:|---:|---:|
| foundation all-around low | 43/64 | 44/64 | 140 | 28.3x | 2.0 tiles |
| foundation all-around normal | 29/64 | 36/64 | 193 | 20.2x | 2.2 tiles |
| foundation large apron low | 1/64 | 0/64 | 123 | 2.5x | 5.6 tiles |
| foundation large apron normal | 1/64 | 0/64 | 179 | 2.4x | 5.7 tiles |
| trench straight both low | 8/64 | 4/64 | 77 | 2.9x | 3.6 tiles |
| trench straight both normal | 18/64 | 2/64 | 161 | 2.9x | 4.0 tiles |
| trench straight one-side low | 18/64 | 7/64 | 80 | 2.3x | 4.1 tiles |
| trench straight one-side normal | 20/64 | 2/64 | 115 | 2.4x | 4.6 tiles |

The all-around and large-apron foundation cells should not share the same
starter stage. The former makes essentially every legal free cell a target
dump cell. The latter is a genuinely constrained transport task with roughly
ten times less relative dump area and more than twice the dump distance.

### 5.3 M1 and M2 are not uniformly harder

For flat:

- all four M1 foundation cells were `0/32` at every checkpoint;
- 37 of the 40 M1 success events came from
  `straight_one_side_light`;
- `segmented_both_light` produced one success event;
- `segmented_one_side_objects` produced two success events;
- `segmented_separated_objects` produced none; and
- M2 produced only two isolated successes, both in `T_both_objects`.

For staged:

- every M1/M2 foundation cell was always zero;
- only two distinct M1 trench maps ever succeeded;
- no M2 map ever succeeded; and
- the final checkpoint had no M1 success at all.

The flat policy solved the nominal M1 `straight_one_side_light` cell more often
than several M0 cells. Stratum labels therefore do not induce a scalar
difficulty order.

The development metadata also changes several causes together:

- M1 procedural foundations introduce a new geometry source and dump
  distances around 6.6-8.2 tiles;
- object cells add roughly 100-130 obstacle cells;
- M2 wall cells add roughly 147-242 obstacle cells;
- T/X trenches change junction topology and often site constraints together;
  and
- one-side/separated dump styles are introduced alongside those geometry and
  site changes.

No current result can identify which single axis caused a failed composite
cell.

### 5.4 Policy churn

Flat M0 behavior is not a stable nested set:

- 40 of 64 M0 maps succeed at least once across checkpoint cadence;
- no M0 map succeeds at all eight checkpoints;
- from the update-1,000 peak to update 4,000, only 7 successes are retained,
  17 are lost, and 4 new maps are gained;
- the peak-to-final success-set Jaccard index is `0.25`; and
- consecutive-checkpoint Jaccard falls to `0.28` by updates 3,500-4,000.

E8 initially solved 12 all-around foundations and no trenches. Flat update
1,000 learned 18 new M0 identities but retained only 6 of those 12 E8
successes. It acquired trench behavior by moving to a different policy mode,
not by monotonically adding skills.

Staged is more stable only because its successful support is narrower:

- 25 M0 maps ever succeed;
- 3 succeed at all eight checkpoints; and
- its best-to-final M0 Jaccard is `0.50`.

At update 4,000, flat and staged each solve 11 M0 maps but share only 4 of
them. This policy-mode sensitivity reinforces the need for repeated sampled
evaluation and logit-margin analysis before changing the encoder.

## 6. Reward audit

### 6.1 Active reward

Both arms used `Rewards.dense()`, `apply_trench_rewards: false`, one excavator,
and normalizer `70`.

The main normalized terms are:

| Event | Raw term | Normalized term |
|---|---:|---:|
| every step | existence `-0.25` | `-0.003571` |
| move | `-0.10` plus collision `-0.20` | `-0.001429` plus `-0.002857` |
| base turn | `-0.05` plus collision `-0.05` | `-0.000714` plus `-0.000714` |
| cabin turn | `-0.02` | `-0.000286` |
| successful initial dig/load | `+1.0` | `+0.014286` |
| wrong dig | `-0.12` | `-0.001714` |
| failed dump | `-1.0` | `-0.014286` |

Successful dumps receive relocation-potential progress, a dump-zone bonus, and
a `-1` offset before normalization. The run overrides use
`dump_bonus_mult=0.5` and `excavator_relocate_*_mult=1.5`.

The distance potential is therefore already a dense signal for nearby
transport. Far dumping should be tested with this dense parent first, not with
a terminal-only reward.

### 6.2 Terminal reward

For a single excavator, the normalized success terminal reward is:

```text
2 / 70 * (
  200 * ((gated_completion - 0.6) / 0.4)^2
  + 40 if gated_completion >= 0.999 else 0
)
```

for `gated_completion >= 0.6`, and zero below it. A perfect success receives
`6.8571`. A timeout receives:

```text
2 / 70 * 20 * gated_completion^2
```

with a maximum of `0.5714`.

For dig-and-dump maps without border enforcement:

```text
gated_completion =
  0.6 * exact_dumped_volume / required_dig_volume
  + 0.4 * dig_completion
```

This is a useful dense feasibility bridge, but it is not the final terminal
objective proposed in the progressive-reward design.

### 6.3 Confirmed termination mismatch

Task termination accepts positive dirt in a one-cell dilation of the target
dump zone. Completion and reward count positive dirt only on exact
`target_map > 0` cells.

At a legal task termination with all digging complete, let `f` be the fraction
of required dirt volume placed on exact target cells. Then
`gated_completion = 0.6 f + 0.4`. Dirt in buffer-only cells can therefore:

- remove the perfect-completion bonus;
- sharply reduce a successful terminal reward; or
- produce zero successful terminal reward when too little volume is on exact
  cells.

All-around foundations largely hide this mismatch because almost every legal
free cell is already an exact target. Apron, one-side, and separated layouts
expose it. This makes the defect directionally consistent with the observed
difficulty gap, but trajectory replay is required before assigning causal
weight.

### 6.4 Logged reward history is insufficient

The current logger:

- averages `rewards/agent_0` across environments at only the last rollout
  step;
- converts `rewards/terminal`, `rewards/trench`, and `rewards/existence` from
  the final state by taking the first array element;
- does not log full-rollout reward or per-episode return by component; and
- does not separate reward components at success from components at timeout.

This is visible empirically. `rewards/terminal` is nonzero in only 14 of 4,000
flat log rows and 15 of 4,000 staged rows despite 585,884 and 926,975 online
successes. `rewards/trench` is always zero because trench shaping was disabled,
and `rewards/existence` is the expected constant `-0.003571`.

No retrospective claim about terminal-reward frequency, return saturation, or
reward-component dominance should be made from those W&B fields.

## 7. Termination and evaluator audit

The reliable training termination fields are:

- `train/completed_episodes`;
- `train/successful_episodes`; and
- `train/episode_success_rate`.

The following fields are easy to misinterpret:

- `progress/episode_completion_rate` is only the fraction terminal on the final
  step of the latest rollout;
- `terminal/episode_count` and `terminal/success_count` are device-0 legacy
  counts; and
- fixed-evaluator `terminated: true` includes horizon timeout, so all 64
  development episodes are reported terminated even when success is zero.

The fixed evaluator does not yet report:

- task success versus timeout versus both on the same transition;
- exact-mask versus accepted-buffer completion;
- reward components by termination reason;
- positive buffer-only volume;
- mass residual;
- invalid action execution;
- target, obstacle, or dump-mask mutation;
- per-map action counts and productive workspace cycles; or
- deterministic logit margin and repeated sampled outcomes.

The first training episode also randomizes hidden `env_steps` in `[0, 450)`.
This violates the full-reset contract but affects only the first 4,096
episodes: about 0.25% of flat completions and 0.21% of staged completions. It
cannot explain the late regression.

Legacy terminal backfill is called unconditionally, but it adds nothing for
these single-agent runs because the backfill window is gated by
`num_agents > k`.

## 8. Evidence-ranked failure hypotheses

| Rank | Hypothesis | Status | Evidence | Deciding action |
|---:|---|---|---|---|
| 1 | pooled `3/3` progression bypasses unsolved cells | confirmed treatment defect | M1 majority occupancy with M0 trench `0/32`; no identity/family conditioning | retire it; use global fixed-bank gates |
| 2 | current strata are non-monotonic and confounded | confirmed | all-around versus apron gap; M1 straight trench easier than many M0 cells | rebuild orthogonal family-specific cells |
| 3 | train-identity fit and/or stochastic policy explains online/offline gap | likely, unresolved | online near 50%, held-out near zero, tiny repeated banks, deterministic evaluator | exact-train deterministic plus repeated sampled eval |
| 4 | heterogeneous exposure drives flat retention regression | likely, unresolved | peak-to-final M0 Jaccard `0.25` under 75% M1/M2 exposure | paired 500-update M0-only versus terminal-mixture fork |
| 5 | reward/termination mismatch depresses constrained-map learning | confirmed defect, unknown effect size | exact versus dilated dump masks | counterfactual replay audit |
| 6 | late excavation/dump cleanup is the immediate skill bottleneck | supported diagnostic | terminal completion near 0.9 with residual edge tiles | success/timeout-stratified terminal metrics and trajectories |
| 7 | clipped global positive heights alias transported volume | confirmed information loss, unknown effect | global `action_map` clipped to `[-1, 1]` | paired-state legality/reward/optimal-action test |
| 8 | PPO entropy or policy drift contributes to churn | possible | moderate entropy and changing fixed success sets | logit/KL/action-mode audit, then bounded fork |
| 9 | current architecture lacks capacity | not supported | stable optimization; no representation-specific failure isolated | do not sweep architecture |
| 10 | the runs simply need more time | contradicted for these treatments | 524M transitions each; staged plateau; flat held-out regression | stop unchanged continuations |

Dynamic infeasibility remains possible for some cells. Static capacity and path
checks explicitly are not action witnesses. It must be tested with bounded
fixed-identity learning and legal trajectory capture.

## 9. Recommended map redesign

### 9.1 Replace one M0-M2 scalar with family-specific cells

Difficulty must be expressed as an explicit vector:

```text
(family, geometry/topology, dump layout, dump distance,
 reachable capacity, site constraint, work volume, partial-state mode)
```

Change one axis at a time in feasibility panels. Composite maps enter the
generalist distribution only after their component axes have dynamic
witnesses.

### 9.2 Foundation ladder

Recommended order:

1. contiguous OSM foundations with all legal free ground dumpable;
2. contiguous procedural foundations with the same all-around dump contract;
3. the same geometries with a broad near apron;
4. one-side and separated dump zones, still without obstacles;
5. internal holes, bearing-wall strips, pillar-like negative components, and
   partially disconnected dig shapes;
6. scattered objects, access roads, and gapped walls one at a time; and
7. combined constraints only after each individual axis passes.

This isolates the current untested question: are procedural foundation
geometries hard, or are only their constrained dump layouts hard?

For the first apron feasibility cells, remove capacity as a bottleneck:
candidate starting gates are at least `3x` nearby reachable dump area,
dig-dump median no more than `3` tiles, and no obstacles. These are hypotheses
to validate, not permanent deployment limits.

### 9.3 Trench ladder

Classify geometry by topology rather than a broad “trench” label:

1. one straight segment;
2. two and then three end-to-end segments without a junction;
3. one T junction;
4. one X junction;
5. `N` intersecting segments, recorded by segment count, junction count, and
   junction degree; and
6. disconnected trench groups.

Within each topology, progress dump layout separately:

1. large both-side side-cast;
2. large one-side side-cast;
3. per-segment allowed sides;
4. separated nearby zones; and
5. remote haul as a conditional robustness track.

Do not introduce a junction, obstacles, and a one-side dump restriction in the
same first exposure.

### 9.4 Controlled dump-distance feasibility

Build paired maps from identical simple geometry, work volume, site, and dump
capacity while shifting only the dump region. Use path-distance bins centered
near 2, 4, 6, and 8 tiles. The bank metadata uses `0.6875 m/tile`, but training
and promotion should record tiles as the canonical unit.

Start with at least `3x` reachable capacity and no obstacles. Keep the
450-action horizon only when a constructive action witness or conservative
lower bound fits it.

Train progressively with the corrected dense reward:

- pass a distance bin before initializing the next from its checkpoint;
- evaluate every 50-100 updates on fixed source-disjoint maps;
- keep earlier bins in a retention bank; and
- if 8 tiles fails, diagnose action feasibility, horizon, reward, and
  observation before exposing a generalist to remote dumping.

Remote haul at 12 or more tiles is a separate specialist feasibility question,
not an early generalist requirement.

## 10. Recommended curriculum controller

Use global checkpoint-bounded stages, not per-environment streaks.

For each family:

1. train on a declared frontier cell mixture;
2. evaluate a frozen, source-disjoint promotion bank;
3. require at least `26/32` family successes and `6/8` in every included cell;
4. require the gate at two consecutive checkpoints;
5. require no integrity failure and no more than a five-point loss on mastered
   cells; and
6. promote only by starting a new recorded run from the passing checkpoint.

If a mastered cell fails retention at two evaluations:

1. stop the current stage;
2. restore the last checkpoint passing all previous gates;
3. return to the previous mixture as a new recorded treatment; and
4. do not silently mutate map weights inside the compiled PPO run.

The initial rehearsal fraction is an experimental parameter. Select it from the
historical retention fork rather than treating 20-30% as established.

Foundation and trench specialists should qualify independently. Only then
train a 50/50 easy-family generalist. A pooled generalist score must never hide
a failed family or cell.

## 11. Minimal execution sequence

### P0 — no-gradient diagnosis and correctness

1. Run the exact-mask versus accepted-buffer replay audit on flat update
   1,000, flat update 4,000, and staged update 4,000.
2. Evaluate E8, flat updates 1,000/4,000, and staged updates 1,000/4,000 on
   exact unique training identities and development identities.
3. Repeat sampled-action M0 evaluation for flat updates 1,000/4,000 with eight
   fixed seeds; report deterministic disagreement and logit margin.
4. Ratify one legal dump-mask contract, then make termination, completion,
   reward, logging, and evaluation share it.
5. remove hidden first-episode horizon randomization for full-reset runs;
6. make map loading exact and fail on missing or duplicated identities unless
   duplication is an explicit treatment; and
7. add global, termination-stratified reward and integrity logging.

No new PPO training is authorized before this phase closes.

### P1 — bounded dynamic-feasibility probes

Train exactly two fixed identities:

- one low-volume all-around foundation; and
- one low-volume straight both-side trench.

Use E8 parameters only, a fresh optimizer, the corrected dense reward, untouched
450-step resets, and the current `_se` architecture. Stop at 50, 100, 250, or
500 updates as soon as `29/32` fixed reset seeds pass twice with a saved legal
trajectory.

E8 is appropriate as the first parent because it already solves all-around
foundations and flat proves that it can adapt to some new straight trenches.
A scratch control is conditional on a failed fixed-identity probe; it is not
part of the first minimal set.

### P2 — small orthogonal feasibility panels

Only after P1 passes:

1. run the paired dump-distance panel on simple foundation and trench
   geometries;
2. create procedural all-around foundations to isolate geometry source;
3. test trench topology by segment/junction count while keeping dump layout
   easy; and
4. admit only dynamically witnessed cells to larger training banks.

These are specialist feasibility instruments, not deployment models.

### P3 — family specialists and easy generalist

1. Generate at least 64 unique train identities per selected primary cell,
   with no repeated slots.
2. Train an easy foundation specialist and an easy trench specialist.
3. Require family and cell gates on separate promotion/development banks.
4. Train a corrected 50/50 easy-family generalist from E8 parameters, not by
   merging specialist weights.
5. Use a 1,000-update initial budget and extend to 2,000 only while fixed-bank
   performance is improving.

### P4 — global map curriculum

Progress one quantitative axis at a time with checkpoint-bounded promotion and
retention. The first matched curriculum comparison is:

```text
qualified easy parent
  +-- flat exposure to the next declared family cells
  `-- global staged exposure with fixed-bank promotion
```

Do not reuse the per-environment `3/3` treatment.

### P5 — separate reward and reset curricula

After a corrected dense parent qualifies:

1. repair the Stage-1 parity conflict in the current local
   `/home/lorenzo/moleworks/terra/PROGRESSIVE_REWARD_CURRICULUM.md`
   specification;
2. run the matched corrected-dense versus dense-to-terminal A/B/C study on one
   fixed qualified foundation family;
3. select lexicographically by success, productive workspace cycles, and
   steps;
4. validate the selected reward sequence separately on trenches; and
5. test 25% mass-conserving partial resets only after the map sampler is
   selected.

Map level, reward stage, and partial-reset probability must never advance in
the same causal comparison.

## 12. Required instrumentation before P1

Every terminal episode must emit one global record containing:

- `map_id`, split, family, primary cell, and active curriculum stage;
- termination reason: `task_done`, `timeout`, or `task_done_and_timeout`;
- exact-target and accepted-mask completion;
- exact, buffer-only, and illegal positive soil volume;
- dig completion, dump purity, moved-to-dump completion, and residual component
  counts;
- total return and every reward component summed over the episode;
- terminal reward before and after normalization;
- episode steps and action histogram;
- invalid-action attempts and invalid executions;
- productive workspace cycles;
- mass residual and target/obstacle/dump-mask integrity; and
- deterministic logit margin when produced by fixed evaluation.

Training aggregation must reduce over all devices. W&B should log totals and
bounded rates, not an arbitrary environment element. The machine-readable
episode receipt remains the source of truth for recomputation.

The fixed evaluator must expose deterministic versus sampled mode explicitly
and must not call a timeout a successful completion.

## 13. Architecture decision

Keep `resnet_spatial_8x8_se` through P0-P3.

Before an architecture experiment:

1. construct paired states that become identical after global positive-height
   clipping but require different legality, reward, or optimal action;
2. add the smallest missing scalar or channel proven by that test;
3. verify fixed identities and family specialists with the corrected input;
   and
4. use xattn only if the remaining failures specifically require
   agent-conditioned selection among spatial dump/obstacle regions.

Do not test v5, a transformer core, recurrence, or higher resolution merely
because the task horizon is long. None restores information discarded before
the encoder.

## 14. Decision boundaries

Do not conclude:

- that M1/M2 maps are dynamically impossible;
- that deterministic evaluation equals stochastic policy ability;
- that flat regression is classical catastrophic forgetting;
- that entropy is the cause;
- that the current encoder is too small;
- that a reward curriculum will repair an invalid map/termination contract; or
- that far dumping fits 450 actions.

Do conclude:

- the tested flat terminal mixture and per-environment `3/3` treatment failed;
- the unchanged runs have saturated their useful scientific value;
- all-around dumping is the correct first foundation stage;
- map difficulty needs quantitative family-specific axes;
- the reward/termination mismatch must be fixed before future-policy training;
  and
- trustworthy reward/termination instrumentation is a launch blocker.

## 15. Goal completion criteria

The curriculum-recovery goal is complete only when:

1. one legal dump-mask definition drives termination, completion, reward,
   metrics, and evaluation;
2. reward and termination receipts are globally aggregated and recomputable;
3. one easy foundation and one easy trench pass fixed-identity feasibility;
4. family specialists pass source-disjoint family and cell gates;
5. a corrected dense 50/50 easy generalist passes two consecutive retention
   gates;
6. a global map curriculum advances by fixed-bank evidence without hiding a
   failed cell;
7. dump-distance limits are measured, with remote haul either qualified or
   explicitly separated as unsupported;
8. progressive rewards and partial resets are selected in separate matched
   experiments; and
9. one sealed composite bank confirms generalization across feasible
   foundations, trenches, dump constraints, and site obstacles.
