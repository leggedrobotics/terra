# C0/T1 fresh-trench dig-alignment pilot — u10,000 readout

Date: 2026-08-21. Both arms were **still training** when this was measured
(jobs 11152229 C0 / 11152230 T1, ~u11,900 of 100,000). Checkpoints were copied
read-only; no run directory, bank, or Slurm job was written to.

This readout reports numbers and states the preregistered rule outcomes
mechanically. It does **not** decide whether to stop the pilot.

---

## 0. Headline

The gate works. The raw completion comparison is the wrong criterion.

| endpoint (trench − net4, 176 slots) | C0 (gate off) | T1 (gate on) | Δ |
|---|---|---|---|
| **raw** strict exact completion | **156/176 = 88.64%** | 68/176 = 38.64% | **−50.00 pp** |
| **admissible** exact completion | **8/176 = 4.55%** | **68/176 = 38.64%** | **+34.09 pp** |

C0 reaches 88.64% by digging from physically inadmissible stations: **83.35%
of its applicable fresh-dig attempts are pose-invalid**, and 166/176 of its
episodes contain at least one inadmissible dig (median 6 in a *successful*
episode). Only 8 of its 154 probe-measured successes used exclusively
pose-valid digs, and all 8 are on single-axis straights.

The two arms are not producing the same kind of output, so the −50 pp raw
figure is not a like-for-like regression.

---

## 1. Protocol actually executed

**Primary endpoint.** `eval_fixed_bank.py --panel-family gate_main
--accepted-panel development --terra-revision
a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 --horizon 450`, one invocation per
arm (fingerprints differ by design), joined externally. 608 slots, 38
conditions. Deterministic (argmax), seed 20260724, completion contract
`exact_visible_dump_v1`.

Endpoint scope = the 14 trench conditions minus the 3 net4 conditions = **11
conditions × 16 = 176 slots**. These 11 are exactly 11 of the 12 pooled
conditions the arms trained on (the 12th, `trn-straight-allfree`, is a
capability-floor condition absent from gate_main). Foundation and net4 are
reported separately and never pooled into the endpoint.

Integrity, verified from the receipts:

- both arms consumed byte-identical maps: equal `reset_verification.layer_sha256`
  and equal `manifest_sha256`;
- both integrity blocks passed: 0 target mutations, 0 obstacle mutations,
  0 mass-residual failures, 0 non-finite states;
- treatment fingerprints differ only in `enforce_trench_dig_alignment`
  (C0 `a68b6376c5a4ddbb`, T1 `85fe0198abd5f48a`).

**Mechanism endpoint.** Terra logs none of this, so it was measured offline
with a new instrumented rollout tool,
`scripts/trench_align_rollout_probe.py` (in the baselines worktree, matching
its `scripts/` convention; sha256 `67d75b8154407e5f…`). It replicates
`eval_mcts.rollout_episode` as `eval_fixed_bank.py` drives it (deterministic
argmax, `prev_actions` history, frozen manifest map/episode seeds,
`step_no_reset` with inactive-slot preservation) and additionally records, per
step and per slot: chosen action; whether the prospective DO was fresh-trench
*applicable*; the three exported alignment scalars **before** the step; the raw
physical yaw (rad) and standoff (m) of the diagnostic section; whether the step
had an effect; how many fresh trench cells it dug; and, optionally, a
differential gate-on/gate-off successor comparison.

- **Fixed map set, identical for all four cells:** the same 176 trench−net4
  slots of `evaluation/gate_main/development`, `DATASET_SIZE` held at the full
  panel 608 so `exact_reset_keys` keeps its slot→map identity.
- Horizon 450, seed 20260724, deterministic. 176 episodes per cell.
- Cells: C0 and T1 × u500 and u10,000. `--differential-gate` on both T1 cells.

**Self-checks built into the probe.** At every step it asserts that its own
transcription of the alignment computation reproduces Terra's exported
`fresh_trench_dig_alignment_valid` / `..._yaw_error` / `..._standoff_error`
exactly (atol 1e-5). All four cells ran to completion with the assertion
active, so the raw physical quantities below are anchored to the shipped
export, not to a re-derivation that might have drifted.

**Cross-validation against the panel.** The probe's independent episode
outcomes reproduce the panel endpoint exactly for T1 (68/176) and to within 2
episodes for C0 (probe 154 vs panel 156). The difference is expected: per-env
step RNG comes from `jrandom.split(rng_step, count)` and `count` is 176 in the
probe versus 608 in the panel. Panel numbers are authoritative for the
endpoint; probe numbers are used self-consistently for anything trace-derived.

---

## 2. Primary endpoint

### 2.1 By scope

| scope | C0 | T1 | Δ pp |
|---|---|---|---|
| **endpoint: trench − net4 (11 cond)** | **156/176 = 88.64%** | **68/176 = 38.64%** | **−50.00** |
| net4 (3 cond, preregistered exclusion) | 36/48 = 75.00% | 1/48 = 2.08% | −72.92 |
| foundation (24 cond, context only) | 12/384 = 3.12% | 4/384 = 1.04% | −2.08 |
| whole gate_main panel (38 cond) | 204/608 = 33.55% | 73/608 = 12.01% | −21.55 |

Macro and micro coincide on every scope because all conditions carry 16 slots.
Foundation is near zero for both arms as expected — these are trench
specialists trained only on the 12 pooled trench conditions. Foundation is
context, not an endpoint.

`gate_main` numbers are **not** comparable to historical 45-condition
main-panel results; the receipts carry `evaluation_panel_family: gate_main`.

### 2.2 By condition — raw vs admissible

Admissible exact completion = the episode completed exactly **and** used only
pose-valid fresh trench digs.

| condition | C0 raw | T1 raw | Δ raw | C0 admissible | T1 admissible | Δ admissible |
|---|---|---|---|---|---|---|
| trn-net3-side1-road | 9/16 | 0/16 | −56.2 | 0/16 | 0/16 | +0.0 |
| trn-net3-side2 | 15/16 | 5/16 | −62.5 | 0/16 | 5/16 | +31.2 |
| trn-net3-side2-s | 16/16 | 2/16 | −87.5 | 0/16 | 2/16 | +12.5 |
| trn-seg2-side2 | 16/16 | 7/16 | −56.2 | 0/16 | 7/16 | +43.8 |
| trn-seg3-side2 | 14/16 | 3/16 | −68.8 | 0/16 | 3/16 | +18.8 |
| trn-straight-altsides | 12/16 | 6/16 | −37.5 | 1/16 | 6/16 | +31.2 |
| trn-straight-side1 | 16/16 | 9/16 | −43.8 | 1/16 | 9/16 | +50.0 |
| trn-straight-side1-tight | 15/16 | 12/16 | −18.8 | 0/16 | 12/16 | +75.0 |
| trn-straight-side2 | 15/16 | 14/16 | −6.2 | 6/16 | 14/16 | +50.0 |
| trn-tee-side2 | 13/16 | 2/16 | −68.8 | 0/16 | 2/16 | +12.5 |
| trn-tee-side2-s | 15/16 | 8/16 | −43.8 | 0/16 | 8/16 | +50.0 |
| **TOTAL** | **156/176** | **68/176** | **−50.00** | **8/176** | **68/176** | **+34.09** |

C0's raw advantage is largest exactly where its admissible completion is zero
(junctions). Its only admissible successes are 6 on `trn-straight-side2` and
one each on `trn-straight-side1` / `trn-straight-altsides`.

Graded rather than all-or-nothing, the picture is the same: C0's admissible
share of all its fresh digs is **16.65%** (188 admissible vs 941
inadmissible), and its median per-episode admissible fraction is **0.000**.

**Caveats on the admissible endpoint.**

1. T1 satisfies it *by construction* — the gate makes an inadmissible dig
   impossible, so T1 cannot fail this criterion. The metric therefore does not
   measure T1's skill; it measures **how much of C0's raw advantage is
   realizable**, and the answer is almost none of it.
2. Terra-gate-admissible is a **necessary, not sufficient** condition for ROS
   physical acceptance. Terra's cone is discrete, the `CABIN_CONTROL` offset
   (−0.274 m, under half a cell) is deliberately not modelled, and there is no
   swept-path check. Real acceptance also needs continuous footprint, reach,
   endpoint, and swept-path validation.
3. Raw completion alone is **not** a valid promotion criterion. This
   retro-justifies the preregistration's promotion rule requiring the ROS
   physical-acceptance lower bound to exceed zero *alongside* the completion
   bound — that clause is load-bearing, not belt-and-braces.

### 2.3 Failure shape

| trench − net4 | C0 | T1 |
|---|---|---|
| failures | 20/176 | 108/176 |
| median dig_fraction of failures | 0.675 | 0.311 |
| failures ≥ 90% dug | 5/20 | 3/108 |
| all failures at the 450 horizon | 20/20 | 108/108 |
| mean dig_fraction (all) | 0.9549 | 0.5841 |
| mean episode steps | 94.3 | 294.8 |
| no-effect actions / episode | 11.9 | 102.5 |

T1's failures are not near-complete timeouts; they stall around a third dug.
"Needs a longer horizon" is ruled out.

The 102.5 no-effect actions per T1 episode are **not** gate blocks: there were
129 invalid attempts across all 176 episodes (0.73 per episode, ≈0.25% of T1's
active steps), two orders of magnitude below the ~20 pp no-effect excess. The
excess is ordinary ineffective actions from an unconverged policy.

---

## 3. Mechanism endpoint

### 3.1 The 2×2

All cells: same 176 maps, same seed, 450 steps, deterministic.

| cell | update | gate | succeeded | applicable states | aligned share | invalid attempts | invalid / applicable | DO-rate while misaligned | admissible aligned digs/episode |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 500 | off | 0/176 | 3,583 | 13.1% | 442 | **0.9888** | 0.1419 | 0.03 |
| C0 | 10,000 | off | 154/176 | 5,937 | 18.1% | 941 | **0.8335** | 0.1936 | 1.07 |
| T1 | 500 | on | 1/176 | 24,340 | 6.6% | 0 | **0.0000** | 0.0000 | 1.38 |
| T1 | 10,000 | on | 68/176 | 31,913 | 41.5% | 129 | **0.1822** | 0.0069 | 3.29 |

"Applicable state" = active, empty excavator, prospective DO would remove fresh
trench soil. "Aligned share" = fraction of those states that are pose-valid.

### 3.2 The control answers the causal question

**Alignment competence is not learned incidentally.** Without the gate, C0
improves only marginally over 9,500 updates — invalid share of applicable
attempts 98.88% → 83.35%, aligned share 13.1% → 18.1% — and remains
catastrophically misaligned, because Terra's completion objective does not
reward alignment. With the gate, T1 sits at 18.22% invalid and reaches aligned
poses **41.5%** of the time, 2.3× C0's rate.

At matched u10,000 the gate reduces the invalid share of applicable fresh-dig
attempts from 0.8335 to 0.1822 — a **78% relative reduction**, and a 28×
reduction in the rate of digging while misaligned (0.1936 → 0.0069).

### 3.3 Raw yaw / standoff of successful fresh digs

Diagnostic-section raw physical quantities. Reported because the *exported*
standoff error is clipped to exactly 0 whenever the pose is in band and so
carries no information for an admitted dig.

| cell | successful fresh digs | raw yaw (deg) mean / p90 | raw standoff (m) mean / p10 / p90 |
|---|---|---|---|
| C0 u10,000 | 188 | 0.0018 / 0.0000 | 4.810 / 4.132 / 5.434 |
| T1 u500 | 242 | 0.0021 / 0.0178 | 4.699 / 3.760 / 5.640 |
| T1 u10,000 | 579 | 0.0015 / 0.0000 | 4.971 / 4.280 / 5.637 |

Successful digs are essentially perfectly parallel in every cell — unsurprising,
since Terra's 12 base headings are 30° apart and the tolerance is 15°, so an
admitted dig is almost always exactly on-axis. The distributions are already at
floor at u500 and cannot tighten further; this endpoint has no headroom in this
setup and should not be read as a trend.

The **invalid** attempts are the informative tail. Of T1's 129 at u10,000:

| failure mode | count |
|---|---|
| yaw out of tolerance only, standoff in band | **128** |
| standoff out of band only, yaw in tolerance | 1 |
| both | 0 |

The yaw errors are a spike at exactly **30.00°** — one heading bin off
parallel — at a healthy 3.97 m standoff. These are not geometrically
impossible stations; they are one rotation step from admissible.

### 3.4 Why T1's u500 baseline is 0.0000 — competence, not incompetence

At u500 T1 was in a **misaligned applicable state 22,733 times, across 3,413
distinct contiguous segments, in 161/176 episodes** (median step index 224;
only 3% within the first 10 steps) — and chose DO in **exactly zero** of them,
while choosing DO at a 15.1% rate in aligned states.

So the zero is not "it never got into position to make a mistake." It is a
policy that had **already learned the constraint perfectly** by the first
scheduled evaluation. u500 is 500 × 65,536 = **32.8M transitions**, not an
untrained policy. The gate's lesson is acquired early; what takes the remaining
updates is learning to *complete trenches* under it.

This is the real reason the preregistered mechanism clause is unevaluable: the
quantity it tracks was already saturated at floor before the first measurement.

### 3.5 Retracted measurement: `pose_valid_axis_available_at_applicable_do`

The probe originally reported, at each applicable DO, whether at least one
*owning* section was pose-valid, intended as independent evidence about whether
an admissible pose was reachable. **It is degenerate and is withdrawn.**

Mechanism: Terra's `axis_pose_valid` already conjoins `axis_has_fresh` (the
axis must own a fresh cell in the selected cone), and the macro `valid` is then
"every fresh cell has a pose-valid owner." On the typical single-owner cone
(mean owning-section count 1.23), "some owning section is pose-valid" and "the
DO is admitted" are the same proposition. Verified as an exact identity over
all 176 × 450 steps: `invalid ⟺ applicable ∧ pose_valid_axis_count == 0`.

Its 81.8% was therefore just 1 − 0.182, the invalid fraction restated. Reading
it as "an admissible pose existed 82% of the time" would have been circular.
The field is retained in the receipts under
`retracted_pose_valid_availability` so the error is auditable.

### 3.6 Retracted measurement: pooled per-step attempt rate

An earlier pass reported a 4.37% "attempt rate at opportunity" for T1 u10,000
as evidence of deterrence. **Withdrawn.** Two independent defects:

1. **Dwell weighting.** 93.8% of opportunity-steps come from timed-out
   episodes. Split: successful episodes 373/818 = 45.6%; failed episodes
   206/12,422 = 1.66%; pooled 4.37%.
2. **Absorbing NOOP loops.** The apparent "39.2% DO_NOTHING at opportunity"
   comes from 17 of 176 episodes, with 12 contiguous runs of ≥50 steps (longest
   435) carrying 97% of those steps, 99.9% inside failed episodes. With a
   deterministic argmax policy, DO_NOTHING does not change the world; once
   `prev_actions` saturates the observation is fixed and the episode burns to
   the horizon.

**Structural confound, applying to any per-step attempt rate:** opportunity
duration is endogenous. Digging removes the soil and *ends* the opportunity;
dawdling extends it. So the statistic is biased in both directions and is not
used as willingness evidence anywhere in this readout.

Dwell-free replacements: episodes that ever dug at an opportunity (T1 98.5% at
u500 → 83.4% at u10,000) and admissible digs per episode (1.38 → **3.29**).
T1 performs 2.4× *more* successful aligned digs per episode at u10,000 than at
u500, which is why the deterrence reading was dropped.

---

## 4. W&B, matched updates

Runs `trench_align_{c0,t1}_f64694a569_s20260818` in `aless-weber-eth/mixed-agents`,
both `running` at pull time; history to u11,871 (C0) and u11,991 (T1).

Data note: W&B serialises `NaN` as the **string** `"NaN"`, which silently
poisons naive aggregation; values must be coerced before filtering.

At u10,001 (nearest logged point to u10,000):

| metric | C0 | T1 | Δ |
|---|---|---|---|
| online_eval/success_within_horizon_rate | 0.9551 | 0.6226 | −0.3325 |
| train/episode_success_rate | 0.9848 | 0.7302 | −0.2546 |
| behavior/absolute_completion | 0.9909 | 0.8094 | −0.1815 |
| behavior/dig_completion | 0.9969 | 0.8780 | −0.1189 |
| behavior/action_fraction/do | 0.1978 | 0.1333 | −0.0645 |
| behavior/action_fraction/no_op | 0.0779 | 0.1106 | +0.0328 |
| behavior/no_effect_action_rate | 0.1784 | 0.2905 | +0.1121 |
| behavior/mean_episode_length | 83.1 | 212.3 | +129.2 |
| reward/episode_return | 6.2382 | 3.1801 | −3.0581 |
| ppo/entropy | 1.2438 | 1.7063 | +0.4625 |
| ppo/value_loss | 0.0226 | 0.0612 | +0.0386 |
| ppo/explained_variance | 0.9923 | 0.9854 | −0.0069 |

`online_eval/termination_within_horizon_rate` equals
`success_within_horizon_rate` in both arms (every terminated episode is a
success; `completed_episode_success_rate` = 1.000 for both), so
"positive terminations" carries no information beyond success rate here.

Trajectory, 1k-update means:

| metric | arm | 0–1k | 1–2k | 2–3k | 3–4k | 4–5k | 5–6k | 6–7k | 7–8k | 8–9k | 9–10k | 10–11k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| online_eval success | C0 | 0.002 | 0.008 | 0.069 | 0.159 | 0.255 | 0.358 | 0.502 | 0.644 | 0.702 | 0.879 | 0.971 |
| online_eval success | T1 | 0.001 | 0.003 | 0.003 | 0.011 | 0.180 | 0.287 | 0.352 | 0.427 | 0.507 | 0.566 | 0.666 |
| dig_completion | C0 | — | 0.766 | 0.766 | 0.790 | 0.825 | 0.861 | 0.895 | 0.929 | 0.944 | 0.979 | 0.997 |
| dig_completion | T1 | — | 0.612 | 0.614 | 0.662 | 0.668 | 0.719 | 0.750 | 0.787 | 0.822 | 0.856 | 0.888 |
| action_fraction/do | C0 | — | 0.137 | 0.127 | 0.127 | 0.130 | 0.135 | 0.144 | 0.154 | 0.163 | 0.178 | 0.223 |
| action_fraction/do | T1 | — | 0.143 | 0.152 | 0.154 | 0.143 | 0.140 | 0.136 | 0.134 | 0.136 | 0.139 | 0.142 |
| no_effect_action_rate | C0 | — | 0.415 | 0.388 | 0.372 | 0.363 | 0.357 | 0.351 | 0.328 | 0.314 | 0.248 | 0.162 |
| no_effect_action_rate | T1 | — | 0.395 | 0.388 | 0.386 | 0.373 | 0.361 | 0.345 | 0.335 | 0.325 | 0.319 | 0.302 |
| ppo/entropy | C0 | 2.011 | 1.991 | 1.989 | 1.962 | 1.934 | 1.902 | 1.862 | 1.789 | 1.727 | 1.482 | 1.106 |
| ppo/entropy | T1 | 1.988 | 1.952 | 1.917 | 1.910 | 1.938 | 1.916 | 1.893 | 1.857 | 1.808 | 1.756 | 1.673 |

**Divergences between arms.**

1. **Delayed onset.** C0 leaves the floor at ~u1.5k; T1 stays flat until
   ~u4k. T1's `dig_completion` is already 0.15 below C0's at u1–2k, before
   either is completing.
2. **C0 undergoes a phase transition T1 has not entered.** Between u9k and
   u11k C0's DO fraction rises 0.178 → 0.223 (0.1348 early → 0.2693 at
   u11,871, 2.00×) while its no-effect rate collapses 0.248 → 0.162 → 0.081.
   T1's DO fraction is flat: u≤1000 mean 0.1380, u≥8000 mean 0.1411 (min
   0.1272), last 0.1531 — a 1.11× change, essentially none. T1's no-effect
   rate stays ~0.28–0.30.
3. **T1 has not lost dig propensity.** Its DO fraction never trends below its
   own early baseline (the run minimum 0.1199 occurs early at u221); the late
   dips are noise around a flat line. So the gate did not cause avoidance in
   the strict sense on the training distribution — it prevented the *escalation*
   in dig rate that C0 undergoes.
4. **Entropy and value loss.** T1 holds higher entropy (1.71 vs 1.24) and
   higher value loss (0.061 vs 0.023) at u10,001 — consistent with a policy
   still exploring rather than one that has converged onto a solution.

Open observation, not resolved here: T1's training `dig_completion` is 0.878
at u10,001 while its panel `dig_fraction` on the endpoint scope is 0.584. Two
confounds are entangled — stochastic-vs-argmax evaluation, and pooled training
maps vs held-out development maps — and this readout does not separate them.

---

## 5. Rule outcomes, stated mechanically

### 5.1 Mechanism check — **NOT EVALUABLE AS WRITTEN**

> "T1 invalid fresh-DO attempt fraction must fall ≥50% from its first
> evaluation."

| | u500 | u10,000 |
|---|---|---|
| invalid / all DO steps (preregistered denominator) | **0.0000** (0/16,746) | **0.0322** (129/4,009) |
| invalid / applicable DO steps | **0.0000** (0/242) | **0.1822** (129/708) |

The baseline is **0.0000** on both denominators. **A ≥50% fall from zero is
undefined.** Mechanically the fraction *rose*; the clause cannot be marked
satisfied or violated.

The naive reading — "it rose, therefore the clause contributing to the stop
condition is met" — is wrong. The rise accompanies the policy getting
dramatically better: applicable-DO share 1.45% → 17.66%, applicable dig poses
242 → 708, successes 1/176 → 68/176. T1 started encountering genuine alignment
decisions it previously never faced. And per §3.4 the zero is competence, not
incompetence: it had already refused 22,733 misaligned opportunities.

The accuracy change 242/242 = 100% → 579/708 = 81.8% is **not** a regression
either: the denominators are not comparable (242 rare events versus 708 diverse
ones). This is a composition effect.

**Substantive numbers in place of the broken clause:**

- T1 aligns correctly on **81.8% (579/708)** of its applicable fresh-dig
  attempts at u10,000;
- against the no-gate control at the same update, **16.7% (188/1,129)**.

**Proposed preregistration fix** (not applied retroactively to this readout):
replace the within-arm fall clause with either

- (i) *"alignment accuracy on applicable fresh-dig attempts must be at least
  X pp above the matched-update control arm"* — the contrast the two-arm design
  actually supports and which needs no within-arm trend; or
- (ii) if a within-arm trend is wanted, gate it on adequacy of the baseline:
  *"the baseline evaluation must contain at least N applicable fresh-dig
  attempts (N ≈ 500) and a non-zero invalid fraction before it may serve as a
  denominator; otherwise the clause is reported as not evaluable."*

### 5.2 Pilot stop rule — **DOES NOT FIRE**

> "Stop if T1 exact completion is more than 5 pp below C0 at **two successive**
> scheduled evaluations **AND** its invalid-DO attempt fraction has not fallen
> by at least half."

- Clause 1, at this single point: T1 raw exact completion is **50.00 pp** below
  C0 — far beyond the 5 pp threshold. But the rule requires **two successive**
  scheduled evaluations and **only u10,000 exists**. The second point is not
  yet available.
- Clause 2: **ill-posed** (§5.1).
- Conjunction: **not evaluable**. The stop rule **does not fire**.

Noted for whoever evaluates the second point: on the admissible endpoint the
sign reverses (+34.09 pp in T1's favour), so which quantity "exact completion"
denotes materially changes clause 1.

### 5.3 Code stop — **NO EVIDENCE OF VIOLATION**

> "Any invalid fresh DO mutates a trench target cell, or any matched
> relift/dump/non-trench transition differs."

From the differential gate-on/gate-off successor comparison on T1 u10,000
(every step re-executed from the same state with `enforce_trench_dig_alignment`
flipped, successor state pytrees compared leaf-by-leaf, excluding the
`env_cfg` subtree that carries the flag itself):

| divergence class | count |
|---|---|
| at invalid **applicable** DO steps (the intended treatment) | **129** |
| at non-DO steps | **0** |
| at valid DO steps (incl. relifts and inapplicable DOs) | **0** |
| at loaded DO steps (dumps) | **0** |
| invalid fresh DO that mutated a trench target cell | **0** |
| invalid DO steps with any effect at all | **0** |
| target map mutated, any slot | **false** |

One-to-one with the 129 invalid attempts: the gate fires on exactly the
intended transitions and nothing else. The panel receipts independently report
0 target mutations for both arms.

Scope limit: the probe rolls trench maps only, so the "non-trench excavation"
transition class is not covered here. Terra's
`terra/tests/test_trench_dig_alignment.py` covers mixed-map and pure
non-trench excavation.

---

## 6. Interpretation: is T1 learning to align, or merely blocked?

The framing was "(a) learning delay vs (b) junction infeasibility." Neither
alone fits; a third reading was tested and rejected; the evidence now supports
a fourth.

**(a) learning delay — partly true, insufficient.** T1's invalid attempts are
one heading bin from admissible (§3.3), not stuck. But T1 is already at 0.69%
dig-while-misaligned; there is little left to learn on *that* axis.

**(b) junction infeasibility — weak.** The axis-count split looks supportive:

| axis class (T1 u10,000) | conditions | applicable DO | invalid | invalid / applicable |
|---|---|---|---|---|
| 1-axis | 4 straight | 250 | 1 | 0.004 |
| 2-axis | seg2, tee ×2 | 143 | 0 | 0.000 |
| 3-axis | net3 ×3, seg3 | 315 | 128 | **0.406** |

— until the per-condition split shows **3 of the 4 three-axis conditions at
exactly 0.000**, with all 128 invalid attempts from `trn-net3-side2-s` alone.
"One condition," not "junctions as a class." Reporting only the axis-class row
would overstate the generality.

**(c) deterrence / learned timidity — tested and rejected.** T1 does 2.4× more
admissible digs per episode at u10,000 than u500, its training dig propensity
never falls, and the evidence originally offered for it was a dwell-weighted
artifact (§3.6).

**(d) the gate is working; the control was banking unrealizable completions —
best supported.** C0's 88.6% is achieved with 83.35% of applicable attempts
inadmissible and only 4.55% of episodes fully admissible. T1's alignment
competence is real, large, and gate-attributable (§3.2). The raw completion
gap is the price of the constraint, not collateral damage.

**What remains genuinely open.** Whether the completion cost is a transient
learning delay or persistent. T1 is still improving at u10,000
(online-eval 0.566 → 0.666 across the last two windows) and has 90,000 updates
of budget left. This readout cannot distinguish "T1 converges to C0-like
completion, admissibly, by u30k" from "T1 plateaus near 40%." Only more
training answers it.

**Leading hypothesis for the residual completion gap, not yet tested here:**
under the gate, continuing a trench past the stretch the current pose admits
requires relocating and re-yawing into a fresh admissible lane, which C0 never
has to do. This predicts the observed axis gradient (single-axis straights near
parity at −6.2 pp raw; junctions −56 to −88 pp) and stalling partway rather
than at the end (median failure dig_fraction 0.311). The decisive test is
per-finite-section attribution on stalled episodes — if one section is
near-complete while siblings are near-zero, the deficit is an unlearned
re-approach maneuver; if all sections sit near ~31%, it is general slowness.
Instrumentation for this was added to the probe
(`summary.section_completion`, using the terminal action map and the per-map
`trench_axis_membership` bitmask) but the measurement is **not** included in
this readout.

**Measurement that would settle (b) properly:** re-run
`tools/audit_trench_alignment_feasibility.py`'s strict-gate cover **conditioned
on the pose set the policy actually visits** (recoverable from the `.npz`
traces) rather than over all 12×12 poses. That answers whether an admissible
station was reachable from where the policy stood — the question the retracted
availability metric only appeared to answer.

---

## 7. Artifacts

Readout: `TRENCH_ALIGNMENT_PILOT_U10000_READOUT_20260821.md` (this file).

Receipts, `tools/trench_align_pilot_u10000_receipts/`:

| file | contents |
|---|---|
| `eval_c0_u010000_gate_main_dev.json` | C0 panel receipt, 608 per-map rows |
| `eval_t1_u010000_gate_main_dev.json` | T1 panel receipt, 608 per-map rows |
| `probe_{c0,t1}_u{000500,010000}.json` | mechanism summaries, per-condition, per-axis-class, per-slot |
| `probe_{c0,t1}_u{000500,010000}.npz` | full per-step traces (action, active, validity, applicability, raw yaw/standoff, dug cells, divergence) |

Tools:

- `scripts/trench_align_rollout_probe.py` in the **baselines** worktree
  (sha256 `67d75b8154407e5fa5ec7cf882b2869ed2560cc2d1420ff344532f9c3df29d8c`
  at the state used for the four cells; later amended to add section
  attribution, which was not used here);
- `tools/trench_align_pilot_readout.py` in this worktree — joins panel +
  probe + W&B receipts into one JSON.

Checkpoint identity: C0 `74848eab42bafaef…`, T1 `20fee53f34e0f86c…`
(sha256 as recorded in the panel receipts). The four copied `.pkl` files were
deleted after measurement.

## 8. Caveats and what was not measured

1. Both arms were still training; these are mid-run snapshots, not final.
2. One seed. The preregistration requires ≥3 matched seeds for promotion.
3. Development panel only; promotion and sealed panels untouched.
4. Deterministic (argmax) evaluation throughout. T1's behaviour differs
   markedly between argmax and sampling, so single-policy argmax numbers
   should not be read as the training-time distribution.
5. Probe vs panel differ by 2 episodes for C0 (batch-size-dependent step RNG).
6. Admissible completion uses Terra's gate as the admissibility oracle — a
   necessary but not sufficient proxy for ROS acceptance. **The deployment
   endpoint (ROS physical acceptance of raw plans) was not measured.**
7. Section-level attribution of stalled episodes: instrumented, not run.
8. Non-trench and foundation transition classes are not covered by the
   differential gate check.
9. Terra runtime is identical between the training pin
   (`a4b838b6cb894fdf982b614d4deea96f778fd7b0`) and the worktree HEAD used for
   evaluation (`dddfc8e0`) — the only diff is the preregistration markdown.
