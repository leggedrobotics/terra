# C0/T1 fresh-trench dig-alignment pilot — u85,000 readout

Date: 2026-08-25. Both arms have **ended**. Jobs 11152229 (C0) and 11152230
(T1) exited `TIMEOUT` at their 119:45 wall on 2026-08-25T19:41:01 with
`ExitCode 0:0`, C0 having reached u86,000 and T1 u85,441. A wall-time exit with
a valid checkpoint is `CONTINUABLE`, not a failure. Nothing is running.

Both arms are read at the **matched update u85,000**. C0's u86,000 checkpoint
exists and is used only for a C0-internal sanity check (§6.3), never for the
cross-arm comparison.

This readout reports numbers and states the preregistered rule outcomes
mechanically. It also gives a continuation recommendation, which the
preregistration does not cover.

---

## 0. Headline

Two things happened between u10,000 and u85,000, and they point the same way.

| endpoint (trench − net4, 176 slots) | C0 (gate off) | T1 (gate on) | Δ |
|---|---|---|---|
| **admissible** exact completion | **2/176 = 1.14%** | **110/176 = 62.50%** | **+61.36 pp** |
| raw strict exact completion | 167/176 = 94.89% | 111/176 = 63.07% | −31.82 pp |

against u10,000:

| endpoint | C0 u10k | T1 u10k | Δ u10k | C0 u85k | T1 u85k | Δ u85k | move |
|---|---|---|---|---|---|---|---|
| **admissible** | 4.55% | 38.64% | **+34.09** | **1.14%** | **62.50%** | **+61.36** | **+27.27 pp for T1** |
| raw | 88.64% | 38.64% | −50.00 | 94.89% | 63.07% | −31.82 | +18.18 pp for T1 |

1. **T1 closed 36% of the raw gap and did enter the phase transition it had
   not entered at u10,000.** Its commitment-escalation midpoint is u15,181
   against C0's u11,271 — T1 was simply 5,000 updates short of its own
   transition when the first readout was taken (§4).

2. **The control moved backwards on admissibility while moving forwards on raw
   completion.** C0's admissible exact completion **fell** 4.55% → 1.14% while
   its raw rose 88.64% → 94.89%; its invalid share of applicable fresh-dig
   attempts **rose** 0.8335 → 0.8608, and its admissible digs per episode fell
   1.07 → 0.90. Eight and a half times more training on a completion objective
   made the no-gate policy *less* physically admissible, not more.

The two arms still do not produce the same kind of output, so the −31.82 pp raw
figure is still not a like-for-like regression. Raw completion alone remains an
invalid promotion criterion.

---

## 1. Protocol actually executed

**Primary endpoint.** `eval_fixed_bank.py --panel-family gate_main
--accepted-panel development --terra-revision
a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 --horizon 450`, one invocation per
arm, joined externally. 608 slots, 38 conditions. Deterministic (argmax), seed
20260724, completion contract `exact_visible_dump_v1`. Endpoint scope = the 14
trench conditions minus the 3 net4 conditions = **11 conditions × 16 = 176
slots**. Identical to u10,000 in every argument.

Integrity, verified from the receipts:

| arm / update | ckpt sha256 | manifest sha256 | treatment fp | integrity | reset |
|---|---|---|---|---|---|
| C0 u10,000 | `74848eab42ba` | `1216bee3be9f` | `a68b6376c5a4ddbb` | all zero, passed | passed |
| C0 u85,000 | `1767873efbcc` | `1216bee3be9f` | `a68b6376c5a4ddbb` | all zero, passed | passed |
| T1 u10,000 | `20fee53f34e0` | `1216bee3be9f` | `85fe0198abd5f48a` | all zero, passed | passed |
| T1 u85,000 | `526607f70fa0` | `1216bee3be9f` | `85fe0198abd5f48a` | all zero, passed | passed |

- all four evaluations consumed **byte-identical maps**: the
  `reset_verification.layer_sha256` block hashes to `1613192908b5` in every
  one, and the manifest sha256 is equal across all four;
- 0 target mutations, 0 obstacle mutations, 0 mass-residual failures, 0
  non-finite states, 0 slot-index or termination disagreements, in all four;
- treatment fingerprints differ only in `enforce_trench_dig_alignment` and are
  unchanged from u10,000.

**Arm matching.** The W&B configs of the two runs differ in exactly four keys:
`enforce_trench_dig_alignment` (false / true), and the three bookkeeping fields
`config_name`, run `name`, and `checkpoint_dir`. Nothing else.

**Checkpoint provenance.** Read from the read-only archive
`/cluster/project/rsl/alesweber/terra_trench_align_v1_final/{c0,t1}/` and
verified against its `CHECKPOINT_SHA256.txt` after copying:
C0 u85,000 `1767873e…`, C0 u86,000 `2c8f3e6a…`, T1 u85,000 `526607f7…` — all
three match. The two u60,000 checkpoints used for the plateau curve (§6) were
copied from the Slurm run directory read-only and verified against a remote
`sha256sum`: C0 `20fcf209…`, T1 `6f13bb1c…`.

**Evaluation runtime is equivalent to the u10,000 readout's.** That readout ran
at Terra worktree `dddfc8e0`; the worktree is now `a7204ef5`. The whole diff
over `terra/` is confined to `terra/env_generation/partial_reset_bank.py` and
the partial-reset sidecar loader in `terra/maps_buffer.py` — another session's
work. `eval_fixed_bank.configure_for_bank` sets `config.partial_reset_root =
None` and `partial_reset_bank_sha256 = None`, and `maps_buffer` calls
`load_partial_reset_action_sidecars` only when `partial_reset_root is not
None`. The probe imports `configure_for_bank` from `eval_fixed_bank`, so both
measurement paths are covered. No changed line executes in either.

**Mechanism endpoint.** `scripts/trench_align_rollout_probe.py` (baselines
worktree, sha256 `67d75b8154407e5f…` — the same file, byte for byte, that
produced the u10,000 cells). Same fixed 176-slot trench−net4 map set, same seed
20260724, horizon 450, deterministic, `DATASET_SIZE` held at the full panel 608
so `exact_reset_keys` keeps its slot→map identity. `--differential-gate` on
T1. Its per-step assertion that its own transcription reproduces Terra's
exported `fresh_trench_dig_alignment_valid` / `..._yaw_error` /
`..._standoff_error` (atol 1e-5) was active and passed for every cell.

**Cross-validation against the panel.** The probe's independent episode
outcomes reproduce the panel endpoint **exactly** for C0 (167/176 both) and to
within 1 episode for T1 (probe 110 vs panel 111). The residual difference has
the same cause as at u10,000: per-env step RNG comes from
`jrandom.split(rng_step, count)` and `count` is 176 in the probe versus 608 in
the panel. Panel numbers are authoritative for the raw endpoint; probe numbers
are used self-consistently for anything trace-derived, including admissible
completion.

**New this round.** A net4-scoped probe pass (`--include-cell 'trn-net4-*'`)
per arm, to put an admissibility number on the conditions where C0's raw
advantage is largest (§2.4). Three further panel evaluations — both arms at
u60,000 and C0 at u86,000 — were run for the plateau assessment (§6) and are
not part of the matched-update endpoint.

---

## 2. Primary endpoint

### 2.1 By scope

| scope | C0 u10k | T1 u10k | Δ u10k | C0 u85k | T1 u85k | Δ u85k |
|---|---|---|---|---|---|---|
| **endpoint: trench − net4 (11 cond)** | 88.64% | 38.64% | −50.00 | **94.89%** | **63.07%** | **−31.82** |
| net4 (3 cond, preregistered exclusion) | 75.00% | 2.08% | −72.92 | 70.83% | 2.08% | −68.75 |
| foundation (24 cond, context only) | 3.12% | 1.04% | −2.08 | 14.84% | 4.17% | −10.68 |
| whole gate_main panel (38 cond) | 33.55% | 12.01% | −21.55 | 42.43% | 21.05% | −21.38 |

Macro and micro coincide on every scope because all conditions carry 16 slots.
Foundation remains context, not an endpoint — both arms are trench specialists
trained only on the 12 pooled trench conditions, and both improved on
foundation without ever seeing one, C0 more than T1.

`gate_main` numbers are **not** comparable to historical 45-condition
main-panel results.

### 2.2 By condition — raw

| condition | C0 u10k | T1 u10k | Δ u10k | C0 u85k | T1 u85k | Δ u85k | ΔΔ |
|---|---|---|---|---|---|---|---|
| trn-net3-side1-road | 9/16 | 0/16 | −56.2 | 15/16 | 7/16 | −50.0 | +6.2 |
| trn-net3-side2 | 15/16 | 5/16 | −62.5 | 15/16 | 12/16 | −18.8 | +43.8 |
| trn-net3-side2-s | 16/16 | 2/16 | −87.5 | 15/16 | 6/16 | −56.2 | +31.2 |
| trn-seg2-side2 | 16/16 | 7/16 | −56.2 | 16/16 | 9/16 | −43.8 | +12.5 |
| trn-seg3-side2 | 14/16 | 3/16 | −68.8 | 16/16 | 5/16 | −68.8 | +0.0 |
| trn-straight-altsides | 12/16 | 6/16 | −37.5 | 14/16 | 10/16 | −25.0 | +12.5 |
| trn-straight-side1 | 16/16 | 9/16 | −43.8 | 15/16 | 14/16 | −6.2 | +37.5 |
| trn-straight-side1-tight | 15/16 | 12/16 | −18.8 | 14/16 | 12/16 | −12.5 | +6.2 |
| trn-straight-side2 | 15/16 | 14/16 | −6.2 | 16/16 | 15/16 | −6.2 | +0.0 |
| trn-tee-side2 | 13/16 | 2/16 | −68.8 | 16/16 | 9/16 | −43.8 | +25.0 |
| trn-tee-side2-s | 15/16 | 8/16 | −43.8 | 15/16 | 12/16 | −18.8 | +25.0 |
| **TOTAL** | **156/176** | **68/176** | **−50.00** | **167/176** | **111/176** | **−31.82** | **+18.18** |

No condition's gap widened. T1 gained on 9 of 11 and held on 2
(`trn-seg3-side2`, `trn-straight-side2`); the latter was already at −6.2. The
gap now concentrates almost entirely in three geometries —
`trn-seg3-side2` (−68.8), `trn-net3-side2-s` (−56.2), `trn-net3-side1-road`
(−50.0) — which is exactly where T1's residual invalid attempts live (§3.3).

### 2.3 By condition — admissible

Admissible exact completion = the episode completed exactly **and** used only
pose-valid fresh trench digs. Probe-derived, so read against the probe's own
episode outcomes (C0 167, T1 110).

| condition | C0 adm u85k | T1 adm u85k | Δ | (C0 adm u10k) | (T1 adm u10k) |
|---|---|---|---|---|---|
| trn-net3-side1-road | 0/16 | 7/16 | +43.8 | 0/16 | 0/16 |
| trn-net3-side2 | 0/16 | 12/16 | +75.0 | 0/16 | 5/16 |
| trn-net3-side2-s | 0/16 | 6/16 | +37.5 | 0/16 | 2/16 |
| trn-seg2-side2 | 0/16 | 9/16 | +56.2 | 0/16 | 7/16 |
| trn-seg3-side2 | 0/16 | 5/16 | +31.2 | 0/16 | 3/16 |
| trn-straight-altsides | 0/16 | 10/16 | +62.5 | 1/16 | 6/16 |
| trn-straight-side1 | 1/16 | 14/16 | +81.2 | 1/16 | 9/16 |
| trn-straight-side1-tight | 0/16 | 12/16 | +75.0 | 0/16 | 12/16 |
| trn-straight-side2 | 1/16 | 15/16 | +87.5 | 6/16 | 14/16 |
| trn-tee-side2 | 0/16 | 8/16 | +50.0 | 0/16 | 2/16 |
| trn-tee-side2-s | 0/16 | 12/16 | +75.0 | 0/16 | 8/16 |
| **TOTAL** | **2/176 = 1.14%** | **110/176 = 62.50%** | **+61.36** | **8/176** | **68/176** |

C0's two remaining admissible successes are one each on `trn-straight-side1`
and `trn-straight-side2` — both single-axis straights, as at u10,000, but four
fewer of them. **C0 is now at exactly zero admissible completion on 9 of the 11
endpoint conditions.**

Graded rather than all-or-nothing: C0's admissible share of all its fresh digs
fell from 16.65% to **13.92%** (159 admissible vs 983 inadmissible), and the
median count of inadmissible digs inside a *successful* C0 episode is still
**6**. 172 of its 176 episodes contain at least one inadmissible dig (was 166).

**Caveats on the admissible endpoint** (unchanged from u10,000, restated
because they still bind):

1. T1 satisfies it *by construction* — the gate makes an inadmissible dig
   impossible, so T1 cannot fail this criterion. The metric does not measure
   T1's skill; it measures **how much of C0's raw advantage is realizable**,
   and that answer got worse, not better, with more training.
2. Terra-gate-admissible is **necessary, not sufficient** for ROS physical
   acceptance: Terra's cone is discrete, the `CABIN_CONTROL` offset (−0.274 m,
   under half a cell) is deliberately not modelled, and there is no swept-path
   check.

### 2.4 net4: the raw advantage is entirely unrealizable

The 3 net4 conditions are the preregistered exclusion — the 2,400-map preflight
found their maps have no complete strict-gate cover, tolerance-independently.
A net4-scoped probe pass now puts a number on what C0's 70.83% there is worth:

| net4 (3 cond, 48 slots) | C0 u85k | T1 u85k |
|---|---|---|
| raw exact | 34/48 = 70.83% | 1/48 = 2.08% |
| **admissible exact** | **0/48 = 0.00%** | **1/48 = 2.08%** |
| fresh digs admissible / inadmissible | 32 / 311 | 81 / 0 |
| invalid share of applicable attempts | 0.9067 | 0.4564 |

**Every one of C0's 34 net4 completions used at least one pose-invalid dig.**
The preflight's geometric prediction and the policy's behaviour agree.

### 2.5 Failure shape

| trench − net4 | C0 u10k | T1 u10k | C0 u85k | T1 u85k |
|---|---|---|---|---|
| failures | 20/176 | 108/176 | 9/176 | 65/176 |
| median dig_fraction of failures | 0.675 | 0.311 | 0.725 | 0.266 |
| failures ≥ 90% dug | 5/20 | 3/108 | 1/9 | 4/65 |
| all failures at the 450 horizon | 20/20 | 108/108 | 9/9 | 65/65 |
| mean dig_fraction (all) | 0.9549 | 0.5841 | 0.9752 | 0.7520 |
| mean episode steps | 94.3 | 294.8 | 64.6 | 197.4 |
| no-effect actions / episode | 11.9 | 102.5 | 2.0 | 27.2 |

T1's mean dig fraction rose sharply (0.584 → 0.752) and its no-effect actions
per episode fell 3.8×. But its *failures* still stall around a quarter dug —
the median failure dig fraction went **down**, 0.311 → 0.266. The 65 episodes
T1 still fails are not near-misses and are not horizon-limited; every one of
them runs to 450 steps having stopped making progress. "Needs a longer
horizon" remains ruled out.

---

## 3. Mechanism endpoint

### 3.1 The 2×3

All cells: same 176 maps, same seed, 450 steps, deterministic.

| cell | update | gate | succeeded | applicable states | aligned occupancy | invalid attempts | invalid / applicable | DO-rate while misaligned | admissible digs/ep |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 500 | off | 0/176 | 3,583 | 13.1% | 442 | **0.9888** | 0.1419 | 0.03 |
| C0 | 10,000 | off | 154/176 | 5,937 | 18.1% | 941 | **0.8335** | 0.1936 | 1.07 |
| C0 | 85,000 | off | 167/176 | 4,916 | 13.3% | 983 | **0.8608** | 0.2306 | 0.90 |
| T1 | 500 | on | 1/176 | 24,340 | 6.6% | 0 | **0.0000** | 0.0000 | 1.38 |
| T1 | 10,000 | on | 68/176 | 31,913 | 41.5% | 129 | **0.1822** | 0.0069 | 3.29 |
| T1 | 85,000 | on | 110/176 | 22,721 | 22.7% | 350 | **0.3293** | 0.0199 | 4.05 |

"Applicable state" = active, empty excavator, prospective DO would remove fresh
trench soil. "Aligned occupancy" = fraction of those states that are pose-valid.

**Two of these columns are dwell-weighted; read them with care.** "Aligned
occupancy" and "DO-rate while misaligned" both have a *number of steps* in the
denominator, so they inherit exactly the endogeneity the handover flags for
attempt rates: a policy that reaches a pose and acts spends fewer steps in that
pose than one that dawdles, and digging ends the state it is counted in. T1's
episodes shortened from 295 to 199 steps and its applicable-state count fell
29%, so the 41.5% → 22.7% occupancy fall does **not** license a "T1 got worse at
reaching admissible poses" reading, and the 0.0069 → 0.0199 rise in DO-rate
while misaligned is partly the same arithmetic running the other way. They are
reported because they were reported at u10,000 and because the cross-arm
contrast at a *matched* update is still meaningful (both arms face the same 176
maps); they are not used as within-arm trend evidence.

The columns without a dwell denominator all move in T1's favour: invalid share
of applicable **attempts** (a per-decision accuracy, not per-step) and
admissible digs per episode, 3.29 → **4.05**. So does the fraction of episodes
containing at least one admissible fresh dig: 0.756 (u500) → 0.801 (u10k) →
**0.847** (u85k). Where the dwell-weighted and dwell-free measures disagree, the
dwell-free measures are the evidence.

*(That episode fraction is computed here as "≥ 1 admissible fresh dig in the
episode", `dug > 0`. It is not the same statistic as the u10,000 readout's
98.5% → 83.4% figure, which used a different predicate; the two are not
comparable and only the trajectory within this definition is used.)*

### 3.2 The control answers the causal question — more sharply than at u10,000

**Alignment competence is not learned incidentally, and it is not even
stationary.** Over 84,500 updates without the gate, C0's invalid share of
applicable fresh-dig attempts went 0.9888 → 0.8335 → **0.8608**: it improved
slightly in the first 10k and then **regressed**. That quantity is a
per-decision accuracy with no dwell denominator, so the within-arm sequence is
readable. Two dwell-free episode-level quantities agree with it: C0's
admissible digs per episode fell 1.07 → **0.90**, and its admissible exact
completion fell by a factor of four, 4.55% → **1.14%**. (Its DO-rate while
misaligned also rose, 0.1419 → 0.1936 → 0.2306, but that statistic is
dwell-weighted and is not what carries the claim.)

Terra's completion objective does not reward alignment, and at convergence it
actively selects against it: the fastest way to finish a trench is to dig from
wherever you are standing. C0's mean episode length fell from 94 to 65 steps
over the same interval.

With the gate, at matched u85,000:

- invalid share of applicable attempts **0.8608 → 0.3293**, a **62% relative
  reduction**;
- rate of digging while misaligned **0.2306 → 0.0199**, an **11.6× reduction**;
- inadmissible digs executed: **983 → 0** (by construction);
- admissible digs per episode **0.90 → 4.05**, a **4.5× increase**.

### 3.3 Raw yaw / standoff, and the invalid tail

Diagnostic-section raw physical quantities. Reported because the *exported*
standoff error is clipped to exactly 0 whenever the pose is in band.

| cell | successful fresh digs | raw yaw (deg) mean / p90 | raw standoff (m) mean / p10 / p90 |
|---|---|---|---|
| C0 u500 | 5 | 0.0040 / 0.0119 | 4.680 / 3.938 / 5.567 |
| C0 u10,000 | 188 | 0.0018 / 0.0000 | 4.810 / 4.132 / 5.434 |
| C0 u85,000 | 159 | 0.0017 / 0.0000 | 4.678 / 3.956 / 5.436 |
| T1 u500 | 242 | 0.0021 / 0.0178 | 4.699 / 3.760 / 5.640 |
| T1 u10,000 | 579 | 0.0015 / 0.0000 | 4.971 / 4.280 / 5.637 |
| T1 u85,000 | 713 | 0.0016 / 0.0000 | 5.006 / 4.320 / 5.526 |

As at u10,000, admitted digs are essentially perfectly parallel in every cell
and the standoff never binds. This endpoint has no headroom in this setup — 12
base headings 30° apart against a 15° tolerance — and **should not be read as a
trend**. The preregistration's "raw successful fresh-dig yaw/standoff" clause
remains vacuous for the reason recorded in the handover §6.

The **invalid** attempts are the informative tail:

| cell | invalid | raw yaw (deg) mean / median / min / max | raw standoff (m) mean / p10 / p90 | yaw-only | standoff-only | both |
|---|---|---|---|---|---|---|
| C0 u10,000 | 941 | 32.26 / 30.00 / 0.00 / 90.00 | 2.544 / 0.291 / 4.869 | 315 | 275 | 351 |
| C0 u85,000 | 983 | 32.11 / 30.00 / 0.00 / 90.00 | 2.813 / 0.314 / 5.035 | 395 | 248 | 340 |
| T1 u10,000 | 129 | 29.77 / 30.00 / 0.00 / 30.00 | 3.970 / 3.974 / 3.974 | **128** | 1 | 0 |
| T1 u85,000 | 350 | **60.00 / 60.00 / 59.99 / 60.00** | 4.686 / 3.590 / 5.351 | **349** | 0 | 1 |

Two findings.

**C0's inadmissible digs are dangerous in the standoff dimension, not only the
yaw one.** 588 of its 983 involve a standoff violation, at a mean standoff of
2.81 m against a 3.5 m minimum and a p10 of 0.31 m — the machine digging
essentially on top of the trench line. This is the failure mode the gate exists
to prevent, and it is unchanged in character from u10,000.

**T1's residual invalid attempts moved from one heading bin off axis to two.**
At u10,000 they were a spike at exactly 30.00°; at u85,000 they are a spike at
exactly **60.00°** (min 59.99, max 60.00, all 350), with standoff comfortably in
band. The diagnostic axis is the *best-scoring* of the offending sections, so
this is not a selection artifact: the most favourable owning section is two
heading bins away.

**This retires the main evidence for the broad-to-strict tolerance
curriculum.** The handover's next-action #3 rested on "128 of T1's 129 invalid
attempts are yaw-only failures at exactly 30.00° — one heading bin from
admissible", which motivated a broad early tolerance annealing to 15°. That
observation was made at u10,000 and does **not** survive to u85,000: admitting
these attempts would require a tolerance above 60°, i.e. a quarter turn, which
is not a relaxation of the constraint but its abolition. The curriculum may
still be worth testing for its exploration benefit early in training, but the
"one bin away" argument for it is gone.

### 3.4 Per-condition and per-axis-class concentration

| axis class | C0 u85k appl. DO / invalid / ratio | T1 u85k appl. DO / invalid / ratio |
|---|---|---|
| 1-axis (4 straight) | 388 / 347 / **0.894** | 258 / 0 / **0.000** |
| 2-axis (seg2, tee ×2) | 310 / 255 / **0.823** | 254 / 58 / **0.228** |
| 3-axis (net3 ×3, seg3) | 444 / 381 / **0.858** | 551 / 292 / **0.530** |

C0's misalignment is now uniform across axis classes (0.82–0.89) and across all
eleven conditions (0.75–0.94). It is systematic, not geometry-driven.

T1's is highly concentrated. **Eight of the eleven conditions are at exactly
0.0000 invalid.** All 350 invalid attempts come from three:

| T1 u85,000 | succ | applicable DO | invalid | invalid / applicable |
|---|---|---|---|---|
| trn-net3-side2-s | 6/16 | 279 | 243 | **0.871** |
| trn-seg2-side2 | 9/16 | 126 | 58 | 0.460 |
| trn-seg3-side2 | 5/16 | 110 | 49 | 0.446 |
| the other 8 conditions | 90/128 | 548 | **0** | **0.000** |

At u10,000 the concentration was even tighter — 128 of 129 from
`trn-net3-side2-s` alone. It has broadened from one condition to three, and
`trn-net3-side2-s` still supplies 69% of the total. The "junctions as a class"
reading is still not supported: two of the four 3-axis conditions
(`trn-net3-side2`, `trn-net3-side1-road`) are at exactly zero, and one of the
three 2-axis conditions accounts for all the 2-axis invalids.

### 3.5 Section attribution

Per-finite-section completion at the terminal state, attributing dug cells to
owning sections via the per-map `trench_axis_membership` bitmask, restricted to
**stalled episodes with ≥ 2 generated sections**. Same 176 slots, seed, horizon;
computed inside the same probe pass as the mechanism numbers.

| stalled multi-section episodes | C0 u10k | T1 u10k | C0 u85k | T1 u85k |
|---|---|---|---|---|
| n | 16 | 85 | **4** | 53 |
| mean best section | 0.763 | 0.572 | 0.865 | 0.491 |
| mean worst section | 0.394 | 0.101 | 0.518 | 0.168 |
| median spread (best − worst) | 0.320 | 0.538 | 0.335 | 0.290 |
| worst section exactly 0.000 | 4/16 (25%) | 53/85 (62%) | 1/4 | **33/53 (62%)** |
| spread < 0.2 ("general slowness") | 5/16 | 25/85 (29%) | 0/4 | **25/53 (47%)** |

Every successful episode in both arms at both updates finishes every section
(mean best = mean worst = 1.000, spread 0.000), as the exact-completion contract
requires.

The u10,000 reading was "an unlearned re-approach maneuver": T1 works one
section and leaves a sibling untouched. That signature **persists at exactly the
same rate** — 62% of stalled multi-section episodes have a completely untouched
sibling, identical to u10,000 — but the population around it changed. The
median spread fell 0.538 → 0.290 and the share of stalled episodes that look
like ordinary general slowness rose 29% → 47%. So T1's remaining failures are
now a *mixture*: about half still show the untouched-sibling signature, and
about half are undifferentiated partial progress.

C0's stalled population has collapsed to n = 4 and cannot support a comparison
at this update.

### 3.6 Retracted measurements — still retracted

The three measurements withdrawn during the u10,000 analysis are not
resurrected here, and the reasons are unchanged:

1. **`pose_valid_axis_available_at_applicable_do`** — algebraically identical
   to the gate decision (`invalid ⟺ applicable ∧ pose_valid_axis_count == 0`),
   so its value is just `1 − invalid/applicable` restated. The probe still
   emits the field; it is retained in the receipts and read nowhere.
2. **Pooled per-step attempt rate at opportunity** — dwell-weighted, and
   contaminated by absorbing `DO_NOTHING` loops in timed-out episodes. **No
   per-step attempt rate is used as willingness evidence anywhere in this
   readout.** The same objection is extended in §3.1 to aligned occupancy and
   to the DO-rate-while-misaligned column, both of which count steps in their
   denominator and are therefore dwell-weighted for the same reason.
3. **The deterrence hypothesis (c)** — refuted at u10,000 by T1's rising
   admissible digs per episode, and refuted more strongly here: 1.38 → 3.29 →
   **4.05**, with the episode-level fraction that ever digs admissibly also
   rising monotonically. T1 never lost dig propensity.

---

## 4. Is the completion cost transient? — the pilot's central open question

At u10,000 T1 trailed C0 by 50 pp raw, led by 34 pp admissible, and had **not**
entered the commitment-escalation phase transition C0 underwent between u7k and
u12k. With 8.5× more training the answer has three parts.

### 4.1 T1 did enter the transition — about 4,000 updates later

Escalation measured on `behavior/action_fraction/do`, 25-point smoothed,
baseline = mean over u ≤ 2,000, plateau = mean over u ≥ 80,000:

| arm | baseline | plateau | ratio | 10% crossing | **50% crossing** | 90% crossing |
|---|---|---|---|---|---|---|
| C0 | 0.1361 | 0.3579 | **2.63×** | u7,991 | **u11,271** | u20,101 |
| T1 | 0.1408 | 0.2292 | **1.63×** | u2,121 | **u15,181** | u28,401 |

and on `behavior/no_effect_action_rate`:

| arm | base | floor | crosses halfway | crosses 90% of range |
|---|---|---|---|---|
| C0 | 0.4153 | 0.0040 | u9,961 | u12,891 |
| T1 | 0.3961 | 0.0256 | u13,721 | u20,011 |

T1's DO fraction went 0.137 (u5–10k) → 0.158 → 0.199 → 0.210 → 0.229, and its
no-effect rate collapsed 0.337 → 0.250 → 0.118 → 0.058 → 0.027. **The u10,000
readout caught T1 roughly 5,000 updates before its own midpoint**, which is why
it read as a flat line then. The transition is real, delayed by ~4k updates at
the midpoint and ~8k at the 90% point, and **smaller in amplitude**: T1
escalates 1.63× where C0 escalates 2.63×, ending at a DO fraction of 0.229
against C0's 0.358. That residual difference is the gate doing its job — a
fraction of C0's DO actions are inadmissible digs T1 is not permitted to make.

### 4.2 The raw gap closed on the training distribution, only partly on held-out maps

| gap (T1 − C0) | at u10,001 | at u85,001 | closure |
|---|---|---|---|
| `online_eval/success_within_horizon_rate` | −0.3325 | **−0.0205** | 94% |
| `train/episode_success_rate` | −0.2546 | **−0.0129** | 95% |
| `behavior/absolute_completion` | −0.1815 | −0.0046 | 97% |
| `behavior/dig_completion` | −0.1189 | −0.0019 | 98% |
| **held-out panel, raw exact** | **−50.00 pp** | **−31.82 pp** | **36%** |

On the maps it trains on, T1 has essentially caught up: 0.9790 vs 0.9995 online
eval, 0.9865 vs 0.9995 train success. On the held-out development panel it has
closed just over a third of the gap. **The completion cost is partly transient
and partly not**, and the split is a generalization gap, not a learning-speed
gap: the gated policy fits its training distribution nearly as well as the
control and transfers it markedly less well.

The held-out gap's own trajectory is −50.00 pp (u10k) → **−36.36 pp** (u60k) →
−31.82 pp (u85k). Most of the closure happened before u60,000; the last 4.5 pp
came roughly two-thirds from T1 gaining (+2.84 pp) and one-third from C0 giving
back (−1.70 pp) (§6.2), and both movements are inside the panel's slot churn
(§6.3). Treat "the gap is still closing" as unsupported past u60,000.

### 4.3 The admissible gap widened

+34.09 pp → **+61.36 pp** in T1's favour. It widened for two independent
reasons: T1 improved (38.64% → 62.50%) *and* C0 deteriorated (4.55% → 1.14%).

**Answer to the central question, in one line:** the raw completion cost is
substantially but not fully transient (36% recovered on held-out maps, ~95% on
the training distribution), while the admissible benefit is not transient at
all and grew by 27 pp — because the control, given more training, converts its
completion advantage further into inadmissible digs.

### 4.4 Matched W&B, u85,001

| metric | C0 | T1 | Δ | (Δ at u10,001) |
|---|---|---|---|---|
| `online_eval/success_within_horizon_rate` | 0.9995 | 0.9790 | −0.0205 | −0.3325 |
| `train/episode_success_rate` | 0.9995 | 0.9865 | −0.0129 | −0.2546 |
| `behavior/absolute_completion` | 0.9995 | 0.9948 | −0.0046 | −0.1815 |
| `behavior/dig_completion` | 1.0000 | 0.9981 | −0.0019 | −0.1189 |
| `behavior/action_fraction/do` | 0.3546 | 0.2266 | −0.1280 | −0.0645 |
| `behavior/no_effect_action_rate` | 0.0062 | 0.0288 | +0.0225 | +0.1121 |
| `behavior/mean_episode_length` | 34.9 | 50.9 | +15.9 | +129.2 |
| `train/episode_timeout_rate` | 0.0005 | 0.0135 | +0.0129 | +0.2546 |
| `reward/episode_return` | 6.7918 | 6.5518 | −0.2400 | −3.0581 |
| `ppo/entropy` | 0.0647 | 0.1780 | +0.1133 | +0.4625 |
| `ppo/value_loss` | 0.0082 | 0.0031 | −0.0051 | +0.0386 |
| `ppo/explained_variance` | 0.9276 | 0.9994 | +0.0719 | −0.0069 |

Both entropy coefficients have annealed to their 0.02 floor. C0's policy
entropy has collapsed to 0.065 — effectively deterministic — while T1 holds
0.178. T1's value function is now the *better* fit of the two (explained
variance 0.9994 vs 0.9276, value loss 0.0031 vs 0.0082), a reversal of the
u10,000 picture.

Data note, still true: W&B serialises `NaN` as the **string** `"NaN"`, which
`float()` silently converts into a real NaN and poisons every downstream mean
and least-squares fit. `tools/trench_align_pilot_readout.py:wandb_series` now
drops non-finite values at ingest.

---

## 5. Preregistered rule outcomes, stated mechanically

### 5.1 Mechanism check — **STILL NOT EVALUABLE AS WRITTEN**

> "T1 invalid fresh-DO attempt fraction must fall ≥50% from its first
> evaluation."

| | u500 | u10,000 | u85,000 |
|---|---|---|---|
| invalid / all DO steps (preregistered denominator) | **0.0000** (0/16,746) | 0.0322 (129/4,009) | **0.1180** (350/2,966) |
| invalid / applicable DO steps | **0.0000** (0/242) | 0.1822 (129/708) | **0.3293** (350/1,063) |

The baseline is **0.0000** on both denominators. A ≥50% fall from zero is
undefined; the clause can be marked neither satisfied nor violated.
Mechanically the fraction *rose again*.

**What the evidence actually supports, as opposed to the rule.** The naive
reading — "it rose twice, so the clause contributing to the stop condition is
met" — is wrong for the reason recorded in the handover §6, and the u85,000
data make it wronger. The zero at u500 was **competence**: T1 was in a
misaligned applicable state 22,733 times across 161/176 episodes and chose DO
in exactly zero of them, while digging at a 15.1% rate when aligned. It had
mastered the constraint by 32.8M transitions. The subsequent rise accompanies
the policy getting dramatically better at everything else: applicable DO poses
242 → 708 → 1,063, successes 1/176 → 68/176 → 110/176, admissible digs per
episode 1.38 → 3.29 → 4.05.

The measurement the two-arm design actually supports is the cross-arm contrast
at a matched update, which needs no within-arm trend:

| alignment accuracy on applicable fresh-dig attempts | u10,000 | u85,000 |
|---|---|---|
| T1 | 81.8% (579/708) | **67.1% (713/1,063)** |
| C0 (matched-update control) | 16.7% (188/1,129) | **13.9% (159/1,142)** |
| T1 − C0 | +65.1 pp | **+53.2 pp** |

Both arms' accuracy fell between u10,000 and u85,000, and the gap narrowed by
12 pp while remaining enormous. Note that the denominators are not comparable
across updates — 708 versus 1,063 applicable poses, from policies that reach
very different states — so the within-arm fall is a composition effect and not
a regression claim. The **cross-arm** contrast at each fixed update is the
supported statement.

**The preregistration fix proposed at u10,000 should now be applied.** Replace
the within-arm fall clause with (i) *"alignment accuracy on applicable
fresh-dig attempts must be at least X pp above the matched-update control
arm"*, or (ii) gate the within-arm trend on baseline adequacy: *"the baseline
evaluation must contain at least N ≈ 500 applicable fresh-dig attempts and a
non-zero invalid fraction before it may serve as a denominator; otherwise the
clause is reported as not evaluable."* This is the second consecutive readout
blocked on it.

### 5.2 Pilot stop rule — **DOES NOT FIRE**

> "Stop if T1 exact completion is more than 5 pp below C0 at **two successive**
> scheduled evaluations **AND** its invalid-DO attempt fraction has not fallen
> by at least half."

u85,000 is the second scheduled evaluation, so the two-evaluation clause can
finally be assessed.

**Mechanically:**

- **Clause 1, on raw exact completion: SATISFIED.** T1 is 50.00 pp below C0 at
  u10,000 and 31.82 pp below at u85,000; both exceed the 5 pp threshold, at two
  successive scheduled evaluations.
- **Clause 1, on admissible exact completion: NOT SATISFIED.** T1 is 34.09 pp
  *above* C0 at u10,000 and 61.36 pp above at u85,000. Which quantity "exact
  completion" denotes flips this clause's sign at both points.
- **Clause 2: ILL-POSED** (§5.1). The baseline is exactly zero, so "fallen by
  at least half" has no value.
- **Conjunction: NOT EVALUABLE.** The rule is a conjunction and one conjunct
  has no truth value. **The stop rule does not fire.**

**What the evidence supports, separately.** Even if clause 2 were repaired so
that the conjunction evaluated, stopping would be the wrong call on this
evidence, for three reasons:

1. The clause-1 trigger is entirely an artifact of reading "exact completion"
   as raw. On the quantity that survives contact with a real machine —
   completion using only physically admissible digs — T1 leads by 61 pp and the
   lead is growing.
2. The gap that does exist **closed** between the two evaluation points:
   −50.00 → −31.82 pp, with no condition widening and 9 of 11 improving. A stop
   rule designed to catch a treatment that is failing is being triggered by one
   that converged. (It has stopped closing since u60,000 — §4.2, §6.2 — which is
   an argument against *more updates*, not an argument for abandoning the
   treatment.)
3. The stop rule's prescribed next step is the broad-to-strict tolerance
   curriculum, and §3.3 has just removed that remedy's main supporting
   observation. Firing the rule would route to an action the u85,000 evidence
   no longer indicates.

### 5.3 Code stop — **NO EVIDENCE OF VIOLATION**

> "Any invalid fresh DO mutates a trench target cell, or any matched
> relift/dump/non-trench transition differs."

From the differential gate-on/gate-off successor comparison on T1 u85,000
(every step re-executed from the same state with
`enforce_trench_dig_alignment` flipped, successor state pytrees compared
leaf-by-leaf, excluding the `env_cfg` subtree that carries the flag itself):

| divergence class | T1 u500 | T1 u10,000 | T1 u85,000 |
|---|---|---|---|
| at invalid **applicable** DO steps (the intended treatment) | 0 | 129 | **350** |
| at non-DO steps | 0 | 0 | **0** |
| at valid DO steps (incl. relifts and inapplicable DOs) | 0 | 0 | **0** |
| at loaded DO steps (dumps) | 0 | 0 | **0** |
| invalid fresh DO that mutated a trench target cell | 0 | 0 | **0** |
| invalid DO steps with any effect at all | 0 | 0 | **0** |
| target map mutated, any slot | false | false | **false** |
| total divergence steps | 0 | 129 | **350** |

One-to-one with the 350 invalid attempts: the gate fires on exactly the
intended transitions and nothing else, at 2.7× the u10,000 event count. The
panel receipts independently report 0 target mutations for both arms.

Scope limit, unchanged: the probe rolls trench maps only, so the "non-trench
excavation" transition class is not covered here.
`terra/tests/test_trench_dig_alignment.py` covers mixed-map and pure
non-trench excavation.

### 5.4 Preflight stop and promotion stop

Preflight stop: not re-evaluated; the bank was untouched this round.

Promotion stop (≥3 matched seeds, seed-stratified paired bootstrap, 95%
LCB(T1−C0 exact) > −2 pp **and** 95% LCB(raw ROS physical acceptance) > 0):
**cannot be assessed.** One seed exists, so the seed-stratified bootstrap the
clause specifies has no strata; and ROS physical acceptance has never been
measured, so its bound has no data at all. The *point estimates* on the single
available seed are −31.82 pp on raw completion and +61.36 pp on admissible
completion — but a point estimate is not a lower confidence bound and neither
number can be substituted for the rule. The clause's second conjunct, the ROS
acceptance bound, remains the load-bearing one and is the single largest
unmeasured quantity in this pilot.

---

## 6. Plateau assessment and continuation recommendation

14,000 updates remain to the u100,000 target. Observed throughput is 86,000
updates in 119:31 = **719.6 updates/hour**, so the remainder is **≈19.5 h** of
a fresh 120 h allocation — the segment would be ~84% idle unless the target is
raised.

### 6.1 Training-distribution metrics are flat

Linear fit over the tail u65,000–u85,441, with the residual scatter it was
fitted through and the change it projects over the remaining 14,000 updates:

| metric | arm | slope per 10k | residual sd | u65–70k | u80k+ | projected over 14k |
|---|---|---|---|---|---|---|
| `train/episode_success_rate` | C0 | +0.00009 | 0.0003 | 0.9994 | 0.9996 | +0.0001 |
| `train/episode_success_rate` | T1 | +0.00061 | 0.0013 | 0.9829 | 0.9840 | **+0.0009** |
| `online_eval/success_within_horizon_rate` | C0 | +0.00021 | 0.0007 | 0.9992 | 0.9995 | +0.0003 |
| `online_eval/success_within_horizon_rate` | T1 | +0.00020 | 0.0031 | 0.9816 | 0.9823 | +0.0003 |
| `behavior/absolute_completion` | C0 | +0.00006 | 0.0002 | 0.9997 | 0.9998 | +0.0001 |
| `behavior/absolute_completion` | T1 | +0.00117 | 0.0010 | 0.9906 | 0.9921 | **+0.0016** |
| `behavior/dig_completion` | T1 | −0.00029 | 0.0004 | 0.9978 | 0.9974 | −0.0004 |
| `behavior/mean_episode_length` | T1 | −0.50250 | 0.598 | 52.49 | 51.69 | −0.70 |
| `reward/episode_return` | T1 | +0.00911 | 0.0151 | 6.5095 | 6.5244 | +0.013 |

5k-window means for T1's `online_eval/success_within_horizon_rate` from u30k
on: 0.979, 0.978, 0.979, 0.981, 0.980, 0.981, 0.982, 0.982, 0.982, 0.982,
0.982, 0.981. That is twelve consecutive windows inside a 0.004 band, spanning
55,400 updates. C0's are 0.999 throughout.

T1's largest remaining trend is `behavior/absolute_completion` at +0.0012 per
10k updates, which projects to **+0.0016 over the entire remaining budget** —
about one part in six hundred, and smaller than the scatter of a single 5k
window. C0's largest is an order of magnitude below that. **Both arms are flat
on the training distribution and have been since roughly u30,000.**

### 6.2 Held-out panel trajectory

Training-distribution flatness does not settle held-out behaviour, so the same
`gate_main/development` panel was run on the u60,000 checkpoint of each arm,
giving a three-point held-out curve.

| arm | update | **endpoint (trench − net4)** | net4 | foundation | whole panel |
|---|---|---|---|---|---|
| C0 | 10,000 | 156/176 = 88.64% | 75.00% | 3.12% | 33.55% |
| C0 | 60,000 | **170/176 = 96.59%** | 79.17% | 20.83% | 47.37% |
| C0 | 85,000 | 167/176 = 94.89% | 70.83% | 14.84% | 42.43% |
| T1 | 10,000 | 68/176 = 38.64% | 2.08% | 1.04% | 12.01% |
| T1 | 60,000 | **106/176 = 60.23%** | 4.17% | 5.47% | 21.22% |
| T1 | 85,000 | 111/176 = 63.07% | 2.08% | 4.17% | 21.05% |

Because all three evaluations use the identical 176 slots with identical frozen
reset seeds, the right instrument is the **paired** slot-level flip count, not
the marginal rate:

| arm | u10,000 → u60,000 | u60,000 → u85,000 | exact McNemar on the tail |
|---|---|---|---|
| C0 | +16 / −2, net **+14** | +5 / −8, net **−3** | p = 0.581 |
| T1 | +47 / −9, net **+38** | +18 / −13, net **+5** | p = 0.473 |

**Neither arm's held-out endpoint is improving detectably over its last 25,000
updates.** C0 peaked at u60,000 and gave back 3 slots. T1 gained 5 net slots out
of 31 that flipped in either direction — the churn dominates the net movement,
and the sign is not distinguishable from a coin flip. The bulk of both arms'
held-out progress happened before u60,000; T1's is later and larger (+38 slots
against C0's +14) but it too has run out by u60,000.

Extrapolating T1's u60k→u85k rate (+2.84 pp over 25,000 updates) over the
remaining 14,000 gives **+1.6 pp**, i.e. 63.07% → ~64.7%, against a paired
churn of ~30 slots per 25,000 updates. The projected gain is smaller than the
measurement's own noise.

### 6.3 C0-internal u85,000 vs u86,000 sanity check

C0's u86,000 checkpoint was evaluated on the same panel, **C0-internally only**.

| C0 | endpoint | net4 | foundation | whole panel |
|---|---|---|---|---|
| u85,000 | 167/176 = 94.89% | 70.83% | 14.84% | 42.43% |
| u86,000 | 168/176 = 95.45% | 68.75% | 19.27% | 45.23% |
| paired flips on the endpoint | **+7 / −6, net +1** | | | |

This is the single most useful calibration number in the section. **1,000
updates of C0 flip 13 of the 176 endpoint slots** — exactly the same slot churn
as its entire 25,000-update tail (§6.2). The endpoint's run-to-run resolution on
this panel is therefore ~±13 slots of churn regardless of how much training
separates the two checkpoints, and any net movement smaller than that is not a
trend.

It also confirms the decision to compare at u85,000 rather than against C0's
u86,000: doing the latter would have handed C0 an arbitrary +0.56 pp drawn from
this churn.

### 6.4 Recommendation

**C0 (control): stop. Do not continue.** It is at 94.89% held-out raw
completion having peaked at 96.59%, its training metrics have been flat since
u30,000, and its only remaining trend is negative. More importantly, the
control's job in this design is to answer *what the completion objective does
without the gate*, and it has answered: it converges to near-ceiling raw
completion built almost entirely on inadmissible digs, and gets **worse** at
admissibility with more training. Another 14,000 updates cannot change that
answer and would only extend a trend that is already reported.

**T1 (treatment): stop this segment too, but for a different reason.** T1 is
not obviously converged in the sense that its held-out endpoint is at 63.07%
with a positive tail sign — but the sign is not significant (p = 0.473), the
projected gain over the whole remaining budget is +1.6 pp, and it would cost a
full 120 h allocation to run 19.5 h of compute. The decision-relevant question
is not "will T1 reach 65%" but "what closes a 31.8 pp generalization gap", and
§4.2 says extra updates are not the answer: T1 already fits its training
distribution to within 1.3 pp of the control. **Compute spent on 14,000 more
updates of this configuration buys ~1.6 pp on a question that is not
update-limited.**

**Recommendation: do not request a continuation segment. Close the pilot here
and spend the next allocation on a different arm.** Concretely, in the order the
evidence supports:

1. **Two more matched seed pairs, bringing the total to the three the promotion
   rule requires.** Every number in this readout is single-seed; the entire
   61 pp admissible lead rests on n = 1. Because both arms plateau on the
   held-out endpoint by u60,000 (§6.2), the replicate seeds need only run to
   u60,000 — 83 h each, cheaper than one continuation segment for both arms
   together, and it converts the headline from an observation into a result.
   **This is the highest-value use of the next allocation.**
2. **ROS physical acceptance on the u85,000 T1 checkpoint.** It needs no GPU
   training at all, it is the deployment endpoint, and it is the second conjunct
   of the promotion rule. It has never been measured and is the largest
   unmeasured quantity in the pilot.
3. **Attack the generalization gap, not the update count.** T1's failure is
   transfer, not fit. The 12 pooled trench conditions × 96 maps is a narrow
   training distribution and the gate makes the reachable-state manifold
   narrower still. Widening the trench training slice, or regularizing, is the
   indicated direction. Note that the broad-to-strict tolerance curriculum —
   the preregistered next step — has lost its main supporting observation
   (§3.3) and should not be run on the strength of the u10,000 evidence alone.

If a continuation is nonetheless wanted for a reason outside this readout, the
one defensible framing is to raise `total_timesteps` well beyond u100,000 so
that a fresh 120 h allocation is actually spent — u100,000 is 19.5 h away and
would leave ~100 h idle.

---

## 7. Artifacts

Readout: `TRENCH_ALIGNMENT_PILOT_U85000_READOUT_20260825.md` (this file).

Receipts, `tools/trench_align_pilot_u85000_receipts/`:

| file | contents |
|---|---|
| `eval_{c0,t1}_u085000_gate_main_dev.json` | panel receipts at the matched update, 608 per-map rows each |
| `eval_{c0,t1}_u060000_gate_main_dev.json` | panel receipts at u60,000, for the held-out plateau curve |
| `eval_c0_u086000_gate_main_dev.json` | C0-internal sanity check |
| `probe_{c0,t1}_u085000.json` | mechanism summaries: per-condition, per-axis-class, per-slot, section attribution |
| `probe_{c0,t1}_u085000.npz` | full per-step traces (action, active, validity, applicability, raw yaw/standoff, dug cells, divergence) |
| `probe_{c0,t1}_u085000_net4.json` / `.npz` | net4-scoped admissibility pass (§2.4) |
| `wandb_{c0,t1}.jsonl.gz` | full W&B histories, 9,460 / 9,398 rows to u86,001 / u85,431 (gzipped: 18 MB → 3.4 MB each; the readout tool reads `.gz` directly) |
| `readout_join_20260825.json` | joined receipt: endpoints, mechanism, admissible completion, W&B matched values, tail trends, rule outcomes |

Tools:

- `scripts/trench_align_rollout_probe.py` in the **baselines** worktree, sha256
  `67d75b8154407e5fa5ec7cf882b2869ed2560cc2d1420ff344532f9c3df29d8c` —
  unchanged from the u10,000 cells;
- `tools/trench_align_pilot_readout.py` in this worktree — extended this round
  with `--matched-update`, `--prior-readout` (which makes the stop rule's
  two-successive-evaluations clause evaluable), a `wandb_tail_trend` block for
  the plateau assessment, an `admissible` block on the primary endpoint, a
  non-finite guard in `wandb_series` (see the `"NaN"`-as-string note in §4.4),
  and transparent gzip reading of the history files.

Exact reproduction command for the join receipt:

```
python tools/trench_align_pilot_readout.py \
  --panel-c0 <R85>/eval_c0_u085000_gate_main_dev.json \
  --panel-t1 <R85>/eval_t1_u085000_gate_main_dev.json \
  --probe-c0-first <R10>/probe_c0_u000500.json --probe-c0-late <R85>/probe_c0_u085000.json \
  --probe-t1-first <R10>/probe_t1_u000500.json --probe-t1-late <R85>/probe_t1_u085000.json \
  --wandb-c0 <R85>/wandb_c0.jsonl.gz --wandb-t1 <R85>/wandb_t1.jsonl.gz \
  --matched-update 85000 --prior-readout <R10>/readout_join_20260821.json \
  --output <R85>/readout_join_20260825.json
```

Checkpoint identity, as recorded in the panel receipts: C0 u85,000
`1767873efbcc…`, T1 u85,000 `526607f70fa0…`, C0 u86,000 `2c8f3e6a5782…`,
C0 u60,000 `20fcf209fc44…`, T1 u60,000 `6f13bb1c1ba2…`. All copied `.pkl`
files were deleted after measurement.

## 8. Caveats and what was not measured

1. **One seed.** The preregistration requires ≥3 matched seeds for promotion.
   Everything here is a single-seed result and no confidence interval on the
   arm difference is available.
2. **Development panel only.** Promotion and sealed panels untouched.
3. **Deterministic (argmax) evaluation throughout.** T1's behaviour differs
   markedly between argmax and sampling, so these numbers should not be read as
   the training-time distribution. The 0.9790 online-eval figure at u85,001 is
   a *stochastic* in-distribution measurement and is not comparable to the
   63.07% argmax held-out panel figure; §4.2 compares each against its own
   matched counterpart, never across.
4. **Probe vs panel differ by 1 episode for T1** (batch-size-dependent step
   RNG). Panel is authoritative for raw; probe for admissible.
5. **Admissible completion uses Terra's gate as the admissibility oracle** — a
   necessary but not sufficient proxy. **The deployment endpoint (ROS physical
   acceptance of raw plans under footprint, reach, endpoint and swept-path
   checks) was not measured, and remains the single largest gap in this
   pilot.** It is also the second conjunct of the promotion rule.
6. **Aligned occupancy and DO-rate-while-misaligned are dwell-weighted** and
   are reported in §3.1 only with that caveat attached; neither carries a
   within-arm trend claim anywhere in this readout.
7. **Non-trench and foundation transition classes are not covered** by the
   differential gate check.
8. **The plateau curve reuses the development panel.** §6.2 evaluates three
   checkpoints per arm on the same `gate_main/development` slots. Those slots
   are held out from *training*, not from *checkpoint selection*, and this
   readout has now looked at them five times. Nothing here selects a checkpoint
   on that basis, but a future promotion decision must not; that is what the
   untouched promotion and sealed panels are for.
9. **The admissible endpoint is probe-derived, the raw endpoint is
   panel-derived.** They differ by 1 episode for T1 (§1), so the two rows of
   the headline table are not read off the same 176 outcomes. The admissible
   figure is internally consistent (110/176 admissible against the probe's own
   110/176 raw), which is the comparison that matters, but it is not
   110-out-of-the-panel's-111.
10. **C0's stalled multi-section population is n = 4** at u85,000, so the
   section-attribution comparison in §3.5 is T1-internal (u10k vs u85k) rather
   than cross-arm at this update.
11. The three retracted u10,000 measurements were not recomputed or reused.
12. **The u10,000-era motivation for the broad-to-strict tolerance curriculum
    no longer holds** (§3.3). No replacement remedy has been tested. What the
    u85,000 data indicate is a *generalization* gap — T1 fits its training
    maps within 1.3 pp of C0 but transfers 31.8 pp worse — which points at
    map diversity or regularization rather than at the tolerance schedule, but
    that hypothesis is untested here.
