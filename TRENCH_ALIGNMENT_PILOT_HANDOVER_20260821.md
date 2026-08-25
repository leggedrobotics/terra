# Handover — fresh-trench dig-alignment C0/T1 pilot

Written 2026-08-21, mid-flight. Both training arms are RUNNING and must not be
disturbed. Read this file, then
`TRENCH_ALIGNMENT_PILOT_U10000_READOUT_20260821.md` (results) and
`TRENCH_FRESH_DIG_ALIGNMENT_RESEARCH_NOTE_20260818.md` (design authority).
`TRENCH_ALIGNMENT_PILOT_PREREGISTRATION_20260819.md` holds the endpoints and
stop rules — but see §6, two of its clauses are broken.

---

## 1. What this experiment is

Terra can refuse a fresh trench dig when the machine's pose is physically
incompatible with the trench section (chassis yaw within 0.2619 rad of the
section axis, perpendicular standoff 3.5–7.0 m). Rejection is an
all-or-nothing macro no-op. The pilot tests that gate causally with two
matched-seed arms differing in exactly one flag:

| arm | job | `enforce_trench_dig_alignment` |
|---|---|---|
| C0 control | 11152229 | false |
| T1 treatment | 11152230 | true |

Everything else is identical: same seed (20260818), bank, revisions, PPO
config, and the width-3 alignment observation in the policy input for **both**
arms. The legacy trench reward stays off everywhere.

## 2. Where everything lives

**Repos** (worktrees, not main checkouts):

| what | path | branch | HEAD |
|---|---|---|---|
| Terra | `.worktrees/terra_trench_fresh_dig_alignment_20260818` | `experiment/trench-fresh-dig-alignment-20260818` | `cdd643e3` |
| baselines | `.worktrees/terra_baselines_trench_pose_alignment_20260818` | `experiment/trench-pose-alignment-20260818` | `9dda8e3` |

Both pushed to origin. The Euler jobs run from an archived snapshot of Terra
`a4b838b6` and baselines `f64694a5`; later commits are analysis artifacts only
and do **not** affect the live runs.

**Euler** — account `alesweber` (SSH `euler-alesweber`), Slurm account
`es_hutter`, partition `gpuhe.120h`, 4× RTX 4090 each, 119:45 wall.

```
RUNS=/cluster/scratch/alesweber/codex_terra_edge_runs/terra_trench_align_v1/runs/f64694a569fbeb1353f2f908c46b9baab5f7e22b/s20260818
$RUNS/{c0,t1}/slurm_<jobid>.out
$RUNS/{c0,t1}/checkpoints/trench_align_{c0,t1}_f64694a569fb_s20260818_update_XXXXXX.pkl
$RUNS/{c0,t1}/run_contract.env
```

**W&B**: `aless-weber-eth/mixed-agents`, runs
`trench_align_{c0,t1}_f64694a569_s20260818`.

**Bank**: `/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819`
— the shipped V8 R2 release enriched with finite trench section metadata.
Training uses the pooled slice `train_pilot_pooled_12cond` (1,152 maps,
`DATASET_SIZE=1152`). Evaluation uses `evaluation/gate_main/*`.

**Launcher**: `scripts/euler_trench_align_v1/{submit.sh,run.sbatch}` in the
baselines worktree. `SUBMIT=0` contract check, `SUBMIT=stage` stage only,
`SUBMIT=1` submit both arms.

## 3. Current state

At handover: ~u13,000 of 100,000, 17h54m elapsed, both healthy, no NaN, no
failure signatures. Throughput ~720 updates/hour **including** eval and
checkpoint overhead.

**The run will not reach u100,000 in one allocation.** 119:45 at 720 upd/h
lands near **u86–87k**. That is by design (the target is deliberately beyond
one segment) and a wall-time exit with a valid checkpoint is `CONTINUABLE`,
not a failure. A continuation segment will be needed; use `--resume_from`, set
`total_timesteps` to the absolute final target, and follow the W&B resume
rules in the `terra-rl` skill.

## 4. Scope decisions already made — do not silently revisit

- **Trench-only training.** 12 trench conditions, no foundation maps. The
  accepted-bank machinery has no trench-only scope, hence the pooled folder and
  the YAML-preset path rather than `--accepted-bank-*`.
- **net4 excluded** (`trn-net4-side2`, `-side2-s`, `-side1-road`). The 2,400-map
  preflight found 61 maps with no complete strict-gate cover, all 4-axis
  junctions, and the failure is **tolerance-independent** (12 headings quantize
  yaw error to 30° bins; identical at 15/20/25°). Reported separately, never
  pooled into the endpoint.
- **v7-trn conditions excluded** — no finite generator provenance exists
  anywhere for those seven geometries. They fail the metadata contract closed.
- **reward_v2** (timing variant 0, protocol
  `obstacle_geodesic_8_physical_global_v1`). Not optional: the legacy
  dense_skill path rejects R2 distance datasets by contract.

## 5. Results at u10,000 — the headline is a reversal

| endpoint (trench − net4, 176 slots) | C0 | T1 | Δ |
|---|---|---|---|
| **raw** exact completion | 88.64% | 38.64% | −50.00 pp |
| **admissible** exact completion | **4.55%** | **38.64%** | **+34.09 pp** |

C0 reaches 88.6% by digging from pose-invalid stations on 83.35% of applicable
attempts; 166/176 of its episodes contain at least one, median 6 in a
*successful* episode. Its 8 admissible successes are all single-axis straights.

**Raw completion is not a valid promotion criterion.** The two arms do not
produce the same kind of output. This is why the preregistration's
ROS-acceptance clause is load-bearing rather than belt-and-braces.

Alignment is **not** learned incidentally: the control's invalid share moves
only 0.9888 → 0.8335 over 9,500 updates, while T1 reaches 41.5% aligned
occupancy (C0: 18.1%) and digs while misaligned 28× less often.

Rule outcomes: **stop rule does not fire** (needs two successive evaluations,
only u10,000 exists; mechanism clause ill-posed). **Code stop clean** — 129
divergences, all one-to-one with invalid applicable digs, zero at non-DO steps,
valid digs, relifts, dumps; zero target-cell mutations.

Mechanism, in one line: the gate suppresses **commitment escalation**. C0
phase-transitions between u7k and u12k (DO fraction 0.135 → 0.269, no-effect
0.248 → 0.081); T1 stays flat (0.138 → 0.141). T1 is **not** deterred — it does
2.4× more admissible digs per episode at u10,000 than at u500 and never loses
dig propensity.

## 6. Traps — read before doing anything

**Two preregistration clauses are broken.** Fix the wording before the u20,000
readout depends on them.

1. *"T1 invalid fresh-DO fraction must fall ≥50% from its first evaluation"* is
   **undefined**: the u500 baseline is exactly 0.0000. And the zero means
   **competence, not incompetence** — at u500 T1 met 22,733 misaligned
   applicable states across 161/176 episodes and dug in **zero** of them, while
   digging at 15.1% when aligned. It had mastered the constraint by 32.8M
   transitions. Naively applying the rule inverts the truth. Substantive
   replacement: alignment accuracy on applicable attempts (T1 81.8% vs C0 16.7%).
2. *"raw successful fresh-dig yaw/standoff"* is **vacuous**: 12 headings at 30°
   against a 15° tolerance means admitted digs are always ~0.00° off-axis.
   Standoff never binds either (4.70–4.97 m inside a 3.5–7.0 m band).

**Do not use per-step attempt rates as willingness evidence.** Opportunity
duration is endogenous — digging ends an opportunity, dawdling extends it — so
the rate is biased in both directions. This killed one measurement already.

**Three measurements were retracted** during the u10,000 analysis, all recorded
with causes in the readout §(f): a pose-availability metric that was
algebraically identical to the gate decision; a pooled attempt rate that was
dwell-weighted; and the deterrence hypothesis, refuted by T1's rising
admissible digs per episode. Do not resurrect them without reading why.

**Operational:**

- `submit.sh` pins `TRENCH_TERRA_REVISION_PIN=a4b838b6`, which no longer equals
  the worktree HEAD. A naive resubmit **will fail the pin check** — that is
  intended. Bump it consciously, or the arms stop being comparable.
- The baselines worktree carries **another session's uncommitted `isaac_sim/`
  work**. Never `git add -A` there. `submit.sh` excludes exactly those paths
  from its cleanliness check.
- Evaluation must pass `--terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4`
  — the bank's pinned protocol revision, *not* the worktree HEAD.
- Local RTX 4090 needs `XLA_FLAGS=--xla_gpu_autotune_level=0` (driver 580 vs
  jax-0.4.26 cuDNN 8.9.7 → `CUDNN_STATUS_EXECUTION_FAILED`). Deliberately not in
  the Euler sbatch; apply there only if the signature appears.
- **Local disk is at 15 GB free.** Checkpoints are 28 MB each; the u20,000
  readout needs room for four. Delete copies when done.
- A `while pgrep -f "<pattern>"` guard **matches its own bash command line** and
  spins forever. This cost ~40 minutes twice. Don't write one.

## 7. Next actions

1. **u20,000 readout** — the earliest point the two-evaluation stop clause can
   even be assessed. Repeat the u10,000 procedure:
   ```
   eval_fixed_bank.py --checkpoint <ckpt> --panel-family gate_main \
     --accepted-panel development \
     --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 --horizon 450
   scripts/trench_align_rollout_probe.py   # mechanism + admissible completion
   ```
   Report **admissible** completion as the endpoint, raw alongside it, never
   raw alone.
2. **Fix the two preregistration clauses** (§6) and commit the amendment before
   using them.
3. **Broad-to-strict tolerance curriculum arm.** This is now evidence-driven,
   not speculative: **128 of T1's 129 invalid attempts are yaw-only failures at
   exactly 30.00°** — one heading bin from admissible, standoff in band. A broad
   early tolerance that anneals to 15° lets the policy complete whole trenches
   first, then tightens precision on a behavior it already has. Per the research
   note, test the curriculum **before** any reward term.
4. **Plan the continuation segment** before the wall (~u86–87k).
5. Open question: is the completion cost transient? T1 was still improving at
   u10,000 with 90k updates of budget left.
6. Unmeasured and important: **ROS physical acceptance**, the real deployment
   endpoint. Terra-gate-admissible is *necessary, not sufficient* — Terra's
   discrete cone, the sub-cell CABIN_CONTROL offset (0.274 m), and the absence
   of swept-path checking all sit between this proxy and the field. Also
   unmeasured: ≥3 seeds, promotion/sealed panels.

## 8. Artifacts

- `TRENCH_ALIGNMENT_PILOT_U10000_READOUT_20260821.md` — full results
- `tools/trench_align_pilot_u10000_receipts/` — 2 panel receipts, 4 probe JSONs
  + traces, 2 section JSONs, `readout_join_20260821.json`
- `tools/trench_align_pilot_readout.py` — the join/analysis tool
- `scripts/trench_align_rollout_probe.py` (baselines worktree) — instrumented
  rollout; records alignment scalars, admissibility, and section attribution,
  and asserts it reproduces Terra's exported values at every step
- `tools/audit_trench_alignment_feasibility.py`, `tools/enrich_trench_finite_metadata.py`,
  `tools/build_trench_pilot_pooled_train.py` — bank construction and preflight

---

## 9. UPDATE 2026-08-25 — pilot complete, this note partly superseded

Both arms hit the 119:45 wall as designed (C0 u86,000, T1 u85,441). Final
checkpoints archived to `/cluster/project/rsl/alesweber/terra_trench_align_v1_final/`
with a SHA-256 manifest — scratch purges, do not rely on it.

**Result at matched u85,000** (see `TRENCH_ALIGNMENT_PILOT_U85000_READOUT_20260825.md`,
Terra `6f608e64`): admissible exact completion **C0 1.14% vs T1 62.50%
(+61.36 pp)**; raw **94.89% vs 63.07% (−31.82 pp)**. The control *regressed* on
admissibility with 8.5× more training. Both arms plateaued since ~u30k.

**Three items in this note are now superseded:**

1. §7.3 broad-to-strict curriculum — **evidence retired.** T1's residual
   invalid attempts moved from exactly 30.00° (one heading bin) at u10k to
   exactly 60.00° (two bins, all 350) at u85k. Admitting them would need a
   >60° tolerance. Do not launch that arm on the u10,000 rationale.
2. §7.4 continuation segment — **do not run.** 14,000 updates is 19.5 h of a
   120 h allocation for a projected +1.6 pp, inside panel noise.
3. §7.5 "is the cost transient" — **answered.** T1 did enter the escalation,
   ~4k updates later and at 1.63× vs 2.63× amplitude. The raw gap closed 36%
   on held-out but stopped closing after u60k while training-distribution
   convergence reached ~95%: the residual is a **generalization** gap.

**Priorities now, in order:** (1) two more matched seed pairs to u60,000 — the
61 pp lead rests on n=1; (2) **ROS physical acceptance**, never measured,
needs no training, and is the promotion rule's second conjunct; (3) the
generalization gap. §6 Traps remains valid in full.
