# The standoff floor traps position; yaw and the junction veto lose the digs

Date: 2026-09-01
Branch: `epoch/terra-footprint-and-soil-containment-20260831`, HEAD `566867db`
(corrected footprint raster + dig-side soil containment)
Status: observation + one cheap measurement over existing traces. No env change,
no training, nothing committed.

## 1. The observation

A human who knows the gate contract exactly, playing against a panel that
prints the yaw error, the perpendicular standoff, the metres of deficit and the
per-section verdict every frame, **dug 0 of 185 required cells in 264 actions
across two slots**.

| slot | condition | actions | `DO` presses | cells dug | median standoff of the aligned section | steps below the 3.5 m floor |
|---|---|---:|---:|---:|---:|---:|
| 296 | trn-net3-side1-road | 162 | 3 (2 out-of-band, 1 not-applicable) | 0 / 88 | 1.12 m | 101 / 115 |
| 294 | trn-net3-side1-road | 102 | 9 (7 out-of-band, 1 both, 1 not-applicable) | 0 / 97 | 0.25 m | 74 / 80 |

The failure mode was identical on both slots and is not a skill failure. The
machine was **correctly aligned** — yaw error 0.0° to the owning section — with
**fresh trench cells inside the workspace cone**, and every `DO` was refused
solely because the base centre sat below the 3.5 m standoff floor (0.25 m on
slot 294). The player's own reading of the map was right; the quantity that
refused him is not visible in the terrain, only in the panel.

His objection, recorded verbatim because it is the useful part: *"the 3.5 to 7
is along the chassis major axis, not lateral distance — otherwise how are we
supposed to dig the trench if we are not on top of the trench?"*

**The second half of the observation matters more than the first.** He was stuck
against section 2 at 0.25 m — hopeless without a long detour — while at the same
pose **section 1 was already in band at 6.90 m and two rotations from aligned**:
the true escape was `RIGHT RIGHT UP`, three actions to `S1 yaw 0.0°, standoff
6.75 m`. Neither he nor the panel spotted it, because the panel lists the
sections symmetrically and he (and I, while coaching him) kept working the
section the cabin happened to face. So the human trial is evidence about
**section selection** at least as much as about the standoff floor — which is
exactly what §3.2 finds for the policy.

## 2. Why forward/backward cannot fix it — measured, not argued

Terra's excavator moves in 5-cell steps along one of 12 headings and has no
lateral motion. When the chassis is parallel to a section (the state the yaw
clause *requires*), forward and backward are tangential to that section, so the
perpendicular standoff is invariant under exactly the two actions a human — and
a policy — reaches for first.

From the human session, over moves taken while parallel (yaw ≤ 15°):

| slot | FORWARD/BACKWARD moves while parallel | mean change in standoff | max abs change |
|---|---:|---:|---:|
| 294 | 35 | **+0.000 m** | **0.000 m** |
| 296 | 71 | +0.001 m | 0.341 m |

On slot 294 the section axis lies on a lattice heading, so the standoff is
*exactly* invariant: 35 moves, zero change, to the last bit. On 296 the axes are
off-lattice, so single moves jitter by up to 0.34 m and still net to zero.

The only escape is a detour: rotate off-axis (≥3 rotations for 90°), drive 1–2
moves to cross the band, rotate back (3 more) — **6–8 actions, no intermediate
reward, and every `DO` in between a silent no-op**. That is the structural cost,
and it is paid out of a 450-step horizon.

## 3. Is T1 living in this trap? Two denominators, two different answers

Cheap query over traces already on disk:
`tools/trench_align_pilot_u85000_receipts/probe_{t1,c0}_u085000.npz`
(450 steps × 176 trench slots, per-step `align_applicable`, `align_valid`,
`raw_standoff_m`, `raw_yaw_rad`, `pose_valid_axis_count`, `fresh_axis_count`,
`fresh_trench_cells_dug`, `action`, `action_had_effect`, `loaded`).

Tail definition (mine, stated so it can be challenged): for each slot, the
**post-dig tail** is every active step after the last step at which a fresh
trench cell was actually removed.

**The two sets must not be mixed.** They answer different questions and they
disagree sharply about the standoff floor:

- **Positional / counterfactual set** — every tail step at which a `DO` *would
  have been* refused. Describes **where the machine stands**. It is
  **dwell-weighted**: a machine that sits still in an illegal spot contributes
  one row per step, so occupancy dominates the count.
- **Attempt set** — steps where the policy *actually pressed* `DO`, empty, with
  the gate applicable, and was refused (`action==DO & active & applicable &
  ~valid & loaded==0`). Describes **which digs it lost**. Not dwell-weighted.

### 3.1 Positional set — where it stands (dwell-weighted)

| quantity | T1 (gate on) | C0 (gate off) |
|---|---:|---:|
| active steps | 35,113 | 11,348 |
| post-dig tail | 27,898 steps (**79.5%** of active) | 4,046 (35.7%) |
| tail steps where the gate was applicable | 69.1% | 54.8% |
| tail steps applicable **and would refuse** | 15,628 (**56.0%** of tail) | 1,879 (46.4%) |
| of those, standoff **< 3.5 m** | 6,355 (**22.8%** of tail, 40.7%) | 1,141 (28.2% of tail) |
| of those, standoff **> 7.0 m** | **0 (0.0%)** | 0.2% |
| of those, in band, yaw-only | 59.4% | 42.7% |
| mean / median standoff | 3.72 / 3.91 m | 2.86 / 3.02 m |
| tail steps whose action had no effect | 16.9% | 6.8% |

### 3.2 Attempt set — which digs it loses (not dwell-weighted)

| quantity | T1 (gate on) | C0 (gate off, counterfactual) |
|---|---:|---:|
| refused `DO` presses | **350** | 983 |
| standoff **< 3.5 m** | **1 (0.3%)** | 588 (59.8%) |
| standoff **in band** | **349 (99.7%)** | 395 (40.2%) |
| standoff **> 7.0 m** | **0 (0.0%)** | 0 (0.0%) |
| yaw **> 15°** | **350 (100.0%)** | 711 (72.3%) |
| ≥1 section pose-valid yet refused (**junction-veto signature**) | **242 (69.1%)** | 48 (4.9%) |
| multi-axis cone (`fresh_axis_count` ≥ 2) | 291 (83.1%) | 287 (29.2%) |
| mean / median standoff, yaw | 4.69 m / 4.76 m, 60.0° / 60.0° | 2.81 m / 2.80 m, 32.1° / 30.0° |
| effective fresh digs in the run | 713 | 1,142 |

### 3.2b Side finding: some frozen start poses are nowhere near a legal lane

The station search built into the debugger (`P`) reports, from **slot 458's
frozen start pose** (row 10, col 21, base 7): 1,500 reachable poses stepped
breadth-first, **0 of them passing even the geometric necessary condition**
(some section simultaneously yaw-aligned and in band). The single section on
that map is 12.62 m away and 30° off at the start. From **slot 294's** start the
same search returns a legal station in **4 actions** (`DOWN DOWN Q Q` →
`S1 yaw 0.0°, standoff 6.93 m`), so this is a per-slot property, not a general
one.

Bounded claim: the 458 search was budget-limited, so this is "no legal station
in the first 1,500 poses explored", not a proof of none within the full
5-action set. It is still the cleanest available account of why the scripted
oracle records `no_admissible_station` on exactly these slots, and it says the
opening problem on some maps is **relocation before any dig is possible**, which
is precisely what a pose-seeded reset would remove.

### 3.3 Readings, in order of confidence

1. **The 7.0 m ceiling never binds — in either set.** 0 of T1's 17,566
   positional refusals, 0 of its 350 actual refused attempts, 0 of C0's 983.
   Operationally the "band" is a **floor**. The u85,000 readout missed this
   because it measured standoff on *admitted* digs (non-binding, and §3.3 of
   that readout says so) instead of decomposing refusals.
2. **The floor traps the policy positionally; it is not what loses its digs.**
   T1 spends 22.8% of its post-dig tail parked applicable-and-too-close — that
   is the wandering. But when it commits to a dig it is **in band 99.7% of the
   time** and fails on **yaw (100% of refused attempts are > 15° off, all at
   exactly 60°)**, with **69.1% carrying the junction-veto signature** (some
   section was pose-valid and the macro action was refused anyway) on a
   **83.1%** multi-axis cone. So "Lorenzo's trap accounts for a quarter of T1's
   tail" is correct about **occupancy** and wrong if read as "a quarter of its
   lost digs" — the lost digs are an **alignment and junction-clause** problem.
3. **The 60.0° signature is the same one the u85,000 readout found** for T1's
   invalid attempts (min 59.99, max 60.00). Two adjacent lattice headings are
   30° apart, so 60° is "two clicks off-axis" — the policy commits from a pose
   that is a fixed, discrete distance from legal, not from a continuum of near
   misses.
4. **C0 is the mirror image and confirms the split is a treatment effect.**
   Without the gate, 59.8% of its would-be-refused presses are below the floor
   and only 4.9% are junction-veto shaped: it digs from wherever it stands. T1
   has learned the floor into its *attempt policy* (99.7% in band) while still
   spending a fifth of its tail standing below it.
5. **Do not read the DO counts as willingness.** 1,551 tail `DO` presses in
   27,898 steps looks like abstention, but the handover's retracted measurement
   warns opportunity duration is endogenous and per-step attempt rates are
   biased in both directions. Counts are raw; no willingness claim is made.

## 4. What this implies for the remedy ranking

The attempt set reorders the priorities, and it reverses the emphasis a
standoff-first reading would give:

1. **Alignment and the junction clause come first.** They are where the digs are
   actually lost: 100% of T1's refused attempts are yaw-failures, all at exactly
   60°, and 69.1% of them carry the junction-veto signature on an 83.1%
   multi-axis cone. Anything that improves *which section the machine commits
   to* — including simply ranking the sections by cost-to-legal rather than
   presenting them symmetrically — attacks the binding constraint. The human
   trial makes the same point from the other end: he spent 264 actions fighting
   the section he was aimed at (S2, 0.25 m, hopeless) while a different section
   was already in band at 6.90 m and two rotations away.
2. **The standoff floor is second, and it is a *positional* remedy, not a dig
   remedy.** It explains the wandering (22.8% of the tail parked
   applicable-and-too-close) rather than the refusals. The right lever is the
   mid-trench / partial-reset curriculum: seed episodes already in a legal lane
   so the policy does not have to solve recovery before it is ever paid for
   digging. **Concrete engineering gap:** the partial-reset machinery
   (`PARTIAL_RESET_FRACTIONS`, `reset_tier`) substitutes only the **action
   map** — `MapsBuffer._select_map` swaps `partial_action_map` and nothing else,
   and the agent pose still comes from `State.new` with the episode key. **Pose
   is never seeded.** That is the most actionable line in this document.
3. **The numbers to quote** when arguing the tail is mechanism, not noise:
   79.5% of T1's active steps are post-dig; 56.0% of those would refuse a `DO`;
   and of its actual refused attempts, 99.7% are in band and 69.1% are
   veto-shaped. A curriculum or observation change that works should move the
   positional numbers *and* shift the attempt mix away from the veto.
4. **Recorded, not acted on: should the gate admit an aligned-but-too-close
   pose?** At the human's pose the cone said "reachable" (2 fresh cells inside
   the annulus) and the gate said no. Reach is radial and already tested by the
   cone (≈3.64–6.50 m, ±30°); the standoff floor is a *lane* constraint on top
   of it, deliberately so — the research note's intent is "the chassis needs a
   safe parallel offset lane, not attraction to the excavation centreline", and
   Terra makes dug cells non-traversable, so a machine working from on the line
   strands itself in its own hole. The audit found the gate non-binding on
   feasibility, and the axis-sweep receipt shows the human's own slot 294 is
   **fully coverable by lane sweeping alone** (97/97 cells, 320 legal lanes,
   best single lane per axis 100% / 100% / 92.3%). The contract stands; nothing
   here justifies changing it.

### 4.1 The asymmetry worth attacking: shaping, not information

The sharpest thing in this document is not that the floor is invisible on the
terrain — it is **where the gradient is missing**.

- **The reward carries no gradient.** Refusal is an all-or-nothing macro no-op.
  0.25 m outside the floor and 3.4 m outside it produce byte-identical
  transitions: same reward, same state, same everything. Nothing in the return
  distinguishes "one rotation from legal" from "a full relocation away".
- **The observation *does* carry the gradient.** Terra exports
  `fresh_trench_dig_standoff_error` **signed and normalised** (negative = too
  close, positive = too far, zero = in band) plus
  `fresh_trench_dig_yaw_error` normalised over 0–90°, and per the pilot design
  both scalars are in the policy input for **both** arms. The policy can see
  exactly how far out of band it is and on which side.

So this is not an observability failure; it is a shaping failure. A policy that
can see a continuous error signal, and receives an identical null return
whichever end of it it sits at, has no reason to descend it — and the descent it
would have to perform is a 6–8 action detour through states that are all equally
worthless under the current return. That is a precise, testable account of why a
recovery the policy can in principle see is one it does not learn, and it points
at a corrected potential on the *exported* errors (the research note's deferred
option 3) rather than at loosening the gate.

## 5. Provenance and reproduction

- Human session logs (JSONL, one record per action with pose, per-section yaw
  and standoff, verdict, reason code, counters, state digest):
  `data/manual_trench_logs/manual_trench_slot296_20260901_141629.jsonl`
  (started on 296, switched to 294 mid-session). Gitignored by `data/*`.
- Tool: `tools/manual_trench_debugger.py` (this branch, uncommitted).
  Launch: `JAX_PLATFORMS=cpu PYTHONPATH=$PWD
  /home/lorenzo/moleworks/.venv-terra-uv/bin/python
  tools/manual_trench_debugger.py --slot 294`
- Trace measurement: recomputed from
  `tools/trench_align_pilot_u85000_receipts/probe_{t1,c0}_u085000.npz`;
  no new rollouts, no checkpoint loads, ~2 s of numpy.
- Lane feasibility for slot 294:
  `tools/trench_align_oracle_receipts_20260831/axis_sweep_feasibility_gate_main_dev.json`,
  entry `trn-net3-side1-road:evaluation/gate_main/development:294`,
  `footprint: corrected`.
- Gate contract authority: `TRENCH_FRESH_DIG_ALIGNMENT_RESEARCH_NOTE_20260818.md`
  ("base-center perpendicular standoff"), implementation
  `terra/state.py:_get_fresh_trench_dig_alignment_details`.

## 6. Non-claims

- No claim that the gate is wrong, or that the band should be widened. The
  contract is deliberate and the feasibility evidence is against changing it.
- No claim about the human's skill; the point is that a fully-informed operator
  with a numeric panel still could not convert alignment into a dig, which
  bounds how discoverable the manoeuvre is.
- Both decompositions use Terra's exported *diagnostic-axis* standoff and yaw
  (the closest blocking section on a refusal). They answer "why was this
  refused", not "how far is the machine from every section".
- The junction-veto share is a **signature**, not a direct measurement: it
  counts refused attempts with `pose_valid_axis_count >= 1`, i.e. some owning
  section was pose-valid and the macro action was refused anyway, which is the
  veto's defining condition. It does not count the exclusive cells themselves.
- The positional set is **dwell-weighted** and the attempt set is not. Mixing
  them produces the wrong conclusion in both directions; §3.1 and §3.2 are
  reported separately for that reason.
- C0's "refused" attempts are counterfactual throughout: its gate is off, so
  those digs actually happened. Its column is a contrast, not an outcome.
- Two slots, one operator, one session for the human trial. That part is an
  existence argument about the mechanism, not a rate.
