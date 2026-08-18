# Terra fresh-trench dig alignment

Date: 2026-08-18

Status: source audit, finite-section feasibility probe, and an opt-in Terra
treatment are complete. No training was launched. The treatment is not enabled
by default and is not yet wired into the Terra-baselines policy encoder.

## Decision

Terra should prevent a fresh trench dig when the base pose is incompatible with
the generated local trench section. ROS postprocessing remains necessary for
continuous `CABIN_CONTROL`, footprint, reach, endpoint, and swept-path
validation, but it cannot repair a policy that repeatedly chooses physically
bad excavation stations without changing the demonstrated plan.

The smallest credible first treatment is **option 2: an all-or-nothing fresh-dig
gate plus observable alignment state**. Keep the existing trench reward off.
Do not add a corrected potential until the gate has been evaluated causally.

The user's metadata direction is essential: the map generator already knows
the finite sections and intersections. Terra should retain those facts, not
infer one dominant infinite axis from the raster at every dig.

## Source-level audit

The audit is against Terra base revision
`25f855db3d913fd638c4e56b1740437a2b7122ca`.

- `State._step()` routes action 6 (`DO`) to `State._handle_do()`
  (`terra/state.py:255-305`). For an excavator, `DO` calls `_handle_dig()` when
  empty and `_handle_dump()` when loaded (`terra/state.py:2816-2908`).
- `_mask_out_wrong_dig_tiles()` first builds the exact workspace cone. If any
  positive staged soil is in the cone, it selects only positive soil;
  otherwise it selects action-map-zero target soil. It also applies depth,
  previous-dig, and foundation masks (`terra/state.py:2119-2179`). Thus
  `target < 0 && action == 0` is the exact fresh-target predicate after the
  existing macro-action selection.
- `_handle_dig()` rejects a workspace that intersects static padding, then
  applies the selected cone, soil mechanics, load accounting, new hole-based
  dumpability, and `last_dig_mask` (`terra/state.py:2438-2563`).
- The default 64-cell map is 36.5714 m wide (0.57143 m/cell), with 12 base and
  12 cabin headings, a five-cell move, and a five-cell radial reach extension
  (`terra/config.py:64-106`). The excavator annulus is approximately
  3.64--6.50 m and spans +/-30 degrees; the base-frame body exclusion also
  makes the exact cone depend on both base and cabin heading
  (`terra/state.py:1409-1485`, `terra/state.py:1996-2029`).
- Movement is unavailable to a loaded excavator. Candidate translation and
  rotation endpoints are checked using the rasterized 7x11 footprint against
  padding, holes, and blocking piles (`terra/state.py:527-704`,
  `terra/state.py:870-966`). This is an endpoint-footprint abstraction, not a
  continuous swept-volume model.
- Dynamic dumpability removes cells within Terra's 5x5 dilation of all holes
  (`terra/map.py:29-54`). Dumping has its own physical, dumpability,
  traversability, last-workspace, and accepted-zone logic
  (`terra/state.py:2680-2801`).

The current trench reward is not the desired contract:

- it is disabled by default with `apply_trench_rewards=False`
  (`terra/config.py:205-219`);
- when enabled, it minimizes base-center distance to the closest **infinite
  trench centerline** (`terra/state.py:3324-3356`);
- it aligns the absolute cabin/arm angle rather than chassis yaw
  (`terra/state.py:3358-3388`); and
- it is added on every excavator action, not only an empty `DO` that would
  excavate fresh trench soil (`terra/state.py:3606-3617`).

That distance term is directionally wrong for execution: the chassis needs a
safe parallel offset lane, not attraction to the excavation centerline.

## Why finite generator metadata is required

V8 and V9 already generate `trench_arms` in one-to-one order with `axes_ABC`;
the current curriculum generator delegates its Terra metadata writer to V9.
Review JSON retains the arms, but the old Terra writer exported only the
infinite lines. Materialization then copied that already-lossy sidecar.

In the latest P5 candidate bank
`/home/lorenzo/moleworks/.artifacts/terra_p5_candidates320_full_20260801_642756cc`,
all 4,478 trench records have consistent finite arms, axes, and counts. Yet a
nearest-infinite-line label disagrees with the nearest finite section for:

| Family | Disagreeing target cells |
|---|---:|
| straight | 0 / 175,144 (0.00%) |
| T | 1,658 / 57,972 (2.86%) |
| network3 | 2,109 / 42,533 (4.96%) |
| network4 | 2,695 / 63,610 (4.24%) |
| network3-road | 2,349 / 45,116 (5.21%) |

An `argmin` label also cannot represent legitimate membership in more than one
section at a junction. A dominant-axis gate could therefore authorize fresh
cells from a perpendicular branch. The implementation carries finite section
endpoints and generated half-width, then caches a per-cell multi-owner bitmask.

Generator edge regularization can add a small fringe outside the ideal arm
capsule. Across the same 4,478 maps (629,496 trench target cells), the maximum
measured excess was 1.3531 cells. Membership therefore uses generated
half-width + 0.5 cell, with a nearest-section fallback bounded at +1.5 cells.
It never assigns an arbitrarily distant negative target on a mixed-purpose map.

## Implemented environment contract

The treatment is global and opt-in:

```text
enforce_trench_dig_alignment = false
trench_dig_yaw_tolerance_rad = 0.2619
trench_dig_standoff_min_m = 3.5
trench_dig_standoff_max_m = 7.0
```

For one prospective `DO`:

1. Use Terra's existing cone and dig-selection logic.
2. Apply only to a type-0 excavator that is empty and whose selected workspace
   contains `target < 0 && action == 0` cells owned by a generated trench
   section.
3. A section is pose-valid when chassis yaw is parallel to its axis within the
   tolerance and base-center perpendicular standoff lies in the configured
   band. Terra's existing cone remains the actual reach test.
4. Every selected fresh trench cell must have at least one pose-valid owning
   section. Shared junction cells may use either section. If the same macro
   cone contains an exclusive cell from a perpendicular, invalid branch, reject
   the complete `DO`.
5. Rejection is an all-or-nothing macro no-op, not a hidden per-cell mask or an
   action mask.
6. Relift, loaded dumping, off-zone staging, navigation, non-excavator actions,
   and non-trench excavation retain their old transitions.

Terra exposes three top-level values:

- `fresh_trench_dig_alignment_valid`;
- `fresh_trench_dig_yaw_error`, normalized from 0 to 1 over 0--90 degrees; and
- `fresh_trench_dig_standoff_error`, signed and normalized (negative is too
  close, positive too far, zero is in-band).

For an inapplicable action they are `(1, 0, 0)`. At a rejected intersection,
the errors describe the closest blocking section rather than an already-valid
section. The canonical loader rejects inconsistent counts, stale axis/endpoint
pairs, and invalid widths; batch reset rejects missing/incomplete finite
metadata before JAX tracing. Lower-level State/TerraEnv use also fails closed
when a declared section is incomplete.

Implementation anchors:

- configuration: `terra/config.py:276-282`;
- metadata validation/loading: `terra/maps_buffer.py:874-994` and
  `terra/env.py:793-870`;
- finite multi-owner cache: `terra/map.py:11-148`;
- gate and diagnostics: `terra/state.py:2181-2452`;
- observation export: `terra/env.py:539-644`;
- V8/V9/current-writer metadata export:
  `tools/map_generation/generate_prototypes_v8.py:2695-2714` and
  `tools/map_generation/generate_prototypes_v9.py:2019-2038`.

### `CABIN_CONTROL` decision

Do not add an explicit continuous `CABIN_CONTROL` offset to this first Terra
treatment. The URDF offset is `[0.0, -0.274] m`
(`description/mole_description/xacro/parts.xacro:319-325` in the read-only ROS
worktree), which is less than half one Terra cell. Terra's base-centered grid,
discrete footprint, and existing dig cone do not support the rest of the
continuous geometry needed to make that sub-cell correction a physical safety
claim. Adding it only to the standoff metric would be false precision.

Base-yaw plus a broad standoff band is sufficient for the simulator abstraction
because the exact Terra cone still establishes reach. ROS remains the owner of
the rotating `CABIN_CONTROL` origin, exact parallel execution line, continuous
footprint/reach, and swept-path clearance. If Terra later adopts a continuous
workspace pivot, the cone, footprint, and alignment metric should move together.

## Feasibility result

The executable probe is `tools/audit_trench_alignment_feasibility.py`; its
64-map receipt is `tools/trench_alignment_feasibility_20260818.json`.
It uses the review-v4 generated bank at
`/home/lorenzo/moleworks/.artifacts/terra_map_distribution_review_v4/review_bank`.
The probe SHA-256 is
`50dedbf467ec3865369f517cf6efd56d58a0df297481bd2457d7c4c176ebc660`;
the JSON SHA-256 is
`7d6a50e7fae7899eef7c9d616d90a9623add46fb236c2bf3dc4ee6ef2560ea1e`.

For each map it uses finite multi-owner sections, the same bounded raster
fringe as runtime, all 12x12 exact State-derived base/cabin cones, JAX-float32
Terra movement deltas, the 7x11 footprint, padding, and a conservative pose
graph in which the entire target trench is already holes. It finds a monotone
all-or-nothing fresh-dig cover within one persistent pose component and requires
every selected dig station to have an actual legal `BACKWARD` successor that
shares a yaw/standoff-valid section.

| Family | Complete maps | Covered fresh cells | Same-base accepted-dump screen |
|---|---:|---:|---:|
| straight | 16 / 16 | 1,894 / 1,894 | 16 / 16 |
| T | 16 / 16 | 2,392 / 2,392 | 16 / 16 |
| network | 16 / 16 | 2,310 / 2,310 | 16 / 16 |
| road T | 16 / 16 | 2,379 / 2,379 | 7 / 16; 2,301 / 2,379 cells |

Thus all 64 representative maps contain a valid strict-gate construction. The
present policy's bad poses are not evidence that these maps are geometrically
impossible. Conversely, the road dump screen shows why same-base dump reach
must not be added to the fresh-dig gate: relay/staging behavior must stay free.

## Option comparison

| Option | Decision | Reason |
|---|---|---|
| 1. Corrected reward only | Not first | Base-yaw/standoff shaping is directionally correct, but cannot enforce safety and adds coefficient sensitivity. |
| 2. Hard gate + observation | **First treatment** | Directly enforces the one physical invariant, preserves macro semantics, and gives the cleanest causal comparison. |
| 3. Gate + corrected reward | Defer | May improve exploration, but initially confounds whether safety came from feasibility or shaping. |
| 4. Broad-to-strict curriculum | Conditional only | Strict 15-degree feasibility already passes 64/64; use only if the policy has valid opportunities but cannot learn to reach them. |

If later justified, a corrected potential should use chassis-yaw error and
distance to the safe standoff band, and should be active only for an empty
prospective fresh trench `DO`. It must not reuse centerline attraction. A
conditional curriculum may start with a broader tolerance and end at
15 degrees plus numerical slack, the nearest-bin bound for 12 headings.

## Minimal causal experiment

The experiment has exactly two arms:

| Arm | Finite metadata | Three observations in policy input | Gate | Legacy trench reward |
|---|---:|---:|---:|---:|
| C0 control | yes | yes | off | off |
| T1 treatment | yes | yes | on | off |

Everything else must match: fresh initialization, Terra and baselines revision,
frozen train/evaluation banks, PPO config, seed, transition budget, checkpoint
schedule, and evaluator. Do not reuse or reinterpret the currently running
relay/feed-forward or GRU experiments; the handover already identifies runtime,
seed, and architecture confounds.

Before training, enrich the exact frozen banks, run this feasibility preflight
over every trench map, and add the three scalars to the Terra-baselines input
list for **both** arms. Run one matched-seed pilot to the preregistered early
checkpoint, then expand to at least three matched seeds only if the mechanism
works without a material completion regression.

Primary endpoint: strict exact completion on the untouched frozen full-start
panel. Mechanism endpoints: invalid fresh-`DO` attempt/no-effect rate, raw
successful fresh-dig yaw/standoff, and completion by family/section. Deployment
endpoint: ROS physical acceptance of the **raw** plans under footprint, reach,
endpoint, and swept-path checks; 100% mask coverage alone is not acceptance.

Stop rules:

- Code stop: any invalid fresh `DO` mutates a trench target cell, or any matched
  relift/dump/non-trench transition differs.
- Preflight stop: any frozen trench map lacks a complete strict-gate cover or
  lacks valid finite metadata.
- Pilot stop: stop if T1 exact completion is more than 5 percentage points
  below C0 at two successive scheduled evaluations and its invalid-`DO`
  attempt fraction has not fallen by at least half from the first T1
  evaluation. Do not add a reward; first test the conditional broad-to-strict
  curriculum.
- Promotion stop: use a seed-stratified paired bootstrap over the frozen map
  panel. Require the 95% lower confidence bound for T1-C0 exact completion to
  exceed -2 percentage points and the corresponding bound for raw ROS physical
  acceptance to exceed zero. A safer-looking yaw histogram without both
  outcomes is insufficient.

## Provenance, verification, and blockers

- Worktree:
  `/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818`
- Branch: `experiment/trench-fresh-dig-alignment-20260818`
- Uncommitted base revision:
  `25f855db3d913fd638c4e56b1740437a2b7122ca`
- Main Terra checkout was not modified.
- No training, commit, push, ROS execution, or current-run mutation occurred.

Focused behavioral coverage is in
`terra/tests/test_trench_dig_alignment.py`: aligned fresh dig, misaligned
fresh-dig rejection and gate-off control, T-junction all-or-nothing behavior,
too-close/too-far standoff, relift, dump, mixed-map and pure non-trench
excavation, JIT parity, metadata failure, complete toy trench, and a real
backward move. Frozen benchmark and old positional checkpoint contracts are
also checked.

Verification commands:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD \
  /home/lorenzo/moleworks/.venv-terra-uv/bin/python -m pytest -q \
  terra/tests/test_trench_dig_alignment.py \
  terra/tests/test_benchmark_protocol.py \
  terra/tests/test_dump_contract.py \
  terra/tests/test_maps_buffer_distance_contract.py \
  terra/tests/test_partial_action_loading.py

JAX_PLATFORMS=cpu PYTHONPATH=$PWD \
  /home/lorenzo/moleworks/.venv-terra-uv/bin/python \
  tools/audit_trench_alignment_feasibility.py \
  --output tools/trench_alignment_feasibility_20260818.json
```

Final result: `67 passed, 69 warnings, 4 subtests passed in 390.58s`. The
warnings are the pre-existing `jax.tree_map` deprecation. The feasibility JSON
parses, contains 64 per-map results and four family summaries, and reports
complete fresh coverage for all 64 maps. Python compilation and
`git diff --check` also pass.

Remaining blockers before any training:

1. The current Terra-baselines adapter explicitly constructs its model input
   without the three new keys
   (`.worktrees/terra_baselines_trench_pose_alignment_20260818/utils/utils_ppo.py:83-115`).
   Add an opt-in encoder change in a separate baselines worktree and use the
   same observation schema in C0 and T1.
2. Existing runtime metadata banks contain only infinite `axes_ABC`; regenerate
   or deterministically enrich the exact frozen train/evaluation sidecars with
   finite endpoints and half-width, then run the all-map preflight.
3. The feasibility witness is constructive Terra-level evidence, not policy
   discovery and not a full episode proof. It does not prove a route from every
   randomized spawn, an episode horizon, accumulated pile/capacity behavior, or
   a continuous swept path.

## Explicit nonclaims

- The gate does not make a Terra plan execution-safe on the physical machine.
- It does not replace ROS postprocessing or ROS fail-closed validation.
- It does not enforce a dump lane, dump pose, exclusively backward full route,
  or continuous swept clearance.
- It does not show that the current policy will discover the feasible sequence.
- The 64-map witness is not a proof for untested banks or generators with more
  than four sections; those must fail preflight rather than be truncated.
- No benefit of a corrected reward, curriculum, GRU, sampler, partial reset, or
  action mask is claimed.
