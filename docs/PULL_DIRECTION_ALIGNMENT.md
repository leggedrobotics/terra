# Bulk cutting room and optional precision edges

`pull_direction_alignment=True` enables the experimental 2.5 m cutting-room
rule. It replaces the legacy chassis-aligned trench gate. The default remains
off, preserving GRU u110000's original environment.

For each fresh target cell, use its actual radial pull toward the base:

1. Find the continuous interval through that cell inside the fixed excavation
   target, stopping at holes, gaps and the outer boundary.
2. Clip that interval to the excavator's legal radial reach. The u110000 machine
   configuration has 4–6.5 m reach.
3. Require at least `dig_pull_min_length_m=2.5` m. This is the chosen macro
   allowance: approximately 0.5 m entry, 1.5 m useful cutting, 0.5 m exit.
4. If `enforce_foundation_border_alignment=True`, also require pulling within
   25 degrees of the local edge tangent in a 0.6 m inner edge band.

A wide bulk excavation can therefore be cut perpendicular to an edge. A narrow
trench rejects that direction because it lacks cutting room. At a connected
corner or junction, either branch can provide space; cells elsewhere in the
same workspace do not inherit that permission. Previously excavated target
cells still provide room for the last residual cells. Loose-soil pickup,
dumping, obstacles, chassis clearance and navigation retain their existing rules.

## Meaning of the ramp allowance

A perpendicular cut leaves a ramp. Bulk tasks accept those margins implicitly;
a requested precision edge adds the parallel-pull requirement. Terra currently
removes whole target cells and does not represent the ramp's remaining height
or volume. Thus even with precision enabled this is an admissibility model,
not a prediction of final physical surface accuracy.

The user-specified context is a 1.3 m bucket and roughly 0.7–1.0 m depth. The
2.5 m allowance is an explicit approximation, not derived from those depths:
a 0.5 m ramp at 45 degrees covers only 0.5 m of vertical change. This change
does not add a swept bucket footprint, force model or three-dimensional tool
trajectory. Raster trench width can differ from its nominal metric width.

Requiring every end cap of a narrow trench to be precise can make that task
infeasible: pulling parallel to a short end cap still needs 2.5 m of room.
There is no silent end-cap exemption. The manual bulk-trench witnesses use
precision disabled and therefore accept ramped ends. A separate finishing
primitive would be needed for cases the current action cannot finish.

## Geometry, discretization and observations

The batch constructor prepares finite contours from the immutable target,
including holes and disconnected components. It simplifies sub-cell stairs,
reducing the tolerance when necessary until every target cell center remains
strictly inside and every other center remains strictly outside. This cannot
recover continuous design geometry lost during rasterization. The 25-degree
edge tolerance accounts approximately for discrete poses and contour error;
the cabin's 60-degree workspace is not added to that tolerance.

At corners, nearest edges within a quarter-cell distance tie allow either
edge tangent. Pulls are continuous vectors even though base and cabin actions
are discrete. The cutting interval is clipped to reach, so the 2.5 m requirement
uses the entire u110000 radial span. Separate passes may be needed for different
rows of a narrow raster trench.

The existing admissible/executable dig observations use the same fresh-cell
permission as native DO. The edge band, edge error and border-diggable channels
follow the optional precision requirement. Input dimensions are unchanged.
Legacy chassis yaw/standoff errors become neutral in the new mode, and old
trench reward shaping must remain disabled. There are no reward bonuses or
termination relaxations.

## Configuration

```python
env = TerraEnvBatch(..., pull_direction_alignment=True)
cfg = EnvConfig(
    pull_direction_alignment=True,
    dig_pull_min_length_m=2.5,
    enforce_foundation_border_alignment=False,  # Bulk, accepting ramped margins.
    edge_band_width_m=0.6,
    edge_pull_tolerance_rad=0.436332313,         # 25 degrees if precision is on.
    apply_trench_rewards=False,
)
```

Set `enforce_foundation_border_alignment=True` for precision-edge environments.
It can differ across entries in a batched `EnvConfig`, allowing a mixture of
bulk and precision tasks without changing the policy input dimensions. There
is no per-segment precision mask in this change. All entries in a batch must
use the same geometry-preparation mode.

The paired baselines exposes the same fields, checkpoint/evaluation forwarding,
and the `gru_generalist_512_pull_direction` preset. The old experimental
`trench_pull_tolerance_rad` field remains reserved for positional compatibility;
the current rule does not impose a separate trench-axis angle threshold.

Low-level `State.new` callers must supply records from
`boundary_records_from_mask(target < 0)`: `[256,7]`, columns
`A,B,C,row0,col0,row1,col1`, sentinel -97 padding. Prepared geometry has its own
record count; the original foundation metadata count is preserved for existing
dump rules. Missing or malformed geometry fails closed for fresh excavation.
Trench-axis metadata is not needed by the new rule.

## Validation and provenance (October 6 snapshot)

The following records the pre-training validation on October 6. Later manual
inspection and its incomplete native foundation attempt are described in the
[manual viewer documentation](../terra/viewer3d/README.md#pull-rule-inspector).

This isolated Terra worktree starts at `d1d128bb`, descended from the u110000
machine-rule runtime `ef406998`; its paired baselines starts at `5d52f9f`.
The separate multiagent workspace branch `31107708` is not part of this change.
Old EnvConfig pickle positions are unchanged; new fields are appended.

All 20,480 training-bank slots passed the strict cell-center classification
check. They require 4–214 finite segments, so the previous capacity of 128 was
increased to 256. Boundary storage is 140 MiB for that bank, excluding other
arrays. Overflow raises an error instead of truncating geometry.

The previous strict angle-only draft scored 14/32 road-network starts with
GRU u110000, versus 32/32 under its legacy rules, at horizon 450. Those numbers
belong to the earlier draft and must not be attributed to this cutting-room
rule. They are preserved in `trench_validation_20261006/comparison.json`.
The new rule is a policy-transfer test, not a replay of the training MDP.

Current manual and geometric evidence is in
`/home/lorenzo/moleworks/.artifacts/terra_pull_direction_20261006/manual_feasibility_20261006/`.
The straight native witness excavates and legally disposes all 70 cells, reaches
native success at action 270, and drives clear at action 281. The L-shaped
corner witness also excavates and disposes all 70 cells, succeeds at action
283, and drives clear at action 307. Both preserve native mass, target,
obstacle, disposal and navigation rules. These are manually authored native
action plans on chosen synthetic cases, not learned-policy performance.

The new 2.5 m bulk rule was also evaluated on the same 32 road-network starts:
**0/32 complete**, mean excavation **12.74%**, and all 32 stall for more than
100 steps (horizon 450). The legacy control remains 32/32. Both arms conserve
mass and preserve targets/obstacles, with finite states. See
`trench_validation_20261006/bulk_stroke_comparison.json`. This candidate is
therefore **not a drop-in replacement for the frozen GRU policy**. The evidence
does not justify enabling it by default or claiming general trench solvability.

A bounded static audit of that exact road map found at least 6 valid integer
base positions for every target cell, with all 143 cells covered. This includes
native chassis bounds and the final 2.5 m helper. It excludes a permanently
unreachable individual-cell explanation for this map, but does not prove an
ordered full-plan/disposal/egress solution.

Current CPU checks pass: 16 legacy trench tests, 27 frozen-benchmark/machine-rule
tests, 7 contour tests, 7 stroke-geometry tests and the new native behavior
cases (including observation/DO agreement, loose-soil recovery and mixed
precision settings). Paired baselines passes 14 focused configuration/loading
checks. The independent final-helper audit validates every removed cell in
both native witnesses. Both repository diffs pass whitespace checks.

The CUDA policy-evaluation path completes, but a disposable two-environment,
two-step PPO update hit its 900-second process limit while compiling (exit124,
zero updates). Its convolution/backward preflight passed. The first-update
training gate remains **UNVERIFIED**; this is not a training-ready promotion.
No production training was launched and the source checkpoint is unchanged.
