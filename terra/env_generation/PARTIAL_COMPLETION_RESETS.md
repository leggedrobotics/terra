# Partial-Completion Resets for Terra

Status: research implementation contract

## Goal

Create normal Terra maps that start partway through an excavation task. These
maps are for early curriculum stages: the agent sees useful endgame states
without first solving the full exploration problem.

The generator does not plan an excavation sequence. It constructs a plausible
state for a narrow skill curriculum:

- one coherent part of the target has already been dug;
- exactly the same soil volume appears in compact piles;
- piles are in the final dump zone, close to it, split between both, or placed
  in a natural source-to-terminal relay corridor; and
- enough excavation remains for the episode to continue.

The first implementation supports the current 64 x 64, one-depth, solo
excavator configuration: a 36.5714 m map with Terra's derived 7 x 11 tile
footprint. Unsupported shapes or nonzero source action maps fail loudly.

## Terra map contract

The target map is unchanged:

- `-1`: tile that must eventually be excavated;
- `0`: neutral tile;
- `+1`: final dump zone.

The generated action map stores the partial state:

- `-1`: already excavated by one depth unit;
- `0`: unchanged ground;
- `+1`, `+2`, ...: soil height.

Multiple excavated tiles can therefore be represented in one pile. For
example, a pile with heights `[1, 2, 1]` contains four units of soil. There is
no need to change the meaning of `-1`.

Terra must accept positive action-map heights when loading a dataset, derive
initial dynamic dumpability from the loaded holes, and sum a lift in `int32`.
Relifts take up to the 127-unit bucket capacity and leave the remaining pile in
place. A large staged pile therefore teaches repeated partial pickup rather
than becoming an artificial no-op.

## Generation algorithm

For each source map, completion fraction, and random seed:

1. Count target excavation tiles and compute
   `K = round(fraction * dig_tile_count)`.
2. Select the `K` completed target tiles:
   - ordinary pile modes grow a compact front from one random boundary seed
     per target component;
   - `relay_corridor` roots every target component at a terminal-facing cell
     and reverse-deletes only boundary cells whose removal preserves
     four-neighbor connectivity to that root. Candidates farther from the
     terminal and root are removed first. On branching trenches this peels
     leaf/prong work before the access spine; at higher completion it consumes
     the spine from its ends instead of cutting through its middle. This rule
     is inferred from target topology and does not depend on named map types or
     authored trench axes. It guarantees connected remaining target material,
     not exact excavator footprint or workspace reachability; the normal access
     validation remains a separate gate.
3. Put `-1` on the selected tiles. The removed volume is exactly `K`. A single
   remaining target tile is valid and actionable.
4. Recompute dynamic dumpability using Terra's five-by-five hole-clearance
   rule.
5. Choose a pile layout:
   - `in_zone`: all soil lies in the dump zone or its one-tile apron;
   - `near_zone`: all soil lies two to eight Manhattan tiles from the dump
     zone and outside its apron;
   - `mixed`: 60-90% lies in-zone and the remainder lies near-zone.
   - `relay_corridor`: choose the most upstream completed tile, compute an
     obstacle-aware four-neighbor route to the terminal, and place compact
     soil in one compact pile within a one-machine-width pocket around the
     early part of that route. The pile center may sit one cell beside a
     shortest route (at most three added route steps, including grid
     discretization). Centers are sought about one workspace reach downstream
     of the source, with four tiles of along-route tolerance so a pile can use
     a connected staging pocket. This keeps soil on the natural working
     direction rather than in a remote corner. The manifest records whether
     conservative pickup and terminal
     service-center proxies overlap; that field is not an exact relocation
     proof.
6. Relay mode uses one pile. Other modes choose one to three separated pile
   centers; `max_piles` is the total, including both parts of a mixed state.
7. Grow each pile bottom-up. A unit is added to the closest legal support cell
   only when the increment preserves:
   - integer height;
   - the configured maximum height; and
   - a four-neighbor height difference of at most one.
8. Combine the negative completed patch and positive pile field.
9. Reject the candidate if any required check below fails. Retry with the same
    requested fraction and pile mode up to the configured bounded attempt
    count; then fail with the last concrete reason.

This is intentionally not random per-tile soil allocation. Soil is accumulated
into compact multi-height mounds around a few centers.

## Required checks

Every emitted action map must satisfy:

- integer values in `[-1, 127]`;
- negative cells occur only on target excavation;
- positive cells overlap neither excavation targets nor obstacles;
- positive cells are initially dynamically dumpable;
- exact mass conservation:

  ```text
  sum(positive heights) == number of completed -1 tiles
  ```

- at least one unfinished excavation tile remains;
- the selected pile-mode support is respected;
- relay mode contains exactly one four-neighbor-connected staged pile;
- pile height and four-neighbor slope limits hold;
- at least one conservative footprint-sized spawn region remains; and
- a footprint-eroded four-neighbor free-space proxy connects spawn regions to
  remaining excavation, staged soil, and the final dump zone.

The manifest reports the largest staged-soil volume seen by a slightly
oversized NumPy cone, the corresponding lower bound on workspace pickups, and
the minimum number of 127-unit bucket loads. These are difficulty diagnostics,
not rejection thresholds.

The spawn and connectivity checks are also static feasibility proxies. They do
not prove that an action sequence exists, and they must not be described as
Terra action-level certification.

## Dataset output

The output keeps Terra's ordinary folder layout:

```text
images/img_N.npy
occupancy/img_N.npy
dumpability/img_N.npy
distance/img_N.npy
actions/img_N.npy
metadata/trench_N.json       # when present in the source
```

Target, occupancy, static dumpability, distance, and optional metadata are
copied unchanged. Only `actions/img_N.npy` is new.

A JSON-lines manifest records the source index, fraction, mode, seed, completed
volume, pile centers, pile heights, and rejected-candidate count. The generator
never overwrites an existing output path.

Example:

```bash
python tools/generate_partial_completion_dataset.py \
  --input /path/to/full_dataset \
  --output /path/to/partial_dataset \
  --completion-fractions 0.25,0.50,0.75,0.90 \
  --variants-per-fraction 2 \
  --mode-weights in_zone=1.0 \
  --seed 0
```

For relocation training, use one explicit path:

```bash
python tools/generate_partial_completion_dataset.py \
  --input /path/to/full_dataset \
  --output /path/to/relay_dataset \
  --completion-fractions 0.50,0.75,0.90 \
  --mode-weights relay_corridor=1.0 \
  --seed 0
```

`in_zone` is the one default path because it applies to the broadest set of
maps. Run explicit `near_zone=1.0` or `mixed=1.0` experiments on source maps
whose nearby staging area passes the static access proxy; unsupported
map/mode combinations fail instead of silently falling back to another mode.

## Relay curriculum progression

The first pilot should mix no more than 25% partial-reset lanes with ordinary
full starts. Move from late cleanup back toward the real initial-state
distribution:

1. `R0`: 90% completed, with conservative pickup and terminal service-center
   proxies still overlapping. This is the easiest relift-and-cleanup reset.
2. `R1`: 75-90% completed with
   `relay_no_shared_conservative_proxy_center=true`. This selects likely
   stage-move-relift cases for inspection; it is not proof that relocation is
   required under exact Terra poses.
3. `R2`: 50-75% completed, then ordinary full starts. This bridges cleanup to
   excavation plus cleanup without inventing remote or lateral pile geometry.

Fractions are independently generated compact fronts, not snapshots from one
demonstration. This is Backplay-inspired start-distribution shaping, not exact
Backplay, and completion fraction must not be used as a proxy for relay length.

Anneal the partial fraction toward zero and judge promotion only on untouched
full-start maps. Partial-reset successes must not update the full-start
curriculum mastery EMA.

## Minimal test gate

The high-value test set is:

1. Generate all pile modes and verify exact mass, support, slope, and
   nonterminal state.
2. Verify a fixed seed is deterministic and a one-tile remainder remains valid.
3. Load a generated dataset through `MapsBuffer` and `State.new`, checking that
   multi-height soil survives and initial dynamic dumpability reflects holes.
4. Exercise actual Terra partial relifts above the bucket boundary and verify
   exact mass conservation across repeated pickup/dump cycles.
5. Generate a small sample from the real review dataset and inspect/reject
   failures explicitly.

No exhaustive orientation matrix, GPU initialization, compatibility framework,
or action-planning proof belongs in this research generator.

## Known limitations

- The partial state is plausible, not a demonstrated outcome of a legal action
  history.
- The relay corridor and workspace-handoff fields are static geometry
  diagnostics, not an executable action witness.
- `relay_no_shared_conservative_proxy_center` is a difficulty hint from an
  under-approximating static pose model, not a proof that exact Terra base
  relocation is necessary.
- Pile shape is a compact stable mound, not a soil-physics simulation.
- The access test ignores dynamic ordering effects.
- Some difficult source maps or pile modes will be rejected. That is preferable
  to silently weakening the checks.
- Curved/intersecting/new target geometries can use the same generator once
  they are present in the source dataset; the partial-reset algorithm does not
  need geometry-specific branches.
