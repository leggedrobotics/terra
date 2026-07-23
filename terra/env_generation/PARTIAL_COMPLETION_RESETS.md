# Partial-Completion Resets for Terra

Status: research implementation contract

## Goal

Create normal Terra maps that start partway through an excavation task. These
maps are for early curriculum stages: the agent sees useful endgame states
without first solving the full exploration problem.

The generator does not plan an excavation sequence and does not try to find an
optimal soil distribution. It only constructs a plausible state:

- one coherent part of the target has already been dug;
- exactly the same soil volume appears in compact piles;
- piles are in the final dump zone, close to it, or split between both; and
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
If one lift would exceed the `int8` bucket capacity of 127, the action is a
no-op instead of wrapping the load.

## Generation algorithm

For each source map, completion fraction, and random seed:

1. Count target excavation tiles and compute
   `K = round(fraction * dig_tile_count)`.
2. Pick one random boundary seed per connected target component, compute
   8-neighbor distance from those seeds, and take the `K` closest target tiles
   with random tie-breaking. This produces a compact advancing excavation
   front without scattered per-tile completion.
3. Repair or reject selections that leave a one-tile excavation component.
4. Put `-1` on the selected tiles. The removed volume is exactly `K`.
5. Recompute dynamic dumpability using Terra's five-by-five hole-clearance
   rule.
6. Choose one of three pile layouts:
   - `in_zone`: all soil lies in the dump zone or its one-tile apron;
   - `near_zone`: all soil lies two to eight Manhattan tiles from the dump
     zone and outside its apron;
   - `mixed`: 60-90% lies in-zone and the remainder lies near-zone.
7. Choose one to three separated pile centers in the selected support.
   `max_piles` is the total, including both parts of a mixed state.
8. Grow each pile bottom-up. A unit is added to the closest legal support cell
   only when the increment preserves:
   - integer height;
   - the configured maximum height; and
   - a four-neighbor height difference of at most one.
9. Combine the negative completed patch and positive pile field.
10. Reject the candidate if any required check below fails. Retry with the same
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

- at least two unfinished excavation tiles remain;
- no unfinished excavation component is a singleton;
- the selected pile-mode support is respected;
- pile height and four-neighbor slope limits hold;
- at least one conservative footprint-sized spawn region remains; and
- a footprint-eroded four-neighbor free-space proxy connects spawn regions to
  remaining excavation, staged soil, and the final dump zone.

The generator uses a slightly oversized NumPy cone as a conservative load
check for staged soil outside the final dump zone. It rejects a candidate if a
possible staged lift exceeds 127. Soil already in the final zone is complete
and need not be lifted; Terra's runtime capacity guard safely rejects an agent
action that nevertheless tries to lift too much at once. The NumPy cone is a
safety proxy, not a claim that it exactly reproduces every float32 Terra
boundary tile.

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

`in_zone` is the one default path because it applies to the broadest set of
maps. Run explicit `near_zone=1.0` or `mixed=1.0` experiments on source maps
whose nearby staging area passes the static access proxy; unsupported
map/mode combinations fail instead of silently falling back to another mode.

## Minimal test gate

The high-value test set is:

1. Generate all three pile modes and verify exact mass, support, slope, and
   nonterminal state.
2. Verify a fixed seed is deterministic and a 90%-complete patch retains no
   singleton excavation component.
3. Load a generated dataset through `MapsBuffer` and `State.new`, checking that
   multi-height soil survives and initial dynamic dumpability reflects holes.
4. Exercise actual Terra lifts at the bucket boundary: accept 127 and reject
   128 without mutating the state.
5. Generate a small sample from the real review dataset and inspect/reject
   failures explicitly.

No exhaustive orientation matrix, GPU initialization, compatibility framework,
or action-planning proof belongs in this research generator.

## Known limitations

- The partial state is plausible, not a demonstrated outcome of a legal action
  history.
- Pile shape is a compact stable mound, not a soil-physics simulation.
- The access test ignores dynamic ordering effects.
- Some difficult source maps or pile modes will be rejected. That is preferable
  to silently weakening the checks.
- Curved/intersecting/new target geometries can use the same generator once
  they are present in the source dataset; the partial-reset algorithm does not
  need geometry-specific branches.
