# Trench axis contract v2 — implementation draft

Date: 2026-08-25
Branch: `experiment/trench-axis-contract-v2-20260825`

## Decision

Use one exact representation from generation through runtime:

1. up to four undirected trench axes as line equations `A*x + B*y + C = 0`;
2. one `uint8[H,W]` map whose bit `i` means that target cell belongs to axis
   `i`.

The environment does not infer finite segments, widths, or intersection
ownership. Foundations carry no trench axes and an all-zero owner map.

## Why this is the smallest correct contract

The old runtime reconstructed section ownership from endpoints, half-widths,
nearest-axis rules, and a raster-fringe exception. That duplicated generator
geometry and made junction behavior depend on an approximation reconstructed at
load time.

The generator already knows which arms produced each target cell. Persisting
that result removes the reconstruction and makes intersections explicit:

- an ordinary arm cell has one bit;
- a true junction cell has multiple bits;
- regularized edge cells are assigned once, deterministically, to the nearest
  generated finite arm;
- every trench target cell must have at least one bit;
- non-target cells must have no bits.

## Runtime rule

For a fresh trench dig, an axis is pose-valid when:

- the base is parallel to the undirected axis within the configured yaw
  tolerance; and
- perpendicular standoff is inside the configured metric band.

A selected fresh cell is valid when its owner bits intersect the pose-valid
axis bits. `DO` remains atomic: if any selected fresh trench cell is invalid,
the whole fresh dig is rejected. The rule is neutral for:

- foundations and other non-trench maps;
- already-dug cells;
- relifting positive soil;
- loaded-agent dumping; and
- non-excavator embodiments.

Missing or inconsistent trench ownership fails closed at loading, batch reset,
and the lower-level state predicate.

## Orientations

The runtime formula accepts arbitrary continuous `A,B,C` axes. The current
general bank samples a 15 degree trench lattice. The tracked base retains 12
headings at 30 degree spacing, and the yaw tolerance is exactly half a bin
(`pi/12`), inclusive with a cosine-domain numerical guard. Therefore every
generated orientation has at least one discrete aligned base heading, including
the exact 15 degree half-bin cases.

The generator's U7 backward-drive witness uses the same set. At a half-bin it
may alternate the two equally aligned base headings, choosing the next move
that minimizes lateral centre-line error. This preserves the existing drift
and footprint gates without falsely requiring a 15 degree base-heading state
that Terra does not have.

This changes orientation diversity inside the existing trench conditions; it
does not add or remove foundation or trench conditions.

## Partial resets

Partial resets need no trench-specific representation. They retain the
canonical target, axes, owner map, obstacles, and dumpability, and replace only
the action map. Runtime alignment quantifies over cells satisfying:

`target < 0 AND action == 0`.

Already-dug trench cells therefore leave the fresh set automatically. Relifted
or staged positive soil remains outside the fresh-dig gate. The same partial
reset bank can cover foundations and trenches as long as its source identity is
bound to the newly generated canonical bank.

## Feasibility versus policy learning

Static admission is intentionally limited to three claims:

- **A0 — ownership:** every trench target cell has valid declared owner bits;
- **A1 — pose cover:** every trench target cell is reachable by some
  obstacle-clear, yaw/standoff-valid base pose;
- **A2 — atomic cover:** every trench target cell appears in at least one fully
  admissible Terra `DO` cone.

The admission audit does not solve or filter on navigation order, re-approach,
spoil placement, rehandling, dump capacity, or episode horizon. Those are the
strategic capabilities the policy must learn and the rollout benchmark must
measure.

## Constrained-map admission

The 320-map production pass exposed a separate generator issue in the two U10
road conditions. A map slot held its one-sided layout fixed across every
candidate. Four `trn-net3-side1-road` slots exhausted all 320 candidates even
though different layouts satisfy every unchanged gate; their best fixed-layout
`plan_delta` values were only 0.074–0.099 against the required 0.10. One
`trn-net4-side1-road` slot needed a different layout before candidates could
satisfy the unchanged turn-dump and station-coverage gates. One shared source
slot in the normal and tight one-sided straight conditions likewise exhausted
its fixed layout on backward-drive clearance.

Trench and planning slots now make at most four deterministic layout re-draws,
each only after the preceding layout exhausts all 320 candidates. Candidate
geometry, road, capacity, lane, backward-drive, turn-dump, and planning
thresholds are unchanged. The four exhausted net3 slots pass the same planning
gate after the first re-draw (`plan_delta` 0.107–0.214); the net4 slot passes on
re-draw four with `plan_delta=0.11224`, strict turn-dump coverage `1.0`, and
station-dump fraction `0.70182`. Both straight variants pass on re-draw one at
attempt 120. Independent map slots may run in spawned CPU workers, but results
are consumed in map-index order. If a rerolled dig duplicates an earlier
accepted dig in that condition, only that slot is replayed in the parent with
the accumulated exact identities so the old serial continue-search semantics
are preserved. Scenario arrays and manifests must be byte-identical between
serial and parallel generation; the generation receipt records the requested
worker count.

## Data path

The owner sidecar travels through the one supported path:

`generate_curriculum_bank.py`
→ `materialize_splits.py`
→ `materialize_loader_bank.py`
→ `load_maps_from_disk()`
→ `MapsBuffer`
→ `State/GridWorld`.

Each trench metadata file binds the owner array with
`trench_axis_contract=generator_owner_bits_v1` and a SHA-256 value. New strict
trench runs require a regenerated bank. Old finite-enriched trench banks are
not silently upgraded at runtime.

## Code reduction

The implementation removes the runtime membership reconstruction, the finite
metadata enrichment/validation path, the partial-reset alignment auditor, the
pilot-specific finite-bank pooler, and the old whole-task feasibility search.
The replacement auditor reports only A0–A2. Map generation keeps one direct
entry point; `--workers` changes execution speed, not generated content.

## Verification gates

Before any training submission:

1. focused state tests cover half-bin orientation, multi-owner intersections,
   atomic rejection, fail-closed ownership, foundation parity, and relifts;
2. generator tests verify exclusive arm bits and multi-bit junctions;
3. split/loader tests verify the sidecar survives materialization;
4. one generated trench map loads through the exact contract;
5. a representative map from every trench geometry family passes A0–A2;
6. partial-reset loading and reset-tier tests remain green;
7. a bounded CPU or GPU environment canary reaches reset and one finite update.

## Training and benchmark sequence

After the implementation gates pass:

1. generate a small all-trench orientation panel and run A0–A2;
2. regenerate the mixed foundation-plus-trench bank with zero owner maps for
   foundations;
3. regenerate or rebind the general partial-reset bank to those source IDs;
4. run a short mixed-policy canary with strict alignment enabled only on
   declared trench maps;
5. benchmark the finished trench specialist and candidate generalist on a
   fixed panel;
6. export trajectory JSON plus GIFs grouped by geometry, orientation, outcome,
   and failure reason;
7. launch the full generalist only after the fixed-panel and artifact-provenance
   receipts are complete.

No production training launch is part of this implementation commit.
