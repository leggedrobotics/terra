# Terra Digging Benchmark Review goal

- Goal ID: `terra-digging-benchmark-review-v1`
- Status: active; first priority inside the Terra curriculum-recovery goal
- Owner-facing outcome: Lorenzo can inspect, compare, and record decisions on
  the actual training-map distribution through a graphical review site while
  benchmark admission work continues.
- Canonical scientific contract:
  [`MAP_BENCHMARK_SPEC.md`](MAP_BENCHMARK_SPEC.md)
- Canonical execution ledger:
  [`TRAINING_TASKS.md`](TRAINING_TASKS.md)

This goal does not replace either canonical document. It turns their
distribution and website requirements into one reviewable deliverable and
records completion evidence back into both.

## Outcome

Deliver a versioned Terra digging benchmark inspector that:

1. normalizes the current map bank into explicit family, geometry, dump,
   capacity, separation, work, site, split, source-group, and validity fields;
2. renders every public scenario with inspectable target, exact legal dump,
   occupancy, initial-soil, and derived-measurement graphics;
3. makes causal parent/counterfactual pairs and the
   Anchor/One-axis/Composed admission graph easy to compare;
4. exposes exact counts, weights, support gaps, repeats, duplicates, split
   leakage, and distribution tails instead of only attractive galleries;
5. exports Lorenzo's approve/reject/quarantine decisions and notes as
   deterministic data; and
6. is available through a reproducible local review server, with a separate
   Nautilus-friendly example tree organized by provisional review depth,
   family, and condition.

The first review dataset is the current 256-row B0a bank and is visibly
labelled **design input, not a frozen benchmark release**. The accepted S2
448-scenario pilot replaces this input only after its S1 schema, capacity,
support, split, and validation gates pass. No sealed scenario raster or
identity is published.

## Build boundary

- Terra owns map identities, conditions, generators, normalized manifests, and
  validation receipts.
- The review-site repository consumes exported public-safe JSON and graphics
  and owns no scientific truth.
- The site never executes a policy or validator.
- The implementation stays static-first and small: one deterministic exporter,
  one data contract, and one client application. It does not introduce a
  database, authentication service, or parallel environment implementation.
- Website work may proceed beside exact-validator optimization, but it cannot
  mark a scenario Static-valid, Witness-valid, admitted, or ranked without the
  corresponding canonical receipt.
- No PPO run is authorized by this goal.

## Required review surfaces

- Overview with dataset status, source hashes, counts, splits, conditions, and
  integrity warnings.
- Map Explorer with filters, stable scenario URLs, layer controls, exact
  measurements, provenance, and validity badges.
- Parent/counterfactual comparison for capacity, separation, side access,
  source geometry, and trench topology.
- Distribution dashboard with exact marginal and joint counts, capacity,
  separation, volume, topology, source-group diversity, pair coverage, and
  rejection/support warnings.
- Curriculum view that distinguishes training exposure from benchmark
  exposure and identifies missing or overrepresented conditions.
- Deterministically ordered review queue with approve, reject, quarantine, and
  note actions plus JSONL export/import.
- A standalone `examples/` tree with Anchor/One-axis/Composed preview folders,
  then family and condition folders, each containing a fixed overview sheet
  plus individual composites and a short README. Because B0a predates the
  canonical admission graph, every such depth label is explicitly
  provisional.
- Clear empty states for policy heatmaps, witnesses, and leaderboard data that
  are intentionally unavailable before S3-S5.

## Verification contract

The primary verifier is a clean rebuild from pinned input hashes followed by:

1. manifest/card/thumbnail count and identity-hash agreement;
2. deterministic export reproduction;
3. graphics checks against the underlying 64 x 64 arrays;
4. filter, stable-link, pair-navigation, layer-toggle, and decision
   export/import tests;
5. responsive screenshots of overview, explorer, comparison, distribution,
   curriculum, and review-queue states;
6. a production build with no missing assets or browser errors; and
7. a locally served build whose URL, process receipt, and review paths are
   recorded.

The final proof recorded in both canonical documents includes the input
receipt, exporter/site commits, build and test commands, screenshot paths,
local server and example-folder paths, and a list of limitations still
blocking S2 admission. A Sites project may be reserved, but saving or
deploying a production version is deferred until Lorenzo asks.

## Execution checklist

- [x] Freeze this goal and bind it to both canonical documents.
- [ ] Define and validate the normalized public review-data schema.
- [ ] Export the current B0a design-input bank and deterministic graphics.
- [ ] Build the required review surfaces and decision-data workflow.
- [ ] Run data, UI, visual, and production-build verification.
- [ ] Start and verify the local review server and organized example tree.
- [ ] Record review decisions and implementation evidence in both canonical
  documents.
- [ ] Swap in the source-disjoint 448-scenario S2 bank after all prerequisite
  gates pass; do not relabel the B0a preview as S2.

## Stop conditions

Pause and revise the canonical spec rather than hiding the problem if:

- exported records cannot be traced to exact map/source hashes;
- a graphic disagrees with the underlying map layers;
- sealed data would be exposed;
- the UI needs to infer a scientific field not present in a receipt; or
- the site architecture starts duplicating Terra generation or validation
  logic.
