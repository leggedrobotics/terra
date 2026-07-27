# Terra Digging Benchmark Review goal

- Goal ID: `terra-digging-benchmark-review-v1`
- Status: local controlled-capacity and B0a previews complete; owner review
  and S1-S2 admission remain active inside the Terra curriculum-recovery goal
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
- [x] Define and validate the normalized public review-data schema.
- [x] Export the current B0a design-input bank and deterministic graphics.
- [x] Build the required review surfaces and decision-data workflow.
- [x] Run data, UI, visual, and production-build verification.
- [x] Start and verify the local review server and organized example tree.
- [x] Record implementation evidence in both canonical documents.
- [x] Add the fresh controlled-capacity bank without upgrading its
  exact-loader-only, Static-pending status.
- [ ] Record Lorenzo's actual approve/reject/quarantine decisions and notes as
  deterministic JSONL and in both canonical documents.
- [ ] Swap in the source-disjoint 448-scenario S2 bank after all prerequisite
  gates pass; do not relabel the B0a preview as S2.

## Local B0a design-input preview receipt

- Site source:
  `/home/lorenzo/moleworks/terra-digging-benchmark-site` at commit
  `276482f7d79d84b92c57ae46c684f5cd6aa917c6`.
- Input: historical B0a paired-panel design input. The identity-manifest
  SHA-256 is
  `911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e`;
  the stronger review-manifest SHA-256
  `c0c27de498561bed00715b05acf7daff442a7835a37c9f0109372a93576e736b`
  also binds the file registry, source registry, validation receipt, and
  exporter source.
- Export: `256` scenarios, `16` legacy cells, `144` source groups, `1,792`
  layer PNGs, `16` overview sheets, `256` individual example composites, and
  zero sealed assets. `review-data.json` SHA-256 is
  `cbfbea5df54a03248faf2c1f294f3fe819f5478f3f4d250326bfc6cdba7622fe`;
  the complete example-tree SHA-256 is
  `3df169695a6073adce26186865ad1daae06df3b1d0f7022c33593f46ea85b968`.
  A second clean export reproduced both hashes exactly.
- Organized examples:
  `/home/lorenzo/moleworks/.artifacts/terra_digging_benchmark_review_20260727/examples`.
- Responsive screenshots:
  `/home/lorenzo/moleworks/.artifacts/terra_digging_benchmark_review_20260727/site_screenshots`.
- Verification passed: five exporter unit tests, array/PNG/tree export
  verification, Python compilation and Black checking, TypeScript checking,
  the production build, OpenNext packaging, and the Playwright review,
  comparison, curriculum-exposure, layer, and decision round trip
  (`3 passed`, `3 project-specific skips`). The importer also rejects
  missing manifest/scenario hashes.
- The owner-review correction moves the decision-and-comment editor above the
  audit details and exposes a plain `Map comment` field. It is writable before
  a decision; a comment-only `terra-map-review-record-v2` remains visibly
  decision-pending and survives exact-hash JSONL export/import. A later
  accept/reject/quarantine action preserves that comment.
- Local review URL: `http://127.0.0.1:4173`. After the comment-UX rebuild,
  `next-server` PID `3244226` was bound only to `127.0.0.1:4173`;
  both `/` and a generated map asset returned HTTP `200`.
- Scientific status: Format validation passed; live-geometry Static
  validation and Witness validation have not run, and no policy result is
  supplied. Canonical `condition_id` and `scenario_id` remain absent and the
  depth labels are provisional legacy groupings. This is unranked design
  input, not S2.
- The Playwright JSONL is synthetic workflow evidence, not Lorenzo's review.
  No owner decision has been recorded yet.
- Lorenzo's first distribution-level observation is recorded separately from
  per-map decisions: current source-bank foundation targets look small
  relative to the available site. The 16 `f_osm_all` preview identities span
  97-179 dig cells (median 155.5), or only 2.37%-4.37% of the 64 x 64 raster.
  Small tasks remain anchors, but S2 must show target-area/required-volume
  coverage and include a larger-footprint candidate review slice before
  claiming broad foundation coverage.
- `npm audit` reports nine high and zero critical transitive advisories in the
  pinned Next/OpenNext tree, with no compatible automatic fix. Loopback review
  is accepted; publication remains gated on resolving or explicitly accepting
  that dependency receipt.

## Local controlled-capacity review receipt

- Site source:
  `/home/lorenzo/moleworks/terra-digging-benchmark-site` at commit
  `91ccf87b2deafdee533bf8bd181f5642680178ae`.
- Input:
  `/home/lorenzo/moleworks/.artifacts/terra_pilot_apron_capacity_review_20260727_v1/`
  with source `files.sha256` SHA-256
  `03b5af27c3fc2acea05b6ecb001e4dcb7cdcbfe3f2677efa64ad2a6e3e7df80f`.
  The site exporter re-verifies every source file before rendering.
- Export: `64` public-train scenarios, `32` exact low/high pairs, `32` source
  groups, `2` capacity cells, `448` layer PNGs, and zero sealed assets.
  `capacity-review-data.json` has SHA-256
  `8fa452adfe47c4ffb5675e002c264cc69f250d061f6c00ae2ccf826008c0bc49`;
  the stronger review-manifest SHA-256 is
  `b287a9a5e65b71e4a00ddda61e0028d07ffd816c11e95520d296325fa0145d22`.
- Verification passed: all source/export integrity checks, seven Python
  tests, TypeScript checking, the production build, desktop paired-review and
  narrow-layout Playwright flows, review-bank switching, and exact-hash JSONL
  export/import.
- The `Map comment` field is enabled before accept/reject/quarantine. A live
  port-4173 test wrote a comment with no decision, reloaded it from local
  storage, and round-tripped it through JSONL bound to the exact capacity
  release, manifest, and scenario hashes.
- Local review URL:
  `http://127.0.0.1:4173/?build=91ccf87&bank=capacity&comments=enabled`.
  `next-server` PID `3731061` is bound to `127.0.0.1:4173`; the URL returns
  HTTP `200`. The historical B0a input remains selectable under
  **Review bank**.
- Scientific status remains non-admission: exact-loader validation passes,
  live Static validation is pending the direct-service cost gate, Witness and
  policy results are absent, and no benchmark or PPO gate is satisfied. The
  `140-189`-cell targets occupy only `3.42%-4.61%` of the map, so this surface
  isolates dump capacity and does not claim broad foundation-size coverage.

## Stop conditions

Pause and revise the canonical spec rather than hiding the problem if:

- exported records cannot be traced to exact map/source hashes;
- a graphic disagrees with the underlying map layers;
- sealed data would be exposed;
- the UI needs to infer a scientific field not present in a receipt; or
- the site architecture starts duplicating Terra generation or validation
  logic.
