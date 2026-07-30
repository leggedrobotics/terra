# Full curriculum visual-review receipt

Date: 2026-07-30

## Source bank

Artifact:
`/home/lorenzo/moleworks/.artifacts/terra_diverse64_full_20260730`

- 32 conditions;
- 64 accepted scenarios per condition;
- 2048 accepted scenarios total;
- zero unsatisfied generator constraints;
- 150 dig-map rerolls, all recorded;
- 1:03:16 wall time;
- 220832 KiB peak RSS; and
- source-footprint pool: 600 arrays, SHA-256
  `12a137cfc2be7949e77ae115b3885d6b7b7d545679022f2530b198814af188c3`.

Most runtime came from rejection sampling for the two composed
road-constrained trench conditions. For example:

- `trn-net3-side1-road`: 4050 start-side, 1451 road-bite, and 278 dump-component
  rejections;
- `trn-net4-side1-road`: 2016 turn/dump-coverage, 1279 road-bite, 678
  station-dump, and 500 backward-drive rejections.

The removed centred-IoU similarity rule was not an admission gate.

This source bank is review-only. Its long process began before the final
source-group and pair-slot identity repair, so accepted conditions must be
regenerated with the committed generator before split freezing.

## Image gallery

Artifact:
`/home/lorenzo/moleworks/.artifacts/terra_diverse64_full_review_20260730`

- 512 selected scenario graphics;
- 32 condition overviews;
- 512 unique scenario hashes;
- 16 displayed scenarios per condition; and
- seven explicit sibling branches: easy anchors, dump capacity, dump distance,
  dump layout, geometry/topology, site constraints, and combined constraints.

## Local review site

- URL: `http://127.0.0.1:4174/`
- Worktree:
  `/home/lorenzo/moleworks/.worktrees/terra_digging_benchmark_diverse64_review_20260730`
- Branch: `diverse64-gallery-review`
- Adapter commit: `240f38f`
- Full-gallery commit: `4a1c1a2`
- Site manifest SHA-256:
  `39f7cd2e8ce565bd384de214da5f2eee5e76764cb554e149c0ba675d815d6d51`

Site validation:

- 13 Python tests passed;
- TypeScript checks passed;
- production build passed;
- six Playwright tests passed with six expected project skips;
- comments persist across reload; and
- Accept, Reject, Quarantine, reviewer, and comment receipts export as JSONL.
