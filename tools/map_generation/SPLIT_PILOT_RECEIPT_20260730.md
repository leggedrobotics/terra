# Split-ready pilot receipt

Date: 2026-07-30

This receipt exercises the real generator and materializer with the pilot split
arithmetic. It is implementation evidence, not the frozen curriculum bank.

## Final-code acceptance smoke

The final generator identity logic generated one scenario for every registered
condition:

- 32/32 conditions accepted;
- 32 scenarios accepted;
- zero unsatisfied constraints;
- 92.21 seconds wall time; and
- 111760 KiB peak RSS.

Artifact:
`/home/lorenzo/moleworks/.artifacts/terra_diverse_finalcode_smoke_20260730`

## Oversized candidate

The generator produced 160 candidates for each representative family anchor:

- `fnd-slab-ring3x`: 160/160;
- `trn-straight-side2`: 160/160;
- 320 total accepted scenarios;
- 13 trench dig rerolls recorded rather than hidden;
- zero unsatisfied constraints;
- 150.93 seconds wall time; and
- 119408 KiB peak RSS.

Artifact:
`/home/lorenzo/moleworks/.artifacts/terra_split_pilot_candidates_20260730`

## Exact split materialization

`materialize_splits.py` selected 128 scenarios per condition with the exact
requested counts:

| Condition | train | promotion | development | sealed |
|---|---:|---:|---:|---:|
| `fnd-slab-ring3x` | 64 | 16 | 16 | 32 |
| `trn-straight-side2` | 64 | 16 | 16 | 32 |

Post-materialization checks:

- 256/256 unique scenario identities;
- zero realized-source groups crossing splits;
- no incomplete selected pair slots;
- assignment SHA-256
  `c95769083e46f1058cb02aa70662c1fa65fe1a6ebb4945349d9231186ec4e5c9`;
- source-manifest SHA-256
  `263a4cdc2c69577661842a31415caafc50d2cc375f73309a6039c7ef1835091f`.

Artifact:
`/home/lorenzo/moleworks/.artifacts/terra_split_pilot_materialized_20260730`

This proves that removing the centred-IoU wall is sufficient to generate and
materialize the requested pilot counts for representative foundation and
trench anchors. It does not approve the remaining conditions or replace
Lorenzo's visual review.
