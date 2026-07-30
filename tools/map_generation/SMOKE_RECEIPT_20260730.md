# Diverse-bank generator smoke receipt

Date: 2026-07-30

## Command

```bash
/home/lorenzo/moleworks/.venv-terra-uv/bin/python \
  tools/map_generation/generate_curriculum_bank.py \
  --source-foundations \
  /home/lorenzo/moleworks/.artifacts/terra_map_audit_20260723/full_data/foundations_dumpzones_v3 \
  --output \
  /home/lorenzo/moleworks/.artifacts/terra_diverse64_smoke_20260730 \
  --maps 64 \
  --only fnd-slab-ring3x,trn-straight-side2
```

## Result

- `fnd-slab-ring3x`: 64/64 accepted.
- `trn-straight-side2`: 64/64 accepted.
- 128 scenarios total.
- Zero exact full-scenario duplicates.
- Zero unsatisfied constraints.
- Source pool: 600 arrays, SHA-256
  `12a137cfc2be7949e77ae115b3885d6b7b7d545679022f2530b198814af188c3`.
- Two independent trench-only runs produced the identical manifest SHA-256
  `4bd3d72760c9d93952825e81a8087bd8518c8286a0a6f83b6760c9d9675d7b5b`.
- The timed trench-only run took 42.87 seconds wall time and 178128 KiB peak
  RSS.

The first generated foundation scenario reproduces the reviewed v6 bank
byte-for-byte across target, occupancy, dumpability, action, and distance
arrays. The five reference/output SHA-256 pairs are identical:

| Array | SHA-256 |
|---|---|
| target | `177695ef97e9640863458087b8ea8e816559a74b3e99acf72287e64bbba06244` |
| occupancy | `c1cf3cde9f391134c25953ffffb43c872dd154f4e621fb5790b0d0400a3f10c1` |
| dumpability | `59424a138729536c55dd8b1cf67ac0d55a4f0af377e78b8fdf0ff262d958edd3` |
| action | `7e3d8926596dd48a0806979e04e4a3f9447d391296949535286eb00bb201a771` |
| distance | `13999732155ef720c01073b410ac7d80758f862123b6cd99c7364ad5312ad323` |

Centred IoU is diagnostic only. The observed nearest-neighbour maxima were
`0.895` for the foundation condition and `1.000` for the trench condition.
These maps are not exact scenarios: both conditions still have 64 unique
scenario identities. This is the intended evidence that the former centred-IoU
gate rejected usable training variation.
