# Curriculum map generation

`generate_curriculum_bank.py` is the only supported entry point. It preserves
the reviewed v6 map semantics while admitting any valid, non-duplicate
scenario. Centred IoU is written to `diversity_report.json`; it does not reject
training maps.

Example:

```bash
python tools/map_generation/generate_curriculum_bank.py \
  --source-foundations /path/to/foundations_dumpzones_v3 \
  --output /path/to/empty/output \
  --maps 64
```

Use `--only condition-a,condition-b` for a bounded generator smoke. The command
fails on unknown conditions, a non-empty output directory, missing source data,
unfilled conditions, or exact duplicate full scenarios.

The generator writes all arrays and manifest rows but renders only 16 evenly
spaced provisional review examples per condition. Set `--review-examples 0` for
data-only generation. The later review exporter may replace this provisional
subset with descriptor-selected examples without changing the bank.

Each row carries two distinct identities:

- `source_group_id` hashes the raw OSM footprint, or the realized procedural
  dig mask when there is no raw source;
- `pair_slot_id` is the declared counterfactual bank slot.

Rerolls can change a condition's realized dig inside one pair slot.
`materialize_splits.py` drops those non-identical pair slots, selects an exact
deterministic set from an oversized candidate bank, keeps every retained pair
slot in one split, and then fails if any realized source would leak across
splits:

```bash
python tools/map_generation/materialize_splits.py \
  --manifest /path/to/candidate/manifest.csv \
  --dataset /path/to/candidate/dataset \
  --output /path/to/new/split-bank \
  --train 64 \
  --promotion 16 \
  --development 16 \
  --sealed 32
```

Generate more than 128 candidate maps per condition so rerolled pair slots can
be discarded without shrinking a split. The materializer does not search for a
near-feasible allocation: it either meets the exact contract or fails.

Create the human-review folder without mutating the bank:

```bash
python tools/map_generation/export_review_gallery.py \
  --bank /path/to/generated/bank \
  --output /path/to/new/review-folder
```

The gallery groups conditions into explicit sibling branches rather than one
ambiguous “one-axis” level. Its `index.csv` pins every displayed image to its
scenario identity and provides editable `decision` and `comment` columns.

The `generate_prototypes*.py` modules are a private snapshot of the reviewed
v6 generator lineage. They are retained to avoid changing validated geometry,
capacity, lane, obstacle, and proximity semantics during the diversity repair.
Do not invoke them directly.

Implementation receipts:

- [`SMOKE_RECEIPT_20260730.md`](SMOKE_RECEIPT_20260730.md): reproducibility and
  representative 64-map generation;
- [`SPLIT_PILOT_RECEIPT_20260730.md`](SPLIT_PILOT_RECEIPT_20260730.md):
  final-code 32-condition acceptance plus real `160 -> 64/16/16/32`
  materialization.
