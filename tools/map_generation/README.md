# Curriculum map generation

The canonical current `v6-main` condition registry, token glossary, anchor/tier
semantics, exact 32-condition table, bank counts, and P5 sampler boundary are in
[`CURRICULUM_TAXONOMY.md`](../../CURRICULUM_TAXONOMY.md). This page documents
the one supported generation/materialization path.

`generate_curriculum_bank.py` is the only supported entry point. It preserves
the reviewed v6 map semantics while admitting any valid, non-duplicate
scenario. Centred IoU is written to `diversity_report.json`; it does not reject
training maps.

Trench targets use the v2 axis contract: `axes_ABC` plus a generated
`uint8` `trench_axis_owners` sidecar. Junction cells may own several axes;
foundation owner maps are zero. The current generator samples trench global
orientations on a 15 degree lattice, while Terra's 12 base headings and
inclusive 15 degree tolerance keep every generated axis locally alignable.

Before generating a P5 accepted bank, compile the manifest-bound review export
into an explicit condition list:

```bash
python tools/map_generation/compile_condition_review.py \
  --review-data /path/to/curriculum-diverse64-review-data.json \
  --decisions /path/to/review-decisions.jsonl \
  --output /path/to/new/review-admission
```

The compiler requires exactly one Accept, Reject, or Quarantine disposition
for every condition, exact release and manifest identity, and at least one
accepted easy anchor for each family. It validates any included map records but
only for their pinned release and scenario identity; it never infers a
condition decision from map votes or comments. The accepted
condition IDs are written as one comma-separated line in
`accepted_conditions.txt`; pass that explicit set to `--only` below.

Example:

```bash
python tools/map_generation/generate_curriculum_bank.py \
  --source-foundations /path/to/foundations_dumpzones_v3 \
  --output /path/to/empty/output \
  --maps 160 \
  --only "$(cat /path/to/review-admission/accepted_conditions.txt)" \
  --review-examples 0
```

The 160 candidates leave room for the exact `64/16/16/32` split after dropping
rerolled pair slots. Use `--only condition-a,condition-b` for a bounded
generator smoke. The command
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

Convert the reviewed split bank into the one loader-ready training/evaluation
layout:

```bash
python tools/map_generation/materialize_loader_bank.py \
  --split-bank /path/to/split-bank \
  --output /path/to/new/accepted-bank \
  --review-admission /path/to/review-admission/review_admission.json \
  --terra-revision "$(git rev-parse HEAD)"
```

`accepted-bank/dataset.json` is the public bank index. Its `train` entries
point to equal-size, per-condition Terra loader levels; `promotion`,
`development`, and `sealed` are contiguous evaluation panels. Every level or
panel contains contiguous arrays plus `manifest.jsonl`, and all of them bind to
the same hashed `source_registry.jsonl`. Evaluation rows add a deterministic
`reset_seed` that selects that row's exact contiguous map slot and
`episode_id = hash(scenario_id, reset_seed, environment_protocol_sha256)`.
`scenario_id` is recomputed from the five canonical scenario arrays. Trench
axis equations and owner bits are geometry metadata, bound separately by the
owner SHA-256 in each `metadata/trench_N.json`. The command
rejects review-only inputs, split leakage, count/support mismatches, identity
collisions, and array/manifest hash disagreement before publishing the output.
Each published level declares `terra_reset_arrays_sha256_v1`; the live loader
recomputes that identity and compares the arrays, manifest, and source registry.
Historical exact datasets remain loadable only when explicitly labeled
`terra_legacy_map_id_v0`.

The loader bank copies the validated `review_admission.json`, hashes it in the
root `dataset.json`, and requires its explicit accepted condition IDs to equal
the split bank's training conditions exactly. The Euler campaign rejects banks
without that binding.

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
  materialization;
- [`FULL_REVIEW_RECEIPT_20260730.md`](FULL_REVIEW_RECEIPT_20260730.md):
  the complete 32-condition visual bank, seven-branch gallery, and local-site
  verification.
