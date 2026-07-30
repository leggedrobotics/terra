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

The `generate_prototypes*.py` modules are a private snapshot of the reviewed
v6 generator lineage. They are retained to avoid changing validated geometry,
capacity, lane, obstacle, and proximity semantics during the diversity repair.
Do not invoke them directly.
