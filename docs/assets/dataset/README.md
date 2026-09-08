# Dataset figures and counts

Generated on September 8, 2026 from saved arrays. The figures show initial task
maps, not policy results, reconstructed terrain, or simulated soil properties.
The accompanying [dataset reference](../../DATASET.md) defines the banks and
categories and records the full audit findings.

| Figure | Contents | Exports |
|---|---|---|
| Task geometry | Six foundation and seven trench conditions added by V7 within V8 | [SVG](terrain_geometry.svg), [PNG](terrain_geometry.png) |
| Site constraints | Eight constraint conditions with the same excavation footprint | [SVG](site_constraints.svg), [PNG](site_constraints.png) |
| All conditions | One example of each of the 47 V8 training conditions | [SVG](all_conditions.svg), [PNG](all_conditions.png) |
| Foundation efficiency suite | Square, rectangle, and L examples from the separate easy-foundation bank | [SVG](foundation_suite.svg), [PNG](foundation_suite.png) |

The maps use a common 64 by 64 cell extent, approximately 36.57 by 36.57 m.
Columns increase right and rows increase up (`origin="lower"`). The 13 geometry
panels show the V7 additions, rather than every geometry in the full V8 bank.
Brown means required excavation, green accepted final dumping, beige neutral
ground that permits temporary staging, purple unoccupied ground where dumping
is forbidden, and dark grey static occupancy. These are task-layer categories;
the runtime also checks footprint, reach, and dynamic terrain feasibility.
Courtyard interiors are neutral staging areas and must be cleared for exact
completion. A purple road is not an occupancy obstacle. The foundation suite
has no neutral exterior: every exterior cell is an accepted dump target.

Every pictured example comes from a training split. Geometry and atlas panels
select the upper median required-dig area within each condition, with the slot
index breaking ties. Foundation-suite examples use the same rule within each
shape. Constraint panels match the source of the median `fnd-slab-ring3x`
training example; the builder checks that their required-excavation masks are
identical. Dump supports may be resampled within those conditions, so the
figure is a taxonomy illustration rather than a one-variable policy ablation.
No policy score is used for selection.

The SVG exports retain editable text and embed the exact raster grids; they
are not vector reconstructions of the original continuous geometries. PNGs are
180 dpi previews. All four previews were visually inspected after generation.

- [Condition counts](condition_counts.csv): 47 rows derived from the full
  finite-enriched bank's 4,512 stored training slots. Both slot counts and
  unique scenario IDs are reported. These are support counts, not measured
  training exposure. Source IDs recur across constraint siblings and therefore
  must not be summed across rows to infer independent geometries.
- [Foundation counts](foundation_counts.csv): nine rows, three shape counts
  for each of train/validation/test, totaling 384 maps.
- [Selected examples](figure_samples.csv): bank-relative source directories,
  condition IDs, slots, map/source IDs, and the selection rule for every panel.
  Paths are relative to the Terra repository working directory in the command
  below.

Rebuild from the Terra repository root with the local project environment:

```bash
/home/lorenzo/moleworks/.venv-terra-uv/bin/python tools/build_dataset_documentation.py \
  --bank ../.artifacts/terra_v8_trench_finite_enriched_20260819 \
  --foundation-bank ../.artifacts/terra_foundation_sweep_20260907/bank \
  --output docs/assets/dataset
```

The [builder](../../../tools/build_dataset_documentation.py) requires NumPy and
Matplotlib, reads the inputs without changing them, and writes only this output
directory. It checks map shapes, manifest-to-array slot agreement, total
training counts, and the matched excavation footprint. It is a documentation
renderer, not a runtime feasibility or policy-evaluation test. The input bank
locations are local research artifacts; these documentation exports do not
constitute a public dataset release.
