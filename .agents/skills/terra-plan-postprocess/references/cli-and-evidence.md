# ROS authoring commands, inputs and evidence

Read this for robot-specific solo edits, conversion or Nav2 checks. Portable
fleet processing and 3D HTML/video rendering now belong to Terra's
`terra-postprocess`; follow the selected Terra checkout's postprocessing README.
The ROS commands below remain the authoring adapter. They are not a fleet
converter or a general continuous-plan-to-native-action compiler.

Commands below are the
interface verified on 2026-09-28; check the selected checkout's README and `--help`
if it differs. Run at that repository root inside a sourced ROS container using
the repository's container/build workflow, not host ROS.

## Input and command contracts

`SOURCE` is a saved converter input directory with `terra_plan.json`,
`arrays.npz`, `map_geometry.json` and optional `grid_map_geometry.json`.
It is not the raw TerraMapMaker export's `map/` directory. `PROFILE` is the
reviewed base profile. `plan evaluate` records necessary profile adaptations.
Use new output paths so before/after evidence stays intact.

```bash
# Inspect native pairs and map-frame poses.
python3 scripts/TerraMapMaker/tmm.py plan show SOURCE/terra_plan.json

# Baseline: routes are attempted only after complete geometry.
python3 scripts/TerraMapMaker/tmm.py plan evaluate SOURCE \
  --profile PROFILE --repair --compact-dump-regions --routes --out BEFORE

# Move both members of native pair 0; map metres and relative yaw degrees.
python3 scripts/TerraMapMaker/tmm.py plan move-station SOURCE/terra_plan.json \
  --step 0 --dx 0.25 --dy 0 --dyaw-deg 15 --out CANDIDATE.json

# Original source target remains the obligation.
python3 scripts/TerraMapMaker/tmm.py plan evaluate SOURCE --plan CANDIDATE.json \
  --profile PROFILE --repair --compact-dump-regions --routes \
  --compare BEFORE --out AFTER

# Render saved evidence without launching a planner.
python3 scripts/TerraMapMaker/tmm.py plan review AFTER/conversion \
  --compare BEFORE/conversion --routes AFTER/navigation --out REVIEW
```

Omit `--routes AFTER/navigation` from `review` when route evidence is absent.
`--repair` and `--compact-dump-regions` are explicit options; retain the same
settings in a matched comparison. Native edits accept inline evaluation flags
`--source`, `--profile`, `--review-out` together, plus comparison/repair/routes.

Other native edits:

```bash
python3 scripts/TerraMapMaker/tmm.py plan drop-step PLAN.json 3 --out NEW.json
python3 scripts/TerraMapMaker/tmm.py plan swap-steps PLAN.json 3 4 --out NEW.json
python3 scripts/TerraMapMaker/tmm.py plan reassign-dig PLAN.json \
  --steps 2 1 0 --axis-deg 0 --cuts -1.142857 0.571429 --out NEW.json
python3 scripts/TerraMapMaker/tmm.py plan set-dump PLAN.json \
  --step 1 --center -4.4 5.1 --radius 0.45 --out NEW.json
```

- Indices are **zero-based native pairs**, not one-based viewer workspace labels.
  Reordering changes indices; inspect `plan show` again and track source action IDs.
- `reassign-dig` preserves the selected fresh-dig union. Selected steps receive
  strips in listed order; axis and absolute cut projections are in the map frame.
- `set-dump` specifies a source-cell-centre permission disk, not pile radius.
  A tiny disk can select no cells or miss useful fine-grid points after rasterizing.
- Commands edit native schema-v2 plans and reject converted plans. Moving a
  station does not move its dig masks. Dropping a pair does not delete its target
  from the source arrays.

If geometry was evaluated without routes, check them separately:

```bash
python3 high_level_planning/terra_planner/scripts/check_converted_terra_routes.py \
  --source AFTER/source --conversion AFTER/conversion --output-dir ROUTES
python3 scripts/TerraMapMaker/tmm.py plan review AFTER/conversion \
  --compare BEFORE/conversion --routes ROUTES --out REVIEW_WITH_ROUTES
```

Keep the original geometry-only summary. A combined final status should link the
exact conversion, route report and matched review; do not rewrite an earlier
`navigation not requested` result as though it ran routes then.

`plan evaluate` is an authoring loop; it does not install a `mole_maps` package or
establish launch readiness. When packaging is requested, use the current
`high_level_planning/terra_planner/scripts/prepare_converted_terra_plan.py` and its
owning README. Keep the edited native plan, authored map, alignment and effective
profile consistent, and rerun the package's preflight and route checks. Internal
routes exclude the initial approach and final exit; verify those separately when
they are part of the requested execution. Route runtime operation through
`$terra-pipeline`.

## Read the outputs

| Artifact | Meaning |
|---|---|
| `request.json`, `source/`, `profile.yaml` | Saved inputs, edits, options and effective settings. Include explicit source-layout edits and assumption sidecars. |
| `summary.json`, `evaluation.log` | Requested-check outcome; inspect `complete_geometric_plan` and `internal_navigation`. |
| `conversion/report.json` | Coverage, continuous geometry, constraints, per-pair admission, witnesses, closure and soil-sequence errors. |
| `conversion/coverage.npz` | Final retained support/completion/deposit masks and poses; use these to measure actual fresh work. |
| `conversion/terra_plan.json` | Emitted only for complete geometry; existence is not a navigation pass. |
| `navigation/report.json`, `route_NNN_MMM.json` and `.npz` | Required/checked/passing legs, paths, arrivals and saved terrain/costmaps. Later legs can be untested after a failure. |
| `review/index.html`, `overview.png`, `review.json` | Interactive timeline, image and adapter. Route adapter is under `review.json.after.routes`. |

Evaluation exit 0 means the **requested** checks pass, 2 means incomplete geometry
or failed requested navigation, and 1 means input/tool error. Review exit 0 means
rendering succeeded, including when the plan fails.

## Coordinate and indexing traps

- Native arrays are **X,Y**. A cell centre is
  `origin + R(yaw) * (([i,j] + 0.5) * metres_per_tile)`.
  Native `pos_base` transforms without the extra half cell.
- Coverage arrays are **Y,X**. Their origin is the centre of cell `[0,0]`;
  drawing extents begin half a conversion cell earlier. Route occupancy `.npz`
  origins instead describe the lower-left cell edge.
- Use `base_pose`, `cell_centers_world`, `cell_geometry` and recorded alignment;
  do not assume unrotated axes or substitute a fine-grid bounding box for the
  exact known source domain.
- `retained_dump_support` holds candidate release centres;
  `retained_deposit_support` holds possible deposition support. Neither supplies
  measured height or actual per-load dump points.
- Match retained rows to `accepted=true` and non-omitted reports and verify
  counts against masks. Rejected/omitted rows and discarded added stations must
  not shift identities. Before/after source IDs can be unmatched after splits.
- Native `images < 0` means excavation, `images > 0` final disposal;
  `dumpability` is separate permission. Actions encode initial terrain state.
  Authoring validation rejects positive dump labels on non-dumpable cells: when
  excluding final ground from dumping, clear its positive labels too.
