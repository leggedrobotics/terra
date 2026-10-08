# LLM-assisted Terra plan refinement

The intended workflow is:

```text
Saved Terra plan and native evidence
  → LLM diagnoses a failure or inefficiency and proposes a supported edit
  → deterministic postprocessor rebuilds and checks the candidate
  → continuous-space plan artifact plus report
  → metric timeline adapter → Terra 3D HTML / MP4 / GIF
```

The LLM operates the existing tools and explains candidate changes. It does not
replace the geometry, material, schedule or route checks. Postprocessing is
optional when the task only needs a native rollout visualization. Command and
installation details are in the [media workflow](README.md).

## Inputs and ownership

| Input or stage | Maintained owner and current contract |
| --- | --- |
| Policy plan/capture | `terra-baselines` and the checkpoint's matching Terra runtime; model, reset and RNG behavior stay with that adapter |
| Native visualization evidence | `terra.viewer3d.ReplayRecorder`; joint endpoints keep requested/effective commands and native reservations |
| Exact fleet input | `terra_fleet_source` v1, from ordered native substeps; `NativeFleetRecorder` checks parity with the same runtime |
| Solo robot conversion | The explicitly selected ROS checkout, machine profile and converter through `terra-postprocess evaluate` |
| Deterministic refinement | `terra.postprocess.fleet` for fleets; ROS conversion plus portable `replay` for solo plans |
| Continuous-space review | `terra.postprocess.timeline` and the shared 3D renderer |

A native replay is not interchangeable with an authoring plan. Joint-round
endpoints cannot reconstruct within-round material ownership and order. Render
them directly; acquire exact substeps before fleet postprocessing.

## The improvement loop

1. Keep the original inputs and run the selected processor to establish a
   baseline. Record the effective geometry, checks and options.
2. Read the report and view the relevant work. Identify the first concrete
   failure or a measurable improvement, rather than optimizing appearance.
3. Make one supported edit or processor change with an expected effect. The
   fleet processor currently removes closed motion loops, refines workspace
   support and retimes retained paths. New stances, holding poses, routes and
   material strategies require additional planning and fresh native evidence.
   Solo source edits use the maintained ROS authoring commands.
4. Rebuild dependent geometry, chronological soil/load state, reservations and
   requested routes. Do not edit output masks, reported success or recorded
   after-states while retaining stale checks.
5. Compare validity and efficiency separately. Stop after meeting the objective
   or exhausting edits supported by the evidence. Report no improvement without
   claiming optimality; retain failures and name any missing input or capability.

Keep the excavation goal, coordinate alignment, machine limits, identities and
material balance fixed in an unchanged-input comparison. Explicitly report site
or disposal-layout changes. Measure coverage and continuous residual, workspaces,
conflicts, rehandling and travel under the scope actually checked. A shorter
route prefix is not evidence that the complete plan is shorter.

## The continuous-space artifact

For fleets, `terra-postprocess fleet` already exports:

- `fleet_source.json`: original ordered native evidence and declared geometry.
- `fleet_plan.json`: source, retained actions, metric poses, workspace/refined
  polygons, material events, proposed schedule and validation report.
- `fleet_report.json`: geometry/material/schedule outcomes and limitations.
- `original.json.gz` and `postprocessed.json.gz`: `terra.postprocessed.v1`
  timelines with metric positions, workspace geometry, the original terrain
  grid and native/loose-soil heights.
- `original.html` and `postprocessed.html`: interactive before/after review.

This is a metric geometric proposal with raster soil support. It is not a
continuous-time controller trajectory or a calibrated physical soil model.
For solo plans, retain the ROS conversion, effective profile, coverage and
requested route evidence alongside the `solo` timeline/report/HTML outputs.
The timeline is a viewing adapter; keep the full source and plan as well.

## What “back to Terra” means

**Back to the Terra viewer:** implemented. The shared renderer accepts
`terra.postprocessed.v1` directly. It preserves metric positions, refined
workspaces, terrain resolution and saved routes without snapping them onto the
native discrete action grid. Missing routes remain jumps and failed verdicts
remain visible.

```bash
terra-postprocess fleet fleet_source.json --out candidate
terra-postprocess render candidate/postprocessed.json.gz --out candidate/refined.mp4
```

**Back to native Terra execution:** a separate validation step. Changed commands
must run in a compatible native adapter from the correct initial state, with
actual order, rejection outcomes and material effects recorded again. The
current postprocessor reports `native_replay_performed: false`. There is no
general compiler from an arbitrary metric plan to valid native Terra actions.
If that adapter is unavailable, the candidate stays native-unverified. Rendering
or rasterizing the proposal cannot substitute for execution.

## Deliverable

Keep the original/edited inputs, change notes, continuous plan, checks, timelines
and before/after 3D pages together; add requested video and gallery outputs.
Include a short numeric comparison and the first unresolved failure. Report
native task success, processed geometry, requested navigation and physical
execution as separate evidence. CPU recaptures retain their own outcomes.

Current video export holds exact recorded endpoints with illustrative timing.
Interactive solo/metric playback can show illustrative motion; joint native
endpoints do not establish intermediate motion or robot readiness.
