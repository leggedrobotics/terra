# Diagnose, edit and review

Use this for the ROS solo-conversion branch when it fails or produces an
inefficient sequence. Prefer a small edit supported by the observed failure
over unconstrained pose guessing. Portable fleet refinement is owned by Terra.

## Soil and geometry constraints

- Temporary dumping on future dig targets and repeated dumping at one place
  are allowed. A later cut removes loose soil and excavates required native
  material only within its actual cut support; soil outside it persists.
- Do not dump into existing excavation or its active clearance band. Final
  disposal belongs outside trench bays. Read the current profile's distances
  and source-cell size; authoring and runtime clearance can differ.
- Permission area, selected release point, possible deposit support and actual
  pile are different. Permission overlap alone does not establish a conflict.
- Replay terrain in execution order. Collection can reopen ground; cuts leave
  holes and later dumps can change access. Future stances are movable proposals.
  A checked route may motivate a dump edit, not a permanent ban around all stances.
- Traversability depends on terrain height and slope. A below-0.5 m assumption
  is not height evidence from a binary spoil mask. Do not invent pile heights,
  capacity or clay repose to clear a route.
- Keep bottom completion, entry reach, dump reach, BASE/CABIN/control origins
  and stop tolerance distinct. Use the effective profile and robot geometry;
  do not alter dump reach as a side effect of another repair.
- Preserve the original goal and continuous required geometry. Site extension
  or disposal-layout changes need separate variants, changed area/cell counts
  and explicit new-ground assumptions.

The converter may require collection before cutting prior spoil or block all
retained deposit support during navigation. Check current implementation;
expose such limits instead of inventing operator rules or claiming unsupported
permissive behavior.

## Choose the relevant correction

| Observation | Check and possible correction |
|---|---|
| All samples covered, but residual area remains | Inspect `workspace_lanes.largest_unclosed_residual_pieces`, centroid, area and rejections. Check full-blade containment and completion reach; samples miss thin seams. Use the recorded residual tolerance. |
| Many tiny stations or narrow trench strips | Compare initial dump admission with the **full planned cut window**. Protecting only required-core samples can select a deposit that later blocks the full-width cut. Keep final actual-cut/support audits active. |
| Body crosses known map boundary | Transform the footprint and margins at the actual proposed pose. Distinguish a stance adjustment from extra ground. An authorized extension retains target world geometry, explicitly labels assumed flat/free ground, and requires fresh routes. |
| No legal dump under a complete cut | Calculate contained deposit centres after full-support and hole-gap checks, including required station stops. A wider permission or final strip may help. Do not shrink physical spread by reducing permission radius. |
| Moving a foundation stance loses bulk coverage | Reassign its target union or retain the bulk visit and add/reuse a finishing visit. Moving BASE alone leaves masks fixed. Check later footing and dump effects before reordering. |
| Early corner cut blocks later stations | Inspect chronological `footprint_prior_dig` evidence. Ground may be needed before the cut; change order or stance family rather than permanently banning the target. |
| Native pose/dump edit appears ineffective | Generated trench lanes can replace native stances. Added stations can choose independent fallback dumps. Inspect chosen poses, source IDs, `planned_trench_lane`, centres and masks before repeating an edit. |
| Clear endpoints but no route | Inspect the whole padded swept body, terrain order, heading and turning radius. Screen approach geometry or a saved-terrain route. Endpoint freedom does not establish maneuvering space. |
| Prior pile seems to block a leg | Remove only that support in a labeled diagnostic, retaining cuts and all other terrain. If reachable, relocate the deposit or add real collection and rebuild the sequence. Diagnostic deletion is not a repaired plan. |
| Smac `Start occupied` with an apparently free pose | Preserve start/arrival, padded footprint, occupancy and raw costs. Compare continuous geometry, raster/heading representation and actual planner settings. A free centre is insufficient; planner rejection alone does not prove a physical collision. Keep the candidate failed while unresolved. |

A specific route can guide dump reselection. For an authoring no-dump region,
save a separate source variant, validate labels, record changed cells/area,
verify alternate reachable dump ground, and label the comparison as a disposal
layout change. Do not generalize it into a hard ban around all future digs.

A passing one-leg probe still requires native reconversion, actual deposition at
new locations, all affected geometry, and complete internal navigation. An
exploratory converted-pose edit cannot be delivered with old masks.

## Efficiency is a separate result

Measure stations and source-to-retained changes, fresh completion area per visit,
cut/collection roles, represented rehandling, heading changes, route lengths and
complete internal driving distance. Separate station spacing from path length,
and witness-band length from useful work. Few new samples can still represent
necessary continuous closure; do not delete a visit from a sample-only metric.

Fewer stations can mean more driving. A failed route prefix is not complete-plan
travel. Claim baseline travel reduction only when comparable baseline routes
exist. Selected cases do not establish a new whole-bank score.

## Visual and independent review

For the ROS 2D review, use maintained `tmm_review.py` and `review/index.html`.
For metric 3D, use `terra-postprocess solo` then the shared `render` command.
Coordinates are in
[cli-and-evidence.md](cli-and-evidence.md). Inspect overview images and important
timeline events. These artifacts aggregate workspaces; do not invent individual
scoop execution or timing.

Follow arrival terrain → work → deposition → departure → next navigation. Show:

- Original goal, required completion, physical cut support and remaining gaps.
- Native proposals, retained BASE poses, bodies, margins and saved arrivals.
- Candidate release centres, possible deposit support and actual observations,
  if any; historical deposits, collected ground and persistent holes.
- Original/extended source boundaries, unknown exterior and changed final zones.
- Station-order connectors, returned paths, failed legs and untested later legs.

Show issue location, source/retained identity, value, threshold and report field.
Do not invent exact residual polygons from centroid/area diagnostics. Retain exact
metrics but distinguish roundoff below the recorded tolerance from real gaps.
Make the failed leg selectable with its terrain and departure pose. For videos,
use step indices or explicitly illustrative timing when timestamps are absent;
do not animate invented measured heights.

For a converter fix, add a small reproducer and run repository-required checks.
Re-evaluate a representative affected plan. Reuse route evidence only through
the maintained strict match/replay check when executable geometry and terrain
are unchanged. Artifact-only edits need their invariants and conversion/route
checks, not a new software test suite.
