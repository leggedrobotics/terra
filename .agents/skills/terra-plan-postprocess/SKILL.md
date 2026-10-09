---
name: terra-plan-postprocess
description: Refine saved Terra excavation plans with LLM-guided edits and the deterministic postprocessor, export continuous-space plan artifacts, and render original/refined 3D views. Use for solo, two-excavator and mixed fleets, including coverage, soil sequence, workspace and route failures. Training belongs to terra-rl; robot execution belongs to terra-pipeline.
---

# Terra Plan Refinement

Follow this loop: **Terra plan → LLM diagnosis and supported edits → deterministic
postprocessing → continuous-space artifact → Terra 3D viewer**. Treat the LLM's
edits as proposals; recomputed checks determine whether they improve the plan.

## Owner and inputs

Start in the user's selected Terra checkout, or locate it from the workspace
instructions. Verify its revision and dirty state. Read `terra/postprocess/README.md` and
`terra/postprocess/REFINEMENT.md`. Use a current isolated checkout when the
canonical checkout contains unrelated work. Terra owns portable processing and
rendering; baselines owns checkpoint capture. No policy rerun is needed to
render an existing recording.

- `terra.viewer3d.v1`: native recorded states for rendering. Joint endpoints
  alone cannot supply exact ordered substeps for fleet processing.
- `terra_fleet_source` v1: exact native substeps, geometry and material effects
  for `terra-postprocess fleet`. Acquire missing substeps with
  `NativeFleetRecorder` in the source's matching runtime; do not invent them.
- Saved solo converter input/conversion: use the explicit ROS adapter through
  `evaluate`/`solo`. Read [ROS commands](references/cli-and-evidence.md) and
  [repair guidance](references/repair-and-review.md) only for this branch.
- `terra.postprocessed.v1`: metric playback of the continuous-space proposal.

## Refinement loop

1. Save the original inputs, effective geometry/profile and baseline report.
   Keep native success, processed validity and requested route checks separate.
2. Inspect the first meaningful failure or measured inefficiency. Use the report
   and 3D scene to distinguish continuous coverage gaps, body/tool conflicts,
   soil/load order, dump placement and failed connections.
3. Propose one supported correction and state the expected effect. The current
   fleet processor cleans closed motion loops, refines support and retimes fixed
   paths; it does not search new stances or routes. Changed native commands or
   poses need fresh execution evidence from the matching runtime. Never rewrite
   recorded after-states or material deltas to pretend an edit was executed.
4. Rebuild affected processing, soil, workspace and route evidence. Keep the
   full excavation goal, stable machine identities, coordinate alignment,
   material/load balance and machine limits. Do not crop gaps or relax limits
   to get a pass. Retain rejected candidates with their actual verdict.
5. Compare validity and efficiency separately: coverage/residual, conflicts,
   workspaces, rehandling, travel and checked/required routes where available.
   Stop when the objective is met or no further edit is supported by the
   evidence. Report no improvement or a concrete missing input/capability
   instead of forcing a gain.

## Export and review

```bash
terra-postprocess fleet fleet_source.json --out candidate
terra-postprocess render candidate/postprocessed.json.gz --out candidate/refined.html
terra-postprocess render candidate/postprocessed.json.gz --out candidate/refined.mp4
```

`fleet` also writes the source, `fleet_plan.json`, `fleet_report.json`, original
and processed metric timelines, and both HTML pages. Exit 2 is a reviewable
rejected proposal; exit 1 is an input/tool error. Video dependencies and optional
browser selection are in the owning README. Current video holds recorded
endpoints; interactive playback retains its documented illustrative motion.

The viewer reads metric artifacts directly. **Rendering does not require
quantizing the continuous plan back onto Terra's native grid.** If native
execution validation is requested, replay the changed action sequence through a
compatible runtime adapter and record the result separately. A general
continuous-plan-to-native-action compiler is not implemented; keep that result
unverified when the needed adapter is absent.

Deliver the original/edited inputs, continuous plan, reports, timeline, before/
after 3D pages and requested videos/gallery, plus a short change log and numeric
comparison. Inspect at least the first failure, edited work and final state.
State remaining blockers and distinguish modeled feasibility, native replay,
ROS route checks and physical execution. ROS-dependent work runs in its normal
container. For execution, use the installed `$terra-pipeline` skill when
available, or the owning ROS execution runbook.
