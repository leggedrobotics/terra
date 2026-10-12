# Environment correctness and finishability checklist

Owner: Terra environment, with the sibling terra-baselines structured trainer.
Updated 2026-10-10. Work through one item at a time; a passing test closes only
the behavior it exercises, not full-bank finishability or physical transfer.

Baseline: Terra `fddedf3e`, baselines `4e6f8ea` (training snapshot `90caf23`).
Evidence: `/home/lorenzo/moleworks/.artifacts/terra_environment_audit_20261010/REPORT.md`.
Keep current training snapshots unchanged while qualifying local fixes.

| ID | Status | Item | Evidence required to close |
| --- | --- | --- | --- |
| 1a | Complete locally | Detect a loaded excavator with no executable full-load unload; end the irrecoverable episode instead of waiting to the cap. | Native trapped-load and legal-staging regressions pass; terminal reward/reset checks pass; explicit failure reason in viewer and training. Production adoption remains item 12. |
| 1b | Bounded probe passed; broader coverage open | Prevent digging into a disposal dead end, if it occurs on real bank states. | 29 saved poses, 144 legal digs/relifts, no trapped loads. Includes nine partially excavated states at 24.7–98.9% completion. Exact post-dig checks include staging; this does not rule out traps elsewhere. |
| 2 | Complete locally | Qualify exact success, decision cap, disabled time deadline, once-only bonuses and reset/GAE boundaries. | Seven environment/helper tests and two new trainer integration tests pass; shared episode helper keeps play and training math identical. |
| 3 | Implemented; manual testing open | Match the manual game to the training protocol and make manual failures reproducible. | Candidate game is live on 8768 with shared timing/rewards/map-dependent caps, no time deadline and the new dead-end handling. 20 CPU viewer/protocol tests and 75 JS tests pass; exports preserve protocol and undo. |
| 4 | Two witnesses; planner update needed | Establish constructive finishability beyond the selected oracle panel. | All nine old successful tapes were replayed with bounded legal reroutes: two complete trench witnesses, seven unresolved route searches. No cap or disposal-dead-end failures. The old nine native successes are not nine structured successes. |
| 5 | Formula corrected; two trench witnesses pass | Check macro-cap adequacy against executable successful tapes and correct the formula documentation. | Code is `max(300, ceil(1.2 * dig_units))`; campaign/launcher/ledger prose corrected. Actual successful structured replays use 86/300 and 102/300 decisions. Foundation/precision margins remain unqualified. |
| 6 | Queued | Diagnose remaining edge/one-cell cleanup rejections. | Capture a failed manual pose; inspect pull room, dug clearance, reach and angle independently; add its regression before relaxing geometry. Keep the agreed 1 m pull setting. |
| 7 | Queued | Qualify productive inner reach. | Compare native assumptions with field evidence and a bounded sensitivity panel. Poor payload below 4.8 m is not a hard kinematic impossibility. |
| 8 | Queued | Calibrate travel, setup and per-bucket swing timing. | Fit consistent physical events without counting dumping twice; test effects on successful-plan ranking. |
| 9 | Queued | Resolve native versus converted material volume. | Choose/document depth and units consistently, check conservation and recompute time references. Current native volume is 14.29% higher than area times 0.5 m. |
| 10 | Queued | Measure precision reward/advantage balance before tuning. | Per-mode removed volume, raw reward, advantage and completion statistics; equal removed volume currently has about 4–5 times smaller progress reward on the larger precision maps. |
| 11 | Runtime baseline verified; mask profile pending | Profile exact DO masks on the production configuration. | Original GH200 probe reached update 3 at 4,057 decisions/s; production update 70 was 3,743/s. No kernel breakdown for GH200 yet. RTX4090 separate-kernel results are not production throughput. |
| 12 | Pending GPU availability and finishability checks | Validate the corrected environment in a bounded training smoke before any production adoption. | Local 4090 was busy with other work; do not interrupt it. Need finite update, terminal/reset exercise and checkpoint/resume on candidate code. Existing CSCS snapshot remains separate. |
| 13 | Fixed baseline; retain regression | No bucket-width constraint in native digging. | Strip gate removed at `8b2037eb`; postprocessing owns trench widening. Do not reintroduce it while fixing finishability. |
| 14 | Fixed baseline; retain regression | Oblique one-cell translation must not silently become straight motion. | Cardinal-only one-cell requests; native rounding from original pose for longer moves; blocked clipping regression. |

## Item 1a work log

The audit reproduced a 45-unit dig followed by twelve failed native unloads
on a synthetic state with no dump ground. It did not establish the frequency
of this failure in the training bank. The three oracle disposal failures were
empty states and used stricter disposal rules than native Terra.

Use the already computed full-load `do_mask`, not the approximate pre-dig
dump observation. A legal off-zone unload must keep an episode alive. This
first fix handles a proven dead end; prevention remains item 1b.

Implemented and checked:

- Environment helper consumes the exact DO mask without running more unload
  simulations. Empty machines, unsupported agent types and unsupported heading
  counts are excluded. Dead-end failure is -1 and distinct from budget timeout.
- Viewer reports `loaded_no_unload`, freezes the episode and retains undo,
  exploration and export. Its post-action mask detects the trap immediately.
- Trainer/evaluator reuse the next observation's mask, take one zero-duration
  fallback WAIT, terminate and reset. Bulk/precision dead-end counts and rates
  are logged. This adds one recorded decision relative to the manual viewer.
  It preserves undiscounted return, but is not GAE-equivalent to immediate
  termination when lambda is below one.
- New training runs default to `--terminate-loaded-dead-ends`. Old checkpoints
  evaluate with their original behavior; native resume requires explicitly
  retaining it with `--no-terminate-loaded-dead-ends`. The ongoing campaign's
  pinned snapshot was not edited, and the game was not restarted.

Validation on 2026-10-10: 26 native structured-action/viewer lifecycle CPU
tests passed (435.998 s); four new trainer dead-end/reset/GAE/protocol tests
passed (2.687 s), plus five existing distribution/GAE/checkpoint tests (4.704 s).
All 74 viewer JavaScript tests and the viewer build passed. Independent source
review found no blocking issues. Repository diff checks passed. No GPU or new
training job was used. These changes are local and not committed or deployed.

## Items 1b–3 follow-up

The first bounded disposal probe found a valid full-load native unload for all
81 admitted digs across 20 saved poses: 10 bank starts and 10 manual poses on
untouched terrain. A second probe added nine partially excavated replay states
at 24.7–98.9% completion: all 63 legal digs/relifts admitted a full unload.
Combined evidence is 29 states and 144 legal candidates, with no native trap
found. This covers some late excavation and spoil states, not the whole bank.
CPU medians
on that geometry were 65.52 ms for the existing DO row, 6.59 ms for a selected
dig plus traversability refresh, and 66.41 ms for its twelve unload checks.
Do not treat those numbers as GH200 throughput or add a 144-check all-heading
guard based on this sample alone. Evidence and exact input states:
`/home/lorenzo/moleworks/.artifacts/terra_environment_audit_20261010/unsafe_dig/REPORT.md`.

`terra.structured_episode` now owns the shared limits and terminal bonus math.
The trainer and viewer call it. Tests cover exact and over-limit completion,
disabled time deadlines, failure with no success bonus, loaded-not-complete,
depth-aware max caps, once-only rewards, reset state/history and GAE isolation.
Seven helper/native tests passed in 10.440 s; two new trainer integration tests
plus four existing dead-end/protocol tests passed in 5.745 s.

The viewer accepts `--structured-protocol`, saves all timing/episode values
with recordings, and rejects mismatched restores. Historical exports preserve
their original dead-end semantics. Independent review caught and resolved a
historical restore mismatch; final 20 CPU lifecycle/protocol tests passed in
4.332 s, and all 75 JS tests/build passed.

## Manual test handoff (item 3)

The candidate game is live at http://127.0.0.1:8768; 8767 is the older game.
Startup selects 17411 / precision / start 0, cap 576, with no time deadline.
It uses campaign timing and rewards plus the new loaded-dead-end handling;
the running CSCS snapshot has not adopted that fix. The complete protocol is
`/home/lorenzo/moleworks/.artifacts/terra_environment_audit_20261010/manual/protocol.json`.
Launcher and recordings are beside it; server tmux session is
`terra-review-game-current`. HTTP session state and protocol were verified.

Test one case at a time. For an unexpected
blocked dig/move, missing unload, or incorrect completion: export before
undo/reset, record what should have happened, and keep the case and step.
Turn each confirmed failure into a focused regression and link it here.

1. Case 17411 precision/start 0: side/bottom edges and final-cell cleanup with
   already-dug pull room. Keep the agreed 1 m stroke rule, with no width gate.
2. At 30/60-degree headings, forward/backward requests of two and five cells;
   one-cell requests intentionally require a cardinal heading.
3. After each dig, check for at least one enabled full-load unload heading.
4. A small trench (15360 bulk/start 0 or 13824 bulk/start 1): verify final
   disposal, empty bucket, and exact completion. These starts now have complete
   structured replay witnesses. Reaching the time reference must not end it.

## Items 4–5: executable oracle replay

All nine previously successful native tapes were checked against actual
structured transitions. Each direct conversion relies on a clipped oblique
one-cell move, now deliberately rejected to preserve movement direction.
Independent native prefixes matched the converted state before that rejection.
Bounded searches for legal routes recovered complete 13824 bulk/start 1 and
15360 bulk/start 0 tapes at 86/300 and 102/300 decisions, respectively.

A second pass also tried routing directly to the next work position. All nine
cases completed the audit: the same two successes and seven bounded route
misses, with no cap exhaustion, loaded-no-unload stop or integrity failure.
17413 bulk reached 87.7% after one such reroute, then encountered another miss.
These are not proofs of unreachability. Navigation and work-position selection
in the oracle need to use the structured transition from the outset; the old
9/20 native figure is not a structured finishability estimate. Keep bank-wide
finishability and foundation/precision cap adequacy open.

Artifacts and complete action tapes:
`/home/lorenzo/moleworks/.artifacts/terra_environment_audit_20261010/structured_replay/REPORT.md`.
The replay was independently reviewed. It checks exact completion and episode
limits after every committed action, with float32 time accumulation and native
material/terrain integrity checks. Planning probes do not spend decisions.
