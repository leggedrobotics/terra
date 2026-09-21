# Terra Research Context

## Objective

Train a global excavation-planning agent that can sequence base motion, cabin
orientation, digging, and dumping to complete trench and foundation earthworks.
The policy should generalize across map geometry, obstacles, dumping
constraints, and relevant excavator embodiments rather than memorize a single
layout.

Terra is the abstract JAX environment. The sibling `terra-baselines` repository
owns PPO training, evaluation, checkpoints, inference, and experiment
operations. `moleworks_ros` owns plan execution on the simulated or real robot.

## Canonical environment sources

- [Audited dataset reference and terrain figures](docs/DATASET.md)
- [Audited environment, rewards, and termination](docs/ENVIRONMENT.md)
- [Paired PPO and evaluation protocol](https://github.com/leggedrobotics/terra-baselines/blob/main/docs/TRAINING_PROTOCOL.md)
- [Environment overview and semantics](README.md)
- [Map-generation workflow](terra/env_generation/README.md)
- [Partial-completion reset contract](terra/env_generation/PARTIAL_COMPLETION_RESETS.md)
- `terra/state.py`: action feasibility and state transitions
- `terra/env.py`: environment and observation assembly
- `terra/config.py`: curriculum and reward/environment configuration
- `terra/maps_buffer.py`: dataset and map metadata loading

## September 21 metadata precision and endpoint clearance

Straight-trench replay exposed two geometry defects. Float16 storage moved a
nominal centerline by 5.23 mm, admitting one pose at a 2 m offset while rejecting
its mirror. The movement check also rejected a chassis corner equal to the map
dimension, although corners are continuous cell boundaries and the footprint
occupancy test uses cell centers.

Trench and foundation-border metadata now remain float32 in MapsBuffer. Closed
trench distance limits use the same 1e-5 m numerical tolerance as the dig cone.
Endpoint and swept-translation bounds accept corners in [0, map dimension];
overhang, soil, holes, obstacles and other chassis remain forbidden.

On the inspected straight geometry, replaying the original greedy actions from
the same initial Agent now completes 80/80 excavation and accepted disposal at
action 46. A manual plan completes at action 62, then exits in seven native
actions to open ground with all four base movements available. This is a
fixed-action simulator diagnosis, not a newly evaluated policy success rate.
Task termination still ends on excavation/disposal completion; post-completion
egress is recorded separately and does not count as additional episode success.

The correction was validated in `terra-geometry-clearance-20260921`, based on
the current experiment source plus its earlier geometry correction, and applied
to the active local experiment checkout. Existing frozen cluster snapshots
retain their original runtime. Evidence and regression/review reports are in
`.artifacts/terra_geometry_clearance_20260921/` in the Moleworks workspace.
Serialized old float16 states must be reconstructed from original metadata;
widening an already quantized table cannot restore geometry. Demonstration
observations must be regenerated for this runtime. The inspected evaluation
map remains outside the generalist training bank.

Euler job 14791590 subsequently evaluated the same u109250 policy before and
after this correction. Straight-trench completion rises from 2/32 to 26/32
(26 gains, two losses). The full panel changes from 383/384 to 382/384
foundations and 208/224 to 211/224 trenches; roads remain 29/32. Native finite
PPO and transition-integrity checks pass. Six straight starts and two newly
lost full-panel cases remain for diagnosis; these results do not qualify exit
from every successful plan. Reports and independent review are under
`.artifacts/terra_latest_geometry_20260921/`.

The observation dictionary also exposes the acting agent's previous effective
work pose as normalized x/y, heading sine/cosine and a validity flag. Reset
context is zero. Legacy policies ignore it; the paired baselines support
optional actor/critic context and native optimizer-preserving migration for
the delayed retained-work cost experiment.

## September 18 workspace-boundary correction

A saved trench failure exposed an observation/transition disagreement: the
executable observation reported eight fresh cells, while DO relifted one loose
soil cell on the sector edge. Batched GPU matrix arithmetic placed that cell
outside the observed sector although scalar geometry included it.

The correction rotates relative coordinates elementwise, gathers the base
position directly, and gives angular/radial sector boundaries a closed
convention with a numerical tolerance of `1e-5` rad/metres. It preserves the
chassis, trench-alignment, soil-priority, disposal and completion rules. Five
GPU geometry tests pass, including scalar/batched/nested agreement over 432
translated base/cabin poses. Replaying the saved failure now reports zero
executable fresh cells and consistently relifts the one soil unit.

Ten existing training plans also complete under the fixed geometry, with
449 actions and no ineffective actions or integrity failures. Native MapsBuffer
replay now also passes with float16 trench/foundation metadata and the original
actions/reset seeds. It supersedes the earlier helper's float32 metadata:
some saved observation values change, so the native export is not byte-identical
to the earlier bank. These four-source, trench-only examples are implementation
smoke data, not a qualified broad imitation curriculum. The paired baselines
implementation has completed two
finite native diagnostic PPO updates with actor imitation; no new production
training or learned-policy improvement is established.

Evidence is under `.artifacts/terra_trench_failures_20260918/boundary_fix/` in
the Moleworks workspace. The paired baselines
`docs/research/TRENCH_DEMONSTRATIONS_20260918.md` records replay limits, the
completed 1,280-plan rehearsal archive, the independent Oracle review, and the
bounded PPO-plus-imitation experiment being prepared. Its soft foundation
targets and explicit source-balanced expert sampling require their own
production-layout qualification; the earlier hard-label smoke does not
establish those checks.

## September 16 Oracle implementation

The observation dictionary now includes `remaining_time`, the fraction of the
finite episode budget left. Reset emits 1; terminal step observations emit 0.
Legacy policies ignore this field. The paired baselines change adds explicit
zero-initialized actor/critic embeddings and native checkpoint migration.

Reward-v2 also offers optional retained-work setup, transfer-distance and
heading costs. They count effective digging, relifting and dumping at exact
base poses; navigation and cabin motion between those events are discarded.
Initial approach and final egress are excluded. Straight-line distance is a
lower bound on a navigable transfer. All new coefficients default to zero and
remain zero for broad training. Before activating them, expose previous
retained pose/validity to the policy and qualify completion. The optional
`State._executable_fresh_dig_union()` diagnostic unions actual eligible cells
across cabin headings; it adds no geometry work to normal training.

Corrected chassis/soil/trench rules and terminal reward-v2 potential are
unchanged. Focused tests cover event accounting, reset/handoff, fresh-cell
union and telescoping returns including failed-terminal potential. On a real
fixture, old state leaves and zero-cost rewards match the previous runtime
exactly for dig, cabin, motion, base turn and WAIT.

Experiment decisions and remaining diagnostics are recorded in the paired
baselines `docs/research/ORACLE_FOLLOWUP_20260916.md`.

## Active research themes

- Generalist global planning across trenches and foundations.
- Reliable completion of both bulk/core excavation and precise edge-finishing
  phases.
- Explicit state-dependent action feasibility and useful affordance features.
- Curricula and reset distributions that deliberately expose partially
  completed and rare endgame states.
- Stable value learning across core excavation, alignment, edge finishing, and
  terminal completion.
- Export of plans with enough geometry and frame metadata for ROS execution.

The detailed edge-finishing diagnosis and literature notes live in
`terra-baselines/docs/edges_trainings/`; treat those as research hypotheses and
verify them against the current Terra code before implementation.

## Movement-feedback pilot runtime

The 2026-08-21 paired pilot uses one common repaired runtime for both arms:
dig/relift excludes the exact current base footprint, the visible
traversability layer preserves blockers beneath the agent overlay, and a
successful dig/relift applies local soil relaxation exactly once in
`_handle_dig`. These are shared runtime invariants, not the compared treatment.

The optional treatment observations are deliberately narrow and unmasked:
four exact tracked-base movement-effect bits and two previous-transition bits
for any physical effect and material-or-load change. Reset observations encode
the previous outcome as `00`. The sibling baselines repository owns the
fresh-scratch control-versus-feedback training contract and evaluation gates.

Both paired 4-GPU arms completed 50,000 updates on 2026-08-23 (Slurm
`11364188` control and `11364189` feedback). Their final 1,000-update online
success is tied at 0.99019 and 0.99037, while feedback reduces the no-effect
rate from 0.03152 to 0.01450. This supports promoting the repaired runtime and
the optional observation path, with feedback disabled by default. It does not
select the feedback policy: the preregistered fixed development-720 and
recurrence panels remain pending in terra-baselines.

## Partial-completion reset distribution

The supported training treatment is one sparse, source-bound
`relay_corridor` sidecar bank over the accepted full-start bank. Canonical
target, obstacle, dumpability, and distance layers remain unchanged; the
sidecar supplies only a mass-conserving nonzero action map. A source is admitted
only as a complete, strictly nested 50/75/90% triplet generated with one shared
source seed. A condition is supported only when at least one complete triplet
exists, and all three tiers sample the same canonical source pool. Unsupported
conditions fail instead of falling back to `in_zone`, `mixed`, or `near_zone`.

The baselines-owned lane schedule is fixed for this treatment. Full starts
remain the majority throughout:

1. updates 0--2,499 hold the partial-reset lane share at 25% and use only
   90%-complete resets;
2. updates 2,500--4,999 keep the 25% share and distribute partial lanes across
   the cumulative 75/90% window;
3. updates 5,000--7,499 keep the 25% share and distribute partial lanes across
   the cumulative 50/75/90% window; and
4. updates 7,500--9,999 retain that cumulative window while fading the total
   partial-reset share linearly from 25% to zero. Updates 10,000 onward use
   ordinary full starts only.

These are mechanically generated synthetic reset states. The treatment is
Backplay-inspired start-distribution shaping, not exact trajectory Backplay and
not a bank of demonstrated successful suffixes.

Full-start evaluation and curriculum mastery remain separate from this
treatment. Partial episodes must not update the full-start mastery EMA or be
pooled into the primary completion metric; report them as tier-stratified
training diagnostics. Record the sidecar digest, generator revision, active
schedule phase, total partial lane share, and per-tier lane shares for every
experiment.

## Experiment identity

The September 11 movement correction preserves strict soil-free chassis
occupancy while replacing independently rounded intermediate tracked poses
with a straight swept-polygon check. The previous integer-prefix rule could
invent sideways collisions at angled headings. Each shorter candidate is
tested along its own straight path; clear endpoints cannot jump obstacles.
This uses Terra's cell-center polygon convention and does not certify Nav2 or
physical vehicle clearance. The sibling baselines reliability design and
foundation behavior note retain the matched frozen-policy diagnosis. Keep the
original ba9cc214 training cohort separate from this environment correction.

For every reported result record the Terra revision, terra-baselines revision,
dataset/map family and identity, curriculum, agent/action type, seed, policy
architecture, checkpoint hash, reset distribution, and evaluation protocol.
Live run state belongs in the training repository, not here.
