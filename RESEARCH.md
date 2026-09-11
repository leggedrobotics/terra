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

- [Environment overview and semantics](README.md)
- [Map-generation workflow](terra/env_generation/README.md)
- [Partial-completion reset contract](terra/env_generation/PARTIAL_COMPLETION_RESETS.md)
- `terra/state.py`: action feasibility and state transitions
- `terra/env.py`: environment and observation assembly
- `terra/config.py`: curriculum and reward/environment configuration
- `terra/maps_buffer.py`: dataset and map metadata loading

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
