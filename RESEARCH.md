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

## Partial-completion reset distribution

Training should not use only untouched maps with an all-zero action state.
Every production curriculum should include a controlled minority of
mass-conserving partial-completion resets so the policy repeatedly encounters
late excavation, cleanup, and dumping decisions without first solving the full
exploration problem.

Initial research distribution:

- keep full-task resets as the majority and begin with roughly 20-30% partial
  resets;
- emphasize 50% and 75% completed `in_zone` states in early training;
- use 25% completed states as a bridge back toward full tasks;
- introduce `mixed` states after in-zone endgames are reliable, and introduce
  `near_zone` states later because they retain the largest soil-relocation
  burden; and
- downweight 90% completed states until generation rejects tiny disconnected
  excavation remnants.

This is a starting hypothesis, not a fixed final weighting. Record the exact
partial-reset probability, completion-fraction distribution, pile mode, and
generator revision for every experiment. Primary evaluation must remain on
untouched full-task resets, with partial-reset results reported as a separate
stratified diagnostic rather than pooled into the main completion metric.

## Experiment identity

For every reported result record the Terra revision, terra-baselines revision,
dataset/map family and identity, curriculum, agent/action type, seed, policy
architecture, checkpoint hash, reset distribution, and evaluation protocol.
Live run state belongs in the training repository, not here.
