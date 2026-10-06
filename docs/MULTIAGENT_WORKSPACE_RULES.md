# Machine workspace separation

`EnvConfig.workspace_guard_enabled=True` is the default for all fleets,
including two excavators. It is saved with the environment configuration.
The same setting controls random reset placement and native joint transitions.

Each active machine reserves its chassis and its complete native work sector.
WAIT, empty travel and a raised skid bucket do not release that sector.
Translations and rotations reserve their continuous sweep, including the full
arm/cabin turn. A one-tile stand-off is included; at the current 64-cell map
scale this is about 0.57 m. These are conservative 2D macro envelopes, not a
physical arm or vehicle collision model.

The native joint step processes the requested actions in its existing execution
order. Each accepted sweep remains reserved until the round ends. A conflicting
candidate is rolled back atomically, including soil, load and carry credit. Its
effective reward action is WAIT. PPO retains the requested action and its own
log probability; the environment does not alter the sampling distribution.
Diagnostics expose blocked slots, effective actions and accepted conflicts.

Geometry projections use explicit highest-precision matrix products, independent
of policy precision. Every sweep also contains both endpoint reservations. This
prevents reduced-precision GPU projections from admitting a move whose next
stationary envelope overlaps a neighbour.

Random resets reject intersecting full envelopes, up to 2,048 placement attempts
per machine. An impossible placement raises an error instead of hanging or
returning an invalid fleet. The public prepared-reset API also rejects
unauthorized overlaps. Low-level `State.new(initial_agent=...)` preserves
caller-owned states: validate them with `workspace_guard.state_has_conflict`,
as the fixed campaign bank does. A registered loading pose may pass that check because the exception
below permits stationary holding.

## Explicit loading exception

No machine-type pair is automatically exempt. Register a directed pair by slot:

```python
from terra.config import EnvConfig
from terra.workspace_interactions import loading_pair_mask

types = (0, 1, 2)  # excavator, truck, skid
cfg = EnvConfig(
    agent_types=types,
    action_types=(0, 0, 0),
    workspace_loading_pairs=loading_pair_mask(types, [(0, 1)]),
)
```

The training CLI accepts `--workspace_loading_pairs '[[0,1]]'`. Participants
must be active and each may have only one registered partner. Wrong roles,
reversed pairs and unknown permissions are rejected at configuration validation.

For that pair, only the excavator work sector may overlap the receiver's body
and work sector, in these phases:

| Requested action | Partner requirement | Additional requirement |
| --- | --- | --- |
| Truck approach, steer or depart | Excavator requests WAIT | No material change |
| Excavator cabin rotation | Truck requests WAIT | Chassis and material unchanged |
| Excavator DO | Truck requests WAIT | Native transfer of the entire load and carry credit into this registered truck |
| Explicit WAIT | None | No physical or material change |

All chassis-to-chassis separation, truck-workspace-to-excavator-chassis
separation, and checks against every third machine remain enforced. A failed
request does not become permission for its partner to move. Loading that fails
because the truck is full cannot fall through to a ground dump inside the shared
workspace. Loaded trucks can depart using tracked or wheeled motion.

Terra represents transfer reach using the truck base centre in the excavator
cone (with its existing one-cell tolerance). It has no separate truck-bed/cab
geometry. This exception therefore tests abstract loading coordination; it does
not establish physical cab or arm clearance. Truck reward support is a separate
contract and is not established by these transition tests.

## Reproducing historical experiments

Set `workspace_guard_enabled=False` explicitly to reproduce a body-only legacy
environment. Old checkpoint configurations receive the appended default fields
when loaded; continued training therefore uses the new rule unless explicitly
disabled. Compare checkpoints under identical environment settings. Historical
two-excavator speedups did not include this workspace rule.
