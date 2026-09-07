# Optional foundation behavior costs and observations

The three behavior costs apply only to `RewardStage.REWARD_V2`. Their
`EnvConfig` defaults are zero, preserving the existing reward. They do not
change action geometry, movement increments, task completion, or trench
section admission.

| Setting | Cost |
|---|---|
| `lateral_dig_cost` | Coefficient × newly excavated required volume / total required target volume × sin²(chassis-relative cabin angle before digging) |
| `base_travel_cost` | Coefficient per metre of executed base displacement |
| `base_turn_cost` | Coefficient per radian of executed base rotation, using the shorter wrapped angle difference |

All costs are subtracted from R2. The lateral term requires actual fresh target
excavation by the acting excavator on an empty-to-loaded transition. Dumping,
loose-soil pickup, cabin swing, and failed digging incur no lateral cost.
Travel and rotation measure the same acting agent slot before and after a
transition, including when the next actor changes. Rejected base actions incur
no additional movement cost; the existing R2 step cost still applies.

Chassis-relative digging is a geometric preference. Terra does not model
physical tipping moments, contact support, or soil bearing capacity. This term
does not establish physical stability.

The signed contributions are exposed as `reward_v2_lateral_dig`,
`reward_v2_base_travel`, and `reward_v2_base_turn`. The corresponding measurements
are `reward_v2_fresh_dig_volume`, `reward_v2_base_travel_m`, and
`reward_v2_base_turn_rad`. Reset and other reward stages provide the same keys
with zeros. The signed costs are also included in `agent_rewards`, so the usual
agent + terminal + existence reconstruction remains valid.

## Executable fresh-volume observation

`executable_dig_observation=False` preserves the existing
`local_map_admissible_dig` semantics. Setting it to true changes this same
12-value vector to executable fresh required volume at each cabin heading,
starting at the current arm direction. It does not add input dimensions.
Checkpoint/config consumers must distinguish these two meanings.

The executable vector shares eligibility with DO: the actual cleaned cone,
whole-cone obstacle rejection, positive-pile priority, last-dig exclusion,
base-footprint exclusion, optional foundation-border alignment, per-cell trench
admission, and carrier capacity check. Loaded excavators and loose-soil pickup
report zero fresh volume. A junction cell remains admitted through any aligned
owning section; unrelated branch cells remain excluded. The affordance does not
execute DO or soil relaxation for its 12 headings.

Set the same `executable_dig_observation` value in `EnvConfig` and in the
`TerraEnvBatch` constructor, or `TerraEnv.new` for direct environment users.
The constructor option is static during JIT compilation, preventing legacy
vmapped training from evaluating the new branch. Standalone
`TerraEnv.wrap_state(state)` reads the saved `EnvConfig` flag; batched standalone
callers can pass the static keyword explicitly.

The four settings are appended to the end of `EnvConfig`, so checkpoints with
older positional constructor arguments receive the zero/false defaults.
