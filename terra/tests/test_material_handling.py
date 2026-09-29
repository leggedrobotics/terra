"""Native relift capacity and action-level material handling contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import RewardStage
from terra.env import TerraEnv
from terra.tests.test_relocation_reward_contract import (
    SHAPE, _cone, _mass, _set_current_agent, _state, _truck_geometry,
)


dig = jax.jit(lambda state: state._handle_dig())
transfer = jax.jit(lambda state: state._try_truck_transfer_on_excavator_dump())
dump = jax.jit(lambda state: state._handle_dump())
behavior = jax.jit(lambda old, new: old._reward_v2_behavior_costs(new))
terms = jax.jit(lambda old, new: old._agent_reward_terms(new, TrackedAction.do().action))


def test_relift_limit_changes_only_loose_material_and_charges_actual_volume():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[:12, :12] = 1
    probe = _state(target)
    cone = _cone(probe, 0).astype(bool)
    footprint = np.asarray(probe._active_base_footprint_mask()).reshape(SHAPE)
    cells = np.argwhere(cone & ~footprint & (target == 0))
    assert len(cells) >= 40
    soil = np.zeros(SHAPE, dtype=np.int8)
    soil[tuple(cells[:40].T)] = 4
    state = _state(target, action=soil)
    for capacity, expected in ((0, 127), (52, 52)):
        cfg = state.env_cfg._replace(
            reward_stage=RewardStage.REWARD_V2,
            excavator_relift_capacity=capacity,
            material_handling_cost=0.25,
        )
        before = state._replace(env_cfg=cfg)
        after = dig(before)
        assert int(after.agent.agent_states[0].loaded[0]) == expected
        assert _mass(after) == _mass(before) == 160
        assert int(np.asarray(after.world.action_map.map).sum()) == 160 - expected
        component = behavior(before, after)
        assert float(component["reward_v2_handled_volume"]) == expected
        np.testing.assert_allclose(component["reward_v2_material_handling"], -0.25 * expected / 160)

    # The matching cap restricts relifts only; normal fresh dig DO is intact.
    target[tuple(cells[:40].T)] = -1
    fresh = _state(target)
    fresh = fresh._replace(env_cfg=fresh.env_cfg._replace(
        excavator_relift_capacity=8, material_handling_cost=0.25))
    dug = dig(fresh)
    assert int(dug.agent.agent_states[0].loaded[0]) == 40
    assert _mass(dug) == _mass(fresh) == 0
    np.testing.assert_allclose(behavior(fresh, dug)["reward_v2_material_handling"], -0.25)


def test_native_transfer_then_dump_keeps_intra_round_pickup_and_terminal_release():
    state, _, _ = _truck_geometry()
    state = state._replace(env_cfg=state.env_cfg._replace(material_handling_cost=0.25))
    loaded = dig(state)
    volume = int(loaded.agent.agent_states[0].loaded[0])
    assert volume > 0
    passed = transfer(loaded)
    truck_ready = _set_current_agent(passed, 1)
    delivered = dump(truck_ready)
    per_action = jax.tree_util.tree_map(
        lambda a, b: jnp.stack([a, b]),
        terms(loaded, passed), terms(truck_ready, delivered),
    )
    diagnostics = TerraEnv._transition_diagnostics(loaded, delivered, per_action)
    # Truck starts and finishes empty in this physical two-action round.
    assert int(loaded.agent.agent_states[1].loaded[0]) == 0
    assert int(delivered.agent.agent_states[1].loaded[0]) == 0
    np.testing.assert_array_equal(diagnostics["transition_pickup_units"], [0, volume, 0, 0])
    np.testing.assert_array_equal(diagnostics["transition_unload_units"], [volume, volume, 0, 0])
    assert int(diagnostics["transition_mass_residual"]) == 0
    assert float(per_action["reward_v2_handled_volume"].sum()) == volume
    np.testing.assert_allclose(per_action["reward_v2_material_handling"].sum(), -0.25)
