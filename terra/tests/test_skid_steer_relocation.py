"""Skid-steer relocation: pickup, delivery, and R2 on haul-only maps.

Transitions use Terra's real handlers; poses are direct test setup. At the
test pose (heading 0) the chassis covers columns 11-21 and the bucket sector
columns 22-27.
"""

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import REWARD_V2_DISTANCE_BOUND
from terra.config import REWARD_V2_DISTANCE_REF_M
from terra.env_generation.distance import compute_reward_v2_distance_map
from terra.tests.test_relocation_reward_contract import SHAPE
from terra.tests.test_relocation_reward_contract import _env_config
from terra.tests.test_relocation_reward_contract import _mass
from terra.tests.test_relocation_reward_contract import _state

CFG = _env_config((2,))
POSE = np.array([32, 16], dtype=np.int16)
DUMP_POSE = np.array([32, 38], dtype=np.int16)  # bucket over columns 44-49
ZONE = (slice(26, 39), slice(44, 57))

pickup = jax.jit(lambda s: s._skid_steer_auto_load_dirt(s))
forward = jax.jit(lambda s: s._handle_move_forward())
backward = jax.jit(lambda s: s._handle_move_backward())
do = jax.jit(lambda s: s._handle_do())
r2_values = jax.jit(lambda s: s._reward_v2_state_values())
r2_reward = jax.jit(
    lambda s, n, success: s._get_reward_v2(n, success, success)
)


def _distance(target):
    return compute_reward_v2_distance_map(
        target,
        np.zeros(SHAPE, dtype=bool),
        tile_size_m=CFG.tile_size,
        distance_ref_m=REWARD_V2_DISTANCE_REF_M,
        distance_bound=REWARD_V2_DISTANCE_BOUND,
    )


def _skid(target, action, *, loaded=0, shovel=0, pose=POSE):
    state = _state(target, action=action, distance=_distance(target), env_cfg=CFG)
    agent = state.agent.agent_states[0]
    return state._set_agent_state_at(
        0,
        agent._replace(
            pos_base=jnp.asarray(pose, dtype=jnp.int16),
            loaded=jnp.array([loaded], dtype=jnp.int8),
            shovel_lifted=jnp.array([shovel], dtype=jnp.int8),
        ),
    )


def _bucket(state):
    return np.asarray(state._build_dig_dump_cone()).reshape(SHAPE).astype(bool)


def _loaded(state):
    return int(np.asarray(state.agent.agent_states[0].loaded)[0])


def _map(state):
    return np.asarray(state.world.action_map.map, dtype=np.int32)


def _zone():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[ZONE] = 1
    return target


def test_pickup_fills_to_capacity_and_conserves_mass():
    empty = np.zeros(SHAPE, dtype=np.int8)
    bucket = _bucket(_skid(_zone(), empty))
    capacity = int(CFG.skidsteer_capacity)
    assert bucket.sum() > capacity
    soil = bucket.astype(np.int8)
    for loaded in (0, 30):
        state = _skid(_zone(), soil, loaded=loaded)
        after = pickup(state)
        assert _loaded(after) == capacity
        assert _mass(after) == _mass(state)
        assert _map(after).min() >= 0
        assert int((soil - _map(after)).sum()) == capacity - loaded
    # Less soil than room: all of it.
    few = np.zeros(SHAPE, dtype=np.int8)
    few[tuple(np.argwhere(bucket)[:10].T)] = 1
    after = pickup(_skid(_zone(), few))
    assert _loaded(after) == 10 and _map(after).sum() == 0
    # Full bucket: no change.
    full = _skid(_zone(), soil, loaded=capacity)
    assert _loaded(pickup(full)) == capacity
    np.testing.assert_array_equal(_map(pickup(full)), soil)


def test_pickup_never_takes_accepted_soil_and_keeps_it_contained():
    target = np.zeros(SHAPE, dtype=np.int8)
    target[:, 25:] = 1  # bucket columns 25-27 lie in the accepted zone
    empty = np.zeros(SHAPE, dtype=np.int8)
    bucket = _bucket(_skid(target, empty))
    soil = np.where(bucket, 1, 0).astype(np.int8)
    soil[bucket & (np.arange(SHAPE[1])[None, :] == 25)] = 3  # tall zone edge
    state = _skid(target, soil)
    after = pickup(state)
    zone = target > 0
    off_zone_soil = int(soil[~zone].sum())
    assert 0 < off_zone_soil <= CFG.skidsteer_capacity
    assert _loaded(after) == off_zone_soil
    assert int(_map(after)[~zone].sum()) == 0
    assert int(_map(after)[zone].sum()) == int(soil[zone].sum())
    assert _mass(after) == _mass(state)


def test_reverse_keeps_the_load():
    # The dump zone is in the bucket, so the removed implicit reverse dump
    # would have fired here.
    target = np.zeros(SHAPE, dtype=np.int8)
    target[:, 22:] = 1
    state = _skid(target, np.zeros(SHAPE, dtype=np.int8), loaded=20, shovel=0)
    after = backward(state)
    assert _loaded(after) == 20
    np.testing.assert_array_equal(_map(after), _map(state))
    assert not np.array_equal(
        np.asarray(after.agent.agent_states[0].pos_base), POSE
    )


def test_r2_on_a_haul_only_map_pays_delivery_and_ends_in_success():
    target = _zone()
    empty = np.zeros(SHAPE, dtype=np.int8)
    bucket = _bucket(_skid(target, empty))
    soil = np.zeros(SHAPE, dtype=np.int8)
    soil[tuple(np.argwhere(bucket)[:40].T)] = 1
    state = _skid(target, soil)
    assert float(state._required_excavation_volume()) == 0.0
    assert float(state.material_v_reset) == 40.0
    q, h, p, phi, valid = (float(x) for x in r2_values(state))
    assert valid == 1.0 and q == 0.0 and p == 0.0
    assert h == float(state.material_h_reset) > 0.0

    # Scoop: FORWARD is blocked by the pile and loads it in place.
    loaded = forward(state)
    assert _loaded(loaded) == 40 and _map(loaded).sum() == 0
    _, _, p_loaded, _, valid = (float(x) for x in r2_values(loaded))
    assert valid == 1.0 and abs(p_loaded) < 1e-5  # carry keeps H continuous
    lifted = do(loaded)
    assert int(np.asarray(lifted.agent.agent_states[0].shovel_lifted)[0]) == 1

    # Drive (teleport) to the zone and dump.
    at_zone = lifted._set_agent_state_at(
        0,
        lifted.agent.agent_states[0]._replace(
            pos_base=jnp.asarray(DUMP_POSE, dtype=jnp.int16)
        ),
    )
    dumped = do(at_zone)
    assert _loaded(dumped) == 0
    assert int(_map(dumped)[target > 0].sum()) == 40
    _, h_done, p_done, _, valid = (float(x) for x in r2_values(dumped))
    assert valid == 1.0 and h_done == 0.0
    np.testing.assert_allclose(p_done, float(state.material_h_reset) / 40.0, rtol=1e-6)
    completion = dumped._get_task_completion(
        dumped.world.action_map.map, dumped.world.target_map.map
    )
    assert float(completion["absolute_completion"]) == 1.0

    for before, after, success in (
        (state, loaded, False),
        (loaded, lifted, False),
        (at_zone, dumped, True),
    ):
        reward, terms = r2_reward(before, after, jnp.bool_(success))
        assert np.isfinite(float(reward))
        assert float(terms["reward_v2_valid"]) == 1.0
    delivery, _ = r2_reward(at_zone, dumped, jnp.bool_(False))
    scoop, _ = r2_reward(state, loaded, jnp.bool_(False))
    assert float(delivery) > float(scoop)


def test_observation_normalizers_use_the_haul_volume():
    from terra.env import TerraEnv

    target = _zone()
    soil = np.zeros(SHAPE, dtype=np.int8)
    soil[30:34, 22:27] = 2  # 40 units
    state = _skid(target, soil)
    obs = jax.jit(TerraEnv._state_to_obs_dict)(state)
    context = np.asarray(obs["reward_v2_reset_context"])
    np.testing.assert_allclose(
        context[1], float(state.material_h_reset) / 40.0, rtol=1e-6
    )
    assert np.all(np.isfinite(context)) and context[1] <= REWARD_V2_DISTANCE_BOUND
