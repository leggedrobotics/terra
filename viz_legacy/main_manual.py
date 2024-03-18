#!/usr/bin/env python3
import jax
import jax.numpy as jnp

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def redraw():
    global timestep, tile_size, batch_cfg
    env.terra_env.render_obs(
        timestep.observation, mode="human", tile_size=tile_size, key_handler=key_handler, info=timestep.info
    )


def key_handler(event):
    global timestep, action_type, n_envs, rng, i, action_mask, ACTIVATE_ACTION_MASKING
    i += 1
    print(f"step {i}")
    rng, _rng = jax.random.split(rng)
    _rng = _rng[None, ...]
    
    def apply_action_mask(action):
        if ACTIVATE_ACTION_MASKING:
            action_item = action.action.item()
            if action_mask[0][action_item].item() == False:
                action = action_type.do_nothing()
                print(f"MASKED OUT: action {action_item}.")
        return action

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    print("pressed", event.key)

    if event.key == "escape":
        env.window.close()

    if event.key == "left":
        action = action_type.anticlock()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "right":
        action = action_type.clock()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "up":
        action = action_type.forward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "down":
        action = action_type.backward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "a":
        action = action_type.cabin_anticlock()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "d":
        action = action_type.cabin_clock()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "e":
        action = action_type.extend_arm()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "r":
        action = action_type.retract_arm()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "o":
        action = action_type.clock_forward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "k":
        action = action_type.clock_backward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "i":
        action = action_type.anticlock_forward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == "l":
        action = action_type.anticlock_backward()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)

    if event.key == " ":
        action = action_type.do()
        action = apply_action_mask(action)
        timestep = env.step(
            env_cfgs,
            timestep,
            repeat_action(action),
            _rng,
        )
        action_mask = timestep.info["action_mask"]
        parse_step(timestep)


def parse_step(timestep):
    redraw()
    print("Reward :", timestep.reward)
    print("Done: ", timestep.done)


ACTIVATE_ACTION_MASKING = False

batch_cfg = BatchConfig()
action_type = batch_cfg.action_type
n_envs_x = 1
n_envs_y = 1
n_envs = n_envs_x * n_envs_y
seed = 24
rng = jax.random.PRNGKey(seed)
env = TerraEnvBatch(rendering=True, n_envs_x_rendering=n_envs_x, n_envs_y_rendering=n_envs_y)

env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
rng, _rng = jax.random.split(rng)
_rng = _rng[None, ...]
timestep = env.reset(env_cfgs, _rng)

tile_size = 16

rng, _rng = jax.random.split(rng)
_rng = _rng[None, ...]
timestep = env.step(
    env_cfgs,
    timestep,
    action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0)),
    _rng,
)
action_mask = timestep.info["action_mask"]
i = 0
env.terra_env.render_obs(
    timestep.observation, key_handler=key_handler, block=True, tile_size=tile_size, info=timestep.info
)
