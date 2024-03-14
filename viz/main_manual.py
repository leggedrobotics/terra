#!/usr/bin/env python3
import jax
import jax.numpy as jnp

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def redraw():
    global obs, states, tile_size, batch_cfg, info
    env.terra_env.render_obs(
        obs, mode="human", tile_size=tile_size, key_handler=key_handler, info=info
    )


def key_handler(event):
    global states, action_type, obs, n_envs, key_maps_buffer, info, i, action_mask, ACTIVATE_ACTION_MASKING
    i += 1
    print(f"step {i}")

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

    elif event.key == "left":
        action = action_type.anticlock()

    elif event.key == "right":
        action = action_type.clock()

    elif event.key == "up":
        action = action_type.forward()

    elif event.key == "down":
        action = action_type.backward()

    elif event.key == "a":
        action = action_type.cabin_anticlock()

    elif event.key == "d":
        action = action_type.cabin_clock()

    elif event.key == "e":
        action = action_type.extend_arm()

    elif event.key == "r":
        action = action_type.retract_arm()

    elif event.key == "o":
        action = action_type.clock_forward()

    elif event.key == "k":
        action = action_type.clock_backward()

    elif event.key == "i":
        action = action_type.anticlock_forward()

    elif event.key == "l":
        action = action_type.anticlock_backward()

    elif event.key == " ":
        action = action_type.do()

    else:
        return

    action = apply_action_mask(action)
    _transition, key_maps_buffer = env.step(
        states,
        repeat_action(action),
        env_cfgs,
        key_maps_buffer,
    )

    states = _transition.next_state
    obs = _transition.obs
    reward = _transition.reward
    done = _transition.done
    info = _transition.infos
    action_mask = _transition.infos["action_mask"]
    parse_step(states, reward, done, info)


def parse_step(obs, reward, done, info):
    redraw()
    print("Reward :", reward)
    print("Done: ", done)


ACTIVATE_ACTION_MASKING = False

batch_cfg = BatchConfig()
action_type = batch_cfg.action_type
n_envs_x = 1
n_envs_y = 1
n_envs = n_envs_x * n_envs_y
seeds = jnp.array([jax.random.PRNGKey(3)])
env = TerraEnvBatch(rendering=True, n_envs_x_rendering=n_envs_x, n_envs_y_rendering=n_envs_y)

env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
states, obs, key_maps_buffer = env.reset(seeds, env_cfgs)

tile_size = 16

transition, maps_buffer_keys = env.step(
    states,
    action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0)),
    env_cfgs,
    key_maps_buffer,
)
action_mask = transition.infos["action_mask"]
i = 0
env.terra_env.render_obs(
    obs, key_handler=key_handler, block=True, tile_size=tile_size, info=transition.infos
)
