#!/usr/bin/env python3
import jax
import jax.numpy as jnp

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch



def _up():
    global states, action_type, obs, n_envs, key_maps_buffer, info, i, action_mask, ACTIVATE_ACTION_MASKING
    action = action_type.forward()
    print("Applying action FORWARD")
    states, (obs, reward, done, info), key_maps_buffer = env.step(
        states,
        repeat_action(action, n_envs),
        env_cfgs,
        key_maps_buffer,
    )
    parse_step(states, reward, done, info)

def redraw():
    global obs, states, tile_size, batch_cfg, info
    env.terra_env.render_obs(
        obs, info=info, generate_gif=True,
    )

def repeat_action(action, n_envs):
        return action_type.new(action.action[None].repeat(n_envs, 0))


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

    

    print("pressed", event.key)

    if event.key == "escape":
        env.window.close()

    if event.key == "left":
        action = action_type.anticlock()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "right":
        action = action_type.clock()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "up":
        action = action_type.forward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "down":
        action = action_type.backward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "a":
        action = action_type.cabin_anticlock()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "d":
        action = action_type.cabin_clock()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "e":
        action = action_type.extend_arm()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "r":
        action = action_type.retract_arm()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "o":
        action = action_type.clock_forward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "k":
        action = action_type.clock_backward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "i":
        action = action_type.anticlock_forward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == "l":
        action = action_type.anticlock_backward()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
        parse_step(states, reward, done, info)

    if event.key == " ":
        action = action_type.do()
        action = apply_action_mask(action)
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action),
            env_cfgs,
            key_maps_buffer,
        )
        action_mask = info["action_mask"]
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

seeds = jnp.array([33])
env = TerraEnvBatch(rendering=True, n_envs_x_rendering=n_envs_x, n_envs_y_rendering=n_envs_y)

env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
states, obs, key_maps_buffer = env.reset(seeds, env_cfgs)

tile_size = 16

states, (obs, rewards, dones, info), key_maps_buffer = env.step(
    states,
    action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0)),
    env_cfgs,
    key_maps_buffer,
)
action_mask = info["action_mask"]
i = 0
env.terra_env.render_obs_pygame(
    obs, info=info, generate_gif=True
)

print(f"{obs=}")

import time
time.sleep(10000)
# for _ in range(10):
#     print(f"{obs=}\n")
#     time.sleep(1)
#     _up()

# env.terra_env.rendering_engine.create_gif()
