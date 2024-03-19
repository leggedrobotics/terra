#!/usr/bin/env python3
import jax
import jax.numpy as jnp

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch


def redraw():
    global obs, states, tile_size, batch_cfg
    env.terra_env.render_obs(
        obs, mode="human", tile_size=tile_size, key_handler=key_handler
    )


def key_handler(event):
    global states, action_type, obs, n_envs, key_maps_buffer

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    print("pressed", event.key)

    if event.key == "escape":
        env.window.close()

    if event.key == "left":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.anticlock()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "right":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.clock()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "up":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.forward()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "down":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.backward()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "a":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action_type.cabin_anticlock()),
            env_cfgs,
            key_maps_buffer,
        )
        parse_step(states, reward, done, info)

    if event.key == "d":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.cabin_clock()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "e":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.extend_arm()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "r":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.retract_arm()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)

    if event.key == "o":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action_type.clock_forward()),
            env_cfgs,
            key_maps_buffer,
        )
        parse_step(states, reward, done, info)

    if event.key == "k":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action_type.clock_backward()),
            env_cfgs,
            key_maps_buffer,
        )
        parse_step(states, reward, done, info)

    if event.key == "i":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action_type.anticlock_forward()),
            env_cfgs,
            key_maps_buffer,
        )
        parse_step(states, reward, done, info)

    if event.key == "l":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states,
            repeat_action(action_type.anticlock_backward()),
            env_cfgs,
            key_maps_buffer,
        )
        parse_step(states, reward, done, info)

    if event.key == " ":
        states, (obs, reward, done, info), key_maps_buffer = env.step(
            states, repeat_action(action_type.do()), env_cfgs, key_maps_buffer
        )
        parse_step(states, reward, done, info)


def parse_step(obs, reward, done, info):
    redraw()
    print("Reward :", reward)
    print("Done: ", done)


batch_cfg = BatchConfig()
action_type = batch_cfg.action_type
n_envs = 3
seeds = jnp.array([24, 35245, 65])
env = TerraEnvBatch(rendering=True, n_imgs_row=n_envs)

env_cfgs = jax.vmap(lambda x: EnvConfig.new())(jnp.arange(n_envs))
states, obs, key_maps_buffer = env.reset(seeds, env_cfgs)

tile_size = 16

states, (obs, rewards, dones, infos), key_maps_buffer = env.step(
    states,
    action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0)),
    env_cfgs,
    key_maps_buffer,
)
env.terra_env.render_obs(obs, key_handler=key_handler, block=True, tile_size=tile_size)
