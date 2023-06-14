#!/usr/bin/env python3
import jax
import jax.numpy as jnp

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.env import TerraEnvBatch
from terra.utils import init_maps_buffer


def redraw():
    global obs, states, tile_size, batch_cfg, maps_buffer
    env.terra_env.render_obs(
        obs, mode="human", tile_size=tile_size, key_handler=key_handler
    )


def key_handler(event):
    global states, action_type, obs, n_envs, maps_buffer

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    print("pressed", event.key)

    maps = maps_buffer.sample(n_envs)

    if event.key == "escape":
        env.window.close()

    if event.key == "left":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "right":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "up":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.forward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "down":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.backward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "a":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.cabin_anticlock()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "d":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.cabin_clock()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "e":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.extend_arm()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "r":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.retract_arm()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "o":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock_forward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "k":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock_backward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "i":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock_forward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == "l":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock_backward()), maps
        )
        parse_step(states, reward, done, info)

    if event.key == " ":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.do()), maps
        )
        parse_step(states, reward, done, info)

    maps_buffer = maps_buffer.shuffle()


def parse_step(obs, reward, done, info):
    redraw()
    print("Reward :", reward)
    print("Done: ", done)


env_cfg = EnvConfig()
batch_cfg = BatchConfig()
action_type = batch_cfg.action_type
n_envs = 3
seeds = jnp.array([24, 33, 16])
env = TerraEnvBatch(rendering=True, n_imgs_row=n_envs)
key_maps_buffer = jax.random.PRNGKey(332)
maps_buffer = init_maps_buffer(key_maps_buffer, env.env_cfg)
maps = maps_buffer.sample(n_envs)
states, obs = env.reset(seeds, maps)

tile_size = 16

states, (obs, rewards, dones, infos) = env.step(
    states,
    action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0)),
    maps,
)
env.terra_env.render_obs(obs, key_handler=key_handler, block=True, tile_size=tile_size)
