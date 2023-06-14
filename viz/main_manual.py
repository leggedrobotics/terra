#!/usr/bin/env python3
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
    global states, action_type, obs, n_envs

    def repeat_action(action, n_times=n_envs):
        return action_type.new(action.action[None].repeat(n_times, 0))

    print("pressed", event.key)

    if event.key == "escape":
        env.window.close()
        # env.window_target.close()

    # if event.key == "backspace":
    #     reset()

    if event.key == "left":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock())
        )
        parse_step(states, reward, done, info)

    if event.key == "right":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock())
        )
        parse_step(states, reward, done, info)

    if event.key == "up":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.forward())
        )
        parse_step(states, reward, done, info)

    if event.key == "down":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.backward())
        )
        parse_step(states, reward, done, info)

    if event.key == "a":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.cabin_anticlock())
        )
        parse_step(states, reward, done, info)

    if event.key == "d":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.cabin_clock())
        )
        parse_step(states, reward, done, info)

    if event.key == "e":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.extend_arm())
        )
        parse_step(states, reward, done, info)

    if event.key == "r":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.retract_arm())
        )
        parse_step(states, reward, done, info)

    if event.key == "o":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock_forward())
        )
        parse_step(states, reward, done, info)

    if event.key == "k":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.clock_backward())
        )
        parse_step(states, reward, done, info)

    if event.key == "i":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock_forward())
        )
        parse_step(states, reward, done, info)

    if event.key == "l":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.anticlock_backward())
        )
        parse_step(states, reward, done, info)

    if event.key == " ":
        states, (obs, reward, done, info) = env.step(
            states, repeat_action(action_type.do())
        )
        parse_step(states, reward, done, info)
        return


def parse_step(obs, reward, done, info):
    redraw()

    print("Reward :", reward)
    print("Done: ", done)


env_cfg = EnvConfig()
batch_cfg = BatchConfig()
action_type = batch_cfg.action_type
n_envs = 1
env = TerraEnvBatch(rendering=True, n_imgs_row=1, n_envs=n_envs)
print(env)
seeds = jnp.array([24])
states, obs = env.reset(seeds)

tile_size = 16

states, (obs, rewards, dones, infos) = env.step(
    states, action_type.new(action_type.do_nothing().action[None].repeat(n_envs, 0))
)
env.terra_env.render_obs(obs, key_handler=key_handler, block=True, tile_size=tile_size)
