#!/usr/bin/env python3

from src.env import TerraEnv
from src.actions import TrackedActionType
from src.config import EnvConfig
# import argparse


def redraw():
    global state
    # if not args.agent_view:
    #     img = env.render("rgb_array", tile_size=args.tile_size)

    # img = env.render(state=state, mode="human", tile_size=32, key_handler=lambda event: key_handler(event, state))
    img = env.render(state=state, mode="human", tile_size=32, key_handler=key_handler)

    env.window.show_img(img)
    # env.window_target.show_img(img_target)


# def reset():
#     env.reset()
#     # env.level_up()
#     # if args.seed != -1:
#     #     env.seed(args.seed)

#     # if hasattr(env, "mission"):
#     #     print("Mission: %s" % env.mission)
#     #     env.window.set_caption(env.mission)

#     redraw()

def key_handler(event):

    global state

    print("pressed", event.key)

    if event.key == "escape":
        env.window.close()
        # env.window_target.close()

    # if event.key == "backspace":
    #     reset()

    # if event.key == "left":
    #     obs, reward, done, info = env.step(env.actions.rotate_base_counter)
    #     parse_step(obs, reward, done, info)

    # if event.key == "right":
    #     obs, reward, done, info = env.step(env.actions.rotate_base_clock)
    #     parse_step(obs, reward, done, info)

    if event.key == "up":
        _, (state, reward, done, info) = env.step(state, TrackedActionType.FORWARD)
        parse_step(state, reward, done, info)

    if event.key == "down":
        _, (state, reward, done, info) = env.step(state, TrackedActionType.BACKWARD)
        parse_step(state, reward, done, info)

    # if event.key == "a":
    #     obs, reward, done, info = env.step(env.actions.rotate_cabin_counter)
    #     parse_step(obs, reward, done, info)

    # if event.key == "d":
    #     obs, reward, done, info = env.step(env.actions.rotate_cabin_clock)
    #     parse_step(obs, reward, done, info)

    # if event.key == " ":
    #     obs, reward, done, info = env.step(env.actions.do)
    #     parse_step(obs, reward, done, info)
    #     return

    # if event.key == "backspace":
    #     print("resetting")
    #     parse_step(0, 0, 0, 0)

    # if event.key == "enter":
    #     # obs, reward, done, info =  (env.actions.done)
    #     print("resetting")
    #     state = env.reset()
    #     redraw()
    #     return


def parse_step(obs, reward, done, info):
    redraw()
    # print("Observations [:, :, 0] \n", obs['image'][:, :, 0])
    # print("Observations [:, :, 1] \n", obs['image'][:, :, 1])
    # print("Observations [:, :, 2] \n", obs['image'][:, :, 2])
    # print("obs \n", obs)
    # print("obs \n", obs)

    print("Reward :", reward)
    print("Done: ", done)


# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--env", help="gym environment to load", default="HeightGrid-Empty-5x5-v0"
# )
# parser.add_argument(
#     "--seed", type=int, help="random seed to generate the environment with", default=-1
# )
# parser.add_argument(
#     "--tile_size", type=int, help="size at which to render tiles", default=32
# )
# parser.add_argument(
#     "--agent_view", 
#     default=False,
#     help="draw the agent sees (partially observable view)",
#     action="store_true",
# )

# args = parser.parse_args()

# # grid_height = np.zeros((5,5))
# # grid_height[1, 3] = 1
# # env = gym.make(args.env)
# # env = EmptyEnv5x5()
# rewards = {"collision_reward": -1, # against wall 0, ok
#            "longitudinal_step_reward": -0.1,
#            "base_turn_reward": -0.2, # ok
#            "dig_reward": 1, # ok
#            "dig_wrong_reward": -2, # ok
#            "move_dirt_reward": 1,
#            "existence_reward": -0.05, # ok
#            "cabin_turn_reward": -0.05, # ok
#            "terminal_reward": 10}

env = TerraEnv(EnvConfig())
print(env)
seed = 24
state = env.reset(seed=seed)
# if args.agent_view:
#     env = FullyObsWrapper(env)
# env = ImgObsWrapper(env)

# window = Window('heightgrid - ' + args.env)
# env.render(state=state, key_handler=lambda event: key_handler(event, state), block=True)
env.render(state=state, key_handler=key_handler, block=True)

# env.window.reg_key_handler(key_handler)
# env.window_target.reg_key_handler(key_handler)

# # reset()

# # # Blocking event loop
# env.window.show(block=True)
# env.window_target.show(block=True)
