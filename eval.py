import numpy as np
import jax
import math
from utils.models import get_model_ready
from utils.helpers import load_pkl_object
from terra.env import TerraEnvBatch
from terra.actions import (
    WheeledAction,
    TrackedAction,
    WheeledActionType,
    TrackedActionType,
)
import jax.numpy as jnp
from utils.utils_ppo import obs_to_model_input, wrap_action

# from utils.curriculum import Curriculum
from tensorflow_probability.substrates import jax as tfp
from train import TrainConfig  # needed for unpickling checkpoints


def load_neural_network(config, env):
    rng = jax.random.PRNGKey(0)
    model, _ = get_model_ready(rng, config, env)
    return model


def _append_to_obs(o, obs_log):
    if obs_log == {}:
        return {k: v[:, None] for k, v in o.items()}
    obs_log = {
        k: jnp.concatenate((v, o[k][:, None]), axis=1) for k, v in obs_log.items()
    }
    return obs_log


def rollout_episode(
    env: TerraEnvBatch,
    model,
    model_params,
    env_cfgs,
    rl_config,
    max_frames,
    deterministic,
    seed,
):
    """
    NOTE: this function assumes it's a tracked agent in the way it computes the stats.
    """
    print(f"Using {seed=}")
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, rl_config.num_test_rollouts)
    timestep = env.reset(env_cfgs, rng_reset)
    prev_actions = jnp.zeros(
        (rl_config.num_test_rollouts, rl_config.num_prev_actions),
        dtype=jnp.int32
    )

    tile_size = env_cfgs.tile_size[0].item()
    move_tiles = env_cfgs.agent.move_tiles[0].item()

    action_type = env.batch_cfg.action_type
    if action_type == TrackedAction:
        move_actions = (TrackedActionType.FORWARD, TrackedActionType.BACKWARD)
        l_actions = ()
        do_action = TrackedActionType.DO
    elif action_type == WheeledAction:
        move_actions = (WheeledActionType.FORWARD, WheeledActionType.BACKWARD)
        l_actions = (
            WheeledActionType.CLOCK_FORWARD,
            WheeledActionType.CLOCK_BACKWARD,
            WheeledActionType.ANTICLOCK_FORWARD,
            WheeledActionType.ANTICLOCK_BACKWARD,
        )
        do_action = WheeledActionType.DO
    else:
        raise (ValueError(f"{action_type=}"))

    obs = timestep.observation
    areas = (obs["target_map"] == -1).sum(
        tuple([i for i in range(len(obs["target_map"].shape))][1:])
    ) * (tile_size**2)
    target_maps_init = obs["target_map"].copy()
    dig_tiles_per_target_map_init = (target_maps_init == -1).sum(
        tuple([i for i in range(len(target_maps_init.shape))][1:])
    )

    t_counter = 0
    reward_seq = []
    episode_done_once = None
    episode_length = None
    move_cumsum = None
    do_cumsum = None
    obs_seq = {}
    while True:
        obs_seq = _append_to_obs(obs, obs_seq)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        if model is not None:
            obs_model = obs_to_model_input(timestep.observation, prev_actions, rl_config)
            v, logits_pi = model.apply(model_params, obs_model)
            if deterministic:
                action = np.argmax(logits_pi, axis=-1)
            else:
                pi = tfp.distributions.Categorical(logits=logits_pi)
                action = pi.sample(seed=rng_act)
            prev_actions = jnp.roll(prev_actions, shift=1, axis=1)
            prev_actions = prev_actions.at[:, 0].set(action)
        else:
            raise RuntimeError("Model is None!")
        rng_step = jax.random.split(rng_step, rl_config.num_test_rollouts)
        timestep = env.step(
            timestep, wrap_action(action, env.batch_cfg.action_type), rng_step
        )
        reward = timestep.reward
        next_obs = timestep.observation
        done = timestep.done

        reward_seq.append(reward)
        print(t_counter)
        print(10 * "=")
        t_counter += 1
        if jnp.all(done).item() or t_counter == max_frames:
            break
        obs = next_obs

        # Log stats
        if episode_done_once is None:
            episode_done_once = done
        if episode_length is None:
            episode_length = jnp.zeros_like(done, dtype=jnp.int32)
        if move_cumsum is None:
            move_cumsum = jnp.zeros_like(done, dtype=jnp.int32)
        if do_cumsum is None:
            do_cumsum = jnp.zeros_like(done, dtype=jnp.int32)

        episode_done_once = episode_done_once | done

        episode_length += ~episode_done_once

        move_cumsum_tmp = jnp.zeros_like(done, dtype=jnp.int32)
        for move_action in move_actions:
            move_mask = (action == move_action) * (~episode_done_once)
            move_cumsum_tmp += move_tiles * tile_size * move_mask.astype(jnp.int32)
        for l_action in l_actions:
            l_mask = (action == l_action) * (~episode_done_once)
            move_cumsum_tmp += 2 * move_tiles * tile_size * l_mask.astype(jnp.int32)
        move_cumsum += move_cumsum_tmp

        do_cumsum += (action == do_action) * (~episode_done_once)

    # Path efficiency -- only include finished envs
    move_cumsum *= episode_done_once
    path_efficiency = (move_cumsum / jnp.sqrt(areas))[episode_done_once]
    path_efficiency_std = path_efficiency.std()
    path_efficiency_mean = path_efficiency.mean()

    # Workspaces efficiency -- only include finished envs
    reference_workspace_area = 0.5 * np.pi * (8**2)
    n_dig_actions = do_cumsum // 2
    workspaces_efficiency = (
        reference_workspace_area
        * ((n_dig_actions * episode_done_once) / areas)[episode_done_once]
    )
    workspaces_efficiency_mean = workspaces_efficiency.mean()
    workspaces_efficiency_std = workspaces_efficiency.std()

    # Coverage scores
    dug_tiles_per_action_map = (obs["action_map"] == -1).sum(
        tuple([i for i in range(len(obs["action_map"].shape))][1:])
    )
    coverage_ratios = dug_tiles_per_action_map / dig_tiles_per_target_map_init
    coverage_scores = episode_done_once + (~episode_done_once) * coverage_ratios
    coverage_score_mean = coverage_scores.mean()
    coverage_score_std = coverage_scores.std()

    stats = {
        "episode_done_once": episode_done_once,
        "episode_length": episode_length,
        "path_efficiency": {
            "mean": path_efficiency_mean,
            "std": path_efficiency_std,
        },
        "workspaces_efficiency": {
            "mean": workspaces_efficiency_mean,
            "std": workspaces_efficiency_std,
        },
        "coverage": {
            "mean": coverage_score_mean,
            "std": coverage_score_std,
        },
    }
    print(episode_done_once)
    return np.cumsum(reward_seq), stats, obs_seq


def print_stats(
    stats,
):
    episode_done_once = stats["episode_done_once"]
    episode_length = stats["episode_length"]
    path_efficiency = stats["path_efficiency"]
    workspaces_efficiency = stats["workspaces_efficiency"]
    coverage = stats["coverage"]

    completion_rate = 100 * episode_done_once.sum() / len(episode_done_once)

    print("\nStats:\n")
    print(f"Completion: {completion_rate:.2f}%")
    # print(f"First episode length average: {episode_length.mean()}")
    # print(f"First episode length min: {episode_length.min()}")
    # print(f"First episode length max: {episode_length.max()}")
    print(
        f"Path efficiency: {path_efficiency['mean']:.2f} ({path_efficiency['std']:.2f})"
    )
    print(
        f"Workspaces efficiency: {workspaces_efficiency['mean']:.2f} ({workspaces_efficiency['std']:.2f})"
    )
    print(f"Coverage: {coverage['mean']:.2f} ({coverage['std']:.2f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-run",
        "--run_name",
        type=str,
        default="ppo_2023_05_09_10_01_23",
        help="es/ppo trained agent.",
    )
    parser.add_argument(
        "-env",
        "--env_name",
        type=str,
        default="Terra",
        help="Environment name.",
    )
    parser.add_argument(
        "-n",
        "--n_envs",
        type=int,
        default=1,
        help="Number of environments.",
    )
    parser.add_argument(
        "-steps",
        "--n_steps",
        type=int,
        default=10,
        help="Number of steps.",
    )
    parser.add_argument(
        "-d",
        "--deterministic",
        type=int,
        default=0,
        help="Deterministic. 0 for stochastic, 1 for deterministic.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed for the environment.",
    )
    args, _ = parser.parse_known_args()
    n_envs = args.n_envs

    log = load_pkl_object(f"{args.run_name}")
    config = log["train_config"]
    # from utils.helpers import load_config
    # config = load_config("agents/Terra/ppo.yaml", 22333, 33222, 5e-04, True, "")["train_config"]

    config.num_test_rollouts = n_envs
    config.num_devices = 1

    # curriculum = Curriculum(rl_config=config, n_devices=n_devices)
    # env_cfgs, dofs_count_dict = curriculum.get_cfgs_eval()
    env_cfgs = log["env_config"]
    env_cfgs = jax.tree_map(
        lambda x: x[0][None, ...].repeat(n_envs, 0), env_cfgs
    )  # take first config and replicate
    shuffle_maps = True
    env = TerraEnvBatch(rendering=False, shuffle_maps=shuffle_maps)
    config.num_embeddings_agent_min = 60

    model = load_neural_network(config, env)
    model_params = log["model"]
    # model_params = jax.tree_map(lambda x: x[0], replicated_params)
    deterministic = bool(args.deterministic)
    print(f"\nDeterministic = {deterministic}\n")

    cum_rewards, stats, _ = rollout_episode(
        env,
        model,
        model_params,
        env_cfgs,
        config,
        max_frames=args.n_steps,
        deterministic=deterministic,
        seed=args.seed,
    )

    print_stats(stats)
