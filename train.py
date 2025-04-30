import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from utils.models import get_model_ready
from terra.env import TerraEnvBatch
from terra.config import EnvConfig
from flax.training.train_state import TrainState
import optax
import wandb
import eval_ppo
from datetime import datetime
from dataclasses import asdict, dataclass
import time
from tqdm import tqdm
from functools import partial
from flax.jax_utils import replicate, unreplicate
from flax import struct
import utils.helpers as helpers
from utils.utils_ppo import select_action_ppo, wrap_action, obs_to_model_input, policy
import os

jax.config.update("jax_threefry_partitionable", True)


@dataclass
class TrainConfig:
    name: str
    num_devices: int = 0
    project: str = "main"
    group: str = "default"
    num_envs_per_device: int = 4096
    num_steps: int = 32
    update_epochs: int = 5
    num_minibatches: int = 32
    total_timesteps: int = 30_000_000_000
    lr: float = 3e-4
    clip_eps: float = 0.5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 5.0
    max_grad_norm: float = 0.5
    eval_episodes: int = 100
    seed: int = 42
    log_train_interval: int = 1  # Number of updates between logging train stats
    log_eval_interval: int = (
        50  # Number of updates between running eval and syncing with wandb
    )
    checkpoint_interval: int = 50  # Number of updates between checkpoints
    # model settings
    num_prev_actions = 5
    clip_action_maps = True  # clips the action maps to [-1, 1]
    local_map_normalization_bounds = [-16, 16]
    loaded_max = 100
    num_rollouts_eval = 400  # max length of an episode in Terra for eval (for training it is in Terra's curriculum)

    def __post_init__(self):
        self.num_devices = (
            jax.local_device_count() if self.num_devices == 0 else self.num_devices
        )
        self.num_envs = self.num_envs_per_device * self.num_devices
        self.total_timesteps_per_device = self.total_timesteps // self.num_devices
        self.eval_episodes_per_device = self.eval_episodes // self.num_devices
        assert (
            self.num_envs % self.num_devices == 0
        ), "Number of environments must be divisible by the number of devices."
        self.num_updates = (
            self.total_timesteps // (self.num_steps * self.num_envs)
        ) // self.num_devices
        print(f"Num devices: {self.num_devices}, Num updates: {self.num_updates}")

    # make object subscriptable
    def __getitem__(self, key):
        return getattr(self, key)


def make_states(config: TrainConfig):
    env = TerraEnvBatch()
    num_devices = config.num_devices
    num_envs_per_device = config.num_envs_per_device

    env_params = EnvConfig()
    env_params = jax.tree_map(
        lambda x: jnp.array(x)[None, None]
        .repeat(num_devices, 0)
        .repeat(num_envs_per_device, 1),
        env_params,
    )
    print(f"{env_params.tile_size.shape=}")

    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)

    network, network_params = get_model_ready(_rng, config, env)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx
    )

    return rng, env, env_params, train_state


class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    # for rnn policy
    prev_actions: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = (
            transition.reward
            + gamma * next_value * (1 - transition.done)
            - transition.value
        )
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    config,
):
    clip_eps = config.clip_eps
    vf_coef = config.vf_coef
    ent_coef = config.ent_coef

    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # Terra: Reshape
        # [minibatch_size, seq_len, ...] -> [minibatch_size * seq_len, ...]
        print(f"ppo_update_networks {transitions.obs['agent_state'].shape=}")
        print(f"ppo_update_networks {transitions.prev_actions.shape=}")
        transitions_obs_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.obs,
        )
        transitions_actions_reshaped = jax.tree_map(
            lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])),
            transitions.prev_actions,
        )
        print(f"ppo_update_networks {transitions_obs_reshaped['agent_state'].shape=}")
        print(f"ppo_update_networks {transitions_actions_reshaped.shape=}")

        # NOTE: can't use select_action_ppo here because it doesn't decouple params from train_state
        obs = obs_to_model_input(transitions_obs_reshaped, transitions_actions_reshaped, config)
        value, dist = policy(train_state.apply_fn, params, obs)
        value = value[:, 0]
        # action = dist.sample(seed=rng_model)
        transitions_actions_reshaped = jnp.reshape(
            transitions.action, (-1, *transitions.action.shape[2:])
        )
        log_prob = dist.log_prob(transitions_actions_reshaped)

        # Terra: Reshape
        value = jnp.reshape(value, transitions.value.shape)
        log_prob = jnp.reshape(log_prob, transitions.log_prob.shape)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(
            -clip_eps, clip_eps
        )
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
        train_state.params
    )
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean(
        (loss, vloss, aloss, entropy, grads), axis_name="devices"
    )
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


def get_curriculum_levels(env_cfg, global_curriculum_levels):
    curriculum_stat = {}
    curriculum_levels = env_cfg.curriculum.level
    for i, global_curriculum_level in enumerate(global_curriculum_levels):
        curriculum_stat[f'Level {i}: {global_curriculum_level["maps_path"]}'] = jnp.sum(
            curriculum_levels == i
        ).item()
    return curriculum_stat


def make_train(
    env: TerraEnvBatch,
    env_params: EnvConfig,
    config: TrainConfig,
):
    def train(
        rng: jax.Array,
        train_state: TrainState,
    ):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(
            _rng, config.num_envs_per_device * config.num_devices
        )
        reset_rng = reset_rng.reshape(
            (config.num_devices, config.num_envs_per_device, -1)
        )

        # TERRA: Reset envs
        reset_fn_p = jax.pmap(env.reset, axis_name="devices")  # vmapped inside
        timestep = reset_fn_p(env_params, reset_rng)
        prev_actions = jnp.zeros(
            (config.num_devices, config.num_envs_per_device, config.num_prev_actions), dtype=jnp.int32
        )
        prev_reward = jnp.zeros((config.num_devices, config.num_envs_per_device))

        # TRAIN LOOP
        @partial(jax.pmap, axis_name="devices")
        def _update_step(runner_state, _):
            """
            Performs a single update step in the training loop.

            This function orchestrates the collection of trajectories from the environment, calculation of advantages, and updating of the network parameters based on the collected data. It involves stepping through the environment to collect data, calculating the advantage estimates for each step, and performing several epochs of updates on the network parameters using the collected data.

            Parameters:
            - runner_state: A tuple containing the current state of the RNG, the training state, the previous timestep, the previous action, and the previous reward.
            - _: Placeholder to match the expected input signature for jax.lax.scan.

            Returns:
            - runner_state: Updated runner state after performing the update step.
            - loss_info: A dictionary containing information about the loss and other metrics for this update step.
            """

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                """
                Executes a step in the environment for all agents.

                This function takes the current state of the runners (agents), selects an action for each agent based on the current observation using the PPO algorithm, and then steps the environment forward using these actions. The environment returns the next state, reward, and whether the episode has ended for each agent. These are then used to create a transition tuple containing the current state, action, reward, and next state, which can be used for training the model.

                Parameters:
                - runner_state: Tuple containing the current rng state, train_state, previous timestep, previous action, and previous reward.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - runner_state: Updated runner state after stepping the environment.
                - transition: A namedtuple containing the transition information (current state, action, reward, next state) for this step.
                """
                rng, train_state, prev_timestep, prev_actions, prev_reward = runner_state

                # SELECT ACTION
                rng, _rng_model, _rng_env = jax.random.split(rng, 3)
                action, log_prob, value, _ = select_action_ppo(
                    train_state, prev_timestep.observation, prev_actions, _rng_model, config
                )

                # STEP ENV
                _rng_env = jax.random.split(_rng_env, config.num_envs_per_device)
                action_env = wrap_action(action, env.batch_cfg.action_type)
                timestep = env.step(prev_timestep, action_env, _rng_env)
                transition = Transition(
                    # done=timestep.last(),
                    done=timestep.done,
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_actions=prev_actions,
                    prev_reward=prev_reward,
                )

                # UPDATE PREVIOUS ACTIONS
                prev_actions = jnp.roll(prev_actions, shift=1, axis=-1)
                prev_actions = prev_actions.at[..., 0].set(action)

                runner_state = (rng, train_state, timestep, prev_actions, timestep.reward)
                return runner_state, transition

            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_actions, prev_reward = runner_state
            rng, _rng = jax.random.split(rng)
            _, _, last_val, _ = select_action_ppo(
                train_state, timestep.observation, prev_actions, _rng, config
            )
            # advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)
            advantages, targets = calculate_gae(
                transitions, last_val, config.gamma, config.gae_lambda
            )

            # UPDATE NETWORK
            def _update_epoch(update_state, _):
                """
                Performs a single epoch of updates on the network parameters.

                This function iterates over minibatches of the collected data and applies updates to the network parameters based on the PPO algorithm. It is called multiple times to perform multiple epochs of updates.

                Parameters:
                - update_state: A tuple containing the current state of the RNG, the training state, and the collected transitions, advantages, and targets.
                - _: Placeholder to match the expected input signature for jax.lax.scan.

                Returns:
                - update_state: Updated state after performing the epoch of updates.
                - update_info: Information about the updates performed in this epoch.
                """

                def _update_minbatch(train_state, batch_info):
                    """
                    Updates the network parameters based on a single minibatch of data.

                    This function applies the PPO update rule to the network parameters using the data from a single minibatch. It is called for each minibatch in an epoch.

                    Parameters:
                    - train_state: The current training state, including the network parameters.
                    - batch_info: A tuple containing the transitions, advantages, and targets for the minibatch.

                    Returns:
                    - new_train_state: The training state after applying the updates.
                    - update_info: Information about the updates performed on this minibatch.
                    """
                    transitions, advantages, targets = batch_info
                    new_train_state, update_info = ppo_update_networks(
                        train_state=train_state,
                        transitions=transitions,
                        advantages=advantages,
                        targets=targets,
                        config=config,
                    )
                    return new_train_state, update_info

                rng, train_state, transitions, advantages, targets = update_state

                # MINIBATCHES PREPARATION
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                # [seq_len, batch_size, ...]
                batch = (transitions, advantages, targets)
                # [batch_size, seq_len, ...], as our model assumes
                batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                shuffled_batch = jtu.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # [num_minibatches, minibatch_size, seq_len, ...]
                minibatches = jtu.tree_map(
                    lambda x: jnp.reshape(
                        x, (config.num_minibatches, -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )
                train_state, update_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (rng, train_state, transitions, advantages, targets)
                return update_state, update_info

            # [seq_len, batch_size, num_layers, hidden_dim]
            update_state = (rng, train_state, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]
            # EVALUATE AGENT
            rng, _rng = jax.random.split(rng)

            runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
            return runner_state, loss_info

        # Setup runner state for multiple devices

        rng, rng_rollout = jax.random.split(rng)
        rng = jax.random.split(rng, num=config.num_devices)
        train_state = replicate(train_state, jax.local_devices()[: config.num_devices])
        runner_state = (rng, train_state, timestep, prev_actions, prev_reward)
        # runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_updates)
        for i in tqdm(range(config.num_updates), desc="Training"):
            start_time = time.time()  # Start time for measuring iteration speed
            runner_state, loss_info = jax.block_until_ready(
                _update_step(runner_state, None)
            )
            end_time = time.time()

            iteration_duration = end_time - start_time
            iterations_per_second = 1 / iteration_duration
            steps_per_second = (
                iterations_per_second
                * config.num_steps
                * config.num_envs
                * config.num_devices
            )

            tqdm.write(
                f"Steps/s: {steps_per_second:.2f}"
            )  # Display steps and iterations per second

            # Use data from the first device for stats and eval
            loss_info_single = unreplicate(loss_info)
            runner_state_single = unreplicate(runner_state)
            _, train_state, timestep, prev_actions = runner_state_single[:4]
            env_params_single = timestep.env_cfg

            if i % config.log_train_interval == 0:
                curriculum_levels = get_curriculum_levels(
                    env_params_single, env.batch_cfg.curriculum_global.levels
                )
                wandb.log(
                    {
                        "performance/steps_per_second": steps_per_second,
                        "performance/iterations_per_second": iterations_per_second,
                        "curriculum_levels": curriculum_levels,
                        "lr": config.lr,
                        **loss_info_single,
                    }
                )

            if i % config.checkpoint_interval == 0:
                checkpoint = {
                    "train_config": config,
                    "env_config": env_params_single,
                    "model": runner_state_single[1].params,
                    "loss_info": loss_info_single,
                }
                helpers.save_pkl_object(checkpoint, f"checkpoints/{config.name}.pkl")

            if i % config.log_eval_interval == 0:
                eval_stats = eval_ppo.rollout(
                    rng_rollout,
                    env,
                    env_params_single,
                    train_state,
                    prev_actions,
                    config,
                )

                # eval_stats = jax.lax.pmean(eval_stats, axis_name="devices")
                n = config.num_envs_per_device * eval_stats.length
                avg_positive_episode_length = jnp.where(
                    eval_stats.positive_terminations > 0,
                    eval_stats.positive_terminations_steps / eval_stats.positive_terminations,
                    jnp.zeros_like(eval_stats.positive_terminations_steps)
                )
                loss_info_single.update(
                    {
                        "eval/rewards": eval_stats.reward / n,
                        "eval/max_reward": eval_stats.max_reward,
                        "eval/min_reward": eval_stats.min_reward,
                        "eval/lengths": eval_stats.length,
                        "eval/FORWARD %": eval_stats.action_0 / n,
                        "eval/BACKWARD %": eval_stats.action_1 / n,
                        "eval/CLOCK %": eval_stats.action_2 / n,
                        "eval/ANTICLOCK %": eval_stats.action_3 / n,
                        "eval/CABIN_CLOCK %": eval_stats.action_4 / n,
                        "eval/CABIN_ANTICLOCK %": eval_stats.action_5 / n,
                        "eval/DO": eval_stats.action_6 / n,
                        "eval/positive_terminations": eval_stats.positive_terminations
                        / config.num_envs_per_device,
                        "eval/total_terminations": eval_stats.terminations
                        / config.num_envs_per_device,
                        "eval/avg_positive_episode_length": avg_positive_episode_length
                    }
                )

                wandb.log(loss_info_single)
        return {"runner_state": runner_state_single, "loss_info": loss_info_single}

    return train


def train(config: TrainConfig):
    run = wandb.init(
        entity="terra-sp-thesis",
        project=config.project,
        group=config.group,
        name=config.name,
        config=asdict(config),
        save_code=True,
    )
    
    # Log config.py and train.py files to wandb
    train_py_path = os.path.abspath(__file__)  # Path to current train.py file
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "terra", "terra", "config.py")
    
    code_artifact = wandb.Artifact(name="source_code", type="code")
    
    # Add train.py
    if os.path.exists(train_py_path):
        code_artifact.add_file(train_py_path, name="train.py")
    
    # Add config.py
    if os.path.exists(config_path):
        code_artifact.add_file(config_path, name="config.py")
    
    # Log the artifact if any files were added
    if code_artifact.files:
        run.log_artifact(code_artifact)

    rng, env, env_params, train_state = make_states(config)

    train_fn = make_train(env, env_params, config)

    print("Training...")
    try:  # Try block starts here
        t = time.time()
        train_info = jax.block_until_ready(train_fn(rng, train_state))
        elapsed_time = time.time() - t
        print(f"Done in {elapsed_time:.2f}s")
    except KeyboardInterrupt:  # Catch Ctrl+C
        print("Training interrupted. Finalizing...")
    finally:  # Ensure wandb.finish() is called
        run.finish()
        print("wandb session finished.")


if __name__ == "__main__":
    DT = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="experiment",
    )
    parser.add_argument(
        "-m",
        "--machine",
        type=str,
        default="local",
    )
    parser.add_argument(
        "-d",
        "--num_devices",
        type=int,
        default=0,
        help="Number of devices to use. If 0, uses all available devices.",
    )
    args, _ = parser.parse_known_args()

    name = f"{args.name}-{args.machine}-{DT}"
    train(TrainConfig(name=name, num_devices=args.num_devices))
