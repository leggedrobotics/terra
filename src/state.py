import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple
from src.config import EnvConfig
from src.map import GridWorld
from src.agent import Agent
from src.actions import Action
from src.utils import Float

class State(NamedTuple):
    """
    Stores the current state of the environment.
    Given perfect information, the observation corresponds to the state.
    """
    seed: jnp.uint32

    env_cfg: EnvConfig

    world: GridWorld
    agent: Agent

    env_steps: int

    @classmethod
    def new(cls, seed: int, env_cfg: EnvConfig) -> "State":
        key = jax.random.PRNGKey(seed)
        world = GridWorld.new(jnp.uint32(seed), env_cfg)

        key, subkey = jax.random.split(key)

        agent = Agent.new(env_cfg)
        agent = jax.tree_map(lambda x: x if isinstance(x, Array) else jnp.array(x), agent)
        # TODO: implement here multiple agents if required (see JUX)

        return State(
            seed=jnp.uint32(seed),
            env_cfg=env_cfg,
            world=world,
            agent=agent,
            env_steps=0
        )

    def _step(self, action: Action) -> "State":
        pass

    def _get_reward(self) -> Float:
        pass

    def _is_done(self) -> jnp.bool_:
        pass
