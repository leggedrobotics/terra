"""An opt-in, single-world manual session backed by Terra's real transitions."""

from pathlib import Path

import numpy as np

from .replay import ReplayRecorder


def demo_maps():
    """A small, partially excavated site; all heights are Terra soil units."""
    shape = (64, 64)
    target = np.zeros(shape, dtype=np.int8)
    target[23:42, 29:33] = -1
    target[23:27, 33:44] = -1
    target[22:43, 8:14] = 1
    action = np.zeros(shape, dtype=np.int8)
    action[23:27, 29:33] = -1
    action[24:28, 9:13] = 1
    padding = np.zeros(shape, dtype=np.int8)
    padding[7:15, 47:56] = 1
    padding[47:55, 7:17] = 1
    dumpability = np.ones(shape, dtype=bool)
    dumpability[49:55, 21:60] = False
    dumpability[padding != 0] = False
    rows, cols = np.indices(shape)
    distance = (
        np.maximum(np.maximum(22 - rows, rows - 42), 0)
        + np.maximum(np.maximum(8 - cols, cols - 13), 0)
    ).astype(np.float32) / 24.0
    return (
        target,
        padding,
        np.full((3, 3), -97, dtype=np.float32),
        np.int32(-1),
        np.full((64, 3), -97, dtype=np.float32),
        np.int32(-1),
        dumpability,
        action,
        distance,
    )


class ManualSession:
    """Keep terminal states until reset and record every accepted action.

    JAX imports live here so viewing an existing recording does not load JAX.
    The HTTP service serializes calls to this object.
    """

    def __init__(
        self,
        *,
        map_path=None,
        seed=0,
        max_steps=400,
        agent_types=(0,),
        action_types=None,
        warmup=True,
    ):
        import jax
        import jax.numpy as jnp

        from terra.config import BatchConfig, EnvConfig
        from terra.env import TerraEnv

        if not 1 <= len(agent_types) <= 4 or any(
            t not in (0, 1, 2) for t in agent_types
        ):
            raise ValueError(
                "Choose one to four agent types: 0 excavator, 1 truck, 2 skid steer."
            )
        if action_types is None:
            action_types = tuple(1 if t == 1 else 0 for t in agent_types)
        if len(action_types) != len(agent_types) or any(
            t not in (0, 1) for t in action_types
        ):
            raise ValueError(
                "Provide one action type per agent: 0 tracked or 1 wheeled."
            )
        if not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be between 1 and 10000.")
        self.seed = seed
        self.max_steps = max_steps
        self.map_path = Path(map_path).resolve() if map_path else None
        self.env = TerraEnv()
        if self.map_path:
            from terra.maps_buffer import load_single_map

            loaded = load_single_map(str(self.map_path))
            # load_single_map batches all fields except foundation_border_type.
            self.maps = tuple(
                value if i == 5 else value[0] for i, value in enumerate(loaded)
            )
        else:
            self.maps = tuple(jnp.asarray(value) for value in demo_maps())
        target = np.asarray(self.maps[0])
        if (
            target.ndim != 2
            or target.shape[0] != target.shape[1]
            or not 32 <= target.shape[0] <= 128
        ):
            raise ValueError(
                "Manual Terra maps must be square, between 32 and 128 cells per side."
            )
        for index in (1, 6, 7, 8):
            if np.asarray(self.maps[index]).shape != target.shape:
                raise ValueError(
                    "Map, occupancy, dumpability, actions, and distance shapes must match."
                )
        if not np.isfinite(np.asarray(self.maps[8])).all():
            raise ValueError("Distance map contains non-finite values.")
        if np.any(target > 0) and not np.any(np.asarray(self.maps[8]) > 0):
            raise ValueError(
                "A designated dump region requires a nonzero distance map."
            )
        baseline = BatchConfig()
        tile_size = baseline.maps.edge_length_m / target.shape[0]

        def odd_tiles(meters):
            count = round(meters / tile_size)
            return count if count % 2 else count + 1

        cfg = EnvConfig()
        self.config = cfg._replace(
            tile_size=tile_size,
            max_steps_in_episode=max_steps,
            agent_types=tuple(agent_types),
            action_types=tuple(action_types),
            maps=cfg.maps._replace(edge_length_px=target.shape[0]),
            agent=cfg.agent._replace(
                width=odd_tiles(baseline.agent.dimensions.HEIGHT),
                height=odd_tiles(baseline.agent.dimensions.WIDTH),
            ),
        )
        from .spawn import validate_spawn

        # The training reset sampler retries forever if no footprint fits. Prove
        # this exact seed can place every agent before entering that sampler.
        self.spawn_poses = validate_spawn(self.seed, self.config, self.maps)
        self.reset()
        if warmup:
            from terra.actions import TrackedAction

            # Compile without consuming a user action or changing recorded state.
            warmed = self.env.step_no_reset(
                self.timestep.state, TrackedAction.do_nothing(), self.config
            )
            jax.block_until_ready(warmed.reward)

    def reset(self):
        import jax
        import jax.numpy as jnp

        self.timestep = self.env.reset(
            jax.random.PRNGKey(self.seed), *self.maps, self.config
        )
        if self.map_path is None:
            # Known free positions make the example repeatable and put the first
            # excavator beside a dig workspace. Dynamics after reset are unmodified.
            positions = ((32, 20), (13, 24), (44, 42), (14, 39))
            state = self.timestep.state
            agents = list(state.agent.agent_states)
            for i in range(len(self.config.agent_types)):
                agents[i] = agents[i]._replace(
                    pos_base=jnp.asarray(positions[i], dtype=agents[i].pos_base.dtype),
                    angle_base=jnp.zeros_like(agents[i].angle_base),
                    angle_cabin=jnp.zeros_like(agents[i].angle_cabin),
                )
            state = state._replace(
                agent=state.agent._replace(
                    agent_states=tuple(agents), current_agent=jnp.int32(0)
                )
            )
            state = self.env.wrap_state(state)
            self.timestep = self.timestep._replace(
                state=state, observation=self.env._state_to_obs_dict(state)
            )
        self.recorder = ReplayRecorder(
            metadata={
                "title": self.map_path.name if self.map_path else "The earthworks yard",
                "source": (
                    "Terra manual play" if self.map_path else "Terra built-in example"
                ),
                "map": str(self.map_path) if self.map_path else "built-in-yard",
                "seed": self.seed,
                "max_steps": self.max_steps,
                "height_units": "abstract soil units",
            }
        )
        self.recorder.append(self.timestep)
        return self.payload()

    def payload(self):
        return {"mode": "manual", "replay": self.recorder.to_dict()}

    def step(self, action):
        import jax.numpy as jnp

        from terra.actions import TrackedAction

        if (
            isinstance(action, bool)
            or not isinstance(action, int)
            or action not in range(8)
        ):
            raise ValueError("action must be an integer from 0 to 7.")
        if bool(np.asarray(self.timestep.done)):
            raise RuntimeError(
                "This episode has ended. Reset to start another episode."
            )
        actor_id = int(np.asarray(self.timestep.state.agent.current_agent))
        wrapped_action = TrackedAction.new(jnp.array([action], dtype=jnp.int8))
        self.timestep = self.env.step_no_reset(
            self.timestep.state, wrapped_action, self.config
        )
        return self.recorder.append(self.timestep, action=action, actor_id=actor_id)

    def play_demo(self):
        """A recorded dig, slew, dump, and relocation through actual Terra steps."""
        if self.map_path or tuple(self.config.agent_types) != (0,):
            raise ValueError(
                "The scripted example requires the built-in single excavator scene."
            )
        for action in [6, *([4] * 6), 6, *([5] * 6), 1, 0, 2, 3]:
            if bool(np.asarray(self.timestep.done)):
                break
            self.step(action)
        self.recorder.metadata["source"] = (
            "Scripted actions in the real Terra environment"
        )
        return self.recorder
