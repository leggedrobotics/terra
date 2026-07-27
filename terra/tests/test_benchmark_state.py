import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.agent import Agent
from terra.agent import AgentState
from terra.benchmark_state import agent_from_record
from terra.benchmark_state import agent_state_sha256
from terra.benchmark_state import agent_to_record
from terra.benchmark_state import validate_benchmark_initial_agent
from terra.config import EnvConfig
from terra.env import TerraEnv
from terra.env import TerraEnvBatch


class BenchmarkAgentStateTest(unittest.TestCase):
    SHAPE = (64, 64)

    @staticmethod
    def _env_config() -> EnvConfig:
        base = EnvConfig()
        return base._replace(
            tile_size=np.float32(36.5714285714 / 64),
            agent=base.agent._replace(width=7, height=11),
            maps=base.maps._replace(edge_length_px=64),
            agent_types=(0,),
            action_types=(0,),
        )

    @staticmethod
    def _state_slot(
        index: int,
        *,
        pos_base: tuple[int, int],
        canonical: bool = False,
    ) -> AgentState:
        value = 0 if canonical else index + 1
        carry_baseline = 0.0 if canonical else value + 0.25
        carry_after_lift = 0.0 if canonical else value + 0.75
        return AgentState(
            pos_base=jnp.asarray(pos_base, dtype=jnp.int16),
            angle_base=jnp.asarray([value % 12], dtype=jnp.int8),
            angle_cabin=jnp.asarray(
                [0 if canonical else (value + 2) % 12], dtype=jnp.int8
            ),
            wheel_angle=jnp.asarray([value % 2], dtype=jnp.int8),
            loaded=jnp.asarray([value], dtype=jnp.int8),
            agent_type=jnp.asarray([value % 3], dtype=jnp.int8),
            action_type=jnp.asarray([value % 2], dtype=jnp.int8),
            shovel_lifted=jnp.asarray([value % 2], dtype=jnp.int8),
            carry_baseline_potential=jnp.asarray(carry_baseline, dtype=jnp.float32),
            carry_potential_after_lift=jnp.asarray(carry_after_lift, dtype=jnp.float32),
        )

    @classmethod
    def _noncanonical_agent(cls) -> Agent:
        return Agent(
            width=jnp.asarray(7, dtype=jnp.int32),
            height=jnp.asarray(11, dtype=jnp.int32),
            moving_dumped_dirt=jnp.asarray(True),
            agent_states=(
                cls._state_slot(0, pos_base=(30, 30)),
                cls._state_slot(1, pos_base=(18, 18)),
                cls._state_slot(2, pos_base=(4, 5)),
                cls._state_slot(3, pos_base=(6, 7)),
            ),
            agent_active=jnp.asarray([1, 1, 0, 0], dtype=jnp.int8),
            num_agents=jnp.asarray(2, dtype=jnp.int32),
            current_agent=jnp.asarray(1, dtype=jnp.int32),
        )

    @classmethod
    def _canonical_agent(cls) -> Agent:
        zero = cls._state_slot(0, pos_base=(0, 0), canonical=True)
        active = zero._replace(pos_base=jnp.asarray([32, 32], dtype=jnp.int16))
        return Agent(
            width=jnp.asarray(7, dtype=jnp.int32),
            height=jnp.asarray(11, dtype=jnp.int32),
            moving_dumped_dirt=jnp.asarray(False),
            agent_states=(active, zero, zero, zero),
            agent_active=jnp.asarray([1, 0, 0, 0], dtype=jnp.int8),
            num_agents=jnp.asarray(1, dtype=jnp.int32),
            current_agent=jnp.asarray(0, dtype=jnp.int32),
        )

    @classmethod
    def _map_arguments(cls):
        return (
            jnp.zeros(cls.SHAPE, dtype=jnp.int8),
            jnp.zeros(cls.SHAPE, dtype=jnp.int8),
            -97.0 * jnp.ones((3, 3), dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            -97.0 * jnp.ones((64, 3), dtype=jnp.float32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.ones(cls.SHAPE, dtype=jnp.bool_),
            jnp.zeros(cls.SHAPE, dtype=jnp.int8),
            jnp.ones(cls.SHAPE, dtype=jnp.float32),
        )

    def _assert_agents_equal(self, actual: Agent, expected: Agent) -> None:
        actual_leaves = jax.tree_util.tree_leaves(actual)
        expected_leaves = jax.tree_util.tree_leaves(expected)
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
            actual_array = np.asarray(actual_leaf)
            expected_array = np.asarray(expected_leaf)
            self.assertEqual(actual_array.dtype, expected_array.dtype)
            np.testing.assert_array_equal(actual_array, expected_array)

    def test_explicit_agent_reset_round_trips_eager_jit_and_batched_vmap(self):
        env = TerraEnv.new(maps_size_px=64)
        env_cfg = self._env_config()
        map_arguments = self._map_arguments()
        initial_agent = self._noncanonical_agent()

        with jax.disable_jit():
            eager = env.reset(
                jax.random.PRNGKey(1),
                *map_arguments,
                env_cfg,
                initial_agent,
            )
        compiled = env.reset(
            jax.random.PRNGKey(2),
            *map_arguments,
            env_cfg,
            initial_agent,
        )
        self._assert_agents_equal(eager.state.agent, initial_agent)
        self._assert_agents_equal(compiled.state.agent, initial_agent)

        batch_env = object.__new__(TerraEnvBatch)
        batch_env.terra_env = env

        def stack_twice(value):
            array = jnp.asarray(value)
            return jnp.stack((array, array))

        env_cfgs = jax.tree_util.tree_map(stack_twice, env_cfg)
        batched_maps = tuple(stack_twice(value) for value in map_arguments)
        initial_agents = jax.tree_util.tree_map(stack_twice, initial_agent)
        batched = batch_env.reset_prepared(
            env_cfgs,
            jax.random.split(jax.random.PRNGKey(3), 2),
            *batched_maps,
            initial_agents,
        )
        for index in range(2):
            selected = jax.tree_util.tree_map(
                lambda value: value[index], batched.state.agent
            )
            self._assert_agents_equal(selected, initial_agent)

    def test_codec_round_trip_golden_hash_and_inactive_mutation(self):
        agent = self._canonical_agent()
        record = agent_to_record(agent)
        decoded = agent_from_record(record)
        self._assert_agents_equal(decoded, agent)

        self.assertEqual(
            agent_state_sha256(agent),
            "debd22b6ff2c8b31d263ceb843e524d5bf9ae1ffe186e26291f1e5ec3d18fb1a",
        )

        mutated_slot = agent.agent_states[3]._replace(
            carry_potential_after_lift=jnp.asarray(1.0, dtype=jnp.float32)
        )
        mutated = agent._replace(agent_states=agent.agent_states[:3] + (mutated_slot,))
        self.assertNotEqual(
            agent_state_sha256(mutated),
            agent_state_sha256(agent),
        )

    def test_codec_rejects_malformed_records(self):
        record = agent_to_record(self._canonical_agent())

        missing = copy.deepcopy(record)
        del missing["agent_states"]["loaded"]
        with self.assertRaisesRegex(ValueError, "missing=.*loaded"):
            agent_from_record(missing)

        wrong_shape = copy.deepcopy(record)
        wrong_shape["agent_states"]["pos_base"][0] = [32]
        with self.assertRaisesRegex(ValueError, "pos_base must have shape"):
            agent_from_record(wrong_shape)

        overflow = copy.deepcopy(record)
        overflow["agent_states"]["angle_base"][0] = 128
        with self.assertRaisesRegex(ValueError, "outside int8"):
            agent_from_record(overflow)

    def test_benchmark_admissibility_uses_canonical_slots_and_live_footprint(self):
        agent = self._canonical_agent()
        env_cfg = self._env_config()
        padding = np.zeros(self.SHAPE, dtype=np.int8)
        actions = np.zeros(self.SHAPE, dtype=np.int8)
        dumpability = np.ones(self.SHAPE, dtype=np.bool_)
        validate_benchmark_initial_agent(
            agent,
            env_cfg=env_cfg,
            padding_mask=padding,
            action_map=actions,
            dumpability_mask=dumpability,
        )

        blocked = padding.copy()
        blocked[32, 32] = 1
        with self.assertRaisesRegex(ValueError, "overlaps an obstacle"):
            validate_benchmark_initial_agent(
                agent,
                env_cfg=env_cfg,
                padding_mask=blocked,
                action_map=actions,
                dumpability_mask=dumpability,
            )

        noncanonical_slot = agent.agent_states[3]._replace(
            pos_base=jnp.asarray([1, 0], dtype=jnp.int16)
        )
        with self.assertRaisesRegex(ValueError, "canonical zero pos_base"):
            validate_benchmark_initial_agent(
                agent._replace(
                    agent_states=agent.agent_states[:3] + (noncanonical_slot,)
                ),
                env_cfg=env_cfg,
                padding_mask=padding,
                action_map=actions,
                dumpability_mask=dumpability,
            )


if __name__ == "__main__":
    unittest.main()
