import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from terra.env import TerraEnvBatch, TimeStep
from terra.state import State


class FakeAgentState(NamedTuple):
    action_type: jax.Array


class FakeAction(NamedTuple):
    action: jax.Array


class DispatchState(NamedTuple):
    action_type: jax.Array
    marker: jax.Array
    current_agent: jax.Array
    env_steps: jax.Array

    def _get_current_agent_state(self):
        return FakeAgentState(self.action_type[None])

    def _mark(self, value):
        return self._replace(marker=jnp.int32(value))

    def _handle_move_forward(self):
        return self._mark(10)

    def _handle_move_backward(self):
        return self._mark(11)

    def _handle_clock(self):
        return self._mark(12)

    def _handle_anticlock(self):
        return self._mark(13)

    def _handle_cabin_clock(self):
        return self._mark(14)

    def _handle_cabin_anticlock(self):
        return self._mark(15)

    def _handle_do(self):
        return self._mark(16)

    def _handle_move_forward_wheeled(self):
        return self._mark(20)

    def _handle_move_backward_wheeled(self):
        return self._mark(21)

    def _handle_turn_wheels_left(self):
        return self._mark(22)

    def _handle_turn_wheels_right(self):
        return self._mark(23)

    def _do_nothing(self):
        return self

    def _swap(self):
        return self._replace(current_agent=1 - self.current_agent)


def reference_step(state, action, turn=True):
    handlers = [
        state._handle_move_forward,
        state._handle_move_backward,
        state._handle_clock,
        state._handle_anticlock,
        state._handle_cabin_clock,
        state._handle_cabin_anticlock,
        state._handle_do,
        state._do_nothing,
        state._handle_move_forward_wheeled,
        state._handle_move_backward_wheeled,
        state._handle_turn_wheels_left,
        state._handle_turn_wheels_right,
        state._handle_cabin_clock,
        state._handle_cabin_anticlock,
        state._handle_do,
        state._do_nothing,
    ]
    offset = jnp.array([0, 8], dtype=jnp.int32) @ jax.nn.one_hot(
        state.action_type, 2, dtype=jnp.int32
    )
    action_idx = jnp.squeeze(action.action)
    result = jax.lax.cond(
        jnp.logical_or(action_idx == -1, action_idx == 7),
        state._do_nothing,
        lambda: jax.lax.switch(offset + action_idx, handlers),
    )
    result = jax.lax.cond(turn, result._swap, lambda: result)
    return result._replace(env_steps=result.env_steps + 1)


class FakeEnvState(NamedTuple):
    value: jax.Array

    def _get_infos(self, action, task_done):
        return {"task_done": task_done}


class FakeTerraEnv:
    @staticmethod
    def step_no_reset(state, action, env_cfg):
        next_state = FakeEnvState(state.value + action + 1)
        done = action == 1
        info = next_state._get_infos(action, done)
        info = {
            **info,
            "reward_components": {"terminal": done.astype(jnp.float32)},
        }
        return TimeStep(
            state=next_state,
            observation={"value": next_state.value},
            reward=next_state.value.astype(jnp.float32),
            done=done,
            info=info,
            env_cfg=env_cfg,
        )

    @staticmethod
    def _reset_existent(
        state,
        target_map,
        padding_mask,
        trench_axis,
        trench_kind,
        foundation_border_axis,
        foundation_border_kind,
        dumpability_mask,
        action_map,
        distance_map,
        env_cfg,
    ):
        del (
            state,
            padding_mask,
            trench_axis,
            trench_kind,
            foundation_border_axis,
            foundation_border_kind,
            dumpability_mask,
            action_map,
            distance_map,
            env_cfg,
        )
        reset_state = FakeEnvState(target_map)
        return reset_state, {"value": reset_state.value}

    def step(
        self,
        state,
        action,
        target_map,
        padding_mask,
        trench_axis,
        trench_kind,
        foundation_border_axis,
        foundation_border_kind,
        dumpability_mask,
        action_map,
        distance_map,
        env_cfg,
    ):
        timestep = self.step_no_reset(state, action, env_cfg)

        def reset(item):
            reset_state, reset_obs = self._reset_existent(
                item.state,
                target_map,
                padding_mask,
                trench_axis,
                trench_kind,
                foundation_border_axis,
                foundation_border_kind,
                dumpability_mask,
                action_map,
                distance_map,
                item.env_cfg,
            )
            info = reset_state._get_infos(action, item.info["task_done"])
            info = {**info, "reward_components": item.info["reward_components"]}
            return item._replace(state=reset_state, observation=reset_obs, info=info)

        return jax.lax.cond(timestep.done, reset, lambda item: item, timestep)


class FakeCurriculum:
    @staticmethod
    def update_cfgs(timestep, keys):
        del keys
        return timestep


class FakeBatch:
    terra_env = FakeTerraEnv()
    curriculum_manager = FakeCurriculum()

    @staticmethod
    def _get_map(keys, env_cfg):
        del keys
        target = jnp.arange(env_cfg.shape[0], dtype=jnp.int32) + 100
        zeros = jnp.zeros_like(target)
        return (
            target,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.zeros((env_cfg.shape[0], 2), dtype=jnp.uint32),
        )


class StepOptimizationTest(unittest.TestCase):
    def test_dispatch_matches_sixteen_branch_reference(self):
        for action_type in (0, 1):
            for action_idx in range(-1, 8):
                for turn in (False, True):
                    state = DispatchState(
                        action_type=jnp.int32(action_type),
                        marker=jnp.int32(-1),
                        current_agent=jnp.int32(0),
                        env_steps=jnp.int32(9),
                    )
                    action = FakeAction(jnp.array([action_idx], dtype=jnp.int32))
                    expected = reference_step(state, action, turn=turn)
                    actual = State._step(state, action, turn=turn)
                    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_conditional_reset_matches_reference(self):
        batch = FakeBatch()
        keys = jax.random.split(jax.random.PRNGKey(0), 3)
        initial = TimeStep(
            state=FakeEnvState(jnp.array([10, 20, 30], dtype=jnp.int32)),
            observation={"value": jnp.array([10, 20, 30], dtype=jnp.int32)},
            reward=jnp.zeros((3,), dtype=jnp.float32),
            done=jnp.zeros((3,), dtype=jnp.bool_),
            info={
                "task_done": jnp.zeros((3,), dtype=jnp.bool_),
                "reward_components": {
                    "terminal": jnp.zeros((3,), dtype=jnp.float32)
                },
            },
            env_cfg=jnp.array([1, 1, 1], dtype=jnp.int32),
        )

        for actions in (
            jnp.array([0, 0, 0], dtype=jnp.int32),
            jnp.array([0, 1, 0], dtype=jnp.int32),
            jnp.array([1, 1, 1], dtype=jnp.int32),
        ):
            expected = TerraEnvBatch.step_unconditional_reset_candidates(
                batch, initial, actions, keys
            )
            actual = TerraEnvBatch.step(batch, initial, actions, keys)
            expected_leaves = jax.tree_util.tree_leaves(expected)
            actual_leaves = jax.tree_util.tree_leaves(actual)
            self.assertEqual(len(actual_leaves), len(expected_leaves))
            for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
                np.testing.assert_array_equal(
                    np.asarray(actual_leaf), np.asarray(expected_leaf)
                )


if __name__ == "__main__":
    unittest.main()
