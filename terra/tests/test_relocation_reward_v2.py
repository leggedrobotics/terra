"""reward-v2: re-digging one's own spoil must pay less than a fresh dig.

The reward-v1 defect (found by the M1-B readout): `_compute_potential_multiplier`
gated the excavator's relocate-dumped discount on `has_transport_agent`, so in a
single-excavator setup — `agent_types: [0]`, i.e. every M1 arm — the discount was
unreachable and a dig -> dump -> re-dig cycle paid the same rate as fresh work.
Combined with the flat `+1.0` `started_loading` dig reward and a
`dig_on_dump_penalty` that only fires on the DESIGNATED dump zone, spoil staged
on legal non-designated ground could be re-handled indefinitely for full reward.

These tests fix the behaviour at three levels:

* the multiplier itself, with no transport agent present (the exact regression);
* the dump reward, on two otherwise byte-identical scripted dumps that differ
  only in whether the load came from previously dumped material;
* the flag's reachability — `_handle_dig` must actually set
  `moving_dumped_dirt` when a single excavator lifts a staged pile, otherwise
  the fix is dead code.

Everything is deterministic: one fixed PRNG key, hand-written maps, no sampling.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.config import BatchConfig
from terra.config import EnvConfig
from terra.config import MapsDimsConfig
from terra.config import check_relocation_multipliers
from terra.env import TerraEnvBatch
from terra.state import State

SEED = 20260730
SHAPE = (64, 64)


def _env_config(*, dumped_mult: float = 0.2, dug_mult: float = 1.5) -> EnvConfig:
    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=64)
    )
    base = EnvConfig()
    batched = base._replace(
        agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32))
    )
    updated = batch_env.update_env_cfgs(batched)
    return base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(
            edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0]),
        ),
        agent_types=(0,),          # single excavator: no transport agent exists
        action_types=(0,),
        foundation_dump_min_free_fraction=0.0,
        excavator_relocate_dumped_mult=dumped_mult,
        excavator_relocate_dug_dirt_mult=dug_mult,
    )


def _state(
    target: np.ndarray,
    *,
    action: np.ndarray | None = None,
    env_cfg: EnvConfig | None = None,
    loaded: int = 0,
    angle_cabin: int = 0,
) -> State:
    if action is None:
        action = np.zeros(SHAPE, dtype=np.int8)
    padding = np.zeros(SHAPE, dtype=np.int8)
    dumpability = np.ones(SHAPE, dtype=np.bool_)
    state = State.new(
        jax.random.PRNGKey(SEED),
        env_cfg if env_cfg is not None else _env_config(),
        target,
        padding,
        -97.0 * np.ones((3, 3), dtype=np.float32),
        np.int32(-1),
        -97.0 * np.ones((64, 3), dtype=np.float32),
        np.int32(-1),
        dumpability,
        action,
        distance_map_override=np.ones(SHAPE, dtype=np.float32),
    )
    current = state._get_current_agent_state()._replace(
        pos_base=jnp.array([32, 32], dtype=jnp.int16),
        angle_base=jnp.array([0], dtype=jnp.int8),
        angle_cabin=jnp.array([angle_cabin], dtype=jnp.int8),
        loaded=jnp.array([loaded], dtype=jnp.int8),
    )
    return state._set_current_agent_state(current)


def _workspace(angle_cabin: int = 0) -> np.ndarray:
    """Cells the excavator can dig/dump from (32, 32) at this cabin angle."""
    empty = np.zeros(SHAPE, dtype=np.int8)
    state = _state(empty, angle_cabin=angle_cabin)
    cone = np.asarray(state._build_dig_dump_cone()).reshape(SHAPE)
    coordinates = np.argwhere(cone)
    if coordinates.size == 0:
        raise AssertionError(f"empty workspace at cabin angle {angle_cabin}")
    return coordinates


class RelocationMultiplierTest(unittest.TestCase):
    """The regression itself: the discount with no transport agent on site."""

    def test_redig_discount_applies_without_a_transport_agent(self):
        state = _state(np.zeros(SHAPE, dtype=np.int8))
        self.assertFalse(bool(state._has_transport_agent()))

        fresh = float(
            state._compute_potential_multiplier(
                jnp.bool_(False), jnp.bool_(False)
            )
        )
        redig = float(
            state._compute_potential_multiplier(
                jnp.bool_(False), jnp.bool_(True)
            )
        )
        self.assertAlmostEqual(fresh, 1.5, places=5)
        self.assertAlmostEqual(redig, 0.2, places=5)
        self.assertLess(redig, fresh)

    def test_transport_agents_keep_their_own_multiplier(self):
        state = _state(np.zeros(SHAPE, dtype=np.int8))
        for moving_dumped in (False, True):
            value = float(
                state._compute_potential_multiplier(
                    jnp.bool_(True), jnp.bool_(moving_dumped)
                )
            )
            self.assertAlmostEqual(value, 1.5, places=5)

    def test_guard_flags_a_config_that_disables_the_discount(self):
        self.assertEqual(check_relocation_multipliers(_env_config()), "")
        warning = check_relocation_multipliers(
            _env_config(dumped_mult=1.5, dug_mult=1.5)
        )
        self.assertIn("re-dig discount is disabled", warning)


class ScriptedDumpRewardTest(unittest.TestCase):
    """Same scripted dump, twice: fresh load vs re-dug spoil."""

    @staticmethod
    def _scripted_dump(*, moving_dumped_dirt: bool, env_cfg: EnvConfig):
        """dig -> (mark the load's provenance) -> dump, and the dump reward."""
        cells = _workspace()
        dig_cells = cells[: len(cells) // 3]
        dump_cells = cells[len(cells) // 3 : 2 * len(cells) // 3]

        target = np.zeros(SHAPE, dtype=np.int8)
        for y, x in dig_cells:
            target[y, x] = -1
        for y, x in dump_cells:
            target[y, x] = 1

        before_dig = _state(target, env_cfg=env_cfg)
        after_dig = before_dig._handle_dig()
        loaded = int(np.asarray(after_dig._get_current_agent_state().loaded)[0])
        # provenance is the ONLY difference between the two runs
        carrying = after_dig._replace(
            agent=after_dig.agent._replace(
                moving_dumped_dirt=jnp.bool_(moving_dumped_dirt)
            )
        )
        after_dump = carrying._handle_dump()
        reward = float(carrying._handle_rewards_dump(after_dump, None))
        return reward, loaded

    def test_redig_dump_pays_strictly_less_than_a_fresh_dump(self):
        cfg = _env_config(dumped_mult=0.2, dug_mult=1.5)
        fresh, fresh_loaded = self._scripted_dump(
            moving_dumped_dirt=False, env_cfg=cfg
        )
        redig, redig_loaded = self._scripted_dump(
            moving_dumped_dirt=True, env_cfg=cfg
        )

        self.assertGreater(fresh_loaded, 0, "the scripted dig must load dirt")
        self.assertEqual(fresh_loaded, redig_loaded, "loads must be identical")
        self.assertLess(
            redig,
            fresh,
            f"re-dug spoil paid {redig} vs {fresh} for a fresh load; the "
            "reward-v1 defect is back",
        )

    def test_reward_v1_configuration_shows_no_discount(self):
        """Characterisation: 1.5 / 1.5 (the M1 presets) is a no-op discount.

        This is what reward-v1 did on EVERY dump, and it is why the code fix
        alone is not enough for a preset that sets the two multipliers equal.
        """
        cfg = _env_config(dumped_mult=1.5, dug_mult=1.5)
        fresh, _ = self._scripted_dump(moving_dumped_dirt=False, env_cfg=cfg)
        redig, _ = self._scripted_dump(moving_dumped_dirt=True, env_cfg=cfg)
        self.assertAlmostEqual(fresh, redig, places=4)


class MovingDumpedDirtFlagTest(unittest.TestCase):
    """The flag must be reachable for a lone excavator, or the fix is dead code."""

    def test_lifting_a_staged_pile_sets_moving_dumped_dirt(self):
        cells = _workspace()
        pile_cells = cells[: len(cells) // 4]

        target = np.zeros(SHAPE, dtype=np.int8)
        # the pile sits on NON-designated ground (target == 0), which spec §8.1
        # makes legal to stage on — this is the loophole reward-v1 paid for
        action = np.zeros(SHAPE, dtype=np.int8)
        for y, x in pile_cells:
            action[y, x] = 2

        state = _state(target, action=action)
        self.assertFalse(bool(state.agent.moving_dumped_dirt))
        after = state._handle_dig()
        self.assertTrue(
            bool(after.agent.moving_dumped_dirt),
            "a lone excavator re-digging staged spoil must be flagged",
        )

    def test_fresh_ground_does_not_set_the_flag(self):
        cells = _workspace()
        target = np.zeros(SHAPE, dtype=np.int8)
        for y, x in cells[: len(cells) // 4]:
            target[y, x] = -1
        state = _state(target)
        after = state._handle_dig()
        self.assertFalse(bool(after.agent.moving_dumped_dirt))


if __name__ == "__main__":
    unittest.main()
