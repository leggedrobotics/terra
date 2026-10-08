"""Tracked moves that keep a turn option instead of shuttling between dead ends."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from terra.actions import TrackedAction
from terra.config import EnvConfig
from terra.dig_direction import boundary_records_from_mask
from terra.state import State

FORWARD, BACKWARD, CLOCK, ANTICLOCK = 0, 1, 2, 3
SHAPE = (64, 64)


def machine_state(action_map, keeps_turn):
    """Machine-rules excavator (centred 7x11 chassis, 0.57 m hole clearance)."""
    cfg = EnvConfig()._replace(
        tile_size=40 / 70,
        agent=EnvConfig().agent._replace(
            width=7, height=11, dig_min_radius_m=4.0, dump_max_radius_m=6.0,
            dug_clearance_m=0.57, centre_chassis_on_base=True),
        maps=EnvConfig().maps._replace(edge_length_px=64),
        agent_types=(0,), action_types=(0,),
        pull_direction_alignment=True,
        tracked_move_keeps_turn=keeps_turn,
    )
    target = np.zeros(SHAPE, np.int8)
    target[30:48, 20:34] = -1
    boundaries, count = boundary_records_from_mask(target < 0)
    return State.new(
        jax.random.PRNGKey(3), cfg, target, np.zeros(SHAPE, np.int8),
        np.full((8, 8), -97, np.float32), np.int32(0), boundaries, np.int32(count),
        np.ones(SHAPE, bool), action_map, distance_map_override=np.ones(SHAPE, np.float32),
    )


@partial(jax.jit, static_argnums=(2,))
def _successors(state, pose, cfg):
    """Native (row, col, heading) after FORWARD, BACKWARD, CLOCK, ANTICLOCK."""
    cur = state._replace(env_cfg=cfg)._get_current_agent_state()._replace(
        pos_base=pose[:2].astype(jnp.int16), angle_base=pose[2:].astype(jnp.int8),
        loaded=jnp.zeros((1,), jnp.int8))
    at = state._replace(env_cfg=cfg)._set_current_agent_state(cur)
    out = []
    for action in range(4):
        new = at._step(TrackedAction.new(jnp.asarray([action], jnp.int8)))._get_current_agent_state()
        out.append(jnp.concatenate([new.pos_base.astype(jnp.int32), new.angle_base.astype(jnp.int32)]))
    return jnp.stack(out)


def successors(state, pose):
    # Python-scalar config is static; a traced config would promote angle dtypes.
    return [tuple(int(v) for v in row) for row in np.asarray(
        _successors(state._replace(env_cfg=None), jnp.asarray(pose, jnp.int32), state.env_cfg))]


class TrackedMoveKeepsTurnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Holes just in front of a chassis facing -row near the bottom edge,
        # the pose left after cutting the near side of a pit from outside.
        trap = np.zeros(SHAPE, np.int8)
        trap[43:48, 25:28] = -1
        cls.trap = {keeps: machine_state(trap, keeps) for keeps in (False, True)}
        lane = np.zeros(SHAPE, np.int8)
        lane[:, 20] = -1
        lane[:, 32] = -1
        cls.lane = {keeps: machine_state(lane, keeps) for keeps in (False, True)}
        cls.open = {keeps: machine_state(np.zeros(SHAPE, np.int8), keeps) for keeps in (False, True)}

    def test_default_move_reproduces_the_shuttle_trap(self):
        state = self.trap[False]
        forward, backward, clock, anticlock = successors(state, (54, 26, 3))
        self.assertEqual((forward, clock, anticlock), ((54, 26, 3),) * 3)
        self.assertEqual(backward, (58, 26, 3))
        forward, backward, clock, anticlock = successors(state, backward)
        self.assertEqual(forward, (54, 26, 3))
        self.assertEqual((backward, clock, anticlock), ((58, 26, 3),) * 3)

    def test_move_stops_where_the_chassis_can_still_turn(self):
        state = self.trap[True]
        back = successors(state, (54, 26, 3))[BACKWARD]
        self.assertEqual(back[1:], (26, 3))
        self.assertTrue(54 < back[0] < 58, back)
        _, edge, clock, anticlock = successors(state, back)
        self.assertNotEqual({clock, anticlock}, {back})
        # The edge stop stays reachable with further moves.
        for _ in range(3):
            edge = successors(state, edge)[BACKWARD]
        self.assertEqual(edge, (58, 26, 3))

    def test_open_ground_and_dead_end_lanes_are_unchanged(self):
        for heading in range(12):
            off = successors(self.open[False], (12, 12, heading))
            on = successors(self.open[True], (12, 12, heading))
            self.assertEqual(off, on, heading)
        # A lane too narrow to turn anywhere keeps the longest clear move.
        off = successors(self.lane[False], (30, 26, 3))
        on = successors(self.lane[True], (30, 26, 3))
        self.assertEqual(on[CLOCK], (30, 26, 3))
        self.assertEqual(off, on)


if __name__ == '__main__':
    unittest.main()
