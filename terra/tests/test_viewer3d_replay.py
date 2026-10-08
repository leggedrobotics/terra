"""CPU-only replay contract tests; no simulator or JAX installation is needed."""

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from terra.viewer3d import (
    ReplayRecorder,
    load_replay,
    snapshot_from_timestep,
    validate_replay,
)
import terra.viewer3d.replay as replay_module


def timestep_fixture():
    """Non-square grid and non-contiguous active slots catch orientation/ID loss."""
    action = np.array(
        [[0, -2, 0, 0, 0], [0, 0, 4, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int16
    )
    world = NS(
        action_map=NS(map=action),
        target_map=NS(
            map=np.array(
                [[0, -2, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]], dtype=np.int8
            )
        ),
        padding_mask=NS(map=np.zeros((3, 5), dtype=bool)),
        dumpability_mask=NS(map=np.ones((3, 5), dtype=bool)),
        dumpability_mask_init=NS(map=np.ones((3, 5), dtype=bool)),
        interaction_mask=NS(map=np.zeros((3, 5), dtype=bool)),
        traversability_mask=NS(map=np.zeros((3, 5), dtype=np.int8)),
    )
    agents = tuple(
        NS(
            pos_base=np.array([1, index], dtype=np.int16),
            angle_base=np.array([index], dtype=np.int8),
            angle_cabin=np.array([index + 1], dtype=np.int8),
            loaded=np.array([index], dtype=np.int16),
            wheel_angle=np.array([-1], dtype=np.int8),
            agent_type=np.array([index % 3], dtype=np.int8),
            action_type=np.array([index % 2], dtype=np.int8),
            shovel_lifted=np.array([0], dtype=np.int8),
        )
        for index in range(4)
    )
    cfg = NS(
        tile_size=0.5, agent=NS(angles_base=12, angles_cabin=8, dig_radius_tiles=5)
    )
    state = NS(
        world=world,
        agent=NS(
            agent_states=agents,
            agent_active=np.array([1, 0, 1, 0], dtype=bool),
            current_agent=np.int32(2),
            width=3,
            height=2,
        ),
        env_steps=np.int32(7),
        env_cfg=cfg,
    )
    return NS(
        state=state,
        env_cfg=cfg,
        reward=np.float32(-0.25),
        done=np.bool_(False),
        info={"task_done": np.bool_(False)},
        observation={"agent_states": np.full((4, 8), 99)},
    )


def batch_fixture(value, shape):
    if isinstance(value, NS):
        return NS(
            **{name: batch_fixture(leaf, shape) for name, leaf in vars(value).items()}
        )
    if isinstance(value, dict):
        return {name: batch_fixture(leaf, shape) for name, leaf in value.items()}
    if isinstance(value, tuple):
        return tuple(batch_fixture(leaf, shape) for leaf in value)
    return np.broadcast_to(value, shape + np.shape(value)).copy()


class SnapshotTest(unittest.TestCase):
    def test_raw_heights_stable_slots_orientation_and_reach(self):
        timestep = timestep_fixture()
        frame = snapshot_from_timestep(timestep, action=6, actor_id=0)
        self.assertEqual(frame["grid"], {"rows": 3, "cols": 5, "tile_size_m": 0.5})
        self.assertEqual(frame["maps"]["action"][0][1], -2)
        self.assertEqual(frame["maps"]["action"][1][2], 4)
        self.assertEqual([agent["id"] for agent in frame["agents"]], [0, 2])
        self.assertEqual(frame["current_agent"], 2)
        self.assertEqual(frame["agents"][1]["position"], [1, 2])
        self.assertAlmostEqual(frame["agents"][1]["base_yaw"], np.pi / 3)
        self.assertAlmostEqual(frame["agents"][1]["cabin_yaw"], 3 * np.pi / 4)
        self.assertEqual(frame["agents"][0]["reach"], [2.5, 7.5])
        self.assertEqual(frame["agents"][1]["reach"], [0.0, 7.0])
        self.assertEqual(frame["agents"][1]["wheel_angle"], -1)
        self.assertIs(frame["done"], False)
        self.assertEqual(json.loads(json.dumps(frame)), frame)
        frame["maps"]["action"][1][2] = 100
        frame["agents"][0]["position"][0] = 2
        self.assertEqual(timestep.state.world.action_map.map[1, 2], 4)
        self.assertEqual(timestep.state.agent.agent_states[0].pos_base[0], 1)

    def test_missing_optional_and_dummy_maps_are_null(self):
        timestep = timestep_fixture()
        del timestep.state.world.dumpability_mask_init
        timestep.state.world.interaction_mask = NS(map=np.zeros((1, 1)))
        timestep.state.world.traversability_mask = None
        frame = snapshot_from_timestep(timestep)
        for name in ("dumpability_static", "interaction", "traversability"):
            self.assertIsNone(frame["maps"][name])
        del timestep.state.world.padding_mask
        with self.assertRaisesRegex(ValueError, "required map"):
            snapshot_from_timestep(timestep)

    def test_explicit_multi_axis_batch_slices_scalars_and_map_together(self):
        batch = batch_fixture(timestep_fixture(), (2, 3))
        batch.reward[1, 2] = 0.75
        batch.done[1, 2] = True
        batch.info["task_done"][1, 2] = True
        batch.state.env_steps[1, 2] = 11
        batch.state.agent.current_agent[1, 2] = 0
        batch.state.agent.agent_states[2].pos_base[1, 2] = [2, 4]
        batch.state.world.action_map.map[1, 2, 1, 2] = 9
        batch.env_cfg.tile_size[1, 2] = 0.25
        # Configurations may contain both batched leaves and shared Python scalars.
        batch.env_cfg.agent.angles_base = 12
        frame = snapshot_from_timestep(batch, env_index=(1, 2))
        self.assertEqual(frame["reward"], 0.75)
        self.assertTrue(frame["done"] and frame["task_done"])
        self.assertEqual(frame["step"], 11)
        self.assertEqual(frame["current_agent"], 0)
        self.assertEqual(frame["agents"][1]["position"], [2, 4])
        self.assertEqual(frame["maps"]["action"][1][2], 9)
        self.assertEqual(frame["agents"][0]["reach"], [3.5, 8.5])
        for index in (None, 1, (1,), (1, 3), (-1, 0), (True, 0)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                snapshot_from_timestep(batch, env_index=index)

    def test_single_batch_dimension_does_not_confuse_agent_slots(self):
        batch = batch_fixture(timestep_fixture(), (4,))
        batch.state.agent.agent_active[3] = [0, 1, 0, 1]
        batch.state.agent.current_agent[3] = 3
        batch.state.agent.agent_states[3].loaded[3, 0] = 12
        frame = snapshot_from_timestep(batch, env_index=3)
        self.assertEqual([agent["id"] for agent in frame["agents"]], [1, 3])
        self.assertEqual(frame["agents"][1]["loaded"], 12)

    def test_device_batch_is_selected_before_host_transfer(self):
        batch = batch_fixture(timestep_fixture(), (2,))
        values = batch.state.world.action_map.map

        class DeviceBatch:
            shape = values.shape

            def __array__(self, *args, **kwargs):
                raise AssertionError("The full device batch must not be transferred")

            def __getitem__(self, index):
                return values[index]

        batch.state.world.action_map.map = DeviceBatch()
        frame = snapshot_from_timestep(batch, env_index=1)
        self.assertEqual(frame["maps"]["action"], values[1].tolist())

    def test_truck_and_skid_steer_reach_match_terra_geometry(self):
        timestep = timestep_fixture()
        timestep.state.agent.agent_active = np.array([1, 1, 1, 0], dtype=bool)
        timestep.state.agent.width = 7
        timestep.state.agent.height = 11
        timestep.env_cfg.tile_size = 4 / 7
        frame = snapshot_from_timestep(timestep)
        self.assertEqual(frame["agents"][0]["reach"], [6.375, 11.375])
        for agent in frame["agents"][1:]:
            self.assertEqual(agent["reach"], [4.0, 11.0])

    def test_replay_import_does_not_import_jax(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import terra.viewer3d; assert 'jax' not in sys.modules",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.recorder = ReplayRecorder(metadata={"title": "Example", "seed": 42})
        self.timestep = timestep_fixture()
        self.recorder.append(self.timestep)

    def test_consecutive_actor_inference_and_detached_recording(self):
        self.timestep.state.env_steps = np.int32(8)
        self.timestep.state.agent.current_agent = np.int32(0)
        frame = self.recorder.append(self.timestep, action=7)
        self.assertEqual(frame["actor_id"], 2)
        self.timestep.state.world.action_map.map[1, 2] = 0
        self.assertEqual(frame["maps"]["action"][1][2], 4)
        exported = self.recorder.to_dict()
        exported["frames"][0]["agents"][0]["loaded"] = 50
        self.assertEqual(self.recorder.frames[0]["agents"][0]["loaded"], 0)
        self.timestep.state.env_steps = np.int32(0)
        self.assertIsNone(self.recorder.append(self.timestep)["actor_id"])

    def test_json_gzip_and_mapping_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("episode.json", "episode.json.gz"):
                with self.subTest(name=name):
                    path = self.recorder.save(Path(directory) / name)
                    self.assertEqual(load_replay(path), self.recorder.to_dict())
        restored = load_replay(self.recorder.to_dict())
        restored["metadata"]["title"] = "Edited"
        self.assertEqual(self.recorder.metadata["title"], "Example")

    def test_offline_html_inlines_assets_and_escapes_script_terminators(self):
        malicious = '</script><script>alert("not executable")</script>'
        self.recorder.metadata["title"] = malicious
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "static").mkdir()
            (root / "static" / "index.html").write_text(
                '<html><head><link rel="stylesheet" href="/static/style.css"></head><body><script defer src="/static/viewer.js"></script></body></html>'
            )
            (root / "static" / "style.css").write_text("body { margin: 0; }")
            (root / "static" / "viewer.js").write_text('const x = "</script>";')
            with patch.object(replay_module, "__file__", str(root / "replay.py")):
                path = self.recorder.save_html(root / "episode.html")
            html = path.read_text()
        self.assertNotIn(malicious, html)
        self.assertNotIn('src="/static/', html)
        self.assertIn("<style>body", html)
        payload = re.search(
            r'<script type="application/json" id="terra-replay">(.*?)</script>', html
        ).group(1)
        self.assertEqual(json.loads(payload)["metadata"]["title"], malicious)

    def test_rejects_malformed_shapes_values_ids_and_schema(self):
        original = self.recorder.to_dict()

        def changed(path, value):
            document = deepcopy(original)
            target = document
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return document

        cases = [
            (("schema",), "terra.viewer3d.v2"),
            (("metadata",), []),
            (("frames",), []),
            (("frames", 0, "reward"), float("nan")),
            (("frames", 0, "action"), True),
            (("frames", 0, "action"), 8),
            (("frames", 0, "actor_id"), 1),
            (("frames", 0, "current_agent"), 1),
            (("frames", 0, "done"), 1),
            (("frames", 0, "task_done"), True),
            (("frames", 0, "grid", "rows"), 129),
            (("frames", 0, "grid", "tile_size_m"), 0),
            (("frames", 0, "maps", "target"), None),
            (("frames", 0, "maps", "action"), [[0]]),
            (("frames", 0, "maps", "action", 0, 0), float("inf")),
            (("frames", 0, "maps", "action", 0, 0), 0.5),
            (("frames", 0, "maps", "padding", 0, 0), 3),
            (("frames", 0, "agents", 1, "id"), 0),
            (("frames", 0, "agents", 0, "position"), [3, 1]),
            (("frames", 0, "agents", 0, "base_yaw"), float("inf")),
            (("frames", 0, "agents", 0, "loaded"), -1),
            (("frames", 0, "agents", 0, "reach"), [5, 3]),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                validate_replay(changed(path, value))
        del original["frames"][0]["current_agent"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_replay(original)


if __name__ == "__main__":
    unittest.main()
