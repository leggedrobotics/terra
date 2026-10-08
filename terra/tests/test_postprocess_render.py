"""Portable renderer checks without a ROS runtime, Node, or policy inference."""

from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from terra.postprocess.render import player_bundle, write_html
from terra.viewer3d import snapshot_from_timestep, validate_replay
from terra.tests.test_viewer3d_replay import timestep_fixture


def native_joint():
    first = snapshot_from_timestep(timestep_fixture())
    first.update(
        joint_actions=None,
        effective_joint_actions=None,
        workspace_blocked=None,
        workspace_polygons=[
            {"id": 2, "component": "work", "vertices": [[0, 0], [0, 5], [3, 5]]}
        ],
    )
    last = deepcopy(first)
    last.update(
        step=first["step"] + 1,
        joint_actions=[0, 7, 6, 7],
        effective_joint_actions=[0, 7, 7, 7],
        workspace_blocked=[False, False, True, False],
        done=True,
    )
    return {
        "schema": "terra.viewer3d.v1",
        "metadata": {"title": "Native </script> fleet", "source": "synthetic test"},
        "frames": [first, last],
    }


def metric_plan():
    return {
        "schema": "terra.postprocessed.v1",
        "metadata": {"title": "Metric </script> plan", "source": "synthetic test"},
        "grid": {"rows": 2, "cols": 3, "resolution_m": 0.1, "origin_xy_m": [8, 9]},
        "agents": [
            {
                "id": 2,
                "type": 0,
                "action_type": 0,
                "width_m": 1,
                "length_m": 2,
                "reach_m": [0, 3],
            }
        ],
        "initial": {
            "native_m": [0.0] * 6,
            "loose_m": [0.0] * 6,
            "agents": [
                {
                    "id": 2,
                    "pose": [8, 9, 0],
                    "cabin_yaw": 0,
                    "load": 0,
                    "wheel_angle": 0,
                    "shovel_lifted": 0,
                }
            ],
        },
        "workspaces": [],
        "frames": [],
    }


class RenderTest(unittest.TestCase):
    def test_native_gzip_preserves_joint_fields_and_inlines_every_asset(self):
        data = native_joint()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native.json.gz"
            with gzip.open(source, "wt") as stream:
                json.dump(data, stream)
            with patch(
                "subprocess.run",
                side_effect=AssertionError("Rendering must not launch a build"),
            ):
                out = write_html(source, Path(directory) / "nested/native.html")
            text = out.read_text()
        payload = re.search(
            r'<script type="application/json" id="terra-replay">(.*?)</script>',
            text,
            re.S,
        )
        self.assertEqual(json.loads(payload.group(1)), data)
        self.assertNotIn("<script src=", text)
        self.assertNotIn('href="/static/', text)
        self.assertIn("Reserved workspaces", text)
        self.assertIn("The MIT License", text)

    def test_metric_export_uses_packaged_player_without_node_or_ros(self):
        data = metric_plan()
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "subprocess.run",
                side_effect=AssertionError("Rendering must not launch a build"),
            ):
                out = write_html(data, Path(directory) / "metric.html")
            text = out.read_text()
        payload = re.search(r"const DATA=(.*?);\s*</script>", text, re.S)
        self.assertEqual(json.loads(payload.group(1)), data)
        self.assertIn("TerraPostprocessedView", text)
        self.assertIn("The MIT License", text)
        self.assertNotIn("<script src=", text)
        self.assertNotIn("Metric </script>", text)

    def test_explicit_bundle_is_a_file_and_cannot_terminate_its_script(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "prebuilt"):
                player_bundle(directory)
            bundle = Path(directory) / "custom.js"
            bundle.write_text('window.note = "</script>";')
            out = write_html(
                metric_plan(), Path(directory) / "metric.html", bundle=bundle
            )
            self.assertIn('window.note = "<\\/script>";', out.read_text())
            with self.assertRaisesRegex(ValueError, "Expected terra"):
                write_html({"schema": "unsupported"}, out)

    def test_import_does_not_load_jax_or_ros_packages(self):
        code = """
import importlib.abc, sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'rclpy', 'workspace_planner', 'terra_planner_runtime'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, RejectRuntime())
from terra.postprocess.render import player_bundle
from terra.postprocess import dashboard
assert 'TerraPostprocessedView' in player_bundle()[0]
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_dashboard_retains_joint_fields_without_inferred_work_events(self):
        from terra.postprocess import dashboard

        data = native_joint()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "native.json"
            source.write_text(json.dumps(data))
            compact = dashboard.terra3d_data(source)
            out, verdicts = dashboard.build(
                {
                    "cases": [
                        {"name": "Native fleet", "terra3d": str(source), "versions": []}
                    ]
                },
                Path(directory) / "dashboard.html",
            )
            self.assertIn("TerraPlan3D", out.read_text())
        self.assertEqual(verdicts, [])
        self.assertTrue(compact["joint"])
        self.assertEqual(compact["events"], [])
        self.assertEqual(compact["pairs"], [])
        for actual, original in zip(compact["frame_extras"], data["frames"]):
            self.assertEqual(
                actual["workspace_polygons"], original["workspace_polygons"]
            )
            self.assertEqual(actual["joint_actions"], original["joint_actions"])
            self.assertEqual(actual["workspace_blocked"], original["workspace_blocked"])

    def test_joint_validation_rejects_inconsistent_attribution_and_polygons(self):
        mutations = [
            {"actor_id": 2},
            {"action": 6},
            {"joint_actions": [0]},
            {"workspace_blocked": [False, False, 1, False]},
            {"effective_joint_actions": [0, 7]},
            {
                "workspace_polygons": [
                    {"id": 1, "component": "work", "vertices": [[0, 0], [0, 5], [3, 5]]}
                ]
            },
        ]
        for changes in mutations:
            with self.subTest(changes=changes):
                data = native_joint()
                data["frames"][1].update(changes)
                with self.assertRaises(ValueError):
                    validate_replay(data)
        reset = native_joint()
        reset["frames"][0]["effective_joint_actions"] = [7] * 4
        with self.assertRaisesRegex(ValueError, "requested slots"):
            validate_replay(reset)


if __name__ == "__main__":
    unittest.main()
