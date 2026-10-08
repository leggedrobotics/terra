"""Offline geometry/model scope stays independent from the ROS and JAX runtimes."""

import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import box

from terra.postprocess import geometry


def test_source_cells_are_full_rectangles_on_the_rotated_xy_lattice():
    mask = np.zeros((4, 7), bool)
    mask[1, 3] = mask[2, 3] = True
    actual = geometry.source_mask_geometry(
        mask,
        dict(
            meters_per_tile=0.5,
            yaw_map_from_plan_rad=math.pi / 2,
            origin_map_xy_m=[5, -2],
        ),
    )
    assert actual.area == pytest.approx(0.5)
    assert actual.bounds == pytest.approx((3, -1.5, 3.5, -0.5))


def test_continuous_bands_preserve_holes_and_finite_ends():
    allowed = box(1, -1, 7, 1).difference(box(3, -0.2, 5, 0.2))
    runs = geometry.continuous_radial_runs(
        allowed, np.array([0, 0]), np.array([1, 0]), 0, 8, 0.5
    )
    np.testing.assert_allclose(runs, [[1, 3], [5, 7]], atol=2e-6, rtol=0)
    for near, far in runs:
        band = geometry.pull_band_polygon([0, 0], [1, 0], near, far, 0.5)
        assert band.difference(allowed).area < 3e-6
    assert (
        geometry.continuous_radial_runs(
            box(1, -0.4, 7, 0.4), [0, 0], np.array([1, 0]), 1, 7, 0.5
        )
        == []
    )


def test_offline_dump_footprint_model_includes_closed_edge_and_one_coarse_cell():
    assert (0, 3) in geometry.dump_footprint_offsets(0.15)
    assert (0, 4) not in geometry.dump_footprint_offsets(0.15)
    assert set(geometry.dump_footprint_offsets(0.6)) == {
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    }


def test_portable_modules_import_without_ros_or_jax(tmp_path):
    script = """
import importlib, importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'rclpy', 'ament_index_python', 'workspace_planner', 'terra_planner_runtime', 'jax', 'jaxlib'}:
            raise AssertionError('runtime dependency: ' + name)
        if name in {'terra.env', 'terra.state'}:
            raise AssertionError('native runtime dependency: ' + name)
sys.meta_path.insert(0, BlockRuntime())
for name in ('fleet','fleet_geometry','timeline','replay','seams','soil','geometry'):
    importlib.import_module('terra.postprocess.' + name)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
