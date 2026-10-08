"""Forecast loose-soil heights (terra_planner_runtime.spoil_heights) used by the converter's opt-in height model."""

import math

import numpy as np
import pytest

from terra.postprocess.soil import (
    SpoilHeights,
    plateau_deposit,
    release_region,
    tall_core_radius,
    toe_radius,
)


def _grid(n=120, res=0.1):
    x, y = np.meshgrid(np.arange(n) * res, np.arange(n) * res)
    return x, y, res


def test_one_bucket_never_reaches_half_a_metre_but_a_workspace_pile_does():
    # A 27 degree cone of 0.255 m3 is about 0.40 m high; 2.21 m3 stands 0.5 m high out to about 0.63 m.
    assert tall_core_radius(0.255, 0.5) == 0.0
    assert tall_core_radius(2.21, 0.5) == pytest.approx(0.63, abs=0.01)
    assert tall_core_radius(0.0, 0.5) == 0.0


def test_loads_conserve_volume_and_the_tall_part_matches_one_cone():
    X, Y, res = _grid()
    heights = SpoilHeights(X, Y, res)
    centre = np.zeros(X.shape, dtype=bool)
    centre[60, 60] = True
    heights.deposit(centre, 2.21)
    assert heights.height.sum() * res**2 == pytest.approx(2.21, rel=1e-6)
    tall = heights.tall(0.5)
    rows, cols = np.nonzero(tall)
    radius = np.hypot(X[rows, cols] - X[60, 60], Y[rows, cols] - Y[60, 60]).max()
    assert radius == pytest.approx(tall_core_radius(2.21, 0.5), abs=0.1)
    # A cut lifts the loose soil in its cells and reports its volume.
    lifted = np.zeros_like(centre)
    lifted[55:66, 55:66] = True
    volume = heights.remove(lifted)
    assert volume > 0.0 and not heights.height[lifted].any()
    assert heights.height.sum() * res**2 == pytest.approx(2.21 - volume, rel=1e-6)


def test_a_workspace_deposit_fills_its_region_to_one_level_with_repose_slopes_around_it():
    # Workspace level (Lorenzo, 2 October 2026): 4 m3 over a 0.35 m disk region is one plateau, not stacked loads.
    # A frustum of top radius r and height h holds pi (r2 h + r h2 / t + h3 / (3 t2)): 0.82 m for 4 m3, against
    # 1.0 m for the cone of the same soil on one cell.
    X, Y, res = _grid()
    tan = math.tan(math.radians(27.0))
    region = np.hypot(X - 6.0, Y - 6.0) <= 0.35 + 1e-9
    heights = SpoilHeights(X, Y, res)
    heights.deposit(region, 4.0)
    assert heights.height.sum() * res**2 == pytest.approx(4.0, rel=1e-6)
    level = heights.height[region]
    assert level.max() - level.min() < 1e-6 and level.max() == pytest.approx(
        0.82, abs=0.03
    )
    # Outside the region the soil falls at the repose slope from the nearest region cell.
    assert heights.height[60, 60 + 10] == pytest.approx(
        level.max() - tan * (1.0 - 0.3), abs=0.02
    )
    cone = SpoilHeights(X, Y, res)
    cell = np.zeros_like(region)
    cell[60, 60] = True
    cone.deposit(cell, 4.0)
    assert cone.height.max() == pytest.approx(
        (3 * 4.0 * tan**2 / math.pi) ** (1 / 3), abs=0.03
    )
    assert cone.height.max() > level.max() + 0.1


def test_a_deposit_stacks_on_the_soil_already_there_and_spreads_only_from_its_region():
    X, Y, res = _grid()
    region = (np.abs(X - 6.0) <= 0.45) & (np.abs(Y - 6.0) <= 0.45)
    heights = SpoilHeights(X, Y, res)
    heights.deposit(region, 1.0)
    first = heights.height.copy()
    heights.deposit(region, 1.0)
    assert heights.height.sum() * res**2 == pytest.approx(2.0, rel=1e-6)
    released = release_region(region, res)
    assert heights.height[released].min() > first[released].max()
    # A pit beyond the toe that the slope would reach only across bare ground gets no soil.
    surface = np.zeros(X.shape)
    pit = (X >= 9.0) & (X <= 10.0)
    surface[pit] = -3.0
    window, added, off_map = plateau_deposit(
        surface, region, 1.0, res, math.tan(math.radians(27.0))
    )
    full = np.zeros(X.shape)
    full[window] = added
    assert (
        not off_map
        and not full[pit].any()
        and full.sum() * res**2 == pytest.approx(1.0, rel=1e-6)
    )
    # Two far-apart cells fill to one level each: equal shares.
    split = np.zeros(X.shape, dtype=bool)
    split[30, 30] = split[90, 90] = True
    heights = SpoilHeights(X, Y, res)
    heights.deposit(split, 2 * 0.255)
    assert heights.height[30, 30] == pytest.approx(heights.height[90, 90], abs=1e-6)
    assert heights.height.max() == pytest.approx(0.398, abs=0.02)


def test_pile_radius_counts_the_soil_a_load_would_join():
    X, Y, res = _grid()
    heights = SpoilHeights(X, Y, res)
    # Bare ground: every centre carries one cone of the workspace's soil.
    bare = heights.pile_radius(2.21)
    assert bare[60, 60] == pytest.approx(toe_radius(2.21), abs=1e-9)
    assert bare.min() == pytest.approx(bare.max())
    # A 5 m3 pile at the centre: a load on it carries most of that pile, a load far away only its own soil.
    centre = np.zeros(X.shape, dtype=bool)
    centre[60, 60] = True
    heights.deposit(centre, 5.0)
    joined = heights.pile_radius(0.5)
    assert toe_radius(0.5) + 0.8 < joined[60, 60] <= toe_radius(5.5) + 1e-9
    assert toe_radius(0.5) < joined[60, 70] < joined[60, 60]
    assert joined[60, 5] == pytest.approx(toe_radius(0.5), abs=1e-6)


def test_a_deposit_is_the_same_on_a_grown_lattice():
    # The same region on a lattice grown by 52 cells per side (a map margin) gets the same soil.
    def lattice(origin, size):
        return np.meshgrid(
            origin + np.arange(size) * 0.1, origin + np.arange(size) * 0.1
        )

    for row, col in ((200, 200), (123, 57), (301, 222)):
        small, large = SpoilHeights(*lattice(-18.25, 366), 0.1), SpoilHeights(
            *lattice(-23.45, 470), 0.1
        )
        centres = np.zeros((366, 366), dtype=bool)
        centres[row - 3 : row + 4, col - 3 : col + 4] = True
        small.deposit(centres, 2.3)
        large.deposit(np.pad(centres, 52), 2.3)
        np.testing.assert_allclose(
            large.height[52:-52, 52:-52], small.height, rtol=0, atol=1e-9
        )


def test_soil_is_released_on_the_centres_whose_pile_footprint_lies_in_the_mask():
    # ROS ranks dump centres with their whole 0.45 m footprint in the mask first: a 0.9 m disk mask releases its soil
    # on its 0.45 m interior, a 0.35 m mask (no such centre) on all of it.
    X, Y, res = _grid()
    distance = np.hypot(X - 6.0, Y - 6.0)
    wide = distance <= 0.9 + 1e-9
    released = release_region(wide, res)
    assert (
        released.any()
        and released.sum() < wide.sum()
        and distance[released].max() <= 0.45 + 1e-9
    )
    compact = distance <= 0.35 + 1e-9
    np.testing.assert_array_equal(release_region(compact, res), compact)
    # So 4 m3 on the wide mask stands nearly as high as on the compact one.
    tops = []
    for mask in (wide, compact):
        heights = SpoilHeights(X, Y, res)
        heights.deposit(mask, 4.0)
        tops.append(heights.height.max())
    assert abs(tops[0] - tops[1]) < 0.12
