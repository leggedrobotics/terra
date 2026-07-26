import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT = Path(__file__).parents[2] / "tools" / "build_b0_feasibility_panels.py"
SPEC = importlib.util.spec_from_file_location("b0_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
b0 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = b0
SPEC.loader.exec_module(b0)


def test_declared_panels_reuse_one_straight_anchor():
    anchor = "t_straight_both_d02"
    assert anchor in b0.PANELS["trench_distance"]
    assert anchor in b0.PANELS["trench_side"]
    assert anchor in b0.PANELS["trench_topology"]
    assert set(b0.PRIMARY_EASY_CELLS["foundation"]) == {
        "f_osm_all",
        "f_procedural_all",
        "f_apron_d02",
        "f_apron_d04",
    }
    assert set(b0.PRIMARY_EASY_CELLS["trench"]) == {
        anchor,
        "t_straight_one_d02",
        "t_segmented2_both_d02",
        "t_segmented3_both_d02",
    }


def test_foundation_apron_hits_each_distance_and_capacity_bin():
    dig = np.zeros((b0.MAP_SIZE, b0.MAP_SIZE), dtype=np.bool_)
    dig[28:36, 27:37] = True
    for center in (2, 4, 6, 8):
        dump, metadata = b0.build_apron_dump(
            dig,
            center,
            side_access="all",
        )
        assert not np.any(dig & dump)
        assert dump.sum() >= b0.MINIMUM_CAPACITY_RATIO * dig.sum()
        assert abs(metadata["p50_tiles"] - center) <= (b0.DISTANCE_TOLERANCE_TILES)


def test_trench_side_pair_has_equal_capacity_and_exact_forbidden_side():
    dig = np.zeros((b0.MAP_SIZE, b0.MAP_SIZE), dtype=np.bool_)
    dig[30:33, 20:44] = True
    both, both_metadata = b0.build_apron_dump(
        dig,
        2,
        side_access="both",
        heading_degrees=0.0,
    )
    one, one_metadata = b0.build_apron_dump(
        dig,
        2,
        side_access="one",
        heading_degrees=0.0,
        side_sign=1,
    )
    projection = b0.trench_projection(dig, 0.0)
    assert both.sum() == one.sum()
    assert (both & (projection <= -1.0)).sum() >= 0.40 * both.sum()
    assert (both & (projection >= 1.0)).sum() >= 0.40 * both.sum()
    assert not np.any(one & (projection < 1.0))
    assert abs(both_metadata["p50_tiles"] - 2) <= 0.75
    assert abs(one_metadata["p50_tiles"] - 2) <= 0.75


def test_dihedral_similarity_recognizes_only_shape_equivalence():
    left = np.zeros((b0.MAP_SIZE, b0.MAP_SIZE), dtype=np.bool_)
    left[20:25, 20:34] = True
    equivalent = np.rot90(left)
    distinct = left.copy()
    distinct[25:31, 20:25] = True
    assert b0.maximum_dihedral_iou(left, equivalent) == 1.0
    assert b0.maximum_dihedral_iou(left, distinct) < 0.9
