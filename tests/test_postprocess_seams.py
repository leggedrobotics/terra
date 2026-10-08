"""Seam lips of a converted plan as the dashboard shows them (tmm_seams)."""

import math
from pathlib import Path

import pytest
from shapely.geometry import box


from terra.postprocess import seams as tmm_seams


def report(b_witnesses, rule=None):
    """A digs x in [3.5, 7] facing +x; B, later, stands at x = 13.5 facing -x (its pulls start at x = 13.5 - far)."""
    pair = lambda pose, witnesses: dict(  # noqa: E731
        accepted=True,
        workspace_type="excavate",
        chosen_pose=list(pose),
        witnesses=witnesses,
    )
    return dict(
        design_wkt=box(2.0, -3.0, 12.0, 3.0).wkt,
        required_geometry_wkt=box(3.6, -0.3, 9.9, 0.3).wkt,
        control_offset_xy_m=[0.0, 0.0],
        envelope=dict(blade_width_m=1.3),
        fresh_cutting_rule=rule,
        per_pair=[
            pair(
                (0.0, 0.0, 0.0),
                [dict(theta_rad=0.0, radius_near_m=3.5, radius_far_m=7.0)],
            ),
            pair((13.5, 0.0, math.pi), b_witnesses),
            dict(
                accepted=True,
                workspace_type="collect_dumped_soil",
                chosen_pose=[0.0, 5.0, 0.0],
                witnesses=[],
            ),
        ],
    )


def test_a_butt_joint_has_no_lip_and_an_overlap_pull_gives_it():
    primary = dict(theta_rad=0.0, radius_near_m=3.5, radius_far_m=6.5)
    out = tmm_seams.seam_lips(report([primary]))
    assert out[2] is None and out[0]["seams"] == []
    length, share = tmm_seams.share(out[1]["seams"])
    assert length == pytest.approx(0.6, abs=0.02) and share == pytest.approx(0.0)
    # 0.31 m: a lip ending exactly 0.3 m in reads as 0.29 (1 cm steps; contains_xy excludes the edge), as in
    # workspace_lanes.seam_overlap_readout. The converter fits overlap pulls slightly past their target.
    lip = dict(theta_rad=0.0, radius_near_m=6.5, radius_far_m=6.81, kind="overlap")
    out = tmm_seams.seam_lips(report([primary, lip]))
    assert tmm_seams.share(out[1]["seams"])[1] == pytest.approx(1.0)
    # Without a completion ring the lip pull is part of the completed ground, so nothing lies past it.
    assert out[1]["lip"].is_empty


def test_with_the_cutting_band_the_lip_past_the_planned_ring_is_reported_separately():
    rule = dict(
        planning_completion_radius_min_m=4.35,
        planning_completion_radius_max_m=6.15,
        completion_radius_min_m=4.0,
        completion_radius_max_m=6.5,
    )
    primary = dict(theta_rad=0.0, radius_near_m=3.5, radius_far_m=6.5)
    lip = dict(theta_rad=0.0, radius_near_m=6.0, radius_far_m=6.5, kind="overlap")
    out = tmm_seams.seam_lips(report([primary, lip], rule))
    # A completes x up to 6.15 m; B up to 13.5 - 6.15 = 7.35 m, so a 1.2 m gap and no seam without the lip band.
    assert out[1]["seams"] == []
    # The lip band reaches x = 7.0, past B's planned ring: reported as B's lip.
    assert out[1]["lip"].bounds[0] == pytest.approx(7.0, abs=1e-6)
