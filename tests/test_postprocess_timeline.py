"""The 3D adapter preserves metric replay, fine masks, route direction and identity."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.postprocess import timeline as tmm_timeline


def example():
    # A non-square map with a changed cell above the old uint16 index ceiling.
    shape = (257, 259)
    masks = np.zeros((2, *shape), dtype=bool)
    masks[0, 256, 258] = True
    masks[1, 0, 0] = True
    native = np.zeros((5, *shape), dtype=np.float32)
    native[1:, 256, 258] = -0.375
    native[3:, 0, 0] = -0.625
    loose = np.zeros_like(native)
    loose[2:, 4, 7] = 0.12345
    loose[4:, 6, 8] = 0.23456
    path = [
        [3.0, -2.0, 3.12],
        [3.6, -2.0, 3.13],
        [3.2, -2.0, -3.13],
        [3.2, -1.0, -3.12],
    ]
    case = SimpleNamespace(
        tag="fine metric replay",
        conversion=Path("/not/a/real/conversion"),
        report={"inputs": {"input_dir": "/not/a/real/source"}},
        shape=shape,
        origin=np.array([-9.2, 12.4]),
        res=0.1,
        footprint=np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]),
        poses=np.array([path[0], path[-1]]),
        known=np.ones(shape, dtype=bool),
        obstacle=np.zeros(shape, dtype=bool),
        target=masks.any(axis=0),
        final_ground=masks[1],
        support=masks,
        completion=masks,
        dump_centres=masks[:, ::-1].copy(),
        converter_deposit=masks,
        routes={(1, 2): {"path": path, "passed": True}},
    )
    events = [
        dict(
            workspace=k + 1,
            kind="excavate",
            source_pair=0 if k == 0 else None,
            route=None if k == 0 else {"blocked": False},
            steps={"cut": 1 + 2 * k},
            cut={"payload_m3": 0.375},
            dump={"loads": [{"step": 2 + 2 * k, "x_m": 2.0, "y_m": 5.0}]},
        )
        for k in range(2)
    ]
    return dict(
        case=case,
        native_frames=native,
        loose_frames=loose,
        timeline=[{}] * 4,
        design=-masks.any(axis=0).astype(float) * 0.5,
        rules=SimpleNamespace(band_min_m=4.0, entry_max_m=7.0),
        events=events,
    )


def test_fleet_timeline_preserves_rotated_grid_material_and_all_identities():
    from terra.postprocess import fleet as tmm_fleet
    from .test_postprocess_fleet import dual_excavator

    source = dual_excavator()
    angle, origin = 0.7, np.array([4.0, -7.0])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    source["grid"].update(yaw_rad=angle, origin_xy_m=origin.tolist())
    for state in [a["initial_state"] for a in source["agents"]] + [
        action[when] for action in source["actions"] for when in ("before", "after")
    ]:
        state["position_xy_m"] = (rotation @ state["position_xy_m"] + origin).tolist()
        state["base_yaw_rad"] += angle
    result = tmm_fleet.postprocess(source)
    data = tmm_timeline.from_fleet(result)
    np.testing.assert_allclose(
        data["grid"]["origin_xy_m"], origin + rotation @ [0.5, 0.5]
    )
    assert data["grid"]["yaw_rad"] == angle
    assert data["metadata"]["source_ids"] == {"alpha": 0, "bravo": 1}
    assert data["frames"][0]["terrain_changes"] == [[3 * 22 + 5, -0.1, 0.0]]
    assert data["workspaces"][0]["masks"]["dig"] == [[3, 5, 1]]
    assert all([a["id"] for a in f["agents"]] == [0, 1] for f in data["frames"])
    assert all(len(f["reservations"]) == 2 for f in data["frames"])
    assert data["initial"]["agents"][0]["cabin_yaw"] == angle
    assert [f["work"][0]["agent_id"] for f in data["frames"]] == [0, 1, 0, 1]


def test_scheduled_timeline_omits_only_noop_waits_and_rejects_partial_schedule():
    from terra.postprocess import fleet as tmm_fleet
    from .test_postprocess_fleet import dual_excavator, action

    source = dual_excavator()
    state = source["agents"][0]["initial_state"]
    source["actions"].insert(0, action("alpha", 0, state, state, code=7))
    result = tmm_fleet.postprocess(source)
    data = tmm_timeline.from_fleet(result)
    assert len(data["frames"]) == 4
    assert data["metadata"]["omitted_wait_ids"] == ["alpha:0:0"]
    assert len(tmm_timeline.from_fleet(result, variant="cleaned")["frames"]) == 5
    result["schedule"]["complete"] = False
    assert len(tmm_timeline.from_fleet(result)["frames"]) == 5
    with pytest.raises(ValueError, match="incomplete schedule"):
        tmm_timeline.from_fleet(result, variant="scheduled")


def test_rotated_nonsquare_mixed_timeline_keeps_initial_soil_and_partial_pickups():
    from terra.postprocess import fleet as tmm_fleet
    from .test_postprocess_fleet import source, state, action

    value = source(("excavator", "skid"), starts=[state(16, 16), state(4, 5)])
    for key in ("initial_terrain", "target", "accepted_dump_mask", "obstacles"):
        value[key] = np.pad(np.asarray(value[key]), ((0, 0), (0, 3))).tolist()
    value["initial_terrain"][6][5] = 2
    value["accepted_dump_mask"][6][6] = True
    first = value["agents"][1]["initial_state"]
    pickup1 = dict(first, position_xy_m=[4.5, 5], load_units=1)
    pickup2 = dict(first, position_xy_m=[5.0, 5], load_units=2)
    empty = dict(pickup2, load_units=0)
    value["actions"] = [
        action("bravo", 0, first, pickup1, [(6, 5, 2, 1)], code=0),
        action("bravo", 1, pickup1, pickup2, [(6, 5, 1, 0)], code=0),
        action("bravo", 2, pickup2, empty, [(6, 6, 0, 2)]),
    ]
    angle, origin = -0.4, np.array([8.0, -3.0])
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    value["grid"].update(yaw_rad=angle, origin_xy_m=origin.tolist())
    for item in [a["initial_state"] for a in value["agents"]] + [
        a[when] for a in value["actions"] for when in ("before", "after")
    ]:
        item["position_xy_m"] = (rotation @ item["position_xy_m"] + origin).tolist()
        item["base_yaw_rad"] += angle
    data = tmm_timeline.from_fleet(tmm_fleet.postprocess(value), variant="cleaned")
    assert (data["grid"]["rows"], data["grid"]["cols"]) == (25, 22)
    assert data["initial"]["loose_m"][5 * 22 + 6] == 0.2
    assert [f["agents"][1]["load"] for f in data["frames"]] == [1, 2, 0]
    assert [f["phase"] for f in data["frames"]] == ["collect", "collect", "deliver"]
    assert [f["terrain_changes"] for f in data["frames"]] == [
        [[5 * 22 + 6, 0.0, 0.1]],
        [[5 * 22 + 6, 0.0, 0.0]],
        [[6 * 22 + 6, 0.0, 0.2]],
    ]
    assert (
        data["frames"][0]["agents"][1]["pose"][:2]
        != data["initial"]["agents"][1]["pose"][:2]
    )


def test_reconstructs_exact_metric_terrain_above_uint16_index_limit():
    result = example()
    data = tmm_timeline.from_replay(result)
    a = np.array(data["initial"]["native_m"])
    b = np.array(data["initial"]["loose_m"])
    seen = []
    for frame in data["frames"]:
        for i, x, y in frame["terrain_changes"]:
            seen.append(i)
            a[i], b[i] = x, y
        if frame["terrain_changes"]:
            assert frame["work"][0]["agent_id"] == 0
    assert max(seen) > 65535
    np.testing.assert_array_equal(a, result["native_frames"][-1].ravel())
    np.testing.assert_array_equal(b, result["loose_frames"][-1].ravel())
    assert data["grid"]["origin_xy_m"] == [-9.2, 12.4]
    assert data["workspaces"][0]["masks"]["dig"] == [[256, 258, 1]]
    assert data["workspaces"][1]["source_pair"] is None
    json.dumps(data, allow_nan=False)


def test_saved_route_keeps_every_pose_reversal_and_yaw_wrap():
    result = example()
    data = tmm_timeline.from_replay(result)
    drives = [f for f in data["frames"] if f["phase"] == "drive"]
    assert [f["agents"][0]["pose"] for f in drives] == result["case"].routes[(1, 2)][
        "path"
    ]
    assert all(f["route_status"] == "checked" for f in drives)
    assert all(not f["terrain_changes"] for f in drives)


def test_radial_arrival_requires_recorded_evidence_or_explicit_validator(tmp_path):
    result = example()
    result["case"].conversion = tmp_path
    waypoints = [{"radial_workspace": {"test_geometry": k}} for k in range(4)]
    (tmp_path / "terra_plan.json").write_text(json.dumps({"waypoints": waypoints}))
    unverified = tmm_timeline.from_replay(result)
    assert all(
        f["route_status"] == "unverified"
        for f in unverified["frames"]
        if f["phase"] == "drive"
    )
    assert any(f["phase"] == "relocate" for f in unverified["frames"])
    seen = []

    def validator(pose, workspace, tolerance):
        seen.append((pose, workspace, tolerance))
        return {"passed": True}

    checked = tmm_timeline.from_replay(result, station_validator=validator)
    assert all(
        f["route_status"] == "checked"
        for f in checked["frames"]
        if f["phase"] == "drive"
    )
    assert not any(f["phase"] == "relocate" for f in checked["frames"])
    assert seen[0][0] == result["case"].routes[(1, 2)]["path"][-1]
    assert seen[0][1]["radial_workspace"] == {"test_geometry": 2}
    assert seen[0][2] == 0
    result["case"].routes[(1, 2)]["station_arrival"] = {"passed": True}
    recorded = tmm_timeline.from_replay(result)
    assert all(
        f["route_status"] == "checked"
        for f in recorded["frames"]
        if f["phase"] == "drive"
    )
    refused = tmm_timeline.from_replay(
        result, station_validator=lambda *_: {"passed": False}
    )
    assert all(
        f["route_status"] == "unverified"
        for f in refused["frames"]
        if f["phase"] == "drive"
    )


@pytest.mark.parametrize(
    "saved,expected", [(None, "missing"), ({"path": [], "passed": False}, "failed")]
)
def test_unavailable_route_is_a_discontinuous_station_connector(saved, expected):
    result = example()
    result["case"].routes = {} if saved is None else {(1, 2): saved}
    data = tmm_timeline.from_replay(result)
    assert not any(f["phase"] == "drive" for f in data["frames"])
    connector = next(f for f in data["frames"] if f["phase"] == "relocate")
    assert connector["route_status"] == expected


def test_route_rejected_on_current_terrain_remains_failed():
    result = example()
    result["events"][1]["route"]["blocked"] = True
    data = tmm_timeline.from_replay(result)
    assert all(
        f["route_status"] == "failed" for f in data["frames"] if f["phase"] == "drive"
    )


def test_stale_route_endpoint_cannot_relocate_the_refined_workspace():
    result = example()
    result["case"].routes[(1, 2)]["path"][-1] = [20.0, 20.0, 0.0]
    data = tmm_timeline.from_replay(result)
    assert all(
        f["route_status"] == "unverified"
        for f in data["frames"]
        if f["phase"] == "drive"
    )
    assert any(f["phase"] == "relocate" for f in data["frames"])
    cut = [f for f in data["frames"] if f["phase"] == "cut"][-1]
    assert cut["agents"][0]["pose"] == result["case"].poses[1].tolist()


def test_stale_start_does_not_hide_a_failed_route():
    result = example()
    result["case"].routes[(1, 2)].update(passed=False)
    result["case"].routes[(1, 2)]["path"][0] = [20.0, 20.0, 0.0]
    data = tmm_timeline.from_replay(result)
    assert all(
        f["route_status"] == "failed" for f in data["frames"] if f["phase"] == "drive"
    )


def test_grid_masks_identity_and_material_ownership_are_validated():
    data = tmm_timeline.from_replay(example())
    for mutation in ("identity", "mask", "ownership", "terrain"):
        bad = deepcopy(data)
        if mutation == "identity":
            bad["frames"][0]["agents"][0]["id"] = 7
        elif mutation == "mask":
            bad["workspaces"][0]["masks"]["dig"] = [[256, 258, 2]]
        elif mutation == "ownership":
            next(f for f in bad["frames"] if f["terrain_changes"])["work"] = []
        else:
            bad["initial"]["loose_m"][0] = float("nan")
        with pytest.raises(ValueError):
            tmm_timeline.validate(bad)
