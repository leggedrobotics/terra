"""Portable workspace replay under the declared Terra-to-machine ground rules."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from terra.postprocess import replay as tmm_replay


def test_runtime_validation_is_explicit_and_missing_evidence_is_not_checked():
    case = dataclasses.replace(
        _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]]), complete=True
    )
    case = dataclasses.replace(case, final_zone=~case.target)
    unchecked = tmm_replay.replay(case)
    assert unchecked["runtime"]["status"] == "not checked"
    assert tmm_replay.validity(unchecked)["checks"][0]["status"] == "not checked"
    refusal = dict(
        status="fail",
        message="waypoint[2] DIG completion must be nonempty",
        error="RuntimeError",
    )
    seen = []

    def validate(actual):
        seen.append(actual)
        return refusal

    result = tmm_replay.replay(case, runtime_validator=validate)
    converter = tmm_replay.validity(result)["checks"][0]
    assert seen == [case]
    assert converter["status"] == "fail"
    assert converter["failures"][0]["workspace"] == 2
    assert converter["failures"][0]["kind"] == "runtime_refused"
    with pytest.raises(ValueError, match="explicit pass/fail"):
        tmm_replay.runtime_checks(case, lambda _: {})
    result["runtime"] = tmm_replay.runtime_checks(case, lambda _: {"status": "fail"})
    converter = tmm_replay.validity(result)["checks"][0]
    assert converter["status"] == "fail"
    assert converter["failures"][0]["kind"] == "runtime_refused"


RES = 0.1

TILE = 4.0 / 7.0


def _grid_mask(shape, origin, x0, x1, y0, y1):
    rows, cols = np.indices(shape)
    x, y = origin[0] + cols * RES, origin[1] + rows * RES
    return (x > x0) & (x < x1) & (y > y0) & (y < y1)


def _disk(shape, origin, cx, cy, radius):
    rows, cols = np.indices(shape)
    return (
        np.hypot(origin[0] + cols * RES - cx, origin[1] + rows * RES - cy)
        <= radius + 1e-9
    )


def _trench_case(dump_patches):
    """A 1 m wide, 3.5 m long, 0.5 m deep strip dug from its left side by two stations.

    Station 1 at (-4.5, 0.5) finishes y in [0, 1]; station 2 at (-4.5, 2.25) finishes y in [1, 3.5].
    ``dump_patches`` holds each station's dump-centre patches as (x, y, radius).
    """
    shape, origin = (200, 200), np.array(
        [-9.95, -9.95]
    )  # cell centres on 0.05 m offsets
    target = _grid_mask(shape, origin, -0.5, 0.5, 0.0, 3.5)
    completion = np.stack(
        [
            target & _grid_mask(shape, origin, -1, 1, 0.0, 1.0),
            target & _grid_mask(shape, origin, -1, 1, 1.0, 3.5),
        ]
    )
    centres = np.stack(
        [
            np.any([_disk(shape, origin, *patch) for patch in patches], axis=0)
            for patches in dump_patches
        ]
    )
    n = len(dump_patches)
    return tmm_replay.Case(
        tag="strip",
        conversion=Path("/nonexistent"),
        report=dict(inputs=dict(input_dir="/nonexistent")),
        origin=origin,
        res=RES,
        target=target,
        required=target.copy(),
        depth_m=0.5,
        tile_m=TILE,
        final_zone=np.zeros(shape, dtype=bool),
        obstacle=np.zeros(shape, dtype=bool),
        known=np.ones(shape, dtype=bool),
        kinds=["excavate"] * n,
        sources=list(range(n)),
        poses=np.array([[-4.5, 0.5, 0.0], [-4.5, 2.25, 0.0]])[:n],
        support=completion[:n].copy(),
        completion=completion[:n],
        dump_centres=centres,
        converter_blocked=np.zeros((n, *shape), dtype=bool),
        converter_deposit=np.zeros((n, *shape), dtype=bool),
        native_poses=[],
        native_dump=[np.zeros(shape, dtype=bool)] * n,
        native_dig=[completion[k].copy() for k in range(n)],
        footprint=np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
        dump_rule=dict(
            reach_radius_m=6.0,
            reach_origin_xyz=[0.0, 0.0, 0.647],
            min_base_radius_m=4.5,
            # One map cell beyond the 1.1 m support: no soil in the excavation, no band (Lorenzo, 1 October 2026).
            min_excavated_clearance_m=RES,
        ),
        routes={},
        complete=False,
    )


def test_repeated_dumps_at_one_point_accumulate_the_whole_volume_without_clipping():
    case = _trench_case([[(0.05, 0.05, 0.05)]])
    surface = np.zeros(case.shape)
    tan = math.tan(math.radians(27.0))
    window, added, apex, off_map = tmm_replay.cone_deposit(
        case, surface, 0.05, 0.05, 0.255, tan, case.known
    )
    surface[window] += added
    first_peak, first_area = apex, int((surface >= 0.01).sum())
    window, added, apex, _ = tmm_replay.cone_deposit(
        case, surface, 0.05, 0.05, 0.255, tan, case.known
    )
    surface[window] += added
    assert surface.sum() * RES**2 == pytest.approx(0.51, rel=1e-9)
    # One 0.255 m3 cone at 27 degrees is ~0.40 m high; the second load raises and widens it.
    assert first_peak == pytest.approx(0.398, abs=0.02)
    assert apex > first_peak + 0.05 and int((surface >= 0.01).sum()) > first_area
    assert not off_map
    # Soil is never clipped at the map edge: it stays on the map and is flagged.
    _, added, _, off_map = tmm_replay.cone_deposit(
        case, np.zeros(case.shape), -9.85, 0.05, 0.255, tan, case.known
    )
    assert off_map and added.sum() * RES**2 == pytest.approx(0.255, rel=1e-9)


def test_a_later_cut_removes_spoil_dumped_on_its_target_then_cuts_native_ground():
    # Station 1 dumps on station 2's future target, 2.5 m from its own pit.
    case = _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    result = tmm_replay.replay(case)
    first, second = result["events"]
    assert [first["kind"], second["kind"]] == ["excavate", "excavate"]
    assert first["cut"]["native_cut_m3"] == pytest.approx(0.5)
    assert all(
        load["choice"] == "ros" and load["pile_clear"]
        for load in first["dump"]["loads"]
    )
    # Station 2 lifts the part of that pile on its ground, then cuts its 1.25 m3 of native soil.
    assert second["cut"]["native_cut_m3"] == pytest.approx(1.25)
    assert 0.0 < second["cut"]["loose_lifted_m3"] < 0.5
    assert second["cut"]["payload_m3"] == pytest.approx(
        1.25 + second["cut"]["loose_lifted_m3"]
    )
    totals = result["totals"]
    assert totals["required_left_m2"] == 0.0 and totals["loose_in_finished_m3"] == 0.0
    # Volume is conserved: everything cut lies on the ground at the end.
    assert totals["loose_on_ground_m3"] == pytest.approx(1.75, abs=1e-3)
    kinds = {issue["kind"] for issue in result["issues"]}
    # The pile's part beyond the design end cannot be dug out by an excavation, and no final zone takes it: soil
    # left outside the final zones at the end breaks the end-state rule (Lorenzo, 1 October 2026).
    left = [
        issue for issue in result["issues"] if issue["kind"] == "temporary_spoil_left"
    ]
    assert left and all(issue["severity"] == "physical" for issue in left)
    assert not kinds & {
        "dump_into_pit",
        "dump_ros_refuses",
        "arrival_blocked",
        "completion_out_of_band",
    }
    # One frame per cut and per load; the missing converter output and the soil left fail this synthetic case.
    assert len(result["timeline"]) == 2 + len(first["dump"]["loads"]) + len(
        second["dump"]["loads"]
    )
    checks = {
        check["id"]: check["status"] for check in tmm_replay.validity(result)["checks"]
    }
    assert checks["coverage"] == checks["pile_clearance"] == "pass"
    assert checks["routes"] == "not checked"
    assert [check for check, status in checks.items() if status == "fail"] == [
        "converter_plan",
        "spoil_at_end",
    ]
    spoil = next(
        check
        for check in tmm_replay.validity(result)["checks"]
        if check["id"] == "spoil_at_end"
    )
    assert (
        len(spoil["failures"]) == len(left)
        and "released off the final zones left outside" in spoil["detail"]
    )


def test_a_dump_that_puts_soil_in_the_pit_is_rejected_and_its_spill_is_reported():
    # A dump region 0.45 m from the pit dug by station 1 (y up to 1.0): the workspace's deposit reaches into it.
    near = [[(0.05, 1.45, 0.05)], [(-0.45, 6.45, 0.05)]]
    result = tmm_replay.replay(_trench_case(near))
    loads = result["events"][0]["dump"]["loads"]
    # Workspace level: one deposit of the station's whole soil over its region (Lorenzo, 2 October 2026).
    assert (
        len(loads) == 1
        and loads[0]["choice"] == "no_clear_point"
        and not loads[0]["ros_admitted"]
    )
    assert loads[0]["volume_m3"] == pytest.approx(
        result["events"][0]["cut"]["payload_m3"], abs=1e-4
    )
    spill = [issue for issue in result["issues"] if issue["kind"] == "dump_into_pit"]
    assert spill and spill[0]["severity"] == "physical" and spill[0]["workspace"] == 1
    # The failure points at the dump, one frame after the cut, and says what an edit needs.
    assert (
        spill[0]["step"] == loads[0]["step"] == result["events"][0]["steps"]["cut"] + 1
    )
    fix = spill[0]["fix"]
    assert (
        fix["terra_step"] == 0
        and fix["needed_centre_distance_m"] > fix["current_centre_distance_m"]
    )
    assert (
        "plan set-dump --step 0" in spill[0]["text"]
        or "No final-zone centre" in spill[0]["text"]
    )
    assert loads[0]["pit_spill_m3"] >= tmm_replay.Rules().min_leftover_m3
    checks = {
        check["id"]: check["status"] for check in tmm_replay.validity(result)["checks"]
    }
    assert checks["pile_clearance"] == "fail"
    # The deposit covers its whole region: a region that also holds a clear patch still spills from its near part.
    both = [[(0.05, 1.45, 0.05), (0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]]
    result = tmm_replay.replay(_trench_case(both))
    (deposit,) = result["events"][0]["dump"]["loads"]
    assert (
        deposit["y_m"] == pytest.approx(2.45, abs=0.01) and deposit["choice"] == "ros"
    )
    assert [issue for issue in result["issues"] if issue["kind"] == "dump_into_pit"]
    # A region of the clear patch alone keeps the deposit out of the pit.
    far = [[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]]
    result = tmm_replay.replay(_trench_case(far))
    assert not [issue for issue in result["issues"] if issue["kind"] == "dump_into_pit"]


def test_a_pile_close_to_the_pit_passes_when_no_soil_lands_in_it():
    # 1.6 m from the pit: ROS admits the centre (1.1 m support and one cell) and neither cone reaches the pit, so the
    # old two-tile band would have failed it but the 1 October rule passes it.
    close = [[(0.05, 2.55, 0.05)], [(-0.45, 6.45, 0.05)]]
    result = tmm_replay.replay(_trench_case(close))
    loads = result["events"][0]["dump"]["loads"]
    assert loads and all(
        load["choice"] == "ros" and load["pile_clear"] for load in loads
    )
    assert all(
        load["pit_spill_m3"] == 0.0 and load["gap_to_excavation_m"] < 2.0 * TILE
        for load in loads
    )
    assert not [
        issue
        for issue in result["issues"]
        if issue["kind"] in ("dump_into_pit", "dump_touches_pit")
    ]
    pile = next(
        check
        for check in tmm_replay.validity(result)["checks"]
        if check["id"] == "pile_clearance"
    )
    assert pile["status"] == "pass"


def test_a_dump_centre_near_the_reach_edge_is_lost_from_far_stops_once_its_pile_grows():
    # Station 2's single centre lies 5.61 m away: ROS reaches it on original ground from every stop within 0.35 m (the
    # converter's old check), but from the far stops (5.96 m) no longer at the top of its 1.25 m3 deposit (0.68 m).
    # Station 1's centre (5.45 m) stays reachable from every stop at the top of its 0.5 m3 deposit.
    case = _trench_case([[(-4.45, -4.95, 0.05)], [(-1.15, -2.25, 0.05)]])
    case = dataclasses.replace(
        case, dump_rule=dict(case.dump_rule, station_tolerance_m=0.35)
    )
    result = tmm_replay.replay(case)
    (first,), (second,) = (event["dump"]["loads"] for event in result["events"])
    assert first["stops"] == 25 and first["stops_without_admitted_centre"] == 0
    # Admitted at the nominal station, lost from some stops.
    assert (
        second["choice"] == "ros"
        and second["reachable"]
        and 0 < second["stops_without_admitted_centre"] < 25
    )
    assert second["pile_top_m"] == pytest.approx(0.68, abs=0.02)
    stranded = [
        issue
        for issue in result["issues"]
        if issue["kind"] == "dump_unreachable_from_stop"
    ]
    assert [
        (issue["workspace"], issue["severity"], issue["loads"]) for issue in stranded
    ] == [(2, "runtime", 1)]
    assert result["totals"]["loads_unreachable_from_some_stop"] == 1
    admission = next(
        check
        for check in tmm_replay.validity(result)["checks"]
        if check["id"] == "dump_admission"
    )
    assert admission["status"] == "unverified"


def test_a_collection_may_lift_loose_soil_outside_the_fresh_cut_band():
    # Station 2 at 3.0 m from its ground: outside the 4.0-6.5 m band as an excavation, fine as a collection.
    case = _trench_case([[(-0.45, 6.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    case = dataclasses.replace(
        case, poses=np.array([[-4.5, 0.5, 0.0], [-3.0, 2.25, 0.0]])
    )
    kinds = {issue["kind"] for issue in tmm_replay.replay(case)["issues"]}
    assert "completion_out_of_band" in kinds
    collected = dataclasses.replace(case, kinds=["excavate", "collect_dumped_soil"])
    assert "completion_out_of_band" not in {
        issue["kind"] for issue in tmm_replay.replay(collected)["issues"]
    }


def test_a_step_the_converter_rejected_for_want_of_a_dump_gets_a_suggested_centre():
    # Only station 1 was kept; Terra step 1 was rejected with no reachable dump. A final zone lies north of the pit.
    case = _trench_case([[(-0.45, 6.45, 0.05)]])
    final = _grid_mask(case.shape, case.origin, -4.0, 1.0, 5.5, 7.5)
    case = dataclasses.replace(
        case,
        final_zone=final,
        native_dig=[
            case.completion[0],
            _grid_mask(case.shape, case.origin, -0.5, 0.5, 1.0, 3.5) & case.target,
        ],
        native_dump=[np.zeros(case.shape, bool), np.zeros(case.shape, bool)],
        report=dict(
            inputs=dict(input_dir="/nonexistent"),
            per_pair=[
                dict(pair_index=0, accepted=True),
                dict(
                    pair_index=1,
                    accepted=False,
                    original_pose=[-4.5, 2.25, 0.0],
                    placement_rejections=dict(
                        no_reachable_destination_dump_candidate=3
                    ),
                ),
            ],
        ),
    )
    result = tmm_replay.replay(case)
    rejected = [
        issue
        for issue in result["issues"]
        if issue["kind"] == "converter_step_rejected"
    ]
    assert (
        len(rejected) == 1
        and rejected[0]["terra_step"] == 1
        and rejected[0]["severity"] == "converter"
    )
    fix = rejected[0]["fix"]
    assert fix["terra_step"] == 1 and fix["pile_m3"] == pytest.approx(1.25, abs=0.05)
    assert (
        "suggested_centre_xy_m" in fix
        and "plan set-dump --step 1" in rejected[0]["text"]
    )
    # The incomplete plan's converter check lists the rejected step under the missing plan.
    converter = next(
        c for c in tmm_replay.validity(result)["checks"] if c["id"] == "converter_plan"
    )
    assert [f["kind"] for f in converter["failures"]] == [
        "no_plan",
        "converter_step_rejected",
    ]


def test_rejected_stations_whose_ground_other_stations_dig_and_dropped_last_loads_are_converter_notes(
    monkeypatch,
):
    # Lorenzo, 2 October 2026: "yes, if other stations dig it it's fine". The converter's notes on a rejected Terra
    # station that does not block the plan, and on a last load Terra never unloads, fail nothing.
    monkeypatch.setattr(
        tmm_replay,
        "runtime_checks",
        lambda case, validator=None: dict(status="pass", message=""),
    )
    case = _trench_case([[(-0.45, 6.45, 0.05)]])
    note = dict(
        pair_index=1,
        source_steps=[2, 3],
        workspace_type="excavate",
        reasons=dict(no_reachable_destination_dump_candidate=3),
        required_left_m2=0.0,
        blocks_completion=False,
        note="other workspaces dig its ground",
    )
    report = dict(
        inputs=dict(input_dir="/nonexistent"),
        complete_geometric_plan=True,
        coverage_fraction=1.0,
        continuous_required_residual_m2=0.01,
        coverage_area=dict(
            required_m2=3.5,
            covered_m2=3.49,
            residual_m2=0.01,
            covered_fraction=3.49 / 3.5,
        ),
        per_pair=[
            dict(pair_index=0, accepted=True),
            dict(
                pair_index=1,
                accepted=False,
                original_pose=[-4.5, 2.25, 0.0],
                placement_rejections=note["reasons"],
            ),
        ],
        rejected_terra_stations=[note],
        dropped_trailing_lifts=[
            dict(
                step=449,
                note="Terra's plan ends before this load is unloaded; it moves no soil",
            )
        ],
    )
    case = dataclasses.replace(
        case,
        report=report,
        native_dig=[case.completion[0], case.completion[0]],
        native_dump=[case.target] * 2,
    )
    for complete in (True, False):
        result = tmm_replay.replay(dataclasses.replace(case, complete=complete))
        converter = next(
            c
            for c in tmm_replay.validity(result)["checks"]
            if c["id"] == "converter_plan"
        )
        assert [n["kind"] for n in converter["notes"]] == [
            "converter_station_rejected",
            "trailing_lift_dropped",
        ]
        assert "other workspaces dig its ground" in converter["notes"][0]["text"]
        assert "covers 3.49 of 3.50 m2 required (99.71 %)" in converter["detail"]
        # Without a written plan the check fails on the missing plan only, not on the excused rejection.
        assert [f["kind"] for f in converter["failures"]] == (
            [] if complete else ["no_plan"]
        )
    assert (
        tmm_replay.report(result)["converter"]["coverage_area"]
        == report["coverage_area"]
    )
    # A rejection without a recorded reason says so.
    quiet = dict(report, rejected_terra_stations=[dict(note, reasons={})])
    result = tmm_replay.replay(dataclasses.replace(case, report=quiet))
    converter = next(
        c for c in tmm_replay.validity(result)["checks"] if c["id"] == "converter_plan"
    )
    assert (
        "no stance dug any of its remaining ground; other workspaces dig its ground"
        in converter["notes"][0]["text"]
    )
    # A rejection that blocks the plan stays a failure.
    blocking = dict(
        report,
        rejected_terra_stations=[
            dict(note, blocks_completion=True, note="1.00 m2 undug")
        ],
    )
    result = tmm_replay.replay(dataclasses.replace(case, report=blocking))
    converter = next(
        c for c in tmm_replay.validity(result)["checks"] if c["id"] == "converter_plan"
    )
    assert [f["kind"] for f in converter["failures"]] == [
        "no_plan",
        "converter_step_rejected",
    ]
    assert [n["kind"] for n in converter["notes"]] == ["trailing_lift_dropped"]


def test_a_cut_takes_the_design_material_of_its_whole_support_and_finishes_only_its_completion():
    # Station 1's exported support reaches 0.5 m into station 2's strip (y in [1, 1.5]). The runtime's pulls cut
    # through the whole support (cut corridor): station 1 lifts that native ground too and station 2 gets only what is
    # left. Required ground counts as finished only where a completion cut reached it.
    case = _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    case.support[0] = case.target & _grid_mask(case.shape, case.origin, -1, 1, 0.0, 1.5)
    corridor = tmm_replay.replay(case)
    completion = tmm_replay.replay(case, tmm_replay.Rules(cut="completion"))
    c1, c2 = (event["cut"]["native_cut_m3"] for event in corridor["events"])
    k1, k2 = (event["cut"]["native_cut_m3"] for event in completion["events"])
    assert k1 == pytest.approx(0.5, abs=0.01) and c1 == pytest.approx(0.75, abs=0.01)
    assert c2 == pytest.approx(k2 - 0.25, abs=0.01) and c1 + c2 == pytest.approx(
        k1 + k2, abs=0.01
    )
    assert (
        corridor["totals"]["required_left_m2"]
        == completion["totals"]["required_left_m2"]
        == 0.0
    )
    # Without station 2 the strip beyond station 1's completion is cut in the forecast, but not finished.
    alone = tmm_replay.replay(dataclasses.replace(case, sources=[0]))
    assert alone["totals"]["native_cut_m3"] == pytest.approx(0.75, abs=0.01)
    assert alone["totals"]["required_left_m2"] == pytest.approx(2.5, abs=0.05)


def test_loose_soil_under_half_a_metre_is_drivable_and_excavated_ground_is_not():
    case = _trench_case([[(0.05, 3.45, 0.05)]])
    rules = tmm_replay.Rules()
    points = tmm_replay.body_points(case.footprint, RES / 2.0)
    pose = np.array([-5.0, -5.0, 0.0])
    body = _grid_mask(case.shape, case.origin, -6.0, -4.0, -6.0, -4.0)

    def check(native, loose, excavated):
        classes = tmm_replay.hazard_classes(case, rules, native, loose, excavated)
        return tmm_replay.body_check(
            case, points, pose, classes, tmm_replay._distance(excavated, RES)
        )

    flat, none = np.zeros(case.shape), np.zeros(case.shape, dtype=bool)
    low = np.where(body, 0.3, 0.0)
    assert (
        not check(flat, low, none)["blocked"]
        and not check(flat, low, none)["unverified"]
    )
    high = np.where(body, 0.7, 0.0)
    assert (
        check(flat, high, none)["unverified"] and not check(flat, high, none)["blocked"]
    )
    assert check(flat, np.where(body, 1.2, 0.0), none)[
        "blocked"
    ]  # ROS /map height obstacle
    assert check(np.where(body, -0.5, 0.0), flat, body)["blocked"]


def test_a_route_nav2_did_not_find_is_reported_with_the_replayed_start():
    case = _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    case.routes = {
        (1, 2): dict(
            from_workspace=1, to_workspace=2, path=[], error_message="Start occupied"
        )
    }
    result = tmm_replay.replay(case)
    missing = [
        issue for issue in result["issues"] if issue["kind"] == "route_not_found"
    ]
    assert (
        len(missing) == 1
        and missing[0]["severity"] == "runtime"
        and "start is clear" in missing[0]["text"]
    )
    assert result["events"][1]["route"] == dict(
        found=False, nav2_error="Start occupied", start_blocked_here=False
    )


def test_a_route_over_spoil_below_the_map_height_passes_with_a_note(monkeypatch):
    # Every body pose stands on loose soil of 0.5 m or more, below the /map height: drivable on a route with chassis
    # balancing (Lorenzo, 29 September), still unverified under a station.
    body_check = tmm_replay.body_check
    monkeypatch.setattr(
        tmm_replay, "body_check", lambda *args: dict(body_check(*args), unverified=True)
    )
    case = _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    path = [[-4.5, 0.5, 0.0], [-4.5, 1.4, 0.0], [-4.5, 2.25, 0.0]]
    case.routes = {
        (1, 2): dict(
            from_workspace=1, to_workspace=2, path=path, path_length_m=1.75, passed=True
        )
    }
    result = tmm_replay.replay(case)
    notes = [i for i in result["issues"] if i["kind"] == "route_over_high_spoil"]
    assert (
        len(notes) == 1
        and notes[0]["severity"] == "efficiency"
        and notes[0]["poses"] == 3
    )
    assert (notes[0]["x_m"], notes[0]["y_m"]) == (
        -4.5,
        0.5,
    )  # the first pose over high spoil
    assert result["events"][1]["route"]["high_spoil_pose_indices"] == [0, 1, 2]
    checks = {check["id"]: check for check in tmm_replay.validity(result)["checks"]}
    assert (
        checks["routes"]["status"] == "pass"
        and "drivable with chassis balancing" in checks["routes"]["detail"]
    )
    assert checks["arrival"]["status"] == "unverified"


def _with_third_station(case):
    """``case`` plus a third station, an empty collection at (-4.5, 4.0): a drive a route check can leave unchecked."""
    empty = np.zeros(case.shape, dtype=bool)[None]
    return dataclasses.replace(
        case,
        kinds=case.kinds + ["collect_dumped_soil"],
        sources=case.sources + [2],
        poses=np.vstack([case.poses, [[-4.5, 4.0, 0.0]]]),
        support=np.concatenate([case.support, empty]),
        completion=np.concatenate([case.completion, empty]),
        dump_centres=np.concatenate([case.dump_centres, empty]),
        converter_blocked=np.concatenate([case.converter_blocked, empty]),
        converter_deposit=np.concatenate([case.converter_deposit, empty]),
        native_dump=case.native_dump + [empty[0]],
        native_dig=case.native_dig + [empty[0]],
    )


def test_the_route_checks_own_failure_fails_routes_and_names_the_drives_it_never_checked():
    case = _with_third_station(
        _trench_case([[(0.05, 3.45, 0.05)], [(-0.45, 6.45, 0.05)]])
    )
    touching = dict(
        passed=False, hazard_intersection_count=16, minimum_body_to_hazard_gap_m=0.0
    )
    path = [[-4.5, 0.5, 0.0], [-4.5, 1.4, 0.0], [-4.5, 2.25, 0.0]]
    case.routes = {
        (1, 2): dict(
            from_workspace=1,
            to_workspace=2,
            path=path,
            path_length_m=1.75,
            passed=False,
        )
    }
    leg = dict(
        from_workspace=1,
        to_workspace=2,
        route_found=True,
        passed=False,
        swept_body=touching,
    )
    case.navigation = dict(
        connected_internal_routes=False, failure="internal_route_failed", legs=[leg]
    )
    result = tmm_replay.replay(case)
    routes = next(
        c for c in tmm_replay.validity(result)["checks"] if c["id"] == "routes"
    )
    # The replay's own body test finds the drive clear; the checker's padded-body sweep does not, and it stopped there.
    assert routes["status"] == "fail" and [f["kind"] for f in routes["failures"]] == [
        "route_check_failed"
    ]
    failure = routes["failures"][0]
    assert (
        failure["workspace"] == 2
        and failure["step"] == result["events"][1]["steps"]["cut"]
    )
    assert (
        "padded body touches hazards at 16 poses (closest gap 0.00 m)"
        in failure["text"]
    )
    assert (
        "never checked the drive to workspace 3" in failure["text"]
        and failure["unchecked_legs"] == 1
    )
    # A drive Nav2 found no path for gets one issue with the checker's reason, not a second route_not_found.
    error = (
        'GridBasedplugin failed to plan from (0, 0) to (1, 1): "no valid path found"'
    )
    case.routes = {
        (1, 2): dict(
            from_workspace=1, to_workspace=2, path=[], error_message=error, passed=False
        )
    }
    leg = dict(
        from_workspace=1,
        to_workspace=2,
        route_found=False,
        error_message=error,
        passed=False,
    )
    case.navigation = dict(
        connected_internal_routes=False, failure="internal_route_failed", legs=[leg]
    )
    issues = [
        i for i in tmm_replay.replay(case)["issues"] if i["kind"].startswith("route_")
    ]
    assert [i["kind"] for i in issues] == ["route_check_failed"]
    assert (
        "Nav2 found no path (no valid path found)" in issues[0]["text"]
        and "start is clear" in issues[0]["text"]
    )


def test_spoil_on_the_converters_final_zone_spread_is_final():

    # Station 1's pile lies past the design's far end, off any painted final zone: soil outside the final zones.
    case = _trench_case([[(0.05, 3.45, 0.05)]])
    unspread = tmm_replay.replay(case)
    assert unspread["totals"]["temporary_spoil_left_m3"] > 0.0
    spoil = next(
        check
        for check in tmm_replay.validity(unspread)["checks"]
        if check["id"] == "spoil_at_end"
    )
    assert spoil["status"] == "fail" and all(
        f["kind"] == "temporary_spoil_left" for f in spoil["failures"]
    )
    # The converter recorded a spread allowance over the ground beside the design (report key and coverage mask).
    case.final_spread = ~case.target
    case.report = dict(case.report, final_zone_spread_m=0.571)
    result = tmm_replay.replay(case)
    assert result["totals"]["temporary_spoil_left_m3"] == 0.0
    assert not [
        issue for issue in result["issues"] if issue["kind"] == "temporary_spoil_left"
    ]
    spoil = next(
        check
        for check in tmm_replay.validity(result)["checks"]
        if check["id"] == "spoil_at_end"
    )
    assert spoil["status"] == "pass"


def test_soil_of_a_load_released_on_a_final_zone_counts_as_final_wherever_it_spreads():
    # Station 1 dumps its 0.5 m3 at (-1.45, 5.05), 4 m from its pit; a 0.6 m square final zone lies around that centre,
    # much smaller than the two cones (toe about 0.8-1.0 m).
    case = _trench_case([[(-1.45, 5.05, 0.05)]])
    zone = _grid_mask(case.shape, case.origin, -1.75, -1.15, 4.75, 5.35)
    on_zone = tmm_replay.replay(dataclasses.replace(case, final_zone=zone))
    loads = on_zone["events"][0]["dump"]["loads"]
    assert loads and all(load["final_release"] and load["pile_clear"] for load in loads)
    totals = on_zone["totals"]
    # The piles spread past the zone; that soil is final.
    assert (
        totals["final_soil_past_final_zones_m3"] > 0.1
        and totals["temporary_spoil_left_m3"] == 0.0
    )
    assert totals["loads_released_off_final"] == 0
    spoil = next(
        check
        for check in tmm_replay.validity(on_zone)["checks"]
        if check["id"] == "spoil_at_end"
    )
    assert (
        spoil["status"] == "pass"
        and "spread past the zones (counts as final)" in spoil["detail"]
    )
    # The same dump beside the zone is released off it: its soil outside the zone must be collected.
    beside = tmm_replay.replay(
        dataclasses.replace(
            case,
            final_zone=_grid_mask(case.shape, case.origin, -1.05, -0.55, 4.75, 5.35),
        )
    )
    assert all(
        not load["final_release"] for load in beside["events"][0]["dump"]["loads"]
    )
    assert beside["totals"]["temporary_spoil_left_m3"] > 0.3
    assert any(issue["kind"] == "temporary_spoil_left" for issue in beside["issues"])
    assert np.array_equal(
        beside["temporary_frames"][-1],
        beside["loose_frames"][-1] >= tmm_replay.Rules().toe_m,
    )
