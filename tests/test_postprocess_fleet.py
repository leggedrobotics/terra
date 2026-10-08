"""Quantitative fleet traces, continuous reservations and bounded retiming."""

import copy
import json
import math
from pathlib import Path
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import shape

from terra.postprocess import fleet
from terra.postprocess import fleet_geometry as geometry


def state(x, y, load=0, *, base=0.0, cabin=0.0, shovel=False, wheel=0.0):
    return {
        "position_xy_m": [x, y],
        "base_yaw_rad": base,
        "cabin_yaw_rad": cabin,
        "load_units": load,
        "wheel_angle_rad": wheel,
        "shovel_lifted": shovel,
    }


def profile(**overrides):
    return geometry.validate_geometry(
        {
            "body_length_m": 1.0,
            "body_width_m": 0.6,
            "work_reach_m": 4.0,
            "work_half_angle_rad": 0.9,
            "clearance_m": 0.1,
            "tool_width_m": 0.4,
            **overrides,
        }
    )


def source(types=("excavator", "excavator"), starts=None, profiles=None):
    starts = starts or [state(3, 3), state(15, 15)]
    profiles = profiles or [profile(), profile()]
    return {
        "kind": fleet.SCHEMA,
        "schema_version": 1,
        "grid": {
            "resolution_m": 1.0,
            "cell_height_m": 0.1,
            "origin_xy_m": [0, 0],
            "yaw_rad": 0,
        },
        "initial_terrain": np.zeros((22, 22)).tolist(),
        "target": np.zeros((22, 22)).tolist(),
        "obstacles": np.zeros((22, 22), bool).tolist(),
        "accepted_dump_mask": np.zeros((22, 22), bool).tolist(),
        "agents": [
            {"id": identity, "type": kind, "initial_state": initial, "geometry": body}
            for identity, kind, initial, body in zip(
                ("alpha", "bravo"), types, starts, profiles
            )
        ],
        "actions": [],
    }


def action(identity, step, before, after, cells=(), code=6, order=0):
    return {
        "id": f"{identity}:{step}:{order}",
        "round": step,
        "order": order,
        "agent_id": identity,
        "action": code,
        "before": copy.deepcopy(before),
        "after": copy.deepcopy(after),
        "changed_cells": [list(cell) for cell in cells],
    }


def dual_excavator():
    result = source(
        starts=[state(3, 3.5), state(15, 15.5)],
        profiles=[profile(tool_width_m=1), profile(tool_width_m=1)],
    )
    a, b = [agent["initial_state"] for agent in result["agents"]]
    loaded_a, loaded_b = dict(a, load_units=1), dict(b, load_units=1)
    result["target"][5][3] = result["target"][17][15] = -1
    result["accepted_dump_mask"][5][4] = result["accepted_dump_mask"][17][16] = True
    result["actions"] = [
        action("alpha", 1, a, loaded_a, [(5, 3, 0, -1)]),
        action("bravo", 1, b, loaded_b, [(17, 15, 0, -1)], order=1),
        action("alpha", 2, loaded_a, a, [(5, 4, 0, 1)]),
        action("bravo", 2, loaded_b, b, [(17, 16, 0, 1)], order=1),
    ]
    return result


def test_two_excavators_keep_stable_ids_global_order_shared_terrain_and_goal(tmp_path):
    original = dual_excavator()
    result = fleet.postprocess(original)
    assert [e["agent_id"] for e in result["events"]] == [
        "alpha",
        "bravo",
        "alpha",
        "bravo",
    ]
    assert [e["previous_agent_event"] for e in result["events"]] == [
        None,
        None,
        "alpha:1:0",
        "bravo:1:1",
    ]
    assert result["events"][1]["previous_material_event"] == "alpha:1:0"
    assert len(result["cycles"]) == 2 and all(
        cycle["complete"] for cycle in result["cycles"]
    )
    assert result["report"]["goal"]["complete"]
    assert result["schedule"]["complete"]
    assert result["report"]["status"] == "GEOMETRIC_CANDIDATE"
    assert all(
        e["workspace"]["refinement"].get("residual_area_m2", 0) <= 1e-6
        for e in result["events"]
    )
    assert result["schedule"]["native_replay_required"]
    assert result["report"]["physical_execution_validated"] is False
    assert original == dual_excavator(), "input must remain untouched"
    files = fleet.write_result(result, tmp_path)
    assert json.loads(Path(files["plan"]).read_text())["source"] == result["source"]


def test_moving_partial_skid_pickups_and_shovel_setup_survive():
    initial = state(4, 5)
    result = source(("excavator", "skid"), [state(16, 16), initial])
    result["initial_terrain"][6][5] = 2
    result["accepted_dump_mask"][6][6] = True
    ready = dict(initial, shovel_lifted=True)
    first = dict(ready, position_xy_m=[4.5, 5], load_units=1)
    second = dict(ready, position_xy_m=[5.0, 5], load_units=2)
    empty = dict(second, load_units=0)
    result["actions"] = [
        action("bravo", 1, initial, ready),
        action("bravo", 2, ready, first, [(6, 5, 2, 1)], code=0),
        action("bravo", 3, first, second, [(6, 5, 1, 0)], code=0),
        action("bravo", 4, second, empty, [(6, 6, 0, 2)]),
    ]
    plan = fleet.postprocess(result)
    assert len(plan["actions"]) == 4 and len(plan["events"]) == 3
    assert plan["actions"][0]["after"]["shovel_lifted"]
    assert [event["load_delta"] for event in plan["events"]] == [1, 1, -2]
    assert plan["events"][1]["material_dependencies"] == [
        {"event_id": "initial:6:5", "units": 1.0}
    ]
    assert plan["events"][0]["preserve_motion_during_event"]
    assert (
        plan["events"][0]["workspace"]["refinement"]["status"]
        == "RECORDED_PICKUP_SWEEP"
    )
    assert plan["report"]["cleanup"]["edited_actions"] == 0
    assert plan["report"]["goal"]["complete"]


def test_closed_loop_cleanup_preserves_material_and_every_setup_action():
    original = dual_excavator()
    a = original["agents"][0]["initial_state"]
    moved = dict(a, position_xy_m=[3, 4.5])
    actions = [
        action("alpha", 0, a, moved, code=0),
        action("alpha", 1, moved, a, code=1),
    ]
    for old in original["actions"]:
        new = copy.deepcopy(old)
        new["round"] += 2
        new["id"] = "work:" + old["id"]
        actions.append(new)
    original["actions"] = actions
    plan = fleet.postprocess(original)
    assert [a["action"] for a in plan["actions"][:2]] == [7, 7]
    assert plan["report"]["cleanup"]["edited_actions"] == 2
    assert plan["report"]["cleanup"]["travel_before_m"]["alpha"] == 2
    assert plan["report"]["cleanup"]["travel_after_m"]["alpha"] == 0
    assert plan["report"]["cleanup"]["native_replay_required"]
    assert plan["report"]["final_state_preserved"]
    # Steering changes are setup, even if a later command restores the angle.
    steering = [
        action("alpha", 0, a, dict(a, wheel_angle_rad=0.2), code=2),
        action("alpha", 1, dict(a, wheel_angle_rad=0.2), a, code=3),
    ]
    cleaned, edits = fleet.clean_motion(steering)
    assert not edits and cleaned == steering


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda s: s["actions"][0]["changed_cells"][0].__setitem__(2, 1),
            "sequential terrain",
        ),
        (
            lambda s: s["actions"][0]["after"].__setitem__("load_units", 2),
            "material is not conserved",
        ),
        (
            lambda s: s["actions"][2]["before"].__setitem__("wheel_angle_rad", 0.2),
            "state is discontinuous",
        ),
        (lambda s: s["actions"][0].__setitem__("action", 7), "WAIT changes"),
        (lambda s: s["agents"][1].__setitem__("id", "alpha"), "distinct"),
    ],
)
def test_invalid_fleet_contract_fails_closed(mutation, match):
    value = dual_excavator()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        fleet.postprocess(value)


def test_bounded_retiming_repairs_crossing_routes_and_checks_waiting_peer():
    small = profile(work_reach_m=0.6, work_half_angle_rad=0.4)
    value = source(starts=[state(3, 6), state(5, 3)], profiles=[small, small])
    a, b = [agent["initial_state"] for agent in value["agents"]]
    value["actions"] = [
        action("alpha", 1, a, state(7, 6), code=0),
        action("bravo", 1, b, state(5, 8), code=0, order=1),
    ]
    result = fleet.postprocess(value)
    assert len(result["report"]["cleaned"]["intermachine_conflicts"]) == 1
    assert result["schedule"]["complete"] and result["schedule"]["scheduled_steps"] == 2
    assert all(len(step["held_agents"]) == 1 for step in result["schedule"]["steps"])
    assert result["schedule"]["native_replay_required"]
    assert result["report"]["status"] == "GEOMETRIC_CANDIDATE"
    bounded = fleet.retime_fixed_paths(
        fleet.validate_source(value), value["actions"], max_search_states=1
    )
    assert bounded["status"] == "SEARCH_LIMIT" and not bounded["complete"]


def test_fixed_path_swap_stays_unresolved_and_never_exports_partial_success():
    small = profile(work_reach_m=0.6, work_half_angle_rad=0.4)
    value = source(starts=[state(3, 5), state(7, 5)], profiles=[small, small])
    a, b = [agent["initial_state"] for agent in value["agents"]]
    value["actions"] = [
        action("alpha", 1, a, state(7, 5), code=0),
        action("bravo", 1, b, state(3, 5), code=0, order=1),
    ]
    result = fleet.postprocess(value)
    assert result["schedule"]["status"] == "FIXED_PATH_CONFLICT"
    assert result["schedule"]["steps"] == []
    assert result["report"]["status"] == "UNRESOLVED_FLEET_PLAN"


def test_work_envelopes_remain_reserved_while_empty_waiting_and_shovel_raised():
    body = profile()
    a = state(3, 5, shovel=True)
    b = state(7, 5, base=math.pi)
    left, right = geometry.reservation(a, a, body), geometry.reservation(b, b, body)
    assert left["body"].distance(right["body"]) > 2
    assert geometry.conflict(left, right, 0)["conflict"]
    # A cabin swing reserves its intermediate sector even when endpoints clear.
    turn = geometry.reservation(a, dict(a, cabin_yaw_rad=math.pi / 2), body)
    assert turn["work"].covers(
        geometry.work_polygon(dict(a, cabin_yaw_rad=math.pi / 4), body)
    )


def test_shared_cut_blocks_later_route_and_source_goal_is_not_invented():
    value = dual_excavator()
    loaded = value["actions"][0]["after"]
    value["actions"] = [
        value["actions"][0],
        action("alpha", 2, loaded, dict(loaded, position_xy_m=[5.5, 3.5]), code=0),
    ]
    result = fleet.postprocess(value)
    assert not result["report"]["goal"]["complete"]
    assert result["report"]["status"] == "INCOMPLETE_MATERIAL_GOAL"
    assert result["report"]["cleaned"]["terrain_conflicts"]
    assert not result["schedule"]["complete"]


def test_refined_radial_bands_are_full_width_inside_recorded_material_permission():
    result = fleet.postprocess(dual_excavator(), retime=False)
    workspace = result["events"][0]["workspace"]
    permission = shape(workspace["geometry"])
    assert workspace["refinement"]["lanes"]
    for lane in workspace["refinement"]["lanes"]:
        # The shared primitive admits the documented 1e-6 transport roundoff.
        assert permission.buffer(1.01e-6, join_style=2).covers(shape(lane["geometry"]))
        assert lane["width_m"] == 1
    assert result["events"][0]["changed_cells"] == [[5, 3, 0.0, -1.0]]


def test_retiming_checks_new_deposit_across_entire_peer_route():
    small = profile(work_reach_m=0.6, work_half_angle_rad=0.4)
    value = source(starts=[state(3, 3, load=1), state(3, 6)], profiles=[small, small])
    a, b = [agent["initial_state"] for agent in value["agents"]]
    value["accepted_dump_mask"][5][6] = True
    # A material effect can extend outside the arm sector (pile spread). Both
    # endpoints of the peer route are clear; its intervening sweep crosses it.
    value["actions"] = [
        action("alpha", 1, a, dict(a, load_units=0), [(5, 6, 0, 1)]),
        action("bravo", 1, b, state(7, 6), code=0, order=1),
    ]
    result = fleet.postprocess(value)
    assert result["schedule"]["complete"]
    assert result["schedule"]["scheduled_steps"] == 2
    assert result["schedule"]["steps"][0]["action_ids"] == ["bravo:1:1"]


def test_missing_tool_refinement_cannot_promote_a_complete_schedule():
    value = dual_excavator()
    del value["agents"][0]["geometry"]["tool_width_m"]
    result = fleet.postprocess(value)
    assert result["schedule"]["complete"] and result["report"]["goal"]["complete"]
    assert result["report"]["status"] == "INCOMPLETE_WORKSPACE_REFINEMENT"
    assert "alpha:1:0" in result["report"]["workspace_refinement_incomplete"]


def test_cell_centre_coverage_does_not_hide_continuous_workspace_residual():
    value = dual_excavator()
    value["agents"][0]["geometry"]["tool_width_m"] = 0.4
    result = fleet.postprocess(value)
    refinement = result["events"][0]["workspace"]["refinement"]
    assert (
        refinement["covered_cells"] == 1
        and refinement["residual_area_m2"] > refinement["residual_tolerance_m2"]
    )
    assert refinement["status"] == "RADIAL_BANDS_INCOMPLETE"
    assert result["report"]["status"] == "INCOMPLETE_WORKSPACE_REFINEMENT"


def _native_archive(value):
    """Encode a simple metric fixture as captured native arrays (one-metre cells)."""
    agents = value["agents"]
    fields = (
        "positions",
        "base_angles",
        "cabin_angles",
        "loads",
        "wheel_angles",
        "shovel_lifted",
    )

    def snapshot(states):
        return {
            "positions": [s["position_xy_m"] for s in states],
            "base_angles": [
                (s["base_yaw_rad"] - math.pi / 2) * 12 / (2 * math.pi) for s in states
            ],
            "cabin_angles": [s["cabin_yaw_rad"] * 12 / (2 * math.pi) for s in states],
            "loads": [s["load_units"] for s in states],
            "wheel_angles": [s["wheel_angle_rad"] for s in states],
            "shovel_lifted": [s["shovel_lifted"] for s in states],
        }

    states = [agent["initial_state"] for agent in agents]
    terrain = np.asarray(value["initial_terrain"])
    frames = [dict(snapshot(states), terrain=terrain.copy())]
    before, after, orders, actions = [], [], [], []
    for offset in range(0, len(value["actions"]), 2):
        before_round, after_round, order, codes = [], [], [], [7, 7]
        for command in value["actions"][offset : offset + 2]:
            slot = next(
                i
                for i, agent in enumerate(agents)
                if agent["id"] == command["agent_id"]
            )
            before_round.append(dict(snapshot(states), terrain=terrain.copy()))
            states[slot] = command["after"]
            for x, y, _, new in command["changed_cells"]:
                terrain[x, y] = new
            after_round.append(dict(snapshot(states), terrain=terrain.copy()))
            codes[slot] = command["action"]
            order.append(slot)
        before.append(before_round)
        after.append(after_round)
        actions.append(codes)
        orders.append(order)
        frames.append(dict(snapshot(states), terrain=terrain.copy()))
    data = {
        key: np.asarray([frame[key] for frame in frames])[:, None]
        for key in fields + ("terrain",)
    }
    for prefix, rows in (("before_", before), ("after_", after)):
        data.update(
            {
                prefix
                + key: np.asarray([[item[key] for item in row] for row in rows])[
                    :, None
                ]
                for key in fields + ("terrain",)
            }
        )
    data.update(
        actions=np.asarray(actions)[:, None],
        orders=np.asarray(orders)[:, None],
        case_ids=np.array([42]),
        rounds=np.array([len(actions)]),
        target=np.asarray(value["target"])[None],
        obstacles=np.asarray(value["obstacles"])[None],
        accepted_mask=np.asarray(value["accepted_dump_mask"])[None],
    )
    return data


def test_npz_import_preserves_exact_substeps_and_rejects_omitted_intermediate_state(
    tmp_path,
):
    value = dual_excavator()
    data = _native_archive(value)
    path = tmp_path / "native.npz"
    np.savez_compressed(path, **data)
    agents = [
        {key: agent[key] for key in ("id", "type", "geometry")}
        for agent in value["agents"]
    ]
    (imported,) = fleet.import_native_npz(
        path, agents=agents, tile_size_m=1, cell_height_m=0.1
    )
    assert imported["case_id"] == 42
    assert [a["agent_id"] for a in imported["actions"]] == [
        "alpha",
        "bravo",
        "alpha",
        "bravo",
    ]
    assert fleet.postprocess(imported, retime=False)["report"]["goal"]["complete"]
    data["before_terrain"][0, 0, 1, 5, 3] = 0
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="substep terrain is discontinuous"):
        fleet.import_native_npz(path, agents=agents, tile_size_m=1, cell_height_m=0.1)


@pytest.mark.parametrize("field", ["obstacles", "before_shovel_lifted", "actions"])
def test_npz_import_refuses_corrupted_numeric_values_before_coercion(tmp_path, field):
    value = dual_excavator()
    data = _native_archive(value)
    data[field] = data[field].astype(float)
    data[field].flat[0] = 2 if field != "actions" else 1.5
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(path, **data)
    agents = [
        {key: agent[key] for key in ("id", "type", "geometry")}
        for agent in value["agents"]
    ]
    with pytest.raises(ValueError, match="binary|integral"):
        fleet.import_native_npz(path, agents=agents, tile_size_m=1, cell_height_m=0.1)


def test_agent_specific_geometry_controls_reservations():
    a = state(3, 3)
    small = profile(work_reach_m=1, body_width_m=0.4)
    large = profile(work_reach_m=7, body_width_m=2)
    assert (
        geometry.work_polygon(a, large).area > 30 * geometry.work_polygon(a, small).area
    )
    assert geometry.body_polygon(a, large).area == pytest.approx(
        5 * geometry.body_polygon(a, small).area
    )
    with pytest.raises(ValueError, match="body_width_m"):
        geometry.validate_geometry({**small, "body_width_m": 0})


def test_rotating_body_fixed_tool_offset_covers_intermediate_arc():
    body = profile(work_reach_m=1, work_half_angle_rad=0.4, work_offset_xy_m=[3, 0])
    before, after = state(10, 10), state(10, 10, base=math.pi / 2)
    swept = geometry.reservation(before, after, body)
    for yaw in np.linspace(0, math.pi / 2, 11):
        assert swept["work"].covers(
            geometry.work_polygon(state(10, 10, base=yaw), body)
        )


class _NativeFixture(SimpleNamespace):
    def _replace(self, **fields):
        return _NativeFixture(**(vars(self) | fields))

    def _apply_action(self, action):
        if int(action[0]) == 7:
            return self
        assert int(action[0]) == 6
        return self.next_state[self.agent.current_agent]

    def _with_traversability_mask(self):
        return self


def _native_state(value, states, terrain):
    agent_group = namedtuple("AgentGroup", "agent_states current_agent num_agents")
    machines = []
    for slot, machine in enumerate(states):
        machines.append(
            SimpleNamespace(
                pos_base=np.asarray(machine["position_xy_m"]),
                angle_base=np.array(
                    [(machine["base_yaw_rad"] - math.pi / 2) * 12 / (2 * math.pi)]
                ),
                angle_cabin=np.array([machine["cabin_yaw_rad"] * 12 / (2 * math.pi)]),
                loaded=np.array([machine["load_units"]]),
                wheel_angle=np.array([0]),
                shovel_lifted=np.array([machine["shovel_lifted"]]),
                agent_type=np.array(
                    [0 if value["agents"][slot]["type"] == "excavator" else 2]
                ),
                action_type=np.array([0]),
            )
        )
    world = SimpleNamespace(
        action_map=SimpleNamespace(map=np.asarray(terrain).copy()),
        target_map=SimpleNamespace(map=np.asarray(value["target"])),
        padding_mask=SimpleNamespace(map=np.asarray(value["obstacles"])),
    )
    return _NativeFixture(world=world, agent=agent_group(machines, 0, 2), next_state={})


def test_fresh_native_recorder_records_substeps_only_after_exact_round_parity():
    value = dual_excavator()
    states = [agent["initial_state"] for agent in value["agents"]]
    terrain = np.asarray(value["initial_terrain"])
    initial = _native_state(value, states, terrain)
    first, second = value["actions"][:2]
    states[0] = first["after"]
    terrain[5, 3] = -1
    middle = _native_state(value, states, terrain)
    states[1] = second["after"]
    terrain[17, 15] = -1
    final = _native_state(value, states, terrain)
    initial.next_state[0] = middle
    middle.next_state[1] = final
    args = dict(
        agents=value["agents"],
        tile_size_m=1,
        cell_height_m=0.1,
        accepted_dump_mask=value["accepted_dump_mask"],
    )
    recorder = fleet.NativeFleetRecorder(initial, **args)
    recorder.record_round(
        initial, final, np.array([6, 6]), np.array([0, 1]), round_index=0
    )
    captured = recorder.source()
    assert (
        len(captured["actions"]) == 2 and captured["provenance"]["verified_rounds"] == 1
    )
    assert captured["actions"][0]["changed_cells"] == [[5, 3, 0.0, -1.0]]
    assert captured["actions"][1]["changed_cells"] == [[17, 15, 0.0, -1.0]]
    rejected = fleet.NativeFleetRecorder(initial, **args)
    with pytest.raises(ValueError, match="does not exactly match"):
        rejected.record_round(
            initial, middle, np.array([6, 6]), np.array([0, 1]), round_index=0
        )
    assert rejected.source()["actions"] == []
    # Direct callback collection cannot export an unverified partial round.
    rejected.append_substep(
        initial, middle, slot=0, action=6, round_index=0, within_round_order=0
    )
    with pytest.raises(ValueError, match="Verify the canonical"):
        rejected.source()
    with pytest.raises(ValueError, match="exactly two ordered substeps"):
        rejected.verify_round(middle)
    with pytest.raises(ValueError, match="joint-step"):
        fleet.NativeFleetRecorder(
            SimpleNamespace(world=initial.world, agent=initial.agent), **args
        )
    wrong = initial.agent.agent_states[1].agent_type.copy()
    initial.agent.agent_states[1].agent_type[0] = 2
    with pytest.raises(ValueError, match="declared agent profiles"):
        fleet.NativeFleetRecorder(initial, **args)
    initial.agent.agent_states[1].agent_type = wrong


def test_guarded_recorder_keeps_requested_effective_order_and_rejections():
    value = dual_excavator()
    states = [agent["initial_state"] for agent in value["agents"]]
    terrain = np.asarray(value["initial_terrain"])
    initial = _native_state(value, states, terrain)
    second = value["actions"][1]
    states[1] = second["after"]
    terrain[17, 15] = -1
    final = _native_state(value, states, terrain)
    initial.next_state[1] = final
    initial.env_cfg = final.env_cfg = SimpleNamespace(workspace_guard_enabled=True)
    recorder = fleet.NativeFleetRecorder(
        initial,
        agents=value["agents"],
        tile_size_m=1,
        cell_height_m=0.1,
        accepted_dump_mask=value["accepted_dump_mask"],
    )
    for kwargs, message in (
        ({}, "require effective_actions"),
        ({"effective_actions": [6, 6], "workspace_blocked": [True, False]}, "disagree"),
        ({"effective_actions": [7, 6], "workspace_blocked": [2, 0]}, "binary"),
    ):
        with pytest.raises(ValueError, match=message):
            recorder.record_round(
                initial, final, [6, 6], [1, 0], round_index=0, **kwargs
            )
        assert recorder.source()["actions"] == []
    recorder.record_round(
        initial,
        final,
        [6, 6],
        [1, 0],
        round_index=0,
        effective_actions=[7, 6],
        workspace_blocked=[True, False],
    )
    captured = recorder.source()
    assert captured["provenance"]["native_workspace_guard_enabled"]
    assert [a["agent_id"] for a in captured["actions"]] == ["bravo", "alpha"]
    assert [a["requested_action"] for a in captured["actions"]] == [6, 6]
    assert [a["effective_action"] for a in captured["actions"]] == [6, 7]
    assert [a["workspace_blocked"] for a in captured["actions"]] == [False, True]
    assert captured["actions"][1]["changed_cells"] == []
    assert captured["actions"][1]["before"] == captured["actions"][1]["after"]


def test_guard_evidence_survives_npz_import_and_round_only_is_rejected(tmp_path):
    value = dual_excavator()
    for command in value["actions"]:
        command["round"] += 1
    value["actions"][:0] = [
        action(
            agent["id"],
            0,
            agent["initial_state"],
            agent["initial_state"],
            code=7,
            order=slot,
        )
        for slot, agent in enumerate(value["agents"])
    ]
    data = _native_archive(value)
    data["effective_actions"] = data["actions"].copy()
    data["workspace_blocked"] = np.zeros_like(data["actions"], dtype=bool)
    data["actions"][0, 0] = [2, 3]
    data["workspace_blocked"][0, 0] = True
    path = tmp_path / "guarded.npz"
    np.savez_compressed(path, **data)
    agents = [
        {k: agent[k] for k in ("id", "type", "geometry")} for agent in value["agents"]
    ]
    (captured,) = fleet.import_native_npz(
        path, agents=agents, tile_size_m=1, cell_height_m=0.1
    )
    assert [a["requested_action"] for a in captured["actions"][:2]] == [2, 3]
    assert [a["effective_action"] for a in captured["actions"][:2]] == [7, 7]
    assert all(a["workspace_blocked"] for a in captured["actions"][:2])
    assert fleet.postprocess(captured)["report"]["goal"]["complete"]
    endpoints = {
        k: v for k, v in data.items() if not k.startswith(("before_", "after_"))
    }
    np.savez_compressed(path, **endpoints)
    with pytest.raises(ValueError, match="lacks exact substeps"):
        fleet.import_native_npz(path, agents=agents, tile_size_m=1, cell_height_m=0.1)


def test_guard_evidence_is_validated_and_preserved_by_cleanup():
    value = dual_excavator()
    start = value["agents"][0]["initial_state"]
    moved = dict(start, position_xy_m=[3, 4])
    loop = [
        action("alpha", 0, start, moved, code=0),
        action("alpha", 1, moved, start, code=1),
    ]
    for command in loop:
        command.update(
            requested_action=command["action"],
            effective_action=command["action"],
            workspace_blocked=False,
        )
    for command in value["actions"]:
        command["round"] += 2
        command["id"] = "work:" + command["id"]
    value["actions"][:0] = loop
    result = fleet.postprocess(value)
    assert [a["action"] for a in result["actions"][:2]] == [7, 7]
    assert [a["requested_action"] for a in result["actions"][:2]] == [0, 1]
    assert [a["effective_action"] for a in result["actions"][:2]] == [0, 1]
    bad = copy.deepcopy(value)
    bad["actions"][0]["workspace_blocked"] = True
    with pytest.raises(ValueError, match="disagrees"):
        fleet.validate_source(bad)
