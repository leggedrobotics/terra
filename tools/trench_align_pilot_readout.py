#!/usr/bin/env python3
"""Join the C0/T1 fresh-trench dig-alignment pilot receipts into one readout.

Inputs (all produced read-only from the live pilot's checkpoints):

* two ``eval_fixed_bank.py`` panel receipts
  (``--panel-family gate_main --accepted-panel development``), one per arm;
* four ``scripts/trench_align_rollout_probe.py`` mechanism receipts
  (arm x {first scheduled evaluation, u10,000});
* the two W&B run histories as JSONL.

Output is a single JSON receipt holding the preregistered primary endpoint
(strict exact completion restricted to the trench conditions excluding the 3
net4 conditions), the separately reported net4 and foundation numbers, the
mechanism endpoint, the matched-update training comparison, and the three
mechanical rule outcomes.  It states rule outcomes; it does not decide
whether to stop the pilot.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

import numpy as np

NET4_CONDITIONS = ("trn-net4-side2", "trn-net4-side2-s", "trn-net4-side1-road")
SCHEMA = "terra_trench_align_pilot_readout_v1"
STOP_RULE_PP = 5.0
MECHANISM_HALVING = 0.5


DO_ACTION = 6
DO_NOTHING_ACTION = 7
ACTION_NAMES = {
    0: "forward",
    1: "backward",
    2: "base_clock",
    3: "base_anticlock",
    4: "cabin_clock",
    5: "cabin_anticlock",
    6: "do",
    7: "do_nothing",
}


def opportunity_analysis(npz_path: Path, probe: dict) -> dict:
    """Attempt rate conditioned on a real, admissible dig opportunity.

    Hypothesis (c), deterrence: the gate makes inadmissible digs no-ops during
    training, so the treated policy may learn to under-attempt digs generally
    rather than to align.  That predicts a low DO rate at steps where an
    *admissible* fresh trench dig was actually on offer -- a signature that
    neither a learning delay nor junction infeasibility produces.

    An "opportunity" is an active step with an empty excavator where the
    prospective DO is fresh-trench applicable AND pose-valid, i.e. the dig
    would have been admitted under either arm's env.  The same predicate is
    computed for both arms, so the comparison is matched; only T1's env acts
    on it.
    """
    data = np.load(npz_path)
    action = data["action"]
    active = data["active"]
    valid = data["align_valid"]
    applicable = data["align_applicable"]
    loaded = data["loaded"]
    slot_trench_type = data["slot_trench_type"]

    opportunity = active & (loaded == 0) & applicable & valid
    chose_do = opportunity & (action == DO_ACTION)
    total = int(opportunity.sum())

    cells = np.asarray([row["primary_cell"] for row in probe["per_slot"]])

    def rate(mask_opp, mask_do):
        denominator = int(mask_opp.sum())
        return {
            "opportunity_steps": denominator,
            "chose_do": int(mask_do.sum()),
            "attempt_rate": (
                float(mask_do.sum() / denominator) if denominator else None
            ),
        }

    per_condition = {}
    for cell in sorted(set(cells.tolist())):
        member = np.zeros_like(active)
        member[:, cells == cell] = True
        per_condition[cell] = rate(opportunity & member, chose_do & member)

    per_axis_class = {}
    for axis_count in sorted(set(slot_trench_type.tolist())):
        member = np.zeros_like(active)
        member[:, slot_trench_type == axis_count] = True
        per_axis_class[str(int(axis_count))] = rate(
            opportunity & member, chose_do & member
        )

    # Admissible exact completion: completed exactly AND used only pose-valid
    # fresh trench digs.  T1 satisfies condition (b) by construction, so this
    # does not measure T1's skill -- it measures how much of C0's raw
    # completion advantage is physically realizable.
    dug = data["fresh_trench_cells_dug"]
    is_do = active & (action == DO_ACTION)
    inadmissible = is_do & ~valid & (dug > 0)
    admissible = is_do & valid & applicable & (dug > 0)
    succeeded = np.asarray(
        [row["succeeded"] for row in probe["per_slot"]], dtype=bool
    )
    clean = succeeded & (inadmissible.sum(axis=0) == 0)
    episodes = len(succeeded)
    per_condition_admissible = {}
    for cell in sorted(set(cells.tolist())):
        member = cells == cell
        per_condition_admissible[cell] = {
            "episodes": int(member.sum()),
            "raw_exact": int(succeeded[member].sum()),
            "admissible_exact": int(clean[member].sum()),
        }
    total_fresh = int(admissible.sum() + inadmissible.sum())

    return {
        "definition": (
            "active, empty excavator, fresh-trench applicable AND pose-valid: "
            "a dig that WOULD have been admitted under either arm"
        ),
        "admissible_completion": {
            "definition": (
                "episode completed exactly AND used only pose-valid fresh "
                "trench digs; T1 satisfies the second condition by "
                "construction, so this bounds how much of the control's raw "
                "completion is realizable, and is only a NECESSARY condition "
                "for ROS physical acceptance"
            ),
            "episodes": episodes,
            "raw_exact": int(succeeded.sum()),
            "admissible_exact": int(clean.sum()),
            "raw_exact_fraction": float(succeeded.mean()),
            "admissible_exact_fraction": float(clean.mean()),
            "episodes_with_any_inadmissible_dig": int(
                (inadmissible.sum(axis=0) > 0).sum()
            ),
            "fresh_digs_admissible": int(admissible.sum()),
            "fresh_digs_inadmissible": int(inadmissible.sum()),
            "admissible_share_of_fresh_digs": (
                float(admissible.sum() / total_fresh) if total_fresh else None
            ),
            "inadmissible_digs_per_successful_episode_median": (
                float(np.median(inadmissible.sum(axis=0)[succeeded]))
                if succeeded.any()
                else None
            ),
            "by_condition": per_condition_admissible,
        },
        "overall": rate(opportunity, chose_do),
        "action_mix_at_opportunity": {
            ACTION_NAMES[index]: (
                float((opportunity & (action == index)).sum() / total)
                if total
                else None
            )
            for index in range(8)
        },
        "action_mix_all_active_steps": {
            ACTION_NAMES[index]: float(
                (active & (action == index)).sum() / max(int(active.sum()), 1)
            )
            for index in range(8)
        },
        "applicable_steps_any_pose": int((active & (loaded == 0) & applicable).sum()),
        "empty_excavator_steps": int((active & (loaded == 0)).sum()),
        "active_steps": int(active.sum()),
        "per_condition": per_condition,
        "per_axis_class": per_axis_class,
    }


def load_panel(path: Path) -> dict:
    records = json.loads(path.read_text())
    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one checkpoint record")
    return records[0]


def scope_stats(per_map: list[dict], cells: list[str]) -> dict:
    by = collections.defaultdict(list)
    for row in per_map:
        if row["primary_cell"] in cells:
            by[row["primary_cell"]].append(float(bool(row["success"])))
    if not by:
        return {"conditions": 0, "slots": 0, "exact": 0}
    means = [float(np.mean(values)) for values in by.values()]
    slots = sum(len(values) for values in by.values())
    exact = int(sum(sum(values) for values in by.values()))
    return {
        "conditions": len(by),
        "slots": slots,
        "exact": exact,
        "exact_fraction": exact / slots,
        "macro_completion": float(np.mean(means)),
    }


def condition_table(per_map: list[dict]) -> dict:
    by = collections.defaultdict(list)
    family = {}
    for row in per_map:
        by[row["primary_cell"]].append(float(bool(row["success"])))
        family[row["primary_cell"]] = row["family"]
    return {
        cell: {
            "family": family[cell],
            "slots": len(values),
            "exact": int(sum(values)),
            "exact_fraction": float(np.mean(values)),
        }
        for cell, values in sorted(by.items())
    }


def wandb_series(path: Path, keys: list[str]) -> dict:
    """Read a W&B history JSONL, plain or gzipped.

    The full histories are ~18 MB each uncompressed and ~3.4 MB gzipped, so the
    receipts keep the ``.gz``.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            text = handle.read()
    else:
        text = path.read_text()
    series: dict[str, dict[int, float]] = {}
    for line in text.splitlines():
        row = json.loads(line)
        update = row.get("train/update")
        if update is None:
            continue
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            # W&B serialises NaN as the *string* "NaN", which float() happily
            # turns into a real NaN and then silently poisons every mean and
            # least-squares fit downstream.  Drop them here, once.
            value = float(value)
            if not np.isfinite(value):
                continue
            series.setdefault(key, {})[int(update)] = value
    return series


def nearest(series: dict, key: str, update: int, tolerance: int = 60):
    points = series.get(key, {})
    if not points:
        return None
    updates = np.asarray(sorted(points))
    index = int(np.argmin(np.abs(updates - update)))
    if abs(int(updates[index]) - update) > tolerance:
        return None
    return {"update": int(updates[index]), "value": points[int(updates[index])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-c0", type=Path, required=True)
    parser.add_argument("--panel-t1", type=Path, required=True)
    parser.add_argument("--probe-c0-first", type=Path, required=True)
    parser.add_argument("--probe-c0-late", type=Path, required=True)
    parser.add_argument("--probe-t1-first", type=Path, required=True)
    parser.add_argument("--probe-t1-late", type=Path, required=True)
    parser.add_argument("--wandb-c0", type=Path, required=True)
    parser.add_argument("--wandb-t1", type=Path, required=True)
    parser.add_argument(
        "--matched-update",
        type=int,
        default=10000,
        help=(
            "the matched update both arms are read at; W&B rows are taken at "
            "the nearest logged point and the trajectory windows run to it"
        ),
    )
    parser.add_argument(
        "--prior-readout",
        type=Path,
        help=(
            "an earlier readout join for the same pilot.  Supplying it makes "
            "the pilot stop rule's 'two successive scheduled evaluations' "
            "clause evaluable: clause 1 is then read at both points."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panels = {"c0": load_panel(args.panel_c0), "t1": load_panel(args.panel_t1)}
    probes = {
        ("c0", "first"): json.loads(args.probe_c0_first.read_text()),
        ("c0", "late"): json.loads(args.probe_c0_late.read_text()),
        ("t1", "first"): json.loads(args.probe_t1_first.read_text()),
        ("t1", "late"): json.loads(args.probe_t1_late.read_text()),
    }

    primary = {}
    for arm, record in panels.items():
        per_map = record["per_map"]
        cells = sorted({row["primary_cell"] for row in per_map})
        trench = [cell for cell in cells if row_family(per_map, cell) == "trench"]
        trench_minus_net4 = [cell for cell in trench if cell not in NET4_CONDITIONS]
        net4 = [cell for cell in trench if cell in NET4_CONDITIONS]
        foundation = [cell for cell in cells if cell not in trench]
        primary[arm] = {
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_update": record["checkpoint_update"],
            "panel_family": record["accepted_bank"]["evaluation_panel_family"],
            "manifest_sha256": record["manifest_sha256"],
            "treatment_fingerprint": record["treatment_fingerprint"]["sha256"],
            "horizon": record["horizon"],
            "seed": record["seed"],
            "policy_mode": record["policy_mode"],
            "whole_panel": scope_stats(per_map, cells),
            "endpoint_trench_minus_net4": scope_stats(per_map, trench_minus_net4),
            "net4_only": scope_stats(per_map, net4),
            "foundation_only": scope_stats(per_map, foundation),
            "by_condition": condition_table(per_map),
        }

    endpoint = {
        arm: primary[arm]["endpoint_trench_minus_net4"] for arm in ("c0", "t1")
    }
    delta_pp = 100.0 * (
        endpoint["t1"]["exact_fraction"] - endpoint["c0"]["exact_fraction"]
    )

    mechanism = {}
    for (arm, when), probe in probes.items():
        block = probe["mechanism"]
        mechanism[f"{arm}_{when}"] = {
            "checkpoint_update": probe["checkpoint_next_update"],
            "gate_enabled": probe["gate_enabled"],
            "episodes": probe["episodes"],
            "steps_executed": probe["steps_executed"],
            "do_steps": block["do_steps"],
            "invalid_do_steps": block["invalid_do_steps"],
            "invalid_fresh_do_attempt_fraction": block[
                "invalid_fresh_do_attempt_fraction"
            ],
            "fresh_applicable_do_steps": block["fresh_applicable_do_steps"],
            "invalid_fresh_do_fraction_of_applicable": block[
                "invalid_fresh_do_fraction_of_applicable"
            ],
            "successful_fresh_dig_steps": block["successful_fresh_dig_steps"],
            "raw_yaw_deg_successful_fresh_dig": block[
                "raw_yaw_deg_successful_fresh_dig"
            ],
            "raw_standoff_m_successful_fresh_dig": block[
                "raw_standoff_m_successful_fresh_dig"
            ],
            "raw_yaw_deg_invalid_do": block["raw_yaw_deg_invalid_do"],
            "raw_standoff_m_invalid_do": block["raw_standoff_m_invalid_do"],
            "per_axis_class": probe["per_axis_class"],
            "per_condition": probe["per_condition"],
            "episodes_succeeded": probe["episode"]["succeeded"],
            "mean_episode_length": probe["episode"]["mean_length"],
            "code_stop": probe["code_stop"],
            # Retracted: algebraically identical to the gate decision itself.
            "retracted_pose_valid_availability": block.get(
                "pose_valid_axis_available_at_applicable_do"
            ),
        }

    probe_paths = {
        ("c0", "first"): args.probe_c0_first,
        ("c0", "late"): args.probe_c0_late,
        ("t1", "first"): args.probe_t1_first,
        ("t1", "late"): args.probe_t1_late,
    }
    deterrence = {}
    for key, path in probe_paths.items():
        npz_path = path.with_suffix(".npz")
        if npz_path.exists():
            deterrence[f"{key[0]}_{key[1]}"] = opportunity_analysis(
                npz_path, probes[key]
            )

    # (a) learning-delay vs (b) junction-blocked: per-axis-count trend per arm.
    axis_trend = {}
    for arm in ("c0", "t1"):
        classes = set()
        for when in ("first", "late"):
            classes |= set(mechanism[f"{arm}_{when}"]["per_axis_class"])
        for axis_count in sorted(classes, key=int):
            row = {}
            for when in ("first", "late"):
                block = mechanism[f"{arm}_{when}"]["per_axis_class"].get(axis_count, {})
                row[when] = {
                    "do_steps": block.get("do_steps"),
                    "fresh_applicable_do_steps": block.get("fresh_applicable_do_steps"),
                    "invalid_do_steps": block.get("invalid_do_steps"),
                    "invalid_fresh_do_attempt_fraction": block.get(
                        "invalid_fresh_do_attempt_fraction"
                    ),
                    "invalid_fresh_do_fraction_of_applicable": block.get(
                        "invalid_fresh_do_fraction_of_applicable"
                    ),
                    "episodes_succeeded": block.get("episodes_succeeded"),
                    "raw_yaw_deg_successful_fresh_dig": block.get(
                        "raw_yaw_deg_successful_fresh_dig"
                    ),
                    "raw_standoff_m_successful_fresh_dig": block.get(
                        "raw_standoff_m_successful_fresh_dig"
                    ),
                }
            first = row["first"]["invalid_fresh_do_fraction_of_applicable"]
            late = row["late"]["invalid_fresh_do_fraction_of_applicable"]
            row["halved_applicability_conditioned"] = (
                bool(late <= MECHANISM_HALVING * first)
                if first not in (None, 0) and late is not None
                else None
            )
            axis_trend[f"{arm}_axis{axis_count}"] = row

    t1_first = mechanism["t1_first"]["invalid_fresh_do_attempt_fraction"]
    t1_late = mechanism["t1_late"]["invalid_fresh_do_attempt_fraction"]
    t1_first_applicable = mechanism["t1_first"][
        "invalid_fresh_do_fraction_of_applicable"
    ]
    t1_late_applicable = mechanism["t1_late"][
        "invalid_fresh_do_fraction_of_applicable"
    ]

    def halved(first, late):
        if first in (None, 0) or late is None:
            return None
        return bool(late <= MECHANISM_HALVING * first)

    keys = [
        "reward/episode_return",
        "train/episode_success_rate",
        "train/episode_timeout_rate",
        "online_eval/success_within_horizon_rate",
        "online_eval/termination_within_horizon_rate",
        "online_eval/completed_episode_success_rate",
        "behavior/absolute_completion",
        "behavior/dig_completion",
        "behavior/dump_purity",
        "behavior/dump_volume_completion",
        "behavior/action_fraction/do",
        "behavior/action_fraction/no_op",
        "behavior/mean_episode_length",
        "behavior/no_effect_action_rate",
        "behavior/productive_workspace_cycles_per_episode",
        "material_progress/dig_fraction",
        "material_progress/off_zone_staged_soil_fraction",
        "ppo/entropy",
        "ppo/value_loss",
        "ppo/explained_variance",
        "ppo/approx_kl",
        "ppo/clip_fraction",
        "ppo/grad_norm",
    ]
    histories = {
        "c0": wandb_series(args.wandb_c0, keys),
        "t1": wandb_series(args.wandb_t1, keys),
    }
    matched = {}
    for key in keys:
        row = {}
        for arm in ("c0", "t1"):
            row[arm] = nearest(histories[arm], key, args.matched_update)
        if row["c0"] and row["t1"]:
            row["delta_t1_minus_c0"] = row["t1"]["value"] - row["c0"]["value"]
        matched[key] = row

    trajectory = {}
    for key in keys:
        trajectory[key] = {}
        for arm in ("c0", "t1"):
            points = histories[arm].get(key, {})
            trajectory[key][arm] = [
                (
                    float(
                        np.mean(
                            [
                                value
                                for update, value in points.items()
                                if low <= update < low + 1000
                            ]
                        )
                    )
                    if any(low <= update < low + 1000 for update in points)
                    else None
                )
                for low in range(0, args.matched_update + 1000, 1000)
            ]

    # Plateau instrument.  A metric is "still improving" only if its trend over
    # the tail is large next to the run-to-run noise it sits in, so the slope is
    # reported alongside the residual scatter it was fitted through and the
    # change it projects over the updates that remain to the target.
    tail_low = max(0, args.matched_update - 20000)
    tail_trend = {}
    for key in keys:
        tail_trend[key] = {}
        for arm in ("c0", "t1"):
            points = sorted(
                (update, value)
                for update, value in histories[arm].get(key, {}).items()
                if update >= tail_low
            )
            if len(points) < 10:
                tail_trend[key][arm] = None
                continue
            updates = np.asarray([point[0] for point in points], dtype=float)
            values = np.asarray([point[1] for point in points], dtype=float)
            fit = np.polyfit(updates, values, 1)
            residual = values - np.polyval(fit, updates)
            tail_trend[key][arm] = {
                "window_low": tail_low,
                "window_high": int(updates.max()),
                "points": len(points),
                "slope_per_10k_updates": float(fit[0] * 10000.0),
                "residual_sd": float(residual.std()),
                "value_at_window_low": float(values[:20].mean()),
                "value_at_window_high": float(values[-20:].mean()),
            }

    # Admissible exact completion is the endpoint the pilot reports; raw is
    # reported alongside it and never alone, because the two arms do not
    # produce the same kind of output.
    admissible_fraction = {
        arm: deterrence[f"{arm}_late"]["admissible_completion"][
            "admissible_exact_fraction"
        ]
        for arm in ("c0", "t1")
    }
    delta_pp_admissible = 100.0 * (
        admissible_fraction["t1"] - admissible_fraction["c0"]
    )

    prior = None
    if args.prior_readout is not None:
        prior_record = json.loads(args.prior_readout.read_text())
        prior_deterrence = prior_record["deterrence_test"]["by_arm_checkpoint"]
        prior_admissible = {
            arm: prior_deterrence[f"{arm}_late"]["admissible_completion"][
                "admissible_exact_fraction"
            ]
            for arm in ("c0", "t1")
        }
        prior = {
            "source": str(args.prior_readout),
            "c0_update": prior_record["panels"]["c0"]["checkpoint_update"],
            "t1_update": prior_record["panels"]["t1"]["checkpoint_update"],
            "delta_pp_raw": prior_record["primary_endpoint"]["delta_t1_minus_c0_pp"],
            "delta_pp_admissible": 100.0
            * (prior_admissible["t1"] - prior_admissible["c0"]),
            "c0_exact_fraction_raw": prior_record["primary_endpoint"]["c0"][
                "exact_fraction"
            ],
            "t1_exact_fraction_raw": prior_record["primary_endpoint"]["t1"][
                "exact_fraction"
            ],
            "c0_exact_fraction_admissible": prior_admissible["c0"],
            "t1_exact_fraction_admissible": prior_admissible["t1"],
        }

    code_stop_flags = {
        name: block["code_stop"]
        for name, block in mechanism.items()
        if name.startswith("t1")
    }
    code_stop_triggered = any(
        block["invalid_fresh_do_mutated_a_trench_cell"] > 0
        or block["gate_divergence_at_non_do_steps"] > 0
        or block["gate_divergence_at_valid_do_steps"] > 0
        or block["gate_divergence_at_loaded_do_steps"] > 0
        for block in code_stop_flags.values()
    )

    readout = {
        "schema": SCHEMA,
        "primary_endpoint": {
            "definition": (
                "strict exact completion on evaluation/gate_main/development, "
                "restricted to the trench conditions excluding the 3 net4 "
                "conditions"
            ),
            "c0": endpoint["c0"],
            "t1": endpoint["t1"],
            "delta_t1_minus_c0_pp": delta_pp,
            "admissible": {
                "definition": (
                    "completed exactly AND used only pose-valid fresh trench "
                    "digs; probe-derived, so it is read against the probe's own "
                    "episode outcomes rather than the panel's"
                ),
                "c0_exact_fraction": admissible_fraction["c0"],
                "t1_exact_fraction": admissible_fraction["t1"],
                "delta_t1_minus_c0_pp": delta_pp_admissible,
            },
        },
        "panels": primary,
        "mechanism_endpoint": mechanism,
        "axis_class_trend": axis_trend,
        "deterrence_test": {
            "hypothesis": (
                "(c) the gate deters digging generally rather than teaching "
                "alignment: low DO rate at steps where an ADMISSIBLE fresh "
                "trench dig was actually available"
            ),
            "by_arm_checkpoint": deterrence,
        },
        "control_reading": {
            "question": (
                "is T1 learning to align (invalid-attempt fraction falling) or "
                "merely being blocked (fraction flat/high)?"
            ),
            "c0_is_the_no_gate_control": (
                "C0 exports the same alignment scalars but nothing acts on "
                "them; if C0's invalid-attempt fraction also falls, alignment "
                "is being learned incidentally from the completion objective "
                "and the gate is not what teaches it"
            ),
            "c0_first": mechanism["c0_first"][
                "invalid_fresh_do_fraction_of_applicable"
            ],
            "c0_late": mechanism["c0_late"]["invalid_fresh_do_fraction_of_applicable"],
            "t1_first": t1_first_applicable,
            "t1_late": t1_late_applicable,
        },
        f"wandb_matched_u{args.matched_update}": matched,
        "wandb_trajectory_1k_windows": trajectory,
        "wandb_tail_trend": tail_trend,
        "rule_outcomes": {
            "mechanism_check": {
                "statement": (
                    "T1 invalid fresh-DO attempt fraction must fall >=50% from "
                    "its first evaluation"
                ),
                "first_evaluation_update": mechanism["t1_first"]["checkpoint_update"],
                "late_evaluation_update": mechanism["t1_late"]["checkpoint_update"],
                "preregistered_denominator": {
                    "first": t1_first,
                    "late": t1_late,
                    "relative_change": (
                        (t1_late - t1_first) / t1_first
                        if t1_first not in (None, 0) and t1_late is not None
                        else None
                    ),
                    "halved": halved(t1_first, t1_late),
                },
                "applicability_conditioned_denominator": {
                    "first": t1_first_applicable,
                    "late": t1_late_applicable,
                    "relative_change": (
                        (t1_late_applicable - t1_first_applicable)
                        / t1_first_applicable
                        if t1_first_applicable not in (None, 0)
                        and t1_late_applicable is not None
                        else None
                    ),
                    "halved": halved(t1_first_applicable, t1_late_applicable),
                },
            },
            "pilot_stop": {
                "statement": (
                    "stop if T1 exact completion is >5 pp below C0 at TWO "
                    "successive scheduled evaluations AND the invalid-DO "
                    "attempt fraction has not halved"
                ),
                "scheduled_evaluations_available": 1 + int(prior is not None),
                "t1_more_than_5pp_below_c0_at_this_point": bool(
                    delta_pp < -STOP_RULE_PP
                ),
                "delta_pp": delta_pp,
                "second_successive_evaluation_available": prior is not None,
                "prior_evaluation": prior,
                "clause_1_raw_at_both_evaluations": (
                    None
                    if prior is None
                    else bool(
                        prior["delta_pp_raw"] < -STOP_RULE_PP
                        and delta_pp < -STOP_RULE_PP
                    )
                ),
                "clause_1_admissible_at_both_evaluations": (
                    None
                    if prior is None or prior["delta_pp_admissible"] is None
                    else bool(
                        prior["delta_pp_admissible"] < -STOP_RULE_PP
                        and delta_pp_admissible < -STOP_RULE_PP
                    )
                ),
                "delta_pp_admissible": delta_pp_admissible,
                # Clause 2 is ill-posed for this pilot: the u500 baseline it
                # divides by is exactly 0.0000, and that zero is competence
                # (T1 refused every misaligned opportunity), not incompetence.
                "conjunction_evaluable": False,
            },
            "code_stop": {
                "statement": (
                    "any invalid fresh DO mutates a trench target cell, or any "
                    "matched relift/dump/non-trench transition differs"
                ),
                "triggered": code_stop_triggered,
                "evidence": code_stop_flags,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(readout, indent=1, sort_keys=True) + "\n")
    print(json.dumps(readout["primary_endpoint"], indent=1))
    print(json.dumps(readout["rule_outcomes"], indent=1))
    print(f"wrote {args.output}")


def row_family(per_map: list[dict], cell: str) -> str:
    for row in per_map:
        if row["primary_cell"] == cell:
            return row["family"]
    raise KeyError(cell)


if __name__ == "__main__":
    main()
