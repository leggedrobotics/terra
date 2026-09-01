#!/usr/bin/env python3
"""Re-score the completed C0/T1 trench-alignment pilot under v2 admissibility.

The v1 gate required the PERPENDICULAR base-centre-to-section standoff to lie
in [3.5, 7.0] m on top of the yaw-parallel clause.  That band is a lateral lane
constraint and it refused the on-axis dig-ahead pose even though Terra's dig
cone already enforces working distance radially (3.64-6.50 m).  v2 drops the
band; see ``TRENCH_GATE_STANDOFF_SEMANTICS_BUG_20260901.md``.

This script re-scores the frozen u85,000 probe traces under v2 WITHOUT any new
rollout.  It reads
``tools/trench_align_pilot_u85000_receipts/probe_{c0,t1}_u085000.{npz,json}``
and prints the v1-vs-v2 verdict table.

What is exactly recoverable and what is not
------------------------------------------
The traces store, per step, the v1 gate verdict (``align_valid``), the
diagnostic-axis raw yaw and standoff (``raw_yaw_rad`` / ``raw_standoff_m``),
the number of sections with fresh cells in the selected cone
(``fresh_axis_count``), the number of v1-pose-valid sections
(``pose_valid_axis_count``) and the fresh trench cells actually removed
(``fresh_trench_cells_dug``).  They do NOT store per-cell section membership.

* v2 pose validity is v1's minus the band clause, so v2 is strictly more
  permissive: every v1-admitted step is v2-admitted.  EXACT.
* ``fresh_axis_count == 1``: every fresh cell in the cone is owned by that one
  section (a second owner would itself have fresh cells), and that section is
  the diagnostic axis, so ``raw_yaw_rad <= tol`` decides v2 exactly.  EXACT.
* ``fresh_axis_count >= 2`` and v1 refused: the junction clause needs per-cell
  ownership, which the traces lack.  AMBIGUOUS -- reported as a bracket
  (pessimistic = all ambiguous steps inadmissible, optimistic = all
  admissible), never as a point estimate.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$PWD python \
        tools/rescore_trench_pilot_v2_admissibility.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WORKTREE = Path(__file__).resolve().parents[1]
RECEIPTS = WORKTREE / "tools" / "trench_align_pilot_u85000_receipts"
DO_ACTION = 6
# terra/config.py EnvConfig.trench_dig_yaw_tolerance_rad (15 deg + slack)
YAW_TOLERANCE_RAD = 0.2619
STANDOFF_MIN_M = 3.5
STANDOFF_MAX_M = 7.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(arm: str) -> tuple[dict, dict]:
    npz = np.load(RECEIPTS / f"probe_{arm}_u085000.npz", allow_pickle=True)
    meta = json.loads((RECEIPTS / f"probe_{arm}_u085000.json").read_text())
    return {key: npz[key] for key in npz.files}, meta


def classify(arrays: dict) -> dict:
    """Per-(step, slot) masks for the v1 and v2 gate verdicts."""
    active = arrays["active"]
    applicable = arrays["align_applicable"]
    valid_v1 = arrays["align_valid"]
    fresh_axes = arrays["fresh_axis_count"]
    raw_yaw = np.nan_to_num(arrays["raw_yaw_rad"], nan=np.inf)

    yaw_ok = raw_yaw <= YAW_TOLERANCE_RAD
    single_axis = fresh_axes <= 1

    # v2 verdict, three-valued: admitted / refused / ambiguous.
    v2_admitted = valid_v1 | (applicable & single_axis & yaw_ok)
    v2_refused = applicable & ~valid_v1 & single_axis & ~yaw_ok
    v2_ambiguous = applicable & ~valid_v1 & ~single_axis
    return {
        "active": active,
        "applicable": applicable,
        "valid_v1": valid_v1,
        "v2_admitted": v2_admitted,
        "v2_refused": v2_refused,
        "v2_ambiguous": v2_ambiguous,
    }


def summarize(arm: str) -> dict:
    arrays, meta = load(arm)
    verdicts = classify(arrays)
    active = verdicts["active"]
    applicable = verdicts["applicable"]

    is_do = arrays["action"] == DO_ACTION
    empty = arrays["loaded"] == 0
    dug_fresh = arrays["fresh_trench_cells_dug"] > 0

    # An "executed fresh trench dig": a step that actually removed fresh trench
    # target cells.  This is the population the published admissible endpoint
    # counts (probe key ``invalid_fresh_do_mutated_a_trench_cell``).
    executed = active & dug_fresh
    # The attempt population: an empty excavator pressed DO with the gate
    # applicable.  For C0 (gate off) every refused attempt still dug.
    attempts = active & is_do & applicable & empty

    v1_ok = verdicts["valid_v1"]
    v2_ok = verdicts["v2_admitted"]
    v2_no = verdicts["v2_refused"]
    v2_amb = verdicts["v2_ambiguous"]

    rows = {
        "arm": arm,
        "checkpoint": meta["checkpoint"],
        "gate_enabled": bool(meta["gate_enabled"]),
        "active_steps": int(active.sum()),
        "executed_fresh_digs": int(executed.sum()),
        "executed_v1_admissible": int((executed & v1_ok).sum()),
        "executed_v1_inadmissible": int((executed & ~v1_ok).sum()),
        "executed_v2_admissible": int((executed & v2_ok).sum()),
        "executed_v2_inadmissible": int((executed & v2_no).sum()),
        "executed_v2_ambiguous": int((executed & v2_amb).sum()),
        "attempts": int(attempts.sum()),
        "attempts_v1_refused": int((attempts & ~v1_ok).sum()),
        "attempts_v1_refused_v2_admitted": int((attempts & ~v1_ok & v2_ok).sum()),
        "attempts_v1_refused_v2_refused": int((attempts & ~v1_ok & v2_no).sum()),
        "attempts_v1_refused_v2_ambiguous": int((attempts & ~v1_ok & v2_amb).sum()),
    }

    # ---- episode-level admissible exact completion ---------------------- #
    per_slot = sorted(meta["per_slot"], key=lambda row: int(row["slot_index"]))
    slot_index = arrays["slot_index"]
    order = {int(value): position for position, value in enumerate(slot_index)}
    succeeded = np.zeros(len(per_slot), dtype=bool)
    cells = []
    for row in per_slot:
        succeeded[order[int(row["slot_index"])]] = bool(row["succeeded"])
        cells.append((order[int(row["slot_index"])], row["primary_cell"]))
    cell_of = dict(cells)

    inadm_v1 = (executed & ~v1_ok).sum(axis=0)
    inadm_v2_certain = (executed & v2_no).sum(axis=0)
    amb_v2 = (executed & v2_amb).sum(axis=0)

    adm_v1 = succeeded & (inadm_v1 == 0)
    adm_v2_pess = succeeded & (inadm_v2_certain == 0) & (amb_v2 == 0)
    adm_v2_opt = succeeded & (inadm_v2_certain == 0)

    rows.update(
        {
            "episodes": int(len(per_slot)),
            "succeeded": int(succeeded.sum()),
            "admissible_v1": int(adm_v1.sum()),
            "admissible_v2_pessimistic": int(adm_v2_pess.sum()),
            "admissible_v2_optimistic": int(adm_v2_opt.sum()),
        }
    )

    # ---- decomposition of v1-refused-but-v2-admitted executed digs ------ #
    rescued = executed & ~v1_ok & v2_ok
    standoff = arrays["raw_standoff_m"][rescued]
    yaw = np.degrees(arrays["raw_yaw_rad"][rescued])
    rows["rescued_digs"] = int(rescued.sum())
    if rescued.any():
        rows["rescued_standoff_m"] = {
            "min": float(standoff.min()),
            "p10": float(np.percentile(standoff, 10)),
            "median": float(np.median(standoff)),
            "p90": float(np.percentile(standoff, 90)),
            "max": float(standoff.max()),
            "below_floor": int((standoff < STANDOFF_MIN_M).sum()),
            "above_ceiling": int((standoff > STANDOFF_MAX_M).sum()),
            "under_1m_on_the_line": int((standoff < 1.0).sum()),
        }
        rows["rescued_yaw_deg"] = {
            "max": float(yaw.max()),
            "median": float(np.median(yaw)),
        }

    # ---- per-condition admissible completion ---------------------------- #
    conditions: dict[str, list[int]] = {}
    for position in range(len(per_slot)):
        conditions.setdefault(cell_of[position], []).append(position)
    rows["per_condition"] = {
        cell: {
            "slots": len(positions),
            "succeeded": int(succeeded[positions].sum()),
            "admissible_v1": int(adm_v1[positions].sum()),
            "admissible_v2_pessimistic": int(adm_v2_pess[positions].sum()),
            "admissible_v2_optimistic": int(adm_v2_opt[positions].sum()),
        }
        for cell, positions in sorted(conditions.items())
    }
    return rows


def main() -> int:
    print("receipt hashes")
    for arm in ("c0", "t1"):
        for suffix in ("npz", "json"):
            path = RECEIPTS / f"probe_{arm}_u085000.{suffix}"
            print(f"  {path.name}  sha256 {sha256(path)}")
    print(f"yaw tolerance {YAW_TOLERANCE_RAD} rad "
          f"({np.degrees(YAW_TOLERANCE_RAD):.2f} deg); "
          f"v1 band [{STANDOFF_MIN_M}, {STANDOFF_MAX_M}] m")

    out = {}
    for arm in ("c0", "t1"):
        rows = summarize(arm)
        out[arm] = rows
        print("\n" + "=" * 72)
        print(f"ARM {arm.upper()}  gate_enabled={rows['gate_enabled']}  "
              f"{Path(rows['checkpoint']).name}")
        for key, value in rows.items():
            if key in ("per_condition", "arm", "checkpoint"):
                continue
            print(f"  {key:38s} {value}")
        print("  per-condition (slots | succeeded | adm-v1 | adm-v2 pess/opt)")
        for cell, stats in rows["per_condition"].items():
            print(
                f"    {cell:28s} {stats['slots']:3d} | {stats['succeeded']:3d} | "
                f"{stats['admissible_v1']:3d} | "
                f"{stats['admissible_v2_pessimistic']:3d}/"
                f"{stats['admissible_v2_optimistic']:3d}"
            )

    print("\n" + "=" * 72)
    print("HEADLINE: admissible exact completion on the 176-slot panel")
    print(f"{'arm':4s} {'raw':>10s} {'v1':>12s} {'v2 (pess)':>12s} {'v2 (opt)':>12s}")
    for arm in ("c0", "t1"):
        rows = out[arm]
        total = rows["episodes"]
        print(
            f"{arm:4s} {rows['succeeded']}/{total} ({100*rows['succeeded']/total:5.2f}%) "
            f"{rows['admissible_v1']}/{total} ({100*rows['admissible_v1']/total:5.2f}%) "
            f"{rows['admissible_v2_pessimistic']}/{total} "
            f"({100*rows['admissible_v2_pessimistic']/total:5.2f}%) "
            f"{rows['admissible_v2_optimistic']}/{total} "
            f"({100*rows['admissible_v2_optimistic']/total:5.2f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
