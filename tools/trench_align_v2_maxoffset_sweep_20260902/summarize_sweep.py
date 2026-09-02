#!/usr/bin/env python3
"""Aggregate the v2 max-offset coverage sweep receipts into one summary + tables.

Reads every ``*_<tag>.json`` receipt in this directory and emits ``summary.json``
plus ``README.md``.  ``off`` is the bound-disabled control (yaw-parallel only),
i.e. the 2026-09-01 v2 semantics; every other tag is a candidate bound.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TILE = 0.5714285969734192

# tag -> (metres, label).  0 = clause disabled.
ORDER = ["b114", "b115", "b171", "b172", "b200", "b229", "b286", "b343", "off"]


def bound_of(tag: str) -> float:
    return 0.0 if tag == "off" else float(tag[1:]) / 100.0


def lanes(metres: float) -> str:
    """Which integer-cell perpendicular lanes an axis-aligned section admits."""
    if metres <= 0:
        return "all"
    k = int(metres / TILE + 1e-9)
    return f"0-{k} cells"


def load(name: str):
    p = HERE / name
    if p.exists():
        return json.loads(p.read_text())
    gz = HERE / (name + ".gz")
    if gz.exists():
        import gzip
        return json.loads(gzip.decompress(gz.read_bytes()).decode())
    return None


def is_net4(condition: str) -> bool:
    return condition.startswith("trn-net4-")


def sum_over(rows, keys, pred=lambda r: True):
    out = {k: 0 for k in keys}
    for r in rows:
        if not pred(r):
            continue
        for k in keys:
            out[k] += int(r.get(k, 0))
    return out


OVERRES_KEYS = [
    "maps", "target_cells", "cells_admissible",
    "cells_reachable_any_pose", "cells_reachable_but_never_admissible",
    "inband_applicable_candidates", "veto_candidates",
    "admissible_stations", "admissible_stations_legal_terra_footprint",
]
COVER_KEYS = [
    "maps", "target_cells", "complete_any", "complete_terra_footprint",
    "complete_legacy_mirror_footprint",
]


def overres_block(doc):
    rows = doc["summary_by_condition"]
    scopes = {
        "all": sum_over(rows, OVERRES_KEYS),
        "no_net4": sum_over(rows, OVERRES_KEYS, lambda r: not is_net4(r["condition"])),
        "net4": sum_over(rows, OVERRES_KEYS, lambda r: is_net4(r["condition"])),
    }
    per_map_losses = [
        {"label": r["label"], "condition": r["condition"],
         "lost": int(r["cells_reachable_but_never_admissible"]),
         "target": int(r["target_cells"])}
        for r in doc["results"]
        if int(r["cells_reachable_but_never_admissible"]) > 0
    ]
    per_map_losses.sort(key=lambda d: -d["lost"])
    by_cond = {}
    for r in doc["summary_by_condition"]:
        by_cond[r["condition"]] = {
            "maps": r["maps"], "target_cells": r["target_cells"],
            "cells_admissible": r["cells_admissible"],
            "lost": r["cells_reachable_but_never_admissible"],
        }
    maps_with_loss = len(per_map_losses)
    return {"scopes": scopes, "by_condition": by_cond,
            "maps_with_any_loss": maps_with_loss,
            "worst_maps": per_map_losses[:20],
            "selfcheck_mismatches": sum(
                int(s["mismatches"]) for s in doc["contract"]["terra_replica_selfcheck"])}


def cover_block(doc):
    rows = doc["summary_by_condition"]
    scopes = {
        "all": sum_over(rows, COVER_KEYS),
        "no_net4": sum_over(rows, COVER_KEYS, lambda r: not is_net4(r["condition"])),
        "net4": sum_over(rows, COVER_KEYS, lambda r: is_net4(r["condition"])),
    }
    incomplete = [
        {"label": r["label"], "condition": r["condition"],
         "covered_terra": int(r["cells_from_persistent_station_terra"])
         if "cells_from_persistent_station_terra" in r else None,
         "target": int(r["target_cells"])}
        for r in doc["results"]
        if not bool(r.get("complete_terra_footprint", r.get("complete_terra", True)))
    ]
    by_cond = {r["condition"]: {"maps": r["maps"],
                                "complete_any": r["complete_any"],
                                "complete_terra": r["complete_terra_footprint"]}
               for r in rows}
    return {"scopes": scopes, "by_condition": by_cond,
            "incomplete_terra_maps": len(incomplete),
            "selfcheck_mismatches": sum(
                int(s["mismatches"]) for s in doc["contract"]["terra_replica_selfcheck"])}


def preflight_block(doc):
    c = doc["contract"]
    by_cond = {}
    for r in doc["results"]:
        cond = r["condition"]
        e = by_cond.setdefault(cond, {"maps": 0, "complete": 0, "cells": 0,
                                      "covered": 0})
        e["maps"] += 1
        e["cells"] += int(r["target_cells"])
        e["covered"] += int(r["unrestricted"]["covered"])
        e["complete"] += int(int(r["unrestricted"]["covered"]) == int(r["target_cells"]))
    return {
        "maps": int(c["maps"]),
        "incomplete_fresh_cover_count": int(c["incomplete_fresh_cover_count"]),
        "preflight_passed": bool(c["preflight_passed"]),
        "wall_seconds": c.get("wall_seconds"),
        "by_condition": by_cond,
        "incomplete_maps": c["incomplete_fresh_cover_maps"][:40],
    }


def axis_block(doc):
    main = {}
    for r in doc["summary_by_condition"]:
        if r["footprint"] != "terra":
            continue
        e = main.setdefault("terra", {"maps": 0, "complete": 0, "cells": 0,
                                      "covered": 0})
        e["maps"] += r["maps"]
        e["complete"] += r["complete_exact"]
        e["cells"] += r["target_cells"]
        e["covered"] += r["cells_covered_exact"]
    onaxis = {}
    for r in doc["onaxis_summary_by_family"]:
        if r["footprint"] != "terra" or r["blocked_model"] != "fresh":
            continue
        key = f"{r['family']}|{r['tolerance_tiles']}"
        onaxis[key] = {
            "family": r["family"], "tol_tiles": r["tolerance_tiles"],
            "maps": r["maps"], "maps_complete": r["maps_complete_exact"],
            "target_cells": r["target_cells"],
            "cells_covered": r["cells_covered_exact"],
            "sections": r["sections"],
            "sections_complete": r["sections_own_lane_complete_exact"],
        }
    return {"main_terra": main.get("terra", {}), "onaxis": onaxis,
            "selfcheck_mismatches": sum(
                int(s["mismatches"]) for s in doc["contract"]["terra_replica_selfcheck"])}


def main():
    summary = {"tile_size_m": TILE, "bounds": {}}
    for tag in ORDER:
        m = bound_of(tag)
        entry = {"tag": tag, "max_offset_m": m, "lanes_admitted": lanes(m)}
        d = load(f"overrestriction_gate_main_dev_{tag}.json")
        if d:
            entry["overres_gate_main"] = overres_block(d)
        d = load(f"overrestriction_train_v2_pooled_{tag}.json")
        if d:
            entry["overres_pooled"] = overres_block(d)
        d = load(f"station_cover_gate_main_dev_{tag}.json")
        if d:
            entry["station_cover_gate_main"] = cover_block(d)
        d = load(f"preflight_full_{tag}.json")
        if d:
            entry["preflight"] = preflight_block(d)
        d = load(f"axis_sweep_gate_main_dev_{tag}.json")
        if d:
            entry["axis_sweep_gate_main"] = axis_block(d)
        if len(entry) > 3:
            summary["bounds"][tag] = entry
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: sorted(v) for k, v in
                      [(t, e) for t, e in summary["bounds"].items()]}, indent=1))


if __name__ == "__main__":
    main()
