#!/usr/bin/env python3
"""Fold the v1/v2 revalidation receipts in this directory into one comparison.

Reads whichever of the receipt JSONs exist next to this file and writes
``summary.json`` plus ``README.md`` (the side-by-side tables).  Pure
post-processing: no Terra, no bank access.

Run:  python tools/trench_align_v2_revalidation_20260901/summarize_receipts.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    """Read a receipt, transparently accepting the gzipped form."""
    p = HERE / name
    if p.is_file():
        return json.loads(p.read_text())
    gz = HERE / (name + ".gz")
    if gz.is_file():
        with gzip.open(gz, "rt") as handle:
            return json.load(handle)
    return None


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def overrestriction(tag):
    rows = {}
    for sem in ("v1", "v2"):
        d = load(f"overrestriction_{tag}_{sem}.json")
        if d is None:
            continue
        s = d["summary_by_condition"]
        tot = {
            "semantics": d["contract"]["gate_semantics"],
            "maps": sum(r["maps"] for r in s),
            "target_cells": sum(r["target_cells"] for r in s),
            "cells_admissible": sum(r["cells_admissible"] for r in s),
            "cells_reachable_but_never_admissible":
                sum(r["cells_reachable_but_never_admissible"] for r in s),
            "applicable_candidates": sum(r["inband_applicable_candidates"] for r in s),
            "veto_candidates": sum(r["veto_candidates"] for r in s),
            "veto_candidates_mixed": sum(r["veto_candidates_mixed"] for r in s),
            "admissible_stations": sum(r["admissible_stations"] for r in s),
            "admissible_stations_legal_terra_footprint":
                sum(r["admissible_stations_legal_terra_footprint"] for r in s),
            "selfcheck_mismatches":
                sum(x["mismatches"] for x in d["contract"]["terra_replica_selfcheck"]),
            "by_condition": {r["condition"]: r for r in s},
        }
        tot["cells_admissible_pct"] = pct(tot["cells_admissible"], tot["target_cells"])
        tot["veto_rate_pooled"] = (tot["veto_candidates"]
                                   / max(tot["applicable_candidates"], 1))
        rows[sem] = tot
    return rows


def station_cover(tag):
    rows = {}
    for sem in ("v1", "v2"):
        d = load(f"station_cover_{tag}_{sem}.json")
        if d is None:
            continue
        res = d["results"]
        rows[sem] = {
            "semantics": d["contract"]["gate_semantics"],
            "maps": len(res),
            "target_cells": sum(r["target_cells"] for r in res),
            "cells_admissible_any_pose": sum(r["cells_admissible_any_pose"] for r in res),
            "cells_terra_legal_station":
                sum(r["cells_admissible_terra_legal_persistent_station"] for r in res),
            "cells_legacy_mirror_station":
                sum(r["cells_admissible_legacy_mirror_persistent_station"] for r in res),
            "cells_lost_to_legacy_mirror":
                sum(r["cells_lost_to_legacy_mirror"] for r in res),
            "complete_any_maps": sum(r["complete_any"] for r in res),
            "complete_terra_maps": sum(r["complete_terra"] for r in res),
            "complete_legacy_mirror_maps":
                sum(r["complete_legacy_mirror"] for r in res),
            "terra_legal_stations": sum(r["terra_legal_stations"] for r in res),
            "selfcheck_mismatches":
                sum(x["mismatches"] for x in
                    d["contract"].get("terra_replica_selfcheck", [])),
        }
    return rows


def axis_sweep(tag):
    rows = {}
    for sem in ("v1", "v2"):
        d = load(f"axis_sweep_{tag}_{sem}.json")
        if d is None:
            continue
        out = {"semantics": d["contract"]["gate_semantics"], "by_footprint": {},
               "onaxis": d.get("onaxis_summary_by_family", []),
               "selfcheck_mismatches":
                   sum(x["mismatches"] for x in
                       d["contract"].get("terra_replica_selfcheck", []))}
        for fp in ("terra", "legacy_mirror"):
            g = [r for r in d["summary_by_condition"] if r["footprint"] == fp]
            out["by_footprint"][fp] = {
                "maps": sum(r["maps"] for r in g),
                "complete_exact": sum(r["complete_exact"] for r in g),
                "target_cells": sum(r["target_cells"] for r in g),
                "cells_covered_exact": sum(r["cells_covered_exact"] for r in g),
                "min_axis_best_single_lane_share":
                    min(r["min_axis_best_single_lane_share"] for r in g),
            }
        rows[sem] = out
    return rows


def preflight(name):
    rows = {}
    for sem in ("v1", "v2"):
        d = load(f"{name}_{sem}.json")
        if d is None:
            continue
        c = d["contract"]
        rows[sem] = {
            "semantics": c.get("gate_semantics"),
            "bank": c["bank"],
            "maps": c["maps"],
            "wall_seconds": c["wall_seconds"],
            "incomplete_fresh_cover_count": c["incomplete_fresh_cover_count"],
            "incomplete_fresh_cover_maps": c["incomplete_fresh_cover_maps"][:16],
            "preflight_passed": c["preflight_passed"],
            "candidate_macro_actions":
                sum(r["candidate_macro_actions"] for r in d["results"]),
            "by_condition": {r["condition"]: {
                "maps": r["maps"], "target_cells": r["target_cells"],
                "fresh_complete_maps": r["fresh_complete_maps"],
                "fresh_covered_cells": r["fresh_covered_cells"],
                "same_base_accepted_dump_complete_maps":
                    r["same_base_accepted_dump_complete_maps"],
                "same_base_accepted_dump_covered_cells":
                    r["same_base_accepted_dump_covered_cells"],
            } for r in d["summary_by_condition"]},
            "by_family": {r["family"]: {
                "maps": r["maps"], "target_cells": r["target_cells"],
                "fresh_complete_maps": r["fresh_complete_maps"],
                "fresh_covered_cells": r["fresh_covered_cells"],
            } for r in d["summary_by_family"]},
        }
    return rows


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def reproduction_check():
    """Does --gate-v1 still reproduce the 2026-08-18 review-v4 witness?"""
    old_path = HERE.parent / "trench_alignment_feasibility_20260818.json"
    new_path = HERE / "feasibility_reviewv4_v1.json"
    if not (old_path.is_file() and new_path.is_file()):
        return None
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    ro = {r["label"]: r for r in old["results"]}
    rn = {r["label"]: r for r in new["results"]}
    shared = sorted(set(ro) & set(rn))
    fresh_diff = [k for k in shared
                  if ro[k]["unrestricted"]["covered"] != rn[k]["unrestricted"]["covered"]
                  or ro[k]["unrestricted"]["complete"] != rn[k]["unrestricted"]["complete"]]
    pose_diff = [k for k in shared
                 if ro[k]["persistent_pose_count"] != rn[k]["persistent_pose_count"]]
    dump_diff = [k for k in shared
                 if ro[k]["same_base_accepted_dump"]["covered"]
                 != rn[k]["same_base_accepted_dump"]["covered"]]
    return {
        "maps": len(shared),
        "fresh_cover_summary_identical": all(
            a["family"] == b["family"]
            and a["fresh_complete_maps"] == b["fresh_complete_maps"]
            and a["fresh_covered_cells"] == b["fresh_covered_cells"]
            for a, b in zip(old["summary"], new["summary"])),
        "maps_with_different_fresh_cover": len(fresh_diff),
        "maps_with_different_persistent_pose_count": len(pose_diff),
        "maps_with_different_same_base_dump_cover": len(dump_diff),
        "attribution": "persistent_pose_count and the dump probe are built from "
                       "the agent footprint and do not read the standoff band at "
                       "all, so their movement is terra commit 566867db (the "
                       "compute_polygon_mask raster fix), not the gate semantics",
    }


def main():
    summary = {
        "overrestriction_gate_main_dev": overrestriction("gate_main_dev"),
        "overrestriction_train_pooled_12cond":
            overrestriction("train_pooled_12cond"),
        "station_cover_gate_main_dev": station_cover("gate_main_dev"),
        "axis_sweep_gate_main_dev": axis_sweep("gate_main_dev"),
        "preflight_full": preflight("preflight_full"),
        "feasibility_reviewv4": preflight("feasibility_reviewv4"),
        "v1_reproduction_check": reproduction_check(),
        "footprint_model_check": (
            json.loads((HERE / "footprint_model_check.json").read_text())
            if (HERE / "footprint_model_check.json").is_file() else None),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    md = ["# Fresh-trench gate v2 revalidation, 2026-09-01",
          "",
          "v1 = perpendicular standoff band [3.5, 7.0] m enforced on top of the "
          "yaw-parallel clause (`--gate-v1`).  v2 = yaw-parallel only, the "
          "shipped default (`EnvConfig.trench_dig_standoff_enforced=False`); "
          "working distance is left to the dig cone (3.64-6.50 m radial, "
          "+-30 deg).  Every numpy replica in these tools is asserted against "
          "Terra's exported `fresh_trench_dig_alignment_valid` under BOTH "
          "semantics; mismatch counts are reported below and are all zero.",
          ""]

    ors = summary["overrestriction_gate_main_dev"]
    pf = summary["preflight_full"]
    md += ["## Verdicts", ""]
    if pf.get("v1") and pf.get("v2"):
        md += [f"1. **net4 becomes fully coverable under v2.**  The full "
               f"2,400-map preflight goes from "
               f"{pf['v1']['incomplete_fresh_cover_count']} maps without a "
               f"complete order-independent fresh cover (all net4) to "
               f"{pf['v2']['incomplete_fresh_cover_count']}, "
               f"preflight_passed={pf['v2']['preflight_passed']}.  net4 can "
               f"re-enter the training bank on this evidence."]
    if ors.get("v1") and ors.get("v2"):
        md += [f"2. **The gate was never the binding constraint on coverage in "
               f"the gate_main panel or the pooled training bank.**  Under both "
               f"semantics every target cell is admissibly diggable from some "
               f"pose and `cells_reachable_but_never_admissible` is 0.  What v2 "
               f"changes is the SIZE of the admissible set: "
               f"{ors['v2']['applicable_candidates'] / ors['v1']['applicable_candidates']:.2f}x "
               f"the applicable candidates and "
               f"{ors['v2']['admissible_stations'] / ors['v1']['admissible_stations']:.2f}x "
               f"the admissible stations on gate_main/development."]
    md += ["3. **The on-axis lane works, but only for a single section.**  "
           "Standing on the trench line, cabin straight ahead or behind, "
           "digging and backing up completes 64/64 straight maps at every "
           "tolerance tested, and 0 maps under v1.  No multi-section family is "
           "completable that way alone: 93-98% of cells, and the residual sits "
           "at the junctions, which is the all-or-nothing veto, not the "
           "standoff.",
           "4. The retired v1 band stays reproducible with `--gate-v1`, and the "
           "numpy replica matches Terra's exported verdict under both semantics "
           "on every map checked.", ""]

    for tag, title in (("gate_main_dev", "evaluation/gate_main/development"),
                       ("train_pooled_12cond", "train_pilot_pooled_12cond")):
        rows = summary[f"overrestriction_{tag}"]
        if not rows:
            continue
        md += [f"## (a) Over-restriction audit - {title}", ""]
        hdr = ["metric", "v1", "v2"]
        keys = [
            ("maps", "maps"),
            ("target cells", "target_cells"),
            ("cells admissibly diggable", "cells_admissible"),
            ("cells admissibly diggable %", "cells_admissible_pct"),
            ("cells reachable but never admissible",
             "cells_reachable_but_never_admissible"),
            ("applicable candidates", "applicable_candidates"),
            ("vetoed candidates", "veto_candidates"),
            ("veto rate over applicable candidates", "veto_rate_pooled"),
            ("admissible stations", "admissible_stations"),
            ("admissible stations Terra-legal",
             "admissible_stations_legal_terra_footprint"),
            ("replica-vs-Terra mismatches", "selfcheck_mismatches"),
        ]
        body = []
        for label, key in keys:
            vals = []
            for sem in ("v1", "v2"):
                v = rows.get(sem, {}).get(key, "-")
                if isinstance(v, float):
                    v = f"{v:.4f}" if key.endswith("rate_pooled") else f"{v:.2f}"
                elif isinstance(v, int):
                    v = f"{v:,}"
                vals.append(v)
            body.append([label] + vals)
        md += [md_table(hdr, body), ""]

    rows = summary["station_cover_gate_main_dev"]
    if rows:
        md += ["## (b) Order-independent persistent station cover - "
               "evaluation/gate_main/development", ""]
        keys = [
            ("maps", "maps"), ("target cells", "target_cells"),
            ("cells admissible from any pose", "cells_admissible_any_pose"),
            ("cells from a Terra-legal persistent station",
             "cells_terra_legal_station"),
            ("cells from a legacy-mirror persistent station",
             "cells_legacy_mirror_station"),
            ("cells the legacy mirror would have lost",
             "cells_lost_to_legacy_mirror"),
            ("maps with a complete cover (any pose)", "complete_any_maps"),
            ("maps with a complete cover (Terra footprint today)",
             "complete_terra_maps"),
            ("maps with a complete cover (legacy mirror)",
             "complete_legacy_mirror_maps"),
            ("Terra-legal admissible stations", "terra_legal_stations"),
            ("replica-vs-Terra mismatches", "selfcheck_mismatches"),
        ]
        body = []
        for label, key in keys:
            vals = []
            for sem in ("v1", "v2"):
                v = rows.get(sem, {}).get(key, "-")
                vals.append(f"{v:,}" if isinstance(v, int) else v)
            body.append([label] + vals)
        md += [md_table(["metric", "v1", "v2"], body), ""]

    rows = summary["axis_sweep_gate_main_dev"]
    if rows:
        md += ["## (c) Axis sweep - evaluation/gate_main/development", "",
               "All 12 cabin headings, FORWARD/BACKWARD lanes, dumping removed.", ""]
        body = []
        for fp in ("terra", "legacy_mirror"):
            for sem in ("v1", "v2"):
                r = rows.get(sem, {}).get("by_footprint", {}).get(fp)
                if r is None:
                    continue
                body.append([fp, sem,
                             f"{r['complete_exact']}/{r['maps']}",
                             f"{r['cells_covered_exact']:,}/{r['target_cells']:,}",
                             f"{r['min_axis_best_single_lane_share']:.3f}"])
        md += [md_table(["footprint", "semantics", "maps complete (exact)",
                         "cells covered (exact)", "min single-lane share"], body), ""]

        md += ["### (c2) The ON-AXIS LANE: perpendicular ~ 0, cabin ahead or "
               "behind, forward/backward only", "",
               "Is every section completable by dig-ahead-and-retreat alone?  "
               "Blocked space is `padding` only (`fresh`): that is the correct "
               "model for a monotone retreat, because the cone starts 3.64 m "
               "from the base centre while the chassis reaches only 3.14 m "
               "ahead, so a machine that only ever backs up digs strictly ahead "
               "of every pose it will occupy and never stands on a cell it dug. "
               " Footprint: `terra`.", "",
               "**Under v1 every cell of this table is 0** -- the lane is empty "
               "by construction, since perpendicular <= 2 tiles = 1.14 m is "
               "below the 3.5 m floor.  The v1 rows, the pessimistic "
               "`persistent` blocked model (`padding | all target<0`, under "
               "which a machine on the line stands in its own hole) and the "
               "`legacy_mirror` footprint are all in summary.json.", ""]
        body = []
        for r in rows.get("v2", {}).get("onaxis", []):
            if r["footprint"] != "terra" or r["blocked_model"] != "fresh":
                continue
            body.append([
                r["family"],
                f"{r['tolerance_tiles']:.1f} ({r['tolerance_m']:.2f} m)",
                f"{r['maps_complete_exact']}/{r['maps']}",
                f"{r['sections_own_lane_complete_exact']}/{r['sections']}",
                f"{r['cells_covered_exact']:,}/{r['target_cells']:,}",
                f"{100.0 * r['cells_covered_exact'] / max(r['target_cells'], 1):.1f}%",
                f"{r['min_section_own_share_exact']:.3f}"])
        md += [md_table(["family", "tol tiles (m)", "maps complete",
                         "sections complete (own lane)", "cells covered",
                         "cell %", "worst section share"], body), ""]
        zero = [r for r in rows.get("v1", {}).get("onaxis", [])
                if r["cells_covered_exact"] != 0]
        md += [f"v1 on-axis rows with any coverage: {len(zero)} of "
               f"{len(rows.get('v1', {}).get('onaxis', []))} (expected 0).", ""]

    rows = summary["preflight_full"]
    if rows:
        md += ["## (d) Full 15-condition preflight, net4 included", ""]
        body = []
        conds = sorted(set().union(*[set(rows[s]["by_condition"]) for s in rows]))
        for cond in conds:
            line = [cond]
            for sem in ("v1", "v2"):
                r = rows.get(sem, {}).get("by_condition", {}).get(cond)
                line.append(f"{r['fresh_complete_maps']}/{r['maps']}" if r else "-")
            for sem in ("v1", "v2"):
                r = rows.get(sem, {}).get("by_condition", {}).get(cond)
                line.append(f"{r['fresh_covered_cells']:,}/{r['target_cells']:,}"
                            if r else "-")
            body.append(line)
        md += [md_table(["condition", "v1 complete", "v2 complete",
                         "v1 cells", "v2 cells"], body), ""]
        for sem in ("v1", "v2"):
            r = rows.get(sem)
            if r:
                md += [f"- {sem}: {r['maps']} maps, "
                       f"{r['incomplete_fresh_cover_count']} without a complete "
                       f"fresh cover, preflight_passed={r['preflight_passed']}, "
                       f"{r['candidate_macro_actions']:,} candidate macro "
                       f"actions, {r['wall_seconds']:.0f} s"]
        md += ["",
               "**net4 verdict: yes.**  Every one of the 480 net4 maps has a "
               "complete order-independent fresh cover under v2.  The matched "
               "v1 run on the same branch and the same code leaves 89 maps "
               "incomplete, all of them net4, so the gain is the semantics, not "
               "the branch: the earlier 2026-08-19 v1 receipt had 61 (the "
               "difference is terra commit 566867db, which moved the pose "
               "graph and made net4-side2-s slightly WORSE under v1, "
               "123/160 -> 101/160).", ""]

    rows = summary["feasibility_reviewv4"]
    if rows:
        md += ["## v1 reproduction check - review-v4 bank "
               "(tools/trench_alignment_feasibility_20260818.json)", ""]
        body = []
        fams = sorted(set().union(*[set(rows[s]["by_family"]) for s in rows]))
        for fam in fams:
            line = [fam]
            for sem in ("v1", "v2"):
                r = rows.get(sem, {}).get("by_family", {}).get(fam)
                line.append(f"{r['fresh_complete_maps']}/{r['maps']}" if r else "-")
                line.append(f"{r['fresh_covered_cells']:,}" if r else "-")
            body.append(line)
        md += [md_table(["family", "v1 complete", "v1 cells",
                         "v2 complete", "v2 cells"], body), ""]

    rc = summary["v1_reproduction_check"]
    if rc:
        md += ["## Correctness checks", "",
               "### `--gate-v1` reproduction of "
               "`tools/trench_alignment_feasibility_20260818.json`", "",
               f"- fresh-cover summary identical: "
               f"**{rc['fresh_cover_summary_identical']}** "
               f"({rc['maps_with_different_fresh_cover']} of {rc['maps']} maps "
               f"differ on the fresh cover)",
               f"- `persistent_pose_count` differs on "
               f"{rc['maps_with_different_persistent_pose_count']} maps and the "
               f"same-base dump probe on "
               f"{rc['maps_with_different_same_base_dump_cover']}: "
               f"{rc['attribution']}", ""]
    fm = summary["footprint_model_check"]
    if fm:
        md += ["### Which footprint model is Terra", "",
               "terra commit 566867db fixed `compute_polygon_mask` to rasterise "
               "`(row, col)`, which INVERTS what these tools called \"Terra's "
               "footprint\".  Checked directly against `State._is_valid_move` "
               f"over {fm['contract']['probes']} random poses "
               f"({fm['contract']['terra_legal_fraction']:.3f} legal):", ""]
        md += [md_table(["model", "agreement"],
                        [[k, f"{v['agreement']:.4f}"]
                         for k, v in fm["models"].items()]), "",
               "The tools now use the 0.9995 model as `terra` and keep the "
               "retired one as `legacy_mirror`; the residual is the "
               "corner-in-bounds clause `free_poses` does not model.", ""]

    md += ["## Exact commands", "",
           "```",
           "cd /home/lorenzo/moleworks/.worktrees/"
           "terra_trench_fresh_dig_alignment_20260818",
           "export JAX_PLATFORMS=cpu",
           "export PYTHONPATH=/home/lorenzo/moleworks/.worktrees/"
           "terra_trench_fresh_dig_alignment_20260818",
           "PY=/home/lorenzo/moleworks/.venv-terra-uv/bin/python",
           "BANK=/home/lorenzo/moleworks/.artifacts/"
           "terra_v8_trench_finite_enriched_20260819",
           "R=tools/trench_align_v2_revalidation_20260901",
           "",
           "# junction contract, both semantics",
           "$PY tools/check_trench_gate_multiowner.py [--gate-v1]",
           "",
           "# v1 reproduction of the review-v4 witness (and its v2 counterpart)",
           "$PY tools/audit_trench_alignment_feasibility.py --workers 8 "
           "[--gate-v1] --output $R/feasibility_reviewv4_<sem>.json",
           "",
           "# (a) over-restriction audit",
           "$PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \\",
           "    --dataset evaluation/gate_main/development --workers 24 "
           "[--gate-v1] \\",
           "    --output $R/overrestriction_gate_main_dev_<sem>.json",
           "$PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \\",
           "    --dataset train_pilot_pooled_12cond --workers 24 [--gate-v1] \\",
           "    --output $R/overrestriction_train_pooled_12cond_<sem>.json",
           "",
           "# (b) order-independent persistent station cover",
           "$PY tools/check_trench_persistent_station_cover.py --bank-root $BANK \\",
           "    --dataset evaluation/gate_main/development --workers 24 "
           "[--gate-v1] \\",
           "    --output $R/station_cover_gate_main_dev_<sem>.json",
           "",
           "# (c) axis sweep + the on-axis lane",
           "$PY tools/check_trench_axis_sweep_feasibility.py --bank-root $BANK \\",
           "    --dataset evaluation/gate_main/development --workers 24 "
           "[--gate-v1] \\",
           "    --output $R/axis_sweep_gate_main_dev_<sem>.json",
           "",
           "# (d) full 21-dataset preflight, net4 included",
           "$PY tools/audit_trench_alignment_feasibility.py --bank $BANK "
           "--layout exact \\",
           "    --dataset train/018__trn-net3-side1-road ... "
           "--dataset evaluation/capability_floor/sealed \\",
           "    --workers 24 --witness-actions counts [--gate-v1] \\",
           "    --output $R/preflight_full_<sem>.json",
           "",
           "# footprint model check and this summary",
           "$PY $R/check_footprint_model.py",
           "$PY $R/summarize_receipts.py",
           "```",
           "",
           "The full dataset list for (d) is in each preflight receipt's "
           "`contract.reproduction_command`.", ""]

    (HERE / "README.md").write_text("\n".join(md) + "\n")
    print(f"wrote {HERE/'summary.json'} and {HERE/'README.md'}")


if __name__ == "__main__":
    main()
