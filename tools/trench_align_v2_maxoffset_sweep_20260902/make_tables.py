#!/usr/bin/env python3
"""Render the sweep receipts as the markdown tables in README.md."""
from __future__ import annotations

import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TILE = 0.5714285969734192


def bound_of(tag):
    return 0.0 if tag == "off" else float(tag[1:]) / 100.0


def lanes(m):
    if m <= 0:
        return "every lane"
    k = int(m / TILE + 1e-9)
    return f"0-{k} cells"


def label(tag):
    m = bound_of(tag)
    return "disabled" if m <= 0 else f"{m:.2f} m"


def tags(pattern):
    out = []
    for f in glob.glob(str(HERE / pattern)) + glob.glob(str(HERE / (pattern + ".gz"))):
        t = Path(f).name.split(".")[0].split("_")[-1]
        out.append(t)
    return sorted(set(out), key=lambda t: (bound_of(t) == 0.0, bound_of(t)))


def is_net4(c):
    return c.startswith("trn-net4-")


def load(name):
    p = HERE / name
    if p.exists():
        return json.loads(p.read_text())
    gz = HERE / (name + ".gz")
    if gz.exists():
        import gzip
        return json.loads(gzip.decompress(gz.read_bytes()).decode())
    return None


def fmt(n):
    return f"{n:,}"


def overres_table(prefix, title, scopes=("no_net4", "net4", "all")):
    lines = [f"### {title}", ""]
    header = ("| bound | lanes admitted | scope | maps | target cells | "
              "cells admissibly diggable | % | reachable-but-never-admissible | "
              "maps losing cells | applicable candidates | admissible stations | "
              "replica mismatches |")
    lines += [header, "|" + "---|" * 12]
    for t in tags(f"{prefix}_*.json"):
        d = load(f"{prefix}_{t}.json")
        if d is None:
            continue
        rows = d["summary_by_condition"]
        mism = sum(int(s["mismatches"])
                   for s in d["contract"]["terra_replica_selfcheck"])
        losers = {}
        for r in d["results"]:
            if int(r["cells_reachable_but_never_admissible"]) > 0:
                losers[r["condition"]] = losers.get(r["condition"], 0) + 1
        for scope in scopes:
            if scope == "all":
                sel = rows
                lose = sum(losers.values())
            elif scope == "net4":
                sel = [r for r in rows if is_net4(r["condition"])]
                lose = sum(v for k, v in losers.items() if is_net4(k))
            else:
                sel = [r for r in rows if not is_net4(r["condition"])]
                lose = sum(v for k, v in losers.items() if not is_net4(k))
            if not sel:
                continue
            g = lambda k: sum(int(r[k]) for r in sel)
            cells, adm = g("target_cells"), g("cells_admissible")
            lines.append(
                f"| {label(t)} | {lanes(bound_of(t))} | {scope} | {g('maps')} | "
                f"{fmt(cells)} | {fmt(adm)} | {100.0 * adm / max(cells, 1):.2f} | "
                f"{fmt(g('cells_reachable_but_never_admissible'))} | {lose} | "
                f"{fmt(g('inband_applicable_candidates'))} | "
                f"{fmt(g('admissible_stations'))} | {mism} |")
    lines.append("")
    return lines


def cover_table():
    lines = ["### (b) Order-independent persistent station cover -- "
             "evaluation/gate_main/development", ""]
    lines += ["| bound | lanes admitted | maps | complete cover, any pose | "
              "complete cover, Terra-legal PERSISTENT station | "
              "cells from a persistent station | replica mismatches |",
              "|" + "---|" * 7]
    for t in tags("station_cover_gate_main_dev_*.json"):
        d = load(f"station_cover_gate_main_dev_{t}.json")
        rows = d["summary_by_condition"]
        g = lambda k: sum(int(r[k]) for r in rows)
        cells = sum(int(r["cells_admissible_terra_legal_persistent_station"])
                    for r in d["results"])
        mism = sum(int(s["mismatches"])
                   for s in d["contract"]["terra_replica_selfcheck"])
        lines.append(
            f"| {label(t)} | {lanes(bound_of(t))} | {g('maps')} | "
            f"{g('complete_any')}/{g('maps')} | "
            f"{g('complete_terra_footprint')}/{g('maps')} | "
            f"{fmt(cells)}/{fmt(g('target_cells'))} | {mism} |")
    lines.append("")
    return lines


def preflight_table():
    lines = ["### (c) Full 2,400-map preflight (net4 included)", ""]
    lines += ["| bound | lanes admitted | maps | maps WITHOUT a complete fresh "
              "cover | preflight_passed | cells covered | wall s |",
              "|" + "---|" * 7]
    per_cond = {}
    order = tags("preflight_full_*.json")
    for t in order:
        d = load(f"preflight_full_{t}.json")
        c = d["contract"]
        cov = sum(int(r["unrestricted"]["covered"]) for r in d["results"])
        cells = sum(int(r["target_cells"]) for r in d["results"])
        lines.append(
            f"| {label(t)} | {lanes(bound_of(t))} | {c['maps']} | "
            f"{c['incomplete_fresh_cover_count']} | {c['preflight_passed']} | "
            f"{fmt(cov)}/{fmt(cells)} | {c.get('wall_seconds', 0):.0f} |")
        agg = {}
        for r in d["results"]:
            e = agg.setdefault(r["condition"], [0, 0])
            e[0] += 1
            e[1] += int(int(r["unrestricted"]["covered"]) == int(r["target_cells"]))
        per_cond[t] = agg
    lines.append("")
    conds = sorted({c for a in per_cond.values() for c in a})
    lines += ["Per condition, maps with a complete cover:", "",
              "| condition | " + " | ".join(label(t) for t in order) + " |",
              "|" + "---|" * (len(order) + 1)]
    for cond in conds:
        cells = [f"{per_cond[t][cond][1]}/{per_cond[t][cond][0]}" for t in order]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def axis_tables():
    order = tags("axis_sweep_gate_main_dev_*.json")
    lines = ["### (d) Axis sweep -- evaluation/gate_main/development", "",
             "Main sweep (all 12 cabins, FORWARD/BACKWARD lanes, dumping "
             "removed, Terra footprint, PERSISTENT blocked model):", "",
             "| bound | lanes admitted | maps complete (exact) | cells covered "
             "(exact) | replica mismatches |", "|" + "---|" * 5]
    for t in order:
        d = load(f"axis_sweep_gate_main_dev_{t}.json")
        rows = [r for r in d["summary_by_condition"] if r["footprint"] == "terra"]
        mism = sum(int(s["mismatches"])
                   for s in d["contract"]["terra_replica_selfcheck"])
        lines.append(
            f"| {label(t)} | {lanes(bound_of(t))} | "
            f"{sum(r['complete_exact'] for r in rows)}/{sum(r['maps'] for r in rows)} | "
            f"{fmt(sum(r['cells_covered_exact'] for r in rows))}/"
            f"{fmt(sum(r['target_cells'] for r in rows))} | {mism} |")
    lines += ["", "The ON-AXIS LANE (perpendicular <= 2 tiles of the axis, "
              "cabin straight ahead or behind, FORWARD/BACKWARD only, "
              "`fresh` blocked model = padding only, Terra footprint):", "",
              "| family | metric | " + " | ".join(label(t) for t in order) + " |",
              "|" + "---|" * (len(order) + 2)]
    fams = ["straight", "tee", "network", "road", "segmented"]
    data = {}
    for t in order:
        d = load(f"axis_sweep_gate_main_dev_{t}.json")
        for r in d["onaxis_summary_by_family"]:
            if r["footprint"] != "terra" or r["blocked_model"] != "fresh":
                continue
            if abs(float(r["tolerance_tiles"]) - 2.0) > 1e-9:
                continue
            data[(t, r["family"])] = r
    for fam in fams:
        for metric, key in (("maps complete", None), ("cells covered", None)):
            cells = []
            for t in order:
                r = data.get((t, fam))
                if r is None:
                    cells.append("-")
                elif metric == "maps complete":
                    cells.append(f"{r['maps_complete_exact']}/{r['maps']}")
                else:
                    cells.append(f"{r['cells_covered_exact']}/{r['target_cells']}")
            lines.append(f"| {fam} | {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def oracle_table():
    lines = ["### Scripted oracle -- 176 gate_main/development slots", "",
             "`--horizon 450 --extended-horizon 900 --verify-action-mask`, "
             "checkpoint `oracle_t1_arm_train_config_only.pkl`, terra revision "
             "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4.", "",
             "| bound | completions | rate | median steps | p90 | mean dig "
             "fraction | stations | on-axis stations | loaded-no-dump deadlocks "
             "| illegal spoil | per-step alignment checks |",
             "|" + "---|" * 11]
    for t in tags("oracle_176slot_*.json"):
        d = load(f"oracle_176slot_{t}.json")
        if d is None:
            continue
        s, c = d["summary"], d["contract"]
        med = s["median_success_step"]
        p90 = s["p90_success_step"]
        lines.append(
            f"| {label(t)} | {s['succeeded']}/176 | {s['rate']:.3f} | "
            f"{med if med is not None else '-'} | "
            f"{p90 if p90 is not None else '-'} | {s['mean_dig_fraction']:.3f} | "
            f"{fmt(s['stations'])} | {fmt(s['stations_on_axis'])} | "
            f"{s['loaded_no_legal_dump_deadlocks']} | "
            f"{s['illegal_soil_units']}u / {s['slots_with_illegal_soil']} slots | "
            f"{fmt(c['alignment_export_checks'])} |")
    lines.append("")
    return lines


def main():
    out = []
    out += overres_table("overrestriction_gate_main_dev",
                         "(a) Over-restriction audit -- evaluation/gate_main/development")
    out += overres_table("overrestriction_train_v2_pooled",
                         "(a) Over-restriction audit -- train_v2_pooled_generalist "
                         "(trench slots)")
    out += cover_table()
    out += preflight_table()
    out += axis_tables()
    out += oracle_table()
    (HERE / "tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
