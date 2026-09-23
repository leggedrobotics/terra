#!/usr/bin/env python3
"""Per-condition old / new / rejected / uncoverable counts for the README."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

G = c.generator


def level_generation(level: str, siblings) -> dict:
    tried = complete = reroll = incomplete = exhausted = 0
    for shard in sorted((c.GENERATION / "v6").glob(f"{level.replace('@', '_at_')}_*")):
        summary = json.loads((shard / "shard_summary.json").read_text())
        for slot in summary["slots"]:
            tried += 1
            status = slot["status"]
            complete += status == "complete"
            reroll += status == "complete_reroll"
            incomplete += status == "incomplete"
            exhausted += status == "no_salt0_dig"
    if len(siblings) == 1:
        root = c.GENERATION / "stock999" / siblings[0][1].id
        if (root / "manifest.csv").is_file():
            rows = [r for r in csv.DictReader((root / "manifest.csv").open())
                    if int(r["map_index"]) >= 320]
            tried += 999 - 320
            complete += sum(r["shared_dig"] == "1" for r in rows)
            reroll += sum(r["shared_dig"] == "0" for r in rows)
            incomplete += (999 - 320) - len(rows)
    return {"indices_tried": tried, "complete_shared": complete, "complete_reroll": reroll,
            "incomplete_or_unsatisfied": incomplete, "no_source_left": exhausted}


def main() -> None:
    assemble = json.loads((c.ROOT / "receipts" / "assemble.json").read_text())
    stage_reports = [json.loads(p.read_text()) for p in sorted((c.ROOT / "receipts").glob("stage_*.json"))]
    rejections = {}
    for report in stage_reports:
        for level, info in report["levels"].items():
            rejections[level] = info
    by_level = {}
    for index, condition in enumerate(G.DATASETS["main"].conditions):
        by_level.setdefault(condition.dig_bank_level, []).append((index, condition))
    level_of = {condition.id: level for level, items in by_level.items() for _, condition in items}
    for control, parent in c.CONTROL_PARENTS.items():
        level_of[control] = level_of[parent]
    table = {}
    for condition, counts in assemble["counts"].items():
        level = level_of.get(condition, "v7")
        info = rejections.get(level, {})
        preflight = assemble["levels"].get(level.replace("@", "_at_"), {})
        generation = (level_generation(level, by_level[level]) if level != "v7" else
                      {"batch_per_geometry": info.get("batch_per_geometry"), "seed": info.get("seed")})
        table[condition] = {
            "level": level,
            "old": counts["old"],
            "new": counts["new"],
            "total": counts["total"],
            "generation": generation,
            "source_excluded_at_ranking": (info.get("generated_complete_slots", 0) or 0)
            - (info.get("ranked_slots", 0) or 0) if level != "v7" else 0,
            "identity_rejected": info.get("rejected", {}),
            "preflight_uncoverable_pair_slots": len(preflight.get("incomplete_pair_slots", [])),
            "preflight_audited_slots": preflight.get("preflight_slots"),
        }
    out = c.ROOT / "receipts" / "condition_counts.json"
    out.write_text(json.dumps(table, indent=1) + "\n")
    print(f"{'condition':34s} {'old':>4s} {'new':>4s} {'tot':>4s} {'tried':>5s} {'cmpl':>5s} {'rrl':>4s} {'inc':>4s} {'srcx':>4s} {'idrej':>5s} {'uncov':>5s}")
    for condition, row in table.items():
        gen = row["generation"]
        print(f"{condition:34s} {row['old']:4d} {row['new']:4d} {row['total']:4d} "
              f"{gen.get('indices_tried', 0):5d} {gen.get('complete_shared', 0):5d} "
              f"{gen.get('complete_reroll', 0):4d} {gen.get('incomplete_or_unsatisfied', 0):4d} "
              f"{row['source_excluded_at_ranking']:4d} "
              f"{sum(row['identity_rejected'].values()):5d} {row['preflight_uncoverable_pair_slots']:5d}")
    print("total", sum(r["total"] for r in table.values()), "new", sum(r["new"] for r in table.values()))


if __name__ == "__main__":
    main()
