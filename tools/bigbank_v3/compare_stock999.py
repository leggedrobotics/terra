#!/usr/bin/env python3
"""Compare a stock ``--maps 999`` run's indices 0..319 with the 320-candidate P5 pool.

Shared-dig maps must be byte-identical (seeds do not depend on --maps). Re-rolled
maps may differ: the re-roll heading wraps modulo --maps and re-rolls are checked
against the whole salt-0 bank.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

POOL = Path("/home/lorenzo/moleworks/.artifacts/terra_p5_candidates320_full_20260801_642756cc")
FOLDERS = ("images", "occupancy", "dumpability", "actions", "distance")


def main() -> None:
    run = Path(sys.argv[1])
    pool_rows = {(r["condition_id"], int(r["map_index"])): r
                 for r in csv.DictReader((POOL / "manifest.csv").open())}
    out = {"shared_identical": 0, "shared_different": 0, "reroll_identical": 0,
           "reroll_different": 0, "different": []}
    for row in csv.DictReader((run / "manifest.csv").open()):
        index = int(row["map_index"])
        if index >= 320:
            continue
        old = pool_rows[(row["condition_id"], index)]
        a = int(row["sample_index"])
        b = int(old["sample_index"])
        same = all(np.load(run / "dataset" / f / f"img_{a}.npy").tobytes()
                   == np.load(POOL / "dataset" / f / f"img_{b}.npy").tobytes() for f in FOLDERS)
        kind = "shared" if row["shared_dig"] == "1" and old["shared_dig"] == "1" else "reroll"
        out[f"{kind}_{'identical' if same else 'different'}"] += 1
        if not same:
            out["different"].append({"map_index": index, "new_shared": row["shared_dig"],
                                     "pool_shared": old["shared_dig"]})
    print(json.dumps({"run": run.name, **out}))


if __name__ == "__main__":
    main()
