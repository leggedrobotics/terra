#!/usr/bin/env python3
"""Byte-compare a p5-named reproduction shard against the P5 candidate pool."""
import json
import sys
from pathlib import Path

import numpy as np

POOL = Path("/home/lorenzo/moleworks/.artifacts/terra_p5_candidates320_full_20260801_642756cc")
FOLDERS = ("images", "occupancy", "dumpability", "actions", "distance")


def main() -> None:
    shard = Path(sys.argv[1])
    summary = {"arrays_identical": 0, "arrays_different": 0, "metadata_identical": 0,
               "metadata_different": 0, "record_identical": 0, "record_different": 0,
               "maps": 0, "differences": []}
    summary["skipped_not_in_pool"] = 0
    for path in sorted((shard / "review_metadata").glob("img_*.json")):
        index = path.stem[4:]
        if not (POOL / "review_metadata" / f"img_{index}.json").is_file():
            summary["skipped_not_in_pool"] += 1
            continue
        summary["maps"] += 1
        for folder in FOLDERS:
            a = np.load(shard / "dataset" / folder / f"img_{index}.npy", allow_pickle=False)
            b = np.load(POOL / "dataset" / folder / f"img_{index}.npy", allow_pickle=False)
            same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            summary["arrays_identical" if same else "arrays_different"] += 1
            if not same:
                summary["differences"].append(f"{folder}/img_{index}")
        a = (shard / "dataset" / "metadata" / f"trench_{index}.json").read_bytes()
        b = (POOL / "dataset" / "metadata" / f"trench_{index}.json").read_bytes()
        summary["metadata_identical" if a == b else "metadata_different"] += 1
        a = path.read_bytes()
        b = (POOL / "review_metadata" / f"img_{index}.json").read_bytes()
        if a == b:
            summary["record_identical"] += 1
        else:
            summary["record_different"] += 1
            ra, rb = json.loads(a), json.loads(b)
            keys = sorted(k for k in set(ra) | set(rb) if ra.get(k) != rb.get(k))
            summary["differences"].append(f"record img_{index}: {keys[:8]}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
