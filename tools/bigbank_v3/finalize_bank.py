#!/usr/bin/env python3
"""Write the receipt block, the per-pool R2 distance sidecar and file hashes.

The sidecar follows the recipe whose ``dataset.json`` hashes to d721a8c6...
(``terra_generalist_broad_teachers_20260915/build_bank.py``): the canonical
distance protocol, one ``rows.jsonl`` record per slot (distance file sha256 and
maximum), and a ``dataset.json`` summary. ``--distance_sidecar_sha256`` is the
sha256 of that ``dataset.json`` file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

BANK = c.ROOT / "bank"
POOLED = BANK / "train_v3_generalist_512"
PROTOCOL = (c.ARTIFACTS / "terra_generalist_teachers_20260915" / "inputs" / "bank" /
            "distance_sidecar" / "distance_protocol.json")


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(c.pinned.TERRA_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemble-report", type=Path, required=True)
    parser.add_argument("--extension-receipt", type=Path, required=True)
    args = parser.parse_args()
    assemble = json.loads(args.assemble_report.read_text())
    extension = json.loads(args.extension_receipt.read_text())
    rows = c.read_jsonl(POOLED / "manifest.jsonl")

    descriptor = json.loads((POOLED / "dataset.json").read_text())
    descriptor["train_v3_generalist"] = {
        **extension,
        "per_condition": assemble["counts"],
        "slot_count": len(rows),
        "old_slots": sum(1 for r in rows if r["train_v3_origin"] == "old"),
        "new_slots": sum(1 for r in rows if r["train_v3_origin"] == "new"),
        "order": "old condition directory name (000__..039__), then old 96 in their "
                 "original slot order, then new maps in admission order",
        "source_registry_rows": {"frozen_v8_release": assemble["registry_rows_old"],
                                 "new_train": assemble["registry_rows_new"]},
        "levels": assemble["levels"],
    }
    write(POOLED / "dataset.json", descriptor)

    sidecar = BANK / "distance_sidecar"
    sidecar.mkdir(exist_ok=True)
    shutil.copyfile(PROTOCOL, sidecar / "distance_protocol.json")
    records = []
    maps = POOLED.name
    for row in rows:
        relative = f"{maps}/distance/img_{row['slot_index']}.npy"
        array = np.load(BANK / relative, allow_pickle=False)
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite distance: {relative}")
        record = {k: row[k] for k in ("map_id", "source_id", "scenario_id", "family", "split",
                                      "slot_index")}
        record.update(dataset_relative_path=maps, distance_path=relative,
                      distance_sha256=c.sha256_file(BANK / relative),
                      normalized_distance_max=float(array.max()))
        records.append(record)
    (sidecar / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    scenarios = len({r["scenario_id"] for r in rows})
    write(sidecar / "dataset.json", {
        "schema": "terra_r2_distance_sidecar_v1", "status": "passed",
        "distance_protocol": "distance_protocol.json",
        "distance_protocol_sha256": c.sha256_file(sidecar / "distance_protocol.json"),
        "rows": "rows.jsonl", "rows_sha256": c.sha256_file(sidecar / "rows.jsonl"),
        "datasets": 1, "slots": len(rows), "scenarios": scenarios,
        "scenario_counts": {"train": scenarios},
        "source_registry_sha256": c.sha256_file(BANK / "source_registry.jsonl"),
        "physical_identity_contract": (
            "train_v3_generalist_512: the finite-metadata generalist pool verbatim plus "
            "new source-disjoint maps from the same generators; no repetition"),
        "observed_global_max": max(r["normalized_distance_max"] for r in records),
    })
    sidecar_sha = c.sha256_file(sidecar / "dataset.json")
    files = sorted(p for p in BANK.rglob("*") if p.is_file() and p.name != "files.sha256")
    (BANK / "files.sha256").write_text("".join(
        f"{c.sha256_file(p)}  {p.relative_to(BANK)}\n" for p in files))
    summary = {
        "slots": len(rows),
        "scenarios": scenarios,
        "families": dict(Counter(r["family"] for r in rows)),
        "distance_sidecar_sha256": sidecar_sha,
        "dataset_json_sha256": c.sha256_file(POOLED / "dataset.json"),
        "manifest_jsonl_sha256": c.sha256_file(POOLED / "manifest.jsonl"),
        "source_registry_sha256": c.sha256_file(BANK / "source_registry.jsonl"),
        "files_sha256_sha256": c.sha256_file(BANK / "files.sha256"),
        "file_count": len(files),
        "terra_head": git("rev-parse", "HEAD"),
    }
    write(c.ROOT / "receipts" / "finalize.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
