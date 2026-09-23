#!/usr/bin/env python3
"""Write bank/README.md from the build receipts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

R = c.ROOT / "receipts"


def load(name):
    return json.loads((R / name).read_text())


def jsonl(name):
    return [json.loads(line) for line in (R / name).read_text().splitlines() if line.strip()]


def main() -> None:
    counts = load("condition_counts.json")
    audit = load("audit.json")
    loader = load("loader_check.json")
    final = load("finalize.json")
    chain = load("reproduce_old_chain.json")
    assemble = load("assemble.json")
    timings = load("timings.json")
    driver = jsonl("reproduction/generator_vs_p5_pool.jsonl")
    stock = jsonl("reproduction/stock999_prefix_vs_p5_pool.jsonl")
    d_maps = sum(r["result"]["maps"] for r in driver)
    d_arrays = sum(r["result"]["arrays_identical"] for r in driver)
    d_diff = sum(r["result"]["arrays_different"] for r in driver)
    s_same = sum(r["shared_identical"] + r["reroll_identical"] for r in stock)
    s_diff = sum(r["shared_different"] + r["reroll_different"] for r in stock)
    overlap = audit["overlap_with_all_evaluation_identities"]
    lines = []
    add = lines.append
    add("# train_v3_generalist_512")
    add("")
    add("The 40 conditions of `train_v2_pooled_generalist` with the original 96 maps per "
        "condition kept verbatim and new maps from the same generators appended. Built "
        "2026-09-23 on CPU from the Terra worktree `.worktrees/terra_gru_bigbank_20260923/terra` "
        f"(branch `gru-bigbank-20260923`, HEAD `{final['terra_head'][:10]}`).")
    add("")
    add(f"- `DATASET_SIZE={final['slots']}` (not 20,480: see Blocker), DATASET_PATH = this `bank/` "
        "directory, maps path `train_v3_generalist_512`.")
    add(f"- `--distance_sidecar_sha256 {final['distance_sidecar_sha256']}` = sha256 of "
        "`distance_sidecar/dataset.json` (per-pool R2 sidecar, same recipe as the d721a8c6... "
        "sidecar of `terra_generalist_broad_teachers_20260915/inputs/bank`).")
    add(f"- archive sha256: see `train_v3_generalist_512.tar.zst.sha256` next to the archive.")
    add("")
    add("## Blocker: slab-lineage conditions cannot reach 512")
    add("")
    add("The V6 slab builders draw OSM footprints from `foundations_dumpzones_v3` (541 slab / 418 "
        "large-slab sources) and allow one source per salt-0 dig-bank entry per level. Indices "
        "0-319 already consume 320 sources per level, so only indices 320-540 (slab) and "
        "320-417 (slab-lg) exist. After the frozen-source rule of the Train-96 builder (no source "
        "of any split of the release, no cross-level reuse) only "
        f"{counts['fnd-slab-ring3x']['new']} slab-level and {counts['fnd-slab-lg-ring3x']['new']} "
        "large-slab new pair slots remain. The 15 slab-level conditions (incl. `fnd-slab-allfree`) "
        "and `fnd-slab-lg-ring3x` therefore have fewer than 512 maps; every other condition has 512. "
        "No source pool, rule or generator was changed to close the gap.")
    add("")
    add("## Per-condition counts")
    add("")
    add("| condition | old | new | total | indices tried | complete slots | source-excluded | identity-rejected | preflight-excluded |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition, row in counts.items():
        gen = row["generation"]
        add(f"| {condition} | {row['old']} | {row['new']} | {row['total']} | "
            f"{gen.get('indices_tried', '-')} | "
            f"{(gen.get('complete_shared', 0) or 0) + (gen.get('complete_reroll', 0) or 0) if 'indices_tried' in gen else '-'} | "
            f"{row['source_excluded_at_ranking']} | {sum(row['identity_rejected'].values())} | "
            f"{row['preflight_uncoverable_pair_slots']} |")
    add("")
    add("V7 rows: one new batch `generate_scenarios(424, 2026092302)` (seed 2026092301 fails the "
        "generator's own exact-duplicate assertion at that size); 416 per condition taken in "
        "generator order after identity filtering.")
    add("")
    add("## Reproduce-before-extend")
    add("")
    add(f"- Generator (`tools/map_generation_60d01307`, byte-identical to `git show 60d01307:tools/map_generation/*.py`; "
        "generator files identical to 642756cc, the P5 pool revision): the per-index driver "
        f"regenerated {d_maps} existing candidates (all 13 levels, old-train indices, a full "
        f"0-319 tee range, re-roll slots) with {d_arrays} identical / {d_diff} different arrays, "
        "identical metadata sidecars and generator records. The stock "
        f"`generate_curriculum_bank.py --maps 999` runs reproduce indices 0-319 of the 320-pool for "
        f"7 conditions: {s_same} identical / {s_diff} different maps.")
    add(f"- Transform chain (V8 1.3 m width, capability controls, R2 distance, finite-section "
        f"enrichment) re-derived {chain['maps']} current-bank maps from generator outputs "
        f"(238 V6/control maps across 34 conditions + all 576 V7 maps from "
        f"`generate_scenarios(96, 2026080801)`): {chain['arrays_identical']}/{chain['maps']} "
        "identical in all five arrays, sidecars and manifest rows.")
    add("")
    add("## Admission")
    add("")
    add("New V6 maps: complete pair slots at never-used indices (>= 320; net3 to 1099, net4 to 1289, "
        "other multi-sibling levels to 998, slab to 540, slab-lg to 417), ranked as in "
        "`build_train96_capability_floor_bank._select_additions` (least-supported level first, "
        "stable hash of the pair-slot id, new dihedral shapes first, source-disjoint from every "
        "source of the release and across levels). Every map then passes the exact chain and is "
        "dropped if its final dig raster, source/parent source or reset-array identity collides "
        "with any evaluation identity, any release identity or another new slot.")
    add("")
    add("Trench preflight (`tools/audit_trench_alignment_feasibility.py`, b2a15ddc) was run on every "
        "staged trench slot twice: with the 2122b2df defaults (v2, on-the-line 2.0 m) and "
        "yaw-parallel only (`--max-offset-m 0`), the v2 contract under which the current bank was "
        "re-admitted (2400/2400 complete, 2026-09-01). Exclusion uses the yaw-only verdict. Under "
        "the defaults the tool's persistent-blocking model (all target cells blocked) fails most "
        "old and new maps alike; the counts are in `receipts/` and the build report.")
    add("")
    add("## Disjointness")
    add("")
    add(f"Evaluation universe: {audit['eval_universe_sizes']}. Overlap of the new maps: "
        f"{overlap['new']}. Old maps: {overlap['old']}.")
    add("")
    add("## Loader")
    add("")
    add(f"`TerraEnvBatch` through the trainer's preset path (trench_align_v2_generalist_gen, maps "
        f"path swapped), R2 protocol, CPU: {loader['host_load_seconds_terra_env_batch']} s, "
        f"MapsBuffer arrays {loader['maps_buffer_array_bytes']} bytes, finite-metadata preflight passed.")
    add("")
    add("## Timings")
    add("")
    for key, value in timings.items():
        add(f"- {key}: {value}")
    add("")
    add("Receipts, generator runs and staged candidates: "
        "`/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/{receipts,generation,candidates}`.")
    (c.ROOT / "bank" / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
