#!/usr/bin/env python3
"""Generate V6-main curriculum candidates at never-used map indices.

Runs the exact generator snapshot that produced the P5 candidate pool
(``tools/map_generation_60d01307`` = ``git show 60d01307:tools/map_generation``;
its generator files are identical to 642756cc, the P5 pool revision). The
current ``tools/map_generation`` gate modules changed after the pool was built
(``terra_service`` / ``turn_dump`` dig-fire rule >=2 -> >0), so they are not used.

Per (level, map_index) this replays ``generate_condition``'s shared-dig phase
(salt 0, attempts ``0..SHARED_DIG_ATTEMPTS-1``, seed
``SeedSequence([SEED_BASE, condition_index, map_index, attempt])``) for every
sibling condition of the level. A salt-0 map depends only on the prefix-built
salt-0 dig bank entry, the layout for ``map_index`` and those seeds, so it is
byte-identical to what ``generate_curriculum_bank.py --maps N`` writes for any
``N > map_index``. When every sibling exhausts the shared-dig phase, the re-roll
phase (attempts ``SHARED_DIG_ATTEMPTS..max_attempts-1``) is replayed exactly as
``generate_condition`` does for a ``--maps N`` run (``--bank-size N`` fixes the
salt-0 bank and heading schedule); the slot is complete when all siblings land
on the same re-rolled dig. A slot with mixed shared/re-rolled siblings is
incomplete and is not expanded further, matching the complete pair-slot
admission of the accepted bank. The per-condition ``accepted_digs`` history is
shard-local; ``reroll_history_hits`` counts the only case where that could
matter (a re-rolled dig equal to an earlier accepted dig of the condition).

The stock CLI caps ``--maps`` at 999 because of its ``1000 * condition + map``
sample index. This driver names outputs with ``10000 * condition + map`` so a
level can be extended past index 999; seeds never depend on the sample index.

Salt-0 banks are built with the generator's own loop; a level whose finite
source pool runs out (``slab``: 541 sources, ``slab-lg``: 418, one source per
salt-0 entry) stops at the first exhausted index instead of raising.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

TERRA_ROOT = Path(__file__).resolve().parents[2]
PINNED = TERRA_ROOT / "tools" / "map_generation_60d01307"
sys.path.insert(0, str(PINNED))
sys.path.insert(1, str(TERRA_ROOT))

import generate_curriculum_bank as g  # noqa: E402
import generate_prototypes_v9 as v9  # noqa: E402

for _module in ("generate_curriculum_bank", "generate_prototypes_v9", "terra_service",
                "turn_dump", "terra_geom", "curriculum_taxonomy"):
    _path = Path(sys.modules[_module].__file__).resolve().parent
    if _path != PINNED:
        raise RuntimeError(f"{_module} resolved outside the pinned snapshot: {_path}")

SOURCE_FOUNDATIONS = Path(
    "/home/lorenzo/moleworks/.artifacts/terra_map_audit_20260723/full_data/"
    "foundations_dumpzones_v3"
)
SOURCE_FOUNDATIONS_SHA256 = (
    "12a137cfc2be7949e77ae115b3885d6b7b7d545679022f2530b198814af188c3"
)
EXT_SAMPLE_STRIDE = 10000


def ext_sample_index(condition_index: int, map_index: int) -> int:
    if condition_index < 0 or not 0 <= map_index < EXT_SAMPLE_STRIDE:
        raise ValueError(f"invalid condition/map index {condition_index}/{map_index}")
    return EXT_SAMPLE_STRIDE * condition_index + map_index


def level_conditions(level: str) -> list[tuple[int, object]]:
    return [
        (index, condition)
        for index, condition in enumerate(g.DATASETS["main"].conditions)
        if condition.dig_bank_level == level
    ]


def make_bank(levels: list[str], n_maps: int) -> tuple[object, dict]:
    """Salt-0 dig banks via the generator's build loop, stopping at exhaustion."""
    dataset = g.DATASETS["main"]
    spec = g.tax.RELEASES[dataset.release]
    t0_levels = frozenset(
        c.dig_bank_level for c in dataset.conditions
        if spec.condition_table[c.id][0] == 0
    )
    factory = g.GeometryFactoryV10(SOURCE_FOUNDATIONS)
    bank = g.DigBankV10(factory, n_maps, t0_levels)
    report = {}
    for level in levels:
        accepted: list = []
        rejections: Counter = Counter()
        exhausted_at = None
        for map_index in range(n_maps):
            for attempt in range(1500):
                dig, meta = bank._sample(level, map_index, 0, attempt)
                if dig is None:
                    rejections["dig_bank_construction"] += 1
                    continue
                reason = bank._acceptable(level, dig, meta, accepted)
                if reason:
                    rejections[reason] += 1
                    continue
                accepted.append((dig, meta))
                break
            else:
                exhausted_at = map_index
                break
        bank.bank[level] = accepted
        report[level] = {
            "requested": n_maps,
            "built": len(accepted),
            "exhausted_at_map_index": exhausted_at,
            "rejections": dict(rejections),
        }
    return bank, report


def shared_dig_sample(bank, dataset, condition_index, condition, map_index):
    """generate_condition's salt-0 phase for one (condition, map_index)."""
    layout = g.layout_for(condition, map_index)
    reasons: Counter = Counter()
    for attempt in range(v9.SHARED_DIG_ATTEMPTS):
        dig, dig_meta = bank.get(condition.dig_bank_level, map_index, 0)
        seed = int(
            np.random.SeedSequence(
                [g.SEED_BASE, condition_index, map_index, attempt]
            ).generate_state(1)[0]
        )
        sample, reason = g.make_map(
            condition, dataset, dig, dig_meta, layout, np.random.default_rng(seed)
        )
        if sample is None:
            reasons[reason] += 1
            continue
        sample.metadata.update(
            {
                "map_index": map_index,
                "attempt": attempt,
                "attempt_seed": seed,
                "seed_base": g.SEED_BASE,
                "condition_index": condition_index,
                "shared_dig": 1,
                "dig_sha256": g.sha256_mask(sample.target < 0),
                "occupancy_sha256": g.sha256_mask(sample.occupancy),
            }
        )
        return sample, reasons
    return None, reasons


def reroll_sample(bank, dataset, condition_index, condition, map_index, max_attempts,
                  history):
    """generate_condition's re-roll phase (salt > 0) for one (condition, map_index)."""
    layout = g.layout_for(condition, map_index)
    reasons: Counter = Counter()
    history_hits = 0
    for attempt in range(v9.SHARED_DIG_ATTEMPTS, max_attempts):
        salt = 1 + (attempt - v9.SHARED_DIG_ATTEMPTS) // v9.REROLL_DUMP_ATTEMPTS
        dig, dig_meta = bank.get(condition.dig_bank_level, map_index, salt)
        if dig is None:
            reasons["dig_reroll_exhausted"] += 1
            continue
        if any(np.array_equal(dig, other) for other in history):
            reasons["condition_dig_exact_duplicate"] += 1
            history_hits += 1
            continue
        seed = int(
            np.random.SeedSequence(
                [g.SEED_BASE, condition_index, map_index, attempt]
            ).generate_state(1)[0]
        )
        sample, reason = g.make_map(
            condition, dataset, dig, dig_meta, layout, np.random.default_rng(seed)
        )
        if sample is None:
            reasons[reason] += 1
            continue
        sample.metadata.update(
            {
                "map_index": map_index,
                "attempt": attempt,
                "attempt_seed": seed,
                "seed_base": g.SEED_BASE,
                "condition_index": condition_index,
                "shared_dig": 0,
                "dig_sha256": g.sha256_mask(sample.target < 0),
                "occupancy_sha256": g.sha256_mask(sample.occupancy),
            }
        )
        return sample, reasons, history_hits
    return None, reasons, history_hits


def run(args: argparse.Namespace) -> None:
    started = time.time()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"--output must be empty: {output}")
    for folder in (output, output / "review_metadata",
                   *(output / "dataset" / name for name in g.ARRAY_FOLDERS)):
        folder.mkdir(parents=True, exist_ok=True)
    if args.check_source:
        digest, count = g.source_pool_sha256(SOURCE_FOUNDATIONS)
        if digest != SOURCE_FOUNDATIONS_SHA256:
            raise SystemExit(f"source pool hash mismatch: {digest} ({count})")

    if args.indices:
        indices = sorted({int(value) for value in args.indices.split(",") if value})
    else:
        indices = list(range(args.start, args.stop))
    bank_size = max(indices) + 1
    if args.bank_size:
        if args.bank_size < bank_size:
            raise SystemExit("--bank-size must exceed every generated index")
        bank_size = args.bank_size
    if args.naming == "p5":
        prefix = "curriculum-diverse-320"
        g.sample_index_of = lambda ci, mi: 1000 * ci + mi
        if bank_size > 320:
            raise SystemExit("p5 naming only reproduces the 320-candidate pool")
    else:
        prefix = "curriculum-diverse-ext"
        g.sample_index_of = ext_sample_index
    dataset = dataclasses.replace(
        g.DATASETS["main"], maps_per_condition=bank_size, map_id_prefix=prefix
    )
    level = args.level
    siblings = level_conditions(level)
    if not siblings:
        raise SystemExit(f"unknown level {level}")
    # Try the historically hardest sibling first: an incomplete slot stops early.
    order = [c for c in args.order.split(",") if c] if args.order else []
    siblings.sort(key=lambda item: (order.index(item[1].id) if item[1].id in order
                                    else len(order), item[0]))

    bank, bank_report = make_bank([level], bank_size)
    built = bank_report[level]["built"]
    per_condition = {condition.id: [] for _, condition in siblings}
    history = {condition.id: [] for _, condition in siblings}
    reroll_history_hits = 0
    slots = []
    for map_index in indices:
        if map_index >= built:
            slots.append({"map_index": map_index, "status": "no_salt0_dig",
                          "reason": "source_pool_exhausted"})
            continue
        slot_started = time.time()
        found = {}
        failed = []
        for condition_index, condition in siblings:
            sample, reasons = shared_dig_sample(
                bank, dataset, condition_index, condition, map_index
            )
            if sample is None:
                failed.append((condition.id, dict(reasons.most_common(4))))
            else:
                found[condition.id] = sample
            if found and failed and not args.all_siblings:
                break
        status = "complete"
        if failed and found:
            status = "incomplete"
        elif failed:
            # every sibling left the shared dig: replay the re-roll phase
            found = {}
            for condition_index, condition in siblings:
                sample, reasons, hits = reroll_sample(
                    bank, dataset, condition_index, condition, map_index,
                    args.max_attempts, history[condition.id],
                )
                reroll_history_hits += hits
                if sample is not None:
                    found[condition.id] = sample
            digs = {s.metadata["dig_sha256"] for s in found.values()}
            status = ("complete_reroll" if len(found) == len(siblings)
                      and len(digs) == 1 else "incomplete")
        record = {
            "map_index": map_index,
            "pair_slot_id": f"{level}:{map_index}",
            "status": status,
            "seconds": round(time.time() - slot_started, 3),
            "attempts": {cid: int(s.metadata["attempt"]) for cid, s in found.items()},
        }
        if failed:
            record["salt0_failed"] = failed
        slots.append(record)
        for condition_id, sample in found.items():
            history[condition_id].append(sample.target < 0)
        if status != "incomplete" or args.keep_partial:
            for condition_id, sample in found.items():
                per_condition[condition_id].append(sample)

    rows = []
    for condition_index, condition in siblings:
        rows.extend(
            g.write_condition(output, dataset, condition, condition_index,
                              per_condition[condition.id], 0)
        )
    v9.write_terra_metadata(output, rows)
    rows.sort(key=lambda row: row["sample_index"])
    g.assert_unique_scenario_rows(rows)
    summary = {
        "schema": "terra_bigbank_v3_v6_extension_shard_v1",
        "generator_snapshot": str(PINNED),
        "level": level,
        "naming": args.naming,
        "map_id_prefix": prefix,
        "indices": [indices[0], indices[-1]] if indices else [],
        "index_count": len(indices),
        "dig_bank": bank_report,
        "conditions": [c.id for _, c in siblings],
        "bank_size": bank_size,
        "max_attempts": args.max_attempts,
        "complete_slots": sum(1 for s in slots if s["status"] == "complete"),
        "complete_reroll_slots": sum(1 for s in slots if s["status"] == "complete_reroll"),
        "reroll_history_hits": reroll_history_hits,
        "incomplete_slots": sum(1 for s in slots if s["status"] == "incomplete"),
        "no_salt0_dig_slots": sum(1 for s in slots if s["status"] == "no_salt0_dig"),
        "slots": slots,
        "maps_written": len(rows),
        "wall_seconds": round(time.time() - started, 2),
    }
    (output / "shard_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "slots"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True)
    parser.add_argument("--start", type=int, default=320)
    parser.add_argument("--stop", type=int, default=0)
    parser.add_argument("--indices", default="")
    parser.add_argument("--naming", choices=("ext", "p5"), default="ext")
    parser.add_argument("--bank-size", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=v9.MAX_ATTEMPTS)
    parser.add_argument("--order", default="")
    parser.add_argument("--all-siblings", action="store_true")
    parser.add_argument("--keep-partial", action="store_true")
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
