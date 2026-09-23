#!/usr/bin/env python3
"""Build an exact-contract relocation bank for solo skid-steer R2 training.

Each map has one accepted dump zone, loose soil piles, optional obstacles and
no dig target. A map is kept only if every soil cell lies in the bucket of
some collision-free chassis pose, with the skid steer's footprint and bucket
taken from Terra for all 12 headings (soil blocks the chassis, so this is
conservative; motion connectivity is not checked).

Strata:
  single     no obstacles, 1-2 piles, 16-48 units, height 1 (one load)
  harder     1-2 obstacles, 3 piles, 40-50 units, height 1 (generate_relocations_harder)
  multitrip  0-2 obstacles, 2-4 piles, 60-150 units, heights 1-2 (2-3 loads)

  python tools/build_skid_relocation_bank.py --output BANK --seed 0 \
      --split train --strata single:512 harder:768 multitrip:768
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from terra.config import REWARD_V2_DISTANCE_BOUND
from terra.config import REWARD_V2_DISTANCE_REF_M
from terra.env_generation.distance import REWARD_V2_DISTANCE_METRIC
from terra.env_generation.distance import REWARD_V2_DISTANCE_NORMALIZATION
from terra.env_generation.distance import REWARD_V2_DISTANCE_PROTOCOL_ID
from terra.env_generation.distance import compute_reward_v2_distance_map
from terra.maps_buffer import RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT
from terra.maps_buffer import contained_dump_capacity_sanity_check
from terra.maps_buffer import reset_array_scenario_sha256

SHAPE = (64, 64)
TILE_SIZE_M = 0.571428571428125  # 36.57 m / 64, as in the training banks
EDGE = 6  # soil keeps this many cells from the map border
STRATA = {
    # obstacles, piles, total units, max pile height, zone edge
    "single": dict(obstacles=(0, 0), piles=(1, 2), units=(16, 48), height=1, zone=(12, 12)),
    "harder": dict(obstacles=(1, 2), piles=(3, 3), units=(40, 50), height=1, zone=(12, 12)),
    "multitrip": dict(obstacles=(0, 2), piles=(2, 4), units=(60, 150), height=2, zone=(12, 14)),
}


def skid_geometry():
    """Chassis and bucket cell offsets from pos_base for each base heading."""
    import jax
    import jax.numpy as jnp

    from terra.config import BatchConfig
    from terra.config import EnvConfig
    from terra.config import MapsDimsConfig
    from terra.env import TerraEnvBatch
    from terra.state import State

    batch_env = object.__new__(TerraEnvBatch)
    batch_env.batch_cfg = BatchConfig()._replace(
        maps_dims=MapsDimsConfig(maps_edge_length=SHAPE[0])
    )
    base = EnvConfig()
    updated = batch_env.update_env_cfgs(
        base._replace(agent=base.agent._replace(dig_depth=jnp.ones((1,), dtype=jnp.int32)))
    )
    cfg = base._replace(
        tile_size=float(np.asarray(updated.tile_size)[0]),
        agent=base.agent._replace(
            width=int(np.asarray(updated.agent.width)[0]),
            height=int(np.asarray(updated.agent.height)[0]),
        ),
        maps=base.maps._replace(edge_length_px=int(np.asarray(updated.maps.edge_length_px)[0])),
        agent_types=(2,),
        action_types=(0,),
    )
    empty = np.zeros(SHAPE, dtype=np.int8)
    target = empty.copy()
    target[0, 0] = 1
    state = State.new(
        jax.random.PRNGKey(0), cfg, target, empty,
        -97.0 * np.ones((3, 3), dtype=np.float32), np.int32(-1),
        -97.0 * np.ones((SHAPE[0], 3), dtype=np.float32), np.int32(-1),
        np.ones(SHAPE, dtype=np.bool_), empty,
        distance_map_override=np.zeros(SHAPE, dtype=np.float32),
    )
    center = np.array([32, 32])
    geometry = []
    for heading in range(int(cfg.agent.angles_base)):
        agent = state.agent.agent_states[0]._replace(
            pos_base=jnp.asarray(center, dtype=jnp.int16),
            angle_base=jnp.array([heading], dtype=jnp.int8),
        )
        posed = state._set_agent_state_at(0, agent)
        chassis = np.asarray(posed._current_base_footprint_mask()).reshape(SHAPE)
        bucket = np.asarray(posed._build_dig_dump_cone()).reshape(SHAPE).astype(bool)
        geometry.append((np.argwhere(chassis) - center, np.argwhere(bucket) - center))
    return geometry, cfg


def shifted(mask: np.ndarray, offset, fill: bool) -> np.ndarray:
    """out[p] = mask[p + offset], ``fill`` outside the map."""
    out = np.full_like(mask, fill)
    dr, dc = int(offset[0]), int(offset[1])
    rows, cols = mask.shape
    out[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)] = (
        mask[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)]
    )
    return out


def scoopable(blocked: np.ndarray, geometry) -> np.ndarray:
    """Cells in the bucket of at least one pose whose chassis is free."""
    reach = np.zeros_like(blocked)
    for chassis, bucket in geometry:
        free = np.ones_like(blocked)
        for offset in chassis:
            free &= ~shifted(blocked, offset, True)
        for offset in bucket:
            reach |= shifted(free, -offset, False)
    return reach


def place_rect(rng, taken, size, margin):
    rows, cols = SHAPE
    for _ in range(50):
        r = int(rng.integers(margin, rows - size[0] - margin + 1))
        c = int(rng.integers(margin, cols - size[1] - margin + 1))
        if not taken[r:r + size[0], c:c + size[1]].any():
            return r, c
    return None


def generate(rng, spec, geometry):
    """One candidate map, or None when a placement or check fails."""
    target = np.zeros(SHAPE, dtype=np.int8)
    occupancy = np.zeros(SHAPE, dtype=bool)
    actions = np.zeros(SHAPE, dtype=np.int8)
    zone_edge = int(rng.integers(spec["zone"][0], spec["zone"][1] + 1))
    corner = place_rect(rng, np.zeros(SHAPE, bool), (zone_edge, zone_edge), 9)
    target[corner[0]:corner[0] + zone_edge, corner[1]:corner[1] + zone_edge] = 1
    taken = target > 0
    for _ in range(int(rng.integers(spec["obstacles"][0], spec["obstacles"][1] + 1))):
        size = (int(rng.integers(4, 9)), int(rng.integers(4, 9)))
        # Obstacles keep two cells from the zone so its rim stays reachable.
        spot = place_rect(rng, _dilate(taken, 2), size, 2)
        if spot is None:
            return None
        occupancy[spot[0]:spot[0] + size[0], spot[1]:spot[1] + size[1]] = True
        taken |= occupancy
    piles = int(rng.integers(spec["piles"][0], spec["piles"][1] + 1))
    total = int(rng.integers(spec["units"][0], spec["units"][1] + 1))
    # Every pile holds at least 8 units, as in generate_relocations_harder.
    shares = rng.multinomial(total - 8 * piles, np.ones(piles) / piles) + 8
    for units in shares:
        height = int(rng.integers(1, spec["height"] + 1))
        cells = int(np.ceil(units / height))
        edge = int(np.ceil(np.sqrt(cells)))
        spot = place_rect(rng, _dilate(taken, 3), (edge, edge), EDGE)
        if spot is None:
            return None
        patch = np.zeros(edge * edge, dtype=np.int8)
        patch[: units // height] = height
        patch[units // height: units // height + (units % height > 0)] += units % height
        actions[spot[0]:spot[0] + edge, spot[1]:spot[1] + edge] = patch.reshape(edge, edge)
        taken |= actions > 0
    if int(actions.astype(np.int64).sum()) != total:
        return None
    soil = actions > 0
    reach = scoopable(occupancy | soil, geometry)
    zone = target > 0
    if not reach[soil].all() or reach[zone].mean() < 0.6:
        return None
    try:
        distance = compute_reward_v2_distance_map(
            target, occupancy, tile_size_m=TILE_SIZE_M,
            distance_ref_m=REWARD_V2_DISTANCE_REF_M, distance_bound=REWARD_V2_DISTANCE_BOUND,
        )
    except ValueError:
        return None
    dumpability = np.ones(SHAPE, dtype=bool)
    contained_dump_capacity_sanity_check(target, occupancy, dumpability, actions)
    arrays = dict(images=target, occupancy=occupancy, dumpability=dumpability,
                  actions=actions, distance=distance)
    info = dict(zone_edge=zone_edge, obstacles=int(occupancy.any() and _count_rects(occupancy)),
                piles=piles, units=total, max_height=int(actions.max()))
    return arrays, info


def _dilate(mask, radius):
    out = mask.copy()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out |= shifted(mask, (dr, dc), False)
    return out


def _count_rects(mask):
    from scipy import ndimage
    return int(ndimage.label(mask)[1])


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--strata", nargs="+", required=True, help="name:count")
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new directory")
    geometry, cfg = skid_geometry()
    if abs(cfg.tile_size - TILE_SIZE_M) > 1e-6:
        raise SystemExit(f"Terra tile size {cfg.tile_size} != bank tile size {TILE_SIZE_M}")
    rng = np.random.default_rng(args.seed)
    for folder in ("images", "occupancy", "dumpability", "actions", "distance", "metadata"):
        (args.output / folder).mkdir(parents=True)
    rows, registry, seen = [], [], set()
    slot = 0
    for entry in args.strata:
        name, count = entry.split(":")
        spec = STRATA[name]
        made = rejected = 0
        while made < int(count):
            candidate = generate(rng, spec, geometry)
            if candidate is None:
                rejected += 1
                continue
            arrays, info = candidate
            scenario = reset_array_scenario_sha256(arrays)
            if scenario in seen:
                rejected += 1
                continue
            seen.add(scenario)
            slot += 1
            made += 1
            for folder, array in arrays.items():
                np.save(args.output / folder / f"img_{slot}.npy", array)
            (args.output / "metadata" / f"trench_{slot}.json").write_text(
                json.dumps(dict(axes_ABC=[], stratum=f"skid-reloc-{name}", **info), sort_keys=True) + "\n"
            )
            map_id = f"skid-reloc:{args.split}:{name}:{made - 1:04d}"
            source_id = f"skid-reloc:{scenario}"
            rows.append(dict(
                slot_index=slot, map_id=map_id, source_id=source_id, split=args.split,
                family="relocation", stratum=f"skid-reloc-{name}", primary_cell=f"skid-reloc-{name}",
                slot_weight=1.0, identity_slot_multiplicity=1, scenario_id=scenario, **info,
            ))
            registry.append(dict(map_id=map_id, source_id=source_id, split=args.split, scenario_id=scenario))
        print(f"{name}: {made} maps, {rejected} candidates rejected")
    write_jsonl(args.output / "manifest.jsonl", rows)
    write_jsonl(args.output / "source_registry.jsonl", registry)
    registry_sha256 = hashlib.sha256((args.output / "source_registry.jsonl").read_bytes()).hexdigest()
    dataset = dict(
        schema="terra_exact_map_dataset_v1", slot_count=slot, unique_identity_count=slot,
        shape=list(SHAPE), accepted_dump_contract="exact_visible_dump_v1",
        scenario_identity_contract=RESET_ARRAY_SCENARIO_IDENTITY_CONTRACT,
        distance_protocol_id=REWARD_V2_DISTANCE_PROTOCOL_ID,
        distance_metric=REWARD_V2_DISTANCE_METRIC,
        distance_normalization=REWARD_V2_DISTANCE_NORMALIZATION,
        tile_size_m=TILE_SIZE_M, distance_ref_m=REWARD_V2_DISTANCE_REF_M,
        distance_bound=REWARD_V2_DISTANCE_BOUND,
        source_registry="source_registry.jsonl", source_registry_sha256=registry_sha256,
        generator=dict(tool="tools/build_skid_relocation_bank.py", seed=args.seed,
                       strata=args.strata, split=args.split),
    )
    (args.output / "dataset.json").write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n")
    print(f"wrote {slot} slots to {args.output}")


if __name__ == "__main__":
    main()
