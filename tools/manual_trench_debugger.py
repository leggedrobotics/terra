#!/usr/bin/env python3
"""Human-playable debugger for the Terra fresh-trench dig-alignment gate.

Play one frozen panel slot by hand against the SAME environment the T1 arm
trained on (gate on, finite trench metadata, reward-v2/R2 protocol, 450-step
horizon) and see, every frame, exactly what a ``DO`` would do and -- if it
would be refused -- why, in plain language.

LAUNCH (needs a display; do NOT set SDL_VIDEODRIVER=dummy)
----------------------------------------------------------
    cd /home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
    JAX_PLATFORMS=cpu PYTHONPATH=$PWD \
      /home/lorenzo/moleworks/.venv-terra-uv/bin/python \
      tools/manual_trench_debugger.py --slot 458

Startup compiles the Terra step (~105 s) and the diagnostic probe (~55 s) on
CPU; both are traced exactly once and the window is interactive after that,
at roughly 5 ms per keypress (4 ms step + 5 ms probe + ~25 ms redraw).  A
persistent JAX compilation cache is kept under ``data/jax_compile_cache``.

KEYS
----
    UP / W          FORWARD                 DOWN / S      BACKWARD
    LEFT / A        base ANTICLOCK (CCW)    RIGHT / D     base CLOCK (CW)
    Q               cabin ANTICLOCK         E             cabin CLOCK
    SPACE / RETURN  DO   (dig when empty, dump when loaded)
    N               DO_NOTHING (burn a step)
    U / BACKSPACE   UNDO one action (full state stack)
    R               reset this slot to its frozen full start
    1 2 3 4 5       jump to recommended slots 296 / 405 / 294 / 455 / 458
    [ / ]           previous / next slot index
    O               overlays on/off        G   geometry (bands, axes) on/off
    T               tint remaining trench cells by owning section on/off
    H               help / legend           ESC / window close   quit
    Keys auto-repeat when held (250 ms delay, 60 ms rate).

OVERLAYS
--------
    dark blue + white       the machine: Terra's own rasterised 7x11 footprint,
      outline               drawn cell-exact on the terrain lattice (this is the
                            occupancy the move legality check actually tests)
    white arrow             chassis forward direction (the yaw the gate tests)
    salmon arrow            cabin/arm direction (where the cone points)
    yellow outline          the workspace cone Terra would act in
    bright green fill       cells this DO would actually remove (admitted)
    orange hatch            fresh trench cells the cone selects but the gate
                            refuses (the cells that cost you the macro action)
    section colours         yellow / cyan / magenta / orange = finite trench
      (axis 0..3)           sections.  Solid line = pose-valid from here,
                            dashed = not pose-valid.  Small arrow = axis
                            direction.  Same colour, thin dotted lines = the
                            3.5 m and 7.0 m standoff band edges for that
                            section: stand between them.
    colour tint (T)         remaining (undug) trench cells, tinted by owning
                            section; a white dot marks a junction cell owned
                            by more than one section
    dark green              already dug target cells (Terra's own colouring)
    blue outline            the accepted dump zone (target > 0, not padding)
    MAGENTA X, loud         spoil that has landed OUTSIDE target > 0.  This
                            breaks the exact_visible_dump_v1 purity contract:
                            the episode cannot complete until it is relifted
                            and dumped inside the accepted zone.
    red outline             padding / obstacle cells inside the cone (these
                            alone refuse the whole DO)

    Terra's terrain colours are its own (purple = to dig, green = dug, beige =
    accepted dump zone, grey-brown = non-dumpable road, blue = spoil, black =
    obstacle).  Terra's agent sprite and its red cone tiles are suppressed:
    the sprite is centred half a cell off Terra's own terrain lattice, and both
    are replaced here by lattice-exact overlays.  The terrain image is cached on
    a hash of the terrain itself, so a pose change costs ~20 ms, not ~140 ms.

RECOMMENDED SLOTS (corrected-footprint scripted-oracle failures, worst first)
----------------------------------------------------------------------------
    296  trn-net3-side1-road       oracle dug 73/88 -- its worst slot
    405  trn-seg3-side2            97/105
    294  trn-net3-side1-road       90/97
    455  trn-straight-side1-tight  73/78, then 859 steps loaded with no legal
                                   dump (a *straight* has no junction veto)
    458  trn-straight-side1-tight  58/60 dug but only 54 of 58 spoil units
                                   landed in the accepted zone -> the purity
                                   failure mode

Everything here is a read-only observer plus the ordinary env step.  The gate
verdict, the yaw/standoff diagnostics, the selected cone, the DO outcome, the
action mask and the completion metrics all come from Terra's own methods.  The
per-section decomposition (which the env does not export) is recomputed here
and asserted against Terra's exported ``fresh_trench_dig_alignment_valid`` /
``fresh_trench_dig_yaw_error`` / ``fresh_trench_dig_standoff_error`` on every
single frame; any divergence aborts the tool loudly.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame as pg

import jax

jax.config.update("jax_threefry_partitionable", True)

import jax.numpy as jnp

WORKTREE = Path(__file__).resolve().parents[1]

DEFAULT_BANK = "/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819"
DEFAULT_PANEL = "evaluation/gate_main/development"
BANK_TERRA_REVISION = "a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4"
ORACLE_RECEIPTS = (
    ("corrected footprint (fix 1 only)", "oracle_176slot_corrected_footprint.json"),
    ("pre-fix baseline", "oracle_176slot_gate_main_dev.json"),
)
RECOMMENDED = (296, 405, 294, 455, 458)

ANGLES_BASE = 12  # State uses AgentConfig().angles_base directly; keep in sync
SECTION_COLORS = ((255, 235, 59), (0, 229, 255), (255, 64, 255), (0, 255, 180))


# --------------------------------------------------------------------------- #
# environment construction                                                    #
# --------------------------------------------------------------------------- #
def build_env(bank: str, panel: str, slot_count: int, gate: bool, rendering: bool):
    """Build the T1-arm environment for one frozen evaluation panel."""
    os.environ["DATASET_PATH"] = bank
    os.environ["DATASET_SIZE"] = str(slot_count)

    from terra.config import BatchConfig, CurriculumGlobalConfig, EnvConfig, RewardStage
    from terra.env import TerraEnvBatch
    from terra.maps_buffer import REWARD_V2_DISTANCE_PROTOCOL_ID

    level_config = [
        {
            "maps_path": panel,
            "max_steps_in_episode": 450,
            "rewards_type": 0,
            "apply_trench_rewards": False,
        }
    ]

    class _Curriculum(CurriculumGlobalConfig):
        # eval_fixed_bank.configure_for_bank: one level, no curriculum motion.
        levels = level_config
        last_level_type = "none"

    batch_cfg = BatchConfig(curriculum_global=_Curriculum())
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        rendering=rendering,
        display=False,  # we own the window; Game draws into an offscreen surface
        distance_protocol_id=REWARD_V2_DISTANCE_PROTOCOL_ID,
    )
    env_cfgs = jax.vmap(
        lambda _: EnvConfig.new()._replace(
            agent_types=(0,),
            action_types=(0,),
            enforce_trench_dig_alignment=bool(gate),
            reward_stage=int(RewardStage.REWARD_V2),
        )
    )(jnp.arange(1))
    return env, env_cfgs, batch_cfg


def load_manifest(bank: str, panel: str) -> list[dict]:
    path = Path(bank) / panel / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: int(row["slot_index"]))
    expected = list(range(1, len(rows) + 1))
    if [int(r["slot_index"]) for r in rows] != expected:
        raise RuntimeError(f"{path} does not enumerate contiguous slots 1..N")
    return rows


def map_selection_key(slot_index0: int, slot_count: int) -> jax.Array:
    """The eval protocol's ordered-slot key: PRNGKey(seed) that selects slot i.

    eval_fixed_bank.exact_reset_keys() searches integer seeds for a key whose
    split-then-randint lands on each slot; this is the same search for one slot.
    """

    def selected(key):
        _, subkey = jax.random.split(key)
        return jax.random.randint(subkey, (), 0, slot_count)

    start = 0
    while start < 1_000_000:
        seeds = jnp.arange(start, start + 4096, dtype=jnp.uint32)
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        hits = np.asarray(jax.vmap(selected)(keys)) == slot_index0
        if bool(hits.any()):
            return jnp.asarray(np.asarray(keys)[int(np.argmax(hits))])
        start += 4096
    raise RuntimeError(f"no ordered-slot key found for slot index {slot_index0}")


class SlotLoader:
    """Reset one panel slot exactly the way eval_fixed_bank.py does."""

    def __init__(self, env, env_cfgs, bank: str, panel: str, rows: list[dict]):
        self.env = env
        self.env_cfgs = env_cfgs
        self.bank = Path(bank)
        self.panel = panel
        self.rows = rows
        self.count = len(rows)
        self._prepared: dict[int, tuple] = {}

    def prepare(self, slot: int):
        if slot not in self._prepared:
            if not 1 <= slot <= self.count:
                raise SystemExit(f"--slot must be in 1..{self.count}, got {slot}")
            map_key = map_selection_key(slot - 1, self.count)
            prepared = self.env._prepare_reset_device(self.env_cfgs, map_key[None])
            target = np.asarray(prepared[1])[0]
            disk = np.load(self.bank / self.panel / "images" / f"img_{slot}.npy")
            if not np.array_equal(target, disk):
                raise RuntimeError(
                    f"slot {slot}: materialized target map != images/img_{slot}.npy"
                )
            self._prepared[slot] = prepared
        return self._prepared[slot]

    def reset(self, slot: int):
        prepared = self.prepare(slot)
        seed = int(self.rows[slot - 1]["reset_seed"])
        env_key = jax.vmap(jax.random.PRNGKey)(jnp.asarray([seed], dtype=jnp.uint32))
        timestep = self.env.reset_prepared(prepared[0], env_key, *prepared[1:])
        # A reset timestep differs from a stepped one in exactly one leaf's
        # aval: state.productive_workspace_cycles is weakly typed int32 after
        # reset and strongly typed after a step.  That alone makes
        # step_no_reset trace twice, at ~100 s a trace.  Strip the weak type on
        # that counter only (a numpy round-trip) so play needs one trace.
        # Do NOT do this to the whole pytree: it would also strip
        # env_cfg.agent.angles_base, whose weak int32 is what keeps
        # decrease_angle_circular in int8 -- strengthening it makes Terra's own
        # _apply_base_rotation_mask lax.cond raise on every base rotation.
        timestep = timestep._replace(
            state=timestep.state._replace(
                productive_workspace_cycles=jnp.asarray(
                    np.asarray(timestep.state.productive_workspace_cycles)
                )
            )
        )
        return timestep, prepared[0]


_ORACLE_CACHE: dict[int, list[tuple[str, dict]]] | None = None


def oracle_rows(slot: int) -> list[tuple[str, dict]]:
    """Per-slot scripted-oracle results, parsed once (the receipts are ~7 MB)."""
    global _ORACLE_CACHE
    if _ORACLE_CACHE is None:
        _ORACLE_CACHE = {}
        root = WORKTREE / "tools" / "trench_align_oracle_receipts_20260831"
        for label, name in ORACLE_RECEIPTS:
            path = root / name
            if not path.exists():
                continue
            for row in json.loads(path.read_text()).get("per_slot", []):
                _ORACLE_CACHE.setdefault(int(row["slot_index"]), []).append((label, row))
    return _ORACLE_CACHE.get(slot, [])


# --------------------------------------------------------------------------- #
# the diagnostic probe: Terra's own logic, plus a per-section decomposition    #
# --------------------------------------------------------------------------- #
def make_probe(batch_cfg):
    from terra.state import (
        _as_2d_map,
        _as_axes_table,
        _as_scalar_int,
        _flat_2d_map,
    )
    from terra.utils import compute_polygon_mask

    dummy_action = batch_cfg.action_type.do_nothing()

    def one(state):
        cur = state._get_current_agent_state()
        shape = _as_2d_map(state.world.action_map.map).shape
        height, width = shape
        target = _as_2d_map(state.world.target_map.map).astype(jnp.int32)
        action = _as_2d_map(state.world.action_map.map).astype(jnp.int32)
        padding = _as_2d_map(state.world.padding_mask.map).astype(jnp.int32)
        dumpability = _as_2d_map(state.world.dumpability_mask.map).astype(jnp.bool_)
        last_dig = _as_2d_map(state.world.last_dig_mask.map).astype(jnp.bool_)
        membership = _as_2d_map(state.world.trench_axis_membership).astype(jnp.uint8)
        accepted = state._accepted_dump_mask().astype(jnp.bool_)

        # ---- Terra's own cone and dig selection -------------------------- #
        cone = state._build_dig_dump_cone().astype(jnp.bool_)
        selected = state._mask_out_wrong_dig_tiles(cone).astype(jnp.bool_)
        (
            terra_valid,
            terra_yaw_norm,
            terra_standoff_norm,
            admitted,
        ) = state._get_fresh_trench_dig_alignment_details(selected)

        cone2d = cone.reshape(shape)
        selected2d = selected.reshape(shape)

        # ---- the gate's own quantities, recomputed per section ------------ #
        records = _as_axes_table(state.world.trench_axes).astype(jnp.float32)
        axes = records[:, :3]
        max_axes = axes.shape[0]
        trench_type = jnp.clip(_as_scalar_int(state.world.trench_type), 0, max_axes)
        valid_axes = jnp.arange(max_axes) < trench_type
        segment_vectors = records[:, 5:7] - records[:, 3:5]
        finite_metadata = jnp.logical_and(
            jnp.all(records[:, 3:8] > jnp.float32(-96.0), axis=1),
            jnp.logical_and(
                records[:, 7] > jnp.float32(0.0),
                jnp.linalg.norm(segment_vectors, axis=1) > jnp.float32(1e-6),
            ),
        )
        declared_metadata_valid = jnp.all(
            jnp.logical_or(~valid_axes, finite_metadata)
        )
        fail_closed = jnp.logical_and(trench_type > 0, ~declared_metadata_valid)

        fresh_target = jnp.logical_and(
            selected2d, jnp.logical_and(target < 0, action == 0)
        )
        fresh_trench = jnp.logical_and(
            fresh_target,
            jnp.logical_or(membership != jnp.uint8(0), fail_closed),
        )
        applicable = jnp.logical_and(
            cur.agent_type[0] == 0,
            jnp.logical_and(
                cur.loaded[0] == 0,
                jnp.logical_and(trench_type > 0, jnp.any(fresh_trench)),
            ),
        )

        bit_values = jnp.left_shift(
            jnp.ones((max_axes,), dtype=jnp.uint8),
            jnp.arange(max_axes, dtype=jnp.uint8),
        )
        section_member = jnp.logical_and(
            valid_axes[:, None, None],
            jnp.bitwise_and(membership[None, :, :], bit_values[:, None, None]) != 0,
        )
        axis_has_fresh = jnp.any(
            jnp.logical_and(section_member, fresh_trench[None, :, :]), axis=(1, 2)
        )

        line_denominators = jnp.maximum(
            jnp.linalg.norm(axes[:, :2], axis=1), jnp.float32(1e-6)
        )
        base_angle = jnp.ravel(state._get_base_angle_rad())[0]
        base_forward = jnp.array(
            [-jnp.sin(base_angle), jnp.cos(base_angle)], dtype=jnp.float32
        )
        trench_tangents = jnp.stack([-axes[:, 0], axes[:, 1]], axis=1)
        tangent_norms = jnp.maximum(
            jnp.linalg.norm(trench_tangents, axis=1), jnp.float32(1e-6)
        )
        parallel_cosines = jnp.clip(
            jnp.abs(trench_tangents @ base_forward) / tangent_norms, 0.0, 1.0
        )
        yaw_errors = jnp.arccos(parallel_cosines)
        yaw_errors_normalized = jnp.clip(yaw_errors / (jnp.pi / 2.0), 0.0, 1.0)

        base_row = cur.pos_base[0].astype(jnp.float32)
        base_col = cur.pos_base[1].astype(jnp.float32)
        signed_cells = (
            axes[:, 0] * base_col + axes[:, 1] * base_row + axes[:, 2]
        ) / line_denominators
        standoffs_m = jnp.abs(signed_cells) * state.env_cfg.tile_size
        standoff_min = jnp.float32(state.env_cfg.trench_dig_standoff_min_m)
        standoff_max = jnp.float32(state.env_cfg.trench_dig_standoff_max_m)
        standoff_errors_normalized = jnp.clip(
            jnp.where(
                standoffs_m < standoff_min,
                (standoffs_m - standoff_min) / jnp.maximum(standoff_min, 1e-6),
                jnp.where(
                    standoffs_m > standoff_max,
                    (standoffs_m - standoff_max) / jnp.maximum(standoff_max, 1e-6),
                    jnp.float32(0.0),
                ),
            ),
            -1.0,
            1.0,
        )
        yaw_ok = yaw_errors <= jnp.float32(state.env_cfg.trench_dig_yaw_tolerance_rad)
        band_ok = jnp.logical_and(
            standoffs_m >= standoff_min, standoffs_m <= standoff_max
        )
        axis_pose_valid = jnp.logical_and(
            valid_axes,
            jnp.logical_and(
                finite_metadata,
                jnp.logical_and(axis_has_fresh, jnp.logical_and(yaw_ok, band_ok)),
            ),
        )
        fresh_cell_pose_valid = jnp.any(
            jnp.logical_and(section_member, axis_pose_valid[:, None, None]), axis=0
        )
        replica_valid_when_applicable = jnp.all(
            jnp.logical_or(~fresh_trench, fresh_cell_pose_valid)
        )
        replica_valid = jnp.where(applicable, replica_valid_when_applicable, True)

        diagnostic_pool = jnp.where(
            replica_valid_when_applicable,
            axis_pose_valid,
            jnp.logical_and(axis_has_fresh, ~axis_pose_valid),
        )
        diagnostic_score = yaw_errors_normalized + jnp.abs(standoff_errors_normalized)
        diagnostic_axis = jnp.argmin(
            jnp.where(diagnostic_pool, diagnostic_score, jnp.float32(jnp.inf))
        )
        replica_yaw_norm = jnp.where(
            applicable, yaw_errors_normalized[diagnostic_axis], jnp.float32(0.0)
        )
        replica_standoff_norm = jnp.where(
            applicable, standoff_errors_normalized[diagnostic_axis], jnp.float32(0.0)
        )

        owner_count = jnp.sum(section_member.astype(jnp.int32), axis=0)
        blocked_cells = jnp.logical_and(fresh_trench, ~fresh_cell_pose_valid)
        per_axis_fresh = jnp.sum(
            jnp.logical_and(section_member, fresh_trench[None, :, :]).astype(jnp.int32),
            axis=(1, 2),
        )
        per_axis_exclusive_blocked = jnp.sum(
            jnp.logical_and(
                jnp.logical_and(section_member, blocked_cells[None, :, :]),
                (owner_count == 1)[None, :, :],
            ).astype(jnp.int32),
            axis=(1, 2),
        )
        per_axis_remaining = jnp.sum(
            jnp.logical_and(
                section_member, jnp.logical_and(target < 0, action == 0)[None, :, :]
            ).astype(jnp.int32),
            axis=(1, 2),
        )

        # ---- what would DO actually do (Terra's own transition) ----------- #
        after_do = state._handle_do()
        action_after = _as_2d_map(after_do.world.action_map.map).astype(jnp.int32)
        loaded_after = after_do._get_current_agent_state().loaded[0].astype(jnp.int32)
        completion_after = after_do._get_task_completion(
            after_do.world.action_map.map, after_do.world.target_map.map
        )
        do_changes_map = jnp.any(action_after != action)
        do_changes_load = loaded_after != cur.loaded[0].astype(jnp.int32)
        do_cells_removed = jnp.sum(
            jnp.logical_and(action_after < action, target < 0).astype(jnp.int32)
        )
        do_volume_removed = jnp.sum(jnp.clip(action - action_after, 0, None))
        do_volume_added = jnp.sum(jnp.clip(action_after - action, 0, None))

        # ---- dump geometry ------------------------------------------------ #
        physical = state._build_dig_dump_cone().astype(jnp.bool_)
        physical = state._exclude_dig_tiles_from_dump_mask(physical)
        physical = state._exclude_dumpability_mask_tiles_from_dump_mask(physical)
        physical = state._exclude_traversability_mask_tiles_from_dump_mask(physical)
        physical = state._exclude_just_moved_tiles_from_dump_mask(physical)
        physical = jnp.logical_and(
            physical, _flat_2d_map(state.world.padding_mask.map == 0)
        )
        dump_lacks_space = state._dump_cone_lacks_free_space(physical)
        physical = jnp.logical_and(physical, ~dump_lacks_space)
        accepted_flat = accepted.reshape(-1)
        dump_legal = jnp.logical_and(physical, accepted_flat)
        dump_wrong = jnp.logical_and(physical, ~accepted_flat)

        def dump_for_cabin(angle):
            probe_state = state._set_current_agent_state(
                cur._replace(angle_cabin=jnp.asarray([angle], dtype=cur.angle_cabin.dtype))
            )
            mask = probe_state._build_dig_dump_cone().astype(jnp.bool_)
            mask = probe_state._exclude_dig_tiles_from_dump_mask(mask)
            mask = probe_state._exclude_dumpability_mask_tiles_from_dump_mask(mask)
            mask = probe_state._exclude_traversability_mask_tiles_from_dump_mask(mask)
            mask = probe_state._exclude_just_moved_tiles_from_dump_mask(mask)
            mask = jnp.logical_and(
                mask, _flat_2d_map(probe_state.world.padding_mask.map == 0)
            )
            mask = jnp.logical_and(
                mask, ~probe_state._dump_cone_lacks_free_space(mask)
            )
            return jnp.array(
                [
                    jnp.sum(jnp.logical_and(mask, accepted_flat).astype(jnp.int32)),
                    jnp.sum(jnp.logical_and(mask, ~accepted_flat).astype(jnp.int32)),
                ]
            )

        dump_by_cabin = jax.lax.map(
            dump_for_cabin, jnp.arange(ANGLES_BASE, dtype=jnp.int32)
        )

        # ---- movement legality and the reason a move is refused ----------- #
        traversability = state._build_traversability_mask(
            _as_2d_map(state.world.action_map.map),
            _as_2d_map(state.world.static_traversability_base.map),
        )

        def footprint_report(corners):
            poly = compute_polygon_mask(corners, width, height).astype(jnp.bool_)
            in_bounds = jnp.all(
                jnp.logical_and(
                    corners >= jnp.array([0, 0]),
                    corners < jnp.array([width, height]),
                )
            )
            return {
                "valid": state._is_valid_move(corners),
                "in_bounds": in_bounds,
                "padding": jnp.sum(jnp.logical_and(poly, padding == 1).astype(jnp.int32)),
                "holes": jnp.sum(jnp.logical_and(poly, action < 0).astype(jnp.int32)),
                "piles": jnp.sum(
                    jnp.logical_and(
                        poly, jnp.logical_and(action > 0, traversability == 1)
                    ).astype(jnp.int32)
                ),
                "cells": jnp.sum(poly.astype(jnp.int32)),
                "mask": poly,
            }

        def translation_corners(one_hot):
            angles = jnp.linspace(0, 2 * jnp.pi, ANGLES_BASE, endpoint=False)
            angles = (angles + (jnp.pi / 2)) % (2 * jnp.pi)
            xy_delta = state.env_cfg.agent.move_tiles * jnp.stack(
                [jnp.cos(angles), jnp.sin(angles)], axis=-1
            )
            delta = one_hot.astype(jnp.float32) @ xy_delta
            candidate = jnp.round(cur.pos_base + delta).astype(jnp.int32)
            candidate = jnp.reshape(candidate, (-1, 2))[0]
            corners = state._get_agent_corners(
                candidate,
                base_orientation=cur.angle_base,
                agent_width=state.env_cfg.agent.width,
                agent_height=state.env_cfg.agent.height,
            )
            return candidate, corners

        def rotation_corners(delta):
            new_angle = (cur.angle_base + ANGLES_BASE + delta) % ANGLES_BASE
            corners = state._get_agent_corners(
                cur.pos_base,
                base_orientation=new_angle,
                agent_width=state.env_cfg.agent.width,
                agent_height=state.env_cfg.agent.height,
            )
            return new_angle, corners

        forward_pos, forward_corners = translation_corners(
            state._base_orientation_to_one_hot_forward(cur.angle_base)
        )
        backward_pos, backward_corners = translation_corners(
            state._base_orientation_to_one_hot_backwards(cur.angle_base)
        )
        clock_angle, clock_corners = rotation_corners(-1)
        anticlock_angle, anticlock_corners = rotation_corners(+1)

        own_corners = state._get_agent_corners(
            cur.pos_base,
            base_orientation=cur.angle_base,
            agent_width=state.env_cfg.agent.width,
            agent_height=state.env_cfg.agent.height,
        )

        completion = state._get_task_completion(
            state.world.action_map.map, state.world.target_map.map
        )
        positive_soil = jnp.clip(action, 0, None)

        return {
            # gate: Terra's own exports
            "terra_valid": terra_valid,
            "terra_yaw_norm": terra_yaw_norm,
            "terra_standoff_norm": terra_standoff_norm,
            "admitted_cells": jnp.sum(admitted.astype(jnp.int32)),
            # gate: replica
            "replica_valid": replica_valid,
            "replica_yaw_norm": replica_yaw_norm,
            "replica_standoff_norm": replica_standoff_norm,
            "applicable": applicable,
            "trench_type": trench_type,
            "valid_axes": valid_axes,
            "finite_metadata": finite_metadata,
            "axis_has_fresh": axis_has_fresh,
            "axis_pose_valid": axis_pose_valid,
            "yaw_errors_rad": yaw_errors,
            "standoffs_m": standoffs_m,
            "signed_cells": signed_cells,
            "yaw_ok": yaw_ok,
            "band_ok": band_ok,
            "axes": axes,
            "segments": records[:, 3:7],
            "half_widths": records[:, 7],
            "per_axis_fresh": per_axis_fresh,
            "per_axis_exclusive_blocked": per_axis_exclusive_blocked,
            "per_axis_remaining": per_axis_remaining,
            "yaw_tolerance_rad": jnp.float32(state.env_cfg.trench_dig_yaw_tolerance_rad),
            "standoff_min_m": standoff_min,
            "standoff_max_m": standoff_max,
            "tile_size": jnp.float32(state.env_cfg.tile_size),
            "gate_enabled": jnp.bool_(state.env_cfg.enforce_trench_dig_alignment),
            # cone content
            "cone": cone2d,
            "selected": selected2d,
            "fresh_trench": fresh_trench,
            "blocked_cells": blocked_cells,
            "admitted_mask": jnp.where(
                jnp.bool_(state.env_cfg.enforce_trench_dig_alignment),
                admitted.reshape(shape),
                selected2d,
            ),
            "remaining_mask": jnp.logical_and(target < 0, action == 0),
            "padding_mask": padding == 1,
            "cone_padding_mask": jnp.logical_and(cone2d, padding == 1),
            "owner_count": owner_count,
            "section_member": section_member,
            "cone_cells": jnp.sum(cone2d.astype(jnp.int32)),
            "cone_padding": jnp.sum(
                jnp.logical_and(cone2d, padding == 1).astype(jnp.int32)
            ),
            "cone_positive_soil": jnp.sum(jnp.where(cone2d, positive_soil, 0)),
            "cone_fresh_any_target": jnp.sum(
                jnp.logical_and(cone2d, jnp.logical_and(target < 0, action == 0)).astype(
                    jnp.int32
                )
            ),
            "cone_last_dig": jnp.sum(
                jnp.logical_and(cone2d, last_dig).astype(jnp.int32)
            ),
            "cone_fully_dug": jnp.sum(
                jnp.logical_and(
                    cone2d,
                    jnp.logical_and(
                        target < 0, action <= -state.env_cfg.agent.dig_depth
                    ),
                ).astype(jnp.int32)
            ),
            "workspace_blocked": state._workspace_intersects_obstacle(),
            # DO outcome
            "do_changes": jnp.logical_or(do_changes_map, do_changes_load),
            "do_cells_removed": do_cells_removed,
            "do_volume_removed": do_volume_removed,
            "do_volume_added": do_volume_added,
            "do_loaded_after": loaded_after,
            "do_illegal_after": completion_after["illegal_dump_volume"],
            "do_purity_after": completion_after["dump_purity"],
            # dump geometry
            "dump_physical": jnp.sum(physical.astype(jnp.int32)),
            "dump_legal": jnp.sum(dump_legal.astype(jnp.int32)),
            "dump_wrong": jnp.sum(dump_wrong.astype(jnp.int32)),
            "dump_lacks_space": dump_lacks_space,
            "dump_by_cabin": dump_by_cabin,
            "accepted_mask": accepted,
            "accepted_free": jnp.sum(
                jnp.logical_and(accepted, dumpability).astype(jnp.int32)
            ),
            # movement
            "action_mask": state._get_action_mask(dummy_action),
            "forward": footprint_report(forward_corners),
            "backward": footprint_report(backward_corners),
            "clock": footprint_report(clock_corners),
            "anticlock": footprint_report(anticlock_corners),
            "forward_pos": forward_pos,
            "backward_pos": backward_pos,
            "clock_angle": jnp.reshape(clock_angle, (-1,))[0],
            "anticlock_angle": jnp.reshape(anticlock_angle, (-1,))[0],
            "footprint": compute_polygon_mask(own_corners, width, height),
            # status
            "pos_base": cur.pos_base,
            "angle_base": cur.angle_base[0],
            "angle_cabin": cur.angle_cabin[0],
            "loaded": cur.loaded[0],
            "base_angle_rad": base_angle,
            "env_steps": state.env_steps,
            "max_steps": state.env_cfg.max_steps_in_episode,
            "required_cells": jnp.sum((target < 0).astype(jnp.int32)),
            "dug_cells": jnp.sum(
                jnp.logical_and(target < 0, action < 0).astype(jnp.int32)
            ),
            "fresh_cells": jnp.sum(
                jnp.logical_and(target < 0, action == 0).astype(jnp.int32)
            ),
            "spoiled_target_cells": jnp.sum(
                jnp.logical_and(target < 0, action > 0).astype(jnp.int32)
            ),
            "illegal_spoil_cells": jnp.sum(
                jnp.logical_and(~accepted, action > 0).astype(jnp.int32)
            ),
            "illegal_spoil_mask": jnp.logical_and(~accepted, action > 0),
            "positive_soil": jnp.sum(positive_soil),
            **{f"completion_{key}": value for key, value in completion.items()},
        }

    return jax.jit(lambda state: jax.vmap(one)(state))


def to_host(tree):
    return jax.tree_util.tree_map(lambda value: np.asarray(value)[0], tree)


# --------------------------------------------------------------------------- #
# refusal explainer                                                           #
# --------------------------------------------------------------------------- #
def check_replica(probe: dict, observation) -> None:
    """Fail loudly if the per-section replica drifts from Terra's exports."""
    terra_valid = bool(probe["terra_valid"])
    replica_valid = bool(probe["replica_valid"])
    problems = []
    if terra_valid != replica_valid:
        problems.append(f"valid: terra={terra_valid} replica={replica_valid}")
    for name in ("yaw_norm", "standoff_norm"):
        a = float(probe[f"terra_{name}"])
        b = float(probe[f"replica_{name}"])
        if abs(a - b) > 1e-5:
            problems.append(f"{name}: terra={a:.7f} replica={b:.7f}")
    if observation is not None:
        obs_valid = bool(np.asarray(observation["fresh_trench_dig_alignment_valid"])[0] > 0.5)
        obs_yaw = float(np.asarray(observation["fresh_trench_dig_yaw_error"])[0])
        obs_standoff = float(np.asarray(observation["fresh_trench_dig_standoff_error"])[0])
        if obs_valid != terra_valid:
            problems.append(f"observation valid={obs_valid} != probe {terra_valid}")
        if abs(obs_yaw - float(probe["terra_yaw_norm"])) > 1e-5:
            problems.append(f"observation yaw={obs_yaw} != probe {float(probe['terra_yaw_norm'])}")
        if abs(obs_standoff - float(probe["terra_standoff_norm"])) > 1e-5:
            problems.append(
                f"observation standoff={obs_standoff} != probe {float(probe['terra_standoff_norm'])}"
            )
    # The move-refusal reasons come from candidate footprints this tool builds
    # itself; check them against Terra's own action mask.  Terra's mask bit is
    # "the pose actually changed", which for an empty excavator is exactly
    # _is_valid_move on the candidate corners.
    if int(probe["loaded"]) == 0:
        mask = np.asarray(probe["action_mask"])
        for index, name in ((0, "forward"), (1, "backward"), (2, "clock"), (3, "anticlock")):
            if bool(mask[index]) != bool(probe[name]["valid"]):
                problems.append(
                    f"{name}: terra action_mask={bool(mask[index])} "
                    f"candidate _is_valid_move={bool(probe[name]['valid'])}"
                )
    if problems:
        raise RuntimeError(
            "GATE REPLICA DIVERGED FROM TERRA -- the explanation cannot be trusted:\n  "
            + "\n  ".join(problems)
        )


def section_line(probe: dict, axis: int) -> str:
    a, b, _ = [float(v) for v in probe["axes"][axis]]
    yaw_deg = np.degrees(float(probe["yaw_errors_rad"][axis]))
    tol_deg = np.degrees(float(probe["yaw_tolerance_rad"]))
    standoff = float(probe["standoffs_m"][axis])
    lo = float(probe["standoff_min_m"])
    hi = float(probe["standoff_max_m"])
    yaw_flag = "OK  " if bool(probe["yaw_ok"][axis]) else "FAIL"
    if bool(probe["band_ok"][axis]):
        band_flag = "OK  "
    elif standoff < lo:
        band_flag = "TOO CLOSE"
    else:
        band_flag = "TOO FAR"
    verdict = "POSE-VALID" if bool(probe["axis_pose_valid"][axis]) else "not usable"
    return (
        f"  S{axis} yaw {yaw_deg:5.1f}/{tol_deg:.1f}deg {yaw_flag} | "
        f"standoff {standoff:5.2f}m [{lo:.1f},{hi:.1f}] {band_flag} | {verdict} | "
        f"fresh-in-cone {int(probe['per_axis_fresh'][axis])}"
    )


def explain_do(probe: dict) -> tuple[str, str, list[str]]:
    """Return (verdict, reason_code, detail lines) for pressing DO right now."""
    lines: list[str] = []
    loaded = int(probe["loaded"])
    n_axes = int(probe["trench_type"])

    if bool(probe["workspace_blocked"]):
        return (
            "DO -> REFUSED (no-op)",
            "padding_in_cone",
            [
                f"  the workspace cone overlaps {int(probe['cone_padding'])} static "
                "padding/obstacle cells",
                "  Terra refuses the whole dig before any soil logic runs "
                "(_handle_dig -> _workspace_intersects_obstacle)",
                "  fix: rotate the cabin or step away so no black cell is in the cone",
            ],
        )

    if loaded > 0:
        legal = int(probe["dump_legal"])
        wrong = int(probe["dump_wrong"])
        by_cabin = np.asarray(probe["dump_by_cabin"])
        cabins_legal = int((by_cabin[:, 0] > 0).sum())
        cabins_any = int((by_cabin.sum(axis=1) > 0).sum())
        if legal > 0:
            return (
                f"DO -> DUMP {loaded}u into the accepted zone",
                "dump_legal",
                [
                    f"  loaded {loaded}u; {legal} accepted dump cells are reachable "
                    f"in this cone",
                    f"  purity after this dump: {float(probe['do_purity_after']):.3f}, "
                    f"illegal spoil after: {int(probe['do_illegal_after'])}u",
                ],
            )
        if wrong > 0:
            return (
                f"DO -> DUMP {loaded}u OUTSIDE the accepted zone (purity loss!)",
                "dump_illegal",
                [
                    "  no accepted (target>0) cell is reachable, so Terra falls back to "
                    f"the {wrong} non-accepted cells in the cone",
                    f"  this creates {int(probe['do_illegal_after']) - int(probe['completion_illegal_dump_volume'])}u "
                    "of illegal spoil and permanently caps dump_purity",
                    f"  cabin angles with a LEGAL dump from this pose: {cabins_legal}/12"
                    + ("  <- rotate the cabin instead" if cabins_legal else ""),
                ],
            )
        return (
            "DO -> REFUSED (no-op): nowhere to dump",
            "dump_no_destination",
            [
                f"  loaded {loaded}u but the dump cone has no physically valid cell "
                f"(dumpability/hole/pile/padding/last-workspace filters)",
                f"  dump cone free-space veto: {bool(probe['dump_lacks_space'])}",
                f"  cabin angles with ANY dump: {cabins_any}/12, with a LEGAL dump: "
                f"{cabins_legal}/12",
                "  an excavator cannot move while loaded -- only cabin rotation can help",
            ],
        )

    # ---- empty excavator: the gate's own refusal comes first -------------- #
    if bool(probe["applicable"]) and not bool(probe["terra_valid"]) and bool(
        probe["gate_enabled"]
    ):
        fresh = int(probe["fresh_trench"].sum())
        blocked = int(probe["blocked_cells"].sum())
        pose_valid_any = bool(np.asarray(probe["axis_pose_valid"]).any())
        exclusive = np.asarray(probe["per_axis_exclusive_blocked"])
        offenders = [
            axis
            for axis in range(n_axes)
            if exclusive[axis] > 0 and not bool(probe["axis_pose_valid"][axis])
        ]
        lines.append(
            f"  the cone selects {fresh} fresh trench cells; {blocked} of them have no "
            "pose-valid owning section"
        )
        for axis in range(n_axes):
            if int(probe["per_axis_fresh"][axis]) > 0 or bool(probe["axis_has_fresh"][axis]):
                lines.append(section_line(probe, axis))
        if pose_valid_any and offenders:
            for axis in offenders:
                yaw_deg = np.degrees(float(probe["yaw_errors_rad"][axis]))
                standoff = float(probe["standoffs_m"][axis])
                why = []
                if not bool(probe["yaw_ok"][axis]):
                    why.append(f"misaligned by {yaw_deg:.1f}deg")
                if not bool(probe["band_ok"][axis]):
                    why.append(f"standoff {standoff:.2f}m out of band")
                lines.append(
                    f"  JUNCTION VETO: the cone contains {int(exclusive[axis])} cells "
                    f"owned EXCLUSIVELY by section {axis}, which is "
                    + " and ".join(why)
                )
            lines.append(
                "  at least one section IS pose-valid here, but DO is one macro action: "
                "Terra refuses all of it rather than digging part of it"
            )
            lines.append(
                "  fix: back off / re-aim so the cone stops covering the perpendicular "
                "branch, or align to a pose valid for both sections"
            )
            return ("DO -> REFUSED (no-op): junction veto", "junction_veto", lines)

        yaw_fail = [
            axis
            for axis in range(n_axes)
            if bool(probe["axis_has_fresh"][axis]) and not bool(probe["yaw_ok"][axis])
        ]
        band_fail = [
            axis
            for axis in range(n_axes)
            if bool(probe["axis_has_fresh"][axis]) and not bool(probe["band_ok"][axis])
        ]
        if yaw_fail and not band_fail:
            code, head = "misaligned", "chassis yaw outside 15deg of every owning section"
            lines.append("  fix: rotate the base (LEFT/RIGHT) until yaw <= 15deg")
        elif band_fail and not yaw_fail:
            code, head = "out_of_band", "perpendicular standoff outside 3.5-7.0 m"
            lines.append(
                "  fix: drive along the dotted band edges of that section's colour "
                "(FORWARD/BACKWARD) until the standoff is inside the band"
            )
        else:
            code, head = "misaligned_and_out_of_band", "yaw AND standoff both fail"
            lines.append("  fix: re-aim the chassis first, then correct the standoff")
        return (f"DO -> REFUSED (no-op): {head}", code, lines)

    if bool(probe["do_changes"]):
        cells = int(probe["do_cells_removed"])
        lines = []
        if cells > 0:
            kind = "fresh trench" if n_axes > 0 else "fresh target"
            head = (
                f"DO -> ADMITTED: removes {cells} {kind} cells "
                f"({int(probe['do_volume_removed'])}u), load becomes "
                f"{int(probe['do_loaded_after'])}u"
            )
            code = "admitted"
            if n_axes > 0:
                lines.append(
                    f"  {int(probe['admitted_cells'])} of {int(probe['selected'].sum())} "
                    f"selected cells pass the gate (gate verdict valid="
                    f"{bool(probe['terra_valid'])})"
                )
            else:
                lines.append(
                    "  this map declares no finite trench section, so the "
                    "dig-alignment gate is inapplicable here"
                )
        else:
            head = (
                f"DO -> RELIFT / non-fresh excavation: {int(probe['do_volume_removed'])}u "
                f"removed, load becomes {int(probe['do_loaded_after'])}u"
            )
            code = "relift"
            lines.append(
                f"  no fresh trench cell is removed; the cone holds "
                f"{int(probe['cone_positive_soil'])}u of loose spoil, so Terra's "
                "ambiguity rule selects positive soil and the gate does not apply"
            )
        for axis in range(n_axes):
            if bool(probe["axis_has_fresh"][axis]):
                lines.append(section_line(probe, axis))
        if not bool(probe["terra_valid"]):
            lines.append(
                "  !! the gate is OFF: with the gate ON this DO would be REFUSED"
            )
        return (head, code, lines)

    # ---- nothing happens: say which filter emptied the selection ---------- #
    lines = [
        f"  cone covers {int(probe['cone_cells'])} cells: "
        f"{int(probe['cone_fresh_any_target'])} fresh trench, "
        f"{int(probe['cone_positive_soil'])}u loose spoil, "
        f"{int(probe['cone_fully_dug'])} already at target depth, "
        f"{int(probe['cone_last_dig'])} in the last-dig exclusion; "
        f"Terra selects {int(probe['selected'].sum())}"
    ]
    if int(probe["cone_fresh_any_target"]) == 0:
        lines.append(
            "  reason: NOT APPLICABLE -- no fresh (target<0, undug) trench cell is in "
            "the cone at all.  Drive/rotate until purple cells enter the yellow cone."
        )
        code = "not_applicable_no_fresh_soil"
    elif int(probe["selected"].sum()) == 0:
        lines.append(
            "  reason: every fresh cell in the cone is excluded by Terra's own dig "
            "masks -- the last-dig exclusion (prevents dump-load cycles), the "
            "max-depth mask, or the spoil-ambiguity rule."
        )
        code = "not_applicable_all_excluded"
    else:
        lines.append(
            "  reason: the selection is non-empty but the dig transition still moves "
            "nothing (bucket-capacity guard, or the load does not fit int8)."
        )
        code = "no_effect"
    return ("DO -> NOTHING HAPPENS (no-op)", code, lines)


def move_lines(probe: dict) -> list[str]:
    mask = np.asarray(probe["action_mask"])
    loaded = int(probe["loaded"])
    out = []
    labels = (
        ("FORWARD", "forward", 0),
        ("BACKWARD", "backward", 1),
        ("BASE CW", "clock", 2),
        ("BASE CCW", "anticlock", 3),
    )
    for label, key, index in labels:
        allowed = bool(mask[index])
        if allowed:
            out.append(f"  {label:9s} legal")
            continue
        if loaded > 0:
            out.append(f"  {label:9s} REFUSED: loaded ({loaded}u) -- no movement while loaded")
            continue
        report = probe[key]
        causes = []
        if not bool(report["in_bounds"]):
            causes.append("footprint leaves the map")
        if int(report["padding"]):
            causes.append(f"footprint hits padding x{int(report['padding'])}")
        if int(report["holes"]):
            causes.append(f"footprint hits dug holes x{int(report['holes'])}")
        if int(report["piles"]):
            causes.append(f"footprint hits blocking spoil x{int(report['piles'])}")
        if not causes:
            causes.append("footprint blocked (traversability)")
        out.append(f"  {label:9s} REFUSED: " + ", ".join(causes))
    out.append(
        f"  CABIN CW  {'legal' if bool(mask[4]) else 'REFUSED'}"
        f"   CABIN CCW {'legal' if bool(mask[5]) else 'REFUSED'}"
        f"   DO {'changes state' if bool(mask[6]) else 'no-op'}"
    )
    return out


def status_lines(probe: dict, slot: int, row: dict, reward: float, done: bool) -> list[str]:
    required = int(probe["required_cells"])
    dug = int(probe["dug_cells"])
    fresh = int(probe["fresh_cells"])
    illegal = int(probe["completion_illegal_dump_volume"])
    purity = float(probe["completion_dump_purity"])
    volume = float(probe["completion_dump_volume_completion"])
    dig = float(probe["completion_dig_completion_total"])
    absolute = float(probe["completion_absolute_completion"])
    loaded = int(probe["loaded"])
    by_cabin = np.asarray(probe["dump_by_cabin"])
    lines = [
        f"SLOT {slot}  {row['primary_cell']}  {row['map_id']}  "
        f"sections={int(probe['trench_type'])}",
        f"step {int(probe['env_steps']):3d}/{int(probe['max_steps'])}   "
        f"reward {reward:+.4f}   gate {'ON' if bool(probe['gate_enabled']) else 'OFF'}"
        f"   {'DONE' if done else ''}",
        f"pose row {int(probe['pos_base'][0]):2d} col {int(probe['pos_base'][1]):2d}   "
        f"base {int(probe['angle_base'])}/12 ({np.degrees(float(probe['base_angle_rad'])):6.1f}deg)"
        f"   cabin {int(probe['angle_cabin'])}/12   loaded {loaded}u",
        f"cells required {required}  dug {dug}  fresh left {fresh}  "
        f"spoil-on-target {int(probe['spoiled_target_cells'])}  dig {dig:.3f}",
        f"dump purity {purity:.3f}  volume {volume:.3f}  "
        f"accepted {int(probe['completion_accepted_dump_volume'])}u  "
        f"ILLEGAL SPOIL {illegal}u ({int(probe['illegal_spoil_cells'])} cells)",
        f"dump legal-in-cone {int(probe['dump_legal'])}  any-in-cone "
        f"{int(probe['dump_physical'])}  cabin angles with legal dump "
        f"{int((by_cabin[:, 0] > 0).sum())}/12  absolute completion {absolute:.3f}",
    ]
    warnings = []
    if illegal > 0:
        warnings.append(
            f"!! PURITY BROKEN: {illegal}u of spoil sit outside target>0. dump_purity "
            "cannot reach 1.0 until they are relifted and dumped inside the zone."
        )
    if fresh == 0 and absolute < 1.0:
        warnings.append(
            "!! every trench cell is dug but the episode is NOT complete "
            f"(purity {purity:.3f}, volume {volume:.3f}, unloaded "
            f"{float(probe['completion_unloaded_completion']):.0f})"
        )
    if loaded > 0 and int(probe["dump_physical"]) == 0 and int((by_cabin.sum(axis=1) > 0).sum()) == 0:
        warnings.append(
            "!! DEADLOCK: loaded, no dump destination at any of the 12 cabin angles, "
            "and an excavator cannot move while loaded."
        )
    if int(probe["env_steps"]) >= int(probe["max_steps"]):
        warnings.append("!! horizon reached")
    return lines + warnings


# --------------------------------------------------------------------------- #
# rendering                                                                   #
# --------------------------------------------------------------------------- #
class View:
    PANEL_WIDTH = 940

    def __init__(self, env, zoom: int):
        self.engine = env.terra_env.rendering_engine
        if self.engine is None:
            raise RuntimeError("TerraEnvBatch was built without rendering")
        self.zoom = zoom
        self.tile = self.engine.tile_size
        self.map_px = self.engine.total_display_size
        self.border_px = 4 * self.tile
        terra_dims = (self.map_px + 2 * self.border_px, self.map_px + 2 * self.border_px)
        # Terra's Game draws into this offscreen surface; we scale it ourselves.
        self.terra_surface = pg.Surface(terra_dims)
        self.engine.screen = self.terra_surface
        self.engine.display = False
        self.cell = self.tile * zoom
        self.map_origin = (12, 12)
        width = self.map_px * zoom + 24 + self.PANEL_WIDTH
        height = max(self.map_px * zoom + 24, 1000)
        pg.font.init()
        self.window = pg.display.set_mode((width, height))
        pg.display.set_caption("Terra fresh-trench dig-alignment debugger")
        self.font = self._font(14)
        self.font_small = self._font(12)
        self._terrain_signature = None
        self._terrain_scaled = None
        self._text_cache: dict[tuple[str, tuple[int, int, int]], pg.Surface] = {}
        self.show_overlays = True
        self.show_geometry = True
        self.show_tint = True
        self.show_help = False
        self.splash(
            [
                "Terra fresh-trench dig-alignment debugger",
                "",
                "starting up ...",
            ]
        )

    @staticmethod
    def _font(size: int):
        for name in ("dejavusansmono", "liberationmono", "couriernew", "monospace"):
            path = pg.font.match_font(name)
            if path:
                return pg.font.Font(path, size)
        return pg.font.Font(None, size + 4)

    def cell_rect(self, row: int, col: int) -> pg.Rect:
        ox, oy = self.map_origin
        return pg.Rect(
            ox + int(col * self.cell), oy + int(row * self.cell), self.cell, self.cell
        )

    def point(self, row: float, col: float) -> tuple[int, int]:
        """Cell coordinates -> window pixels, at the CELL CENTRE.

        Terra's terrain tiles span [grid*tile, grid*tile+tile] while its agent
        glyph is centred on grid*tile, i.e. the glyph sits half a cell up-left
        of the terrain lattice.  Overlays follow the terrain, so line geometry
        (axes, standoff bands, heading) is drawn through cell centres and lands
        half a cell off the glyph.
        """
        ox, oy = self.map_origin
        return (
            int(ox + (col + 0.5) * self.cell),
            int(oy + (row + 0.5) * self.cell),
        )

    def _blit_map(self, env, timestep) -> None:
        """Blit Terra's own terrain rendering, cached on the terrain itself.

        Terra's Game walks 64x64 cells in Python twice per frame (~100 ms), so
        re-running it for a pure pose change would eat the whole frame budget.
        The terrain only changes on an effective DO, so it is rendered on a
        terrain hash and cached; the machine, its cone and the pose geometry are
        drawn as overlays on the same lattice.  The agent glyph and Terra's red
        cone are suppressed (agent_active=0, interaction_mask=0) so the cached
        image stays valid for any pose.
        """
        observation = timestep.observation
        action_map = np.asarray(observation["action_map"])
        dumpability = np.asarray(observation["dumpability_mask"])
        signature = hashlib.blake2b(
            action_map.tobytes() + dumpability.tobytes(), digest_size=8
        ).digest()
        if signature != self._terrain_signature:
            zeros_int = np.zeros_like(np.asarray(observation["agent_active"]))
            self.engine.run(
                active_grid=action_map,
                target_grid=np.asarray(observation["target_map"]),
                padding_mask=np.asarray(observation["padding_mask"]),
                dumpability_mask=dumpability,
                interaction_mask=np.zeros_like(
                    np.asarray(observation["interaction_mask"])
                ),
                agent_states=np.asarray(observation["agent_states"]),
                agent_active=zeros_int,
                num_agents=np.zeros((zeros_int.shape[0],), dtype=np.int32),
                generate_gif=False,
                target_tiles=None,
            )
            self._terrain_scaled = pg.transform.scale(
                self.terra_surface,
                (
                    self.terra_surface.get_width() * self.zoom,
                    self.terra_surface.get_height() * self.zoom,
                ),
            )
            self._terrain_signature = signature
        # Terra insets its map by border_px inside its own surface.
        self.window.blit(
            self._terrain_scaled,
            (self.map_origin[0] - self.border_px * self.zoom,
             self.map_origin[1] - self.border_px * self.zoom),
        )

    def splash(self, lines: list[str], env=None, timestep=None) -> None:
        """Paint the window during startup so it is never a black rectangle."""
        self.window.fill((24, 24, 28))
        if env is not None and timestep is not None:
            self._blit_map(env, timestep)
            # Terra's sprite is suppressed and the probe (which owns the exact
            # footprint) has not compiled yet, so mark the machine directly from
            # the observation so the startup frame is not a map with no machine.
            state = np.asarray(timestep.observation["agent_states"])[0][0]
            row, col = float(state[0]), float(state[1])
            angle = 2.0 * np.pi * float(state[2]) / ANGLES_BASE
            arm = 2.0 * np.pi * (float(state[2]) + float(state[3])) / ANGLES_BASE
            centre = self.point(row, col)
            pg.draw.circle(self.window, (0, 43, 91), centre, int(2.5 * self.cell))
            pg.draw.circle(self.window, (255, 255, 255), centre, int(2.5 * self.cell), 2)
            pg.draw.line(
                self.window,
                (255, 255, 255),
                centre,
                self.point(row - 4.5 * np.sin(angle), col + 4.5 * np.cos(angle)),
                3,
            )
            pg.draw.line(
                self.window,
                (255, 120, 120),
                centre,
                self.point(row - 6.5 * np.sin(arm), col + 6.5 * np.cos(arm)),
                2,
            )
        self.draw_panel(lines)
        pg.event.pump()
        pg.display.flip()

    def draw(self, env, timestep, probe: dict, texts: list[str]) -> None:
        self.window.fill((24, 24, 28))
        self._blit_map(env, timestep)
        if self.show_overlays:
            self.draw_overlays(probe)
        self.draw_panel(texts)
        if self.show_help:
            self.draw_help()
        pg.display.flip()

    def _cells(self, mask, color, width=1, inset=0):
        rows, cols = np.nonzero(mask)
        for row, col in zip(rows, cols):
            rect = self.cell_rect(int(row), int(col))
            if inset:
                rect = rect.inflate(-inset, -inset)
            pg.draw.rect(self.window, color, rect, width)

    def draw_overlays(self, probe: dict) -> None:
        n_axes = int(probe["trench_type"])

        # accepted dump zone
        self._cells(probe["accepted_mask"], (60, 130, 255), 1)

        # remaining trench cells tinted per owning section, junctions dotted
        if self.show_tint:
            member = np.asarray(probe["section_member"])
            owner = np.asarray(probe["owner_count"])
            remaining = np.asarray(probe["remaining_mask"])
            for axis in range(n_axes):
                self._cells(
                    member[axis] & remaining & (owner == 1),
                    SECTION_COLORS[axis % 4],
                    1,
                    inset=4,
                )
            rows, cols = np.nonzero(remaining & (owner > 1))
            for row, col in zip(rows, cols):
                pg.draw.circle(
                    self.window, (255, 255, 255), self.point(int(row), int(col)), 2
                )

        # padding inside the cone: the hard blocker
        self._cells(probe["cone_padding_mask"], (255, 0, 0), 2)

        # illegal spoil, loud
        rows, cols = np.nonzero(probe["illegal_spoil_mask"])
        for row, col in zip(rows, cols):
            rect = self.cell_rect(int(row), int(col))
            pg.draw.rect(self.window, (255, 0, 200), rect, 0)
            pg.draw.line(self.window, (0, 0, 0), rect.topleft, rect.bottomright, 2)
            pg.draw.line(self.window, (0, 0, 0), rect.topright, rect.bottomleft, 2)

        # cone, admitted cells, blocked cells
        self._cells(probe["cone"], (255, 255, 120), 2)
        rows, cols = np.nonzero(probe["admitted_mask"] & probe["fresh_trench"])
        for row, col in zip(rows, cols):
            pg.draw.rect(self.window, (0, 255, 120), self.cell_rect(int(row), int(col)), 0)
        rows, cols = np.nonzero(probe["blocked_cells"])
        for row, col in zip(rows, cols):
            rect = self.cell_rect(int(row), int(col))
            pg.draw.rect(self.window, (255, 140, 0), rect, 0)
            pg.draw.line(self.window, (60, 30, 0), rect.topleft, rect.bottomright, 1)

        # geometry: sections, axis direction, standoff band edges
        if self.show_geometry:
            for axis in range(n_axes):
                color = SECTION_COLORS[axis % 4]
                a, b, c = [float(v) for v in probe["axes"][axis]]
                seg = [float(v) for v in probe["segments"][axis]]
                pose_valid = bool(probe["axis_pose_valid"][axis])
                start = self.point(seg[0], seg[1])
                end = self.point(seg[2], seg[3])
                pg.draw.line(self.window, color, start, end, 3 if pose_valid else 1)
                if not pose_valid:
                    pg.draw.circle(self.window, color, start, 4, 1)
                    pg.draw.circle(self.window, color, end, 4, 1)
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                # axis direction: the tangent the chassis yaw is compared against
                denom = float(np.hypot(a, b)) or 1.0
                tangent = np.array([-a, b]) / denom  # (row, col)
                tip = self.point(
                    (seg[0] + seg[2]) / 2.0 + 3.0 * tangent[0],
                    (seg[1] + seg[3]) / 2.0 + 3.0 * tangent[1],
                )
                pg.draw.line(self.window, color, mid, tip, 2)
                pg.draw.circle(self.window, color, tip, 3)
                label = self.font_small.render(
                    f"S{axis} {np.degrees(float(probe['yaw_errors_rad'][axis])):.0f}deg "
                    f"{float(probe['standoffs_m'][axis]):.1f}m",
                    True,
                    color,
                    (0, 0, 0),
                )
                self.window.blit(label, (mid[0] + 8, mid[1] - 8))
                self._draw_band(probe, axis, color)

        # the machine, drawn from Terra's own rasterised footprint so it sits on
        # the same lattice as the cells the env actually tests
        self._cells(probe["footprint"], (0, 43, 91), 0)
        self._cells(probe["footprint"], (255, 255, 255), 1)
        row = float(probe["pos_base"][0])
        col = float(probe["pos_base"][1])
        centre = self.point(row, col)
        angle = float(probe["base_angle_rad"])
        head = self.point(row - 4.5 * np.sin(angle), col + 4.5 * np.cos(angle))
        pg.draw.line(self.window, (255, 255, 255), centre, head, 3)
        pg.draw.circle(self.window, (255, 255, 255), head, 4)
        # where the cabin (and therefore the workspace cone) points
        arm = 2.0 * np.pi * (
            int(probe["angle_base"]) + int(probe["angle_cabin"])
        ) / ANGLES_BASE
        arm_tip = self.point(row - 6.5 * np.sin(arm), col + 6.5 * np.cos(arm))
        pg.draw.line(self.window, (255, 120, 120), centre, arm_tip, 2)
        pg.draw.circle(self.window, (255, 120, 120), arm_tip, 3)

    def _draw_band(self, probe: dict, axis: int, color) -> None:
        """Draw the 3.5 m / 7.0 m standoff band edges for one section."""
        a, b, c = [float(v) for v in probe["axes"][axis]]
        denom = float(np.hypot(a, b))
        if denom < 1e-6:
            return
        tile = float(probe["tile_size"])
        seg = [float(v) for v in probe["segments"][axis]]
        # unit tangent in (row, col)
        tangent = np.array([-a, b]) / denom
        normal = np.array([b, a]) / denom  # (row, col) normal to the line
        p0 = np.array([seg[0], seg[1]])
        p1 = np.array([seg[2], seg[3]])
        for metres in (float(probe["standoff_min_m"]), float(probe["standoff_max_m"])):
            offset_cells = metres / tile
            for sign in (-1.0, 1.0):
                q0 = p0 + sign * offset_cells * normal
                q1 = p1 + sign * offset_cells * normal
                self._dashed(self.point(*q0), self.point(*q1), color)

    def _dashed(self, start, end, color, dash=7, gap=6) -> None:
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        length = float(np.hypot(*(end - start)))
        if length < 1e-6:
            return
        direction = (end - start) / length
        position = 0.0
        while position < length:
            a = start + direction * position
            b = start + direction * min(position + dash, length)
            pg.draw.line(self.window, color, a, b, 1)
            position += dash + gap

    def _wrap(self, line: str, limit: int) -> list[str]:
        if self.font.size(line)[0] <= limit:
            return [line]
        indent = " " * (len(line) - len(line.lstrip()) + 4)
        out: list[str] = []
        current = ""
        for word in line.split(" "):
            candidate = word if not current else current + " " + word
            if self.font.size(candidate)[0] > limit and current:
                out.append(current)
                current = indent + word
            else:
                current = candidate
        if current:
            out.append(current)
        return out

    def draw_panel(self, texts: list[str]) -> None:
        x = self.map_px * self.zoom + 24
        pg.draw.rect(
            self.window, (14, 14, 18), pg.Rect(x - 8, 0, self.PANEL_WIDTH + 8, self.window.get_height())
        )
        y = 10
        wrapped: list[str] = []
        for line in texts:
            wrapped.extend(self._wrap(line, self.PANEL_WIDTH - 16))
        for line in wrapped:
            color = (235, 235, 235)
            if line.strip().startswith("!!"):
                color = (255, 90, 90)
            elif "REFUSED" in line or "VETO" in line or "OUTSIDE" in line:
                color = (255, 170, 60)
            elif "ADMITTED" in line or "DUMP" in line:
                color = (120, 255, 150)
            elif line.startswith("---") or line.startswith("SLOT"):
                color = (150, 200, 255)
            key = (line, color)
            surface = self._text_cache.get(key)
            if surface is None:
                if len(self._text_cache) > 4000:
                    self._text_cache.clear()
                surface = self.font.render(line, True, color)
                self._text_cache[key] = surface
            self.window.blit(surface, (x, y))
            y += 17

    def draw_help(self) -> None:
        lines = __doc__.split("KEYS\n----\n", 1)[1].split("RECOMMENDED")[0].splitlines()
        surface = pg.Surface((self.window.get_width() - 40, 30 + 15 * len(lines)))
        surface.set_alpha(235)
        surface.fill((10, 10, 30))
        for index, line in enumerate(lines):
            surface.blit(self.font_small.render(line, True, (230, 230, 255)), (12, 10 + 15 * index))
        self.window.blit(surface, (20, 20))


# --------------------------------------------------------------------------- #
# session                                                                     #
# --------------------------------------------------------------------------- #
def state_digest(timestep) -> str:
    parts = []
    state = timestep.state
    for value in (
        state.world.action_map.map,
        state.world.target_map.map,
        state.world.dumpability_mask.map,
        state.world.last_dig_mask.map,
        state.agent.agent_states[0].pos_base,
        state.agent.agent_states[0].angle_base,
        state.agent.agent_states[0].angle_cabin,
        state.agent.agent_states[0].loaded,
        state.env_steps,
    ):
        parts.append(np.ascontiguousarray(np.asarray(value)).tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()[:16]


KEY_ACTIONS = {
    pg.K_UP: ("FORWARD", "forward"),
    pg.K_w: ("FORWARD", "forward"),
    pg.K_DOWN: ("BACKWARD", "backward"),
    pg.K_s: ("BACKWARD", "backward"),
    pg.K_LEFT: ("ANTICLOCK", "anticlock"),
    pg.K_a: ("ANTICLOCK", "anticlock"),
    pg.K_RIGHT: ("CLOCK", "clock"),
    pg.K_d: ("CLOCK", "clock"),
    pg.K_q: ("CABIN_ANTICLOCK", "cabin_anticlock"),
    pg.K_e: ("CABIN_CLOCK", "cabin_clock"),
    pg.K_SPACE: ("DO", "do"),
    pg.K_RETURN: ("DO", "do"),
    pg.K_n: ("DO_NOTHING", "do_nothing"),
}

SCRIPT_KEYS = {
    "UP": pg.K_UP,
    "DOWN": pg.K_DOWN,
    "LEFT": pg.K_LEFT,
    "RIGHT": pg.K_RIGHT,
    "W": pg.K_w,
    "S": pg.K_s,
    "A": pg.K_a,
    "D": pg.K_d,
    "Q": pg.K_q,
    "E": pg.K_e,
    "SPACE": pg.K_SPACE,
    "RETURN": pg.K_RETURN,
    "N": pg.K_n,
    "U": pg.K_u,
    "BACKSPACE": pg.K_BACKSPACE,
    "R": pg.K_r,
    "O": pg.K_o,
    "G": pg.K_g,
    "T": pg.K_t,
    "H": pg.K_h,
    "ESC": pg.K_ESCAPE,
    "1": pg.K_1,
    "2": pg.K_2,
    "3": pg.K_3,
    "4": pg.K_4,
    "5": pg.K_5,
    "LBRACKET": pg.K_LEFTBRACKET,
    "RBRACKET": pg.K_RIGHTBRACKET,
}


class Session:
    def __init__(self, args):
        self.args = args
        self.rows = load_manifest(args.bank, args.panel)
        self.slot_count = len(self.rows)
        print(f"panel {args.panel}: {self.slot_count} slots", flush=True)

        t0 = time.time()
        self.env, self.env_cfgs, self.batch_cfg = build_env(
            args.bank, args.panel, self.slot_count, not args.gate_off, rendering=True
        )
        print(f"env built in {time.time() - t0:.1f}s", flush=True)
        self.loader = SlotLoader(self.env, self.env_cfgs, args.bank, args.panel, self.rows)
        self.probe_fn = make_probe(self.batch_cfg)
        self.action_type = self.batch_cfg.action_type
        self.step_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray([0], dtype=jnp.uint32))

        self.slot = args.slot
        self.timestep = None
        self.stack: list = []
        self.probe = None
        self.reward = 0.0
        self.done = False
        self.seq = 0
        self.timing = {"step": 0.0, "probe": 0.0, "render": 0.0}
        self.dirty = True
        self.last_event = "reset"
        self.verdict = ""
        self.reason_code = ""
        self.detail: list[str] = []

        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"manual_trench_slot{args.slot}_{stamp}.jsonl"
        self.log_file = self.log_path.open("w")
        print(f"session log -> {self.log_path}", flush=True)

        self.view = View(self.env, args.zoom)
        pg.key.set_repeat(250, 60)

        self.load_slot(self.slot, warm=True)

    # ---------------- environment plumbing ---------------- #
    def load_slot(self, slot: int, warm: bool = False) -> None:
        t0 = time.time()
        self.slot = slot
        row = self.rows[slot - 1]
        header = [
            f"SLOT {slot}  {row['primary_cell']}  {row['map_id']}",
            "",
        ]
        self.view.splash(
            header + [f"loading slot {slot} from the frozen panel ..."]
        )
        self.timestep, _ = self.loader.reset(slot)
        jax.block_until_ready(self.timestep.reward)
        if warm:
            print(f"reset compiled in {time.time() - t0:.1f}s", flush=True)
            print("compiling the diagnostic probe (one-off; cached)...", flush=True)
        self.stack = []
        self.reward = 0.0
        self.done = False
        self.last_event = f"load_slot_{slot}"
        self.view.splash(
            header
            + [
                "map loaded.  Compiling the gate/refusal probe (~55 s on CPU,",
                "one trace per launch).  The window stays up; keys are queued.",
            ],
            self.env,
            self.timestep,
        )
        t1 = time.time()
        self.refresh(assert_replica=True)
        if warm:
            print(f"probe compiled in {time.time() - t1:.1f}s", flush=True)
            self.view.splash(
                header
                + [
                    "probe ready.  Compiling Terra's env step (~105 s on CPU,",
                    "one trace per launch).  Play starts as soon as it lands.",
                ],
                self.env,
                self.timestep,
            )
            # Terra's reset and step timesteps do not share one pytree/aval
            # signature (the reset info block is a zero stand-in), so the step
            # is traced twice: once from a reset state and once from a stepped
            # state.  Pay both here instead of during play.
            action = self.wrap(self.action_type.do_nothing())
            for index in range(2):
                t2 = time.time()
                warm_timestep = self.env.step_no_reset(
                    self.timestep if index == 0 else warm_timestep,
                    action,
                    self.step_keys,
                )
                jax.block_until_ready(warm_timestep.reward)
                print(
                    f"step variant {index} compiled in {time.time() - t2:.1f}s",
                    flush=True,
                )
            # discard the warm-up transitions: play starts at the frozen start
            self.timestep, _ = self.loader.reset(slot)
            self.refresh(assert_replica=True)
        self.print_startup()
        self.log("START")

    def wrap(self, action):
        return action.new(action.action[None].repeat(1, 0))

    def refresh(self, assert_replica: bool = True) -> None:
        t0 = time.time()
        raw = self.probe_fn(self.timestep.state)
        self.probe = to_host(raw)
        self.timing["probe"] = 1000 * (time.time() - t0)
        if assert_replica:
            check_replica(self.probe, self.timestep.observation)
        self.verdict, self.reason_code, self.detail = explain_do(self.probe)

    def apply(self, name: str) -> None:
        action = getattr(self.action_type, name.lower())()
        self.stack.append((self.timestep, self.reward, self.done))
        if len(self.stack) > 600:
            self.stack.pop(0)
        t0 = time.time()
        self.timestep = self.env.step_no_reset(
            self.timestep, self.wrap(action), self.step_keys
        )
        jax.block_until_ready(self.timestep.reward)
        self.timing["step"] = 1000 * (time.time() - t0)
        self.reward = float(np.asarray(self.timestep.reward).reshape(-1)[0])
        self.done = bool(np.asarray(self.timestep.done).reshape(-1)[0])
        self.last_event = name
        self.refresh(assert_replica=True)
        self.log(name)

    def undo(self) -> None:
        if not self.stack:
            self.last_event = "undo_empty"
            return
        self.timestep, self.reward, self.done = self.stack.pop()
        self.last_event = "undo"
        self.refresh(assert_replica=True)
        self.log("UNDO")

    # ---------------- reporting ---------------- #
    def print_startup(self) -> None:
        row = self.rows[self.slot - 1]
        probe = self.probe
        print("=" * 78, flush=True)
        print(
            f"SLOT {self.slot} / {self.slot_count}   condition {row['primary_cell']}   "
            f"family {row['family']}   map {row['map_id']}",
            flush=True,
        )
        print(
            f"  finite sections {int(probe['trench_type'])}   required cells "
            f"{int(probe['required_cells'])}   accepted dump cells "
            f"{int(probe['accepted_mask'].sum())}   horizon {int(probe['max_steps'])}",
            flush=True,
        )
        print(
            f"  gate {'ON' if bool(probe['gate_enabled']) else 'OFF'}  "
            f"yaw tol {np.degrees(float(probe['yaw_tolerance_rad'])):.2f}deg  "
            f"standoff band {float(probe['standoff_min_m'])}-{float(probe['standoff_max_m'])} m  "
            f"tile {float(probe['tile_size']):.4f} m",
            flush=True,
        )
        print(
            f"  bank {self.args.bank}\n  protocol terra_revision {BANK_TERRA_REVISION}, "
            f"reset_seed {row['reset_seed']}",
            flush=True,
        )
        for label, oracle in oracle_rows(self.slot):
            print(
                f"  ORACLE [{label}]: dug {oracle['dug_cells']}/{oracle['required_cells']} "
                f"(dig {oracle['dig_fraction']:.3f})  succeeded {oracle['succeeded']}  "
                f"spoil {oracle['positive_soil']}u of which accepted "
                f"{oracle['accepted_soil']}u  stations {oracle['stations']}  "
                f"digs {oracle['digs']} dumps {oracle['dumps']}  reasons {oracle['reasons']}",
                flush=True,
            )
        print(
            "  YOUR TARGET: beat the oracle's dug-cell count, and finish with every "
            "spoil unit inside the accepted zone.",
            flush=True,
        )
        print("=" * 78, flush=True)

    def texts(self) -> list[str]:
        row = self.rows[self.slot - 1]
        lines = status_lines(self.probe, self.slot, row, self.reward, self.done)
        lines.append("-" * 74)
        lines.append(self.verdict)
        lines.extend(self.detail)
        lines.append("-" * 74)
        lines.append("legal actions:")
        lines.extend(move_lines(self.probe))
        lines.append("-" * 74)
        for label, oracle in oracle_rows(self.slot)[:1]:
            lines.append(
                f"oracle [{label}]: {oracle['dug_cells']}/{oracle['required_cells']} cells, "
                f"succeeded={oracle['succeeded']}"
            )
        lines.append(
            f"last {self.last_event}   undo depth {len(self.stack)}   "
            f"step {self.timing['step']:.0f}ms probe {self.timing['probe']:.0f}ms "
            f"render {self.timing['render']:.0f}ms"
        )
        lines.append(
            "arrows/WASD drive | Q/E cabin | SPACE DO | N nothing | U undo | R reset"
        )
        lines.append(
            "1-5 slots 296/405/294/455/458 | [ ] slot -/+ | O overlays | G geometry | "
            "T tint | H help | ESC quit"
        )
        return lines

    def log(self, event: str) -> None:
        probe = self.probe
        record = {
            "seq": self.seq,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "slot": self.slot,
            "condition": self.rows[self.slot - 1]["primary_cell"],
            "event": event,
            "env_steps": int(probe["env_steps"]),
            "reward": self.reward,
            "done": self.done,
            "pose": {
                "row": int(probe["pos_base"][0]),
                "col": int(probe["pos_base"][1]),
                "angle_base": int(probe["angle_base"]),
                "angle_cabin": int(probe["angle_cabin"]),
                "loaded": int(probe["loaded"]),
            },
            "do_verdict": self.verdict,
            "do_reason": self.reason_code,
            "do_detail": self.detail,
            "gate": {
                "valid": bool(probe["terra_valid"]),
                "applicable": bool(probe["applicable"]),
                "yaw_error_normalized": float(probe["terra_yaw_norm"]),
                "standoff_error_normalized": float(probe["terra_standoff_norm"]),
                "yaw_errors_deg": [
                    round(float(np.degrees(v)), 3)
                    for v in probe["yaw_errors_rad"][: int(probe["trench_type"])]
                ],
                "standoffs_m": [
                    round(float(v), 3)
                    for v in probe["standoffs_m"][: int(probe["trench_type"])]
                ],
                "axis_pose_valid": [
                    bool(v) for v in probe["axis_pose_valid"][: int(probe["trench_type"])]
                ],
                "exclusive_blocked": [
                    int(v)
                    for v in probe["per_axis_exclusive_blocked"][: int(probe["trench_type"])]
                ],
            },
            "status": {
                "required_cells": int(probe["required_cells"]),
                "dug_cells": int(probe["dug_cells"]),
                "fresh_cells": int(probe["fresh_cells"]),
                "dig_fraction": float(probe["completion_dig_completion_total"]),
                "dump_purity": float(probe["completion_dump_purity"]),
                "dump_volume_completion": float(probe["completion_dump_volume_completion"]),
                "absolute_completion": float(probe["completion_absolute_completion"]),
                "illegal_spoil_units": int(probe["completion_illegal_dump_volume"]),
                "accepted_dump_volume": int(probe["completion_accepted_dump_volume"]),
                "legal_dump_in_cone": int(probe["dump_legal"]),
                "cabin_angles_with_legal_dump": int(
                    (np.asarray(probe["dump_by_cabin"])[:, 0] > 0).sum()
                ),
            },
            "action_mask": [bool(v) for v in probe["action_mask"]],
            "digest": state_digest(self.timestep),
            "timing_ms": {k: round(v, 2) for k, v in self.timing.items()},
        }
        self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()
        self.seq += 1
        print(
            f"[{record['seq']:04d}] {event:14s} step {record['env_steps']:3d} "
            f"r {self.reward:+.4f} | {self.verdict}",
            flush=True,
        )
        for line in self.detail:
            print("        " + line.strip(), flush=True)

    # ---------------- main loop ---------------- #
    def handle_key(self, key: int) -> bool:
        self.dirty = True
        if key in (pg.K_ESCAPE,):
            return False
        if key in KEY_ACTIONS:
            self.apply(KEY_ACTIONS[key][0])
        elif key in (pg.K_u, pg.K_BACKSPACE):
            self.undo()
        elif key == pg.K_r:
            self.load_slot(self.slot)
        elif key == pg.K_o:
            self.view.show_overlays = not self.view.show_overlays
            self.last_event = f"overlays_{self.view.show_overlays}"
        elif key == pg.K_g:
            self.view.show_geometry = not self.view.show_geometry
            self.last_event = f"geometry_{self.view.show_geometry}"
        elif key == pg.K_t:
            self.view.show_tint = not self.view.show_tint
            self.last_event = f"tint_{self.view.show_tint}"
        elif key == pg.K_h:
            self.view.show_help = not self.view.show_help
            self.last_event = f"help_{self.view.show_help}"
        elif key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5):
            index = (pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5).index(key)
            self.load_slot(RECOMMENDED[index])
        elif key == pg.K_LEFTBRACKET:
            self.load_slot(max(1, self.slot - 1))
        elif key == pg.K_RIGHTBRACKET:
            self.load_slot(min(self.slot_count, self.slot + 1))
        return True

    def screenshot(self, name: str) -> None:
        path = self.log_dir / f"{name}.png"
        pg.image.save(self.view.window, str(path))
        print(f"screenshot -> {path}", flush=True)

    def run(self, script: list[str] | None, script_delay: float) -> None:
        playing = True
        pending = list(script or [])
        last_draw = 0.0
        while playing:
            if script is not None and not pending and not self.args.hold:
                break
            if pending:
                token = pending.pop(0)
                if token.startswith("SHOT"):
                    self.render_once()
                    last_draw = time.time()
                    self.screenshot(
                        token.split(":", 1)[1] if ":" in token else f"shot{self.seq}"
                    )
                    continue
                if token == "WAIT":
                    self.dirty = True
                elif token not in SCRIPT_KEYS:
                    raise SystemExit(f"unknown script token {token!r}")
                else:
                    pg.event.post(
                        pg.event.Event(pg.KEYDOWN, key=SCRIPT_KEYS[token], mod=0)
                    )
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    playing = False
                elif event.type == pg.KEYDOWN:
                    if not self.handle_key(event.key):
                        playing = False
            # Only recompose the window when something changed: Terra's own
            # renderer walks 64x64 cells in Python, so redrawing at frame rate
            # would waste the whole budget for nothing.
            if self.dirty or (time.time() - last_draw) > 1.0:
                self.render_once()
                self.dirty = False
                last_draw = time.time()
            if script is not None and script_delay:
                time.sleep(script_delay)
            elif script is None:
                pg.time.wait(8)
        self.log_file.close()
        print(f"session log written: {self.log_path}", flush=True)

    def render_once(self) -> None:
        t0 = time.time()
        self.view.draw(self.env, self.timestep, self.probe, self.texts())
        self.timing["render"] = 1000 * (time.time() - t0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Human-playable Terra fresh-trench dig-alignment debugger.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slot", type=int, default=458, help="panel slot (1-based)")
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument(
        "--zoom", type=int, default=5, help="map magnification (3*zoom px per cell)"
    )
    parser.add_argument(
        "--log-dir", default=str(WORKTREE / "data" / "manual_trench_logs")
    )
    parser.add_argument(
        "--jax-cache", default=str(WORKTREE / "data" / "jax_compile_cache")
    )
    parser.add_argument("--gate-off", action="store_true", help="disable the gate (A/B)")
    parser.add_argument("--headless", action="store_true", help="SDL dummy driver (tests only)")
    parser.add_argument("--script", default=None, help="comma-separated key tokens")
    parser.add_argument("--script-delay", type=float, default=0.0)
    parser.add_argument(
        "--hold", action="store_true", help="stay interactive after --script finishes"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    cache = Path(args.jax_cache)
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

    script = None
    if args.script:
        script = [t.strip().upper() for t in args.script.replace(",", " ").split() if t.strip()]

    session = Session(args)
    session.run(script, args.script_delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
