#!/usr/bin/env python3
"""Targeted checks of the two junction clauses of the fresh-trench gate contract.

The research note's contract item 4 says:

    Every selected fresh trench cell must have at least one pose-valid owning
    section.  Shared junction cells may use either section.  If the same macro
    cone contains an exclusive cell from a perpendicular, invalid branch,
    reject the complete DO.

``terra/tests/test_trench_dig_alignment.py`` already covers the veto and the
"finish one branch, then the other" resolution.  It does *not* isolate the
middle sentence -- that a cell owned by *both* sections is diggable from a pose
valid for either one.  That is the clause an over-restrictive implementation
would silently drop (e.g. by using an argmin owner instead of a bitmask), so it
is checked directly here, together with the +0.5 membership slack.

Both junction clauses are independent of the standoff semantics, so the checks
run under whichever ``EnvConfig.trench_dig_standoff_enforced`` says (v2, the
default) and under ``--gate-v1`` (the retired perpendicular band).  What DOES
change with the semantics is where a machine has to stand to be pose-valid for
a section, so the lane position is chosen per semantics: the v1 lateral lane
(8 tiles = 4.57 m off the axis, inside [3.5, 7.0] m) under ``--gate-v1``, and
the ON-AXIS lane (offset 0, the dig-ahead-retreat pose) under v2 with the "on
the line" clause active.  Running it under both is the point: the junction
verdicts must not move.

Check 6 pins the clause itself: under v2 with a finite
``EnvConfig.trench_dig_max_offset_m`` the v1 lateral lane must be REFUSED, and
with the clause disabled (``--max-offset-m 0``) it must be admitted again.
See TRENCH_GATE_STANDOFF_SEMANTICS_BUG_20260901.md.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=<terra> python tools/check_trench_gate_multiowner.py
  JAX_PLATFORMS=cpu PYTHONPATH=<terra> python tools/check_trench_gate_multiowner.py --gate-v1
  JAX_PLATFORMS=cpu PYTHONPATH=<terra> python tools/check_trench_gate_multiowner.py --max-offset-m 0
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np

from terra.map import compute_trench_axis_membership
from terra.tests.test_trench_dig_alignment import FreshTrenchDigAlignmentTest as T

SHAPE = T.SHAPE
FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {name} {detail}")
    if not condition:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-v1", action="store_true",
                    help="force the retired v1 semantics (perpendicular "
                         "standoff band enforced on top of yaw-parallel)")
    ap.add_argument("--max-offset-m", type=float, default=None,
                    help="override the v2 'on the line' bound "
                         "(EnvConfig.trench_dig_max_offset_m, metres); "
                         "<= 0 disables the clause (yaw-parallel only). "
                         "Inert under --gate-v1.")
    args = ap.parse_args()
    T.setUpClass()
    cfg = T.cfg_v1 if args.gate_v1 else T.cfg
    if args.max_offset_m is not None:
        cfg = cfg._replace(trench_dig_max_offset_m=float(args.max_offset_m))
    semantics = "v1" if cfg.trench_dig_standoff_enforced else "v2"
    max_offset = float(cfg.trench_dig_max_offset_m)
    on_line = (not cfg.trench_dig_standoff_enforced) and max_offset > 0.0
    print(f"tile_size={cfg.tile_size:.6f} agent={cfg.agent.width}x{cfg.agent.height} "
          f"gate={cfg.enforce_trench_dig_alignment} "
          f"semantics={semantics} "
          f"standoff_band_enforced={cfg.trench_dig_standoff_enforced} "
          f"yaw_tol={cfg.trench_dig_yaw_tolerance_rad} "
          f"standoff=[{cfg.trench_dig_standoff_min_m}, {cfg.trench_dig_standoff_max_m}] "
          f"max_offset_m={max_offset} on_line_clause={on_line}")
    # The lane a machine can legally occupy for a section depends on the
    # semantics.  v1: the lateral band, 8 tiles = 4.57 m off the axis.  v2 with
    # the on-the-line clause: the axis itself.  With the clause disabled either
    # lane is legal; the on-axis one is used, so the run stays comparable.
    lateral_lane = {"A (row=24)": (0, (32, 40)), "B (col=40)": (3, (24, 32))}
    on_axis_lane = {"A (row=24)": (0, (24, 32)), "B (col=40)": (3, (32, 40))}
    lanes = lateral_lane if cfg.trench_dig_standoff_enforced else on_axis_lane
    lane_name = "v1 lateral (4.57 m off axis)" if cfg.trench_dig_standoff_enforced \
        else "on-axis (offset 0)"
    print(f"lane model: {lane_name}")

    # A cross: horizontal section at row 24, vertical section at col 40.
    # (24, 40) is the shared cell -- within half-width of BOTH sections.
    axes = T._axes(
        [0, 1, -24, 24, 20, 24, 50, 1],    # A*col + B*row + C = 0  ->  row = 24
        [1, 0, -40, 16, 40, 42, 40, 1],    # col = 40
    )
    shared = (24, 40)

    print("\n1. membership bitmask at the shared junction cell")
    target = np.zeros(SHAPE, dtype=np.int8)
    target[shared] = -1
    membership = np.asarray(compute_trench_axis_membership(
        jnp.asarray(target), jnp.asarray(axes), jnp.int32(2)
    )).astype(np.uint8)
    bits = int(membership[shared])
    check("shared cell owned by BOTH sections (bitmask 0b11)", bits == 0b11,
          f"bitmask={bits:#05b}")

    def cabin_containing(state_maker, cell):
        """Smallest cabin index whose cone covers ``cell`` (cabin is relative)."""
        for cb in range(12):
            st = state_maker(cb)
            cone = np.asarray(st._build_dig_dump_cone()).reshape(SHAPE)
            if cone[cell]:
                return cb, st
        return None, None

    print("\n2. the shared cell is diggable from a pose valid for EITHER section")
    for label, (bh, pos) in lanes.items():
        cb, st = cabin_containing(
            lambda c: T._state(target, axes, base_angle=bh, cabin_angle=c,
                               position=pos, cfg=cfg),
            shared,
        )
        check(f"a cabin heading reaches the shared cell from the {label} lane",
              cb is not None, f"cabin={cb}")
        if cb is None:
            continue
        valid, _, _ = st._get_fresh_trench_dig_alignment()
        after = np.asarray(st._handle_do().world.action_map.map).reshape(SHAPE)
        check(f"gate admits the DO from the {label} lane", bool(valid))
        check(f"shared cell dug from the {label} lane", int(after[shared]) == -1,
              f"action_map{shared}={int(after[shared])}")

    print("\n3. an exclusive perpendicular cell in the same cone still vetoes")
    target2 = np.zeros(SHAPE, dtype=np.int8)
    target2[shared] = -1
    # Inside the 3.64-6.50 m annulus from the A lane, and 2 tiles off section A
    # so it is owned by B only.
    exclusive = (22, 40)
    target2[exclusive] = -1
    a_bh, a_pos = lanes["A (row=24)"]
    found = None
    for cb in range(12):
        st = T._state(target2, axes, base_angle=a_bh, cabin_angle=cb,
                      position=a_pos, cfg=cfg)
        cone = np.asarray(st._build_dig_dump_cone()).reshape(SHAPE)
        if cone[shared] and cone[exclusive]:
            found = (cb, st)
            break
    check("a cone from the A lane holds both the shared and the exclusive-B cell",
          found is not None, f"cabin={found[0] if found else None}")
    if found is not None:
        cb, st = found
        m2 = np.asarray(compute_trench_axis_membership(
            jnp.asarray(target2), jnp.asarray(axes), jnp.int32(2))).astype(np.uint8)
        check("the exclusive cell is owned by B only", int(m2[exclusive]) == 0b10,
              f"bitmask={int(m2[exclusive]):#05b}")
        valid_m, _, _ = st._get_fresh_trench_dig_alignment()
        after = np.asarray(st._handle_do().world.action_map.map).reshape(SHAPE)
        check("gate rejects the mixed cone (contract item 4, sentence 3)",
              not bool(valid_m))
        check("no cell mutated by the rejected DO", not np.any(after))

    print("\n4. membership slack and the bounded nearest-section fallback")
    t = np.zeros(SHAPE, dtype=np.int8)
    t[25, 30] = -1     # 1.0 tiles off the section centre  -> generated (<= 1.0+0.5)
    t[26, 31] = -1     # 2.0 tiles off                     -> fringe    (<= 1.0+1.5)
    t[28, 33] = -1     # 4.0 tiles off                     -> beyond the fallback
    m = np.asarray(compute_trench_axis_membership(
        jnp.asarray(t), jnp.asarray(T._axes([0, 1, -24, 24, 20, 24, 50, 1])),
        jnp.int32(1))).astype(np.uint8)
    check("1.0 tiles off  -> owned (generated half width + 0.5)", int(m[25, 30]) == 1)
    check("2.0 tiles off  -> owned (bounded nearest-section fallback, +1.5)",
          int(m[26, 31]) == 1)
    check("4.0 tiles off  -> NOT owned (fallback is bounded)", int(m[28, 33]) == 0)

    print("\n5. an empty owner bitmask makes the gate neutral (non-trench excavation)")
    t2 = np.zeros(SHAPE, dtype=np.int8)
    t2[28, 33] = -1
    st = T._state(t2, T._axes([0, 1, -24, 24, 20, 24, 50, 1]),
                  base_angle=3, cabin_angle=0, position=(28, 25), cfg=cfg)
    v, _, _ = st._get_fresh_trench_dig_alignment()
    check("unowned target cell reports valid (gate not applicable)", bool(v))

    print("\n6. the v2 'on the line' clause: the sideways lane is refused")
    # Same shared cell, same yaw-parallel chassis, but standing in the v1
    # LATERAL lane (8 tiles = 4.57 m off section A).  Under v2 with a finite
    # trench_dig_max_offset_m that pose is off the line and must be refused;
    # with the clause disabled, or under v1 (where 4.57 m is inside the band),
    # it must be admitted.  This is the pose Lorenzo could still dig from under
    # the yaw-only v2 gate.
    side_bh, side_pos = lateral_lane["A (row=24)"]
    cb, st = cabin_containing(
        lambda c: T._state(target, axes, base_angle=side_bh, cabin_angle=c,
                           position=side_pos, cfg=cfg),
        shared,
    )
    check("a cabin heading reaches the shared cell from the v1 lateral lane",
          cb is not None, f"cabin={cb}")
    if cb is not None:
        offset = abs(side_pos[0] - 24) * float(cfg.tile_size)
        valid_side, _, _ = st._get_fresh_trench_dig_alignment()
        after_side = np.asarray(
            st._handle_do().world.action_map.map).reshape(SHAPE)
        if on_line:
            check(f"gate REFUSES the lateral lane at {offset:.2f} m "
                  f"(limit {max_offset:.2f} m)", not bool(valid_side))
            check("no cell mutated by the refused sideways DO",
                  not np.any(after_side))
        else:
            check(f"gate admits the lateral lane at {offset:.2f} m "
                  f"(on-line clause inactive)", bool(valid_side))
            check("shared cell dug from the lateral lane",
                  int(after_side[shared]) == -1)

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print(f"all multi-owner contract checks passed under gate semantics "
          f"{semantics} (on-line clause {'active' if on_line else 'inactive'}, "
          f"max_offset={max_offset} m)")


if __name__ == "__main__":
    main()
