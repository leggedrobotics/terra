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

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=<terra> python tools/check_trench_gate_multiowner.py
"""

from __future__ import annotations

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
    T.setUpClass()
    print(f"tile_size={T.cfg.tile_size:.6f} agent={T.cfg.agent.width}x{T.cfg.agent.height} "
          f"gate={T.cfg.enforce_trench_dig_alignment} "
          f"yaw_tol={T.cfg.trench_dig_yaw_tolerance_rad} "
          f"standoff=[{T.cfg.trench_dig_standoff_min_m}, {T.cfg.trench_dig_standoff_max_m}]")

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
    for label, bh, pos in (("A (row=24)", 0, (32, 40)), ("B (col=40)", 3, (24, 32))):
        cb, st = cabin_containing(
            lambda c: T._state(target, axes, base_angle=bh, cabin_angle=c, position=pos),
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
    # 10 tiles from the base, so inside the 3.64-6.50 m annulus, and 2 tiles off
    # section A so it is owned by B only.
    exclusive = (22, 40)
    target2[exclusive] = -1
    found = None
    for cb in range(12):
        st = T._state(target2, axes, base_angle=0, cabin_angle=cb, position=(32, 40))
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
                  base_angle=3, cabin_angle=0, position=(28, 25))
    v, _, _ = st._get_fresh_trench_dig_alignment()
    check("unowned target cell reports valid (gate not applicable)", bool(v))

    print()
    if FAILURES:
        print(f"FAILED: {FAILURES}")
        sys.exit(1)
    print("all multi-owner contract checks passed")


if __name__ == "__main__":
    main()
