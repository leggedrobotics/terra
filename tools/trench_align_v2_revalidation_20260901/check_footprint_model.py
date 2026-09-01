#!/usr/bin/env python3
"""Which footprint model do the replica tools have to use to be Terra?

terra commit 566867db fixed ``compute_polygon_mask`` to rasterise ``(row, col)``.
Before it, the mask was transposed and ``State._is_valid_move`` therefore tested
occupancy at the mirror position, which is what
``tools/check_trench_persistent_station_cover.py`` and
``tools/check_trench_axis_sweep_feasibility.py`` were built to model.  After the
fix the two models in ``geometry()`` swap roles, so this asserts which one is
Terra today by driving ``State._is_valid_move`` directly.

Writes footprint_model_check.json next to this file.

Run:  JAX_PLATFORMS=cpu PYTHONPATH=<terra worktree> python \
        tools/trench_align_v2_revalidation_20260901/check_footprint_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from audit_trench_gate_overrestriction import (  # noqa: E402
    MAX_AXES, NH, SHAPE, env_config, free_poses, geometry,
)
from terra.state import State  # noqa: E402

PROBES = 4000
DENSITY = 0.004          # sparse enough that ~72% of poses are legal


def main():
    cfg = env_config()
    _cones, fp_true, fp_masked, _fwd, _bwd = geometry(cfg)
    rng = np.random.default_rng(0)
    blocked = rng.random(SHAPE) < DENSITY

    state = State.new(
        jax.random.PRNGKey(0), cfg,
        np.zeros(SHAPE, np.int8), blocked.astype(np.int8),
        -97.0 * np.ones((MAX_AXES, 8), np.float32), np.int32(-1),
        -97.0 * np.ones((64, 3), np.float32), np.int32(-1),
        np.ones(SHAPE, np.bool_), np.zeros(SHAPE, np.int8),
        distance_map_override=np.ones(SHAPE, np.float32))

    def one(r, c, b):
        cur = state._get_current_agent_state()._replace(
            pos_base=jnp.stack([r, c]).astype(jnp.int16),
            angle_base=jnp.reshape(b, (1,)).astype(jnp.int8),
            angle_cabin=jnp.zeros((1,), jnp.int8),
            loaded=jnp.zeros((1,), jnp.int8))
        posed = state._set_current_agent_state(cur)
        corners = posed._get_agent_corners(
            cur.pos_base, base_orientation=cur.angle_base,
            agent_width=cfg.agent.width, agent_height=cfg.agent.height)
        return posed._is_valid_move(corners)

    batched = jax.jit(jax.vmap(one))
    probes = np.stack([rng.integers(8, 56, PROBES), rng.integers(8, 56, PROBES),
                       rng.integers(0, NH, PROBES)], axis=1).astype(np.int32)
    terra = np.asarray(batched(jnp.asarray(probes[:, 0]), jnp.asarray(probes[:, 1]),
                               jnp.asarray(probes[:, 2]))).astype(bool).reshape(-1)

    models = {
        "fp_true_untransposed (the tools' 'terra' model)":
            free_poses(blocked, fp_true, transposed=False),
        "fp_masked_transposed (the tools' 'legacy_mirror' model)":
            free_poses(blocked, fp_masked, transposed=True),
        "fp_masked_untransposed": free_poses(blocked, fp_masked, transposed=False),
        "fp_true_transposed": free_poses(blocked, fp_true, transposed=True),
    }
    rows = {}
    for name, f in models.items():
        pred = f[probes[:, 2], probes[:, 0], probes[:, 1]]
        rows[name] = {
            "agreement": float(np.mean(pred == terra)),
            "false_legal": int(np.sum(pred & ~terra)),
            "false_illegal": int(np.sum(~pred & terra)),
        }
        print(f"  {name:56s} agree={rows[name]['agreement']:.4f}")

    payload = {
        "schema": "terra_trench_footprint_model_check_v1",
        "contract": {
            "probes": PROBES, "obstacle_density": DENSITY,
            "terra_legal_fraction": float(terra.mean()),
            "reference": "State._is_valid_move on the same random obstacle field",
            "note": "free_poses does not model the corner-in-bounds clause, "
                    "which accounts for the small residual on the exact model",
        },
        "models": rows,
    }
    (HERE / "footprint_model_check.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {HERE / 'footprint_model_check.json'}")


if __name__ == "__main__":
    main()
